"""
================================================================================
 PART B / STEP 2.4a : BUILD FULL EVENT TABLE  (streaming, memory-safe)
 SmartCross
================================================================================

 WHY THIS EXISTS
     Step 2.3 used a 1-in-5 customer sample because the whole file would not
     fit in memory. This builds the event table from ALL 956,645 customers
     without ever holding the 13.6M-row file in RAM.

 HOW IT STAYS SMALL AND FAST
     The file is ordered by snapshot date. We stream it one chunk at a time
     and keep a few small arrays indexed by customer:

         mask      -> 24-bit holdings mask, last month seen   (uint32)
         mask2     -> the same, one observation earlier       (uint32)
         streak    -> months each product has been held in a row (24 x uint8)
         last_acq, n_acq, first_mi, seen

     Each month's rows are processed together with numpy, so the 13.6M rows
     take minutes rather than a Python loop per row.

 WHAT IT EMITS
     One row per (customer, month, product added), with features computed
     from state BEFORE that month - so nothing from the month being predicted
     leaks into its own features.

         X_prev   what they held last month                       (24)
         X_demo   behaviour + demographics                        (11)
         X_hist   holdings HISTORY, new in this version           (48)
                    held_2m_ago_*   what they held one observation earlier,
                                    so a product dropped last month shows
                    held_months_*   how many observations in a row they
                                    have held each product (capped at 15)

     X_prev, X_demo, y, mi and cust are identical to the previous version of
     this script; X_hist is added beside them.

 OUTPUT
     ../data/sant_events.npz    feature matrices, labels, month index, ids
================================================================================
"""

import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

RAW = None          # None = auto-find train_ver2.csv near this script
OUT = os.path.join(HERE, "..", "data", "sant_events.npz")
CHUNK = 1_000_000   # rows per chunk; lower to 500_000 if RAM is tight
STREAK_CAP = 15


def find_raw():
    """Locate train_ver2.csv without hardcoding a path."""
    if RAW and os.path.exists(RAW):
        return RAW
    cands = [
        os.path.join("..", "data", "train_ver2.csv"),
        os.path.join("..", "..", "data", "raw", "stander again lol", "train_ver2.csv"),
        os.path.join("..", "..", "data", "raw", "santander_bank", "train_ver2.csv"),
        "train_ver2.csv",
    ]
    for c in cands:
        q = c if os.path.isabs(c) else os.path.join(HERE, c)
        if os.path.exists(q):
            return os.path.normpath(q)
    root = os.path.normpath(os.path.join(HERE, "..", ".."))
    for dirpath, _, files in os.walk(root):
        if "train_ver2.csv" in files:
            return os.path.join(dirpath, "train_ver2.csv")
    return None

PROD = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
        'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
        'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
        'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
        'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
        'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
        'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
        'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']

NAMES = ['savings_account', 'guarantees', 'current_account',
         'derivatives_account', 'payroll_account', 'junior_account',
         'mas_particular_account', 'particular_account',
         'particular_plus_account', 'short_term_deposits',
         'medium_term_deposits', 'long_term_deposits', 'e_account', 'funds',
         'mortgage', 'pension_plan', 'loans', 'taxes', 'credit_card',
         'securities', 'home_account', 'payroll', 'pensions', 'direct_debit']

HIST_NAMES = ([f"held_2m_ago_{n}" for n in NAMES] +
              [f"held_months_{n}" for n in NAMES])

DEMO = ['sexo', 'age', 'ind_nuevo', 'antiguedad',
        'ind_actividad_cliente', 'renta', 'segmento']
COLS = ['fecha_dato', 'ncodpers'] + DEMO + PROD

SEG = {"01 - TOP": 0, "02 - PARTICULARES": 1, "03 - UNIVERSITARIO": 2}
BITS = (1 << np.arange(24)).astype(np.uint32)


def num(s):
    return pd.to_numeric(s.astype(str).str.strip().replace(
        {"NA": None, "": None, "nan": None}), errors="coerce")


def unpack(m):
    """[n] uint32 masks -> [n x 24] uint8 flags"""
    return ((m[:, None] >> np.arange(24, dtype=np.uint32)) & 1).astype(np.uint8)


class State:
    """Per-customer state in arrays, grown as new customers appear."""

    def __init__(self, n=1_200_000):
        self.idx = {}
        self.mask = np.zeros(n, np.uint32)
        self.mask2 = np.zeros(n, np.uint32)
        self.streak = np.zeros((n, 24), np.uint8)
        self.last_acq = np.full(n, -1, np.int32)
        self.n_acq = np.zeros(n, np.int32)
        self.first_mi = np.zeros(n, np.int32)
        self.seen = np.zeros(n, bool)

    def index(self, cust):
        get, out = self.idx.setdefault, np.empty(len(cust), np.int64)
        for i, c in enumerate(cust.tolist()):
            out[i] = get(c, len(self.idx))
        need = len(self.idx)
        if need > len(self.mask):
            grow = max(need, int(len(self.mask) * 1.5))
            for name in ("mask", "mask2", "streak", "last_acq", "n_acq", "first_mi", "seen"):
                a = getattr(self, name)
                b = np.zeros((grow,) + a.shape[1:], a.dtype)
                if name == "last_acq":
                    b[:] = -1
                b[:len(a)] = a
                setattr(self, name, b)
        return out


def main():
    t0 = time.time()
    print("=" * 78)
    print("STEP 2.4a : BUILDING FULL EVENT TABLE (all customers, with history)")
    print("=" * 78)
    raw = find_raw()
    if raw is None:
        print("\n[X] Could not find train_ver2.csv. Set RAW at the top.\n")
        sys.exit(1)
    print(f"\nSource: {raw}")
    print(f"Output: {os.path.normpath(OUT)}\n")

    S = State()
    months = {}
    out = {k: [] for k in ("prev", "demo", "hist", "y", "mi", "cust")}
    rows_seen = 0

    for ch in pd.read_csv(raw, usecols=COLS, chunksize=CHUNK, low_memory=False):
        rows_seen += len(ch)
        for d in ch["fecha_dato"].unique():
            if d not in months:
                months[d] = len(months)
        mi_col = ch["fecha_dato"].map(months).values.astype(np.int16)
        cust = ch["ncodpers"].values.astype(np.int64)
        P = ch[PROD].apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.uint8).values
        cur_mask = (P.astype(np.uint32) * BITS).sum(axis=1).astype(np.uint32)
        demo_cols = np.column_stack([
            num(ch["age"]).fillna(-1).values, num(ch["renta"]).fillna(-1).values,
            num(ch["antiguedad"]).fillna(-1).values, num(ch["ind_actividad_cliente"]).fillna(-1).values,
            num(ch["ind_nuevo"]).fillna(-1).values, ch["segmento"].map(SEG).fillna(-1).values,
            (ch["sexo"].astype(str).str.strip() == "V").values,
        ]).astype(np.float32)
        ix_all = S.index(cust)

        # one month at a time: a customer appears at most once per month
        for m in np.unique(mi_col):
            r = np.flatnonzero(mi_col == m)
            ix, cm = ix_all[r], cur_mask[r]
            seen = S.seen[ix]

            # ---- events: customers already seen who switched something on
            pm = S.mask[ix]
            added = np.where(seen, cm & ~pm, 0).astype(np.uint32)
            ev = np.flatnonzero(added)
            if len(ev):
                e_ix = ix[ev]
                pbits = unpack(pm[ev])
                la = S.last_acq[e_ix]
                feats = np.column_stack([
                    pbits.sum(axis=1), m - S.first_mi[e_ix],
                    np.where(la >= 0, m - la, -1), S.n_acq[e_ix],
                    demo_cols[r[ev]],
                ]).astype(np.float32)
                hist = np.hstack([unpack(S.mask2[e_ix]), S.streak[e_ix]]).astype(np.uint8)
                addb = unpack(added[ev])                      # [n_ev x 24]
                k_row, k_prod = np.nonzero(addb)              # row order, products ascending
                out["prev"].append(pbits[k_row])
                out["demo"].append(feats[k_row])
                out["hist"].append(hist[k_row])
                out["y"].append(k_prod.astype(np.int16))
                out["mi"].append(np.full(len(k_row), m, np.int16))
                out["cust"].append(cust[r[ev]][k_row])
                S.n_acq[e_ix] += addb.sum(axis=1)
                S.last_acq[e_ix] = m

            # ---- state update, AFTER the features were taken
            new = ~seen
            S.first_mi[ix[new]] = m
            S.mask2[ix[seen]] = pm[seen]
            held = unpack(cm).astype(bool)
            st = S.streak[ix]
            st = np.where(held, np.minimum(st.astype(np.int16) + 1, STREAK_CAP), 0).astype(np.uint8)
            S.streak[ix] = st
            S.mask[ix] = cm
            S.seen[ix] = True

        n_ev = sum(len(a) for a in out["y"])
        print(f"  {rows_seen:,} rows | {len(months)} months | {len(S.idx):,} customers | "
              f"{n_ev:,} events | {time.time()-t0:.0f}s", flush=True)

    X_prev = np.concatenate(out["prev"]).astype(np.uint8)
    X_demo = np.concatenate(out["demo"]).astype(np.float32)
    X_hist = np.concatenate(out["hist"]).astype(np.uint8)
    y = np.concatenate(out["y"]).astype(np.int16)
    mi = np.concatenate(out["mi"]).astype(np.int16)
    cu = np.concatenate(out["cust"]).astype(np.int64)

    print(f"\n  Rows read         : {rows_seen:,}")
    print(f"  Customers         : {len(S.idx):,}")
    print(f"  Months            : {len(months)}")
    print(f"  ACQUISITION EVENTS: {len(y):,}")
    print(f"  history features  : {X_hist.shape[1]}")

    os.makedirs(os.path.dirname(os.path.normpath(OUT)), exist_ok=True)
    np.savez_compressed(OUT, X_prev=X_prev, X_demo=X_demo, X_hist=X_hist, y=y, mi=mi,
                        cust=cu, names=np.array(NAMES), hist_names=np.array(HIST_NAMES),
                        month_keys=np.array(sorted(months, key=months.get)))
    print(f"\n  Saved {os.path.normpath(OUT)}  ({os.path.getsize(OUT)/1e6:.0f} MB)")
    print("=" * 78)
    print(f"STEP 2.4a COMPLETE  ({time.time()-t0:.0f}s)")
    print("=" * 78)


if __name__ == "__main__":
    main()
