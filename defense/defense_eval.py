"""
================================================================================
 SmartCross — defence evidence
================================================================================

 Puts error bars, significance tests and ablations around the pipeline the API
 serves. It re-trains the same engines with the same hyper-parameters on the
 same time-based split as web/backend/build_model.py, then measures the
 questions an examiner asks.

     python defense_eval.py            # ~15-25 min, writes defense/outputs/

 What it measures
   1. every engine, with 95% bootstrap confidence intervals
   2. paired significance: is each difference real, or split noise?
   3. the WHOLE four-layer pipeline over EVERY held-out event, not 400 draws
   4. the alpha slider: what mixing raw CF back into the fused model does
   5. per-product precision / recall / F1, and macro F1 on the learnable subset
   6. ablation: which inputs carry the signal (holdings, demographics, peers)
   7. stability month by month across the three held-out months
   8. calibration of the served model's probabilities
   9. class-weighted training -- the README's "lever 1", as an experiment only

 Fairness rules applied to EVERY engine alike
   * products the customer already holds are never recommended (as app.py)
   * when an engine scores two products identically, the tie counts AGAINST it

 The eligibility rules and regime tilts are read straight out of app.py's
 source, and the collaborative engine is imported from peers.py, so this file
 cannot drift from what the API serves.
================================================================================
"""

import ast
import json
import os
import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, precision_recall_fscore_support

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
BACKEND = os.path.join(ROOT, "web", "backend")
NPZ = os.path.join(ROOT, "partB_santander", "data", "sant_events.npz")
OUT = os.path.join(HERE, "outputs")
sys.path.insert(0, BACKEND)
import peers                                                      # noqa: E402

# --- identical to build_model.py; change there, change here
TEST_MONTHS = 3
MIN_EVENTS = 50
SEED = 42
HGB_PARAMS = dict(max_iter=300, max_depth=10, learning_rate=0.1,
                  l2_regularization=1.0, early_stopping=True,
                  validation_fraction=0.1, random_state=SEED)
XGB_PARAMS = dict(n_estimators=300, max_depth=8, learning_rate=0.1,
                  subsample=0.9, colsample_bytree=0.9, tree_method="hist",
                  objective="multi:softprob", random_state=SEED, n_jobs=-1,
                  verbosity=0)

# --- defence settings
N_BOOT = 1000          # bootstrap resamples for top-k / MAP (cheap)
N_BOOT_F1 = 200        # resamples for macro F1 (needs a full F1 per resample)
LEARNABLE_BASE = 0.005 # products below 0.5% base rate reported separately
DEMO_AGE, DEMO_INCOME, DEMO_SENIORITY = 4, 5, 6   # column positions in X_demo

rng = np.random.default_rng(SEED)
RESULTS = {}


# ============================================================ app.py, verbatim
def load_app_rules():
    """Pull RULES, REGIMES and check_rule out of app.py without importing it.

    Importing app.py would load the pickled model and start a web app.
    Executing just these definitions keeps the rules byte-for-byte identical
    to what the API applies.
    """
    src = open(os.path.join(BACKEND, "app.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    wanted_assign = {"MISSING", "FIELD_LABEL", "RULES", "REGIMES"}
    wanted_def = {"known", "check_rule"}
    keep = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(getattr(t, "id", None) in wanted_assign for t in targets):
                keep.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in wanted_def:
            keep.append(node)
    ns = {"Dict": dict, "List": list}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "app.py", "exec"), ns)
    return ns["RULES"], ns["REGIMES"], ns["check_rule"]


# ================================================================== metrics
def ranks_of_truth(proba, y):
    """0-based rank of the true product, ties counted AGAINST the engine.

    Count-based engines (popularity, pattern CF) give many products the same
    score. Breaking those ties in the engine's favour would flatter it, so a
    product tied with the truth is ranked above it.
    """
    true_score = proba[np.arange(len(y)), y][:, None]
    return (proba > true_score).sum(axis=1) + (proba == true_score).sum(axis=1) - 1


def per_row(proba, y):
    r = ranks_of_truth(proba, y)
    return {"top1": (r < 1).astype(float), "top3": (r < 3).astype(float),
            "top5": (r < 5).astype(float),
            "map7": np.where(r < 7, 1.0 / (r + 1), 0.0)}


def macro_f1(y, pred):
    """Same convention as build_model.py: averaged over classes present."""
    return float(f1_score(y, pred, average="macro", zero_division=0))


def boot_means(v, n):
    """Bootstrap means of a per-row vector, in chunks to keep memory flat.

    The generator is re-seeded on every call, so every engine is resampled on
    exactly the same rows -- which is what makes the comparisons paired.
    """
    g = np.random.default_rng(SEED + 1)
    out = []
    for _ in range(N_BOOT // 100):
        idx = g.integers(0, n, size=(100, n), dtype=np.int32)
        out.append(v[idx].mean(axis=1))
    return np.concatenate(out)


def ci(samples):
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return [float(lo), float(hi)]


def scored(name, proba, y, boot_f1):
    rows = per_row(proba, y)
    pred = proba.argmax(axis=1)
    out = {"name": name}
    for k, v in rows.items():
        out[k] = float(v.mean())
        out[k + "_ci"] = ci(boot_means(v, len(y)))
    out["macro_f1"] = macro_f1(y, pred)
    out["macro_f1_ci"] = ci([macro_f1(y[b], pred[b]) for b in boot_f1])
    out["_rows"], out["_pred"] = rows, pred
    print(f"  {name:40} macro F1 {out['macro_f1']:.4f} "
          f"[{out['macro_f1_ci'][0]:.4f},{out['macro_f1_ci'][1]:.4f}]  "
          f"top-1 {out['top1']:.3f}  top-3 {out['top3']:.3f} "
          f"[{out['top3_ci'][0]:.3f},{out['top3_ci'][1]:.3f}]  MAP@7 {out['map7']:.3f}")
    return out


def paired(a, b, y, boot_f1, label):
    """Paired bootstrap of (a - b) on identical resamples of the test rows."""
    res = {"comparison": label}
    for k in ("top1", "top3", "map7"):
        diff = a["_rows"][k] - b["_rows"][k]
        d = boot_means(diff, len(y))
        res[k] = {"diff": float(diff.mean()), "ci": ci(d),
                  "p_two_sided": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))}
    df = np.array([macro_f1(y[bi], a["_pred"][bi]) - macro_f1(y[bi], b["_pred"][bi]) for bi in boot_f1])
    res["macro_f1"] = {"diff": a["macro_f1"] - b["macro_f1"], "ci": ci(df),
                       "p_two_sided": float(min(1.0, 2 * min((df <= 0).mean(), (df >= 0).mean())))}
    sig = lambda r: "significant" if (r["ci"][0] > 0 or r["ci"][1] < 0) else "not significant"
    print(f"  {label:46} d top-3 {res['top3']['diff']:+.4f} {sig(res['top3']):16} "
          f"d macro F1 {res['macro_f1']['diff']:+.4f} {sig(res['macro_f1'])}")
    return res


def public(e):
    return {k: v for k, v in e.items() if not k.startswith("_")}


def mask_held(proba, held):
    """app.py zeroes products the customer already holds, then renormalises."""
    p = np.where(held > 0, 0.0, proba)
    s = p.sum(axis=1, keepdims=True)
    return np.where(s > 0, p / np.clip(s, 1e-12, None), proba)


def hgb(Xtr, ytr, **kw):
    return HistGradientBoostingClassifier(**HGB_PARAMS, **kw).fit(Xtr, ytr)


# ===================================================================== main
def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    print("=" * 84)
    print("SMARTCROSS — DEFENCE EVIDENCE for the served pipeline")
    print("=" * 84)

    RULES, REGIMES, check_rule = load_app_rules()
    print(f"Rules read from app.py   : {len(RULES)} gated products, {len(REGIMES)} regimes")

    d = np.load(NPZ, allow_pickle=True)
    X_prev = d["X_prev"].astype(np.float32)
    X_demo = d["X_demo"].astype(np.float32)
    X_hist = d["X_hist"].astype(np.float32)          # 48 holdings-history features
    y = d["y"].astype(int)
    mi = d["mi"].astype(int)
    all_names = [str(s) for s in d["names"]]
    month_keys = [str(s)[:7] for s in d["month_keys"]]

    counts = np.bincount(y, minlength=len(all_names))
    keep = np.where(counts >= MIN_EVENTS)[0]
    mask = np.isin(y, keep)
    remap = np.full(len(all_names), -1)
    remap[keep] = np.arange(len(keep))
    y = remap[y[mask]]
    X_prev, X_demo, X_hist, mi = X_prev[mask], X_demo[mask], X_hist[mask], mi[mask]
    X = np.hstack([X_prev, X_demo])
    products = [all_names[j] for j in keep]
    K = len(products)
    held = X_prev[:, keep]

    cut = mi.max() - TEST_MONTHS + 1
    tr, te = mi < cut, mi >= cut
    y_tr, y_te = y[tr], y[te]
    n_te = int(te.sum())
    h_te = held[te]
    print(f"Products {K} · train {int(tr.sum()):,} events (months <{cut}) · "
          f"test {n_te:,} events (months {cut}-{mi.max()})")
    boot_f1 = [rng.integers(0, n_te, size=n_te, dtype=np.int32) for _ in range(N_BOOT_F1)]

    base = np.bincount(y_tr, minlength=K).astype(float)
    base /= base.sum()
    lift = (X_prev[tr].T @ np.eye(K)[y_tr]) / np.clip(X_prev[tr].sum(axis=0)[:, None], 1, None)
    lift = lift / np.clip(base[None, :], 1e-9, None)

    # peers: training rows from earlier months only, test rows from training months
    T = peers.fit(X_prev[tr], y_tr, K)
    F = peers.expanding_features(X_prev, y, mi, K, np.unique(mi[tr]))
    F[te] = peers.features(T, X_prev[te])
    XF = np.hstack([X, F])                            # 79: fused, last month only
    XH = np.hstack([X, F, X_hist])                    # 127: + holdings history = served

    # ------------------------------------------------------------ 1. engines
    print(f"\n[1] Every engine, 95% bootstrap CIs ({N_BOOT} resamples; {N_BOOT_F1} for macro F1)")
    pop_raw = np.tile(base, (n_te, 1))
    old_cf = X_prev[te] @ lift
    rs = old_cf.sum(axis=1, keepdims=True)
    old_cf = np.where(rs > 0, old_cf / np.clip(rs, 1e-9, None), base[None, :])
    cf = peers.pattern(T, X_prev[te])

    t1 = time.time()
    content = hgb(X[tr], y_tr).predict_proba(X[te])
    fused_nh = hgb(XF[tr], y_tr).predict_proba(XF[te])
    fused_model = hgb(XH[tr], y_tr)
    fused = fused_model.predict_proba(XH[te])
    print(f"  (content-only and two fused boosters trained in {time.time()-t1:.0f}s)")
    xgb = None
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(num_class=K, **XGB_PARAMS).fit(X[tr], y_tr).predict_proba(X[te])
    except ImportError:
        print("  XGBoost not installed -- its row is omitted, not quoted.")

    E = {}
    E["pop_unmasked"] = scored("Popularity, held NOT masked (old README)", pop_raw, y_te, boot_f1)
    E["pop"] = scored("Popularity (no ML)", mask_held(pop_raw, h_te), y_te, boot_f1)
    E["cf_old"] = scored("Item-item CF, lift (previous)", mask_held(old_cf, h_te), y_te, boot_f1)
    E["cf"] = scored("Collaborative, portfolio pattern", mask_held(cf, h_te), y_te, boot_f1)
    E["content"] = scored("Content only (HistGB)", mask_held(content, h_te), y_te, boot_f1)
    if xgb is not None:
        E["xgb"] = scored("Content only (XGBoost)", mask_held(xgb, h_te), y_te, boot_f1)
    E["blend_old"] = scored("Weighted blend 0.5/0.5 (previous)",
                            mask_held(0.5 * content + 0.5 * old_cf, h_te), y_te, boot_f1)
    E["fused_nh"] = scored("Fused hybrid, no history (previous)", mask_held(fused_nh, h_te), y_te, boot_f1)
    fused_m = mask_held(fused, h_te)
    E["fused"] = scored("FUSED + HISTORY (served)", fused_m, y_te, boot_f1)
    RESULTS["chance_macro_f1"] = 1.0 / K
    RESULTS["split"] = {"train_events": int(tr.sum()), "test_events": n_te,
                        "test_months": month_keys[cut:mi.max() + 1], "products": K}

    # ------------------------------------------------------- 2. significance
    print("\n[2] Paired bootstrap: is each gap real?")
    comps = [paired(E["fused"], E["fused_nh"], y_te, boot_f1, "Served (with history) vs fused without"),
             paired(E["fused"], E["content"], y_te, boot_f1, "Served vs content only"),
             paired(E["fused_nh"], E["content"], y_te, boot_f1, "Fused (no history) vs content only"),
             paired(E["fused"], E["blend_old"], y_te, boot_f1, "Served vs previous 50/50 blend"),
             paired(E["fused"], E["pop"], y_te, boot_f1, "Served vs popularity"),
             paired(E["cf"], E["pop"], y_te, boot_f1, "Pattern CF vs popularity"),
             paired(E["cf"], E["cf_old"], y_te, boot_f1, "Pattern CF vs lift CF (previous)"),
             paired(E["blend_old"], E["content"], y_te, boot_f1, "Previous blend vs content only")]
    if xgb is not None:
        comps.append(paired(E["xgb"], E["content"], y_te, boot_f1, "XGBoost vs HistGB (content only)"))
    RESULTS["significance"] = comps

    # -------------------------------------- 3. the whole pipeline, every event
    print("\n[3] Full four-layer pipeline over EVERY held-out event")
    regime_by_month, regime_detail = {}, {}
    try:
        from macro import MacroHistory
        mh = MacroHistory()
        for m in range(cut, mi.max() + 1):
            row = mh.at(month_keys[m]) if mh.available else None
            regime_by_month[m] = row["regime"] if row else "RECOVERY"
            regime_detail[month_keys[m]] = (
                {"regime": row["regime"], "agreement": row["agreement"], "n_series": row["n_series"]}
                if row else {"regime": "RECOVERY", "note": "fallback"})
    except Exception as exc:
        print(f"  macro unavailable ({exc}); using RECOVERY for every month")
        regime_by_month = {m: "RECOVERY" for m in range(cut, mi.max() + 1)}
    print(f"  regime per test month (Auto, from the price series): "
          + ", ".join(f"{k} {v['regime']}" for k, v in regime_detail.items()))

    mi_te = mi[te]
    tvec = np.array([[REGIMES[regime_by_month[m]].get(p, 1.0) for p in products] for m in mi_te])
    s3 = fused_m * tvec
    s3 = s3 / np.clip(s3.sum(axis=1, keepdims=True), 1e-12, None)

    demo_te, prev_te = X_demo[te], X_prev[te]      # index once; [te][i] in the loop is quadratic
    allowed = np.ones((n_te, K), dtype=bool)
    missing_block = np.zeros((n_te, K), dtype=bool)
    gated = [(j, p) for j, p in enumerate(products) if p in RULES]
    for i in range(n_te):
        c = {"age": float(demo_te[i, DEMO_AGE]), "income": float(demo_te[i, DEMO_INCOME]),
             "seniority_months": float(demo_te[i, DEMO_SENIORITY]),
             "holdings": {all_names[h] for h in np.flatnonzero(prev_te[i])}}
        for j, p in gated:
            ok, _, absent = check_rule(p, c)
            if not ok:
                allowed[i, j] = False
                missing_block[i, j] = bool(absent)
    final = np.where(allowed, s3, 0.0)
    fallback = final.sum(axis=1) <= 0
    final[fallback] = s3[fallback]

    E["tilt"] = scored("+ macro tilt (Auto regime)", s3, y_te, boot_f1)
    E["pipe"] = scored("+ eligibility = FULL PIPELINE", final, y_te, boot_f1)
    truth_blocked = ~allowed[np.arange(n_te), y_te]
    truth_blocked_missing = missing_block[np.arange(n_te), y_te]
    per_prod_block = {products[j]: {"block_rate_of_actual_takers": float(truth_blocked[y_te == j].mean()),
                                    "of_which_missing_data": float(truth_blocked_missing[y_te == j].mean()),
                                    "takers": int((y_te == j).sum())}
                      for j, _ in gated if (y_te == j).any()}
    RESULTS["pipeline"] = {
        "regime_by_month": regime_detail,
        "truth_removed_by_rule": float(truth_blocked.mean()),
        "truth_removed_by_rule_ci": ci(boot_means(truth_blocked.astype(float), n_te)),
        "truth_removed_because_data_missing": float(truth_blocked_missing.mean()),
        "all_blocked_fallback": float(fallback.mean()),
        "top1_change_from_tilt": float((s3.argmax(1) != fused_m.argmax(1)).mean()),
        "per_product_block": per_prod_block,
    }
    print(f"  true product removed by a rule : {truth_blocked.mean():.4f} "
          f"(of which because data was missing {truth_blocked_missing.mean():.4f})")
    print(f"  tilt changes the #1 pick for   : {RESULTS['pipeline']['top1_change_from_tilt']:.4f}")
    print(f"  all-blocked fallback           : {fallback.mean():.4f}")

    # ------------------------------------------------------------ 4. alpha
    print("\n[4] The alpha slider: fused model mixed with raw pattern CF")
    sweep = []
    for a in np.round(np.linspace(0, 1, 11), 2):
        p = mask_held(a * fused + (1 - a) * cf, h_te)
        r = per_row(p, y_te)
        sweep.append({"alpha": float(a), "top1": float(r["top1"].mean()),
                      "top3": float(r["top3"].mean()), "map7": float(r["map7"].mean()),
                      "macro_f1": macro_f1(y_te, p.argmax(1))})
        print(f"  alpha {a:.1f}  top-1 {sweep[-1]['top1']:.3f}  top-3 {sweep[-1]['top3']:.3f}  "
              f"MAP@7 {sweep[-1]['map7']:.3f}  macro F1 {sweep[-1]['macro_f1']:.4f}")
    RESULTS["alpha_sweep"] = sweep

    # ------------------------------------------------------ 5. per product
    print("\n[5] Per-product scores of the served model")
    pr, rc, f1, sup = precision_recall_fscore_support(
        y_te, fused_m.argmax(1), labels=np.arange(K), zero_division=0)
    top3_hit = E["fused"]["_rows"]["top3"]
    per_product = []
    for j in np.argsort(-base):
        t3 = float(top3_hit[y_te == j].mean()) if sup[j] else None
        per_product.append({"product": products[j], "base_rate": float(base[j]),
                            "test_support": int(sup[j]), "precision": float(pr[j]),
                            "recall": float(rc[j]), "f1": float(f1[j]), "top3_recall": t3})
        print(f"  {products[j]:24} base {base[j]*100:6.2f}%  n {sup[j]:6}  P {pr[j]:.3f}  "
              f"R {rc[j]:.3f}  F1 {f1[j]:.3f}  top-3 recall {'-' if t3 is None else f'{t3:.3f}'}")
    present = sup > 0
    learn = (base >= LEARNABLE_BASE) & present
    RESULTS["per_product"] = per_product
    RESULTS["learnable_subset"] = {
        "threshold_base_rate": LEARNABLE_BASE, "n_products": int(learn.sum()),
        "excluded": [products[j] for j in np.flatnonzero(present & ~learn)],
        "macro_f1": float(f1[learn].mean()), "chance": float(1.0 / learn.sum()),
        "macro_f1_all_present": float(f1[present].mean())}
    print(f"  learnable subset ({learn.sum()} products, base >= {LEARNABLE_BASE:.1%}): "
          f"macro F1 {f1[learn].mean():.4f} vs all present {f1[present].mean():.4f}")

    # ------------------------------------------------------- 6. ablation
    print("\n[6] Ablation -- which inputs carry the signal (same booster each time)")
    n_hold, n_prof = X_prev.shape[1], X.shape[1]
    abl = []
    for name, cols in (("holdings only (24)", slice(0, n_hold)),
                       ("demographics + behaviour only (11)", slice(n_hold, n_prof)),
                       ("peer scores only (44)", slice(n_prof, XF.shape[1])),
                       ("holdings history only (48)", slice(XF.shape[1], XH.shape[1]))):
        p = mask_held(hgb(XH[tr][:, cols], y_tr).predict_proba(XH[te][:, cols]), h_te)
        abl.append(public(scored(name, p, y_te, boot_f1)))
    abl.append({**public(E["content"]), "name": "profile: holdings + demographics (35)"})
    abl.append({**public(E["fused_nh"]), "name": "profile + peers (79)"})
    abl.append({**public(E["fused"]), "name": "profile + peers + history = served (127)"})
    RESULTS["ablation"] = abl

    # ------------------------------------------------- 7. month stability
    print("\n[7] Stability across the three held-out months")
    stab = []
    for m in range(cut, mi.max() + 1):
        sel = mi_te == m
        row = {"month": month_keys[m], "events": int(sel.sum())}
        for key in ("fused", "content", "pop"):
            r = E[key]["_rows"]
            row[key + "_top3"] = float(r["top3"][sel].mean())
            row[key + "_top1"] = float(r["top1"][sel].mean())
        row["fused_macro_f1"] = macro_f1(y_te[sel], E["fused"]["_pred"][sel])
        stab.append(row)
        print(f"  {month_keys[m]}  n {sel.sum():6}  fused top-3 {row['fused_top3']:.3f}  "
              f"content {row['content_top3']:.3f}  popularity {row['pop_top3']:.3f}  "
              f"fused macro F1 {row['fused_macro_f1']:.4f}")
    RESULTS["month_stability"] = stab

    # ---------------------------------------------------- 8. calibration
    print("\n[8] Calibration of the served model's top probability")
    conf = fused_m.max(axis=1)
    hit = (fused_m.argmax(1) == y_te).astype(float)
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(conf, bins) - 1, 0, 9)
    rel, ece = [], 0.0
    for b in range(10):
        s = idx == b
        if s.any():
            rel.append({"bin": f"{bins[b]:.1f}-{bins[b+1]:.1f}", "n": int(s.sum()),
                        "mean_confidence": float(conf[s].mean()), "accuracy": float(hit[s].mean())})
            ece += s.mean() * abs(conf[s].mean() - hit[s].mean())
    brier = float(((fused_m - np.eye(K)[y_te]) ** 2).sum(axis=1).mean())
    RESULTS["calibration"] = {"ece": float(ece), "brier": brier, "reliability": rel}
    print(f"  expected calibration error {ece:.4f}   multiclass Brier {brier:.4f}")

    # ------------------------------------------- 9. class-weighted (lever 1)
    print("\n[9] Class-weighted fused model -- README lever 1, experiment only")
    try:
        p = mask_held(hgb(XH[tr], y_tr, class_weight="balanced").predict_proba(XH[te]), h_te)
        E["cw"] = scored("fused, class_weight=balanced", p, y_te, boot_f1)
        RESULTS["class_weighted"] = {
            "engine": public(E["cw"]),
            "vs_served": paired(E["cw"], E["fused"], y_te, boot_f1, "class-weighted vs served")}
    except TypeError:
        print("  this scikit-learn has no class_weight on HistGB (needs >= 1.2) -- skipped")

    RESULTS["engines"] = {k: public(v) for k, v in E.items()}
    # the validation-month decisions behind the served model, carried alongside
    for key, fn in (("improve", "improve_experiment.json"), ("xgb_gpu", "xgb_gpu_check.json")):
        path = os.path.join(OUT, fn)
        if os.path.exists(path):
            RESULTS[key] = json.load(open(path))
    RESULTS["runtime_seconds"] = round(time.time() - t0)
    import sklearn
    RESULTS["environment"] = {"python": sys.version.split()[0], "numpy": np.__version__,
                              "scikit_learn": sklearn.__version__}
    with open(os.path.join(OUT, "defense_results.json"), "w") as f:
        json.dump(RESULTS, f, indent=2)
    make_charts(E, sweep, abl, rel)
    print("\n" + "=" * 84)
    print(f"DONE in {time.time()-t0:.0f}s  ->  defense/outputs/defense_results.json + charts")
    print("=" * 84)


def make_charts(E, sweep, abl, rel):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed -- charts skipped, JSON written")
        return
    violet, grey, mint = "#6d28d9", "#b4abce", "#0e8f5e"

    keys = [k for k in ("pop", "cf_old", "cf", "content", "xgb", "blend_old", "fused_nh", "fused", "tilt", "pipe") if k in E]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    names = [E[k]["name"] for k in keys]
    vals = [E[k]["top3"] for k in keys]
    err = np.array([[E[k]["top3"] - E[k]["top3_ci"][0], E[k]["top3_ci"][1] - E[k]["top3"]] for k in keys]).T
    cols = [violet if k == "fused" else (grey if k in ("pop", "cf_old", "blend_old", "fused_nh") else
            (mint if k in ("tilt", "pipe") else "#8b5cf6")) for k in keys]
    ax.barh(names, vals, xerr=err, color=cols, capsize=4)
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center")
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()
    ax.set_xlabel("top-3 hit rate on the held-out months (95% bootstrap CI)")
    ax.set_title("Every engine and every pipeline stage, one split, scored the same way")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "d1_engines_ci.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    a = [s["alpha"] for s in sweep]
    for k, lab in (("top3", "top-3"), ("top1", "top-1"), ("map7", "MAP@7")):
        ax.plot(a, [s[k] for s in sweep], marker="o", label=lab)
    ax.set_xlabel("alpha  (1 = fused model as served, 0 = raw pattern CF alone)")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("The peer-mix slider")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "d2_alpha_sweep.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.barh([r["name"] for r in abl], [r["top3"] for r in abl],
            color=[violet if "served" in r["name"] else "#8b5cf6" for r in abl])
    for i, r in enumerate(abl):
        ax.text(r["top3"] + 0.01, i, f"top-3 {r['top3']:.3f} · macro F1 {r['macro_f1']:.3f}", va="center")
    ax.set_xlim(0, 1.3)
    ax.invert_yaxis()
    ax.set_title("Ablation -- which inputs carry the signal")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "d3_ablation.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], ls="--", c="grey")
    ax.plot([r["mean_confidence"] for r in rel], [r["accuracy"] for r in rel], marker="o", c=violet)
    ax.set_xlabel("predicted probability of the #1 product")
    ax.set_ylabel("observed hit rate")
    ax.set_title("Calibration of the served model")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "d4_calibration.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
