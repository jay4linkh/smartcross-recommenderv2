"""
================================================================================
 PART B / STEP 2.2 : ANALYSIS OF THE SANTANDER DATA
 Smart Financial Cross-Selling Recommender System
================================================================================

 THE QUESTION THIS ANSWERS
     Phases 1-4 proved LendingClub could not beat a popularity baseline,
     because loan PURPOSE is not encoded in credit features. The whole reason
     for switching datasets is that Santander asks an answerable question:

         "given the products this customer already holds,
          which one will they add next?"

     This script checks whether that signal is actually there before we build
     anything on top of it. Specifically:

       1. how many products each customer holds       (portfolio sizes)
       2. how common each product is                  (imbalance check)
       3. ACQUISITION EVENTS - the 0 -> 1 flips       <- the whole point
       4. which products get bought together          (co-occurrence)
       5. sequence rules: "after X, customers add Y"  (your Example B)

     Step 3 is the one LendingClub structurally could not produce. If the
     acquisition counts here are healthy, the switch was worth it.

 HOW TO RUN IN VSCODE
     1. Run phase2_1_translate.py FIRST (this reads its output)
     2. Right-click -> "Run Python File in Terminal"
        (or: python phase4_5b_analysis.py)

 OUTPUT
     phase4_5_analysis.json       every number, for your report
     phase4_5_products.png        ownership + acquisition charts
     phase4_5_cooccurrence.png    which products travel together
     santander_acquisitions.csv   the acquisition events, for Phase 5
================================================================================
"""

import os
import sys
import json
import time

try:
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    print(f"\n[X] Missing package: {exc.name}")
    print("    pip install pandas numpy matplotlib\n")
    sys.exit(1)


# ============================================================== CONFIG ====

INPUT_PATH = os.path.join("..", "data", "santander_english.csv")   # output of phase4_5a
TOP_N_RULES = 20                       # how many sequence rules to print
MIN_RULE_SUPPORT = 50                  # ignore rules seen fewer times than this

# =========================================================================


HERE = os.path.dirname(os.path.abspath(__file__))


def banner(text, ch="="):
    print("\n" + ch * 78)
    print(text)
    print(ch * 78)


def short(name):
    """prod_credit_card -> credit_card"""
    return name.replace("prod_", "")


# ================================================================== MAIN ==
def main():
    t0 = time.time()
    banner("PART B / STEP 2.2 : ANALYSIS OF SANTANDER DATA")

    path = INPUT_PATH if os.path.isabs(INPUT_PATH) else os.path.join(HERE, INPUT_PATH)
    if not os.path.exists(path):
        print(f"\n[X] {INPUT_PATH} not found.")
        print("    Run  python phase2_1_translate.py  first.\n")
        sys.exit(1)

    print(f"\nReading {path} ...")
    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded {len(df):,} rows x {df.shape[1]} columns")

    product_cols = sorted([c for c in df.columns if c.startswith("prod_")])
    print(f"Products: {len(product_cols)}")

    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
    months = sorted(df["snapshot_date"].dropna().unique())
    n_cust = df["customer_id"].nunique()

    print(f"Customers: {n_cust:,}")
    print(f"Months   : {len(months)}  "
          f"({pd.Timestamp(months[0]).date()} -> {pd.Timestamp(months[-1]).date()})")

    report = {
        "n_rows": int(len(df)),
        "n_customers": int(n_cust),
        "n_months": len(months),
        "n_products": len(product_cols),
    }

    # ---------------------------------------------- 1. portfolio sizes
    banner("1. PORTFOLIO SIZE - how many products per customer")

    latest = df.sort_values("snapshot_date").groupby("customer_id").tail(1)
    sizes = latest[product_cols].sum(axis=1)

    print(f"\n  mean   {sizes.mean():.2f} products")
    print(f"  median {sizes.median():.0f}")
    print(f"  max    {sizes.max():.0f}")

    print("\n  distribution:")
    dist = sizes.value_counts().sort_index()
    for k, v in dist.head(12).items():
        bar = "#" * int(60 * v / dist.max())
        print(f"    {int(k):2d} products  {v:>8,}  {bar}")

    multi = int((sizes >= 2).sum())
    print(f"\n  Customers holding 2+ products: {multi:,} "
          f"({100*multi/len(sizes):.1f}%)")
    print("  -> THIS is what LendingClub could not give us. Every LendingClub")
    print("     customer had exactly one product, so there was nothing to")
    print("     cross-sell from.")

    report["portfolio"] = {
        "mean": float(sizes.mean()),
        "median": float(sizes.median()),
        "max": int(sizes.max()),
        "pct_multi_product": float(100 * multi / len(sizes)),
    }

    # ---------------------------------------------- 2. ownership counts
    banner("2. PRODUCT OWNERSHIP - imbalance check")

    own = latest[product_cols].sum().sort_values(ascending=False)
    print(f"\n  {'product':32s} {'holders':>10s} {'% of base':>10s}")
    print("  " + "-" * 56)
    for c, v in own.items():
        print(f"  {short(c):32s} {int(v):>10,} {100*v/len(latest):>9.2f}%")

    top, bottom = own.iloc[0], own[own > 0].iloc[-1]
    print(f"\n  Imbalance ratio: {top/max(bottom,1):.0f} : 1")
    print("  Same discipline as Phase 2 applies - do not let the common")
    print("  products swamp the rare ones.")

    report["ownership"] = {short(c): int(v) for c, v in own.items()}

    # ------------------------------------- 3. ACQUISITION EVENTS (0 -> 1)
    banner("3. ACQUISITION EVENTS - the 0 -> 1 flips  (THE KEY NUMBER)")
    print("\n  A customer who did NOT hold a product last month and DOES")
    print("  hold it this month = one cross-sell event. This is the thing")
    print("  LendingClub structurally could not produce.\n")

    df = df.sort_values(["customer_id", "snapshot_date"])
    prev = df.groupby("customer_id")[product_cols].shift(1)

    # 0 -> 1 only. Rows where prev is NaN are a customer's first month,
    # which is not an acquisition, so they fall out naturally.
    acquired = ((df[product_cols] == 1) & (prev == 0))

    acq_counts = acquired.sum().sort_values(ascending=False)
    total_acq = int(acq_counts.sum())

    print(f"  {'product':32s} {'acquisitions':>14s}")
    print("  " + "-" * 48)
    for c, v in acq_counts.items():
        print(f"  {short(c):32s} {int(v):>14,}")

    print(f"\n  TOTAL ACQUISITION EVENTS: {total_acq:,}")
    print(f"  Customers with 1+ acquisition: "
          f"{df.loc[acquired.any(axis=1), 'customer_id'].nunique():,}")

    if total_acq > 10_000:
        print("\n  VERDICT: plenty of cross-sell events to learn from.")
        print("           The switch to Santander was worth it.")
    else:
        print("\n  VERDICT: fewer events than hoped. Raise N_CUSTOMERS in")
        print("           phase4_5a_translate.py and re-run.")

    report["acquisitions"] = {short(c): int(v) for c, v in acq_counts.items()}
    report["total_acquisition_events"] = total_acq

    # save the events for Phase 5
    ev = df.loc[acquired.any(axis=1),
                ["customer_id", "snapshot_date", "age",
                 "gross_income", "segment"]].copy()
    ev["acquired"] = acquired[acquired.any(axis=1)].apply(
        lambda r: ",".join(short(c) for c in product_cols if r[c]), axis=1)
    ev.to_csv(os.path.join(HERE, "../data/santander_acquisitions.csv"), index=False)
    print(f"\n  Saved ../data/santander_acquisitions.csv ({len(ev):,} rows)")

    # ---------------------------------------------- 4. co-occurrence
    banner("4. CO-OCCURRENCE - which products are held together")

    M = latest[product_cols].values
    counts = M.sum(axis=0)
    both = M.T @ M                       # both[i,j] = customers holding i AND j

    print("\n  P(hold B | hold A) - strongest pairs:\n")
    pairs = []
    for i in range(len(product_cols)):
        for j in range(len(product_cols)):
            if i == j or counts[i] < 100:
                continue
            p = both[i, j] / counts[i]
            if p > 0.3:
                pairs.append((p, short(product_cols[i]),
                              short(product_cols[j]), int(both[i, j])))
    pairs.sort(reverse=True)

    print(f"  {'if customer holds':30s} {'they also hold':30s} {'prob':>6s}")
    print("  " + "-" * 70)
    for p, a, b, n in pairs[:TOP_N_RULES]:
        print(f"  {a:30s} {b:30s} {p:>5.1%}")

    report["cooccurrence_top"] = [
        {"if_holds": a, "also_holds": b, "probability": float(p), "n": n}
        for p, a, b, n in pairs[:TOP_N_RULES]
    ]

    # ------------------------------- 5. sequence rules (your Example B)
    banner("5. SEQUENCE RULES - 'after X, customers add Y'")
    print("\n  This is Example B from your spec: 'customers who take a")
    print("  mortgage add insurance within 60 days'. We can finally test it.\n")

    held_before = (prev == 1)
    acq_np = acquired.values
    held_np = held_before.fillna(False).values

    # Base rate: how often is each product held by ANYONE in a given month.
    # Without this, every rule just says "current_account -> everything",
    # because 65% of customers hold a current account. That is the same
    # popularity trap Phases 1-4 exposed, wearing a different hat.
    base_rate = held_np.mean(axis=0)

    rules = {}
    for j, target in enumerate(product_cols):
        rows = np.where(acq_np[:, j])[0]
        if len(rows) < MIN_RULE_SUPPORT:
            continue
        for i, source in enumerate(product_cols):
            if i == j or base_rate[i] < 1e-6:
                continue
            n_with = int(held_np[rows, i].sum())
            if n_with >= MIN_RULE_SUPPORT:
                conf = n_with / len(rows)
                lift = conf / base_rate[i]      # >1 means better than base rate
                rules[(short(source), short(target))] = (
                    lift, conf, base_rate[i], n_with, len(rows))

    ranked = sorted(rules.items(), key=lambda kv: -kv[1][0])

    print("  Ranked by LIFT = confidence / base rate.")
    print("  Lift 1.0 means the rule tells you nothing you didn't already know")
    print("  from the product's overall popularity. Higher is a real pattern.\n")
    print(f"  {'already had':26s} {'then added':24s} {'conf':>6s} "
          f"{'base':>6s} {'lift':>6s}")
    print("  " + "-" * 74)
    for (src, tgt), (lift, conf, base, n, tot) in ranked[:TOP_N_RULES]:
        print(f"  {src:26s} {tgt:24s} {conf:>5.1%} {base:>5.1%} "
              f"{lift:>5.2f}x  ({n:,}/{tot:,})")

    strong = [r for r in ranked if r[1][0] >= 1.2]
    print(f"\n  Rules with lift >= 1.2x : {len(strong)}")
    if strong:
        print("  -> real cross-sell patterns exist beyond raw popularity.")
    else:
        print("  -> nothing beats base rates yet. Raise N_CUSTOMERS and re-run.")

    report["sequence_rules_top"] = [
        {"had": s, "then_added": t, "confidence": float(c),
         "base_rate": float(b), "lift": float(l), "n": n, "of": tot}
        for (s, t), (l, c, b, n, tot) in ranked[:TOP_N_RULES]
    ]
    report["n_rules_lift_above_1_2"] = len(strong)

    # ------------------------------------------------------------ charts
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

        y = np.arange(len(own))
        ax1.barh(y, own.values[::-1], color="#4C72B0")
        ax1.set_yticks(y)
        ax1.set_yticklabels([short(c) for c in own.index][::-1], fontsize=8)
        ax1.set_xlabel("customers holding")
        ax1.set_title("Product ownership")

        ac = acq_counts[acq_counts > 0]
        y2 = np.arange(len(ac))
        ax2.barh(y2, ac.values[::-1], color="#DD8452")
        ax2.set_yticks(y2)
        ax2.set_yticklabels([short(c) for c in ac.index][::-1], fontsize=8)
        ax2.set_xlabel("acquisition events (0 -> 1)")
        ax2.set_title("Cross-sell events - impossible on LendingClub")

        fig.tight_layout()
        fig.savefig(os.path.join(HERE, "../outputs/phase2_products.png"), dpi=120)
        plt.close(fig)
        print("\n  Saved ../outputs/phase2_products.png")

        rate = both / np.maximum(counts[:, None], 1)
        fig, ax = plt.subplots(figsize=(11, 9))
        im = ax.imshow(rate, cmap="YlOrRd", vmin=0, vmax=1)
        lbl = [short(c) for c in product_cols]
        ax.set_xticks(range(len(lbl)))
        ax.set_yticks(range(len(lbl)))
        ax.set_xticklabels(lbl, rotation=90, fontsize=7)
        ax.set_yticklabels(lbl, fontsize=7)
        ax.set_title("P(holds column | holds row)")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, "../outputs/phase2_cooccurrence.png"), dpi=120)
        plt.close(fig)
        print("  Saved ../outputs/phase2_cooccurrence.png")
    except Exception as exc:
        print(f"  (charts skipped: {exc})")

    # ------------------------------------------------------------- save
    with open(os.path.join(HERE, "../outputs/phase2_analysis.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("  Saved ../outputs/phase2_analysis.json")

    # ---------------------------------------------------------- summary
    banner("WHAT THIS MEANS FOR THE PROJECT")
    print(f"""
  LendingClub                        Santander
  ----------------------------------------------------------------
  1 product per customer             {sizes.mean():.1f} products per customer
  no acquisition history             {total_acq:,} acquisition events
  no user x item matrix              {n_cust:,} x {len(product_cols)} matrix
  Example A only (as a proxy)        Examples A and B both buildable

  Phases 1-4 showed the popularity baseline could not be beaten, because
  loan PURPOSE was not in the features. Here the question is different:
  given what a customer already holds, what comes next. The co-occurrence
  and sequence tables above are the evidence that signal exists.

  Next: Phase 5 builds the real collaborative filtering model on this.
""")

    banner(f"PART B STEP 2.2 COMPLETE  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
