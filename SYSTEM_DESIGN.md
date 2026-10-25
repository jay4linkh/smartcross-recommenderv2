# SmartCross — System Design Document

**Smart Financial Cross-Selling Recommender System**
MBA Capstone · NMIMS Mumbai

---

## 1. Objective

### 1.1 Problem statement

Acquiring a new banking customer costs several times more than selling an
additional product to an existing one, yet most banks still push the same
generic product list to everybody. The result is a click-through rate around
1%, customer irritation, and no lift in lifetime value.

### 1.2 Project objective

> **Build a hybrid recommender system that, for any given customer, returns a
> ranked top-3 list of financial products they are genuinely likely to take
> next — personalised to their behaviour, adjusted for the macroeconomic
> regime, and filtered by eligibility rules so that every recommendation is
> one the bank could actually approve.**

### 1.3 Success criteria

| Criterion | Target | Achieved (104,167 held-out events) |
|---|---|---|
| Beat a no-ML popularity baseline | required | **yes** — +18.5 pts top-3, 95% CI [+18.2, +18.8] |
| Top-3 hit rate, whole pipeline | > 80% | **85.3%** (model alone 90.4%) |
| Lift over random chance (macro F1) | > 3x | **7.8x** (0.355) |
| Both engines useful, hybrid beats its parts | required | yes — CF 84.8% alone; fused > content (macro F1 real gain) |
| Explainable output (a reason per recommendation) | required | yes |
| Eligibility rules validated, not just asserted | required | yes — 10.8% of real acquisitions blocked, reported |

### 1.4 Scope boundary

In scope: ranking which products to show. Out of scope: credit approval
decisions, pricing, and limit-setting. The system recommends; it does not
underwrite.

---

## 2. Folder and file layout

```
RecommenderSysProject/
│
├── README.md                             project overview + results
├── SYSTEM_DESIGN.md                      this document
│
├── data/raw/
│   ├── user_transaction_data/            ◄── PART A  (LendingClub)
│   │   ├── lending_club_model_ready.csv      1.95M rows, 16 features
│   │   │
│   │   ├── phase1.py                         1.1  baseline
│   │   ├── phase2.py                         1.2  rebalancing
│   │   ├── phase3.py                         1.3  multi-model + baseline
│   │   ├── phase4.py                         1.4  peer CF + hybrid
│   │   ├── phase5.py                         1.5  macro regime engine
│   │   ├── phase6.py                         1.6  rules + dashboard
│   │   │
│   │   ├── phase3_split/                     frozen train/test split
│   │   ├── phase2_rf_model.pkl               trained artifacts
│   │   ├── phase3_best_model.pkl
│   │   ├── phase4_recommender.pkl            ◄ the deployable engine
│   │   ├── phase5_regime.json                current regime + tilts
│   │   │
│   │   ├── phase6_dashboard.html             ◄ open this
│   │   ├── phase6_cards.json                 sample top-3 cards
│   │   ├── phase*_results.json               all metrics
│   │   └── phase*.png                        charts for the report
│   │
│   ├── sector_assets/                    ETF prices  → Macro Regime Engine
│   │   └── XLE, XLK, EEM, AAPL, MSFT, JPM, BAC, GS
│   ├── macro_economic/                   macro indicators → Macro Regime Engine
│   └── stander again lol/                raw Kaggle Santander download
│       └── train_ver2.csv                    2.29 GB, 13.6M rows
│
├── web/                                 ◄── PART C  the system as a service
│   ├── backend/peers.py                      collaborative engine (train + serve)
│   ├── backend/build_model.py                trains and measures every engine
│   ├── backend/app.py                        FastAPI: the four layers as an API
│   └── frontend/                             console + scroll story, no libraries
│
├── defense/                              ◄── evidence: CIs, significance, ablation
│   ├── cf_experiment.py                      how CF and the fusion were chosen
│   ├── defense_eval.py                       measures the served pipeline
│   └── DEFENSE.md                            viva questions, answered
│
└── partB_santander/                      ◄── PART B  (Santander)
    ├── scripts/
    │   ├── phase2_1_translate.py             2.1  Spanish → English
    │   ├── phase2_2_analysis.py              2.2  acquisition events + lift
    │   ├── phase2_3_model.py                 2.3  next-product model
    │   ├── phase2_4a_build.py                2.4a full event table (streaming)
    │   └── phase2_4b_train.py                2.4b full-data training
    ├── data/
    │   ├── santander_english.csv             460 MB, 2.73M rows
    │   ├── sant_events.npz                   563,310 acquisition events
    │   ├── santander_acquisitions.csv
    │   └── santander_column_map.json         ES→EN mapping (report appendix)
    └── outputs/
        ├── phase2_analysis.json              ownership, co-occurrence, lift
        ├── phase2_4_results.json             final model metrics
        ├── phase2_products.png
        ├── phase2_cooccurrence.png
        ├── phase2_4_comparison.png
        └── phase2_4_importance.png
```

**Why two parts.** Part A is the diagnostic: a complete pipeline on LendingClub
that establishes, against a no-ML baseline, that the ceiling is in the features
rather than the algorithm. Part B is the working engine, built on data with the
structure the problem actually requires. Part A motivates Part B; neither is
redundant.

---

## 3. System architecture

### 3.1 Layer diagram

```
┌──────────────────────────────────────────────────────────────────┐
│  INPUT                                                           │
│  customer profile  ·  current product holdings  ·  today's date  │
└────────────────────────────┬─────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│  LAYER 1 — PROFILE MATRIX                          (feature prep)│
│                                                                  │
│  demographics      age, income, seniority, segment, activity     │
│  holdings          24 binary ownership flags                     │
│  behavioural lag   months_since_last_acquisition                 │
│                    acquisitions_so_far, months_observed          │
│                    n_products_held                               │
│  history (48)      held one observation earlier (24 flags)       │
│                    months each product held in a row (24)        │
│                                                                  │
│  OUT: 35 + 48 = 83-dimensional feature vector                    │
└────────────────────────────┬─────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│  LAYER 2 — PERSONALISED RECOMMENDATION ENGINE   (two engines,    │
│                                                   fused)         │
│   ┌──────────────────────────┐                                   │
│   │  COLLABORATIVE ENGINE    │  holdings only                    │
│   │  P(add j | exact         │  "customers with this portfolio   │
│   │    portfolio), shrunk    │   added j next"                   │
│   │  toward mean_i P(j | i)  │  84.8% top-3 on its own           │
│   └────────────┬─────────────┘                                   │
│                │ 44 peer scores (22 co-occurrence + 22 pattern)  │
│                ▼                                                 │
│   ┌──────────────────────────┐                                   │
│   │  CONTENT ENGINE          │  HistGradientBoosting             │
│   │  83 profile + 44 peer    │  learns per customer how far to   │
│   │  = 127 inputs            │  trust the peers                  │
│   └──────────────────────────┘  90.4% top-3, macro F1 0.355      │
│                                                                  │
│  Fusion is feature augmentation, not a weighted average. Peer    │
│  features for each training month come from earlier months only. │
│  OUT: probability for each of 22 products                        │
└────────────────────────────┬─────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│  LAYER 3 — MACRO REGIME ENGINE                      (reranking)  │
│                                                                  │
│  9 price series ──► 6-month momentum  ┐                          │
│  (ETFs, stocks)     realised vol      ├─► vote per series        │
│                     own-median cutoff ┘   EXPANSION / RECOVERY   │
│                                           SLOWDOWN  / STRESS     │
│                             │                                    │
│                             ▼                                    │
│              per-product tilt multipliers                        │
│              e.g. STRESS → current account ×1.30, funds ×0.70    │
│                                                                  │
│  OUT: macro-adjusted scores  (#1 changes for 1.4%; top-3 same)   │
└────────────────────────────┬─────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│  LAYER 4 — KNOWLEDGE-BASED CONSTRAINTS            (hard filter)  │
│                                                                  │
│  one auditable rule per gated product (13 of 22):                │
│    mortgage     age 25-65 ∧ income ≥ 30k ∧ seniority ≥ 24 mo     │
│    funds        income ≥ 35k  (suitability)                      │
│    pensions     requires a payroll account                       │
│  missing data ("income not on file") never passes a rule         │
│                                                                  │
│  ineligible products are REMOVED, not down-weighted              │
│  validated: removes the product actually taken 10.8% of the time │
│                                                                  │
│  OUT: eligible, ranked scores                                    │
└────────────────────────────┬─────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│  LAYER 5 — OUTPUT                                                │
│  top-3 cards: product · score · eligibility · reason · next step │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 Why this order

Scoring → tilting → filtering, never the reverse.

Filtering first would hide data from the model and bias the learned
probabilities. Tilting after filtering would boost products the customer cannot
have. Each layer narrows the previous one's output, and the hard constraint
runs last so that nothing downstream can re-admit an ineligible product.

---

## 4. Interface contract

### 4.1 Input

```json
{
  "customer_id": "C1050611",
  "as_of_date": "2016-05-28",
  "demographics": {
    "age": 32,
    "gross_income": 75000,
    "seniority_months": 48,
    "segment": "Individual",
    "is_active": 1,
    "is_new_customer": 0
  },
  "holdings": {
    "current_account": 1,
    "payroll_account": 1,
    "credit_card": 1,
    "pension_plan": 0
  }
}
```

Everything in Layer 1's behavioural block is derived server-side from the
customer's holdings history; the caller does not supply it.

### 4.2 Output

```json
{
  "customer_id": "C1050611",
  "macro_regime": "RECOVERY",
  "recommendations": [
    {
      "rank": 1,
      "product": "pensions",
      "score": 0.412,
      "eligible": true,
      "macro_tilt": 1.00,
      "reason": "Customers holding a payroll account add this 9.0 times more often than the base rate."
    },
    {
      "rank": 2,
      "product": "direct_debit",
      "score": 0.187,
      "eligible": true,
      "macro_tilt": 1.00,
      "reason": "Strong co-occurrence with existing payroll and card holdings."
    },
    {
      "rank": 3,
      "product": "long_term_deposits",
      "score": 0.094,
      "eligible": true,
      "macro_tilt": 1.02,
      "reason": "Profile matches customers who added term deposits; mildly boosted by the RECOVERY regime."
    }
  ],
  "filtered_out": ["mortgage"],
  "model_version": "2.4"
}
```

Every recommendation carries a reason. Explainability was a stated success
criterion, so a bare score is not an acceptable output.

---

## 5. Results

### 5.1 Part A — LendingClub (100,000 held-out customers, 13 products)

| stage | macro F1 | top-1 | top-3 | unreachable products |
|---|---:|---:|---:|---:|
| Popularity (no ML) | 0.0556 | 0.567 | 0.865 | 12 of 13 |
| 1.3 Content | 0.1222 | 0.341 | 0.796 | 1 |
| 1.4 Peer CF | 0.1025 | 0.554 | 0.874 | 2 |
| **1.4 Hybrid** | **0.1337** | 0.470 | 0.874 | 2 |
| 1.5 + macro tilt | 0.1335 | 0.463 | 0.872 | 2 |
| 1.6 + eligibility | 0.1292 | 0.441 | 0.775 | 2 |

Chance = 0.0769. Final = **1.68x chance**, up from 1.00x at step 1.1.

**Finding.** No configuration beats the popularity baseline on top-3. Loan
*purpose* is not recoverable from credit features, because income, FICO and DTI
describe borrowing *capacity*, not *intent*. Four different algorithms plus a
structurally different CF method converge on the same wall, which locates the
limit in the data rather than the model.

### 5.2 Part B/C — Santander (563,310 events, 22 products, time-based split)

Test months 2016-03 to 2016-05 (104,167 events), never seen in training. Held
products masked for every engine; ties count against the engine.

| engine | macro F1 | top-1 | top-3 | MAP@7 |
|---|---:|---:|---:|---:|
| Popularity (no ML) | 0.0624 | 0.388 | 0.705 | 0.580 |
| Item-item CF, lift (first version) | 0.0261 | 0.031 | 0.312 | 0.208 |
| Collaborative, portfolio pattern | 0.1741 | 0.566 | 0.848 | 0.715 |
| Content only (HistGB) | 0.2648 | 0.589 | 0.889 | 0.744 |
| Weighted blend 0.5/0.5 (first version) | 0.2641 | 0.587 | 0.874 | 0.734 |
| Fused hybrid, no history (previous version) | 0.2770 | 0.589 | 0.890 | 0.744 |
| **Fused hybrid + history (served)** | **0.3550** | **0.635** | **0.904** | **0.775** |
| + macro tilt + rules = **full pipeline** | 0.3400 | 0.629 | **0.853** | 0.740 |

Chance = 0.0455 for macro F1 and top-1, ~13.6% for top-3. Served = **7.8x
chance** on macro F1; full pipeline **+14.8 points** of top-3 over popularity
even after the rules remove 10.8% of true answers.

### 5.3 Reading these numbers

**Top-3 is the operative metric.** The interface shows three recommendations,
so "was the right product on screen" is the question that matches the product.
Top-1 and MAP@7 are reported beside it.

**Macro F1 is not comparable across different class counts.** Part B step 2.3
scored 18 products and step 2.4 scored 22; the apparent drop is an artifact of
four additional rare classes entering the average. Read it against chance.

**The first hybrid lost to its own content engine** (−1.44 pts top-3, a real
loss in a paired bootstrap), because its collaborative half ranked by lift —
surprise, not likelihood. Fixing the CF engine and fusing instead of averaging
turned that into a small gain: +0.12 pts top-3 (borderline, p 0.046) and
+0.012 macro F1 (clear). Stated as measured, not rounded up.

**New information beat more tuning.** 5x more training data had moved top-1 by
only 0.6 points (2.3 → 2.4), and a hyperparameter search, per-product
thresholds and XGBoost on the GPU found nothing better. Three months of holdings
history — new information, not more rows — added +4.6 pts top-1, +1.46 pts
top-3 and +28% macro F1, every gain with an interval excluding zero.

---

## 6. Design decisions and their justification

| Decision | Why |
|---|---|
| Macro F1 as primary metric | Accuracy is meaningless under 917:1 imbalance; 57% hid 9 dead classes |
| Hybrid resampling, not pure SMOTE | 917:1 would have inflated training data to ~11.5M rows |
| Resample training only | Rebalancing the test set would score the model on a fantasy distribution |
| Time-based split in Part B | A random split lets the model see a customer's future and predict their past |
| Lift, not confidence, for rules | Confidence alone ranks `current_account → everything` first, since 58.8% hold one |
| Rules remove, not down-weight | A compliance constraint that can be outvoted by a score is not a constraint |
| Rule book validated against outcomes | Caught a real bug: `home_ownership_numeric` is a 4-way code, not a flag |
| Popularity baseline at every stage | Without it, 0.134 macro F1 looks like success instead of a ceiling |
| Baseline scored fairly | Popularity must not recommend products already held; unmasked it scores 56.9% instead of 70.5% and flatters everything above it |
| CF ranks by probability, lift only explains | Lift ranks rare products first; as a ranker it scored 31.2% top-3, below popularity |
| Fuse, do not average | A fixed 50/50 average measurably lost to content alone; feeding CF scores in as features lets the model weigh peers per customer |
| Choose on a validation month | CF smoothing and the fusion were picked on 2016-02; the test months were scored once |
| Peer features from earlier months only | Otherwise each training row's own outcome leaks into its inputs |
| Rules quoted only with 30+ customers behind them | A 45x lift from four customers is an anecdote, not a reason |
| Holdings history as features | The strongest single input group (88.3% top-3 alone); chosen on the validation month |
| Ship the trained model with a version lock | Collaborators run the app without retraining; `run.ps1` rebuilds only on a version mismatch |

---

## 7. Known limitations

1. **Part A does not beat its baseline**, and this is reported rather than
   hidden. It is the finding that justifies Part B.
2. **Rare products stay hard.** Macro F1 is 0.355 overall and 0.434 over the
   13 products above a 0.5% base rate; mortgages, loans and home accounts have
   10–20 test events each, too few to learn at this sample size.
3. **Eligibility thresholds are illustrative** retail-credit norms, not a
   bank's credit policy. They remove 10.8% of the products customers actually
   took. In deployment they would come from the policy document and that figure
   should approach zero.
4. **The macro tilt lowers measured accuracy slightly** (−0.0002 macro F1). It
   buys business relevance, not predictive gain, and is reported as measured.
5. **Two datasets, two geographies.** LendingClub (US) and Santander (Spain)
   cannot be merged; they answer different questions and are kept separate.

---

## 8. Reproduction

```powershell
# Part A
cd data\raw\user_transaction_data
python phase1.py ; python phase2.py ; python phase3.py
python phase4.py ; python phase5.py ; python phase6.py

# Part B
cd ..\..\..\partB_santander\scripts
python phase2_1_translate.py
python phase2_2_analysis.py
python phase2_4a_build.py
python phase2_4b_train.py
```

```powershell
# Part C -- the served system
cd ..\..\web\backend
python build_model.py          # ~7 min: every engine, one split
python app.py                  # http://127.0.0.1:8000

# The evidence behind every number above
cd ..\..\defense
python cf_experiment.py        # how the CF engine and fusion were chosen
python defense_eval.py         # ~15 min: CIs, significance, ablation
```

```
pip install -r requirements.txt
```

Each script has one CONFIG block at the top with RAM-tunable settings. Paths
resolve relative to the script, so no editing is required.
