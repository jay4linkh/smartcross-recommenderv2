# SmartCross — project structure

Hybrid financial product cross-selling recommender system.
MBA capstone. Two parts, two datasets, two different questions.

---

## The short version

**Part A** builds a complete recommender on **LendingClub** and establishes,
against a no-ML popularity baseline, that the performance ceiling is in the
*features* rather than the algorithm. Loan *purpose* is not recoverable from
credit attributes, because income, FICO and DTI describe borrowing *capacity*,
not borrowing *intent*.

**Part B** tests that diagnosis on **Santander**, a dataset with the structure
the problem actually requires: 24 product-ownership flags per customer per
month across 17 monthly snapshots. It finds 112,063 acquisition events and 175
sequence rules above 1.2x lift — signal LendingClub could not produce at all.

**Part C** turns the Santander findings into a running system: `web/` is the
whole four-engine pipeline served by one FastAPI process, with a console that
can pull a real held-out customer and mark its own recommendation right or
wrong against what that customer actually did next. Its two engines are fused,
not averaged, and the model reads three months of each customer's holdings
history. Over all 104,167 held-out events the whole pipeline reaches
**85.3% top-3** (95% CI 85.1–85.6%), against about 14% for three random picks
and 70.5% for always recommending the most popular products.

## Quick start

```powershell
git clone https://github.com/jay4linkh/smartcross-recommenderv2.git
cd smartcross-recommenderv2
py -3.12 -m venv .venv ; .venv\Scripts\activate
pip install -r requirements-lock.txt
Expand-Archive data_pack\smartcross_data.zip -DestinationPath . -Force
cd web\backend ; python app.py           # http://127.0.0.1:8000
```

The trained model (`web/backend/model/smartcross.pkl`, 9 MB) is committed, so
there is nothing to train. It was built with the exact versions in
`requirements-lock.txt`; with other numpy / scikit-learn versions, run
`python build_model.py` once (~9 min), or use `.\web\run.ps1`, which notices
and rebuilds on its own.

Every stage in Part A runs on LendingClub. Nothing in Part A touches Santander.

---

## Folder layout

```
RecommenderSysProject/
├── data/raw/
│   ├── user_transaction_data/     <- PART A: LendingClub data + all scripts
│   ├── sector_assets/             <- ETF prices  (used by step 1.5)
│   ├── macro_economic/            <- macro indicators (used by step 1.5)
│   └── stander again lol/         <- raw Kaggle Santander download
│
├── partB_santander/               <- PART B: self-contained
│   ├── scripts/
│   │   ├── phase2_1_translate.py
│   │   ├── phase2_2_analysis.py
│   │   ├── phase2_3_model.py
│   │   ├── phase2_4a_build.py             builds data/sant_events.npz
│   │   └── phase2_4b_train.py
│   ├── data/
│   │   ├── sant_events.npz                563,310 acquisition events
│   │   ├── santander_english.csv          (460 MB, 2.73M rows)
│   │   ├── santander_acquisitions.csv
│   │   └── santander_column_map.json
│   └── outputs/
│       ├── phase2_analysis.json
│       ├── phase2_products.png
│       ├── phase2_cooccurrence.png
│       └── phase2b_output.log
│
├── web/                           <- PART C: the system as a service
│   ├── run.ps1                    one-command start (checks, builds, serves)
│   ├── backend/
│   │   ├── peers.py               the collaborative engine (shared by train + serve)
│   │   ├── build_model.py         trains and measures every engine
│   │   ├── macro.py               regime history from the real price series
│   │   ├── app.py                 FastAPI: the four engines as an API
│   │   └── model/                 smartcross.pkl - the trained model, committed
│   └── frontend/
│       ├── index.html
│       ├── app.css
│       ├── app.js                 every chart drawn by hand, no libraries
│       └── story.js               the scroll story: one real customer, four engines
│
└── defense/                       <- the evidence behind every claim
    ├── cf_experiment.py           how the CF engine and the fusion were chosen
    ├── defense_eval.py            CIs, significance, full pipeline, ablation
    ├── DEFENSE.md                 viva questions, answered from the numbers
    └── outputs/                   defense_results.json + four charts
```

---

## Data

The full raw data is about 12.8 GB and is **not** in this repository: GitHub
caps files at 100 MB, and the Santander data is Kaggle competition data that
may not be redistributed.

**To run the app and reproduce every Part C number, you only need the data
pack (9 MB, committed):**

```powershell
Expand-Archive data_pack\smartcross_data.zip -DestinationPath . -Force
```

It restores, at their original paths:

| file | size | what it is |
|---|---:|---|
| `partB_santander/data/sant_events.npz` | 9 MB | 563,310 acquisition events with 3 months of holdings history — the training/test data for Part C |
| `partB_santander/data/santander_column_map.json` | 3 KB | Spanish → English column map |
| `data/raw/sector_assets/*.csv` | 0.9 MB | 8 ETF / stock price series for the macro regime |
| `data/raw/macro_economic/*.csv` | 0.1 MB | Brent crude (DXY has no usable rows and is skipped) |

**To rerun Parts A and B from scratch**, the two raw source files are in
`data_pack/raw/`, stored with [Git LFS](https://git-lfs.com) (run `git lfs pull`
after cloning). Put them back where the scripts look for them:

| file in `data_pack/raw/` | size | unpack to |
|---|---:|---|
| `santander_train_ver2.csv.zip` | 214 MB | `data/raw/stander again lol/train_ver2.csv` |
| `lendingclub_accepted_2007_to_2018Q4.csv.gz` | 374 MB | `data/raw/user_transaction_data/lending-club/` |
| `lending_club_model_ready.csv.zip` | 42 MB | `data/raw/user_transaction_data/` — what Part A's scripts read |

Everything else under `data/` (cleaned CSVs, translated samples, model-ready
tables) is regenerated by the scripts. The original sources on Kaggle:

| dataset | file | size |
|---|---|---:|
| [Santander Product Recommendation](https://www.kaggle.com/c/santander-product-recommendation/data) | `train_ver2.csv` → `data/raw/stander again lol/` | 2.2 GB (13.6 M rows) |
| [Lending Club loans 2007–2018](https://www.kaggle.com/datasets/wordsforthewise/lending-club) | `accepted_2007_to_2018Q4.csv` → `data/raw/user_transaction_data/lending-club/` | 1.6 GB (2.26 M loans) |

---

## PART A — LendingClub pipeline

Run in order from `data/raw/user_transaction_data/`.
Each step loads the previous step's saved artifacts, so the order matters.

| Step | Script | What it does | Result |
|------|--------|--------------|--------|
| 1.1 | `phase1.py` | Random Forest baseline | 57% accuracy, macro F1 **0.0768** |
| 1.2 | `phase2.py` | Class rebalancing (undersample + SMOTE) | 28% accuracy, macro F1 **0.1181** |
| 1.3 | `phase3.py` | 4 models + top-K metrics + popularity baseline | macro F1 **0.1222** |
| 1.4 | `phase4.py` | Peer collaborative filtering + hybrid blend | macro F1 **0.1337** |
| 1.5 | `phase5.py` | Macro regime engine (sector momentum/volatility) | regime **RECOVERY** |
| 1.6 | `phase6.py` | Eligibility rules + top-3 cards + dashboard | macro F1 **0.1292** |

Filenames keep their original numbering because the scripts are chained by
name (`phase4.py` loads `phase3_best_model.pkl`, and so on). The 1.x labels
above are the report's section numbering.

### Part A results — 100,000 held-out customers

| stage | macro F1 | top-1 | top-3 | products never recommended |
|-------|---------:|------:|------:|---------------------------:|
| 0. Popularity (no ML) | 0.0556 | 0.567 | 0.865 | 12 of 13 |
| 1.3 Content | 0.1222 | 0.341 | 0.796 | 1 |
| 1.4 Peer CF | 0.1025 | 0.554 | 0.874 | 2 |
| 1.4 **Hybrid** | **0.1337** | 0.470 | 0.874 | 2 |
| 1.5 + macro tilt | 0.1335 | 0.463 | 0.872 | 2 |
| 1.6 + eligibility | 0.1292 | 0.441 | 0.775 | 2 |

Random chance for 13 products is 0.0769. The final system reaches **1.68x
chance**, up from 1.00x at step 1.1.

### Reading these numbers honestly

**Accuracy falling from 57% to 28% at step 1.2 was the goal, not a regression.**
The baseline predicted `debt_consolidation` for almost everyone and never
predicted 9 of the 13 products even once. Its macro F1 of 0.0768 is
indistinguishable from random guessing (0.0769). Rebalancing gave up that
borrowed accuracy and bought coverage of every product — which is the only
thing a cross-selling engine can actually use.

**Quote top-3, not top-1.** The product surfaces three recommendations, so
"was the right product on screen" is the question that matches the interface.
Grading a three-slot recommender on its single best guess understates it.

**The popularity baseline reaches 86.5% on top-3 and we do not beat it.**
That is reported rather than hidden, because it is the central finding. But
that baseline shows the *same three products to all 100,000 customers*: its
macro F1 is 0.0556, below chance, and 12 of 13 products are never recommended.
It cannot cross-sell at all. Comparable top-3, fundamentally different
capability.

**Steps 1.5 and 1.6 are not expected to raise accuracy.** The macro tilt
reranks an existing list and the eligibility filter removes options; neither
improves prediction. Their measured deltas (−0.0002 and −0.0043 macro F1) are
reported as measured. The tilt changes the top recommendation for 2.0% of
customers, which is the point: the same engine recommends a different mix in a
downturn than in an expansion.

### Rule-book validation (step 1.6)

The eligibility rules are checked against reality by measuring how often they
would block the product a customer *actually took*:

- Overall: **13.0%**
- `renewable_energy` 46.8%, `vacation` 39.8%, `house` 36.0% — the deliberate
  discretionary affordability gates
- `major_purchase`, `medical`, `other`, `moving` — all under 1%

This check caught a real bug: `home_ownership_numeric` is a four-way code where
`1` is the renting category and `0`/`2` are ownership categories. A naive `> 0`
test was blocking genuine homeowners and killed 79% of actual
`home_improvement` borrowers. After the fix that product blocks 8.5%.

---

## PART B — Santander validation

Run in order from `partB_santander/scripts/`.

| Step | Script | What it does |
|------|--------|--------------|
| 2.1 | `phase2_1_translate.py` | Reads `train_ver2.csv` (2.29 GB), renames all 48 columns and value codes to English, samples customers keeping all 17 months each |
| 2.2 | `phase2_2_analysis.py` | Portfolio sizes, ownership imbalance, acquisition events, co-occurrence, lift-ranked sequence rules |

### Why sampling is done by customer, not by row

A random sample of *rows* would return March for one customer and September for
another, destroying the acquisition history. Cross-selling is about sequence —
"held a mortgage in March, added a pension in April" — so every month is kept
for whichever customers are sampled.

### Part B results

- **191,102 customers x 24 products x 17 months** (2,725,768 rows)
- **112,063 acquisition events** — the 0 -> 1 flips, impossible on LendingClub
- 25.5% of customers hold 2+ products
- Ownership imbalance 37,474:1 — the same discipline as step 1.2 applies

**Top sequence rules by lift** (lift = confidence / base rate):

| already held | then added | confidence | base rate | lift |
|---|---|---:|---:|---:|
| short_term_deposits | long_term_deposits | 7.5% | 0.2% | **40.7x** |
| mas_particular_account | e_account | 29.3% | 0.9% | **32.5x** |
| long_term_deposits | funds | 40.2% | 4.1% | **9.8x** |
| payroll_account | pensions | 67.6% | 7.5% | **9.0x** |
| payroll_account | payroll | 65.4% | 7.5% | **8.7x** |
| payroll_account | credit_card | 45.1% | 7.5% | **6.0x** |
| mortgage | credit_card | 3.6% | 0.6% | **6.2x** |

**175 rules exceed 1.2x lift.**

### Why rules are ranked by lift, not confidence

Confidence alone would put `current_account -> everything` at the top simply
because 58.8% of customers hold a current account — the same popularity trap
step 1.3 exposed, in rule form. Dividing by each product's base rate means only
patterns that beat popularity survive the ranking.

---

## Architecture mapping

The project specification defines four components. Where each one lives:

| Component | Implemented in |
|---|---|
| 1. Personalized Recommendation Engine | Steps 1.3–1.4 (content + peer CF + hybrid) |
| 2. Macro Regime Engine | Step 1.5 |
| 3. Knowledge-Based Constraints | Step 1.6 |
| 4. Final Output Layer (top-3 cards) | Step 1.6 + `phase6_dashboard.html` |

All four are re-implemented against the Santander data in Part C, where they
run as a live service rather than a batch script: the recommendation engine as
two independently-scored engines plus a blend, the regime engine as
`macro.py` classifying from real price series, the constraints as one auditable
rule dictionary, and the output layer as the **Advise** view of the console.

The three scenarios from the specification:

- **Example A — peer group match**: step 1.4, as a demographic proxy for
  collaborative filtering; Part B co-occurrence for the real version
- **Example B — next best step**: not buildable on LendingClub (one product per
  customer, no time dimension); Part B step 2.2 sequence rules
- **Example C — risk and eligibility guardrails**: step 1.6 rule book

---

## Environment

```
pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib joblib pyarrow
```

Each script has a single CONFIG block at the top with RAM-tunable settings
(`SUBSAMPLE`, `PEER_POOL`, `N_CUSTOMERS`). Paths resolve automatically relative
to the script, so no path editing is needed when files sit in their folders.

---

## PART C — the system as a service

`web/` is the whole pipeline running as one FastAPI process, serving a trained
model and a hand-written console at `http://127.0.0.1:8000`.

```powershell
pip install -r requirements-lock.txt
cd web\backend
python app.py             # serves http://127.0.0.1:8000 with the committed model
python build_model.py     # optional: retrains and re-measures every engine, ~9 min
```

Or from the project root, `.\web\run.ps1` — it checks the packages, verifies
the committed model actually loads in *your* Python, rebuilds it only if it
does not, and starts the server.

**Why versions matter.** A pickled model only opens under numpy and
scikit-learn close to the ones that wrote it. The committed model records its
own versions (numpy 1.26.4, scikit-learn 1.4.2); `requirements-lock.txt` pins
them, `app.py` warns on a mismatch, and `run.ps1` rebuilds automatically.

### Part C results — 104,167 held-out events, 22 products

Chance is 4.55% for top-1 and macro F1 (1 in 22) and about 13.6% for top-3
(3 in 22). Every row comes from one `build_model.py` run on one time-based
split: trained on 2015-02 → 2016-02, tested on 2016-03 → 2016-05, every table
the API serves computed from the training months only. Every engine is scored
the way the API serves — **products the customer already holds are never
recommended** — and where an engine gives two products the same score the tie
counts against it.

| engine | macro F1 | top-1 | top-3 | MAP@7 |
|---|---:|---:|---:|---:|
| Popularity (no ML) | 0.0624 | 0.388 | 0.705 | 0.580 |
| Item–item CF, lift (first version) | 0.0261 | 0.031 | 0.312 | 0.208 |
| Collaborative, portfolio pattern | 0.1741 | 0.572 | 0.850 | 0.720 |
| Content only (HistGB) | 0.2648 | 0.589 | 0.889 | 0.744 |
| Content only (XGBoost) | 0.2704 | 0.591 | 0.889 | 0.745 |
| Weighted blend 0.5/0.5 (first version) | 0.2641 | 0.587 | 0.874 | 0.734 |
| Fused hybrid, no history (previous version) | 0.2770 | 0.589 | 0.890 | 0.744 |
| **Fused hybrid + history — served** | **0.3550** | **0.635** | **0.904** | **0.775** |

### How the model got here

1. **The collaborative engine ranks by probability.** The first version ranked
   by lift, which measures surprise rather than likelihood and put rare
   products first: 31.2% top-3, below popularity. "Customers with exactly this
   portfolio added X", shrunk toward single-product co-occurrence when the
   portfolio is rare, reaches 84.8–85.0% on its own.
2. **Fusion, not averaging** (feature augmentation, in Burke's hybrid
   taxonomy). The first hybrid averaged the two engines 50/50 and was measurably
   *worse* than content alone (−1.44 pts top-3). Now the collaborative engine's
   44 scores are inputs to the content engine, and for each training month they
   come from strictly earlier months, so no row's own outcome leaks in.
3. **Three months of holdings history** (48 features): what the customer held
   one observation earlier — so a product dropped last month is visible — and
   how many months in a row they have held each product. Chosen on the
   validation month (2016-02) before the test months were scored
   (`defense/improve_experiment.py`).

Lift is still the right number for an *explanation* ("payroll holders add
pensions 9× more often"), so the console keeps it for reasons — but only for
rules backed by at least 30 training customers.

### Is the improvement real?

`defense/defense_eval.py`: 1,000 bootstrap resamples of the held-out customers,
every engine on identical resamples, a difference called real only when its 95%
interval excludes zero. The console shows the same tables on its Performance view.

| comparison | Δ top-3 | Δ macro F1 | verdict |
|---|---:|---:|---|
| **Served (with history) vs fused without** | **+1.46 pts** | **+0.078** | **real gain on both** |
| Served vs content only | +1.58 pts | +0.090 | real gain on both |
| Served vs previous 50/50 blend | +3.02 pts | +0.091 | real gain on both |
| Served vs popularity | +20.0 pts | +0.293 | real gain on both |
| Pattern CF vs lift CF | +53.6 pts | +0.148 | real gain |
| Previous blend vs content only | −1.44 pts | −0.001 | real loss on top-3 |
| XGBoost vs HistGB (content only) | +0.09 pts | +0.006 | within noise |

Against v2 the history features add **+4.6 pts top-1** as well (58.9% → 63.5%).

**Where the signal comes from** (the same booster on each input group alone):
holdings 85.6% top-3, demographics + behaviour 84.0%, peer scores 86.0%,
**holdings history 88.3%** — the strongest single group — and all 127 inputs
together 90.4%.

**Stable across months:** 90.8%, 89.5%, 90.9% top-3 for 2016-03, -04, -05;
popularity 71.2%, 68.7%, 71.4%. **Well calibrated:** expected calibration error
1.0% — when the model says 60%, it is right about 60% of the time.

**Tried and rejected,** each judged on the validation month or measured on test:

| idea | result | why rejected |
|---|---|---|
| Class-weighted training | top-3 81.9%, macro F1 0.301 | real loss on both |
| Per-product decision thresholds | validation macro F1 0.343 → 0.392 | cost 0.7 pts of top-3 (limit 0.2) |
| Hyperparameter search (5 settings) | best within 0.04 pts top-3 | lower macro F1; current settings kept |
| XGBoost on the GPU (RTX 3060) | top-3 level, 3–4× faster to train | lower macro F1 (0.320–0.340 vs 0.343) |

### Validated against ground truth, end to end

The whole four-layer pipeline — fused model, macro tilt (regime classified from
the price series for each test month), eligibility rules — over **every one of
the 104,167 held-out events**, not a sample:

| | |
|---|---:|
| top-3 after the macro tilt | 90.4% |
| **top-3 after the rules (what the customer experiences)** | **85.3%** (95% CI 85.1–85.6%) |
| top-1 after the rules | 62.9% |
| the product they actually took was **removed by a rule** | **10.8%** (0.4% because data was missing) |
| macro tilt changes the #1 product for | 1.7% of customers |
| every product blocked (fallback path) | 0.0% |

**Quote 85.3%, not 90.4%.** 90.4% is the model with no constraints applied; it
is not what a customer standing at the counter would experience. The gap is the
cost of hard eligibility gates, reported rather than hidden.

### The macro regime is classified, not chosen

`web/backend/macro.py` reads the price files already in `data/raw/` and builds a
month-by-month regime history — nine usable series, 2015-07 to 2026-08, 134
months. (`dxy_usdollar.csv` has no usable rows and is reported as skipped, not
silently dropped.)

| | |
|---|---|
| momentum | 6-month change in the monthly close |
| volatility | annualised stdev of the last 6 monthly returns |
| threshold | **each series' own median volatility** across its history — not a number picked to make the output look good |
| vote | one per series per month, majority wins, ties broken by the sign of mean momentum |

| regime | months | share |
|---|---:|---:|
| EXPANSION | 57 | 42.5% |
| RECOVERY | 41 | 30.6% |
| STRESS | 24 | 17.9% |
| SLOWDOWN | 12 | 9.0% |

The console defaults to **Auto**, which classifies the regime for whichever
month you pick. The four named regimes stay selectable and the UI labels those
as *a scenario, not a measurement*; every recommendation carries a
`regime_source` saying which it was, how many series agreed, and which dissented.

**A finding worth stating in the viva:** the Santander window
(2015-08 → 2016-06) classifies almost entirely as **STRESS / SLOWDOWN** — the
2015–16 oil collapse and China slowdown. The cross-sell behaviour this model
learned was learned under stress conditions, which is precisely the argument for
having a regime layer at all: applying those patterns unchanged during an
expansion would push the wrong products.

Run it standalone to see the history: `cd web\backend && python macro.py`

### Real records are not tidy

Roughly **28% of customers have no income on file** (the dataset stores `-1`),
seniority uses `-999999`, and ages run from 2 to 110. A hard eligibility gate
must not read "unknown" as "failed the threshold", nor as "passed". So each rule
declares the fields it depends on, a missing field produces its own reason
("income at least 35,000 — income not on file"), and the response carries a
`missing_fields` list that the console surfaces.

### The console

Five views, one process.

- **Advise** — opens with a scroll story: a real held-out customer walks into
  the bank, their record becomes the profile vector, and the four engines play
  out on their actual scores, ending with what they really took next. Then the
  intake form, *draw a real customer* (marked correct / missed / removed-by-rule),
  the gate-by-gate walk-through, the top-3 cards, and **Why this answer**: every
  product in plain words — recommended and why, removed and by which rule,
  everything else and where it ranked. The α slider re-scores live and says
  what moved.
- **Inside the engines** — the architecture as an interactive diagram: the
  collaborative engine feeding the content engine, the market data and rule book
  feeding their stages, a measured number on every box, and the train/test split.
- **Performance** — every engine on the same split, plus **Is the improvement
  real?**: confidence intervals, paired significance tests, ablation, month
  stability, and the ideas that were tried and rejected.
- **Evidence** — the strongest well-supported sequence rules (each with how many
  customers stand behind it) and the full item–item lift matrix as a heatmap.
- **Session** — every profile advised since the server started.

`web/README.md` has the API table. `SYSTEM_DESIGN.md` has the component design.

---

## Future work

Everything the previous version listed as "not done yet" has now been tried and
measured: holdings history was kept (above), class weighting, per-product
thresholds, a hyperparameter search and XGBoost on the GPU were rejected on the
numbers, and library versions are pinned. What remains needs things this
project does not have:

1. **A real credit policy.** The eligibility thresholds in `app.py` are
   illustrative retail-credit norms. In a deployment they come from the bank's
   own policy — which is exactly why they live in one readable dictionary a
   compliance officer can audit without reading the model — and the 10.8% of
   real acquisitions they remove should then approach zero.
2. **Rare products need more data, not a better model.** Macro F1 is 0.355
   (7.8× chance); over the 13 products above a 0.5% base rate it is **0.434**.
   Mortgages, loans and home accounts have 10–20 test events each — too few to
   learn at this sample size.
3. **Longer history and live feedback.** The dataset spans 17 months. A bank
   with years of history, and with outcomes of the recommendations it actually
   showed, could learn sequences and measure lift from the console itself.
