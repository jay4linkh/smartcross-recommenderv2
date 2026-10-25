# SmartCross — the recommender as a working web application

A Multi-Engine Recommender System for Financial Cross-Selling.
One customer profile in, a ranked and explainable top-3 out, each with an
eligibility check and a plain-language reason.

Everything runs on your own machine. No internet, no CDN, no API key.

---

## Run it

Two commands, from this folder.

```powershell
# 1. install the server packages (once)
pip install fastapi uvicorn pydantic scikit-learn joblib numpy

# 2. build the model the API serves (once, ~2-4 minutes)
cd backend
python build_model.py

# 3. start it
python app.py
```

Then open **http://127.0.0.1:8000**

Or use the helper script from the project root:

```powershell
.\web\run.ps1
```

`run.ps1` checks the packages, builds the model if `backend/model/smartcross.pkl`
is missing, and starts the server.

---

## What build_model.py does

It reads `partB_santander/data/sant_events.npz` — the 563,310 real acquisition
events built in phase 2.4a — and produces one artefact, `backend/model/smartcross.pkl`:

| key | what it is |
|---|---|
| `clf` | HistGradientBoostingClassifier, the content engine |
| `products` | the 22 products with enough events to learn |
| `base_rate` | each product's share of observed acquisitions |
| `lift` | the item–item lift matrix, `P(add j | holds i) / base(j)` |
| `metrics` | the held-out scores, so the API reports measured numbers |

**The split is by time, not at random.** The last three months are held out
entirely; the model only ever sees earlier months. The lift matrix is computed
from the training months too — computing it over all months would leak the test
period into something the API serves as if it had been learned beforehand.

---

## The four engines, in order

```
customer profile
      │
      ▼
┌──────────────────┐   24 holdings flags + 11 behavioural / demographic columns
│ 1  profile matrix│   months_since_last_acq and acquisitions_so_far matter
└────────┬─────────┘   more than income
         ▼
┌──────────────────┐   content:  P(next product | profile), gradient boosting
│ 2  recommendation│   CF:       Σ lift[held, j] over the item–item matrix
└────────┬─────────┘   blend:    α·content + (1−α)·CF,  α = 0.5
         ▼
┌──────────────────┐   9 real price series → momentum × volatility →
│ 3  macro regime  │   EXPANSION / RECOVERY / SLOWDOWN / STRESS, each with
└────────┬─────────┘   per-product multipliers
         ▼
         ▼
┌──────────────────┐   one readable predicate per product; a product that
│ 4  constraints   │   fails is REMOVED, never down-weighted
└────────┬─────────┘
         ▼
   ranked top-3, each with a reason and the rule it passed
```

Order is not an implementation detail. Filtering before scoring would bias the
probabilities. Tilting after filtering would promote products the customer
cannot have. The hard constraint runs last so nothing downstream can re-admit
an ineligible product.

---

## The macro regime is classified, not chosen

`backend/macro.py` reads the price files already in the project and turns them
into a month-by-month regime history:

```
data/raw/macro_economic/   brent_crude_oil.csv
data/raw/sector_assets/    XLK, XLE, EEM, JPM, BAC, GS, AAPL, MSFT
```

Nine usable series, **2015-07 to 2026-08, 134 months**. (`dxy_usdollar.csv` has
no rows and is skipped — reported, not silently dropped.)

| | |
|---|---|
| momentum | 6-month change in the monthly close |
| volatility | annualised stdev of the last 6 monthly returns |
| threshold | **each series' own median volatility** across its history — not a number picked to make the output look good |
| vote | one vote per series per month, majority wins, ties broken by the sign of mean momentum |

| regime | months | share |
|---|---|---|
| EXPANSION | 57 | 42.5% |
| RECOVERY | 41 | 30.6% |
| STRESS | 24 | 17.9% |
| SLOWDOWN | 12 | 9.0% |

The console defaults to **Auto**, which classifies the regime from these series
for whichever month you pick. The four named regimes are still selectable, and
the UI labels those as *a scenario, not a measurement*. Every recommendation
carries a `regime_source` saying which it was, how many series agreed, and which
ones dissented.

**A finding worth stating in the viva:** the Santander window
(2015-08 → 2016-06) classifies almost entirely as **STRESS / SLOWDOWN** — the
2015–16 oil collapse and China slowdown. The cross-sell behaviour the model
learned was learned under stress conditions. That is precisely the argument for
having a regime layer at all: applying those learned patterns unchanged during
an expansion would push the wrong products.

Run it standalone to see the history:

```powershell
cd backend
python macro.py
```

---

## Measured results

Held out: the last three months. 22 products, so chance is 4.55%.

**Every row below is produced by `build_model.py` in one run, on one split.**
Nothing is quoted from an earlier experiment. If XGBoost is not installed its
row is absent rather than filled in from memory.

| engine | macro F1 | top-1 | top-3 | MAP@7 |
|---|---|---|---|---|
| Popularity (no ML) | 0.0224 | 0.289 | 0.569 | 0.482 |
| Item–item CF | 0.0254 | 0.030 | 0.311 | 0.206 |
| Content (HistGB) — *served* | 0.2648 | 0.589 | 0.889 | 0.744 |
| Content (XGBoost) | **0.2705** | **0.591** | **0.889** | **0.745** |
| Hybrid (α = 0.5) | 0.2641 | 0.587 | 0.874 | 0.734 |

### What to say about these numbers, and what not to

Five things are true and should be stated rather than smoothed over:

1. **The blend does not beat its best component on any metric.** This is a
   correction to an earlier claim. When all five engines are measured in one
   run on one split, the hybrid scores 0.2641 macro F1 against HistGB's 0.2648
   and XGBoost's 0.2705, and loses on top-1, top-3 and MAP@7 as well. Mixing in
   an engine that reaches 31.1% top-3 alone pulls the average down. The
   multi-engine architecture is justified by **coverage and explainability** —
   CF surfaces products the profile model scores at exactly zero, and the
   constraints layer is what makes the output defensible — and *not* by the
   blend ranking better.
2. **Macro F1 is not comparable across different product counts.** Read it
   against chance. 0.2705 against a 0.0455 chance level is 5.95×. An earlier
   18-product run scored a higher raw 0.3098 but only 5.58× its own chance —
   the bigger number was the weaker result.
3. **The macro tilt costs about −0.0002 macro F1.** It is there for business
   relevance and timing. It reorders; it does not predict better.
4. **The eligibility rules block 13.0% of acquisitions that actually
   happened.** That is the price of a hard gate, and it is reported in the
   console rather than hidden.
5. **Five times more data moved top-1 by 0.6 points.** The ceiling here is
   feature-level, not data-level.

### Measured live, end to end, on real customers

The console can draw an actual customer out of the **104,168 held-out events**
(`GET /api/sample`). Because the event records what that customer took next,
the recommendation is marked right or wrong against ground truth rather than
just looking plausible. Over 400 random draws, through all four engines:

| | |
|---|---|
| top-1 | **58.8%** |
| top-3 | **81.2%** |
| the product they actually took was **removed by a rule** | **13.5%** |
| every product blocked (fallback path) | 0.0% |

Note the gap: the model alone reaches **88.9%** top-3, the whole pipeline
**81.2%**. The missing 7.7 points are the eligibility layer deleting the right
answer. That 13.5% independently reproduces the 13.0% block rate measured back
in phase 6, by a completely different route.

**Quote 81.2%, not 88.9%, as the system's accuracy.** 88.9% is what the model
scores with no constraints applied; it is not what a customer standing at the
counter would experience.

### Real records are not tidy

Roughly **28% of customers have no income on file** (the dataset stores `-1`),
seniority uses `-999999`, and ages run from 2 to 110. A hard eligibility gate
must not read "unknown" as "failed the threshold", nor as "passed". So each rule
declares the fields it depends on, a missing field produces its own reason
("income at least 35,000 — income not on file"), and the response carries a
`missing_fields` list that the console surfaces.

### Why HistGB is served when XGBoost scores higher

XGBoost wins by 0.0057 macro F1 and 0.002 top-1 — real but small — and it is a
third-party dependency that may not be installed. HistGradientBoosting ships
with scikit-learn, so the app runs anywhere. The console states the gap on the
Performance view rather than hiding it. To serve XGBoost instead, save `xgb` in
place of `clf` in `build_model.py`.

---

## The console

Five views, all served from the same process at `/`.

- **Advise** — intake form, **Draw a real customer from the held-out data**
  (which then marks the answer correct / missed / removed-by-rule against what
  that customer actually did), three example customers (Rahul, Aman, Vikram), an
  animated **branch walk-through** where the customer passes four gates and each
  one reports what that engine did to their profile, the top-3 cards with
  per-engine bars, what was removed and why, and a cascade chart of every
  product at all four stages.
- **Inside the engines** — the architecture as an interactive flow; select a
  stage to see what it did to the loaded profile.
- **Performance** — the held-out metrics, every engine compared on the same
  split, and the per-product base rates with their rules.
- **Evidence** — the strongest measured sequence rules and the full item–item
  lift matrix as a heatmap.
- **Session** — every profile advised since the server started; select one to
  reopen it.

---

## API

| method | path | returns |
|---|---|---|
| GET | `/api/status` | model, product count, held-out metrics |
| GET | `/api/meta` | products, rules, regimes, base rates |
| POST | `/api/recommend` | a profile in, a ranked top-3 and the full board out |
| GET | `/api/sample` | a real held-out customer, with the product they took next |
| GET | `/api/macro` | the regime history classified from the price series |
| GET | `/api/lift/rules` | the strongest measured sequence rules |
| GET | `/api/lift/matrix` | the item–item lift matrix |
| GET | `/api/performance` | metrics and per-product base rates |
| GET | `/api/session` | recommendations made since start |

Example:

```powershell
curl -X POST http://127.0.0.1:8000/api/recommend `
  -H "Content-Type: application/json" `
  -d '{\"age\":29,\"income\":48000,\"seniority_months\":26,\"holdings\":[\"current_account\",\"payroll_account\"],\"regime\":\"RECOVERY\"}'
```

---

## Layout

```
web/
├── README.md
├── run.ps1
├── backend/
│   ├── build_model.py        trains and measures every engine
│   ├── macro.py              regime history from the real price series
│   ├── app.py                FastAPI: the four engines as a service
│   └── model/
│       └── smartcross.pkl    created by build_model.py
└── frontend/
    ├── index.html
    ├── app.css
    └── app.js                every chart drawn by hand, no libraries
```

The eligibility thresholds in `app.py` (`RULES`) are illustrative retail-credit
norms. In a deployment they come from the bank's own credit policy, which is
exactly why they live in one readable dictionary a compliance officer can audit
without reading the model.
