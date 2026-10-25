"""
================================================================================
 SmartCross API — the four-engine recommender as a service
================================================================================

     python build_model.py      once, builds model/smartcross.pkl
     python app.py              starts the API and serves the console

 Then open http://127.0.0.1:8000

 ENDPOINTS
     GET  /api/status            model, products, held-out metrics
     GET  /api/meta              products, rules, regimes, base rates
     POST /api/recommend         a customer profile in, a ranked top-3 out
     GET  /api/lift/rules        the strongest measured sequence rules
     GET  /api/lift/matrix       the item-item lift matrix, for the heatmap
     GET  /api/performance       held-out metrics and per-product base rates
     GET  /api/session           recommendations made since the server started
     GET  /api/session/{id}

 THE PIPELINE, IN ORDER
     1  profile matrix     35 features assembled from holdings + demographics
     2  recommendation     fused hybrid: the content classifier reads the
                           collaborative engine's scores as inputs (peers.py)
     3  macro regime       per-product tilt multipliers
     4  constraints        ineligible products removed, not down-weighted

 Order matters and is not an implementation detail. Filtering before scoring
 would bias the probabilities; tilting after filtering would boost products the
 customer cannot have. The hard constraint runs last so nothing downstream can
 re-admit an ineligible product. Every response carries the per-stage scores so
 the console can show that cascade rather than assert it.
================================================================================
"""

import os
import sys
import uuid
import datetime
from typing import Dict, List, Optional

import numpy as np

try:
    import joblib
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field
    import uvicorn
except ImportError as exc:
    print(f"\n[X] Missing package: {exc.name}")
    print("    pip install fastapi uvicorn pydantic scikit-learn joblib numpy\n")
    sys.exit(1)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import peers                                   # the collaborative engine, shared with build_model.py
MODEL = os.path.join(HERE, "model", "smartcross.pkl")
FRONTEND = os.path.normpath(os.path.join(HERE, "..", "frontend"))

if not os.path.exists(MODEL):
    print(f"\n[X] {MODEL} not found.  Run:  python build_model.py\n")
    sys.exit(1)

# A pickle is only readable by libraries close to the ones that wrote it.
# Opening a numpy 2.x artefact under numpy 1.x fails deep inside the unpickler
# with a message about BitGenerators that says nothing useful, so translate it.
try:
    B = joblib.load(MODEL)
except Exception as exc:
    import numpy as _np
    import sklearn as _sk
    print("\n" + "=" * 70)
    print("[X] The model file cannot be read by this Python environment.")
    print("=" * 70)
    print(f"\n    {type(exc).__name__}: {exc}")
    print(f"\n    This interpreter : {sys.executable}")
    print(f"    numpy            : {_np.__version__}")
    print(f"    scikit-learn     : {_sk.__version__}")
    print("\n    smartcross.pkl was written by a different version of numpy or")
    print("    scikit-learn. Pickles are not portable across major versions.")
    print("\n    FIX - rebuild it with the libraries you are running now:")
    print("\n        python build_model.py")
    print("\n    That takes a few minutes and writes a file this environment can read.\n")
    sys.exit(1)
if "peers" not in B or "hist_names" not in B:
    print("\n[X] smartcross.pkl predates the current model (no collaborative tables or history).")
    print("    Rebuild it:  python build_model.py\n")
    sys.exit(1)
_built = B.get("versions", {})
_here = {"numpy": np.__version__, "scikit_learn": __import__("sklearn").__version__}
if any(_built.get(k) and _built[k] != v for k, v in _here.items()):
    print(f"[!] model built with numpy {_built.get('numpy')} / scikit-learn {_built.get('scikit_learn')}, "
          f"running {_here['numpy']} / {_here['scikit_learn']}. It loaded, but rebuild with "
          f"python build_model.py for identical results.")
CLF = B["clf"]
PEERS = B["peers"]
HIST_NAMES: List[str] = B["hist_names"]
STREAK_CAP = 15                                # as in phase2_4a_build.py
PRODUCTS: List[str] = B["products"]
ALL: List[str] = B["all_names"]
BASE: np.ndarray = B["base_rate"]
LIFT: np.ndarray = B["lift"]
LIFT_NAMES: List[str] = B["lift_names"]
METRICS: Dict = B["metrics"]
K = len(PRODUCTS)

try:
    from macro import MacroHistory, REGIME_NOTE as MACRO_NOTE
    MACRO = MacroHistory()
except Exception as _exc:                      # never let the macro files stop the API
    print(f"[!] macro series unavailable ({_exc}); regime falls back to manual choice")
    MACRO, MACRO_NOTE = None, {}

# ---------------------------------------------------------------- real customers
# The three named personas are illustrations. This loads the actual held-out
# events so the console can draw a REAL customer at random -- and, because the
# event records what they took next, score the recommendation against ground
# truth instead of against a story.
EVENTS = os.path.join(HERE, "..", "..", "partB_santander", "data", "sant_events.npz")
SAMPLES = None
try:
    _d = np.load(os.path.normpath(EVENTS), allow_pickle=True)
    _names = [str(s) for s in _d["names"]]
    _mi = _d["mi"].astype(int)
    _cut = _mi.max() - 3 + 1                   # the same three test months
    _te = np.where(_mi >= _cut)[0]
    SAMPLES = {
        "prev": _d["X_prev"][_te].astype(np.int8),
        "demo": _d["X_demo"][_te].astype(np.float32),
        "hist": _d["X_hist"][_te].astype(np.int16),
        "y": _d["y"][_te].astype(int),
        "mi": _mi[_te],
        "names": _names,
        "n": len(_te),
        "months": (int(_cut), int(_mi.max())),
    }
    print(f"[i] {SAMPLES['n']:,} held-out customers available for sampling")
except Exception as _exc:
    print(f"[!] events file unavailable ({_exc}); random real customers disabled")
    SAMPLES = None

LIFT_IDX = {n: i for i, n in enumerate(LIFT_NAMES)}
ALL_IDX = {n: i for i, n in enumerate(ALL)}
SEGMENTS = {"VIP": 0, "Individual": 1, "College": 2}

SESSION: List[dict] = []

# ----------------------------------------------------------- macro regimes
# Business rules, not learned parameters. A regime reorders the list; it never
# claims to predict better. Measured effect on macro F1 is about -0.0002.
REGIMES: Dict[str, Dict[str, float]] = {
    "EXPANSION": {"funds": 1.25, "securities": 1.25, "mortgage": 1.20,
                  "long_term_deposits": 1.15, "pension_plan": 1.15,
                  "credit_card": 1.10, "loans": 1.10,
                  "short_term_deposits": 0.85, "current_account": 0.90},
    "RECOVERY": {"mortgage": 1.20, "credit_card": 1.15, "funds": 1.10,
                 "payroll_account": 1.10, "securities": 1.05, "loans": 1.05,
                 "short_term_deposits": 0.95},
    "SLOWDOWN": {"direct_debit": 1.15, "current_account": 1.10,
                 "short_term_deposits": 1.10, "taxes": 1.05, "funds": 0.90,
                 "securities": 0.85, "mortgage": 0.85, "loans": 0.90},
    "STRESS": {"current_account": 1.30, "short_term_deposits": 1.25,
               "direct_debit": 1.20, "long_term_deposits": 1.10, "funds": 0.70,
               "securities": 0.65, "mortgage": 0.60, "loans": 0.70,
               "pension_plan": 0.80},
}
REGIME_NOTE = {
    "EXPANSION": "Rising market, low volatility. Households take on discretionary and investment products.",
    "RECOVERY": "Rising market, still volatile. Growth returns, caution remains.",
    "SLOWDOWN": "Falling market, low volatility. Consolidation beats consumption.",
    "STRESS": "Falling market, high volatility. Liquidity and safety dominate.",
}

# ------------------------------------------------- knowledge-based constraints
# One readable predicate per product, so a compliance officer can audit them
# without reading the model. Thresholds are illustrative retail-credit norms;
# in deployment they come from the bank's own credit policy.
#
# Real records are not tidy. In the Santander data income is -1 for roughly a
# third of customers and seniority uses -999999, both meaning "not on file".
# A credit gate must not treat unknown as failed-the-threshold, and must not
# treat it as passed either: it has to say the data is missing. So each rule
# declares the fields it needs, and a missing field produces its own reason.
#
MISSING = {"income": lambda v: v is None or v < 0,
           "seniority_months": lambda v: v is None or v < 0,
           "age": lambda v: v is None or v <= 0 or v > 120}

FIELD_LABEL = {"income": "income", "seniority_months": "time with the bank", "age": "age"}


def known(c: dict, field: str) -> bool:
    test = MISSING.get(field)
    return not test(c.get(field)) if test else True


# product -> (plain-language rule, predicate, fields the predicate relies on)
RULES = {
    "junior_account": ("age under 25", lambda c: c["age"] < 25, ("age",)),
    "pension_plan": ("age 30+ and income at least 25,000",
                     lambda c: c["age"] >= 30 and c["income"] >= 25000, ("age", "income")),
    "mortgage": ("age 25-65, income at least 30,000, 24 months seniority",
                 lambda c: 25 <= c["age"] <= 65 and c["income"] >= 30000 and c["seniority_months"] >= 24,
                 ("age", "income", "seniority_months")),
    "loans": ("income at least 15,000", lambda c: c["income"] >= 15000, ("income",)),
    "securities": ("income at least 40,000 — suitability",
                   lambda c: c["income"] >= 40000, ("income",)),
    "funds": ("income at least 35,000 — suitability",
              lambda c: c["income"] >= 35000, ("income",)),
    "long_term_deposits": ("income at least 20,000", lambda c: c["income"] >= 20000, ("income",)),
    "medium_term_deposits": ("income at least 15,000", lambda c: c["income"] >= 15000, ("income",)),
    "payroll": ("requires a payroll account",
                lambda c: "payroll_account" in c["holdings"], ()),
    "pensions": ("requires a payroll account",
                 lambda c: "payroll_account" in c["holdings"], ()),
    "particular_plus_account": ("income at least 30,000",
                                lambda c: c["income"] >= 30000, ("income",)),
    "home_account": ("age 25+ and income at least 25,000",
                     lambda c: c["age"] >= 25 and c["income"] >= 25000, ("age", "income")),
    "taxes": ("12 months seniority", lambda c: c["seniority_months"] >= 12, ("seniority_months",)),
}


def check_rule(product: str, c: dict):
    """(passes, reason, missing_fields). Unknown data never silently passes."""
    r = RULES.get(product)
    if not r:
        return True, "no eligibility gate", []
    text, pred, needs = r
    absent = [f for f in needs if not known(c, f)]
    if absent:
        labels = " and ".join(FIELD_LABEL.get(f, f) for f in absent)
        return False, f"{text} — {labels} not on file", absent
    try:
        return bool(pred(c)), text, []
    except Exception:
        return False, f"{text} — could not be evaluated", list(needs)


class Customer(BaseModel):
    name: str = Field("", max_length=80)
    # Bounds are set by what the source data actually contains, not by what a
    # tidy form would allow. Income -1 and seniority -999999 are the dataset's
    # "not on file" sentinels and must be accepted, then handled as missing.
    age: int = Field(34, ge=0, le=120)
    income: float = Field(52000, ge=-1)
    seniority_months: int = Field(48, ge=-999999)
    segment: str = Field("Individual")
    is_active: int = Field(1, ge=0, le=1)
    is_new_customer: int = Field(0, ge=0, le=1)
    is_male: int = Field(0, ge=0, le=1)
    holdings: List[str] = Field(default_factory=lambda: ["current_account", "payroll_account", "direct_debit"])
    months_since_last_acq: int = Field(7, ge=-1)
    acquisitions_so_far: int = Field(3, ge=0)
    # how many months of history the bank has observed. Training had this as a
    # column of its own; -1 means "use seniority", which is what a form-filled
    # profile can offer. A sampled real customer supplies the true value.
    months_observed: int = Field(-1, ge=-1)
    # Holdings history, 48 numbers in the order of phase2_4a_build.py: what was
    # held one observation earlier (24 flags), then months each product has been
    # held in a row (24). A sampled real customer supplies the true values; a
    # hand-entered profile leaves it empty and an estimate is used, and said so.
    history: Optional[List[float]] = Field(None, min_length=48, max_length=48)
    # "AUTO" classifies the regime from the real price series instead of
    # taking the adviser's word for it. A named regime is still accepted,
    # because scenario work ("what would we say in a crash?") is legitimate.
    regime: str = Field("AUTO")
    as_of: str = Field("")          # 'YYYY-MM'; empty means the latest month
    # 1.0 serves the fused model as trained. The peers are already inside it;
    # lowering alpha mixes the raw collaborative scores in again, for what-if work.
    alpha: float = Field(1.0, ge=0.0, le=1.0)
    top_k: int = Field(3, ge=1, le=10)
    # False for live re-scores while the adviser drags a control, so the
    # Session view keeps one record per advice rather than one per pixel.
    record: bool = Field(True)


app = FastAPI(title="SmartCross", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def pretty(p: str) -> str:
    return p.replace("_", " ")


def history_for(c: dict, prev: np.ndarray):
    """(48 history features, source). Real when supplied, otherwise estimated.

    The estimate assumes a stable customer: what they hold now they also held
    the month before, for as long as they have been with the bank (capped as in
    training). It is the least-surprising guess, and the response labels it.
    """
    h = c.get("history")
    if h is not None:
        return np.asarray(h, dtype=np.float32), "record"
    tenure = c.get("months_observed", -1)
    if tenure is None or tenure < 0:
        tenure = c["seniority_months"]
    months = float(min(max(tenure, 1), STREAK_CAP))
    return np.hstack([prev, prev * months]).astype(np.float32), "estimated"


def build_features(c: dict) -> np.ndarray:
    """Layer 1 — the profile matrix: 24 holdings flags + 11 behavioural/demographic,
    then the 44 collaborative scores and the 48 holdings-history features the
    fused model reads alongside them."""
    held = set(c["holdings"])
    prev = np.zeros(len(ALL), dtype=np.float32)
    for h in held:
        if h in ALL_IDX:
            prev[ALL_IDX[h]] = 1.0
    obs = c.get("months_observed", -1)
    if obs is None or obs < 0:
        obs = c["seniority_months"]
    demo = np.array([
        len(held), obs, c["months_since_last_acq"],
        c["acquisitions_so_far"], c["age"], c["income"], c["seniority_months"],
        c["is_active"], c["is_new_customer"],
        SEGMENTS.get(c["segment"], -1), c["is_male"],
    ], dtype=np.float32)
    hist, _ = history_for(c, prev)
    return np.hstack([prev, demo, peers.features(PEERS, prev.reshape(1, -1))[0], hist]).reshape(1, -1)


def holdings_vector(held: set) -> np.ndarray:
    v = np.zeros((1, len(ALL)), dtype=np.float32)
    for h in held:
        if h in ALL_IDX:
            v[0, ALL_IDX[h]] = 1.0
    return v


def cf_scores(held: set) -> np.ndarray:
    """Layer 2b — collaborative: what customers with this portfolio added next.

    Probability, not lift (see peers.py). Lift is kept for the reasons only.
    """
    return peers.pattern(PEERS, holdings_vector(held))[0]


# A rule is only quoted when enough customers stand behind it. Lift on a rare
# product is noisy: 45x from four customers is an anecdote, not evidence.
MIN_SUPPORT = 30
SUPPORT = PEERS["co"]          # [held i x added j] training-month event counts, rows in LIFT_NAMES order


def best_rule_for(j: int, held: set):
    """The strongest well-supported lift from something the customer already holds."""
    best, best_lift = None, 1.0
    for h in held:
        i = LIFT_IDX.get(h)
        if i is None or SUPPORT[i, j] < MIN_SUPPORT:
            continue
        v = float(LIFT[i, j])
        if v > best_lift:
            best, best_lift = h, v
    return (best, best_lift) if best else None


@app.get("/api/status")
def status():
    return {
        "ok": True,
        "products": K,
        "built_at": B["built_at"],
        "metrics": METRICS,
        "model": type(CLF).__name__,
        "n_features": len(B["feature_names"]),
    }


@app.get("/api/meta")
def meta():
    return {
        "products": PRODUCTS,
        "all_products": ALL,
        "regimes": [{"id": r, "note": REGIME_NOTE[r],
                     "tilts": {p: REGIMES[r].get(p, 1.0) for p in PRODUCTS}}
                    for r in REGIMES],
        "segments": list(SEGMENTS.keys()),
        "rules": {p: v[0] for p, v in RULES.items()},
        "base_rate": {p: float(BASE[j]) for j, p in enumerate(PRODUCTS)},
        "metrics": METRICS,
        "built_at": B["built_at"],
        "macro": (MACRO.summary() if (MACRO and MACRO.available)
                  else {"available": False}),
    }


SEG_BY_CODE = {v: k for k, v in SEGMENTS.items()}


@app.get("/api/sample")
def sample(seed: int = -1, min_holdings: int = 1):
    """One real customer from the held-out months, with what they took next.

    The profile is reconstructed exactly as it was when the event happened, so
    the recommendation can be checked against the product they actually chose.
    Nothing about this row was seen in training.
    """
    if SAMPLES is None:
        raise HTTPException(503, "events file not available; random real customers disabled")

    rng = np.random.default_rng(None if seed < 0 else seed)
    for _ in range(400):                       # a few draws to respect min_holdings
        i = int(rng.integers(0, SAMPLES["n"]))
        prev = SAMPLES["prev"][i]
        if int(prev.sum()) >= max(0, min_holdings):
            break

    d = SAMPLES["demo"][i]
    holdings = [SAMPLES["names"][j] for j in np.nonzero(prev)[0]]
    actual = SAMPLES["names"][SAMPLES["y"][i]]

    return {
        "index": i,
        "month_index": int(SAMPLES["mi"][i]),
        "customer": {
            "name": f"Customer #{i:,}",
            "age": int(d[4]),
            "income": float(d[5]),
            "seniority_months": int(d[6]),
            "months_observed": int(d[1]),
            "months_since_last_acq": int(d[2]),
            "acquisitions_so_far": int(d[3]),
            "is_active": int(d[7]),
            "is_new_customer": int(d[8]),
            "is_male": int(d[10]),
            "segment": SEG_BY_CODE.get(int(d[9]), "Individual"),
            "holdings": holdings,
            "history": [int(v) for v in SAMPLES["hist"][i]],   # the real record, as at the event
        },
        # ground truth: the product this customer actually took next
        "actual_next": actual,
        "actual_label": pretty(actual),
        "actual_in_catalogue": actual in PRODUCTS,
        "pool": {"n": SAMPLES["n"], "months": SAMPLES["months"],
                 "note": "held-out months only; never seen in training"},
    }


@app.get("/api/macro")
def macro(limit: int = 0):
    """The regime history classified from the real price series."""
    if MACRO is None or not MACRO.available:
        return {"available": False,
                "reason": "no usable price series under data/raw/"}
    s = MACRO.summary()
    tl = MACRO.timeline[-limit:] if limit else MACRO.timeline
    s["timeline"] = [{"month": r["month"], "regime": r["regime"],
                      "momentum": round(r["momentum"], 4),
                      "volatility": round(r["volatility"], 4),
                      "agreement": round(r["agreement"], 3)} for r in tl]
    s["notes"] = REGIME_NOTE
    return s


@app.get("/api/lift/rules")
def lift_rules(top: int = 40, min_lift: float = 1.2, min_support: int = MIN_SUPPORT):
    """The strongest measured sequence rules: held product -> next product.

    Only rules backed by at least min_support training events are listed;
    how many were dropped for thin support is reported, not hidden.
    """
    out, thin = [], 0
    for i, src in enumerate(LIFT_NAMES):
        for j, dst in enumerate(PRODUCTS):
            if src == dst:
                continue
            v = float(LIFT[i, j])
            if v >= min_lift:
                n = int(SUPPORT[i, j])
                if n < min_support:
                    thin += 1
                    continue
                out.append({"from": src, "from_label": pretty(src),
                            "to": dst, "to_label": pretty(dst),
                            "lift": v, "base_rate": float(BASE[j]),
                            "confidence": float(v * BASE[j]), "support": n})
    out.sort(key=lambda r: -r["lift"])
    return {"rules": out[:top], "total_above_threshold": len(out),
            "min_lift": min_lift, "min_support": min_support,
            "dropped_thin_support": thin}


@app.get("/api/lift/matrix")
def lift_matrix():
    """The full item-item matrix, for the heatmap."""
    idx = [LIFT_IDX[p] for p in PRODUCTS if p in LIFT_IDX]
    names = [p for p in PRODUCTS if p in LIFT_IDX]
    m = LIFT[np.ix_(idx, range(K))]
    return {"rows": names, "cols": PRODUCTS,
            "matrix": [[float(v) for v in r] for r in m]}


EVIDENCE = os.path.normpath(os.path.join(HERE, "..", "..", "defense", "outputs", "defense_results.json"))


@app.get("/api/evidence")
def evidence():
    """Confidence intervals, significance tests and ablations from defense_eval.py.

    Served as written by that script; if it has not been run the console says so
    rather than showing numbers from anywhere else.
    """
    if not os.path.exists(EVIDENCE):
        raise HTTPException(404, "run defense/defense_eval.py to produce the evidence file")
    import json
    with open(EVIDENCE, encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/performance")
def performance():
    order = np.argsort(-BASE)
    return {
        "metrics": METRICS,
        # measured by build_model.py on this same split, in this same run.
        # Nothing here is quoted from an earlier experiment.
        "engines": METRICS.get("engines", []),
        "served_engine": METRICS.get("served_engine", "Content (HistGB)"),
        "xgboost_available": METRICS.get("xgboost_available", False),
        "products": [{"product": PRODUCTS[j], "label": pretty(PRODUCTS[j]),
                      "base_rate": float(BASE[j]),
                      "rule": RULES.get(PRODUCTS[j], ("no gate", None))[0]}
                     for j in order],
        "built_at": B["built_at"],
    }


def resolve_regime(cd: dict):
    """Either classify the regime from the price series, or take the one asked for.

    Returns (regime, source) where source explains the choice to the console.
    """
    want = (cd.get("regime") or "AUTO").upper()
    if want == "AUTO":
        if MACRO is None or not MACRO.available:
            return "RECOVERY", {"mode": "fallback",
                                "detail": "no price series found; defaulted to RECOVERY"}
        row = MACRO.at(cd.get("as_of") or None)
        disagree = sorted({r for r in row["votes"].values() if r != row["regime"]})
        return row["regime"], {
            "mode": "classified",
            "month": row["month"],
            "momentum": round(row["momentum"], 4),
            "volatility": round(row["volatility"], 4),
            "agreement": round(row["agreement"], 3),
            "n_series": row["n_series"],
            "tie_broken": row["tie_broken"],
            "votes": row["votes"],
            "dissenting": disagree,
            "detail": f"classified from {row['n_series']} price series for "
                      f"{row['month']} ({int(row['agreement']*100)}% agreement)",
        }
    if want not in REGIMES:
        raise HTTPException(400, f"unknown regime {want}")
    return want, {"mode": "manual",
                  "detail": "chosen by the adviser, not classified from data"}


@app.post("/api/recommend")
def recommend(c: Customer):
    cd = c.model_dump()
    cd["regime"], regime_source = resolve_regime(cd)
    held = set(cd["holdings"])

    # --- 1. profile matrix
    X = build_features(cd)

    # --- 2. recommendation engine
    content = CLF.predict_proba(X)[0]
    cf = cf_scores(held)
    a = cd["alpha"]
    blend = a * content + (1 - a) * cf
    for p in held:                               # never recommend what they hold
        if p in PRODUCTS:
            blend[PRODUCTS.index(p)] = 0.0
    s2 = blend / blend.sum() if blend.sum() > 0 else blend

    # --- 3. macro regime tilt
    tilts = REGIMES[cd["regime"]]
    tvec = np.array([tilts.get(p, 1.0) for p in PRODUCTS])
    t = s2 * tvec
    s3 = t / t.sum() if t.sum() > 0 else t

    # --- 4. knowledge-based constraints: remove, never down-weight
    blocked, final, missing_fields = [], s3.copy(), set()
    for j, p in enumerate(PRODUCTS):
        ok, reason, absent = check_rule(p, cd)
        if not ok:
            if s3[j] > 0:
                blocked.append({"product": p, "label": pretty(p), "rule": reason,
                                "missing": absent, "score_before": float(s3[j])})
            missing_fields.update(absent)
            final[j] = 0.0
    fallback = bool(final.sum() <= 0)   # plain bool: numpy.bool_ is not JSON-serialisable
    if fallback:
        final = s3.copy()
    s4 = final / final.sum() if final.sum() > 0 else final

    # --- output
    order = np.argsort(-s4)[: cd["top_k"]]
    recs = []
    for rank, j in enumerate(order, 1):
        p = PRODUCTS[j]
        r = best_rule_for(j, held)
        if r:
            reason = (f"Customers holding a {pretty(r[0])} add this "
                      f"{r[1]:.1f} times more often than the base rate.")
        else:
            reason = (f"No sequence rule fires. Ranked on profile fit and a "
                      f"{BASE[j]*100:.2f}% base adoption rate.")
        tilt = float(tvec[j])
        if tilt > 1.001:
            reason += f" Boosted by the {cd['regime']} regime."
        elif tilt < 0.999:
            reason += f" Tempered by the {cd['regime']} regime."
        recs.append({
            "rank": rank, "product": p, "label": pretty(p),
            "score": float(s4[j]),
            "content": float(content[j]), "cf": float(cf[j]),
            "hybrid": float(s2[j]), "after_macro": float(s3[j]),
            "macro_tilt": tilt, "base_rate": float(BASE[j]),
            "eligible": True, "reason": reason,
            "rule": check_rule(p, cd)[1],
            "top_rule": {"from": r[0], "from_label": pretty(r[0]),
                         "lift": float(r[1])} if r else None,
        })

    # the whole board, for the cascade chart
    board = [{
        "product": PRODUCTS[j], "label": pretty(PRODUCTS[j]),
        "content": float(content[j]), "cf": float(cf[j]),
        "hybrid": float(s2[j]), "after_macro": float(s3[j]),
        "final": float(s4[j]), "tilt": float(tvec[j]),
        "held": bool(PRODUCTS[j] in held),
        "blocked": bool(any(b["product"] == PRODUCTS[j] for b in blocked)),
    } for j in np.argsort(-s3)]

    rec = {
        "id": uuid.uuid4().hex[:8],
        "created": datetime.datetime.now().strftime("%H:%M:%S"),
        "customer": {"name": cd["name"], "age": cd["age"], "income": cd["income"],
                     "seniority_months": cd["seniority_months"],
                     "segment": cd["segment"], "holdings": sorted(held),
                     "n_holdings": len(held)},
        "regime": cd["regime"], "regime_note": REGIME_NOTE[cd["regime"]],
        "regime_source": regime_source,
        # whether the 48 history features came from the customer's real record
        # or were estimated from today's holdings (hand-entered profile)
        "history_source": "record" if cd.get("history") is not None else "estimated",
        "alpha": a,
        "recommendations": recs,
        "blocked": sorted(blocked, key=lambda b: -b["score_before"]),
        "all_blocked": fallback,
        "board": board,
        "n_scored": int((s2 > 0).sum()),
        "n_blocked": len(blocked),
        "n_products": K,
        # what the bank does not know about this customer, and so could not
        # evidence. Reported rather than quietly treated as a failed threshold.
        "missing_fields": sorted(FIELD_LABEL.get(f, f) for f in missing_fields),
    }
    if cd["record"]:
        SESSION.insert(0, rec)
        del SESSION[60:]
    return rec


@app.get("/api/session")
def session():
    return [{"id": r["id"], "created": r["created"],
             "name": r["customer"]["name"], "age": r["customer"]["age"],
             "income": r["customer"]["income"], "regime": r["regime"],
             "top": r["recommendations"][0]["label"] if r["recommendations"] else "-",
             "score": r["recommendations"][0]["score"] if r["recommendations"] else 0,
             "n_blocked": r["n_blocked"]} for r in SESSION]


@app.get("/api/session/{rid}")
def session_one(rid: str):
    for r in SESSION:
        if r["id"] == rid:
            return r
    raise HTTPException(404, "no such record in this session")


if os.path.isdir(FRONTEND):
    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(FRONTEND, "index.html"))


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("SMARTCROSS API")
    print("=" * 70)
    print(f"  Model built : {B['built_at']}")
    print(f"  Products    : {K}")
    print(f"  Held out    : macro F1 {METRICS['macro_f1']:.4f} | "
          f"top-3 {METRICS['top3']:.3f} | MAP@7 {METRICS['map7']:.3f}")
    print("\n  Open http://127.0.0.1:8000\n")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
