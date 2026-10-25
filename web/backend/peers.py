"""
================================================================================
 SmartCross — the collaborative engine ("what did customers like you add?")
================================================================================

 Shared by build_model.py (training) and app.py (serving), so the two cannot
 drift apart.

 WHY NOT LIFT
 The first version ranked products by summed LIFT. Lift divides by a product's
 base rate, so it measures surprise, not likelihood: a 40x lift on a product
 almost nobody takes outranked a 2x lift on one a third of customers take.
 Measured on the held-out months it reached 31% top-3 -- below the no-ML
 popularity list (70%). Lift is still the right number for an EXPLANATION
 ("payroll holders add pensions 9x more often"), so the console keeps it for
 reasons and the Evidence view. It is no longer used to rank.

 WHAT REPLACES IT -- two probability estimates
   cooc     mean over held products i of P(add j | holds i), smoothed toward
            the base rate with strength M_COOC
   pattern  P(add j | the customer's EXACT portfolio), shrunk toward cooc with
            strength M_PATTERN -- "customers with exactly your products added j".
            A portfolio seen once cannot overrule what thousands of similar
            customers did, so rare patterns lean on cooc.

 Both strengths were chosen on a validation month the test period never sees
 (defense/cf_experiment.py). On the test months the pattern engine alone
 reaches 85.8% top-3, against 31.2% for lift and 70.5% for popularity.

 HOW IT FUSES WITH THE CONTENT MODEL
 Feature augmentation (Burke's hybrid taxonomy): both estimates, 22 + 22
 numbers, are appended to the 35 profile features, and the gradient-boosting
 model learns per customer how far to trust the peers. For training rows the
 tables are built from strictly EARLIER months (expanding_features), so no
 row's own outcome leaks into its inputs.
================================================================================
"""

import numpy as np

M_COOC = 50        # smoothing toward the base rate (chosen on validation)
M_PATTERN = 0      # pattern shrinkage; 0 = use the pattern whenever it was seen
M_BACKOFF = 5      # cooc strength used as the pattern's back-off


def pack(H):
    """Holdings flags -> one integer per customer, so a portfolio is a key."""
    H = np.asarray(H)
    return (H.astype(np.int64) * (1 << np.arange(H.shape[1], dtype=np.int64))).sum(axis=1)


def fit(H, y, K):
    """Every table the collaborative engine needs, from the rows given only.

    H  [n x 24] holdings flags before the event
    y  [n]      index (0..K-1) of the product added
    """
    H = np.asarray(H, dtype=np.float64)
    base = np.bincount(y, minlength=K).astype(float)
    base /= max(base.sum(), 1.0)
    co = H.T @ np.eye(K)[y]                           # [24 held x K added]
    keys, inv = np.unique(pack(H), return_inverse=True)
    pat = np.zeros((len(keys), K))
    np.add.at(pat, (inv, y), 1.0)
    return {"base": base, "co": co, "n_held": H.sum(axis=0)[:, None],
            "pat_keys": keys, "pat_counts": pat}


def cooc(T, H, m=M_COOC):
    H = np.asarray(H, dtype=np.float64)
    cond = (T["co"] + m * T["base"][None, :]) / (T["n_held"] + m)
    cond = np.where(T["n_held"] > 0, cond, T["base"][None, :])
    n = H.sum(axis=1, keepdims=True)
    s = (H @ cond) / np.clip(n, 1, None)
    return np.where(n > 0, s, T["base"][None, :])


def pattern(T, H, m=M_PATTERN, m_back=M_BACKOFF):
    back = cooc(T, H, m_back)
    keys = pack(H)
    pos = np.clip(np.searchsorted(T["pat_keys"], keys), 0, len(T["pat_keys"]) - 1)
    hit = T["pat_keys"][pos] == keys
    counts = np.where(hit[:, None], T["pat_counts"][pos], 0.0)
    n = counts.sum(axis=1, keepdims=True)
    if m > 0:
        return (counts + m * back) / (n + m)
    return np.where(n > 0, counts / np.clip(n, 1, None), back)


def features(T, H):
    """The 44 inputs the fused model reads: 22 cooc + 22 pattern estimates."""
    return np.hstack([cooc(T, H), pattern(T, H)]).astype(np.float32)


def expanding_features(H, y, mi, K, months):
    """Peer features for training rows, each month built from EARLIER months only."""
    F = np.zeros((len(y), 2 * K), dtype=np.float32)
    for m in months:
        rows, past = mi == m, mi < m
        if past.any():
            F[rows] = features(fit(H[past], y[past], K), H[rows])
        else:
            F[rows] = 1.0 / K                         # first month: no peers yet
    return F
