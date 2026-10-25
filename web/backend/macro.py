"""
================================================================================
 SmartCross — the macro regime engine, from the real series
================================================================================

 Reads the price files in data/raw/macro_economic and data/raw/sector_assets,
 turns them into a month-by-month regime history, and exposes the regime that
 was actually in force in any given month.

 Before this, the regime was a dropdown the user picked. That made the third
 engine a what-if toggle rather than a measurement. Now the dropdown is still
 there for scenario work, but the default is classified from the data and the
 console can show which series voted which way.

 THE CLASSIFICATION
     momentum    6-month change in the monthly close
     volatility  standard deviation of the last 6 monthly returns, annualised

     rising + calm      EXPANSION   discretionary and investment products
     rising + volatile  RECOVERY    growth returns, caution remains
     falling + calm     SLOWDOWN    consolidation beats consumption
     falling + volatile STRESS      liquidity and safety dominate

 The volatility threshold is the MEDIAN volatility across the whole history,
 not a number picked to make the output look good. Half the months are
 therefore "volatile" by construction, which is the honest way to split a
 series when no external definition of "high volatility" is available.

 Every series gets one vote per month. The regime is the majority, ties broken
 by the sign of mean momentum. Carrying the per-series votes through to the API
 means the console can show disagreement rather than hiding it behind a label.

 The files are yfinance exports with a three-line header:
     Price,Close[,Volume]
     Ticker,BZ=F[,BZ=F]
     Date,[,]
     2015-01-02,56.41999816894531[,...]
================================================================================
"""

import os
import csv
import math
import statistics
from collections import Counter, OrderedDict
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))

# where to look for the series, relative to this file
SEARCH = [
    os.path.join(HERE, "..", "..", "data", "raw", "macro_economic"),
    os.path.join(HERE, "..", "..", "data", "raw", "sector_assets"),
]

MOMENTUM_WINDOW = 6     # months
VOL_WINDOW = 6          # months
MIN_MONTHS = 12         # a series shorter than this cannot be classified

REGIME_NOTE = {
    "EXPANSION": "Rising market, low volatility. Households take on discretionary and investment products.",
    "RECOVERY": "Rising market, still volatile. Growth returns, caution remains.",
    "SLOWDOWN": "Falling market, low volatility. Consolidation beats consumption.",
    "STRESS": "Falling market, high volatility. Liquidity and safety dominate.",
}

# A readable name for each file, so the console does not show file names.
PRETTY = {
    "brent_crude_oil": "Brent crude",
    "dxy_usdollar": "US dollar index",
    "XLK_Technology": "Technology (XLK)",
    "XLE_Energy": "Energy (XLE)",
    "EEM_Emerging Markets": "Emerging markets (EEM)",
    "JPM_stock": "JPMorgan",
    "BAC_stock": "Bank of America",
    "GS_stock": "Goldman Sachs",
    "AAPL_stock": "Apple",
    "MSFT_stock": "Microsoft",
}


def _monthly_closes(path: str) -> "OrderedDict[str, float]":
    """Last close of each month. Returns {} for a file that is empty or odd."""
    out = OrderedDict()
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as fh:
            rd = csv.reader(fh)
            for row in rd:
                if not row or len(row) < 2:
                    continue
                date, close = row[0].strip(), row[1].strip()
                # skip the three header lines without assuming their exact text
                if len(date) < 10 or date[4] != "-" or date[7] != "-":
                    continue
                try:
                    v = float(close)
                except ValueError:
                    continue
                if v <= 0 or math.isnan(v):
                    continue
                out[date[:7]] = v          # later rows overwrite: last close wins
    except OSError:
        return OrderedDict()
    return out


def _classify_series(closes: "OrderedDict[str, float]") -> List[dict]:
    """Per-month momentum, volatility and regime for one series."""
    months = list(closes.keys())
    if len(months) < MIN_MONTHS:
        return []

    vals = [closes[m] for m in months]
    rets = [None] + [(vals[i] / vals[i - 1] - 1.0) for i in range(1, len(vals))]

    rows = []
    for i in range(len(months)):
        if i < MOMENTUM_WINDOW or i < VOL_WINDOW:
            continue
        mom = vals[i] / vals[i - MOMENTUM_WINDOW] - 1.0
        window = [r for r in rets[i - VOL_WINDOW + 1:i + 1] if r is not None]
        if len(window) < 2:
            continue
        vol = statistics.pstdev(window) * math.sqrt(12)      # annualised
        rows.append({"month": months[i], "momentum": mom, "volatility": vol})

    if not rows:
        return []

    # the threshold is this series' own median volatility, not a magic number
    thresh = statistics.median([r["volatility"] for r in rows])
    for r in rows:
        rising = r["momentum"] > 0
        volatile = r["volatility"] > thresh
        r["regime"] = ("RECOVERY" if volatile else "EXPANSION") if rising \
            else ("STRESS" if volatile else "SLOWDOWN")
        r["threshold"] = thresh
    return rows


def _discover() -> Dict[str, str]:
    """{display name: path} for every readable series we can find."""
    found = {}
    for d in SEARCH:
        d = os.path.normpath(d)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.lower().endswith(".csv"):
                continue
            stem = os.path.splitext(fn)[0]
            found[PRETTY.get(stem, stem.replace("_", " "))] = os.path.join(d, fn)
    return found


class MacroHistory:
    """The regime in force each month, and who voted for it."""

    def __init__(self):
        self.series: Dict[str, List[dict]] = {}
        self.timeline: List[dict] = []
        self.skipped: List[str] = []
        self.available = False
        self.load()

    def load(self):
        for name, path in _discover().items():
            rows = _classify_series(_monthly_closes(path))
            if rows:
                self.series[name] = rows
            else:
                self.skipped.append(name)

        if not self.series:
            return

        # one vote per series per month
        by_month: Dict[str, List[dict]] = {}
        for name, rows in self.series.items():
            for r in rows:
                by_month.setdefault(r["month"], []).append(dict(r, series=name))

        for month in sorted(by_month):
            votes = by_month[month]
            counts = Counter(v["regime"] for v in votes)
            top = counts.most_common()
            mean_mom = sum(v["momentum"] for v in votes) / len(votes)
            mean_vol = sum(v["volatility"] for v in votes) / len(votes)

            if len(top) > 1 and top[0][1] == top[1][1]:
                # a tie is broken by the average direction, and we say so
                tied = [r for r, c in top if c == top[0][1]]
                rising = mean_mom > 0
                pref = ["EXPANSION", "RECOVERY"] if rising else ["SLOWDOWN", "STRESS"]
                regime = next((p for p in pref if p in tied), tied[0])
                tie = True
            else:
                regime = top[0][0]
                tie = False

            self.timeline.append({
                "month": month,
                "regime": regime,
                "momentum": mean_mom,
                "volatility": mean_vol,
                "agreement": top[0][1] / len(votes),
                "n_series": len(votes),
                "tie_broken": tie,
                "votes": {v["series"]: v["regime"] for v in votes},
            })

        self.available = bool(self.timeline)

    # ------------------------------------------------------------------ api
    def at(self, month: Optional[str] = None) -> Optional[dict]:
        """The regime row for a month ('YYYY-MM'), or the latest one."""
        if not self.timeline:
            return None
        if month is None:
            return self.timeline[-1]
        exact = [r for r in self.timeline if r["month"] == month]
        if exact:
            return exact[0]
        earlier = [r for r in self.timeline if r["month"] <= month]
        return earlier[-1] if earlier else self.timeline[0]

    def summary(self) -> dict:
        counts = Counter(r["regime"] for r in self.timeline)
        return {
            "available": self.available,
            "n_series": len(self.series),
            "series": sorted(self.series.keys()),
            "skipped": self.skipped,
            "from_month": self.timeline[0]["month"] if self.timeline else None,
            "to_month": self.timeline[-1]["month"] if self.timeline else None,
            "n_months": len(self.timeline),
            "distribution": dict(counts),
            "latest": self.at(),
            "method": {
                "momentum": f"{MOMENTUM_WINDOW}-month change in the monthly close",
                "volatility": f"annualised stdev of the last {VOL_WINDOW} monthly returns",
                "threshold": "each series' own median volatility across its history",
                "vote": "one vote per series per month, majority wins, "
                        "ties broken by the sign of mean momentum",
            },
        }


if __name__ == "__main__":
    h = MacroHistory()
    s = h.summary()
    print("=" * 70)
    print("MACRO REGIME ENGINE")
    print("=" * 70)
    if not s["available"]:
        print("\n[X] No usable price series found under data/raw/.")
        raise SystemExit(1)
    print(f"\nSeries   : {s['n_series']}  ({', '.join(s['series'])})")
    if s["skipped"]:
        print(f"Skipped  : {', '.join(s['skipped'])}  (too short or empty)")
    print(f"Covers   : {s['from_month']} to {s['to_month']}  ({s['n_months']} months)")
    print(f"\nRegime distribution:")
    for k, v in sorted(s["distribution"].items(), key=lambda kv: -kv[1]):
        print(f"   {k:10} {v:4}  {v/s['n_months']*100:5.1f}%")
    print(f"\nLatest ({s['latest']['month']}): {s['latest']['regime']}"
          f"   momentum {s['latest']['momentum']:+.3f}"
          f"   volatility {s['latest']['volatility']:.3f}"
          f"   agreement {s['latest']['agreement']*100:.0f}%")
    print("\nThe Santander period:")
    for r in h.timeline:
        if "2015-01" <= r["month"] <= "2016-06":
            print(f"   {r['month']}  {r['regime']:10} mom {r['momentum']:+.3f}  "
                  f"vol {r['volatility']:.3f}  agree {r['agreement']*100:3.0f}%")
