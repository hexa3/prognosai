"""NASA C-MAPSS turbofan data loading + time-series feature engineering for RUL.

Each engine runs from healthy to failure in the training set. In the test set
each trajectory is cut off at a random point, and the goal is to predict the
remaining useful life (RUL) at that last cycle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import DATA_DIR

CMAPSS_DIR = DATA_DIR / "cmapss"
COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]

# Sensors that stay (almost) constant in FD001/FD003 carry no information.
DROP_SENSORS = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]
SENSORS = [f"s{i}" for i in range(1, 22) if f"s{i}" not in DROP_SENSORS]

SENSOR_NAMES = {
    "s2": "LPC outlet temp (°R)",
    "s3": "HPC outlet temp (°R)",
    "s4": "LPT outlet temp (°R)",
    "s7": "HPC outlet pressure (psia)",
    "s8": "Physical fan speed (rpm)",
    "s9": "Physical core speed (rpm)",
    "s11": "HPC outlet static pressure (psia)",
    "s12": "Fuel flow / Ps30 ratio",
    "s13": "Corrected fan speed (rpm)",
    "s14": "Corrected core speed (rpm)",
    "s15": "Bypass ratio",
    "s17": "Bleed enthalpy",
    "s20": "HPT coolant bleed (lbm/s)",
    "s21": "LPT coolant bleed (lbm/s)",
}

RUL_CAP = 125  # piecewise-linear target: degradation is not observable early in life
WINDOWS = (5, 10, 20, 30)


def _read(path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.iloc[:, : len(COLS)]
    df.columns = COLS
    return df


def load_subset(subset: str = "FD001"):
    """Return (train, test, test_truth_rul) for a C-MAPSS subset."""
    train = _read(CMAPSS_DIR / f"train_{subset}.txt")
    test = _read(CMAPSS_DIR / f"test_{subset}.txt")
    truth = pd.read_csv(CMAPSS_DIR / f"RUL_{subset}.txt", header=None, names=["rul"])
    truth["unit"] = np.arange(1, len(truth) + 1)

    max_cycle = train.groupby("unit")["cycle"].transform("max")
    train["rul_true"] = (max_cycle - train["cycle"]).astype(float)
    train["rul"] = train["rul_true"].clip(upper=RUL_CAP)

    # The true RUL of every test row is known too: truth at the last cycle plus the cycles still to come.
    last = test.groupby("unit")["cycle"].transform("max")
    test = test.merge(truth.rename(columns={"rul": "rul_end"}), on="unit")
    test["rul_true"] = (test["rul_end"] + last - test["cycle"]).astype(float)
    test["rul"] = test["rul_true"].clip(upper=RUL_CAP)
    test = test.drop(columns="rul_end")
    return train, test, truth


def _rolling_slope(s: pd.Series, w: int) -> pd.Series:
    """Least-squares slope over a trailing window (vectorised via rolling sums)."""
    x = pd.Series(np.arange(len(s), dtype=float), index=s.index)
    n = s.rolling(w, min_periods=2).count()
    sx = x.rolling(w, min_periods=2).sum()
    sy = s.rolling(w, min_periods=2).sum()
    sxy = (x * s).rolling(w, min_periods=2).sum()
    sxx = (x * x).rolling(w, min_periods=2).sum()
    denom = n * sxx - sx**2
    return ((n * sxy - sx * sy) / denom.replace(0, np.nan)).fillna(0.0)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add trailing-window features per engine. Only past data is used, so there is no look-ahead leakage."""
    parts = []
    for _, g in df.sort_values(["unit", "cycle"]).groupby("unit", sort=False):
        g = g.copy()
        feats = {}
        for s in SENSORS:
            col = g[s]
            # Exponentially weighted mean smooths sensor noise while reacting faster than a plain rolling mean.
            feats[f"{s}_ewm"] = col.ewm(alpha=0.1, adjust=False).mean()
            for w in WINDOWS:
                r = col.rolling(w, min_periods=1)
                feats[f"{s}_mean{w}"] = r.mean()
                if w in (10, 30):
                    feats[f"{s}_std{w}"] = r.std().fillna(0.0)
            feats[f"{s}_slope20"] = _rolling_slope(col, 20)
            feats[f"{s}_slope50"] = _rolling_slope(col, 50)
            feats[f"{s}_delta"] = col - col.iloc[: min(10, len(col))].mean()  # drift from this engine's own baseline
        parts.append(pd.concat([g, pd.DataFrame(feats, index=g.index)], axis=1))
    return pd.concat(parts).reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"unit", "rul", "rul_true", "op1", "op2", "op3"} | set(DROP_SENSORS)
    return [c for c in df.columns if c not in excluded]


def nasa_score(y_true, y_pred) -> float:
    """PHM08 asymmetric scoring function: late predictions are penalised more than early ones."""
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)))


class MondrianConformal:
    """Bin-conditional (Mondrian) split-conformal prediction intervals around a point RUL model.

    Errors depend strongly on how far an engine is from failure: near the end of life
    sensors are informative and errors are small. Far from failure the cap and noise
    dominate. We therefore calibrate a separate residual distribution for each band of
    *predicted* RUL. Within each band the interval has finite-sample coverage of at least
    `level` under exchangeability (Vovk et al., 2005; Angelopoulos & Bates, 2021).
    """

    EDGES = (0, 15, 30, 50, 75, 100, RUL_CAP + 1)

    def __init__(self, level: float = 0.8):
        self.level = level
        self.lo_: np.ndarray | None = None
        self.hi_: np.ndarray | None = None

    def _bin(self, pred):
        return np.clip(np.digitize(pred, self.EDGES[1:-1]), 0, len(self.EDGES) - 2)

    def fit(self, pred, y):
        pred, y = np.asarray(pred, float), np.asarray(y, float)
        res, b = y - pred, self._bin(pred)
        a = (1 - self.level) / 2
        lo, hi = [], []
        for k in range(len(self.EDGES) - 1):
            r = np.sort(res[b == k])
            n = len(r)
            if n < 20:  # too few points: fall back to the global distribution
                r, n = np.sort(res), len(res)
            lo.append(r[max(int(np.floor((n + 1) * a)) - 1, 0)])
            hi.append(r[min(int(np.ceil((n + 1) * (1 - a))) - 1, n - 1)])
        self.lo_, self.hi_ = np.array(lo), np.array(hi)
        return self

    def interval(self, pred):
        pred = np.asarray(pred, float)
        b = self._bin(pred)
        return np.clip(pred + self.lo_[b], 0, RUL_CAP), np.clip(pred + self.hi_[b], 0, RUL_CAP)


def conformal_from_dict(d: dict) -> MondrianConformal:
    c = MondrianConformal(level=d.get("level", 0.8))
    c.lo_, c.hi_ = np.asarray(d["lo"], float), np.asarray(d["hi"], float)
    return c


def conformal_to_dict(c: MondrianConformal) -> dict:
    return {"level": c.level, "edges": list(c.EDGES), "lo": c.lo_.tolist(), "hi": c.hi_.tolist()}
