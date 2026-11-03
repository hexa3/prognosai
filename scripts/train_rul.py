"""Train + evaluate remaining-useful-life (RUL) models on NASA C-MAPSS FD001 and FD003.

Protocol
--------
* Official train/test split. Test RUL truth comes from NASA's RUL_FDxxx.txt.
* 5-fold GroupKFold by engine on the training set for model selection, so no
  engine appears in both train and validation.
* Point model (L2) + Mondrian split-conformal calibration on out-of-fold residuals
  gives 80% prediction intervals with a coverage guarantee per RUL band.
* The policy backtest turns predictions into maintenance decisions and scores
  them in ₹ against the true RUL of held-out engines.

Run:  python scripts/train_rul.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from prognosai import MODEL_DIR, REPORT_DIR
from prognosai.cmapss import RUL_CAP, SENSORS, MondrianConformal, add_features, conformal_to_dict, feature_columns, load_subset, nasa_score
from prognosai.maintenance import CostModel, realized_cost, rolling_backtest

SEED = 42
FIG = REPORT_DIR / "figures"

# Published deep-learning results on the same test sets (RMSE), for context.
LITERATURE = {
    "FD001": {"CNN (Babu et al., 2016)": 18.45, "LSTM (Zheng et al., 2017)": 16.14, "DCNN (Li et al., 2018)": 12.61},
    "FD003": {"CNN (Babu et al., 2016)": 19.82, "LSTM (Zheng et al., 2017)": 16.18, "DCNN (Li et al., 2018)": 12.64},
}

PARAMS = dict(
    n_estimators=900,
    learning_rate=0.02,
    num_leaves=31,
    min_child_samples=40,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.5,
    reg_lambda=5.0,
    random_state=SEED,
    verbose=-1,
)


def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


def cv_rmse(make_model, X, y, groups):
    oof = np.zeros(len(y))
    for tr, va in GroupKFold(n_splits=5).split(X, y, groups):
        oof[va] = make_model().fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[va])
    return rmse(y, np.clip(oof, 0, RUL_CAP))


def best_fixed_interval(train: pd.DataFrame, cost: CostModel) -> float:
    """Tune the conventional time-based policy on training engines, so the baseline is a fair, well-tuned one."""
    life = train.groupby("unit")["cycle"].max().to_numpy().astype(float)
    grid = np.arange(50, 400)
    totals = [realized_cost(np.full_like(life, T), life, cost).sum() for T in grid]
    return float(grid[int(np.argmin(totals))])


def run(subset: str, cost: CostModel):
    train, test, truth = load_subset(subset)
    trf, tef = add_features(train), add_features(test)
    feats = feature_columns(trf)
    X, y, g = trf[feats], trf["rul"], trf["unit"]
    last = tef.groupby("unit").tail(1).sort_values("unit")
    print(f"\n=== {subset}: {train.unit.nunique()} train engines, {test.unit.nunique()} test engines, {len(feats)} features")

    candidates = {
        "Ridge regression (raw sensors)": (lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)), ["cycle"] + SENSORS),
        "LightGBM (raw sensors)": (lambda: lgb.LGBMRegressor(**PARAMS), ["cycle"] + SENSORS),
        "LightGBM + temporal features (PrognosAI)": (lambda: lgb.LGBMRegressor(**PARAMS), feats),
    }
    ablation = []
    for name, (mk, cols) in candidates.items():
        cv = cv_rmse(mk, trf[cols], y, g)
        m = mk().fit(trf[cols], y)
        p = np.clip(m.predict(last[cols]), 0, RUL_CAP)
        row = {
            "model": name,
            "cv_rmse": cv,
            "test_rmse": rmse(last["rul"], p),
            "test_mae": float(mean_absolute_error(last["rul"], p)),
            "test_nasa_score": nasa_score(last["rul"], p),
        }
        ablation.append(row)
        print(f"{name:45s} CV RMSE={cv:6.2f}  test RMSE={row['test_rmse']:6.2f}  MAE={row['test_mae']:5.2f}  NASA={row['test_nasa_score']:8.1f}")

    def make_point():
        return lgb.LGBMRegressor(**PARAMS)

    # Out-of-fold predictions on the training engines (each engine is scored by a model that never saw it).
    oof = trf[["unit", "cycle", "rul", "rul_true"]].copy()
    oof["rul_p50"] = 0.0
    for tr_i, va_i in GroupKFold(n_splits=5).split(X, y, g):
        m = make_point().fit(X.iloc[tr_i], y.iloc[tr_i])
        oof.iloc[va_i, oof.columns.get_loc("rul_p50")] = np.clip(m.predict(X.iloc[va_i]), 0, RUL_CAP)

    # Uncertainty: Mondrian split-conformal intervals calibrated on out-of-fold residuals.
    conformal = MondrianConformal(level=0.8).fit(oof["rul_p50"], oof["rul"])
    print("conformal residual bands (lo, hi) per predicted-RUL bin:",
          [(round(a, 1), round(b, 1)) for a, b in zip(conformal.lo_, conformal.hi_)])
    oof["rul_p10"], oof["rul_p90"] = conformal.interval(oof["rul_p50"])

    point = make_point().fit(X, y)

    def predict(frame):
        mid = np.clip(point.predict(frame[feats]), 0, RUL_CAP)
        lo, hi = conformal.interval(mid)
        return lo, mid, hi

    tef["rul_p10"], tef["rul_p50"], tef["rul_p90"] = predict(tef)
    last = tef.groupby("unit").tail(1).sort_values("unit").reset_index(drop=True)
    final = {
        "test_rmse": rmse(last["rul"], last["rul_p50"]),
        "test_mae": float(mean_absolute_error(last["rul"], last["rul_p50"])),
        "test_nasa_score": nasa_score(last["rul"], last["rul_p50"]),
        "interval_80_coverage": float(((last["rul"] >= last["rul_p10"]) & (last["rul"] <= last["rul_p90"])).mean()),
        "within_15_cycles": float((np.abs(last["rul"] - last["rul_p50"]) <= 15).mean()),
        # Accuracy where it matters most: engines close to failure.
        "rmse_true_rul_le_50": rmse(last.loc[last.rul_true <= 50, "rul"], last.loc[last.rul_true <= 50, "rul_p50"]),
        # Coverage over every test row (all truncation points, 10k+ predictions), not just the 100 final cycles.
        "interval_80_coverage_all_rows": float(((tef["rul"] >= tef["rul_p10"]) & (tef["rul"] <= tef["rul_p90"])).mean()),
        "test_rows": int(len(tef)),
    }
    print("final:", json.dumps({k: round(v, 3) for k, v in final.items()}))

    fixed_T = best_fixed_interval(train, cost)
    rbt = rolling_backtest(oof, cost, fixed_T)
    print("rolling backtest (full lives, out-of-fold, re-decided every cycle):")
    print(rbt.to_string(index=False))

    # ---- figures ----
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    order = np.argsort(last["rul"].to_numpy())
    xs = np.arange(len(order))
    ax[0].fill_between(xs, last["rul_p10"].to_numpy()[order], last["rul_p90"].to_numpy()[order], alpha=0.25, label="80% interval")
    ax[0].plot(xs, last["rul_p50"].to_numpy()[order], "o", ms=3.5, label="Predicted RUL")
    ax[0].plot(xs, last["rul"].to_numpy()[order], "k-", lw=2, label="True RUL (capped)")
    ax[0].set(title=f"{subset} test engines, sorted by true RUL (RMSE {final['test_rmse']:.2f})", xlabel="Engine (sorted)", ylabel="RUL (cycles)")
    ax[0].legend()
    err = last["rul_p50"] - last["rul"]
    ax[1].hist(err, bins=25, edgecolor="white")
    ax[1].axvline(0, c="k")
    ax[1].set(title="Prediction error (negative = early = safe side)", xlabel="Predicted − true (cycles)", ylabel="Engines")
    plt.tight_layout()
    plt.savefig(FIG / f"rul_{subset}.png", dpi=150)
    plt.close("all")

    imp = pd.Series(point.booster_.feature_importance("gain"), index=feats).sort_values(ascending=False)
    point.booster_.save_model(str(MODEL_DIR / f"rul_{subset}.txt"))
    (MODEL_DIR / f"rul_{subset}_meta.json").write_text(json.dumps({"features": feats, "conformal": conformal_to_dict(conformal)}, indent=1))
    keep = ["unit", "cycle", "rul", "rul_true", "rul_p10", "rul_p50", "rul_p90"] + SENSORS
    tef[keep].round(4).to_csv(MODEL_DIR / f"rul_{subset}_test_predictions.csv", index=False)
    return {
        "subset": subset,
        "train_engines": int(train.unit.nunique()),
        "test_engines": int(test.unit.nunique()),
        "n_features": len(feats),
        "ablation": ablation,
        "final": final,
        "literature_rmse": LITERATURE[subset],
        "fixed_interval_age": fixed_T,
        "conformal_bins": {"edges": list(conformal.EDGES), "lo": conformal.lo_.round(2).tolist(), "hi": conformal.hi_.round(2).tolist()},
        "rolling_backtest": rbt.to_dict(orient="records"),
        "top_features": (imp.head(15) / imp.sum()).round(4).to_dict(),
    }


def main():
    cost = CostModel()
    results = {"cost_model": cost.__dict__, "rul_cap": RUL_CAP, "subsets": {}}
    for subset in ("FD001", "FD003"):
        results["subsets"][subset] = run(subset, cost)
    (MODEL_DIR / "rul_metrics.json").write_text(json.dumps(results, indent=2))
    print("saved ->", MODEL_DIR)


if __name__ == "__main__":
    main()
