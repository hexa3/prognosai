"""Train + evaluate the machine-failure classifier and the failure-mode (root-cause) models on AI4I 2020.

Protocol
--------
* 80/20 stratified split. The test set is touched only once, for final reporting.
* 5-fold stratified CV on the training set produces out-of-fold probabilities.
  These are used to pick the decision threshold, with no peeking at test data.
* Ablation: logistic regression vs. LightGBM on raw features vs. LightGBM on
  physics-informed features.

Run:  python scripts/train_failure.py
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
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from prognosai import MODEL_DIR, REPORT_DIR
from prognosai.ai4i import FAILURE_MODES, FEATURES, RAW_FEATURES, load_dataset

SEED = 42
FIG = REPORT_DIR / "figures"
FIG.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Business costs for a mid-size CNC shop (₹). Missing a failure costs far more than an extra inspection.
COST_FN = 50_000  # missed failure: scrapped part, spindle damage, unplanned downtime
COST_FP = 5_000   # false alarm: technician inspection + short stop

LGB_PARAMS = dict(
    n_estimators=600,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=10,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=SEED,
    verbose=-1,
)


def lgbm():
    return lgb.LGBMClassifier(**LGB_PARAMS)


def oof_probs(make_model, X, y, n_splits=5):
    oof = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for tr, va in skf.split(X, y):
        m = make_model().fit(X.iloc[tr], y.iloc[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    return oof


def metrics_at(y, p, thr):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {
        "threshold": float(thr),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "f1": float(f1_score(y, yhat, zero_division=0)),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat, zero_division=0)),
        "accuracy": float((yhat == y).mean()),
        "brier": float(brier_score_loss(y, p)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "business_cost_inr": int(fn * COST_FN + fp * COST_FP),
    }


def best_thresholds(y, p):
    grid = np.linspace(0.01, 0.95, 189)
    f1s = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in grid]
    costs = []
    for t in grid:
        yhat = (p >= t).astype(int)
        fn = int(((yhat == 0) & (y == 1)).sum())
        fp = int(((yhat == 1) & (y == 0)).sum())
        costs.append(fn * COST_FN + fp * COST_FP)
    return float(grid[int(np.argmax(f1s))]), float(grid[int(np.argmin(costs))]), grid, np.array(costs)


def main():
    df = load_dataset()
    y = df["failure"].astype(int)
    train_idx, test_idx = train_test_split(df.index, test_size=0.2, stratify=y, random_state=SEED)
    tr, te = df.loc[train_idx], df.loc[test_idx]
    ytr, yte = y.loc[train_idx], y.loc[test_idx]
    print(f"train={len(tr)}  test={len(te)}  failure rate={y.mean():.3%}")

    # ---------- Ablation on the held-out test set ----------
    candidates = {
        "Logistic regression (raw)": (lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")), RAW_FEATURES),
        "LightGBM (raw features)": (lgbm, RAW_FEATURES),
        "LightGBM + physics features (PrognosAI)": (lgbm, FEATURES),
    }
    ablation = []
    for name, (mk, feats) in candidates.items():
        oof = oof_probs(mk, tr[feats], ytr)
        thr_f1, _, _, _ = best_thresholds(ytr.to_numpy(), oof)
        model = mk().fit(tr[feats], ytr)
        p = model.predict_proba(te[feats])[:, 1]
        m = metrics_at(yte.to_numpy(), p, thr_f1)
        m["cv_pr_auc"] = float(average_precision_score(ytr, oof))
        m["model"] = name
        ablation.append(m)
        print(f"{name:45s} test ROC-AUC={m['roc_auc']:.4f} PR-AUC={m['pr_auc']:.4f} F1={m['f1']:.4f} (CV PR-AUC {m['cv_pr_auc']:.4f})")

    # ---------- Final model ----------
    oof = oof_probs(lgbm, tr[FEATURES], ytr)
    thr_f1, thr_cost, grid, cost_curve = best_thresholds(ytr.to_numpy(), oof)
    model = lgbm().fit(tr[FEATURES], ytr)
    p_test = model.predict_proba(te[FEATURES])[:, 1]
    final = {
        "f1_optimal": metrics_at(yte.to_numpy(), p_test, thr_f1),
        "cost_optimal": metrics_at(yte.to_numpy(), p_test, thr_cost),
        "default_0.5": metrics_at(yte.to_numpy(), p_test, 0.5),
    }
    # A naive "never predict failure" baseline sets the reference point for cost.
    final["never_alarm_cost_inr"] = int(yte.sum() * COST_FN)
    print(json.dumps(final["cost_optimal"], indent=1))

    # ---------- Failure-mode models (root cause) ----------
    mode_models, mode_metrics = {}, {}
    for mode in FAILURE_MODES:
        ym_tr, ym_te = df.loc[train_idx, mode], df.loc[test_idx, mode]
        mm = lgb.LGBMClassifier(**{**LGB_PARAMS, "n_estimators": 400}).fit(tr[FEATURES], ym_tr)
        pm = mm.predict_proba(te[FEATURES])[:, 1]
        mode_models[mode] = mm
        mode_metrics[mode] = {
            "positives_test": int(ym_te.sum()),
            "roc_auc": float(roc_auc_score(ym_te, pm)) if ym_te.sum() else None,
            "pr_auc": float(average_precision_score(ym_te, pm)) if ym_te.sum() else None,
        }
        print(f"mode {mode}: {mode_metrics[mode]}")

    # ---------- Explainability ----------
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(te[FEATURES])
    sv = sv[1] if isinstance(sv, list) else sv
    importance = pd.Series(np.abs(sv).mean(0), index=FEATURES).sort_values(ascending=False)

    plt.figure()
    shap.summary_plot(sv, te[FEATURES], show=False, max_display=12)
    plt.title("What drives failure risk? (SHAP, test set)")
    plt.tight_layout()
    plt.savefig(FIG / "failure_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    # ---------- Figures ----------
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    fpr, tpr, _ = roc_curve(yte, p_test)
    ax[0].plot(fpr, tpr, lw=2, label=f"AUC = {final['cost_optimal']['roc_auc']:.3f}")
    ax[0].plot([0, 1], [0, 1], "--", c="grey")
    ax[0].set(title="ROC (test)", xlabel="False positive rate", ylabel="True positive rate")
    ax[0].legend()
    prec, rec, _ = precision_recall_curve(yte, p_test)
    ax[1].plot(rec, prec, lw=2, label=f"AP = {final['cost_optimal']['pr_auc']:.3f}")
    ax[1].axhline(yte.mean(), ls="--", c="grey", label=f"base rate {yte.mean():.1%}")
    ax[1].set(title="Precision-Recall (test)", xlabel="Recall", ylabel="Precision")
    ax[1].legend()
    ax[2].plot(grid, cost_curve / 1e5, lw=2)
    ax[2].axvline(thr_cost, ls="--", c="red", label=f"cost-optimal = {thr_cost:.2f}")
    ax[2].axvline(thr_f1, ls=":", c="green", label=f"F1-optimal = {thr_f1:.2f}")
    ax[2].set(title="Business cost vs threshold (CV, train)", xlabel="Alarm threshold", ylabel="Cost (₹ lakh)")
    ax[2].legend()
    plt.tight_layout()
    plt.savefig(FIG / "failure_curves.png", dpi=150)
    plt.close("all")

    # ---------- Persist ----------
    # Portable format: LightGBM text models + JSON metadata (loads in any LightGBM 4.x, including in-browser Pyodide).
    model.booster_.save_model(str(MODEL_DIR / "failure_model.txt"))
    for mode, mm in mode_models.items():
        mm.booster_.save_model(str(MODEL_DIR / f"failure_mode_{mode}.txt"))
    (MODEL_DIR / "failure_model_meta.json").write_text(json.dumps(
        {"features": FEATURES, "threshold_cost": thr_cost, "threshold_f1": thr_f1, "cost_fn": COST_FN, "cost_fp": COST_FP}, indent=1))
    out = {
        "dataset": "AI4I 2020 Predictive Maintenance (UCI #601), 10,000 rows",
        "split": {"train": len(tr), "test": len(te), "test_failures": int(yte.sum())},
        "ablation": ablation,
        "final": final,
        "failure_modes": mode_metrics,
        "shap_importance": importance.round(5).to_dict(),
    }
    (MODEL_DIR / "failure_metrics.json").write_text(json.dumps(out, indent=2))
    te.assign(p_failure=p_test).to_csv(MODEL_DIR / "failure_test_predictions.csv", index=False)
    print("saved ->", MODEL_DIR)


if __name__ == "__main__":
    main()
