"""
NHL Model Training & Evaluation
--------------------------------
Replaces Prediction.ipynb.

Runs a full train/test evaluation and prints a breakdown of:
  - Per-model precision
  - Ensemble precision
  - Per-team prediction accuracy
  - Calibration summary (how well probabilities reflect actual outcomes)

Run this whenever you want to re-evaluate model quality before deploying
or after re-scraping fresh training data.
"""

import warnings

import pandas as pd
import numpy as np

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    brier_score_loss,
    precision_score,
    accuracy_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

# Import shared logic from predictor
from predictor import (
    ENSEMBLE_WEIGHTS,
    ROLLING_WINDOWS,
    ROLLING_STAT_COLS,
    BASE_PREDICTORS,
    prepare,
    _rolling_averages,
    _build_models,
)

warnings.filterwarnings("ignore")

TRAIN_CUTOFF = "2024-06-30"
TRAIN_FILE = "nhl_matches_2021_2025.csv"


# ---------------------------------------------------------------------------
# Load & prepare
# ---------------------------------------------------------------------------

def load_data(path: str = TRAIN_FILE) -> tuple[pd.DataFrame, list[str]]:
    raw = pd.read_csv(path)
    prepared = prepare(raw)
    rolled, rolling_cols = _rolling_averages(prepared)
    predictors = BASE_PREDICTORS + rolling_cols
    return rolled, predictors


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(data: pd.DataFrame, predictors: list[str]) -> None:
    train = data[data["Date"] < TRAIN_CUTOFF]
    test = data[data["Date"] >= TRAIN_CUTOFF]

    print(f"Train rows : {len(train):,}  ({train['Date'].min().date()} – {train['Date'].max().date()})")
    print(f"Test rows  : {len(test):,}  ({test['Date'].min().date()} – {test['Date'].max().date()})\n")

    X_train, y_train = train[predictors], train["target"]
    X_test, y_test = test[predictors], test["target"]

    raw_models = _build_models()
    fitted: dict = {}
    all_probs: list[np.ndarray] = []

    # -----------------------------------------------------------------------
    # Per-model metrics
    # -----------------------------------------------------------------------
    print(f"{'Model':<22} {'Precision':>10} {'Accuracy':>10} {'AUC-ROC':>10} {'Brier':>8}")
    print("-" * 64)

    for name, model in raw_models.items():
        calibrated = CalibratedClassifierCV(model, cv=3, method="sigmoid")
        calibrated.fit(X_train, y_train)
        fitted[name] = calibrated

        probs = calibrated.predict_proba(X_test)[:, 1]
        preds = (probs > 0.5).astype(int)
        all_probs.append(probs * ENSEMBLE_WEIGHTS[name])

        precision = precision_score(y_test, preds)
        accuracy = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        brier = brier_score_loss(y_test, probs)

        print(f"{name:<22} {precision:>10.3f} {accuracy:>10.3f} {auc:>10.3f} {brier:>8.4f}")

    # -----------------------------------------------------------------------
    # Ensemble metrics
    # -----------------------------------------------------------------------
    ensemble_probs = np.sum(all_probs, axis=0)
    ensemble_preds = (ensemble_probs > 0.5).astype(int)

    e_precision = precision_score(y_test, ensemble_preds)
    e_accuracy = accuracy_score(y_test, ensemble_preds)
    e_auc = roc_auc_score(y_test, ensemble_probs)
    e_brier = brier_score_loss(y_test, ensemble_probs)

    print("-" * 64)
    print(f"{'ensemble (weighted)':<22} {e_precision:>10.3f} {e_accuracy:>10.3f} {e_auc:>10.3f} {e_brier:>8.4f}")

    # -----------------------------------------------------------------------
    # Calibration check
    # -----------------------------------------------------------------------
    print("\n--- Calibration (fraction of positives per confidence bucket) ---")
    print(f"{'Bucket':>14} {'Mean pred prob':>16} {'Actual win rate':>16} {'Count':>8}")
    frac_pos, mean_pred = calibration_curve(y_test, ensemble_probs, n_bins=5)
    counts, edges = np.histogram(ensemble_probs, bins=5)
    for i, (fp, mp) in enumerate(zip(frac_pos, mean_pred)):
        label = f"{edges[i]:.2f}–{edges[i+1]:.2f}"
        print(f"{label:>14} {mp:>16.3f} {fp:>16.3f} {counts[i]:>8,}")

    # -----------------------------------------------------------------------
    # Per-team accuracy
    # -----------------------------------------------------------------------
    combined = test.assign(predicted=ensemble_preds, win_prob=ensemble_probs)
    team_summary = (
        combined.groupby("Team")
        .apply(
            lambda g: pd.Series({
                "games": len(g),
                "actual_wins": int(g["target"].sum()),
                "predicted_wins": int(g["predicted"].sum()),
                "accuracy": round((g["target"] == g["predicted"]).mean(), 3),
                "avg_confidence": round(g["win_prob"].apply(lambda p: max(p, 1-p)).mean(), 3),
            }),
            include_groups=False,
        )
        .sort_values("accuracy", ascending=False)
        .reset_index()
    )

    print("\n--- Per-team accuracy (test set) ---")
    print(team_summary.to_string(index=False))

    # -----------------------------------------------------------------------
    # Rolling window contribution check
    # -----------------------------------------------------------------------
    print("\n--- Feature importance by rolling window ---")
    # Use the RF model (most interpretable)
    rf_base = raw_models["rf"]
    rf_base.fit(X_train, y_train)
    importance = pd.Series(
        rf_base.feature_importances_, index=predictors
    ).sort_values(ascending=False)

    for window in ROLLING_WINDOWS:
        window_cols = [c for c in predictors if c.startswith(f"avg{window}_")]
        window_importance = importance[window_cols].sum()
        print(f"  avg{window}_* total importance : {window_importance:.3f}")

    print(f"  base features total importance: {importance[BASE_PREDICTORS].sum():.3f}")
    print(f"\n  Top 10 individual features:")
    print(importance.head(10).to_string())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 64)
    print("  NHL Prediction Model — Training & Evaluation")
    print("=" * 64 + "\n")

    data, predictors = load_data()
    evaluate(data, predictors)
