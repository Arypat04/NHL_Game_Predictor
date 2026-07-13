"""
MLB Model Training & Evaluation (matchup model).

Temporal train/test split on the one-row-per-game matchup dataset (both teams'
offense/staff + both starting pitchers). Reports per-model and ensemble
precision / accuracy / AUC-ROC / Brier, plus feature importance and the starting
pitcher's share of it. Config is imported from the live modules so this always
evaluates production.

Only knob here is TEST_SEASON.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, brier_score_loss, precision_score, roc_auc_score,
)

import mlb_matchup as M
from mlb_pitchers import sp_feature_cols
from mlb_predictor import ENSEMBLE_WEIGHTS, TRAIN_CSV, _build_models

warnings.filterwarnings("ignore")

TEST_SEASON = 2025


def load_data() -> tuple[pd.DataFrame, list[str]]:
    raw = pd.read_csv(TRAIN_CSV)
    return M.build_matchup(raw)


def evaluate(game: pd.DataFrame, cols: list[str]) -> None:
    train = game[game["Season"] < TEST_SEASON]
    test  = game[game["Season"] == TEST_SEASON]
    if train.empty or test.empty:
        print(f"⚠ No data for split at season {TEST_SEASON} "
              f"(seasons present: {sorted(game['Season'].unique())})")
        return

    Xtr, ytr = train[cols].fillna(train[cols].median()), train["home_win"]
    Xte, yte = test[cols].fillna(train[cols].median()), test["home_win"]
    print(f"Train {len(train):,} games | Test (season {TEST_SEASON}) {len(test):,} games\n")

    print(f"{'Model':<22} {'Precision':>10} {'Accuracy':>10} {'AUC-ROC':>10} {'Brier':>8}")
    print("-" * 64)
    ens = np.zeros(len(Xte))
    for name, model in _build_models().items():
        cal = CalibratedClassifierCV(model, cv=3, method="sigmoid")
        cal.fit(Xtr, ytr)
        probs = cal.predict_proba(Xte)[:, 1]
        ens += probs * ENSEMBLE_WEIGHTS[name]
        preds = (probs >= 0.5).astype(int)
        print(f"{name:<22} {precision_score(yte, preds, zero_division=0):>10.3f} "
              f"{accuracy_score(yte, preds):>10.3f} {roc_auc_score(yte, probs):>10.3f} "
              f"{brier_score_loss(yte, probs):>8.4f}")

    ep = (ens >= 0.5).astype(int)
    print("-" * 64)
    print(f"{'Ensemble (weighted)':<22} {precision_score(yte, ep, zero_division=0):>10.3f} "
          f"{accuracy_score(yte, ep):>10.3f} {roc_auc_score(yte, ens):>10.3f} "
          f"{brier_score_loss(yte, ens):>8.4f}")
    print(f"\n  Baseline (always home): {yte.mean():.3f}")

    rf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=10,
                                n_jobs=-1, random_state=42)
    rf.fit(Xtr, ytr)
    imp = pd.Series(rf.feature_importances_, index=cols)
    sp_cols = [f"home_{c}" for c in sp_feature_cols()] + [f"away_{c}" for c in sp_feature_cols()]
    print(f"\n  Starting-pitcher importance share: {imp[sp_cols].sum():.3f}")
    print("\n--- Top 12 features ---")
    print(imp.sort_values(ascending=False).head(12).to_string())


if __name__ == "__main__":
    print("=" * 64)
    print("  MLB Matchup Model — Training & Evaluation")
    print("=" * 64 + "\n")
    game, cols = load_data()
    evaluate(game, cols)
