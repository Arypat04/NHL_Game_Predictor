"""
NHL Game Predictor — matchup model.

Predicts P(home win) from BOTH teams' rolling form (offense + possession +
special teams). This replaced the older "one team's stats + opponent code"
model; using both teams' Corsi/Fenwick form improves probability quality.
Goalie / individual-scorer features were tested and dropped (no lift — team
shot-share already prices them in). See nhl_matchup.py.

Data: MongoDB first, CSV fallback. Model: nhl_model.pkl (joblib).
Output schema is unchanged (Time/Away/Home/Predicted_Winner/probs/Confidence),
so main.py, /results, /edges and the frontend are unaffected.
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

import nhl_matchup as M
from nhl_teams import TEAM_NAME_TO_ABBREV  # re-exported for scraper/main imports

warnings.filterwarnings("ignore")

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "nhl_model.pkl")
TRAIN_CSV    = os.path.join(BASE_DIR, "../data/nhl_matches_2021_2025.csv")
SCHEDULE_CSV = os.path.join(BASE_DIR, "../data/nhl_matches_2026.csv")

# The exhaustive search (model_search.py) found bagged trees (RandomForest,
# ExtraTrees) clearly beat boosting (XGBoost/GB) on this noisy data — the old
# xgb-heavy ensemble was over-weighting the weaker models. Blend the two winners.
ENSEMBLE_WEIGHTS = {"rf": 0.5, "et": 0.5}


def _build_models() -> dict:
    return {
        "rf": RandomForestClassifier(
            n_estimators=300, max_depth=7, min_samples_leaf=15,
            max_features="sqrt", random_state=42, n_jobs=-1),
        "et": ExtraTreesClassifier(
            n_estimators=400, max_depth=12, min_samples_leaf=8,
            max_features="sqrt", random_state=42, n_jobs=-1),
    }


# ---------------------------------------------------------------------------
# Data loading (MongoDB first, CSV fallback)
# ---------------------------------------------------------------------------

def _load_team_data() -> pd.DataFrame:
    """All team-game logs (training seasons + rich current-season completed
    games) — the source for both training and the rolling lookups."""
    if os.getenv("MONGODB_URI"):
        try:
            from nhl_database import get_training_data
            df = get_training_data()   # games collection incl. rich current-season games
            if not df.empty:
                print(f"  Loaded {len(df):,} team-game rows from MongoDB")
                return df
        except Exception as e:
            print(f"  ⚠ MongoDB load failed: {e} — falling back to CSV")
    print("  Loading team-game data from CSV...")
    df = pd.read_csv(TRAIN_CSV)
    df["Date"] = pd.to_datetime(df["Date"])
    # include rich current-season completed games for up-to-date rolling
    try:
        cur = pd.read_csv(SCHEDULE_CSV)
        cur["Date"] = pd.to_datetime(cur["Date"])
        cur = cur.dropna(subset=["Rslt"])
        if not cur.empty and "CF%" in cur.columns:   # only if the file is rich
            df = pd.concat([df, cur], ignore_index=True)
    except Exception:
        pass
    return df


def _load_schedule_data() -> pd.DataFrame:
    if os.getenv("MONGODB_URI"):
        try:
            from nhl_database import get_schedule_df
            df = get_schedule_df()
            if not df.empty:
                return df
        except Exception:
            pass
    df = pd.read_csv(SCHEDULE_CSV)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


# ---------------------------------------------------------------------------
# NHLPredictor
# ---------------------------------------------------------------------------

class NHLPredictor:
    def __init__(self):
        self.models: dict            = {}
        self._feature_cols: list[str] = []
        self._team_feats: list[str]   = []
        self._rolled: pd.DataFrame    = pd.DataFrame()
        self._schedule: pd.DataFrame  = pd.DataFrame()
        self._load_and_train()

    def _should_retrain(self) -> bool:
        return not os.path.exists(MODEL_PATH)

    def _load_and_train(self) -> None:
        # all team-game logs incl. rich current-season completed games
        team = _load_team_data()

        # rolling frame used for live prediction lookups
        self._rolled, self._team_feats = M.team_rolling(team)
        self._feature_cols = M.feature_cols(self._team_feats)

        if self._should_retrain():
            print("No saved NHL model — training matchup model from scratch...")
            self._train_and_save(team)
        else:
            print("Loading saved model...")
            saved = joblib.load(MODEL_PATH)
            if saved.get("feature_cols") != self._feature_cols:
                print("  ⚠ Saved feature set is stale — retraining...")
                self._train_and_save(team)
            else:
                self.models = saved["models"]
                print("✓ Model loaded.\n")

        self._load_schedule()

    def _train_and_save(self, team: pd.DataFrame) -> None:
        game, cols = M.build_matchup(team)
        X, y = game[cols], game["home_win"]
        print(f"  Training on {len(X):,} games, {len(cols)} features...")
        for name, model in _build_models().items():
            print(f"    {name}...")
            cal = CalibratedClassifierCV(model, cv=3, method="sigmoid")
            cal.fit(X, y)
            self.models[name] = cal
        joblib.dump({"models": self.models, "feature_cols": self._feature_cols,
                     "team_feats": self._team_feats}, MODEL_PATH)
        print("✓ Model trained and saved.\n")

    def _load_schedule(self) -> None:
        schedule = _load_schedule_data()
        schedule["Date"] = pd.to_datetime(schedule["Date"])
        opp_col = "Opponent" if "Opponent" in schedule.columns else "Opp"
        schedule["Opp_abbrev"] = schedule[opp_col].map(TEAM_NAME_TO_ABBREV)
        self._schedule = schedule

    # -- prediction ---------------------------------------------------------

    def _latest_rolling(self, team: str, before: pd.Timestamp) -> pd.Series | None:
        rows = self._rolled[(self._rolled["Team"] == team) &
                            (self._rolled["Date"] < before)].sort_values("Date")
        return rows.iloc[-1] if not rows.empty else None

    def _ensemble_prob(self, X: pd.DataFrame) -> float:
        return float(sum(m.predict_proba(X)[:, 1][0] * ENSEMBLE_WEIGHTS[n]
                         for n, m in self.models.items()))

    def predict_date(self, date_str: str) -> pd.DataFrame | None:
        if date_str.lower() == "today":
            target = pd.Timestamp.now().normalize()
        else:
            try:
                target = pd.to_datetime(date_str)
            except Exception:
                print(f"Invalid date format '{date_str}'.")
                return None

        games = self._schedule[(self._schedule["Date"] == target) &
                               (self._schedule["Home_Away"] == "Home")].copy()
        if games.empty:
            print(f"No games scheduled for {target.date()}.")
            return None

        rows = []
        for _, g in games.iterrows():
            home = g["Team"]
            away = g.get("Opp_abbrev") or TEAM_NAME_TO_ABBREV.get(g.get("Opp", ""))
            if not away:
                continue
            hr = self._latest_rolling(home, target)
            ar = self._latest_rolling(away, target)
            if hr is None or ar is None:
                continue

            feat = {}
            for f in self._team_feats:
                feat[f"home_{f}"] = hr[f]
                feat[f"away_{f}"] = ar[f]
            feat["dow"]       = target.dayofweek
            feat["home_rest"] = min(max((target - hr["Date"]).days, 0), 7)
            feat["away_rest"] = min(max((target - ar["Date"]).days, 0), 7)
            X = pd.DataFrame([feat])[self._feature_cols]

            p = self._ensemble_prob(X)
            rows.append({
                "Time":             g.get("Time", ""),
                "Away":             away,
                "Home":             home,
                "Predicted_Winner": home if p > 0.5 else away,
                "Home_Win_Prob":    p,
                "Away_Win_Prob":    1 - p,
                "Confidence":       max(p, 1 - p),
            })

        if not rows:
            return None
        return pd.DataFrame(rows).sort_values("Time").reset_index(drop=True)


def predict_games(date: str = "today") -> pd.DataFrame | None:
    return NHLPredictor().predict_date(date)


if __name__ == "__main__":
    predict_games("today")
