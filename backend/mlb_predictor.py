"""
MLB Game Predictor — matchup model.

Predicts P(home win) from BOTH teams' offense/staff rolling form AND BOTH
starting pitchers' rolling form. The starting pitcher is the lever in baseball
that team stats can't see; adding it is the main improvement over the old
"one team's stats + opponent code" model. See mlb_matchup.py / mlb_pitchers.py.

At prediction time the probable starters come from
/schedule?hydrate=probablePitcher; if a probable isn't posted yet we fall back
to each team's most recent starter's form. Output schema is unchanged.

Data: MongoDB first, CSV fallback. Model: models_mlb.pkl (joblib).
"""

import os
import warnings

import joblib
import pandas as pd
import requests
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

import mlb_matchup as M
from mlb_pitchers import sp_feature_cols
from mlb_teams import TEAM_ID_TO_ABBREV

warnings.filterwarnings("ignore")

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "models_mlb.pkl")
TRAIN_CSV    = os.path.join(BASE_DIR, "../data/mlb_matches_2021_2025.csv")
SCHEDULE_CSV = os.path.join(BASE_DIR, "../data/mlb_matches_2026.csv")
MLB_API      = "https://statsapi.mlb.com/api/v1"
CURRENT_SEASON = 2026

# Exhaustive search (model_search.py): bagged trees beat boosting here too — the
# old xgb-heavy ensemble was suboptimal. Blend the two winning families.
ENSEMBLE_WEIGHTS = {"rf": 0.5, "et": 0.5}


def _build_models() -> dict:
    return {
        "rf": RandomForestClassifier(
            n_estimators=300, max_depth=7, min_samples_leaf=8,
            max_features="sqrt", random_state=42, n_jobs=-1),
        "et": ExtraTreesClassifier(
            n_estimators=400, max_depth=12, min_samples_leaf=8,
            max_features="sqrt", random_state=42, n_jobs=-1),
    }


def _load_team_data() -> pd.DataFrame:
    if os.getenv("MONGODB_URI"):
        try:
            from mlb_database import get_training_data
            df = get_training_data()
            if not df.empty:
                print(f"  Loaded {len(df):,} team-game rows from MongoDB")
                return df
        except Exception as e:
            print(f"  ⚠ MongoDB load failed: {e} — falling back to CSV")
    print("  Loading team-game data from CSV...")
    df = pd.read_csv(TRAIN_CSV); df["Date"] = pd.to_datetime(df["Date"]); return df


def _load_schedule_data() -> pd.DataFrame:
    if os.getenv("MONGODB_URI"):
        try:
            from mlb_database import get_schedule_df
            df = get_schedule_df()
            if not df.empty:
                return df
        except Exception:
            pass
    df = pd.read_csv(SCHEDULE_CSV); df["Date"] = pd.to_datetime(df["Date"]); return df


# ---------------------------------------------------------------------------
# MLBPredictor
# ---------------------------------------------------------------------------

class MLBPredictor:
    def __init__(self):
        self.models: dict             = {}
        self._feature_cols: list[str] = []
        self._team_feats: list[str]   = []
        self._medians: pd.Series      = pd.Series(dtype=float)
        self._rolled: pd.DataFrame    = pd.DataFrame()
        self._sp_form: pd.DataFrame   = pd.DataFrame()
        self._schedule: pd.DataFrame  = pd.DataFrame()
        self._load_and_train()

    def _should_retrain(self) -> bool:
        return not os.path.exists(MODEL_PATH)

    def _load_and_train(self) -> None:
        team_full = _load_team_data()
        team_full["Season"] = team_full["Season"].apply(M.normalize_season)
        seasons = sorted(team_full["Season"].unique())

        # team offense rolling frame for live lookups (+ current-season games)
        team_all = team_full
        try:
            sched = _load_schedule_data()
            completed = sched.dropna(subset=["Rslt"]).copy() if "Rslt" in sched.columns else pd.DataFrame()
            if not completed.empty:
                team_all = pd.concat([team_full, completed], ignore_index=True)
        except Exception:
            pass
        self._rolled, self._team_feats = M.team_rolling(team_all)
        self._feature_cols = M.feature_cols(self._team_feats)

        # pitcher form frame for live lookups (training seasons + current)
        forms = []
        for y in seasons + [CURRENT_SEASON]:
            try:
                f = M.pitcher_form_cached(y)
                if not f.empty:
                    forms.append(f)
            except Exception:
                pass
        self._sp_form = pd.concat(forms, ignore_index=True) if forms else pd.DataFrame()

        if self._should_retrain():
            print("No saved MLB model — training matchup model from scratch...")
            self._train_and_save(team_full)
        else:
            print("Loading saved model...")
            saved = joblib.load(MODEL_PATH)
            if saved.get("feature_cols") != self._feature_cols:
                print("  ⚠ Saved feature set is stale — retraining...")
                self._train_and_save(team_full)
            else:
                self.models  = saved["models"]
                self._medians = saved["medians"]
                print("✓ Model loaded.\n")

        self._load_schedule()

    def _train_and_save(self, team: pd.DataFrame) -> None:
        game, cols = M.build_matchup(team)
        X, y = game[cols], game["home_win"]
        self._medians = X.median()
        print(f"  Training on {len(X):,} games, {len(cols)} features...")
        for name, model in _build_models().items():
            print(f"    {name}...")
            cal = CalibratedClassifierCV(model, cv=3, method="sigmoid")
            cal.fit(X, y)
            self.models[name] = cal
        joblib.dump({"models": self.models, "feature_cols": self._feature_cols,
                     "team_feats": self._team_feats, "medians": self._medians}, MODEL_PATH)
        print("✓ Model trained and saved.\n")

    def _load_schedule(self) -> None:
        schedule = _load_schedule_data()
        schedule["Date"] = pd.to_datetime(schedule["Date"])
        self._schedule = schedule

    # -- lookups ------------------------------------------------------------

    def _off_latest(self, team: str, before: pd.Timestamp) -> pd.Series | None:
        rows = self._rolled[(self._rolled["Team"] == team) &
                            (self._rolled["Date"] < before)].sort_values("Date")
        return rows.iloc[-1] if not rows.empty else None

    def _sp_by_pid(self, pid, before: pd.Timestamp) -> pd.Series | None:
        if self._sp_form.empty or pid is None:
            return None
        rows = self._sp_form[(self._sp_form["pid"] == pid) &
                             (self._sp_form["date"] < before)].sort_values("date")
        return rows.iloc[-1] if not rows.empty else None

    def _sp_by_team(self, team: str, before: pd.Timestamp) -> pd.Series | None:
        if self._sp_form.empty:
            return None
        rows = self._sp_form[(self._sp_form["team"] == team) &
                             (self._sp_form["date"] < before)].sort_values("date")
        return rows.iloc[-1] if not rows.empty else None

    def _probable_pitchers(self, date: str) -> dict:
        """{(home_abbrev, away_abbrev): (home_pid, away_pid)} from the MLB API."""
        try:
            resp = requests.get(f"{MLB_API}/schedule", params={
                "sportId": 1, "date": date, "gameType": "R",
                "hydrate": "probablePitcher",
            }, timeout=5)
            if resp.status_code != 200:
                return {}
            out = {}
            for d in resp.json().get("dates", []):
                for g in d.get("games", []):
                    home = TEAM_ID_TO_ABBREV.get(g["teams"]["home"]["team"]["id"])
                    away = TEAM_ID_TO_ABBREV.get(g["teams"]["away"]["team"]["id"])
                    hp = g["teams"]["home"].get("probablePitcher") or {}
                    ap = g["teams"]["away"].get("probablePitcher") or {}
                    if home and away:
                        out[(home, away)] = (hp.get("id"), ap.get("id"))
            return out
        except Exception:
            return {}

    def predict_date(self, date_str: str) -> pd.DataFrame | None:
        if date_str.lower() == "today":
            target = pd.Timestamp.now().normalize()
            date_str = target.strftime("%Y-%m-%d")
        else:
            try:
                target = pd.to_datetime(date_str)
            except Exception:
                print(f"Invalid date: {date_str}")
                return None

        games = self._schedule[(self._schedule["Date"] == target) &
                               (self._schedule["Home_Away"] == "Home")].copy()
        if games.empty:
            print(f"No games scheduled for {target.date()}.")
            return None

        probable = self._probable_pitchers(date_str)
        sp_cols  = sp_feature_cols()
        rows = []

        for _, g in games.iterrows():
            home, away = g["Team"], g["Opp"]
            home_off = self._off_latest(home, target)
            away_off = self._off_latest(away, target)
            if home_off is None or away_off is None:
                continue

            hp, ap = probable.get((home, away), (None, None))
            home_sp = self._sp_by_pid(hp, target) if hp else self._sp_by_team(home, target)
            away_sp = self._sp_by_pid(ap, target) if ap else self._sp_by_team(away, target)

            feat = {}
            for f in self._team_feats:
                feat[f"home_{f}"] = home_off[f]
                feat[f"away_{f}"] = away_off[f]
            for c in sp_cols:
                feat[f"home_{c}"] = home_sp[c] if home_sp is not None else None
                feat[f"away_{c}"] = away_sp[c] if away_sp is not None else None
            feat["dow"]       = target.dayofweek
            feat["home_rest"] = min(max((target - home_off["Date"]).days, 0), 7)
            feat["away_rest"] = min(max((target - away_off["Date"]).days, 0), 7)

            X = pd.DataFrame([feat]).reindex(columns=self._feature_cols)
            X = X.fillna(self._medians)
            p = float(sum(m.predict_proba(X)[:, 1][0] * ENSEMBLE_WEIGHTS[n]
                          for n, m in self.models.items()))
            rows.append({
                "Time":             g.get("Time", ""),
                "Away":             away,
                "Home":             home,
                "Predicted_Winner": home if p >= 0.5 else away,
                "Home_Win_Prob":    p,
                "Away_Win_Prob":    1 - p,
                "Confidence":       max(p, 1 - p),
            })

        if not rows:
            return None
        return pd.DataFrame(rows).sort_values("Time").reset_index(drop=True)


def predict_games(date: str = "today") -> pd.DataFrame | None:
    return MLBPredictor().predict_date(date)


if __name__ == "__main__":
    predict_games("today")
