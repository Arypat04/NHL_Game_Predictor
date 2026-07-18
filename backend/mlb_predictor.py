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
from sklearn.ensemble import RandomForestClassifier

import mlb_matchup as M
from mlb_pitchers import sp_feature_cols
from mlb_teams import TEAM_ID_TO_ABBREV
from seasons import TRAIN_WINDOW

warnings.filterwarnings("ignore")

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "models_mlb.pkl")
TRAIN_CSV    = os.path.join(BASE_DIR, "../data/mlb_matches_train.csv")
SCHEDULE_CSV = os.path.join(BASE_DIR, "../data/mlb_matches_current.csv")
MLB_API      = "https://statsapi.mlb.com/api/v1"

# Exhaustive search (model_search.py): bagged trees beat boosting here too, and
# RandomForest was the single best family (0.560). As with NHL we deploy one lean
# RF rather than an RF+ExtraTrees blend — the blend was inside the noise band but
# cost ~8x the training, which the ~512 MB web dyno can't afford.
ENSEMBLE_WEIGHTS = {"rf": 1.0}
CALIBRATION_CV = 2


def _build_models() -> dict:
    return {
        "rf": RandomForestClassifier(
            n_estimators=200, max_depth=7, min_samples_leaf=8,
            max_features="sqrt", random_state=42, n_jobs=1),
    }


def _load_team_data() -> pd.DataFrame:
    """Team-game logs trimmed to the rolling training window."""
    df = None
    if os.getenv("MONGODB_URI"):
        try:
            from mlb_database import get_training_data
            loaded = get_training_data()
            if not loaded.empty:
                print(f"  Loaded {len(loaded):,} team-game rows from MongoDB")
                df = loaded
        except Exception as e:
            print(f"  ⚠ MongoDB load failed: {e} — falling back to CSV")
    if df is None:
        print("  Loading team-game data from CSV...")
        df = pd.read_csv(TRAIN_CSV); df["Date"] = pd.to_datetime(df["Date"])
    # data-driven rolling window — newest TRAIN_WINDOW+1 seasons present
    labels = df["Season"].apply(M.normalize_season)
    cur = int(labels.max())
    return df[labels.between(cur - TRAIN_WINDOW, cur)].copy()


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


def _load_sp_form(seasons: list[int]) -> pd.DataFrame:
    """Pitcher rolling-form for the given seasons. Reads raw starts from MongoDB
    (fast — the scraper populates them) and computes rolling form per season, so
    the web app doesn't re-collect thousands of API calls at startup. Falls back
    to the local per-season CSV cache / live API collection when offline."""
    from mlb_pitchers import form_from_starts
    if os.getenv("MONGODB_URI"):
        try:
            from mlb_database import get_sp_starts
            raw = get_sp_starts()
            if not raw.empty and "season" in raw.columns:
                frames = [form_from_starts(raw[raw["season"] == y].copy()) for y in seasons]
                frames = [f for f in frames if not f.empty]
                if frames:
                    print(f"  Loaded pitcher form from MongoDB ({len(raw):,} starts)")
                    return pd.concat(frames, ignore_index=True)
            print("  ⚠ mlb_sp_starts is empty — run weekly_refresh.py to populate it")
        except Exception as e:
            print(f"  ⚠ Mongo pitcher-form load failed: {e}")
        # When MONGODB_URI is set, Mongo is the source of truth. Falling through
        # to API collection here would fire thousands of requests and block
        # startup for ~20 minutes (this is what hung the Render dyno and 503'd
        # MLB). Degrade instead: SP features fall back to medians, app stays up.
        print("  → continuing without pitcher form (SP features use medians)")
        return pd.DataFrame()
    # No MongoDB configured (local/offline dev) — cached CSV or live collection.
    frames = []
    for y in seasons:
        try:
            f = M.pitcher_form_cached(y)
            if not f.empty:
                frames.append(f)
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


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
        self._window: list[int]       = []
        self._load_and_train()

    def _should_retrain(self) -> bool:
        return not os.path.exists(MODEL_PATH)

    def _load_and_train(self) -> None:
        team_full = _load_team_data()
        team_full["Season"] = team_full["Season"].apply(M.normalize_season)
        self._window = sorted(team_full["Season"].unique().tolist())

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

        # pitcher form for live lookups — from MongoDB (fast) with local fallback
        seasons = sorted(int(y) for y in team_all["Season"].apply(M.normalize_season).unique())
        self._sp_form = _load_sp_form(seasons)

        if self._should_retrain():
            print("No saved MLB model — training matchup model from scratch...")
            self._train_and_save(team_full)
        else:
            print("Loading saved model...")
            saved = joblib.load(MODEL_PATH)
            if saved.get("feature_cols") != self._feature_cols:
                print("  ⚠ Saved feature set is stale — retraining...")
                self._train_and_save(team_full)
            elif saved.get("seasons") != self._window:
                print("  ⚠ Season window has rolled over — retraining...")
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
            cal = CalibratedClassifierCV(model, cv=CALIBRATION_CV, method="sigmoid")
            cal.fit(X, y)
            self.models[name] = cal
        joblib.dump({"models": self.models, "feature_cols": self._feature_cols,
                     "team_feats": self._team_feats, "medians": self._medians,
                     "seasons": self._window}, MODEL_PATH)
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
