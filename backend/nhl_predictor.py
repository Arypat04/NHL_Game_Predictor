"""
NHL Game Predictor — Elo rating model.

Predicts P(home win) from team Elo ratings (home-ice bump included). This
replaced the RandomForest-on-rolling-features model: an out-of-sample test
showed a plain Elo beats that stack on accuracy, AUC and logloss — a rating
model that knows only who-beat-whom captures team strength more efficiently than
the engineered Corsi/Fenwick/PDO features did.

The win for the free tier is structural: Elo is a dict of floats rebuilt in one
cheap pass over completed games, so there is NO model training, no pickled model,
and no MongoDB model cache — the things that caused the 512 MB OOM and the slow
cold-start retrain. Warmup just loads game results and the schedule.

Ratings are rebuilt as-of each prediction date (only games strictly before it),
so a prediction never sees its own result. Output schema is unchanged
(Time/Away/Home/Predicted_Winner/probs/Confidence), so main.py, /results,
/edges and the frontend are unaffected. See elo.py for the engine.

Data: MongoDB first, CSV fallback.
"""

import os
import warnings

import pandas as pd

import elo
from nhl_teams import TEAM_NAME_TO_ABBREV  # re-exported for scraper/main imports
from seasons import TRAIN_WINDOW

warnings.filterwarnings("ignore")

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
# Vestigial: Elo has no model file. Kept so main.py's /status field and any
# importers don't break. Nothing reads or writes this path any more.
MODEL_PATH   = os.path.join(BASE_DIR, "nhl_model.pkl")
TRAIN_CSV    = os.path.join(BASE_DIR, "../data/nhl_matches_train.csv")
SCHEDULE_CSV = os.path.join(BASE_DIR, "../data/nhl_matches_current.csv")


# ---------------------------------------------------------------------------
# Data loading (MongoDB first, CSV fallback)
# ---------------------------------------------------------------------------

def _load_team_data() -> pd.DataFrame:
    """All team-game logs (training seasons + current-season completed games),
    trimmed to the rolling window — the source for building Elo ratings."""
    df = None
    if os.getenv("MONGODB_URI"):
        try:
            from nhl_database import get_training_data
            loaded = get_training_data()   # games collection incl. current season
            if not loaded.empty:
                print(f"  Loaded {len(loaded):,} team-game rows from MongoDB")
                df = loaded
        except Exception as e:
            print(f"  ⚠ MongoDB load failed: {e} — falling back to CSV")
    if df is None:
        print("  Loading team-game data from CSV...")
        df = pd.read_csv(TRAIN_CSV)
        df["Date"] = pd.to_datetime(df["Date"])
        # include current-season completed games for up-to-date ratings
        try:
            cur = pd.read_csv(SCHEDULE_CSV)
            cur["Date"] = pd.to_datetime(cur["Date"])
            cur = cur.dropna(subset=["Rslt"])
            if not cur.empty:
                df = pd.concat([df, cur], ignore_index=True)
        except Exception:
            pass

    # Normalize the season label FIRST: MongoDB accumulated the same games under
    # two formats (4-digit label 2025 and 8-digit API id 20242025), so without
    # this each game appears twice. One row per (Team, Date) is the invariant.
    df = df.copy()
    df["Season"] = df["Season"].apply(elo.normalize_season)
    before = len(df)
    df = df.sort_values(["Team", "Date"]).drop_duplicates(subset=["Team", "Date"], keep="first")
    if before != len(df):
        print(f"  Dropped {before - len(df):,} duplicate team-game rows")

    # keep the newest TRAIN_WINDOW+1 seasons present — plenty of history for Elo,
    # and the oldest season drops out automatically as the calendar advances
    cur = int(df["Season"].max())
    return df[df["Season"].between(cur - TRAIN_WINDOW, cur)].copy()


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
        self._results: pd.DataFrame  = pd.DataFrame()   # completed games (home view)
        self._schedule: pd.DataFrame = pd.DataFrame()
        self._load()

    def _load(self) -> None:
        team = _load_team_data()
        self._results = elo.results_frame(team, score_cols=("GF", "GA"))
        print(f"✓ Elo ready - {len(self._results):,} completed games "
              f"(seasons {self._results['Season'].min()}-{self._results['Season'].max()}).\n")
        self._load_schedule()

    def _load_schedule(self) -> None:
        schedule = _load_schedule_data()
        schedule["Date"] = pd.to_datetime(schedule["Date"])
        opp_col = "Opponent" if "Opponent" in schedule.columns else "Opp"
        schedule["Opp_abbrev"] = schedule[opp_col].map(TEAM_NAME_TO_ABBREV)
        self._schedule = schedule

    # -- prediction ---------------------------------------------------------

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

        # ratings from games strictly before the target date (no leakage)
        model = elo.Elo.for_sport("nhl").fit(self._results[self._results["Date"] < target])

        rows = []
        for _, g in games.iterrows():
            home = g["Team"]
            away = g.get("Opp_abbrev") or TEAM_NAME_TO_ABBREV.get(g.get("Opp", ""))
            if not away:
                continue
            p = model.prob_home(home, away)
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
