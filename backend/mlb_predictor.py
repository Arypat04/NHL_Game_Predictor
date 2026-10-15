"""
MLB Game Predictor — Elo rating model (clean core, no pitchers).

Predicts P(home win) from team Elo ratings (home-field bump included). This
replaced the RandomForest model (team offense rolling + both starting pitchers'
form): an out-of-sample test showed a plain team Elo already ties that stack —
even though the old model had the starting-pitcher signal and Elo has no features
at all. So we start from a clean team-only Elo and can add a starting-pitcher
rating adjustment back on top later, measuring whether it actually helps.

Like the NHL predictor, this is structurally cheap: Elo is a dict of floats
rebuilt in one pass over completed games, so there's no training, no pickled
model, no MongoDB model cache, and — unlike the old MLB model — no startup
collection of thousands of starting-pitcher API calls. Warmup just loads game
results and the schedule.

Ratings are rebuilt as-of each prediction date (only games strictly before it).
Output schema is unchanged. See elo.py for the engine.

Data: MongoDB first, CSV fallback.
"""

import os
import warnings

import pandas as pd

import elo
from seasons import TRAIN_WINDOW

warnings.filterwarnings("ignore")

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
# Vestigial: Elo has no model file. Kept so main.py's /status field and any
# importers don't break. Nothing reads or writes this path any more.
MODEL_PATH   = os.path.join(BASE_DIR, "models_mlb.pkl")
TRAIN_CSV    = os.path.join(BASE_DIR, "../data/mlb_matches_train.csv")
SCHEDULE_CSV = os.path.join(BASE_DIR, "../data/mlb_matches_current.csv")


def _load_team_data() -> pd.DataFrame:
    """All team-game logs (training seasons + current-season completed games),
    trimmed to the rolling window — the source for building Elo ratings."""
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

    # One row per (Team, Date) — MongoDB accumulated the same game under two
    # season formats, so dedup after normalizing the label.
    df = df.copy()
    df["Season"] = df["Season"].apply(elo.normalize_season)
    before = len(df)
    df = df.sort_values(["Team", "Date"]).drop_duplicates(subset=["Team", "Date"], keep="first")
    if before != len(df):
        print(f"  Dropped {before - len(df):,} duplicate team-game rows")

    cur = int(df["Season"].max())
    return df[df["Season"].between(cur - TRAIN_WINDOW, cur)].copy()


def _load_schedule_data() -> pd.DataFrame:
    if os.getenv("MONGODB_URI"):
        try:
            from mlb_database import get_schedule_df
            df = get_schedule_df()
            if not df.empty:
                return df
        except Exception:
            pass
    df = pd.read_csv(SCHEDULE_CSV)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


# ---------------------------------------------------------------------------
# MLBPredictor
# ---------------------------------------------------------------------------

class MLBPredictor:
    def __init__(self):
        self._results: pd.DataFrame  = pd.DataFrame()
        self._schedule: pd.DataFrame = pd.DataFrame()
        self._load()

    def _load(self) -> None:
        team = _load_team_data()
        self._results = elo.results_frame(team)
        print(f"✓ Elo ready - {len(self._results):,} completed games "
              f"(seasons {self._results['Season'].min()}-{self._results['Season'].max()}).\n")
        self._load_schedule()

    def _load_schedule(self) -> None:
        schedule = _load_schedule_data()
        schedule["Date"] = pd.to_datetime(schedule["Date"])
        self._schedule = schedule

    # -- prediction ---------------------------------------------------------

    def predict_date(self, date_str: str) -> pd.DataFrame | None:
        if date_str.lower() == "today":
            target = pd.Timestamp.now().normalize()
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

        # ratings from games strictly before the target date (no leakage)
        model = elo.Elo.for_sport("mlb").fit(self._results[self._results["Date"] < target])

        rows = []
        for _, g in games.iterrows():
            home, away = g["Team"], g["Opp"]
            if not home or not away:
                continue
            p = model.prob_home(home, away)
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
