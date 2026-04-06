"""
NHL Game Predictor
------------------
Trains an ensemble model on game logs pulled from MongoDB and predicts
outcomes for any date in the current season schedule.

Model persistence via joblib:
  - First run: trains all models, saves to models.pkl
  - Subsequent runs: loads models.pkl instantly (~2-3 seconds)
  - Retrains automatically when triggered (no CSV timestamp — use DELETE models.pkl)
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models.pkl")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEAM_NAME_TO_ABBREV: dict[str, str] = {
    "Anaheim Ducks": "ANA",
    "Arizona Coyotes": "ARI",
    "Boston Bruins": "BOS",
    "Buffalo Sabres": "BUF",
    "Calgary Flames": "CGY",
    "Carolina Hurricanes": "CAR",
    "Chicago Blackhawks": "CHI",
    "Colorado Avalanche": "COL",
    "Columbus Blue Jackets": "CBJ",
    "Dallas Stars": "DAL",
    "Detroit Red Wings": "DET",
    "Edmonton Oilers": "EDM",
    "Florida Panthers": "FLA",
    "Los Angeles Kings": "LAK",
    "Minnesota Wild": "MIN",
    "Montreal Canadiens": "MTL",
    "Montréal Canadiens": "MTL",
    "Nashville Predators": "NSH",
    "New Jersey Devils": "NJD",
    "New York Islanders": "NYI",
    "New York Rangers": "NYR",
    "Ottawa Senators": "OTT",
    "Philadelphia Flyers": "PHI",
    "Pittsburgh Penguins": "PIT",
    "San Jose Sharks": "SJS",
    "Seattle Kraken": "SEA",
    "St. Louis Blues": "STL",
    "St Louis Blues": "STL",
    "Tampa Bay Lightning": "TBL",
    "Toronto Maple Leafs": "TOR",
    "Utah Hockey Club": "UTA",
    "Utah Mammoth": "UTA",
    "Vancouver Canucks": "VAN",
    "Vegas Golden Knights": "VGK",
    "Washington Capitals": "WSH",
    "Winnipeg Jets": "WPG",
}

ROLLING_STAT_COLS = [
    "GF", "GA", "SOG", "PIM", "PPG", "PPO", "SHG",
    "SOG_OPP", "PIM_OPP", "PPG_OPP", "PPO_OPP", "SHG_OPP",
    "FOW", "FOL", "FO%", "CF", "CA", "CF%", "FF", "FA", "FF%",
    "oZS%", "PDO",
]

ROLLING_WINDOWS = [3, 5, 10]

BASE_PREDICTORS = [
    "Team_code", "Opponent_code", "Arena_code",
    "Day_code", "OT_code",
    "days_rest", "back_to_back", "well_rested",
]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _encode_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Arena_code"] = df["Home_Away"].astype("category").cat.codes
    df["Opponent_code"] = df["Opp"].astype("category").cat.codes
    df["Day_code"] = df["Date"].dt.dayofweek
    df["Team_code"] = df["Team"].astype("category").cat.codes
    if "OT" in df.columns:
        df["OT_code"] = df["OT"].astype("category").cat.codes + 1
    else:
        df["OT_code"] = 0
    df = df.sort_values(["Team", "Date"]).reset_index(drop=True)
    df["days_rest"] = df.groupby("Team")["Date"].diff().dt.days.fillna(2)
    df["back_to_back"] = (df["days_rest"] == 1).astype(int)
    df["well_rested"] = (df["days_rest"] >= 3).astype(int)
    return df


def _encode_target(df: pd.DataFrame) -> pd.DataFrame:
    if "Rslt" not in df.columns:
        return df
    mapping = {"W": 1, "L": 0}
    df["target"] = df["Rslt"].map(mapping)
    return df.dropna(subset=["target"]).astype({"target": int})


def _rolling_averages(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    available_cols = [c for c in ROLLING_STAT_COLS if c in df.columns]
    new_col_names: list[str] = []

    global_medians = df[available_cols].median()
    df[available_cols] = (
        df.groupby("Team")[available_cols]
        .transform(lambda s: s.fillna(s.median()))
        .fillna(global_medians)
    )

    all_rolled = []
    for team, group in df.groupby("Team"):
        group = group.sort_values("Date").copy()
        for window in ROLLING_WINDOWS:
            rolled = group[available_cols].rolling(window, closed="left").mean()
            for col in available_cols:
                group[f"avg{window}_{col}"] = rolled[col]
        all_rolled.append(group)

    df_rolled = pd.concat(all_rolled, ignore_index=True)

    for window in ROLLING_WINDOWS:
        new_col_names.extend([f"avg{window}_{c}" for c in available_cols])

    df_rolled = df_rolled.dropna(subset=[f"avg{ROLLING_WINDOWS[0]}_{available_cols[0]}"])

    for col in available_cols:
        for i in range(1, len(ROLLING_WINDOWS)):
            big_col = f"avg{ROLLING_WINDOWS[i]}_{col}"
            small_col = f"avg{ROLLING_WINDOWS[i-1]}_{col}"
            df_rolled[big_col] = df_rolled[big_col].fillna(df_rolled[small_col])

    return df_rolled, new_col_names


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = _encode_features(df)
    df = _encode_target(df)
    return df


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _build_models() -> dict:
    rf = RandomForestClassifier(
        n_estimators=726, max_depth=8, min_samples_split=7,
        min_samples_leaf=1, bootstrap=True, max_features=None,
        random_state=1,
    )
    gb = GradientBoostingClassifier(
        n_estimators=331, learning_rate=0.01316, max_depth=3,
        min_samples_leaf=2, min_samples_split=7, subsample=0.7210,
        random_state=1,
    )
    xgb = XGBClassifier(
        n_estimators=177, learning_rate=0.03147, max_depth=3,
        subsample=0.6927, colsample_bytree=0.6689, gamma=0.3963,
        reg_lambda=1.0774, reg_alpha=0.1021,
        eval_metric="logloss", random_state=1, verbosity=0,
    )
    return {"rf": rf, "gb": gb, "xgb": xgb}


ENSEMBLE_WEIGHTS = {"rf": 0.2, "gb": 0.4, "xgb": 0.4}


# ---------------------------------------------------------------------------
# NHLPredictor
# ---------------------------------------------------------------------------

class NHLPredictor:
    """
    Train on game logs from MongoDB and predict future game outcomes.

    To retrain: delete models.pkl and restart the server.
    Model retrains from MongoDB data automatically.
    """

    def __init__(self):
        self.models: dict = {}
        self.predictors: list[str] = []
        self._team_code_map: dict[str, int] = {}
        self._rolling_data: pd.DataFrame = pd.DataFrame()
        self._schedule: pd.DataFrame = pd.DataFrame()
        self._load_and_train()

    def _should_retrain(self) -> bool:
        return not os.path.exists(MODEL_PATH)

    def _load_and_train(self) -> None:
        if self._should_retrain():
            print("No saved model found — training from MongoDB...")
            self._train_and_save()
        else:
            print("Loading saved model...")
            self._load_saved()
            print("✓ Model loaded.\n")
        self._load_schedule()

    def _train_and_save(self) -> None:
        from database import get_training_data
        print("  Loading training data from MongoDB...")
        train_raw = get_training_data()

        if train_raw.empty:
            raise RuntimeError(
                "No training data found in MongoDB. "
                "Run scraper.py first to populate the database."
            )

        print(f"  Loaded {len(train_raw):,} game rows")
        train_prepared = prepare(train_raw)

        print("  Calculating rolling averages...")
        train_rolled, rolling_cols = _rolling_averages(train_prepared)
        self._rolling_data = train_rolled

        self._team_code_map = (
            train_rolled[["Team", "Team_code"]]
            .drop_duplicates()
            .set_index("Team")["Team_code"]
            .to_dict()
        )

        self.predictors = BASE_PREDICTORS + rolling_cols

        X = train_rolled[self.predictors]
        y = train_rolled["target"]

        raw_models = _build_models()
        for name, model in raw_models.items():
            print(f"  Training {name}...")
            calibrated = CalibratedClassifierCV(model, cv=3, method="sigmoid")
            calibrated.fit(X, y)
            self.models[name] = calibrated

        print("  Saving model to disk...")
        joblib.dump({
            "models": self.models,
            "predictors": self.predictors,
            "team_code_map": self._team_code_map,
            "rolling_data": self._rolling_data,
        }, MODEL_PATH)
        print("✓ Model trained and saved.\n")

    def _load_saved(self) -> None:
        saved = joblib.load(MODEL_PATH)
        self.models = saved["models"]
        self.predictors = saved["predictors"]
        self._team_code_map = saved["team_code_map"]
        self._rolling_data = saved["rolling_data"]

    def _load_schedule(self) -> None:
        from database import get_schedule_df
        schedule = get_schedule_df()

        if schedule.empty:
            print("⚠ No schedule data in MongoDB")
            self._schedule = pd.DataFrame()
            return

        schedule["Date"] = pd.to_datetime(schedule["Date"])
        schedule["Team"] = schedule["Team"].replace("VEG", "VGK")

        if "Opponent" in schedule.columns:
            schedule["Opponent"] = schedule["Opponent"].str.replace("é", "e").str.replace("É", "E")
            schedule["Opp_abbrev"] = schedule["Opponent"].map(TEAM_NAME_TO_ABBREV)
            missing = schedule["Opp_abbrev"].isna().sum()
            if missing:
                unknown = schedule.loc[schedule["Opp_abbrev"].isna(), "Opponent"].unique()
                print(f"  ⚠ {missing} schedule rows unmapped: {unknown}")

        self._schedule = schedule

    def _latest_stats(self, team: str, before_date: pd.Timestamp) -> pd.Series | None:
        rows = self._rolling_data[
            (self._rolling_data["Team"] == team)
            & (self._rolling_data["Date"] < before_date)
        ].sort_values("Date")
        if rows.empty:
            return None
        return rows.iloc[-1]

    def _team_code(self, team: str) -> int:
        return self._team_code_map.get(team, 0)

    def _ensemble_prob(self, feature_row: pd.DataFrame) -> float:
        prob = 0.0
        for name, model in self.models.items():
            prob += model.predict_proba(feature_row)[:, 1][0] * ENSEMBLE_WEIGHTS[name]
        return float(prob)

    def _build_feature_row(self, team_stats, opponent_abbrev, home_away, game_date, rolling_cols):
        row: dict = {
            "Team_code": [team_stats["Team_code"]],
            "Opponent_code": [self._team_code(opponent_abbrev)],
            "Arena_code": [1 if home_away == "Home" else 0],
            "Day_code": [game_date.dayofweek],
            "OT_code": [0],
            "days_rest": [team_stats["days_rest"]],
            "back_to_back": [team_stats["back_to_back"]],
            "well_rested": [team_stats["well_rested"]],
        }
        for col in rolling_cols:
            row[col] = [team_stats.get(col, np.nan)]
        return pd.DataFrame(row)[self.predictors]

    def predict_date(self, date_str: str) -> pd.DataFrame | None:
        if date_str.lower() == "today":
            target_date = pd.Timestamp.now().normalize()
        else:
            try:
                target_date = pd.to_datetime(date_str)
            except Exception:
                print(f"Invalid date format '{date_str}'.")
                return None

        if self._schedule.empty:
            return None

        games_today = self._schedule[self._schedule["Date"] == target_date].copy()
        if games_today.empty:
            return None

        rolling_cols = [c for c in self.predictors if c not in BASE_PREDICTORS]
        predictions = []

        for _, game in games_today.iterrows():
            team = game["Team"]
            opp_abbrev = game.get("Opp_abbrev") or TEAM_NAME_TO_ABBREV.get(game.get("Opponent", ""))

            if not opp_abbrev:
                continue

            team_stats = self._latest_stats(team, target_date)
            if team_stats is None:
                continue

            features = self._build_feature_row(
                team_stats, opp_abbrev, game["Home_Away"], target_date, rolling_cols
            )
            win_prob = self._ensemble_prob(features)

            if game["Home_Away"] == "Home":
                home_team, away_team = team, opp_abbrev
                home_prob, away_prob = win_prob, 1 - win_prob
            else:
                home_team, away_team = opp_abbrev, team
                home_prob, away_prob = 1 - win_prob, win_prob

            predictions.append({
                "Time": game.get("Time", ""),
                "Away": away_team,
                "Home": home_team,
                "Predicted_Winner": home_team if home_prob > 0.5 else away_team,
                "Home_Win_Prob": home_prob,
                "Away_Win_Prob": away_prob,
                "Confidence": max(home_prob, away_prob),
                "_matchup_key": "_".join(sorted([home_team, away_team])),
            })

        if not predictions:
            return None

        df = pd.DataFrame(predictions)
        df = (
            df.sort_values("Confidence", ascending=False)
            .drop_duplicates("_matchup_key")
            .drop(columns=["_matchup_key"])
            .sort_values("Time")
            .reset_index(drop=True)
        )
        return df


def predict_games(date: str = "today") -> pd.DataFrame | None:
    predictor = NHLPredictor()
    return predictor.predict_date(date)


if __name__ == "__main__":
    predict_games("today")