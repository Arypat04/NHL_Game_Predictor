"""
MongoDB database layer
----------------------
Single source of truth for all NHL Predictor data.

Collections:
  games       — all scraped game logs across all seasons
  schedule    — upcoming games for the current season
  predictions — predictions served to users
  results     — actual outcomes vs predictions
  edges       — edge opportunities surfaced
"""

import os
from datetime import datetime, timedelta
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure
import pandas as pd

_client = None
_db = None


def get_db():
    """Get database connection — creates once and reuses."""
    global _client, _db
    if _db is None:
        uri = os.getenv("MONGODB_URI")
        if not uri:
            print("⚠ MONGODB_URI not set")
            return None
        try:
            _client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            _client.server_info()
            _db = _client["nhl_predictor"]
            _ensure_indexes()
            print("✓ MongoDB connected")
        except ConnectionFailure as e:
            print(f"⚠ MongoDB connection failed: {e}")
            _db = None
    return _db


def _ensure_indexes():
    """Create indexes for fast queries."""
    db = _db
    db["games"].create_index([("team", ASCENDING), ("date", ASCENDING)])
    db["games"].create_index([("date", ASCENDING)])
    db["games"].create_index([("season", ASCENDING)])
    db["schedule"].create_index([("date", ASCENDING), ("team", ASCENDING)])
    db["results"].create_index([("date", ASCENDING)])
    db["predictions"].create_index([("date", ASCENDING)])
    db["edges"].create_index([("date", ASCENDING)])


# ---------------------------------------------------------------------------
# Game log operations
# ---------------------------------------------------------------------------

def upsert_games(df: pd.DataFrame) -> int:
    """
    Insert or update game log rows from a DataFrame.
    Uses team + date + season as unique key.
    Returns number of games upserted.
    """
    db = get_db()
    if db is None:
        return 0

    count = 0
    for _, row in df.iterrows():
        doc = row.where(pd.notna(row), None).to_dict()

        # ensure date is stored as datetime
        if "Date" in doc and doc["Date"] is not None:
            doc["Date"] = pd.to_datetime(doc["Date"])

        # lowercase keys for consistency
        doc = {k.lower(): v for k, v in doc.items()}

        db["games"].update_one(
            {
                "team": doc.get("team"),
                "date": doc.get("date"),
                "season": doc.get("season"),
            },
            {"$set": doc},
            upsert=True,
        )
        count += 1

    return count


def upsert_schedule(df: pd.DataFrame) -> int:
    """
    Insert or update schedule rows.
    Uses team + date as unique key.
    """
    db = get_db()
    if db is None:
        return 0

    count = 0
    for _, row in df.iterrows():
        doc = row.where(pd.notna(row), None).to_dict()

        if "Date" in doc and doc["Date"] is not None:
            doc["Date"] = pd.to_datetime(doc["Date"])

        doc = {k.lower(): v for k, v in doc.items()}

        db["schedule"].update_one(
            {
                "team": doc.get("team"),
                "date": doc.get("date"),
            },
            {"$set": doc},
            upsert=True,
        )
        count += 1

    return count


def get_training_data() -> pd.DataFrame:
    """
    Pull all completed games from MongoDB for model training.
    Returns a DataFrame with the same schema as the old CSV.
    """
    db = get_db()
    if db is None:
        return pd.DataFrame()

    # only completed games (have a result)
    cursor = db["games"].find(
        {"rslt": {"$in": ["W", "L"]}},
        {"_id": 0}
    )

    df = pd.DataFrame(list(cursor))
    if df.empty:
        return df

    # restore original column capitalisation expected by predictor.py
    rename_map = {
        "date": "Date", "home_away": "Home_Away", "opp": "Opp",
        "rslt": "Rslt", "gf": "GF", "ga": "GA", "ot": "OT",
        "sog": "SOG", "pim": "PIM", "ppg": "PPG", "ppo": "PPO",
        "shg": "SHG", "sog_opp": "SOG_OPP", "pim_opp": "PIM_OPP",
        "ppg_opp": "PPG_OPP", "ppo_opp": "PPO_OPP", "shg_opp": "SHG_OPP",
        "fow": "FOW", "fol": "FOL", "fo%": "FO%", "cf": "CF", "ca": "CA",
        "cf%": "CF%", "ff": "FF", "fa": "FA", "ff%": "FF%",
        "ozs%": "oZS%", "pdo": "PDO", "season": "Season", "team": "Team",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df["Date"] = pd.to_datetime(df["Date"])

    return df


def get_schedule_df() -> pd.DataFrame:
    """
    Pull the current season schedule from MongoDB.
    Returns a DataFrame with the same schema as the old schedule CSV.
    """
    db = get_db()
    if db is None:
        return pd.DataFrame()

    cursor = db["schedule"].find({}, {"_id": 0})
    df = pd.DataFrame(list(cursor))
    if df.empty:
        return df

    rename_map = {
        "date": "Date", "time": "Time", "home_away": "Home_Away",
        "opponent": "Opponent", "season": "Season", "team": "Team",
        "gp": "GP",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df["Date"] = pd.to_datetime(df["Date"])

    return df


# ---------------------------------------------------------------------------
# Prediction / result / edge logging
# ---------------------------------------------------------------------------

def log_predictions(date: str, predictions: list) -> None:
    db = get_db()
    if db is None or not predictions:
        return
    try:
        docs = [
            {
                "date": date,
                "logged_at": datetime.utcnow(),
                "away": p["Away"],
                "home": p["Home"],
                "predicted_winner": p["Predicted_Winner"],
                "home_win_prob": p["Home_Win_Prob"],
                "away_win_prob": p["Away_Win_Prob"],
                "confidence": p["Confidence"],
            }
            for p in predictions
        ]
        db["predictions"].insert_many(docs)
    except Exception as e:
        print(f"⚠ Failed to log predictions: {e}")


def log_results(date: str, results: list) -> None:
    db = get_db()
    if db is None or not results:
        return
    try:
        for r in results:
            db["results"].update_one(
                {"date": date, "away": r["Away"], "home": r["Home"]},
                {
                    "$set": {
                        "date": date,
                        "logged_at": datetime.utcnow(),
                        "away": r["Away"],
                        "home": r["Home"],
                        "away_score": r["Away_Score"],
                        "home_score": r["Home_Score"],
                        "actual_winner": r["Actual_Winner"],
                        "predicted_winner": r["Predicted_Winner"],
                        "correct": r["Correct"],
                        "status": r["Status"],
                    }
                },
                upsert=True,
            )
    except Exception as e:
        print(f"⚠ Failed to log results: {e}")


def log_edges(date: str, edges: list) -> None:
    db = get_db()
    if db is None or not edges:
        return
    try:
        for e in edges:
            db["edges"].update_one(
                {"date": date, "away": e["Away"], "home": e["Home"]},
                {
                    "$set": {
                        "date": date,
                        "logged_at": datetime.utcnow(),
                        "away": e["Away"],
                        "home": e["Home"],
                        "home_odds": e["Home_Odds"],
                        "away_odds": e["Away_Odds"],
                        "home_edge": e["Home_Edge"],
                        "away_edge": e["Away_Edge"],
                        "best_edge": e["Best_Edge"],
                        "best_bet": e["Best_Bet"],
                        "bookmaker": e["Bookmaker"],
                    }
                },
                upsert=True,
            )
    except Exception as e:
        print(f"⚠ Failed to log edges: {e}")


def get_season_stats() -> dict:
    """Calculate real season accuracy from stored results."""
    db = get_db()
    if db is None:
        return None
    try:
        total = db["results"].count_documents({})
        correct = db["results"].count_documents({"correct": True})
        accuracy = round(correct / total, 4) if total > 0 else 0.0
        return {
            "total_predictions": total,
            "season_accuracy": accuracy,
        }
    except Exception as e:
        print(f"⚠ Failed to get season stats: {e}")
        return None