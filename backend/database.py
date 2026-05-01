"""
MongoDB database layer
----------------------
Single source of truth for all NHL Predictor data.

Collections:
  games       — all scraped game logs across all seasons
  schedule    — current season games (with Rslt for completed games)
  predictions — predictions served to users
  results     — actual outcomes vs predictions
  edges       — edge opportunities surfaced
"""

import os
from datetime import datetime
from pymongo import MongoClient, ASCENDING, UpdateOne
from pymongo.errors import ConnectionFailure
import pandas as pd

_client = None
_db     = None

# Stat columns that must be stored as floats, not strings
NUMERIC_COLS = {
    "gf", "ga", "sog", "pim", "ppg", "ppo", "shg",
    "sog_opp", "pim_opp", "ppg_opp", "ppo_opp", "shg_opp",
    "fow", "fol", "fopct", "cf", "ca", "cfpct",
    "ff", "fa", "ffpct", "ozspct", "pdo",
    "season", "gp",
}


def get_db():
    global _client, _db
    if _db is None:
        uri = os.getenv("MONGODB_URI")
        if not uri:
            print("⚠ MONGODB_URI not set")
            return None
        try:
            _client = MongoClient(uri, serverSelectionTimeoutMS=5000, socketTimeoutMS=10000)
            _client.server_info()
            _db = _client["nhl_predictor"]
            _ensure_indexes()
            print("✓ MongoDB connected")
        except ConnectionFailure as e:
            print(f"⚠ MongoDB connection failed: {e}")
            _db = None
    return _db


def _ensure_indexes():
    db = _db
    db["games"].create_index([("team", ASCENDING), ("date", ASCENDING)])
    db["games"].create_index([("date", ASCENDING)])
    db["games"].create_index([("season", ASCENDING)])
    db["schedule"].create_index([("date", ASCENDING), ("team", ASCENDING)])
    db["results"].create_index([("date", ASCENDING)])
    db["predictions"].create_index([("date", ASCENDING)])
    db["edges"].create_index([("date", ASCENDING)])


def _clean_doc(doc: dict) -> dict:
    """
    Normalize a document for MongoDB:
    - Lowercase all keys
    - Replace dots, %, spaces in key names
    - Drop unnamed/junk columns
    - Coerce known numeric columns to float so they never come back as strings
    """
    clean = {}
    for k, v in doc.items():
        new_key = (
            str(k)
            .lower()
            .replace(".", "")
            .replace("%", "pct")
            .replace(" ", "_")
        )
        if "unnamed" in new_key:
            continue

        # coerce numeric columns to float at write time
        if new_key in NUMERIC_COLS and v is not None:
            try:
                v = float(v)
            except (ValueError, TypeError):
                v = None

        clean[new_key] = v
    return clean


# ---------------------------------------------------------------------------
# Game log operations
# ---------------------------------------------------------------------------

def upsert_games(df: pd.DataFrame) -> int:
    db = get_db()
    if db is None or df.empty:
        return 0

    print(f"  Writing {len(df):,} rows to MongoDB games...")
    operations = []

    for _, row in df.iterrows():
        doc = row.where(pd.notna(row), None).to_dict()
        doc = _clean_doc(doc)

        if "date" in doc and doc["date"] is not None:
            doc["date"] = pd.to_datetime(doc["date"])

        operations.append(
            UpdateOne(
                {"team": doc.get("team"), "date": doc.get("date"), "season": doc.get("season")},
                {"$set": doc},
                upsert=True,
            )
        )

    if not operations:
        return 0

    try:
        result = db["games"].bulk_write(operations, ordered=False)
        return result.upserted_count + result.modified_count
    except Exception as e:
        print(f"⚠ Mongo bulk write failed (games): {e}")
        return 0


def upsert_schedule(df: pd.DataFrame) -> int:
    db = get_db()
    if db is None or df.empty:
        return 0

    print(f"  Writing {len(df):,} rows to MongoDB schedule...")
    operations = []

    for _, row in df.iterrows():
        doc = row.where(pd.notna(row), None).to_dict()
        doc = _clean_doc(doc)

        if "date" in doc and doc["date"] is not None:
            doc["date"] = pd.to_datetime(doc["date"])

        operations.append(
            UpdateOne(
                {"team": doc.get("team"), "date": doc.get("date")},
                {"$set": doc},
                upsert=True,
            )
        )

    if not operations:
        return 0

    try:
        result = db["schedule"].bulk_write(operations, ordered=False)
        return result.upserted_count + result.modified_count
    except Exception as e:
        print(f"⚠ Mongo bulk write failed (schedule): {e}")
        return 0


# ---------------------------------------------------------------------------
# Fetching data for training
# ---------------------------------------------------------------------------

def get_training_data() -> pd.DataFrame:
    db = get_db()
    if db is None:
        return pd.DataFrame()

    cursor = db["games"].find({"rslt": {"$in": ["W", "L"]}}, {"_id": 0})
    df = pd.DataFrame(list(cursor))
    if df.empty:
        return df

    rename_map = {
        "date": "Date", "home_away": "Home_Away", "opp": "Opp",
        "rslt": "Rslt", "gf": "GF", "ga": "GA", "ot": "OT",
        "sog": "SOG", "pim": "PIM", "ppg": "PPG", "ppo": "PPO", "shg": "SHG",
        "sog_opp": "SOG_OPP", "pim_opp": "PIM_OPP", "ppg_opp": "PPG_OPP",
        "ppo_opp": "PPO_OPP", "shg_opp": "SHG_OPP",
        "fow": "FOW", "fol": "FOL", "fopct": "FO%",
        "cf": "CF", "ca": "CA", "cfpct": "CF%",
        "ff": "FF", "fa": "FA", "ffpct": "FF%",
        "ozspct": "oZS%", "pdo": "PDO",
        "season": "Season", "team": "Team",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def get_schedule_df() -> pd.DataFrame:
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
        "gp": "GP", "rslt": "Rslt", "gf": "GF", "ga": "GA",
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
                "date": date, "logged_at": datetime.utcnow(),
                "away": p["Away"], "home": p["Home"],
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
        operations = [
            UpdateOne(
                {"date": date, "away": r["Away"], "home": r["Home"]},
                {"$set": {
                    "date": date, "logged_at": datetime.utcnow(),
                    "time": r.get("Time", ""),
                    "away": r["Away"], "home": r["Home"],
                    "away_score": r["Away_Score"], "home_score": r["Home_Score"],
                    "actual_winner": r["Actual_Winner"],
                    "predicted_winner": r["Predicted_Winner"],
                    "home_win_prob": r["Home_Win_Prob"],
                    "away_win_prob": r["Away_Win_Prob"],
                    "correct": r["Correct"], "status": r["Status"],
                }},
                upsert=True,
            )
            for r in results
        ]
        db["results"].bulk_write(operations, ordered=False)
    except Exception as e:
        print(f"⚠ Failed to log results: {e}")


def log_edges(date: str, edges: list) -> None:
    db = get_db()
    if db is None or not edges:
        return
    try:
        operations = [
            UpdateOne(
                {"date": date, "away": e["Away"], "home": e["Home"]},
                {"$set": {
                    "date": date, "logged_at": datetime.utcnow(),
                    "away": e["Away"], "home": e["Home"],
                    "home_odds": e["Home_Odds"], "away_odds": e["Away_Odds"],
                    "home_edge": e["Home_Edge"], "away_edge": e["Away_Edge"],
                    "best_edge": e["Best_Edge"], "best_bet": e["Best_Bet"],
                    "bookmaker": e["Bookmaker"],
                }},
                upsert=True,
            )
            for e in edges
        ]
        db["edges"].bulk_write(operations, ordered=False)
    except Exception as e:
        print(f"⚠ Failed to log edges: {e}")


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def get_season_stats() -> dict | None:
    db = get_db()
    if db is None:
        return None
    try:
        total    = db["results"].count_documents({})
        correct  = db["results"].count_documents({"correct": True})
        accuracy = round(correct / total, 4) if total > 0 else 0.0
        return {"total_predictions": total, "season_accuracy": accuracy}
    except Exception as e:
        print(f"⚠ Failed to get season stats: {e}")
        return None