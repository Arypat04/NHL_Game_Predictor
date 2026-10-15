"""
Shared MongoDB layer for all sports.

`BaseDatabase` holds every operation that NHL and MLB had in common
(connection, document cleaning, upserts, prediction/result/edge logging,
season stats). Each sport instantiates one with its own collection names,
numeric columns and rename maps — see nhl_database.py / mlb_database.py.
"""

import os
from datetime import datetime

import pandas as pd
from pymongo import ASCENDING, MongoClient, UpdateOne
from pymongo.errors import ConnectionFailure

# MongoDB Atlas free tier times out on bulk writes larger than ~1000 rows,
# so every bulk operation is written in chunks of this size.
CHUNK_SIZE = 500


class BaseDatabase:
    def __init__(
        self,
        db_name: str,
        collections: dict[str, str],
        numeric_cols: set[str],
        training_rename: dict[str, str],
        schedule_rename: dict[str, str],
        key_replacements: tuple[tuple[str, str], ...] = (),
        connect_label: str = "",
    ):
        self.db_name          = db_name
        self.collections      = collections          # logical name -> collection name
        self.numeric_cols     = numeric_cols
        self.training_rename  = training_rename
        self.schedule_rename  = schedule_rename
        self.key_replacements = key_replacements     # extra key char swaps for _clean_doc
        self.connect_label    = connect_label
        self._client: MongoClient | None = None
        self._db = None

    # -- connection ---------------------------------------------------------

    def get_db(self):
        if self._db is None:
            uri = os.getenv("MONGODB_URI")
            if not uri:
                print("⚠ MONGODB_URI not set")
                return None
            try:
                # Generous timeouts: the web dyno is CPU/network throttled and
                # some reads are big (mlb_sp_starts is ~27k docs). The old 10s
                # socket timeout made that read raise on Render, which silently
                # fell back to re-collecting pitcher data from the API — that
                # blocked startup for ~20 minutes and 503'd every MLB endpoint.
                self._client = MongoClient(
                    uri,
                    serverSelectionTimeoutMS=30000,
                    connectTimeoutMS=30000,
                    socketTimeoutMS=120000,
                )
                self._client.server_info()
                self._db = self._client[self.db_name]
                self._ensure_indexes()
                print(f"✓ MongoDB connected{self.connect_label}")
            except ConnectionFailure as e:
                print(f"⚠ MongoDB connection failed: {e}")
                self._db = None
        return self._db

    def _ensure_indexes(self) -> None:
        db = self._db
        c  = self.collections
        db[c["games"]].create_index([("team", ASCENDING), ("date", ASCENDING)])
        db[c["games"]].create_index([("date", ASCENDING)])
        db[c["games"]].create_index([("season", ASCENDING)])
        db[c["schedule"]].create_index([("date", ASCENDING), ("team", ASCENDING)])
        db[c["results"]].create_index([("date", ASCENDING)])
        db[c["predictions"]].create_index([("date", ASCENDING)])
        db[c["edges"]].create_index([("date", ASCENDING)])

    # -- document cleaning --------------------------------------------------

    def _clean_doc(self, doc: dict) -> dict:
        """
        Normalize a document for MongoDB:
        - Lowercase all keys
        - Replace dots, %, spaces (plus any sport-specific) in key names
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
            for old, new in self.key_replacements:
                new_key = new_key.replace(old, new)

            if "unnamed" in new_key:
                continue

            # coerce numeric columns to float at write time
            if new_key in self.numeric_cols and v is not None:
                try:
                    v = float(v)
                except (ValueError, TypeError):
                    v = None

            clean[new_key] = v
        return clean

    # -- bulk write helper --------------------------------------------------

    def _bulk_write_chunks(self, collection: str, operations: list,
                           chunk_size: int = CHUNK_SIZE) -> int:
        """Write operations in chunks to avoid Atlas socket timeouts."""
        db = self._db
        total = 0
        for i in range(0, len(operations), chunk_size):
            chunk = operations[i: i + chunk_size]
            try:
                result = db[collection].bulk_write(chunk, ordered=False)
                total += result.upserted_count + result.modified_count
            except Exception as e:
                print(f"⚠ Chunk write failed ({collection}, chunk {i // chunk_size + 1}): {e}")
        return total

    def _rows_to_ops(self, df: pd.DataFrame, key_fields: list[str]) -> list:
        operations = []
        for _, row in df.iterrows():
            doc = row.where(pd.notna(row), None).to_dict()
            doc = self._clean_doc(doc)
            if "date" in doc and doc["date"] is not None:
                doc["date"] = pd.to_datetime(doc["date"])
            operations.append(
                UpdateOne(
                    {f: doc.get(f) for f in key_fields},
                    {"$set": doc},
                    upsert=True,
                )
            )
        return operations

    # -- game log operations ------------------------------------------------

    def upsert_games(self, df: pd.DataFrame) -> int:
        db = self.get_db()
        if db is None or df.empty:
            return 0
        coll = self.collections["games"]
        print(f"  Writing {len(df):,} rows to MongoDB {coll}...")
        operations = self._rows_to_ops(df, ["team", "date", "season"])
        if not operations:
            return 0
        n = self._bulk_write_chunks(coll, operations)
        print(f"  ✓ {coll}: {n:,} rows upserted")
        return n

    def upsert_schedule(self, df: pd.DataFrame) -> int:
        db = self.get_db()
        if db is None or df.empty:
            return 0
        coll = self.collections["schedule"]
        print(f"  Writing {len(df):,} rows to MongoDB {coll}...")
        operations = self._rows_to_ops(df, ["team", "date"])
        if not operations:
            return 0
        n = self._bulk_write_chunks(coll, operations)
        print(f"  ✓ {coll}: {n:,} rows upserted")
        return n

    # -- fetching data for training -----------------------------------------

    def get_training_data(self) -> pd.DataFrame:
        db = self.get_db()
        if db is None:
            return pd.DataFrame()
        cursor = db[self.collections["games"]].find(
            {"rslt": {"$in": ["W", "L"]}}, {"_id": 0}
        )
        df = pd.DataFrame(list(cursor))
        if df.empty:
            return df
        df = df.rename(columns={k: v for k, v in self.training_rename.items()
                                if k in df.columns})
        df["Date"] = pd.to_datetime(df["Date"])
        return df

    def get_schedule_df(self) -> pd.DataFrame:
        db = self.get_db()
        if db is None:
            return pd.DataFrame()
        cursor = db[self.collections["schedule"]].find({}, {"_id": 0})
        df = pd.DataFrame(list(cursor))
        if df.empty:
            return df
        df = df.rename(columns={k: v for k, v in self.schedule_rename.items()
                                if k in df.columns})
        df["Date"] = pd.to_datetime(df["Date"])
        return df

    # -- prediction / result / edge logging ---------------------------------

    def log_predictions(self, date: str, predictions: list) -> None:
        db = self.get_db()
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
            db[self.collections["predictions"]].insert_many(docs)
        except Exception as e:
            print(f"⚠ Failed to log predictions: {e}")

    def log_results(self, date: str, results: list) -> None:
        db = self.get_db()
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
            db[self.collections["results"]].bulk_write(operations, ordered=False)
        except Exception as e:
            print(f"⚠ Failed to log results: {e}")

    def log_edges(self, date: str, edges: list) -> None:
        db = self.get_db()
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
            db[self.collections["edges"]].bulk_write(operations, ordered=False)
        except Exception as e:
            print(f"⚠ Failed to log edges: {e}")

    # -- stats --------------------------------------------------------------

    def get_season_stats(self) -> dict | None:
        db = self.get_db()
        if db is None:
            return None
        try:
            coll     = self.collections["results"]
            total    = db[coll].count_documents({})
            correct  = db[coll].count_documents({"correct": True})
            accuracy = round(correct / total, 4) if total > 0 else 0.0
            return {"total_predictions": total, "season_accuracy": accuracy}
        except Exception as e:
            print(f"⚠ Failed to get season stats: {e}")
            return None
