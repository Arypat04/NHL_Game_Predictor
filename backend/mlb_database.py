"""
MongoDB database layer for MLB.

Thin configuration over BaseDatabase. All real logic lives in base_database.py;
this file only declares MLB's collection names, numeric columns and rename maps,
then re-exports the operations as module-level functions for backwards
compatibility with existing imports.

Collections:
  mlb_games       — all scraped game logs across all seasons
  mlb_schedule    — current season games (with Rslt for completed games)
  mlb_predictions — predictions served to users
  mlb_results     — actual outcomes vs predictions
  mlb_edges       — edge opportunities surfaced
"""

import pandas as pd
from pymongo import UpdateOne

from base_database import BaseDatabase

COLLECTIONS = {
    "games":       "mlb_games",
    "schedule":    "mlb_schedule",
    "results":     "mlb_results",
    "predictions": "mlb_predictions",
    "edges":       "mlb_edges",
}

# Raw per-start starting-pitcher rows (the web app reads these instead of
# re-collecting from the API on startup; the scraper/refresh writes them).
SP_STARTS_COLLECTION = "mlb_sp_starts"

# Stat columns that must be stored as floats, not strings
NUMERIC_COLS = {
    # Core game stats
    "r", "ra", "rd", "gtm",
    # Hitting
    "h", "hr", "bb", "so", "sb", "lob", "rbi",
    "2b", "3b", "obp", "slg", "ops", "avg", "babip",
    # Pitching
    "p_era", "p_whip", "p_ip", "p_k", "p_bb", "p_hr",
    "p_k9", "p_bb9", "p_hr9", "p_h9", "p_k_bb", "p_pc",
    # Context
    "season",
}

TRAINING_RENAME = {
    # Core
    "date": "Date", "home_away": "Home_Away", "opp": "Opp",
    "rslt": "Rslt", "r": "R", "ra": "RA", "rd": "RD",
    "season": "Season", "team": "Team", "time": "Time",
    # Hitting
    "h": "H", "hr": "HR", "bb": "BB", "so": "SO",
    "sb": "SB", "lob": "LOB", "rbi": "RBI",
    "2b": "2B", "3b": "3B",
    "obp": "OBP", "slg": "SLG", "ops": "OPS",
    "avg": "AVG", "babip": "BABIP",
    # Pitching
    "p_era": "P_ERA", "p_whip": "P_WHIP",
    "p_ip": "P_IP", "p_k": "P_K", "p_bb": "P_BB", "p_hr": "P_HR",
    "p_k9": "P_K9", "p_bb9": "P_BB9", "p_hr9": "P_HR9",
    "p_h9": "P_H9", "p_k_bb": "P_K_BB", "p_pc": "P_PC",
}

SCHEDULE_RENAME = {
    # Core
    "date": "Date", "time": "Time", "home_away": "Home_Away",
    "opp": "Opp", "season": "Season", "team": "Team",
    "rslt": "Rslt", "r": "R", "ra": "RA", "rd": "RD",
    # Hitting
    "h": "H", "hr": "HR", "bb": "BB", "so": "SO",
    "sb": "SB", "lob": "LOB", "rbi": "RBI",
    "2b": "2B", "3b": "3B",
    "obp": "OBP", "slg": "SLG", "ops": "OPS",
    "avg": "AVG", "babip": "BABIP",
    # Pitching
    "p_era": "P_ERA", "p_whip": "P_WHIP",
    "p_ip": "P_IP", "p_k": "P_K", "p_bb": "P_BB", "p_hr": "P_HR",
    "p_k9": "P_K9", "p_bb9": "P_BB9", "p_hr9": "P_HR9",
    "p_h9": "P_H9", "p_k_bb": "P_K_BB", "p_pc": "P_PC",
}

db = BaseDatabase(
    db_name="mlb_predictor",
    collections=COLLECTIONS,
    numeric_cols=NUMERIC_COLS,
    training_rename=TRAINING_RENAME,
    schedule_rename=SCHEDULE_RENAME,
    # MLB stat keys can contain "/" (P_K/BB) and "-" — normalize to "_"
    key_replacements=(("/", "_"), ("-", "_")),
    connect_label=" (MLB)",
)

def upsert_sp_starts(df: pd.DataFrame) -> int:
    """Upsert raw starting-pitcher start rows (keyed by pitcher + game)."""
    conn = db.get_db()
    if conn is None or df.empty:
        return 0
    ops = []
    for _, row in df.iterrows():
        doc = {k: (None if pd.isna(v) else v) for k, v in row.items()}
        if doc.get("date") is not None:
            doc["date"] = pd.to_datetime(doc["date"])
        ops.append(UpdateOne({"pid": doc.get("pid"), "gamePk": doc.get("gamePk")},
                             {"$set": doc}, upsert=True))
    return db._bulk_write_chunks(SP_STARTS_COLLECTION, ops)


def sp_start_counts() -> dict[int, int]:
    """{season: rows stored} for mlb_sp_starts — lets the weekly refresh skip
    completed seasons, whose starts are immutable once the season ends."""
    conn = db.get_db()
    if conn is None:
        return {}
    out = {}
    for season in conn[SP_STARTS_COLLECTION].distinct("season"):
        try:
            out[int(season)] = conn[SP_STARTS_COLLECTION].count_documents({"season": season})
        except (TypeError, ValueError):
            continue
    return out


def get_sp_starts() -> pd.DataFrame:
    """All raw starting-pitcher start rows from MongoDB."""
    conn = db.get_db()
    if conn is None:
        return pd.DataFrame()
    rows = pd.DataFrame(list(conn[SP_STARTS_COLLECTION].find({}, {"_id": 0})))
    if not rows.empty and "date" in rows.columns:
        rows["date"] = pd.to_datetime(rows["date"])
    return rows


# -- module-level API (preserves existing `from mlb_database import ...`) -----
get_db            = db.get_db
upsert_games      = db.upsert_games
upsert_schedule   = db.upsert_schedule
get_training_data = db.get_training_data
get_schedule_df   = db.get_schedule_df
log_predictions   = db.log_predictions
log_results       = db.log_results
log_edges         = db.log_edges
get_season_stats  = db.get_season_stats
save_model        = db.save_model
load_model        = db.load_model
_clean_doc        = db._clean_doc
