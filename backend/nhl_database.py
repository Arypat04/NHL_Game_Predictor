"""
MongoDB database layer for NHL.

Thin configuration over BaseDatabase. All real logic lives in base_database.py;
this file only declares NHL's collection names, numeric columns and rename maps,
then re-exports the operations as module-level functions for backwards
compatibility with existing imports.

Collections:
  games       — all scraped game logs across all seasons
  schedule    — current season games (with Rslt for completed games)
  predictions — predictions served to users
  results     — actual outcomes vs predictions
  edges       — edge opportunities surfaced
"""

from base_database import BaseDatabase

COLLECTIONS = {
    "games":       "games",
    "schedule":    "schedule",
    "results":     "results",
    "predictions": "predictions",
    "edges":       "edges",
}

# Stat columns that must be stored as floats, not strings
NUMERIC_COLS = {
    "gf", "ga", "sog", "pim", "ppg", "ppo", "shg",
    "sog_opp", "pim_opp", "ppg_opp", "ppo_opp", "shg_opp",
    "fow", "fol", "fopct", "cf", "ca", "cfpct",
    "ff", "fa", "ffpct", "ozspct", "pdo",
    "season", "gp",
}

TRAINING_RENAME = {
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

SCHEDULE_RENAME = {
    "date": "Date", "time": "Time", "home_away": "Home_Away",
    "opponent": "Opponent", "season": "Season", "team": "Team",
    "gp": "GP", "rslt": "Rslt", "gf": "GF", "ga": "GA",
}

db = BaseDatabase(
    db_name="nhl_predictor",
    collections=COLLECTIONS,
    numeric_cols=NUMERIC_COLS,
    training_rename=TRAINING_RENAME,
    schedule_rename=SCHEDULE_RENAME,
)

# -- module-level API (preserves existing `from nhl_database import ...`) -----
def get_game_keys() -> set[tuple[str, str]]:
    """Every stored ("YYYY-MM-DD", TEAM) pair in the games collection.

    The weekly refresh runs on an ephemeral CI runner, so nhl_scraper's on-disk
    cache is always empty and it would re-reconstruct the entire season (~2 API
    calls per game) every run. Seeding the skip-list from MongoDB instead makes
    the refresh incremental. Deliberately unfiltered by season: the collection
    holds both 4-digit and 8-digit season formats, and (date, team) is
    unambiguous regardless.
    """
    import pandas as pd

    conn = db.get_db()
    if conn is None:
        return set()
    keys = set()
    for doc in conn[COLLECTIONS["games"]].find({}, {"_id": 0, "date": 1, "team": 1}):
        date, team = doc.get("date"), doc.get("team")
        if date is None or team is None:
            continue
        try:
            keys.add((str(pd.to_datetime(date).date()), str(team)))
        except Exception:
            continue
    return keys


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
