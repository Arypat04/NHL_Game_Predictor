"""
Shared pytest configuration.

Adds backend/ to the import path and forces offline mode (no MONGODB_URI) so the
predictors load their saved .pkl and fall back to CSV — tests are deterministic
and never touch MongoDB or the network.
"""

import os
import sys

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

# Force offline before any predictor/database import happens.
os.environ.pop("MONGODB_URI", None)

# Public columns every predict_date() result must expose (and nothing internal).
PUBLIC_PREDICTION_COLS = [
    "Time", "Away", "Home", "Predicted_Winner",
    "Home_Win_Prob", "Away_Win_Prob", "Confidence",
]


def first_date_with_games(predictor, schedule_csv: str) -> str:
    """Return the first scheduled date (YYYY-MM-DD) that yields predictions."""
    import pandas as pd
    df = pd.read_csv(schedule_csv)
    df["Date"] = pd.to_datetime(df["Date"])
    for d in sorted(df["Date"].dt.strftime("%Y-%m-%d").unique()):
        if predictor.predict_date(d) is not None:
            return d
    raise AssertionError(f"No date with predictions found in {schedule_csv}")
