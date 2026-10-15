"""
Sport-agnostic guards for the data plumbing shared by every sport.

These cover the failure modes that took the deployment down, so a regression
shows up here rather than as a 503 in production. Everything runs offline.
"""

import pandas as pd
import pytest

import elo
from base_database import BaseDatabase


# ---------------------------------------------------------------------------
# Season labels — 8-digit API ids and 4-digit labels must collapse together
# ---------------------------------------------------------------------------

def test_normalize_season_maps_both_formats():
    """A game can arrive as an 8-digit API id (20242025) or a 4-digit label
    (2025); if these don't collapse, the same game is stored twice under two
    labels (this happened: ~12k duplicate rows in the NHL games collection)."""
    assert elo.normalize_season(20242025) == elo.normalize_season(2025) == 2025
    assert elo.normalize_season("20242025") == 2025
    assert elo.normalize_season(2025) == 2025


# ---------------------------------------------------------------------------
# One row per (Team, Date) — duplicates distort the ratings
# ---------------------------------------------------------------------------

def test_duplicate_games_distort_elo():
    """Documents WHY deduping matters: Elo moves a team's rating once per game,
    so a game counted twice moves it twice. If this ever stops distorting, the
    dedupe guard in _load_team_data is no longer load-bearing."""
    n = 10
    clean = pd.DataFrame({
        "Date": pd.date_range("2025-01-01", periods=n, freq="D"),
        "Season": [2025] * n,
        "home": ["BOS"] * n, "away": ["TOR"] * n, "home_win": [1] * n,
    })
    doubled = pd.concat([clean, clean], ignore_index=True)

    r_clean = elo.Elo.for_sport("nhl").fit(clean).rating["BOS"]
    r_dup = elo.Elo.for_sport("nhl").fit(doubled).rating["BOS"]
    assert r_clean != pytest.approx(r_dup)


def test_results_frame_one_row_per_game_drops_unplayed():
    """results_frame takes the home-side rows (one per game) and drops games
    with no result, so Elo never trains on an unplayed game."""
    team = pd.DataFrame({
        "Date": ["2025-01-01", "2025-01-01", "2025-01-02"],
        "Season": [2025, 2025, 2025],
        "Team": ["BOS", "TOR", "BOS"],
        "Opp": ["TOR", "BOS", "MTL"],
        "Home_Away": ["Home", "Away", "Home"],
        "Rslt": ["W", "L", None],          # 3rd game not yet played
    })
    rf = elo.results_frame(team)
    assert len(rf) == 1
    row = rf.iloc[0]
    assert row["home"] == "BOS" and row["away"] == "TOR" and row["home_win"] == 1


# ---------------------------------------------------------------------------
# Mongo connection settings — a tight socket timeout caused a 20-minute hang
# ---------------------------------------------------------------------------

def test_socket_timeout_is_generous_enough_for_large_reads():
    """The web dyno is throttled and some reads are large; a 10s socket timeout
    made a big read raise on Render, which cascaded into a ~20-minute startup
    hang. Keep the generous timeouts."""
    import inspect

    src = inspect.getsource(BaseDatabase.get_db)
    assert "socketTimeoutMS=120000" in src.replace(" ", "")
    assert "serverSelectionTimeoutMS=30000" in src.replace(" ", "")
