"""
Sport-agnostic guards for the data + model plumbing shared by every sport.

These cover the failure modes that took the deployment down, so a regression
shows up here rather than as a 503 in production. Everything runs offline —
MongoDB is faked with a tiny in-memory double.
"""

import pandas as pd
import pytest

import mlb_matchup as MM
import nhl_matchup as NM
from base_database import BaseDatabase


# ---------------------------------------------------------------------------
# One row per (Team, Date) — duplicates silently corrupt rolling form
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module", [NM, MM], ids=["nhl", "mlb"])
def test_normalize_season_maps_both_formats(module):
    """8-digit API ids and 4-digit labels must collapse to the same season, or
    the same game gets stored twice under two labels (this happened: ~12k
    duplicate rows in the NHL games collection)."""
    assert module.normalize_season(20242025) == module.normalize_season(2025) == 2025
    assert module.normalize_season("20242025") == 2025
    assert module.normalize_season(2025) == 2025


def test_team_rolling_is_distorted_by_duplicate_rows():
    """Documents WHY deduping matters: a duplicated game halves the effective
    span of a rolling window, so form features stop meaning what they say."""
    n = 8
    clean = pd.DataFrame({
        "Team": ["BOS"] * n,
        "Date": pd.date_range("2025-01-01", periods=n, freq="D"),
        "Season": [2025] * n,
        "Opp": ["TOR"] * n, "Home_Away": ["Home"] * n, "Rslt": ["W"] * n,
    })
    # Ramp every rolling stat 1..n so a window's mean is easy to reason about.
    for col in NM.TEAM_ROLL_COLS:
        clean[col] = list(range(1, n + 1))

    doubled = pd.concat([clean, clean], ignore_index=True)

    rolled_clean, feats = NM.team_rolling(clean)
    rolled_dup, _ = NM.team_rolling(doubled)

    # Duplicates reach the feature builder rather than being ignored...
    assert len(rolled_dup) == 2 * len(rolled_clean)

    # ...and they change the feature VALUES: with each game duplicated, a
    # 5-game window covers only ~2.5 distinct games, so the trailing mean lags.
    col = f"avg5_{NM.TEAM_ROLL_COLS[0]}"
    assert col in feats
    last_clean = rolled_clean.sort_values("Date")[col].iloc[-1]
    last_dup = rolled_dup.sort_values("Date")[col].iloc[-1]
    assert last_clean != pytest.approx(last_dup), (
        "duplicated games must visibly distort rolling form — if this ever "
        "passes, the dedupe guard in _load_team_data is no longer load-bearing"
    )


# ---------------------------------------------------------------------------
# Model cache — must never serve a model built for a different feature set
# ---------------------------------------------------------------------------

class _FakeCollection:
    def __init__(self):
        self.docs = {}

    def replace_one(self, flt, doc, upsert=False):
        self.docs[flt["_id"]] = doc

    def find_one(self, flt):
        return self.docs.get(flt["_id"])


class _FakeDB:
    def __init__(self):
        self.colls = {}

    def __getitem__(self, name):
        return self.colls.setdefault(name, _FakeCollection())


@pytest.fixture
def db(monkeypatch):
    base = BaseDatabase(db_name="t", collections={}, numeric_cols=set(),
                        training_rename={}, schedule_rename={})
    fake = _FakeDB()
    monkeypatch.setattr(base, "get_db", lambda: fake)
    return base


def _write(path, payload=b"trained-model-bytes"):
    path.write_bytes(payload)
    return str(path)


def test_model_cache_roundtrip(db, tmp_path):
    src = _write(tmp_path / "m.pkl")
    meta = {"feature_cols": ["a", "b"], "seasons": [2024, 2025]}
    assert db.save_model("m", src, meta) is True

    dest = tmp_path / "restored.pkl"
    assert db.load_model("m", str(dest), meta) is True
    assert dest.read_bytes() == b"trained-model-bytes"


def test_model_cache_refuses_stale_feature_set(db, tmp_path):
    src = _write(tmp_path / "m.pkl")
    db.save_model("m", src, {"feature_cols": ["a"], "seasons": [2025]})

    dest = tmp_path / "restored.pkl"
    # A feature-set change must force a retrain, not silently serve a model
    # whose columns no longer line up with the data.
    assert db.load_model("m", str(dest), {"feature_cols": ["a", "b"], "seasons": [2025]}) is False
    assert not dest.exists()


def test_model_cache_refuses_rolled_season_window(db, tmp_path):
    src = _write(tmp_path / "m.pkl")
    db.save_model("m", src, {"feature_cols": ["a"], "seasons": [2024, 2025]})

    dest = tmp_path / "restored.pkl"
    assert db.load_model("m", str(dest), {"feature_cols": ["a"], "seasons": [2025, 2026]}) is False


def test_model_cache_miss_returns_false(db, tmp_path):
    assert db.load_model("never-saved", str(tmp_path / "x.pkl"), {}) is False


def test_model_cache_rejects_oversized_model(db, tmp_path, monkeypatch):
    """A single BSON document caps at 16 MB — an oversized model must be
    declined cleanly rather than raising mid-training."""
    import base_database
    monkeypatch.setattr(base_database, "MAX_MODEL_BYTES", 10)
    src = _write(tmp_path / "big.pkl", b"x" * 100)
    assert db.save_model("big", src, {}) is False


# ---------------------------------------------------------------------------
# Mongo connection settings — a tight socket timeout caused a 20-minute hang
# ---------------------------------------------------------------------------

def test_socket_timeout_is_generous_enough_for_large_reads():
    """mlb_sp_starts is ~27k documents; a 10s socket timeout made that read
    raise on a throttled dyno, which silently fell back to ~20 minutes of API
    collection and blocked startup."""
    import inspect

    src = inspect.getsource(BaseDatabase.get_db)
    assert "socketTimeoutMS=120000" in src.replace(" ", "")
    assert "serverSelectionTimeoutMS=30000" in src.replace(" ", "")
