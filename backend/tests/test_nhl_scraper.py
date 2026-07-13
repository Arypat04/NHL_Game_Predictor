"""
Tests for nhl_scraper (official NHL API).

Two layers:
  - Pure-function unit tests (offline, always run).
  - A live reconstruction test that hits the NHL API and checks the output
    against Hockey-Reference's recorded values for a known game. It skips
    automatically when the API is unreachable, so the default suite stays green
    offline.
"""

import socket

import pandas as pd
import pytest

import nhl_scraper as s
from nhl_predictor import TRAIN_CSV


# -- offline pure-function tests ---------------------------------------------

def test_season_id():
    assert s.season_id(2024) == "20232024"
    assert s.season_id(2021) == "20202021"


def test_parse_pp():
    assert s._parse_pp("2/5") == (2, 5)
    assert s._parse_pp("0/0") == (0, 0)
    assert s._parse_pp(None) == (0, 0)


def test_as_int():
    assert s._as_int("31") == 31
    assert s._as_int(34) == 34
    assert s._as_int("nope") == 0


def test_norm_relocations():
    assert s.norm("ARI") == "UTA"
    assert s.norm("VEG") == "VGK"
    assert s.norm("BOS") == "BOS"


def test_to_eastern_october_is_edt():
    # 23:30 UTC in October → EDT (UTC-4) → 7:30 PM
    assert s.to_eastern("2023-10-13T23:30:00Z") == "7:30 PM"
    assert s.to_eastern("") == ""


# -- live reconstruction vs Hockey-Reference ---------------------------------

def _api_online() -> bool:
    try:
        socket.create_connection(("api-web.nhle.com", 443), timeout=5).close()
        return True
    except OSError:
        return False


requires_api = pytest.mark.skipif(not _api_online(), reason="NHL API unreachable")

# NYR @ WSH on 2024-01-13 — validated by hand against Hockey-Reference.
GAME_ID = 2023020653
GAME_DATE = "2024-01-13"


@pytest.fixture(scope="module")
def reconstructed():
    rows = s.reconstruct_game(GAME_ID)          # self-fetches via boxscore
    return {r["Team"]: r for r in rows}


@pytest.fixture(scope="module")
def hockeyref():
    df = pd.read_csv(TRAIN_CSV)
    df["Date"] = pd.to_datetime(df["Date"])
    day = df[df["Date"] == GAME_DATE]
    return {r["Team"]: r for _, r in day.iterrows()}


@requires_api
def test_two_team_rows(reconstructed):
    assert set(reconstructed) == {"WSH", "NYR"}


@requires_api
@pytest.mark.parametrize("team", ["WSH", "NYR"])
def test_direct_stats_exact(reconstructed, hockeyref, team):
    api, hr = reconstructed[team], hockeyref[team]
    for col in ["GF", "GA", "SOG", "PIM", "PPG", "PPO", "FOW", "FOL"]:
        assert api[col] == hr[col], f"{team} {col}: {api[col]} != {hr[col]}"
    assert api["FO%"] == pytest.approx(hr["FO%"], abs=0.2)
    assert api["Rslt"] == hr["Rslt"]


@requires_api
@pytest.mark.parametrize("team", ["WSH", "NYR"])
def test_advanced_stats_within_tolerance(reconstructed, hockeyref, team):
    api, hr = reconstructed[team], hockeyref[team]
    assert api["CF"]   == pytest.approx(hr["CF"],   abs=2)
    assert api["FF"]   == pytest.approx(hr["FF"],   abs=2)
    assert api["FA"]   == pytest.approx(hr["FA"],   abs=2)
    assert api["oZS%"] == pytest.approx(hr["oZS%"], abs=1.0)
    assert api["PDO"]  == pytest.approx(hr["PDO"],  abs=5.0)
