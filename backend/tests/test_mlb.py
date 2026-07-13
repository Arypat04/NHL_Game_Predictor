"""MLB predictor, team-constant and database tests."""

import pytest

import mlb_database
from conftest import PUBLIC_PREDICTION_COLS, first_date_with_games
from mlb_predictor import MLBPredictor, SCHEDULE_CSV
from mlb_teams import TEAM_ID_TO_ABBREV, VALID_ABBREVS, VALID_TEAM_IDS


@pytest.fixture(scope="module")
def predictor():
    return MLBPredictor()


@pytest.fixture(scope="module")
def predictions(predictor):
    date = first_date_with_games(predictor, SCHEDULE_CSV)
    return predictor.predict_date(date)


# -- predictions --------------------------------------------------------------

def test_predict_date_returns_public_columns(predictions):
    assert list(predictions.columns) == PUBLIC_PREDICTION_COLS


def test_no_internal_columns_leak(predictions):
    assert not [c for c in predictions.columns if c.startswith("_")]


def test_probabilities_are_complementary(predictions):
    for _, row in predictions.iterrows():
        assert row["Home_Win_Prob"] + row["Away_Win_Prob"] == pytest.approx(1.0)
        assert 0.0 <= row["Home_Win_Prob"] <= 1.0
        assert row["Confidence"] == pytest.approx(
            max(row["Home_Win_Prob"], row["Away_Win_Prob"])
        )


def test_predicted_winner_is_a_participant(predictions):
    for _, row in predictions.iterrows():
        assert row["Predicted_Winner"] in (row["Home"], row["Away"])


def test_predicted_winner_uses_inclusive_half_threshold(predictions):
    # MLB picks the home team when Home_Win_Prob >= 0.5
    for _, row in predictions.iterrows():
        assert (row["Predicted_Winner"] == row["Home"]) == (row["Home_Win_Prob"] >= 0.5)


def test_one_row_per_matchup(predictions):
    keys = ["_".join(sorted([r["Home"], r["Away"]])) for _, r in predictions.iterrows()]
    assert len(keys) == len(set(keys))


def test_unknown_date_returns_none(predictor):
    assert predictor.predict_date("1990-06-15") is None


# -- team constants -----------------------------------------------------------

def test_athletics_is_ath_not_oak():
    assert TEAM_ID_TO_ABBREV[133] == "ATH"
    assert "OAK" not in VALID_ABBREVS


def test_t_prefixed_teams_are_valid():
    # Guard against the startswith("T") filter bug that dropped these.
    assert {"TOR", "TBR", "TEX"}.issubset(VALID_ABBREVS)


def test_thirty_teams():
    assert len(VALID_TEAM_IDS) == 30
    assert len(VALID_ABBREVS) == 30


# -- database document cleaning (pure, no Mongo) ------------------------------

def test_clean_doc_handles_slash_keys_and_coercion():
    doc = mlb_database._clean_doc({
        "P_K/BB": "2.5", "HR": "1", "OBP": "0.340", "Unnamed: 0": "junk", "Team": "NYY",
    })
    assert "unnamed" not in str(doc)
    assert doc["p_k_bb"] == 2.5            # "/" normalized to "_"
    assert doc["hr"] == 1.0 and isinstance(doc["hr"], float)
    assert doc["obp"] == 0.340
    assert doc["team"] == "NYY"
