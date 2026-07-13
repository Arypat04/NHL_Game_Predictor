"""NHL predictor, team-constant and database tests."""

import pytest

import nhl_database
from conftest import PUBLIC_PREDICTION_COLS, first_date_with_games
from nhl_predictor import NHLPredictor, SCHEDULE_CSV
from nhl_teams import RELOCATION_MAP, TEAM_NAME_TO_ABBREV, VALID_ABBREVS


@pytest.fixture(scope="module")
def predictor():
    return NHLPredictor()


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


def test_predicted_winner_uses_strict_half_threshold(predictions):
    # NHL picks the home team only when Home_Win_Prob > 0.5
    for _, row in predictions.iterrows():
        assert (row["Predicted_Winner"] == row["Home"]) == (row["Home_Win_Prob"] > 0.5)


def test_one_row_per_matchup(predictions):
    keys = ["_".join(sorted([r["Home"], r["Away"]])) for _, r in predictions.iterrows()]
    assert len(keys) == len(set(keys))


def test_unknown_date_returns_none(predictor):
    assert predictor.predict_date("1990-06-15") is None


# -- team constants -----------------------------------------------------------

def test_known_team_mappings():
    assert TEAM_NAME_TO_ABBREV["Vegas Golden Knights"] == "VGK"
    assert TEAM_NAME_TO_ABBREV["Toronto Maple Leafs"] == "TOR"


def test_montreal_accent_normalization():
    assert TEAM_NAME_TO_ABBREV["Montréal Canadiens"] == "MTL"
    assert TEAM_NAME_TO_ABBREV["Montreal Canadiens"] == "MTL"


def test_relocation_map():
    assert RELOCATION_MAP["VEG"] == "VGK"
    assert RELOCATION_MAP["ARI"] == "UTA"


def test_abbrevs_are_clean():
    assert None not in VALID_ABBREVS
    assert all(a.isupper() and len(a) == 3 for a in VALID_ABBREVS)


# -- database document cleaning (pure, no Mongo) ------------------------------

def test_clean_doc_normalizes_and_coerces():
    doc = nhl_database._clean_doc({
        "FO%": "55.1", "oZS%": "48.2", "GF": "3", "Unnamed: 7": "junk", "Team": "BOS",
    })
    assert "unnamed" not in str(doc)
    assert doc["fopct"] == 55.1
    assert doc["ozspct"] == 48.2
    assert doc["gf"] == 3.0 and isinstance(doc["gf"], float)
    assert doc["team"] == "BOS"
