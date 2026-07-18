import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from importlib import import_module

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from nhl_predictor import NHLPredictor, MODEL_PATH as NHL_MODEL_PATH
from mlb_predictor import MLBPredictor, MODEL_PATH as MLB_MODEL_PATH
from nhl_teams import TEAM_NAME_TO_ABBREV as NHL_TEAM_NAME_TO_ABBREV
from mlb_teams import (
    TEAM_NAME_TO_ABBREV as MLB_TEAM_NAME_TO_ABBREV,
    TEAM_ID_TO_ABBREV as MLB_TEAM_ID_TO_ABBREV,
)

load_dotenv()

SPORTS = {
    "nhl": {
        "odds_sport":    "icehockey_nhl",
        "team_name_map": NHL_TEAM_NAME_TO_ABBREV,
        "model_path":    NHL_MODEL_PATH,
        "active":        True,
    },
    "mlb": {
        "odds_sport":    "baseball_mlb",
        "team_name_map": MLB_TEAM_NAME_TO_ABBREV,
        "model_path":    MLB_MODEL_PATH,
        "active":        True,
    },
}

STATE_MAP = {
    "OFF": "Final", "LIVE": "Live", "PRG": "Live",
    "CRIT": "Live", "FUT": "Scheduled", "PRE": "Scheduled",
}

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
NHL_API_BASE  = "https://api-web.nhle.com/v1"
MLB_API_BASE  = "https://statsapi.mlb.com/api/v1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_team_name(name: str) -> str:
    return name.replace("é", "e").replace("É", "E").replace("è", "e")


def american_to_implied(odds: int) -> float:
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def get_predictor(request: Request, sport: str):
    sport = sport.lower()
    if sport not in ("nhl", "mlb"):
        raise HTTPException(status_code=400, detail=f"Unknown sport '{sport}'. Use 'nhl' or 'mlb'.")
    predictor = (request.app.state.nhl_predictor if sport == "nhl"
                 else request.app.state.mlb_predictor)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is still warming up — try again shortly.")
    return predictor


def get_mlb_postponed_games(date: str) -> set:
    """
    Live-check the MLB Stats API for any postponed games on a given date.
    Returns a set of matchup keys (e.g. "ATH_HOU") so we can filter them
    from predictions and edges without needing a scraper re-run.

    Only marks a game as postponed if it has NO matching Final entry for the
    same gamePk — same-day rainouts played later will have a Final entry and
    should still show.
    """
    try:
        resp = requests.get(
            f"{MLB_API_BASE}/schedule",
            params={"sportId": 1, "date": date, "gameType": "R"},
            timeout=5,
        )
        if resp.status_code != 200:
            return set()

        data = resp.json()

        # Build a lookup of gamePk → all statuses on this date
        pk_statuses: dict[int, list[str]] = {}
        for date_entry in data.get("dates", []):
            for game in date_entry.get("games", []):
                if game.get("officialDate") != date:
                    continue
                pk = game.get("gamePk")
                coded = game.get("status", {}).get("codedGameState", "")
                pk_statuses.setdefault(pk, []).append(coded)

        postponed = set()
        for date_entry in data.get("dates", []):
            for game in date_entry.get("games", []):
                if game.get("officialDate") != date:
                    continue
                coded    = game.get("status", {}).get("codedGameState", "")
                detailed = game.get("status", {}).get("detailedState", "")

                if coded != "D" and detailed != "Postponed":
                    continue

                pk = game.get("gamePk")
                # Only filter if there's no Final for this same gamePk
                has_final = "F" in pk_statuses.get(pk, [])
                if has_final:
                    continue

                away_id = game["teams"]["away"]["team"]["id"]
                home_id = game["teams"]["home"]["team"]["id"]
                away    = MLB_TEAM_ID_TO_ABBREV.get(away_id)
                home    = MLB_TEAM_ID_TO_ABBREV.get(home_id)
                if away and home:
                    postponed.add("_".join(sorted([away, home])))

        return postponed
    except Exception:
        return set()  # fail silently — never block predictions


def calculate_season_accuracy(sport: str) -> dict:
    """Season accuracy from MongoDB results, with a clean zero fallback."""
    if os.getenv("MONGODB_URI"):
        try:
            db_module = "nhl_database" if sport == "nhl" else "mlb_database"
            stats = import_module(db_module).get_season_stats()
            if stats and stats["total_predictions"] > 0:
                return {"accuracy": stats["season_accuracy"], "total": stats["total_predictions"]}
        except Exception as e:
            print(f"  ⚠ MongoDB stats failed ({sport}): {e}")

    return {"accuracy": 0.0, "total": 0}


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

def _warmup(app: FastAPI) -> None:
    """Load predictors + season stats. Runs in a background thread so the web
    server can bind its port immediately — loading the models (and, on a cold
    cache, collecting MLB starting-pitcher data) can take minutes, which would
    otherwise blow past Render's startup port-scan timeout.

    Each stage is isolated: a sport that loads slowly or fails must not stop the
    others from coming up. (A single shared try-block previously meant one slow
    MLB load left season accuracy permanently at 0.0%.)"""
    # Season stats read straight from MongoDB and don't need the predictors, so
    # do them first — the cards populate even if a model is still loading.
    for sport in ("nhl", "mlb"):
        try:
            stats = calculate_season_accuracy(sport)
            setattr(app.state, f"{sport}_accuracy", stats["accuracy"])
            setattr(app.state, f"{sport}_total", stats["total"])
            print(f"✓ {sport.upper()} accuracy: {stats['accuracy']:.1%} over {stats['total']} games")
        except Exception as e:
            print(f"⚠ {sport.upper()} season accuracy failed: {e}")

    for sport, label, cls in (("nhl", "NHL", NHLPredictor), ("mlb", "MLB", MLBPredictor)):
        try:
            print(f"Loading {label} predictor...")
            setattr(app.state, f"{sport}_predictor", cls())
            print(f"✓ {label} predictor ready")
        except Exception as e:
            print(f"⚠ {label} predictor failed to load: {e}")

    print("✓ Warmup complete\n")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # sensible defaults so routes respond (503 for predictions) before warmup finishes
    app.state.nhl_predictor = None
    app.state.mlb_predictor = None
    app.state.nhl_accuracy = 0.0; app.state.nhl_total = 0
    app.state.mlb_accuracy = 0.0; app.state.mlb_total = 0

    threading.Thread(target=_warmup, args=(app,), daemon=True).start()
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(lifespan=lifespan)

_origins = ["http://localhost:5173", "http://localhost:8000"]
if os.getenv("FRONTEND_URL"):
    _origins.append(os.getenv("FRONTEND_URL"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {"status": "Line Lab API is running", "sports": ["nhl", "mlb"]}


@app.get("/predictions")
def get_predictions(request: Request, date: str, sport: str = "nhl"):
    predictor = get_predictor(request, sport)
    results   = predictor.predict_date(date)
    if results is None:
        return []

    payload = results.to_dict(orient="records")

    # MLB: live-check for postponements so predictions stay accurate
    # even if the scraper hasn't been re-run since the postponement was announced
    if sport == "mlb":
        postponed = get_mlb_postponed_games(date)
        if postponed:
            before = len(payload)
            payload = [
                g for g in payload
                if "_".join(sorted([g["Away"], g["Home"]])) not in postponed
            ]
            removed = before - len(payload)
            if removed:
                print(f"  Filtered {removed} postponed game(s) from predictions for {date}")

    if os.getenv("MONGODB_URI"):
        try:
            if sport == "nhl":
                from nhl_database import log_predictions
            else:
                from mlb_database import log_predictions
            log_predictions(date, payload)
        except Exception:
            pass

    return payload


@app.get("/status")
def get_status(request: Request, sport: str = "nhl"):
    sport = sport.lower()
    if sport not in SPORTS:
        raise HTTPException(status_code=400, detail=f"Unknown sport '{sport}'")

    model_path = SPORTS[sport]["model_path"]
    accuracy   = request.app.state.nhl_accuracy if sport == "nhl" else request.app.state.mlb_accuracy
    total      = request.app.state.nhl_total    if sport == "nhl" else request.app.state.mlb_total

    return {
        "status":              "ok",
        "sport":               sport,
        "model_last_trained":  datetime.fromtimestamp(
            os.path.getmtime(model_path)
        ).isoformat() if os.path.exists(model_path) else None,
        "total_predictions":   total,
        "season_accuracy":     accuracy,
        "odds_api_configured": bool(os.getenv("ODDS_API_KEY")),
    }


@app.get("/results")
def get_results(request: Request, date: str, sport: str = "nhl"):
    sport     = sport.lower()
    predictor = get_predictor(request, sport)
    if sport == "nhl":
        return _nhl_results(date, predictor)
    return _mlb_results(date, predictor)


def _nhl_results(date: str, predictor) -> list:
    resp = requests.get(f"{NHL_API_BASE}/score/{date}")
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail="NHL API error")

    games = resp.json().get("games", [])
    if not games:
        return []

    predictions_df = predictor.predict_date(date)
    if predictions_df is None:
        return []

    pred_lookup = {
        "_".join(sorted([r["Home"], r["Away"]])): r
        for _, r in predictions_df.iterrows()
    }

    results = []
    for game in games:
        state = game.get("gameState", "")
        if state not in ("OFF", "LIVE", "PRG", "CRIT"):
            continue

        away = game["awayTeam"]["abbrev"]
        home = game["homeTeam"]["abbrev"]
        away = "VGK" if away == "VEG" else away
        home = "VGK" if home == "VEG" else home
        away_score = game["awayTeam"].get("score", 0)
        home_score = game["homeTeam"].get("score", 0)

        actual_winner = home if home_score > away_score else away
        pred = pred_lookup.get("_".join(sorted([home, away])))
        if pred is None:
            continue

        results.append({
            "Time":             pred["Time"],
            "Status":           STATE_MAP.get(state, "Unknown"),
            "Away":             away,
            "Home":             home,
            "Away_Score":       away_score,
            "Home_Score":       home_score,
            "Actual_Winner":    actual_winner,
            "Predicted_Winner": pred["Predicted_Winner"],
            "Home_Win_Prob":    pred["Home_Win_Prob"],
            "Away_Win_Prob":    pred["Away_Win_Prob"],
            "Correct":          actual_winner == pred["Predicted_Winner"],
        })

    if os.getenv("MONGODB_URI") and results:
        try:
            from nhl_database import log_results
            log_results(date, results)
        except Exception:
            pass

    return results


def _mlb_results(date: str, predictor) -> list:
    resp = requests.get(
        f"{MLB_API_BASE}/schedule",
        params={"sportId": 1, "date": date, "gameType": "R"},
        timeout=10,
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail="MLB API error")

    dates = resp.json().get("dates", [])
    if not dates:
        return []

    predictions_df = predictor.predict_date(date)
    if predictions_df is None:
        return []

    pred_lookup = {
        "_".join(sorted([r["Home"], r["Away"]])): r
        for _, r in predictions_df.iterrows()
    }

    results       = []
    seen_matchups = set()

    for game in dates[0].get("games", []):
        if game.get("officialDate") != date:
            continue

        coded    = game.get("status", {}).get("codedGameState", "")
        detailed = game.get("status", {}).get("detailedState", "")
        abstract = game.get("status", {}).get("abstractGameState", "")

        # Skip postponed — same-day rainouts played later keep their Final entry
        if coded == "D" or detailed == "Postponed":
            continue

        if abstract not in ("Final", "Live"):
            continue

        away_id = game["teams"]["away"]["team"]["id"]
        home_id = game["teams"]["home"]["team"]["id"]
        away    = MLB_TEAM_ID_TO_ABBREV.get(away_id)
        home    = MLB_TEAM_ID_TO_ABBREV.get(home_id)
        if not away or not home:
            continue

        matchup_key = "_".join(sorted([home, away]))

        # Deduplicate — prefer Final over Live when both appear
        if matchup_key in seen_matchups:
            if abstract == "Final":
                results = [r for r in results if "_".join(sorted([r["Home"], r["Away"]])) != matchup_key]
            else:
                continue

        seen_matchups.add(matchup_key)

        away_score    = game["teams"]["away"].get("score", 0) or 0
        home_score    = game["teams"]["home"].get("score", 0) or 0
        status        = "Final" if abstract == "Final" else "Live"
        actual_winner = home if home_score > away_score else away

        pred = pred_lookup.get(matchup_key)
        if pred is None:
            continue

        results.append({
            "Time":             pred["Time"],
            "Status":           status,
            "Away":             away,
            "Home":             home,
            "Away_Score":       away_score,
            "Home_Score":       home_score,
            "Actual_Winner":    actual_winner,
            "Predicted_Winner": pred["Predicted_Winner"],
            "Home_Win_Prob":    pred["Home_Win_Prob"],
            "Away_Win_Prob":    pred["Away_Win_Prob"],
            "Correct":          actual_winner == pred["Predicted_Winner"],
        })

    if os.getenv("MONGODB_URI") and results:
        try:
            from mlb_database import log_results
            log_results(date, results)
        except Exception:
            pass

    return results


@app.get("/edges")
def get_edges(request: Request, date: str, sport: str = "nhl"):
    sport   = sport.lower()
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ODDS_API_KEY not configured")
    if sport not in SPORTS:
        raise HTTPException(status_code=400, detail=f"Unknown sport '{sport}'")

    odds_sport    = SPORTS[sport]["odds_sport"]
    team_name_map = SPORTS[sport]["team_name_map"]
    predictor     = get_predictor(request, sport)

    resp = requests.get(
        f"{ODDS_API_BASE}/sports/{odds_sport}/odds",
        params={
            "apiKey":     api_key,
            "regions":    "us",
            "markets":    "h2h",
            "oddsFormat": "american",
            "dateFormat": "iso",
        },
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail="Odds API error")

    games_on_date = [
        g for g in resp.json()
        if g.get("commence_time", "")[:10] == date
    ]
    if not games_on_date:
        return []

    predictions_df = predictor.predict_date(date)
    if predictions_df is None:
        return []

    pred_lookup = {
        "_".join(sorted([r["Home"], r["Away"]])): r
        for _, r in predictions_df.iterrows()
    }

    # MLB: filter postponed games from edges too
    postponed = get_mlb_postponed_games(date) if sport == "mlb" else set()

    edges = []
    for game in games_on_date:
        home_full = normalize_team_name(game["home_team"])
        away_full = normalize_team_name(game["away_team"])
        home      = team_name_map.get(home_full)
        away      = team_name_map.get(away_full)
        if not home or not away:
            continue

        if "_".join(sorted([away, home])) in postponed:
            continue

        best_home_odds = best_away_odds = best_bookmaker = None
        for bookmaker in game.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market["key"] != "h2h":
                    continue
                for outcome in market["outcomes"]:
                    name = normalize_team_name(outcome["name"])
                    if name == home_full:
                        if best_home_odds is None or outcome["price"] > best_home_odds:
                            best_home_odds = outcome["price"]
                            best_bookmaker = bookmaker["title"]
                    elif name == away_full:
                        if best_away_odds is None or outcome["price"] > best_away_odds:
                            best_away_odds = outcome["price"]

        if best_home_odds is None or best_away_odds is None:
            continue

        home_implied = american_to_implied(best_home_odds)
        away_implied = american_to_implied(best_away_odds)

        pred = pred_lookup.get("_".join(sorted([home, away])))
        if pred is None:
            continue

        home_model = pred["Home_Win_Prob"]
        away_model = pred["Away_Win_Prob"]
        home_edge  = home_model - home_implied
        away_edge  = away_model - away_implied

        edges.append({
            "Time":              pred["Time"],
            "Away":              away,
            "Home":              home,
            "Home_Win_Prob":     round(home_model, 4),
            "Away_Win_Prob":     round(away_model, 4),
            "Home_Odds":         best_home_odds,
            "Away_Odds":         best_away_odds,
            "Home_Implied_Prob": round(home_implied, 4),
            "Away_Implied_Prob": round(away_implied, 4),
            "Home_Edge":         round(home_edge, 4),
            "Away_Edge":         round(away_edge, 4),
            "Best_Edge":         round(max(home_edge, away_edge), 4),
            "Best_Bet":          "Home" if home_edge > away_edge else "Away",
            "Bookmaker":         best_bookmaker,
        })

    edges.sort(key=lambda x: x["Best_Edge"], reverse=True)

    if os.getenv("MONGODB_URI") and edges:
        try:
            if sport == "nhl":
                from nhl_database import log_edges
            else:
                from mlb_database import log_edges
            log_edges(date, edges)
        except Exception:
            pass

    return edges
