from fastapi import FastAPI, Request, HTTPException
import requests
from fastapi.middleware.cors import CORSMiddleware
from predictor import NHLPredictor, MODEL_PATH, TEAM_NAME_TO_ABBREV
import os
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd

load_dotenv()

# --- market registry ---
MARKETS = {
    "moneyline": {
        "description": "NHL game winner predictions",
        "model": NHLPredictor,
        "odds_market": "h2h",
        "active": True,
    },
    "player_props": {
        "description": "Player prop predictions",
        "model": None,
        "odds_market": "player_points",
        "active": False,
    },
}

STATE_MAP = {
    "OFF": "Final",
    "LIVE": "Live",
    "PRG": "Live",
    "CRIT": "Live",
    "FUT": "Scheduled",
    "PRE": "Scheduled",
}

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT = "icehockey_nhl"
NHL_API_BASE = "https://api-web.nhle.com/v1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_team_name(name: str) -> str:
    return name.replace("é", "e").replace("É", "E").replace("è", "e")


def american_to_implied(odds: int) -> float:
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def calculate_season_accuracy(predictor) -> dict:
    if os.getenv("MONGODB_URI"):
        try:
            from database import get_season_stats
            stats = get_season_stats()
            if stats and stats["total_predictions"] > 0:
                print(f"Using MongoDB stats: {stats['season_accuracy']:.1%}")
                return {"accuracy": stats["season_accuracy"], "total": stats["total_predictions"]}
        except Exception as e:
            print(f"MongoDB fallback failed: {e}")

    try:
        schedule_path = os.path.join(os.path.dirname(MODEL_PATH), "../data/nhl_matches_2026.csv")
        schedule = pd.read_csv(schedule_path)
        completed = schedule.dropna(subset=["Rslt"])

        if completed.empty:
            return {"accuracy": 0.571, "total": 0}

        completed["Date"] = pd.to_datetime(completed["Date"])
        dates = completed["Date"].dt.strftime("%Y-%m-%d").unique()

        correct, total = 0, 0

        for date_str in dates:
            predictions_df = predictor.predict_date(date_str)
            if predictions_df is None:
                continue

            day_games = completed[completed["Date"].dt.strftime("%Y-%m-%d") == date_str]

            for _, pred_row in predictions_df.iterrows():
                home, away = pred_row["Home"], pred_row["Away"]
                predicted_winner = pred_row["Predicted_Winner"]

                actual = day_games[(day_games["Team"] == home) | (day_games["Team"] == away)]

                for _, actual_row in actual.iterrows():
                    team = actual_row["Team"]
                    rslt = actual_row["Rslt"]
                    actual_winner = team if rslt == "W" else (away if team == home else home)

                    if predicted_winner == actual_winner:
                        correct += 1
                    total += 1
                    break

        accuracy = round(correct / total, 4) if total > 0 else 0.571
        return {"accuracy": accuracy, "total": total}

    except Exception as e:
        print(f"Accuracy calc failed: {e}")
        return {"accuracy": 0.571, "total": 0}


# ---------------------------------------------------------------------------
# App (NO BLOCKING STARTUP)
# ---------------------------------------------------------------------------

app = FastAPI()

app.state.predictor = None
app.state.season_accuracy = None
app.state.total_predictions = None


def get_predictor(request: Request):
    if request.app.state.predictor is None:
        print("Initializing predictor (lazy load)...")

        predictor = NHLPredictor()
        stats = calculate_season_accuracy(predictor)

        request.app.state.predictor = predictor
        request.app.state.season_accuracy = stats["accuracy"]
        request.app.state.total_predictions = stats["total"]

        print(f"✓ Season accuracy: {stats['accuracy']:.1%}")

    return request.app.state.predictor


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

_origins = [
    "http://localhost:5173",
    "http://localhost:8000",
    "https://linelab-frontend.onrender.com"
]

if os.getenv("FRONTEND_URL"):
    _origins.append(os.getenv("FRONTEND_URL"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {"status": "NHL Predictor API is running"}


@app.get("/predictions")
def get_predictions(request: Request, date: str):
    predictor = get_predictor(request)
    results = predictor.predict_date(date)

    if results is None:
        return []

    payload = results.to_dict(orient="records")

    if os.getenv("MONGODB_URI"):
        try:
            from database import log_predictions
            log_predictions(date, payload)
        except Exception:
            pass

    return payload


@app.get("/status")
def get_status(request: Request):
    try:
        model_time = os.path.getmtime(MODEL_PATH)
        model_time_iso = datetime.fromtimestamp(model_time).isoformat()
    except Exception:
        model_time_iso = None

    return {
        "status": "ok",
        "model_last_trained": model_time_iso,
        "total_predictions": getattr(request.app.state, "total_predictions", 0) or 0,
        "season_accuracy": getattr(request.app.state, "season_accuracy", 0.571),
        "odds_api_configured": bool(os.getenv("ODDS_API_KEY")),
    }


@app.get("/results")
def get_results(request: Request, date: str):
    resp = requests.get(f"{NHL_API_BASE}/score/{date}")
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail="NHL API error")

    games = resp.json().get("games", [])
    if not games:
        return []

    predictor = get_predictor(request)
    predictions_df = predictor.predict_date(date)

    if predictions_df is None:
        return []

    pred_lookup = {
        "_".join(sorted([row["Home"], row["Away"]])): row
        for _, row in predictions_df.iterrows()
    }

    results = []

    for game in games:
        state = game.get("gameState", "")
        if state not in ("OFF", "LIVE", "PRG", "CRIT"):
            continue

        away = "VGK" if game["awayTeam"]["abbrev"] == "VEG" else game["awayTeam"]["abbrev"]
        home = "VGK" if game["homeTeam"]["abbrev"] == "VEG" else game["homeTeam"]["abbrev"]

        away_score = game["awayTeam"].get("score", 0)
        home_score = game["homeTeam"].get("score", 0)

        actual_winner = home if home_score > away_score else away

        key = "_".join(sorted([home, away]))
        pred = pred_lookup.get(key)

        if not pred:
            continue

        results.append({
            "Time": pred["Time"],
            "Status": STATE_MAP.get(state, "Unknown"),
            "Away": away,
            "Home": home,
            "Away_Score": away_score,
            "Home_Score": home_score,
            "Actual_Winner": actual_winner,
            "Predicted_Winner": pred["Predicted_Winner"],
            "Home_Win_Prob": pred["Home_Win_Prob"],
            "Away_Win_Prob": pred["Away_Win_Prob"],
            "Correct": actual_winner == pred["Predicted_Winner"],
        })

    return results


@app.get("/edges")
def get_edges(request: Request, date: str):
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ODDS_API_KEY not configured")

    resp = requests.get(
        f"{ODDS_API_BASE}/sports/{SPORT}/odds",
        params={
            "apiKey": api_key,
            "regions": "us",
            "markets": "h2h",
            "oddsFormat": "american",
            "dateFormat": "iso",
        },
    )

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail="Odds API error")

    games_on_date = [g for g in resp.json() if g.get("commence_time", "")[:10] == date]
    if not games_on_date:
        return []

    predictor = get_predictor(request)
    predictions_df = predictor.predict_date(date)

    if predictions_df is None:
        return []

    pred_lookup = {
        "_".join(sorted([row["Home"], row["Away"]])): row
        for _, row in predictions_df.iterrows()
    }

    edges = []

    for game in games_on_date:
        home_full = normalize_team_name(game["home_team"])
        away_full = normalize_team_name(game["away_team"])

        home = TEAM_NAME_TO_ABBREV.get(home_full)
        away = TEAM_NAME_TO_ABBREV.get(away_full)
        if not home or not away:
            continue

        best_home_odds, best_away_odds, best_bookmaker = None, None, None

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

        key = "_".join(sorted([home, away]))
        pred = pred_lookup.get(key)
        if not pred:
            continue

        home_model = pred["Home_Win_Prob"]
        away_model = pred["Away_Win_Prob"]

        home_edge = home_model - home_implied
        away_edge = away_model - away_implied

        best_bet = "Home" if home_edge > away_edge else "Away"
        best_edge = max(home_edge, away_edge)

        edges.append({
            "Time": pred["Time"],
            "Away": away,
            "Home": home,
            "Home_Win_Prob": round(home_model, 4),
            "Away_Win_Prob": round(away_model, 4),
            "Home_Odds": best_home_odds,
            "Away_Odds": best_away_odds,
            "Home_Implied_Prob": round(home_implied, 4),
            "Away_Implied_Prob": round(away_implied, 4),
            "Home_Edge": round(home_edge, 4),
            "Away_Edge": round(away_edge, 4),
            "Best_Edge": round(best_edge, 4),
            "Best_Bet": best_bet,
            "Bookmaker": best_bookmaker,
        })

    edges.sort(key=lambda x: x["Best_Edge"], reverse=True)

    return edges