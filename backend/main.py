from fastapi import FastAPI, Request, HTTPException
import requests
from fastapi.middleware.cors import CORSMiddleware
from predictor import NHLPredictor, MODEL_PATH, TEAM_NAME_TO_ABBREV
import os
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd

load_dotenv()

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT = "icehockey_nhl"
NHL_API_BASE = "https://api-web.nhle.com/v1"


# ---------------------------------------------------------------------------
# APP SETUP
# ---------------------------------------------------------------------------

app = FastAPI()

app.state.predictor = None
app.state.season_accuracy = None
app.state.total_predictions = None


# ---------------------------------------------------------------------------
# CORS (FIXED FOR RENDER + FRONTEND)
# ---------------------------------------------------------------------------

origins = [
    "http://localhost:5173",
    "http://localhost:8000",
    "https://linelab-frontend.onrender.com",
]

if os.getenv("FRONTEND_URL"):
    origins.append(os.getenv("FRONTEND_URL"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,   # ❗ NO "*" (this was breaking debugging)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def normalize_team_name(name: str) -> str:
    return name.replace("é", "e").replace("É", "E").replace("è", "e")


def american_to_implied(odds: int) -> float:
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


# ---------------------------------------------------------------------------
# MONGO SAFE STATS LOADING
# ---------------------------------------------------------------------------

def load_season_stats():
    try:
        if os.getenv("MONGODB_URI"):
            from database import get_season_stats
            stats = get_season_stats()
            if stats:
                return stats
    except Exception as e:
        print("MongoDB error:", e)

    return {
        "season_accuracy": 0.571,
        "total_predictions": 0
    }


# ---------------------------------------------------------------------------
# LAZY PREDICTOR INIT
# ---------------------------------------------------------------------------

def get_predictor(request: Request):
    if request.app.state.predictor is None:
        print("Initializing predictor...")

        predictor = NHLPredictor()
        stats = load_season_stats()

        request.app.state.predictor = predictor
        request.app.state.season_accuracy = stats.get("season_accuracy", 0.571)
        request.app.state.total_predictions = stats.get("total_predictions", 0)

        print(f"✓ Predictor ready | accuracy: {stats.get('season_accuracy', 0.571):.1%}")

    return request.app.state.predictor


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {"status": "NHL Predictor API running"}


@app.get("/mongo-test")
def mongo_test():
    try:
        from database import get_season_stats
        return get_season_stats()
    except Exception as e:
        return {"error": str(e)}


@app.get("/status")
def get_status(request: Request):
    stats = load_season_stats()

    try:
        model_time = datetime.fromtimestamp(
            os.path.getmtime(MODEL_PATH)
        ).isoformat()
    except Exception:
        model_time = None

    return {
        "status": "ok",
        "model_last_trained": model_time,
        "total_predictions": stats["total_predictions"],
        "season_accuracy": stats["season_accuracy"],
        "odds_api_configured": bool(os.getenv("ODDS_API_KEY")),
    }

# ---------------------------------------------------------------------------
# PREDICTIONS
# ---------------------------------------------------------------------------

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
        except Exception as e:
            print("Mongo log failed:", e)

    return payload


# ---------------------------------------------------------------------------
# RESULTS
# ---------------------------------------------------------------------------

@app.get("/results")
def get_results(request: Request, date: str):
    try:
        resp = requests.get(f"{NHL_API_BASE}/score/{date}")

        if resp.status_code != 200:
            print("NHL API error:", resp.text)
            return []

        data = resp.json()
        games = data.get("games", [])

        if not games:
            return []

        predictor = get_predictor(request)
        predictions_df = predictor.predict_date(date)

        if predictions_df is None or predictions_df.empty:
            print("No predictions for date:", date)
            return []

        pred_lookup = {}

        for _, row in predictions_df.iterrows():
            try:
                key = "_".join(sorted([row["Home"], row["Away"]]))
                pred_lookup[key] = row
            except Exception as e:
                print("Bad prediction row skipped:", e)

        results = []

        for game in games:
            try:
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

                key = "_".join(sorted([home, away]))
                pred = pred_lookup.get(key)

                if not pred:
                    continue

                results.append({
                    "Time": pred.get("Time"),
                    "Status": state,
                    "Away": away,
                    "Home": home,
                    "Away_Score": away_score,
                    "Home_Score": home_score,
                    "Actual_Winner": actual_winner,
                    "Predicted_Winner": pred.get("Predicted_Winner"),
                    "Home_Win_Prob": pred.get("Home_Win_Prob"),
                    "Away_Win_Prob": pred.get("Away_Win_Prob"),
                    "Correct": actual_winner == pred.get("Predicted_Winner"),
                })

            except Exception as e:
                print("Game processing error:", e)
                continue

        return results

    except Exception as e:
        print("🔥 /results crashed fully:", str(e))
        return []


# ---------------------------------------------------------------------------
# EDGES
# ---------------------------------------------------------------------------

@app.get("/edges")
def get_edges(request: Request, date: str):
    api_key = os.getenv("ODDS_API_KEY")

    if not api_key:
        raise HTTPException(status_code=500, detail="ODDS_API_KEY not configured")

    resp = requests.get(
        f"{ODDS_API_BASE}/sports/icehockey_nhl/odds",
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

    games_on_date = [
        g for g in resp.json()
        if g.get("commence_time", "")[:10] == date
    ]

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

        best_home_odds = None
        best_away_odds = None
        best_bookmaker = None

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