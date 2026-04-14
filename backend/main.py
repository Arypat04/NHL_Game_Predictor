from fastapi import FastAPI, Request, HTTPException
import requests
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from predictor import NHLPredictor, MODEL_PATH, TEAM_NAME_TO_ABBREV
from database import (
    get_db, log_predictions, log_results,
    log_edges, get_season_stats
)
import os
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd

load_dotenv()

MARKETS = {
    "moneyline": {
        "description": "NHL game winner predictions",
        "model":       NHLPredictor,
        "odds_market": "h2h",
        "active":      True
    },
    "player_props": {
        "description": "Player prop predictions",
        "model":       None,
        "odds_market": "player_points",
        "active":      False
    }
}

STATE_MAP = {
    "OFF":  "Final",
    "LIVE": "Live",
    "PRG":  "Live",
    "CRIT": "Live",
    "FUT":  "Scheduled",
    "PRE":  "Scheduled",
}

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT         = "icehockey_nhl"
NHL_API_BASE  = "https://api-web.nhle.com/v1"


def normalize_team_name(name: str) -> str:
    return name.replace("é", "e").replace("É", "E").replace("è", "e")


def calculate_season_accuracy(predictor) -> dict:
    """
    Calculate season accuracy from MongoDB games collection.
    All keys are lowercase after _clean_doc fix.
    """
    try:
        db = get_db()
        if db is None:
            return {"accuracy": 0.571, "total": 0}

        print("  Loading completed 2026 games from MongoDB...")

        # all keys lowercase — use lowercase field names
        cursor = db["games"].find(
            {"season": 2026, "rslt": {"$in": ["W", "L"]}},
            {"_id": 0}
        )

        df = pd.DataFrame(list(cursor))

        if df.empty:
            print("  ⚠ No 2026 games found in MongoDB")
            return {"accuracy": 0.571, "total": 0}

        print(f"  Found {len(df):,} completed games")

        # lowercase column names from mongo — map to what predictor expects
        col_map = {
            "date": "Date", "team": "Team", "rslt": "Rslt",
            "home_away": "Home_Away", "opponent": "Opponent",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        df["Date"] = pd.to_datetime(df["Date"])
        df["date_str"] = df["Date"].dt.strftime("%Y-%m-%d")

        dates = df["date_str"].unique()
        correct = 0
        total = 0

        print(f"  Processing {len(dates)} dates...")

        for date_str in dates:
            predictions_df = predictor.predict_date(date_str)
            if predictions_df is None:
                continue

            day_games = df[df["date_str"] == date_str]

            for _, pred_row in predictions_df.iterrows():
                home = pred_row["Home"]
                away = pred_row["Away"]
                predicted_winner = pred_row["Predicted_Winner"]

                actual = day_games[
                    (day_games["Team"] == home) | (day_games["Team"] == away)
                ]

                for _, actual_row in actual.iterrows():
                    team = actual_row["Team"]
                    rslt = actual_row["Rslt"]
                    actual_winner = team if rslt == "W" else (away if team == home else home)

                    if predicted_winner == actual_winner:
                        correct += 1
                    total += 1
                    break

        accuracy = round(correct / total, 4) if total > 0 else 0.571
        print(f"  ✓ Accuracy: {accuracy:.2%} over {total} games")
        return {"accuracy": accuracy, "total": total}

    except Exception as e:
        print(f"  ⚠ Could not calculate season accuracy: {e}")
        return {"accuracy": 0.571, "total": 0}


def backfill_results(predictor) -> None:
    """
    Populate MongoDB results collection from games already in MongoDB.
    Runs at startup — skips dates already processed.
    All field names are lowercase (after _clean_doc fix).
    """
    db = get_db()
    if db is None:
        return

    print("Backfilling historical results into MongoDB...")

    # lowercase field names
    cursor = db["games"].find(
        {"rslt": {"$in": ["W", "L"]}},
        {"_id": 0}
    )
    games_list = list(cursor)

    if not games_list:
        print("  ⚠ No completed games found — run scraper first")
        return

    df = pd.DataFrame(games_list)

    # rename lowercase mongo keys to usable names
    col_map = {
        "date": "Date", "team": "Team", "rslt": "Rslt",
        "home_away": "Home_Away", "opponent": "Opponent",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    df["Date"] = pd.to_datetime(df["Date"])

    unique_dates = sorted(df["Date"].dt.strftime("%Y-%m-%d").unique())
    total_added = 0

    for date_str in unique_dates:
        # skip if already processed
        if db["results"].count_documents({"date": date_str}) > 0:
            continue

        predictions_df = predictor.predict_date(date_str)
        if predictions_df is None:
            continue

        pred_lookup = {
            "_".join(sorted([row["Home"], row["Away"]])): row
            for _, row in predictions_df.iterrows()
        }

        day_games = df[df["Date"].dt.strftime("%Y-%m-%d") == date_str]
        results = []

        for _, game in day_games.iterrows():
            team     = game["Team"]
            rslt     = game["Rslt"]
            home_away = game.get("Home_Away", "")
            opponent  = game.get("Opponent", "")

            home = team     if home_away == "Home" else opponent
            away = opponent if home_away == "Home" else team

            key  = "_".join(sorted([home, away]))
            pred = pred_lookup.get(key)
            if pred is None:
                continue

            actual_winner = home if rslt == "W" and home_away == "Home" else \
                            away if rslt == "W" else \
                            away if home_away == "Home" else home

            results.append({
                "Away":             away,
                "Home":             home,
                "Away_Score":       0,
                "Home_Score":       0,
                "Actual_Winner":    actual_winner,
                "Predicted_Winner": pred["Predicted_Winner"],
                "Home_Win_Rob":     pred["Home_Win_Prob"],
                "Away_Win_Prob":    pred["Away_Win_Prob"],
                "Correct":          actual_winner == pred["Predicted_Winner"],
                "Status":           "Final",
            })

        if results:
            log_results(date_str, results)
            total_added += len(results)

    print(f"✓ Backfill complete — {total_added} results added\n")


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_db()
    app.state.predictor = NHLPredictor()

    # backfill historical results first
    backfill_results(app.state.predictor)

    # then calculate accuracy from results collection
    print("Calculating season accuracy...")
    db_stats = get_season_stats()
    if db_stats and db_stats["total_predictions"] > 0:
        app.state.season_accuracy   = db_stats["season_accuracy"]
        app.state.total_predictions = db_stats["total_predictions"]
    else:
        # fallback to computing from games collection
        stats = calculate_season_accuracy(app.state.predictor)
        app.state.season_accuracy   = stats["accuracy"]
        app.state.total_predictions = stats["total"]

    print(f"✓ Season accuracy: {app.state.season_accuracy:.1%} over {app.state.total_predictions} games\n")
    yield


app = FastAPI(lifespan=lifespan)

origins = [
    "http://localhost:5173",
    "http://localhost:8000",
    "https://nhl-predictor-frontend.onrender.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "NHL Predictor API is running"}


@app.get("/predictions")
def get_predictions(request: Request, date: str):
    predictor = request.app.state.predictor
    results = predictor.predict_date(date)
    if results is None:
        return []
    data = results.to_dict(orient="records")
    log_predictions(date, data)
    return data


@app.get("/status")
def get_status(request: Request):
    db_stats = get_season_stats()
    return {
        "status": "ok",
        "model_last_trained": datetime.fromtimestamp(
            os.path.getmtime(MODEL_PATH)
        ).isoformat() if os.path.exists(MODEL_PATH) else None,
        "total_predictions": db_stats["total_predictions"] if db_stats else request.app.state.total_predictions,
        "season_accuracy":   db_stats["season_accuracy"]   if db_stats else request.app.state.season_accuracy,
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

    predictor = request.app.state.predictor
    predictions_df = predictor.predict_date(date)
    if predictions_df is None:
        return []

    pred_lookup = {}
    for _, row in predictions_df.iterrows():
        key = "_".join(sorted([row["Home"], row["Away"]]))
        pred_lookup[key] = row

    results = []
    for game in games:
        state = game.get("gameState", "")
        if state not in ("OFF", "LIVE", "PRG", "CRIT"):
            continue

        away = game["awayTeam"]["abbrev"]
        home = game["homeTeam"]["abbrev"]
        away_score = game["awayTeam"].get("score", 0)
        home_score = game["homeTeam"].get("score", 0)

        away = "VGK" if away == "VEG" else away
        home = "VGK" if home == "VEG" else home

        actual_winner = home if home_score > away_score else away

        key  = "_".join(sorted([home, away]))
        pred = pred_lookup.get(key)
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

    log_results(date, results)
    return results


def american_to_implied(odds: int) -> float:
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


@app.get("/edges")
def get_edges(request: Request, date: str):
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ODDS_API_KEY not configured")

    resp = requests.get(
        f"{ODDS_API_BASE}/sports/{SPORT}/odds",
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

    predictor = request.app.state.predictor
    predictions_df = predictor.predict_date(date)
    if predictions_df is None:
        return []

    pred_lookup = {}
    for _, row in predictions_df.iterrows():
        key = "_".join(sorted([row["Home"], row["Away"]]))
        pred_lookup[key] = row

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
                    outcome_name = normalize_team_name(outcome["name"])
                    if outcome_name == home_full:
                        if best_home_odds is None or outcome["price"] > best_home_odds:
                            best_home_odds = outcome["price"]
                            best_bookmaker = bookmaker["title"]
                    elif outcome_name == away_full:
                        if best_away_odds is None or outcome["price"] > best_away_odds:
                            best_away_odds = outcome["price"]

        if best_home_odds is None or best_away_odds is None:
            continue

        home_implied = american_to_implied(best_home_odds)
        away_implied = american_to_implied(best_away_odds)

        key  = "_".join(sorted([home, away]))
        pred = pred_lookup.get(key)
        if pred is None:
            continue

        home_model = pred["Home_Win_Prob"]
        away_model = pred["Away_Win_Prob"]
        home_edge  = home_model - home_implied
        away_edge  = away_model - away_implied
        best_bet   = "Home" if home_edge > away_edge else "Away"
        best_edge  = max(home_edge, away_edge)

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
            "Best_Edge":         round(best_edge, 4),
            "Best_Bet":          best_bet,
            "Bookmaker":         best_bookmaker,
        })

    edges.sort(key=lambda x: x["Best_Edge"], reverse=True)
    log_edges(date, edges)
    return edges