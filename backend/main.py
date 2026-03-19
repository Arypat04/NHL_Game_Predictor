from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import sys
import os
from dotenv import load_dotenv
load_dotenv()
# --- your imports from predictor.py go here ---
from predictor import NHLPredictor
# --- market registry goes here ---
MARKETS = {
    "moneyline": {
        "description": "NHL game winner predictions",
        "model":       NHLPredictor,        # the class itself
        "odds_market": "h2h",               # The Odds API market key
        "active":      True
    },
    "player_props": {
        "description": "Player prop predictions",
        "model":       None,                # not built yet
        "odds_market": "player_points",
        "active":      False               # flag it off until ready
    }
}

# --- lifespan (replaces @app.on_event("startup")) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = NHLPredictor()  # instantiate your predictor here
    # load your NHLPredictor once here and store it
    # so every request can use it without retraining
    yield

# --- app init ---
app = FastAPI(lifespan=lifespan)

# --- CORS middleware goes here ---

origins = [
    "http://localhost:5173",
    "http://localhost:8000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- routes ---
@app.get("/")
def root():
    return {"status": "NHL Predictor API is running"}

@app.get("/predictions")
def get_predictions(request : Request, date: str):
    predictor = request.app.state.predictor 
    results = predictor.predict_date(date)
    return results.to_dict(orient="records")

@app.get("/edges")
def get_edges(date: str):
    pass