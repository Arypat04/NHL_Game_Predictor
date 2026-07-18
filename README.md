# Line Lab

A sports betting screener: it predicts moneyline outcomes for **NHL** and **MLB**
games, compares its probabilities against bookmaker odds, and surfaces the gaps
as "edges".

FastAPI backend + React (Vite) frontend + MongoDB Atlas, deployed on Render.

## How it predicts

Each sport uses a **matchup model** — one row per game built from *both* teams'
leakage-safe rolling form (every rolling feature is computed with `closed="left"`
so a game never sees itself). A single calibrated RandomForest per sport turns
that into P(home win).

What actually carries signal, measured rather than assumed:

| Sport | Dominant signal | Notes |
|---|---|---|
| NHL | Team shot-share (Corsi/Fenwick) | Goalie and top-scorer features were built, measured, and **dropped** — possession already prices them in |
| MLB | The **starting pitcher** | ~50% of feature importance. Team pitching stats are redundant once the actual starter's form is a feature |

Rolling windows differ by sport because the sports differ: NHL uses `[5, 10, 20]`
(hockey form is streaky), MLB uses `[10, 20, 40]` (baseball stabilizes slowly).

## Accuracy

Season walk-forward cross-validation — the honest numbers:

| Sport | Accuracy | AUC |
|---|---|---|
| NHL | ~0.585 | ~0.614 |
| MLB | ~0.559 | ~0.579 |

Both sports sit at their **data ceiling**. An exhaustive search (2040 NHL / 816
MLB configurations across logreg, RF, ExtraTrees, GB, HistGB, XGBoost, stacking,
MLP and SVM) found 117 of 2040 configs within 0.005 accuracy of the best — all
inside the ~0.017 season-to-season noise band. Model choice is exhausted; further
accuracy needs new *data* (confirmed lineups, injuries), not new models.

> The "season accuracy" shown in the UI back-tests on games the model trained on,
> so it reads slightly optimistic. Trust the CV numbers above.

## Layout

```
backend/
  main.py                FastAPI app: /predictions /results /edges /status
  seasons.py             single source of truth for season years (rolling window)
  base_database.py       shared MongoDB layer + trained-model cache
  {nhl,mlb}_database.py  per-sport collections and rename maps
  {nhl,mlb}_scraper.py   data collection (official APIs)
  {nhl,mlb}_matchup.py   rolling features + one-row-per-game dataset
  {nhl,mlb}_predictor.py training, model load/cache, prediction
  mlb_pitchers.py        per-pitcher rolling form (MLB's key feature)
  model_search.py        offline model/feature search (research tool)
  weekly_refresh.py      incremental MongoDB refresh
  tests/                 per-sport tests + shared data-integrity guards
frontend/                React + Vite UI
.github/workflows/       weekly refresh (free scheduled runner)
```

## Data sources

Both are **official public APIs** — no HTML scraping. (An earlier version scraped
Hockey-Reference; it was validated stat-by-stat against the NHL API and retired.)

- NHL: `api-web.nhle.com` — standings, schedules, boxscores, play-by-play
  (5v5 possession is reconstructed from PBP events)
- MLB: `statsapi.mlb.com` — schedules, team game logs, pitcher game logs
- Odds: The Odds API (free tier — 500 requests/month)

## Running it

```bash
python -m venv venv && venv/Scripts/activate      # Windows
pip install -r backend/requirements.txt -r backend/requirements-dev.txt

# .env at the repo root
#   MONGODB_URI=...
#   ODDS_API_KEY=...

cd backend && uvicorn main:app --reload            # API on :8000
cd frontend && npm install && npm run dev          # UI  on :5173

pytest backend/tests -q                            # tests (offline, no network)
```

The predictors work without MongoDB, falling back to CSVs in `data/`.

### Collecting data

```bash
python backend/nhl_scraper.py     # full history (resumable, cached per season)
python backend/mlb_scraper.py
python backend/weekly_refresh.py  # incremental — only what changed
```

`weekly_refresh.py` runs every Monday on GitHub Actions (free). It's incremental:
games already in MongoDB are never re-fetched, and completed pitcher seasons are
skipped, since finished games are immutable.

## Deployment notes

Render's **free tier has no persistent disk**, so the filesystem is wiped on every
spin-down. Two consequences shaped the design:

1. **Models are cached in MongoDB** (`base_database.save_model/load_model`) and
   restored on cold start in ~2s instead of being retrained. They're stored as
   single BSON documents, which only works because the deployed models are ~4 MB.
2. **Models are deliberately lean** — one RandomForest, `n_jobs=1`. The previous
   RF+ExtraTrees blend OOM'd the 512 MB dyno (`n_jobs=-1` forks a copy of the data
   per worker), and the accuracy difference is inside the noise band.

Warmup runs in a background thread so the port binds immediately; endpoints return
`503` until their sport is ready.

## Roadmap

- [x] NHL moneyline
- [x] MLB moneyline
- [ ] NBA + NFL moneyline (`seasons.py` already has both; NFL's QB should play the
      role MLB's starting pitcher does)
- [ ] Player props
