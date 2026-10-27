# Line Lab

A sports-betting screener. It predicts moneyline outcomes for **NHL** and **MLB**
games, compares its probabilities against live bookmaker odds, and surfaces the
gaps as **edges** — flagging where the model disagrees with the market.

FastAPI backend · React (Vite) frontend · MongoDB Atlas · deployed on Render.

---

## How it predicts

Both sports use an **Elo rating model**. Every team carries one number (strength)
that moves after each game; before a game the home win probability is a logistic
function of the rating gap plus a home-ice/field bump:

```
P(home win) = 1 / (1 + 10^((R_away − R_home − HFA − rest) / 400))
```

On top of plain Elo, two improvements that were measured, not assumed, and kept:

- **Margin-of-victory multiplier** (both sports) — a blowout moves ratings more
  than a one-goal game, with an autocorrelation term so favorites don't run away.
- **Rest / back-to-back adjustment** (NHL) — a tired team is penalized relative to
  a rested one. (Off for MLB, which plays nearly every day.)

There is **no training, no model file, and no feature pipeline** — the model is a
dict of floats rebuilt in one cheap pass over completed games, rebuilt as-of each
prediction date so a game never sees its own result.

> This replaced an earlier RandomForest-on-rolling-features stack. An
> out-of-sample test showed a plain Elo matches or beats it, with none of the
> training cost — which also dissolved the free-tier memory/cold-start problems.
> Goalie (NHL) and starting-pitcher (MLB) features were built and measured, and
> **do not help** the moneyline — team ratings already price them in.

**Full write-up — model math, the survey of how NHL models are built, and the
honest record of what did and didn't improve ours — is in [`MODELING.md`](MODELING.md).**

### Accuracy (walk-forward, 3 held-out seasons)

| Sport | Accuracy | AUC | logloss |
|---|---|---|---|
| NHL | 0.597 | 0.634 | 0.661 |
| MLB | 0.562 | 0.584 | 0.680 |

Single-game NHL prediction has a well-documented **ceiling around 59–62%**; we're
there. The value is in well-calibrated probabilities, not raw accuracy.

### Edges

The market's implied probabilities include the bookmaker's margin (the vig), so
they sum to more than 100%. `/edges` **de-vigs** the two best prices into a fair
probability that sums to 1, then:

```
edge = model probability − no-vig fair probability
```

The model is kept **independent of the market** on purpose — blending the market
in would improve accuracy but erase the edge signal, which is the whole point.

---

## Layout

```
backend/
  elo.py                 the Elo engine (ratings, MOV multiplier, rest, as-of predictions)
  elo_evaluate.py        walk-forward evaluator (dev tool)
  main.py                FastAPI app: /predictions /results /edges /status
  seasons.py             single source of truth for season years (rolling window)
  base_database.py       shared MongoDB layer
  {nhl,mlb}_database.py   per-sport collections and rename maps
  {nhl,mlb}_scraper.py    data collection (official APIs)
  {nhl,mlb}_predictor.py  load data, build ratings, predict
  mlb_pitchers.py         per-pitcher rolling form (staged for future run-based markets)
  weekly_refresh.py       incremental MongoDB refresh
  tests/                  per-sport tests + shared data-integrity guards
frontend/                 React + Vite UI (see DESIGN.md for the design system)
.github/workflows/        weekly refresh (free scheduled runner)
MODELING.md               model + research reference
DESIGN.md / PRODUCT.md    frontend design system / product brief
```

---

## Data sources

All **official public APIs** — no HTML scraping.

- **NHL** — `api-web.nhle.com`: standings, schedules, box scores, play-by-play
  (shot/score data drives the margin-of-victory multiplier).
- **MLB** — `statsapi.mlb.com`: schedules and team game logs.
- **Odds** — The Odds API (free tier, 500 requests/month).

---

## Running it

```bash
python -m venv venv && venv/Scripts/activate       # Windows (use bin/activate on *nix)
pip install -r backend/requirements.txt -r backend/requirements-dev.txt

# .env at the repo root (optional — the app falls back to CSVs in data/)
#   MONGODB_URI=...
#   ODDS_API_KEY=...

cd backend && uvicorn main:app --reload            # API on :8000
cd frontend && npm install && npm run dev          # UI  on :5173

pytest backend/tests -q                            # tests (offline, no network)
python backend/elo_evaluate.py                     # measure the model, walk-forward
```

The predictors work without MongoDB, falling back to the CSVs in `data/`.

### Collecting / refreshing data

```bash
python backend/nhl_scraper.py     # full history (resumable, cached per season)
python backend/mlb_scraper.py
python backend/weekly_refresh.py  # incremental — only games not already stored
```

`weekly_refresh.py` runs **every Monday on GitHub Actions** (free). It only fetches
games MongoDB doesn't already have, and stamps a timestamp that the UI shows as
"Updated". Ratings are therefore current as of the last weekly refresh; a game
played mid-week influences predictions after the following Monday.

---

## Deployment

Render free tier, backend + frontend from this repo. Because the model is Elo
(no training, no pickled model, no model cache), warmup is just a quick data load
from MongoDB — the 512 MB memory limit and ~15-minute cold-start spin-downs that
shaped the old design are no longer a constraint. The schema served by
`/predictions`, `/results` and `/edges` is stable, so the frontend is decoupled
from model internals.

---

## Roadmap

- [x] NHL moneyline (Elo + margin-of-victory + rest)
- [x] MLB moneyline (Elo + margin-of-victory)
- [ ] Poisson score layer — unlocks **totals / puck line / run line / props**
- [ ] NBA + NFL moneyline (`seasons.py` already configures both)
- [ ] Closing-line-value tracking for the edges
