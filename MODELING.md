# Modeling — Line Lab

How the prediction models work, the research behind them, and the honest record
of what has and hasn't improved them. For the code, see `backend/elo.py`,
`backend/nhl_predictor.py`, `backend/mlb_predictor.py`, and the walk-forward
evaluator `backend/elo_evaluate.py`.

---

## 1. The model: Elo + margin-of-victory + rest

Both sports use an **Elo rating model**. Every team has one number (start 1500)
that moves after each game. There is no training, no model file, no feature
pipeline — it is a dict of floats rebuilt in one cheap pass over completed games,
which is why it loads instantly and sidesteps the free-tier memory/cold-start
problems that the old RandomForest stack hit.

### Win probability
```
P(home win) = 1 / (1 + 10^((R_away − R_home − HFA − rest_adj) / 400))
```
- **Rating gap** `R_home − R_away` — the core team-strength signal.
- **HFA** — home-ice/field advantage in Elo points. At equal ratings this gives
  the real home-win rate (NHL ~0.540 at HFA 35; MLB ~0.531 at HFA 24).
- **`/400`** — the standard Elo scale (a 400-pt gap ≈ 91% win prob).

### Rating update (with the margin-of-victory multiplier)
```
error = actual − P            # actual = 1 if home won, else 0
K_eff = K · ln(|margin| + 1) · ( C / (C + D · gap_winner) )
R_home += K_eff · error ;  R_away −= K_eff · error
```
- **K** — base learning rate (how far one game moves a rating).
- **Margin-of-victory multiplier** — a blowout is stronger evidence than a
  1-goal/1-run squeaker, so it moves ratings more.
- **Autocorrelation damping** `C/(C + D·gap_winner)` — shrinks the move when a
  big favorite wins, so top teams don't run away. (NHL validated to `D=0`, i.e.
  pure margin; MLB uses a light `D`.)

### Rest / back-to-back adjustment (NHL)
A rested team gets an effective-rating bump; a tired/back-to-back team is
relatively penalized (`rest_adj = rest_scale · (min(rest, cap) − 1)`, centered so
one day's rest is neutral). This is real signal in the NHL and **off for MLB**
(`rest_scale = 0`), which plays nearly every day.

### Between seasons
Each team's rating is pulled 30% toward the mean (`R = 1500 + 0.70·(R − 1500)`)
to account for roster turnover.

### Parameters (`elo.PARAMS`, out-of-sample validated)
| | k | HFA | regress | MOV (c, d) | rest_scale, cap |
|---|---|---|---|---|---|
| NHL | 8 | 35 | 0.70 | 1.0, 0.0 | 24, 2 |
| MLB | 4 | 24 | 0.70 | 1.0, 0.001 | 0 (off) |

Ratings are rebuilt **as of each prediction date** (only games strictly before
it), so a prediction never sees its own result. Output schema is unchanged, so
`main.py`, `/edges` and the frontend are unaffected.

### Measured performance (walk-forward mean, 3 held-out seasons)
| | accuracy | AUC | logloss |
|---|---|---|---|
| NHL | 0.597 | 0.634 | 0.661 |
| MLB | 0.562 | 0.584 | 0.680 |

NHL sits at the **academic accuracy ceiling** (~59–62% for single games; see
below), so the gains worth chasing are in probability *quality* (AUC / logloss),
not raw accuracy.

---

## 2. The landscape: how NHL models are built

A survey of public models, academic work, and betting models, with provenance
and trade-offs.

### Elo / rating systems — *our family*
- **Where:** FiveThirtyEight's NHL Elo (built by Neil Paine, data to 1917-18),
  now deprecated; rebuilds at Hockey Alchemy, Harvitronix, Neil Paine's Substack.
- **538 spec** (we are ~95% this): `K=6`, `+50` home ice, MOV multiplier
  `0.6686·ln(MOV)+0.8048`, autocorrelation `2.05/(gap·0.001+2.05)`, ×1.25 in
  playoffs, regress 70%/toward 1505.
- **Pros:** simple, instant, no training, well-calibrated, no overfitting;
  matches ML on single games. **Cons:** knows only wins/losses — blind to shot
  quality; slow to react to roster changes.

### xG-based team models — *the best public NHL model (MoneyPuck)*
- **Where:** MoneyPuck (Peter Tanner). Shot-level xG via gradient boosting +
  "flurry adjustment"; pre-game win model trained 2017–2024.
- **How:** blends recent win%, shooting-adjusted **xG** + shot volume, and
  **goaltending (29% weight)** plus a starting-goalie/rest model.
- **Pros:** captures shot *quality*, reacts faster, handles goalie/rest.
  **Cons:** needs a shot-by-shot xG pipeline; heavier; needs confirmed starters.
- **Key insight:** MoneyPuck weights goaltending 29%, yet *our* goalie
  experiment found no lift — because their base is **xG** (which ignores how well
  the goalie actually stopped pucks, so they must add it back), while our base is
  **results** (where goalie play is already in the W/L). Same logic explains why
  our MLB starting-pitcher experiment failed.

### Player-aggregation (The Athletic)
- **Where:** Dom Luszczyszyn — Game Score Value Added → "Net Rating" (2023);
  team strength = Σ player values → 50,000 Monte Carlo sims.
- **Pros:** handles injuries/trades/lineups directly. **Cons:** needs per-player
  projections (heavy), paywalled, not clearly better per-game.

### Poisson / bivariate-Poisson goal models
- **Where:** adapted from soccer (Dixon-Coles 1997); Dobson & Goddard (2011) did
  NHL. 538 uses it for simulation (`goals ~ Poisson(2.84 + 0.0042·EloDiff)`).
- **Pros:** gives the full **score distribution** — needed for totals / puck line
  / props. **Cons:** not more accurate than Elo for moneyline; the Dixon-Coles
  low-score correction is tuned for soccer draws.

### Pure ML ensembles
- **Where:** Corsica (14 sub-models, CatBoost best), academic ML (Weissbock
  2014), Kaggle projects.
- **Pros:** can fuse many features. **Cons:** this is what Elo *beat* in our own
  tests — overfits noisy box-score features, same ceiling, far more complexity.

### Market-based / betting
- **Where:** Benter's two-stage method (horse racing); Hubáček et al. (2019).
- **Pros:** the closing line is the hardest, best-calibrated benchmark; a logit
  combining model + market improves calibration. **Cons — critical for us:** for
  *profit* you want to **decorrelate** from the market, not blend into it.
  Blending kills the edge signal. So we keep the model independent and use the
  market only as the comparison (which `/edges` does). Track **CLV** as the real
  scorecard.

### The accuracy ceiling
Rigorous single-game NHL models land at **~59–62%** (Weissbock 2014: 59.8%,
argued ~62% ceiling; neural nets ~59.4%). Claims >65% on single games are usually
data leakage or tiny test sets. Our NHL ~59.7% is at the wall. Series/playoff
prediction is easier (~75%).

---

## 3. What we tested (the honest scorecard)

Every idea was measured walk-forward (tune on one season, confirm on later ones).
Kept only what beat the current model out-of-sample.

| Idea | Result | Verdict |
|---|---|---|
| **Margin-of-victory multiplier** | NHL +AUC/+acc/−logloss; MLB +AUC/−logloss | ✅ **Shipped** (both) |
| **Rest / back-to-back (NHL)** | +0.0040 AUC, −0.0017 logloss on test | ✅ **Shipped** (NHL) |
| Rest (MLB) | tuned to zero effect (plays ~daily) | ⏸️ off |
| **NHL goalie features / GSAx** | 23% feature importance, **no** test lift | ❌ team results price it in |
| **MLB starting-pitcher Elo adjustment** | tuner drove scale→0, slightly hurt test | ❌ same reason |
| **Glicko-style rating uncertainty** | boosted accuracy but **hurt** test logloss | ❌ overfit, dropped |
| Market blending | improves accuracy, destroys edge signal | 🚫 deliberately avoided |

**Pattern:** position-specific signals (goalies, pitchers) that *feel* huge are
largely redundant with a results-based team rating for predicting *who wins*.
The wins came from the team-strength model itself (MOV) and a situational factor
the ratings can't see (rest).

---

## 4. Where it could go next

- **Poisson score layer** on top of Elo — unlocks **totals / puck line / props**
  (the product roadmap). Doesn't help moneyline, but it's the real expansion.
- **xG-based ratings** (the MoneyPuck direction) — the biggest potential core
  improvement, but a substantial build (shot pipeline), and to be measured
  skeptically given the goalie/pitcher findings.
- **Glicko-2 done properly** — principled early-season/roster-churn handling
  (our lightweight proxy overfit); modest expected gain, with an OT/draw caveat.
- **CLV tracking** — the honest measure of whether the edges are real; keep the
  model market-independent.

---

### Sources
FiveThirtyEight [methodology](https://fivethirtyeight.com/methodology/how-our-nhl-predictions-work)
and Neil Paine's [formula](https://neilpaine.substack.com/p/how-my-nhl-elo-ratings-and-forecast);
MoneyPuck [about](https://moneypuck.com/about.htm);
ML review / Weissbock ([arxiv](https://arxiv.org/pdf/2410.21484));
hockey Poisson modeling ([researchgate](https://www.researchgate.net/publication/272536864_Modeling_and_prediction_of_ice_hockey_match_results));
market combination (Benter, [oddspapi](https://oddspapi.io/blog/?p=3174));
NHL rest/travel ([RotoWire](https://www.rotowire.com/hockey/article/how-time-zone-travel-hurts-nhl-teams-11-seasons-of-data-122663),
[ESPN](https://www.espn.com/nhl/story/_/id/39512201/)).
