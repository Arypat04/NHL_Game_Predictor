# Product

<!-- impeccable:product-schema 1 -->

<!-- Captured from extensive prior work on this codebase in-session rather than a
cold interview; every fact below is confirmed from the code/behavior. Flag
anything that has changed. -->

## Platform

web

## Users

Sports bettors who want a data-driven edge on game outcomes — the kind of user
who already reads box scores and shops lines, not a casual fan. They arrive with
a question in mind ("who should I bet tonight, and is there value?") and leave
once they've scanned today's slate. Secondary: the owner/operator (the builder)
checking model health.

## Product Purpose

Line Lab predicts moneyline outcomes for NHL and MLB games and compares its own
win probabilities against the bookmaker's implied odds, surfacing the gaps as
"edges." Success is a bettor trusting the number enough to act on it: a clear
read on who the model favors, how confidently, and where the market disagrees.

## Positioning

Not a tout service and not a raw stats site. The mechanism a neighbor can't
truthfully copy: a transparent, leakage-safe matchup model (both teams' rolling
form; the starting pitcher as the lever in baseball) whose probabilities are
calibrated and then held up against the live market — so the product's unit of
value is an *edge* (model prob − implied prob), not a hot take.

## Operating Context

Checked in short sessions, typically the day of or day before games, often on a
phone. The core loop: pick a sport, pick a date, scan predictions and edges,
filter to a confidence threshold, and (for past dates) see how the model did.
Odds come from a free tier with a monthly request cap, so edges can be
temporarily unavailable — a real state the UI must handle gracefully, not an
error. The season rolls over automatically; NHL and MLB have offseasons where a
date legitimately has no games.

## Capabilities and Constraints

- Sports: NHL and MLB today; NBA and NFL are on the roadmap (the UI should not
  hardcode "two sports"). Bet type: moneyline today; player props are planned
  (there is already a disabled "Player Props — Soon" affordance).
- Surfaces per sport/date: **Predictions** (model pick, home/away win %,
  confidence), **Edges** (model vs. market, the value column), **Results** (past
  dates: predicted vs. actual, correct/incorrect), and a **status bar** (season
  accuracy, total predictions, last trained, odds-API health).
- A min-confidence filter (currently 50–75%) trims both predictions and edges.
- Backend is FastAPI + MongoDB; the free host spins down, so the app can return
  a 503 "warming up" state the UI must show as transient, not broken.
- Numbers shown: probabilities and confidence as percentages; accuracy as a
  season figure (acknowledged in-house to be slightly optimistic vs. true
  unseen CV).

## Brand Commitments

- Name: **Line Lab** (current logo treatment sets "Line" + accented "Lab").
- Honest-by-construction voice: the product's credibility rests on not
  overclaiming. No fabricated guarantees, no "lock of the day" hype. Confidence
  and edges are shown as what they are — probabilities, not promises.

## Evidence on Hand

- Real model output and live data via the backend API (predictions, edges,
  results, status). No testimonials, user counts, or performance guarantees
  exist — future work must not fabricate them.
- Real season accuracy figures (NHL ~58% in-app / MLB ~54%); these are genuine
  and must not be inflated.

## Product Principles

- **The edge is the product.** The model vs. market gap is the reason to be
  here; it should never be buried beside the raw pick.
- **Confidence is a signal, shown honestly.** Make certainty legible (a 71% pick
  should look different from a 51% coin-flip) without dressing a probability up
  as a sure thing.
- **Scannable in one sitting.** A bettor checks the slate fast; hierarchy and
  density serve the scan, on phone first.
- **Degrade with dignity.** Warming-up, no-games, and odds-quota-exhausted are
  normal states, designed for — not bugs to apologize for.
- **Built to add sports and bet types.** NBA/NFL and player props are coming;
  structure anticipates them.

## Accessibility & Inclusion

No formal standard established yet. Practical requirement: the interface must
stay legible and usable on a phone, and must not encode the win/loss and
correct/incorrect meaning in color alone (color-blind-safe), since red/green
outcome signaling is the obvious trap for this category.
