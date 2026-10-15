"""
Elo model evaluation (dev tool, run by hand).

Walk-forward: every game is predicted with ratings from the games before it
(Elo.predictions), then we report accuracy / AUC / logloss / Brier per held-out
season and in aggregate, plus the current top teams as a sanity check. Offline —
reads the training CSVs.

    python elo_evaluate.py          # both sports
    python elo_evaluate.py nhl      # one
"""
import sys

import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

import elo

TRAIN_CSV = {
    "nhl": "../data/nhl_matches_train.csv",
    "mlb": "../data/mlb_matches_train.csv",
}
SCORE_COLS = {"nhl": ("GF", "GA"), "mlb": ("R", "RA")}


def evaluate(sport: str, n_test: int = 3) -> None:
    team = pd.read_csv(TRAIN_CSV[sport])
    team["Season"] = team["Season"].apply(elo.normalize_season)
    rf = elo.results_frame(team, score_cols=SCORE_COLS[sport], with_rest=True)

    preds = elo.Elo.for_sport(sport).predictions(rf)
    test_seasons = sorted(preds["Season"].unique())[-n_test:]

    print(f"\n===== {sport.upper()}  (Elo {elo.PARAMS[sport]}) =====")
    acc = auc = ll = br = 0.0
    for ts in test_seasons:
        d = preds[preds["Season"] == ts]
        y = d["home_win"]
        a = accuracy_score(y, (d["p"] >= 0.5).astype(int))
        u = roc_auc_score(y, d["p"]); g = log_loss(y, d["p"]); b = brier_score_loss(y, d["p"])
        acc += a; auc += u; ll += g; br += b
        print(f"  test {ts}: acc={a:.4f} auc={u:.4f} logloss={g:.4f} brier={b:.4f}  (n={len(d)})")
    n = len(test_seasons)
    print(f"  MEAN   : acc={acc/n:.4f} auc={auc/n:.4f} logloss={ll/n:.4f} brier={br/n:.4f}")

    ratings = elo.Elo.for_sport(sport).fit(rf)
    print("  top 8 teams:", [f"{t}:{r:.0f}" for t, r in ratings.top(8)])
    print(f"  home-win base rate: {rf['home_win'].mean():.3f}")


if __name__ == "__main__":
    for s in (sys.argv[1:] or ["nhl", "mlb"]):
        evaluate(s)
