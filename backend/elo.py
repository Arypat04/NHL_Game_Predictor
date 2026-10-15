"""
Elo rating model for game win probability.

One rating per team, updated after every game. Before a game, the home win
probability is a logistic function of the rating gap plus a home-ice/field bump:

    P(home win) = 1 / (1 + 10 ** ((R_away - R_home - HFA) / 400))

After the game each team's rating moves toward the result by K * (actual -
expected). Between seasons ratings regress toward the mean so last year's
standings don't dominate a new season.

Why this exists: an out-of-sample test showed a plain Elo matches or beats the
RandomForest-on-rolling-features stack (clearly for NHL, a tie for MLB) — with
none of the training cost. Elo is a dict of floats updated in one cheap pass, so
it sidesteps the free-tier memory/cold-start problems entirely: no model file,
no training, no Mongo model cache.

The engine is sport-agnostic; per-sport constants live in PARAMS. It operates on
a "results frame": one row per completed game with columns
Date / Season / home / away / home_win, which `results_frame` builds from the
team-game logs the predictors already load.
"""
from __future__ import annotations

import pandas as pd

BASE_RATING = 1500.0

# Chosen by out-of-sample validation (tune on one season, confirm on the next).
# k = update speed, hfa = home advantage in Elo points, regress = fraction of a
# team's rating carried past the between-season pull toward the mean.
PARAMS = {
    "nhl": {"k": 6.0, "hfa": 35.0, "regress": 0.70},
    "mlb": {"k": 4.0, "hfa": 24.0, "regress": 0.70},
}


def normalize_season(s) -> int:
    """Season may be a 4-digit label (2025) or an 8-digit API id (20242025) → the
    label. The same game can arrive under both forms, so normalizing keeps the
    season window and the between-season regression consistent."""
    s = int(s)
    return int(str(s)[4:]) if s > 9999 else s


class Elo:
    def __init__(self, k: float, hfa: float, regress: float, base: float = BASE_RATING):
        self.k = k
        self.hfa = hfa
        self.regress = regress
        self.base = base
        self.rating: dict[str, float] = {}

    @classmethod
    def for_sport(cls, sport: str) -> "Elo":
        return cls(**PARAMS[sport.lower()])

    def _p_home(self, rh: float, ra: float) -> float:
        return 1.0 / (1.0 + 10 ** ((ra - rh - self.hfa) / 400))

    def _run(self, games: pd.DataFrame, record: bool) -> list[float]:
        """One chronological pass, rebuilding ratings from scratch. Each game is
        scored with the ratings BEFORE its own update (so a recorded prediction
        never sees its own result). `games` must be pre-sorted by Date."""
        self.rating = {}
        cur_season = None
        out: list[float] = []
        for home, away, y, season in zip(
            games["home"], games["away"], games["home_win"], games["Season"], strict=False
        ):
            if season != cur_season:                       # season rollover
                cur_season = season
                for t in self.rating:
                    self.rating[t] = self.base + self.regress * (self.rating[t] - self.base)
            rh = self.rating.get(home, self.base)
            ra = self.rating.get(away, self.base)
            p = self._p_home(rh, ra)
            if record:
                out.append(p)
            self.rating[home] = rh + self.k * (y - p)
            self.rating[away] = ra + self.k * ((1 - y) - (1 - p))
        return out

    def fit(self, games: pd.DataFrame) -> "Elo":
        """Rebuild ratings from a results frame (Date / Season / home / away /
        home_win). Leaves the final ratings in `self.rating`."""
        self._run(games.sort_values("Date", kind="stable"), record=False)
        return self

    def predictions(self, games: pd.DataFrame) -> pd.DataFrame:
        """As-of P(home win) for every game in a results frame — each game scored
        on the ratings before it. Returns the frame with a `p` column. Used for
        walk-forward evaluation (see elo_evaluate.py)."""
        games = games.sort_values("Date", kind="stable").reset_index(drop=True)
        out = games.copy()
        out["p"] = self._run(games, record=True)
        return out

    def prob_home(self, home: str, away: str) -> float:
        """P(home win) from current ratings (unknown team → base rating)."""
        return self._p_home(self.rating.get(home, self.base), self.rating.get(away, self.base))

    def top(self, n: int = 10) -> list[tuple[str, float]]:
        return sorted(self.rating.items(), key=lambda kv: kv[1], reverse=True)[:n]


def results_frame(team_df: pd.DataFrame) -> pd.DataFrame:
    """Team-game logs → one row per completed game (home perspective).

    Takes the home-side rows, so each game appears once; `home_win` is the home
    team's result. Rows without a win/loss result (unplayed) are dropped."""
    df = team_df[team_df["Home_Away"] == "Home"].copy()
    df = df.dropna(subset=["Rslt"])
    df = df.rename(columns={"Team": "home", "Opp": "away"})
    df["home_win"] = (df["Rslt"] == "W").astype(int)
    df["Date"] = pd.to_datetime(df["Date"])
    return df[["Date", "Season", "home", "away", "home_win"]].sort_values("Date")
