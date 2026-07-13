"""
NHL matchup feature engineering.

One row per game:
    target  = home win
    features = home team rolling + away team rolling (offense + possession +
               special teams) + context (rest, day-of-week)

Goalie and individual-scorer features were tested and dropped — they add no
measurable lift because team shot-share (Corsi/Fenwick) already prices them in.

`team_rolling` is shared by training (build_matchup) and live prediction (the
predictor looks up each team's latest rolling row as of the game date). All
rolling is leakage-safe (closed="left").
"""

import os

import pandas as pd

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "../data")
TRAIN_CSV = os.path.join(DATA_DIR, "nhl_matches_2021_2025.csv")

# Feature set + windows: an exhaustive search (2040 combos, 5-season walk-forward
# CV; see model_search.py) showed windows and feature sets barely move accuracy
# (all top configs sit within the ~0.017 season-to-season noise). This is the
# best-accuracy config; kept lean deliberately since richer sets don't help.
TEAM_ROLL_COLS = [
    "GF", "GA", "SOG", "SOG_OPP", "CF%", "FF%", "PDO", "oZS%",
    "PPG", "PPO", "PIM", "FO%",
]
TEAM_WINDOWS = [5, 10, 20]
CONTEXT_COLS = ["dow", "home_rest", "away_rest"]


def normalize_season(s) -> int:
    """Season may be a label (2025) or 8-digit API id (20242025) → label."""
    s = int(s)
    return int(str(s)[4:]) if s > 9999 else s


def team_rolling(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Per-team rolling means (closed-left) over TEAM_WINDOWS. Returns the
    augmented team-game frame and the list of rolling feature names."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    cols = [c for c in TEAM_ROLL_COLS if c in df.columns]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    gmed = df[cols].median()
    df[cols] = df.groupby("Team")[cols].transform(lambda s: s.fillna(s.median())).fillna(gmed)

    out, names = [], []
    for _, g in df.groupby("Team"):
        g = g.sort_values("Date").copy()
        for w in TEAM_WINDOWS:
            r = g[cols].rolling(w, closed="left").mean()
            for c in cols:
                g[f"avg{w}_{c}"] = r[c]
        out.append(g)
    rolled = pd.concat(out, ignore_index=True)
    for w in TEAM_WINDOWS:
        names.extend([f"avg{w}_{c}" for c in cols])
    for c in cols:
        for i in range(1, len(TEAM_WINDOWS)):
            big, small = f"avg{TEAM_WINDOWS[i]}_{c}", f"avg{TEAM_WINDOWS[i-1]}_{c}"
            rolled[big] = rolled[big].fillna(rolled[small])
    return rolled, names


def feature_cols(team_feats: list[str]) -> list[str]:
    return ([f"home_{c}" for c in team_feats]
            + [f"away_{c}" for c in team_feats]
            + CONTEXT_COLS)


def build_matchup(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build the one-row-per-game training matrix from team-game logs."""
    raw = raw.copy()
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw["Season"] = raw["Season"].apply(normalize_season)

    rolled, team_feats = team_rolling(raw)
    keep = ["Date", "Season", "Team", "Opp", "Home_Away", "Rslt"] + team_feats
    home = rolled[rolled["Home_Away"] == "Home"][keep].copy()
    away = rolled[rolled["Home_Away"] == "Away"][keep].copy()
    home = home.rename(columns={c: f"home_{c}" for c in team_feats})
    away = away.rename(columns={**{c: f"away_{c}" for c in team_feats},
                                "Team": "Opp", "Opp": "Team"})

    game = home.merge(
        away[["Date", "Team", "Opp"] + [f"away_{c}" for c in team_feats]],
        on=["Date", "Team", "Opp"], how="inner",
    ).drop_duplicates(["Date", "Team", "Opp"])
    game = game.rename(columns={"Team": "home", "Opp": "away"})
    game["home_win"] = (game["Rslt"] == "W").astype(int)

    game["dow"] = game["Date"].dt.dayofweek
    game = game.sort_values(["home", "Date"])
    game["home_rest"] = game.groupby("home")["Date"].diff().dt.days.fillna(2).clip(0, 7)
    game = game.sort_values(["away", "Date"])
    game["away_rest"] = game.groupby("away")["Date"].diff().dt.days.fillna(2).clip(0, 7)
    game = game.sort_values("Date").reset_index(drop=True)

    cols = feature_cols(team_feats)
    game[cols] = game[cols].fillna(game[cols].median())
    return game, cols


if __name__ == "__main__":
    df = pd.read_csv(TRAIN_CSV)
    game, cols = build_matchup(df)
    print(f"games: {len(game)}  features: {len(cols)}  home win rate: {game['home_win'].mean():.3f}")
