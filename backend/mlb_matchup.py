"""
MLB matchup feature engineering.

One row per game:
    target  = home win
    features = home team offense/staff rolling, away team offense/staff rolling,
               home starting-pitcher rolling, away starting-pitcher rolling,
               + context (rest, day-of-week)

`team_rolling` and the cached pitcher-form frame are shared by training
(build_matchup) and live prediction (the predictor looks up each team's latest
offense rolling and each probable pitcher's latest form as of the game date).
All rolling is leakage-safe.
"""

import os

import pandas as pd

from mlb_pitchers import (
    pivot_form_to_game, season_pitcher_form, sp_feature_cols,
)

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "../data")
CACHE_DIR = os.path.join(DATA_DIR, "mlb_cache")
TRAIN_CSV = os.path.join(DATA_DIR, "mlb_matches_2021_2025.csv")

# Feature set + windows from exhaustive search (model_search.py, 4-season CV):
# team OFFENSE only — the team-level pitching stats (P_ERA/P_WHIP/...) are
# redundant once the actual STARTING PITCHER's form is a feature. Longer windows
# (10/20/40) beat short ones because baseball team stats stabilize slowly.
TEAM_ROLL_COLS = ["RD", "R", "OBP", "SLG", "BABIP", "HR"]
TEAM_WINDOWS = [10, 20, 40]
CONTEXT_COLS = ["dow", "home_rest", "away_rest"]


def normalize_season(s) -> int:
    s = int(s)
    return int(str(s)[4:]) if s > 9999 else s


def team_rolling(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
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
    sp = sp_feature_cols()
    return ([f"home_{c}" for c in team_feats] + [f"away_{c}" for c in team_feats]
            + [f"home_{c}" for c in sp] + [f"away_{c}" for c in sp]
            + CONTEXT_COLS)


# ---------------------------------------------------------------------------
# Pitcher form (cached per season)
# ---------------------------------------------------------------------------

def pitcher_form_cached(year: int) -> pd.DataFrame:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"sp_form_{year}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path); df["date"] = pd.to_datetime(df["date"]); return df
    print(f"  collecting starting-pitcher data for {year}...")
    df = season_pitcher_form(year)
    if not df.empty:
        df.to_csv(path, index=False)
    return df


def sp_game_table(year: int) -> pd.DataFrame:
    form = pitcher_form_cached(year)
    return pivot_form_to_game(form) if not form.empty else form


# ---------------------------------------------------------------------------
# Build matchup dataset
# ---------------------------------------------------------------------------

def build_matchup(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    raw = raw.copy()
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw["Season"] = raw["Season"].apply(normalize_season)
    seasons = sorted(raw["Season"].unique())

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

    # starting-pitcher features
    sp_all = pd.concat([sp_game_table(y) for y in seasons], ignore_index=True)
    sp_cols = [f"home_{c}" for c in sp_feature_cols()] + [f"away_{c}" for c in sp_feature_cols()]
    sp_join = sp_all[["date", "home_team", "away_team"] + sp_cols].rename(
        columns={"date": "Date", "home_team": "home", "away_team": "away"}
    ).drop_duplicates(["Date", "home", "away"])
    game = game.merge(sp_join, on=["Date", "home", "away"], how="left")

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
    print(f"SP coverage: {game['home_SP5_ERA'].notna().mean():.3f}")
