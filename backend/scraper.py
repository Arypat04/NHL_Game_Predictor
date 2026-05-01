"""
NHL Game Log Scraper
--------------------
Scrapes game log data for all NHL teams from Hockey-Reference.com.

Produces two CSVs:
  - nhl_matches_2021_2025.csv  (training data, full stats)
  - nhl_matches_2026.csv       (current season, full stats + Rslt column)

If MONGODB_URI is set:
  - Writes game logs to MongoDB games/schedule collections
  - Runs predictor against every completed 2026 game and stores results
  - MongoDB results collection stays in sync after every weekly scrape
"""

import os
import sys
import time
import random

import requests
import pandas as pd
from bs4 import BeautifulSoup, Comment
from io import StringIO
from dotenv import load_dotenv



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data")

load_dotenv(os.path.join(BASE_DIR, "../.env"))
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

BASE_URL = "https://www.hockey-reference.com"

RELOCATION_MAP = {
    "ARI": "UTA",
    "VEG": "VGK",
}

TRAINING_SEASONS = list(range(2021, 2026))
CURRENT_SEASON   = 2026


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def polite_get(url: str, pause: tuple = (2, 5)) -> requests.Response:
    time.sleep(random.uniform(*pause))
    return requests.get(url, headers=HEADERS)


def rename_opp_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(
        columns={col: col.replace(".1", "_OPP") for col in df.columns if str(col).endswith(".1")}
    )


def fix_home_away(df: pd.DataFrame) -> pd.DataFrame:
    col = df.columns[3]
    df[col] = df[col].fillna("Home").replace("@", "Away")
    return df.rename(columns={col: "Home_Away"})


def flatten_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    new_cols = []
    for i, col in enumerate(df.columns):
        top, bottom    = str(col[0]), str(col[1])
        top_unnamed    = "Unnamed" in top
        bottom_unnamed = "Unnamed" in bottom or not bottom
        if not top_unnamed and top == "Opponent" and not bottom_unnamed:
            new_cols.append(f"{bottom}_OPP")
        elif not bottom_unnamed:
            new_cols.append(bottom)
        elif not top_unnamed:
            new_cols.append(top)
        else:
            new_cols.append(f"Col_{i}")
    df.columns = new_cols
    return df


def strip_header_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df.iloc[:, 0] != "Rk"]
    return df.dropna(subset=[df.columns[0]])


def derive_rslt(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Rslt"] = None
    if "GF" in df.columns and "GA" in df.columns:
        gf   = pd.to_numeric(df["GF"], errors="coerce")
        ga   = pd.to_numeric(df["GA"], errors="coerce")
        mask = gf.notna() & ga.notna()
        df.loc[mask & (gf > ga), "Rslt"] = "W"
        df.loc[mask & (gf < ga), "Rslt"] = "L"
    return df


def get_team_urls(year: int) -> list[str]:
    standings_url = f"{BASE_URL}/leagues/NHL_{year}.html"
    print(f"Fetching standings for {year}...")
    resp  = polite_get(standings_url, pause=(1, 3))
    soup  = BeautifulSoup(resp.text, "html.parser")
    comments     = soup.find_all(string=lambda t: isinstance(t, Comment))
    comment_soup = BeautifulSoup("".join(comments), "html.parser")
    standings_table = comment_soup.select("table")[0]
    links = [
        a.get("href")
        for a in standings_table.find_all("a")
        if a.get("href") and "/teams/" in a.get("href")
    ]
    return [f"{BASE_URL}{link}" for link in links]


def team_abbrev_from_url(url: str) -> str:
    abbrev = url.split("/teams/")[1].split("/")[0]
    return RELOCATION_MAP.get(abbrev, abbrev)


# ---------------------------------------------------------------------------
# Training data scraper (2021–2025)
# ---------------------------------------------------------------------------

def scrape_training_seasons(years: list[int] = TRAINING_SEASONS) -> pd.DataFrame:
    all_frames = []

    for year in years:
        team_urls = get_team_urls(year)
        for team_url in team_urls:
            base        = team_url.replace(".html", "")
            team        = team_abbrev_from_url(team_url)
            gamelog_url = f"{base}_gamelog.html"

            print(f"  Fetching {team} ({year})")
            resp = polite_get(gamelog_url)

            if resp.status_code != 200:
                print(f"  ⚠ Skipping {team} — HTTP {resp.status_code}")
                continue

            try:
                tables = pd.read_html(StringIO(resp.text), attrs={"id": "team_games"}, flavor="lxml")
                df     = tables[0]
            except ValueError:
                print(f"  ⚠ No table found for {team} ({year})")
                continue

            df = flatten_multiindex(df)
            df = fix_home_away(df)
            df = strip_header_rows(df)
            df["Season"] = year
            df["Team"]   = team
            all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True)
    combined = rename_opp_columns(combined)
    return combined


# ---------------------------------------------------------------------------
# Current season scraper (2026)
# ---------------------------------------------------------------------------

def scrape_current_season(year: int = CURRENT_SEASON) -> pd.DataFrame:
    team_urls  = get_team_urls(year)
    all_frames = []

    for team_url in team_urls:
        base        = team_url.replace(".html", "")
        team        = team_abbrev_from_url(team_url)
        gamelog_url = f"{base}_games.html"

        print(f"  Fetching {team} ({year})")
        resp = polite_get(gamelog_url)

        if resp.status_code != 200:
            print(f"  ⚠ Skipping {team} — HTTP {resp.status_code}")
            continue

        try:
            tables = pd.read_html(StringIO(resp.text), attrs={"id": "games"}, flavor="lxml")
            df     = tables[0]
        except ValueError:
            print(f"  ⚠ No table found for {team} ({year})")
            continue

        df = flatten_multiindex(df)
        df = fix_home_away(df)
        df = strip_header_rows(df)
        df = derive_rslt(df)
        df["Season"] = year
        df["Team"]   = team
        all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True)
    combined = rename_opp_columns(combined)
    return combined


# ---------------------------------------------------------------------------
# Results generation from scraped data
# ---------------------------------------------------------------------------

def generate_results(current_df: pd.DataFrame) -> None:
    if not os.getenv("MONGODB_URI"):
        return

    try:
        from database import log_results, get_db
        db = get_db()
        if db is None:
            return
    except Exception as e:
        print(f"  ⚠ Could not connect to MongoDB for results: {e}")
        return

    try:
        from predictor import NHLPredictor, TEAM_NAME_TO_ABBREV
        predictor = NHLPredictor()
    except Exception as e:
        print(f"  ⚠ Could not load predictor: {e}")
        return

    completed = current_df.dropna(subset=["Rslt"]).copy()
    completed["Date"] = pd.to_datetime(completed["Date"])
    completed["GF"]   = pd.to_numeric(completed["GF"], errors="coerce")
    completed["GA"]   = pd.to_numeric(completed["GA"], errors="coerce")

    # map opponent full names to abbreviations once
    completed["Opp_abbrev"] = completed["Opponent"].map(TEAM_NAME_TO_ABBREV)

    dates = sorted(completed["Date"].dt.strftime("%Y-%m-%d").unique().tolist())
    print(f"\n  Generating results for {len(dates)} completed game dates...")

    total_games   = 0
    total_correct = 0

    for date_str in dates:
        predictions_df = predictor.predict_date(date_str)
        if predictions_df is None:
            continue

        pred_lookup = {}
        for _, row in predictions_df.iterrows():
            key = "_".join(sorted([row["Home"], row["Away"]]))
            pred_lookup[key] = row

        day_games  = completed[completed["Date"].dt.strftime("%Y-%m-%d") == date_str]
        home_games = day_games[day_games["Home_Away"] == "Home"]

        results = []
        for _, game_row in home_games.iterrows():
            home = game_row["Team"]

            # match using abbreviation instead of full name
            away_rows = day_games[
                (day_games["Home_Away"] == "Away") &
                (day_games["Opp_abbrev"] == home)
            ]
            if away_rows.empty:
                continue
            away = away_rows.iloc[0]["Team"]

            rslt = game_row["Rslt"]
            if rslt not in ("W", "L"):
                continue

            actual_winner = home if rslt == "W" else away
            home_score    = int(game_row["GF"]) if pd.notna(game_row.get("GF")) else 0
            away_score    = int(game_row["GA"]) if pd.notna(game_row.get("GA")) else 0

            key  = "_".join(sorted([home, away]))
            pred = pred_lookup.get(key)
            if pred is None:
                continue

            predicted_winner = pred["Predicted_Winner"]
            correct          = actual_winner == predicted_winner

            results.append({
                "Time":             pred["Time"],
                "Status":           "Final",
                "Away":             away,
                "Home":             home,
                "Away_Score":       away_score,
                "Home_Score":       home_score,
                "Actual_Winner":    actual_winner,
                "Predicted_Winner": predicted_winner,
                "Home_Win_Prob":    pred["Home_Win_Prob"],
                "Away_Win_Prob":    pred["Away_Win_Prob"],
                "Correct":          correct,
            })

            if correct:
                total_correct += 1
            total_games += 1

        if results:
            log_results(date_str, results)

    if total_games > 0:
        accuracy = total_correct / total_games
        print(f"  ✓ Results stored: {total_games:,} games, {accuracy:.1%} accuracy")
    else:
        print("  ⚠ No results could be generated")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Scraping training seasons (2021–2025) ===")
    training_df = scrape_training_seasons()
    out_path = os.path.join(DATA_DIR, "nhl_matches_2021_2025.csv")
    training_df.to_csv(out_path, index=False)
    print(f"✓ Saved {out_path} ({len(training_df):,} rows, {training_df['Team'].nunique()} teams)\n")

    # write training data to MongoDB if configured
    if os.getenv("MONGODB_URI"):
        try:
            from database import upsert_games
            n = upsert_games(training_df)
            print(f"  ✓ MongoDB games: {n:,} rows upserted")
        except Exception as e:
            print(f"  ⚠ MongoDB write failed: {e}")

    print("\n=== Scraping current season (2026) ===")
    current_df = scrape_current_season()
    out_path = os.path.join(DATA_DIR, "nhl_matches_2026.csv")
    current_df.to_csv(out_path, index=False)
    print(f"✓ Saved {out_path} ({len(current_df):,} rows, {current_df['Team'].nunique()} teams)")

    # write schedule to MongoDB if configured
    if os.getenv("MONGODB_URI"):
        try:
            from database import upsert_schedule
            n = upsert_schedule(current_df)
            print(f"  ✓ MongoDB schedule: {n:,} rows upserted")
        except Exception as e:
            print(f"  ⚠ MongoDB write failed: {e}")

    # generate and store results in MongoDB
    print("\n=== Generating results ===")
    generate_results(current_df)

    print("\n=== Done ===")
    rslt_count = current_df["Rslt"].notna().sum() if "Rslt" in current_df.columns else 0
    print(f"Completed 2026 games: {rslt_count:,}")