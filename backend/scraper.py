"""
NHL Game Log Scraper
--------------------
Scrapes game log data from Hockey-Reference and writes directly to MongoDB.

Two operations:
  scrape_training_seasons() — historical data (2021-2025), goes to games collection
  scrape_current_season()   — current season, games go to games collection,
                              schedule (unplayed) goes to schedule collection

Run weekly to keep data fresh. The model retrains automatically on next
server start when MongoDB data is newer than models.pkl.
"""

import os
import time
import random
from datetime import datetime

import requests
import pandas as pd
from bs4 import BeautifulSoup, Comment
from io import StringIO
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

BASE_URL = "https://www.hockey-reference.com"
RELOCATION_MAP = {"ARI": "UTA",
                  "VEG":"VGK"}
TRAINING_SEASONS = list(range(2021, 2026))
CURRENT_SEASON = 2026


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def polite_get(url: str, pause: tuple = (2, 5)) -> requests.Response:
    time.sleep(random.uniform(*pause))
    return requests.get(url, headers=HEADERS)


def rename_opp_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(
        columns={
            col: col.replace(".1", "_OPP")
            for col in df.columns
            if str(col).endswith(".1")
        }
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
        top, bottom = str(col[0]), str(col[1])
        top_unnamed = "Unnamed" in top
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
    """Derive W/L result from GF and GA for current season games."""
    df = df.copy()
    df["Rslt"] = None
    if "GF" in df.columns and "GA" in df.columns:
        gf = pd.to_numeric(df["GF"], errors="coerce")
        ga = pd.to_numeric(df["GA"], errors="coerce")
        mask = gf.notna() & ga.notna()
        df.loc[mask & (gf > ga), "Rslt"] = "W"
        df.loc[mask & (gf < ga), "Rslt"] = "L"
    return df


def get_team_urls(year: int) -> list[str]:
    standings_url = f"{BASE_URL}/leagues/NHL_{year}.html"
    print(f"Fetching standings for {year}...")
    resp = polite_get(standings_url, pause=(1, 3))
    soup = BeautifulSoup(resp.text, "html.parser")
    comments = soup.find_all(string=lambda t: isinstance(t, Comment))
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
# Training data scraper (2021-2025)
# ---------------------------------------------------------------------------

def scrape_training_seasons(years: list[int] = TRAINING_SEASONS) -> pd.DataFrame:
    all_frames = []

    for year in years:
        team_urls = get_team_urls(year)

        for team_url in team_urls:
            base = team_url.replace(".html", "")
            team = team_abbrev_from_url(team_url)
            gamelog_url = f"{base}_gamelog.html"

            print(f"  Fetching {team} ({year})")
            resp = polite_get(gamelog_url)

            if resp.status_code != 200:
                print(f"  ⚠ Skipping {team} — HTTP {resp.status_code}")
                continue

            try:
                tables = pd.read_html(
                    StringIO(resp.text),
                    attrs={"id": "team_games"},
                    flavor="lxml",
                )
                df = tables[0]
            except ValueError:
                print(f"  ⚠ No table found for {team} ({year})")
                continue

            df = flatten_multiindex(df)
            df = fix_home_away(df)
            df = strip_header_rows(df)
            df["Season"] = year
            df["Team"] = team
            all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True)
    combined = rename_opp_columns(combined)
    return combined


# ---------------------------------------------------------------------------
# Current season scraper (2026)
# ---------------------------------------------------------------------------

def scrape_current_season(year: int = CURRENT_SEASON) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns two DataFrames:
      completed — games with results (goes to games collection)
      scheduled — future games (goes to schedule collection)
    """
    team_urls = get_team_urls(year)
    all_frames = []

    for team_url in team_urls:
        base = team_url.replace(".html", "")
        team = team_abbrev_from_url(team_url)
        gamelog_url = f"{base}_games.html"

        print(f"  Fetching {team} ({year})")
        resp = polite_get(gamelog_url)

        if resp.status_code != 200:
            print(f"  ⚠ Skipping {team} — HTTP {resp.status_code}")
            continue

        try:
            tables = pd.read_html(
                StringIO(resp.text),
                attrs={"id": "games"},
                flavor="lxml",
            )
            df = tables[0]
        except ValueError:
            print(f"  ⚠ No table found for {team} ({year})")
            continue

        df = flatten_multiindex(df)
        df = fix_home_away(df)
        df = strip_header_rows(df)
        df = derive_rslt(df)
        df["Season"] = year
        df["Team"] = team
        all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True)
    combined = rename_opp_columns(combined)

    # split into completed and scheduled
    completed = combined[combined["Rslt"].notna()].copy()
    scheduled = combined[combined["Rslt"].isna()].copy()

    return completed, scheduled


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from database import get_db, upsert_games, upsert_schedule

    db = get_db()
    if db is None:
        print("❌ Cannot connect to MongoDB — check MONGODB_URI in .env")
        exit(1)

    print("=== Scraping training seasons (2021-2025) ===")
    training_df = scrape_training_seasons()
    count = upsert_games(training_df)
    print(f"✓ Upserted {count:,} training game rows to MongoDB\n")

    print("=== Scraping current season (2026) ===")
    completed_df, scheduled_df = scrape_current_season()
    count_games = upsert_games(completed_df)
    count_schedule = upsert_schedule(scheduled_df)
    print(f"✓ Upserted {count_games:,} completed 2026 games to games collection")
    print(f"✓ Upserted {count_schedule:,} upcoming games to schedule collection\n")

    print("=== Done ===")
    print(f"Total training games in DB: {db['games'].count_documents({})}")
    print(f"Total scheduled games in DB: {db['schedule'].count_documents({})}")