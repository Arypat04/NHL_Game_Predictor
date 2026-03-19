"""
NHL Game Log Scraper
--------------------
Scrapes game log data for all NHL teams from Hockey-Reference.com.

Produces two CSVs:
  - nhl_matches_2021_2025.csv  (training data, full stats)
  - nhl_matches_2026.csv       (current season, full stats)

Column rename (.1 → _OPP) is applied once here at save time so all
downstream code receives clean column names.
"""

import os
import time
import random

import requests
import pandas as pd
from bs4 import BeautifulSoup, Comment
from io import StringIO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data")

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

RELOCATION_MAP = {"ARI": "UTA"}

TRAINING_SEASONS = list(range(2021, 2026))
CURRENT_SEASON = 2026


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def polite_get(url: str, pause: tuple = (2, 5)) -> requests.Response:
    time.sleep(random.uniform(*pause))
    return requests.get(url, headers=HEADERS)


def rename_opp_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Safety net rename — flatten_multiindex handles most cases directly."""
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
    """Collapse two-row headers. Opponent stats get _OPP suffix directly."""
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
# Training data scraper (2021–2025)
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

def scrape_current_season(year: int = CURRENT_SEASON) -> pd.DataFrame:
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
        df["Season"] = year
        df["Team"] = team
        all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True)
    combined = rename_opp_columns(combined)
    return combined


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Scraping training seasons (2021–2025) ===")
    training_df = scrape_training_seasons()
    out_path = os.path.join(DATA_DIR, "nhl_matches_2021_2025.csv")
    training_df.to_csv(out_path, index=False)
    print(f"✓ Saved {out_path} ({len(training_df):,} rows, {training_df['Team'].nunique()} teams)\n")

    print("=== Scraping current season (2026) ===")
    current_df = scrape_current_season()
    out_path = os.path.join(DATA_DIR, "nhl_matches_2026.csv")
    current_df.to_csv(out_path, index=False)
    print(f"✓ Saved {out_path} ({len(current_df):,} rows, {current_df['Team'].nunique()} teams)\n")

    print("=== Done ===")
    print(f"Training columns : {training_df.columns.tolist()}")
    print(f"Current  columns : {current_df.columns.tolist()}")