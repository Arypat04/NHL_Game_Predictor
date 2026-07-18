"""
MLB Stats API Scraper
----------------------
Data sources:
  - /schedule         → dates, scores, results, game times
  - /teams/{id}/stats → per-game hitting + pitching logs

Produces:
  - mlb_matches_train.csv    (training window)
  - mlb_matches_current.csv  (current season)

Postponed games (codedGameState=D) are filtered out.
Same-day rainouts played later keep their Final entry.
Game times converted from UTC to Eastern (EDT = UTC-4).
"""

import os
import time
from datetime import datetime, timezone, timedelta

import requests
import pandas as pd
from dotenv import load_dotenv

from mlb_teams import TEAM_ID_TO_ABBREV, VALID_TEAM_IDS
from seasons import current_season, training_seasons

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "../data")
TRAIN_CSV   = os.path.join(DATA_DIR, "mlb_matches_train.csv")
CURRENT_CSV = os.path.join(DATA_DIR, "mlb_matches_current.csv")

load_dotenv(os.path.join(BASE_DIR, "../.env"))

API_BASE = "https://statsapi.mlb.com/api/v1"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# Rolling year windows — auto-advance with the calendar (see seasons.py)
TRAINING_SEASONS = training_seasons("mlb")
CURRENT_SEASON   = current_season("mlb")


def api_get(endpoint: str, params: dict = None) -> dict | None:
    url = f"{API_BASE}{endpoint}"
    time.sleep(0.1)
    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
        if resp.status_code == 200:
            return resp.json()
        print(f"  ⚠ {resp.status_code}: {endpoint}")
    except Exception as e:
        print(f"  ⚠ Request error: {e}")
    return None


def team_abbrev(team_id: int) -> str | None:
    return TEAM_ID_TO_ABBREV.get(team_id)


def parse_game_time(raw_dt: str) -> str:
    """
    Convert UTC gameDate string to Eastern Time.
    Baseball season (Apr-Oct) is always EDT = UTC-4.
    """
    if not raw_dt or "T" not in raw_dt:
        return ""
    try:
        utc_dt  = datetime.strptime(raw_dt, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        eastern = utc_dt - timedelta(hours=4)
        return eastern.strftime("%I:%M %p")  # e.g. "07:05 PM"
    except Exception:
        return raw_dt.split("T")[1][:5]      # fallback: raw HH:MM


def scrape_schedule(year: int, include_unfinished: bool = False) -> pd.DataFrame:
    data = api_get("/schedule", {
        "sportId":   1,
        "gameType":  "R",
        "startDate": f"{year}-03-01",
        "endDate":   f"{year}-11-30",
    })

    if not data or "dates" not in data:
        return pd.DataFrame()

    rows = []
    for date_entry in data["dates"]:
        for game in date_entry.get("games", []):
            away_data = game["teams"]["away"]
            home_data = game["teams"]["home"]

            away_id = away_data["team"]["id"]
            home_id = home_data["team"]["id"]

            if away_id not in VALID_TEAM_IDS or home_id not in VALID_TEAM_IDS:
                continue
            if game.get("gameType") != "R":
                continue

            # Skip postponed — same-day rainouts played later keep their Final entry
            if game.get("status", {}).get("codedGameState") == "D":
                continue

            away_abbr  = team_abbrev(away_id)
            home_abbr  = team_abbrev(home_id)
            game_date  = game.get("officialDate", "")
            game_time  = parse_game_time(game.get("gameDate", ""))
            status     = game.get("status", {}).get("abstractGameState", "")

            away_score = away_data.get("score")
            home_score = home_data.get("score")

            home_rslt = away_rslt = None
            if status == "Final" and home_score is not None and away_score is not None:
                if home_score > away_score:
                    home_rslt, away_rslt = "W", "L"
                elif home_score < away_score:
                    home_rslt, away_rslt = "L", "W"
                else:
                    continue  # tie

            if not include_unfinished and home_rslt is None:
                continue

            rows.append({
                "Date": game_date, "Time": game_time,
                "Home_Away": "Home", "Team": home_abbr, "Opp": away_abbr,
                "Rslt": home_rslt, "R": home_score, "RA": away_score,
            })
            rows.append({
                "Date": game_date, "Time": game_time,
                "Home_Away": "Away", "Team": away_abbr, "Opp": home_abbr,
                "Rslt": away_rslt, "R": away_score, "RA": home_score,
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["Date"] = pd.to_datetime(df["Date"])
    return df


def scrape_team_game_log(team_id: int, year: int) -> pd.DataFrame:
    def fetch_log(group: str) -> list[dict]:
        data = api_get(f"/teams/{team_id}/stats", {
            "stats": "gameLog", "group": group,
            "season": year, "gameType": "R",
        })
        if not data:
            return []
        return data.get("stats", [{}])[0].get("splits", [])

    hitting_rows  = {}
    pitching_rows = {}

    for split in fetch_log("hitting"):
        date = split.get("date", "")
        s    = split.get("stat", {})
        hitting_rows[date] = {
            "H": s.get("hits"), "HR": s.get("homeRuns"),
            "BB": s.get("baseOnBalls"), "SO": s.get("strikeOuts"),
            "SB": s.get("stolenBases"), "LOB": s.get("leftOnBase"),
            "RBI": s.get("rbi"), "2B": s.get("doubles"), "3B": s.get("triples"),
            "OBP": s.get("obp"), "SLG": s.get("slg"), "OPS": s.get("ops"),
            "AVG": s.get("avg"), "BABIP": s.get("babip"),
        }

    for split in fetch_log("pitching"):
        date = split.get("date", "")
        s    = split.get("stat", {})
        ip   = float(s.get("inningsPitched", 0) or 0)
        k    = float(s.get("strikeOuts",     0) or 0)
        bb   = float(s.get("baseOnBalls",    0) or 0)
        hr   = float(s.get("homeRuns",       0) or 0)
        pitching_rows[date] = {
            "P_ERA": s.get("era"), "P_WHIP": s.get("whip"),
            "P_IP": ip, "P_K": k, "P_BB": bb, "P_HR": hr,
            "P_K9":   round(k  / ip * 9, 3) if ip > 0 else None,
            "P_BB9":  round(bb / ip * 9, 3) if ip > 0 else None,
            "P_HR9":  round(hr / ip * 9, 3) if ip > 0 else None,
            "P_H9":   s.get("hitsPer9Inn"),
            "P_K_BB": round(k / bb, 3) if bb > 0 else None,
            "P_PC":   s.get("numberOfPitches"),
        }

    all_dates = set(hitting_rows) | set(pitching_rows)
    merged = [{"Date": d, **hitting_rows.get(d, {}), **pitching_rows.get(d, {})}
              for d in sorted(all_dates)]

    df = pd.DataFrame(merged)
    if not df.empty:
        df["Date"] = pd.to_datetime(df["Date"])
    return df


def scrape_season(year: int, include_unfinished: bool = False) -> pd.DataFrame:
    print("  Fetching schedule...")
    schedule = scrape_schedule(year, include_unfinished=include_unfinished)
    if schedule.empty:
        print(f"  ⚠ No schedule data for {year}")
        return pd.DataFrame()

    completed = schedule["Rslt"].notna().sum()
    print(f"  Schedule: {len(schedule)} rows ({completed} with results)")

    print("  Fetching game logs for 30 teams...")
    log_frames = []
    for team_id, abbrev in TEAM_ID_TO_ABBREV.items():
        log = scrape_team_game_log(team_id, year)
        if log.empty:
            print(f"    ⚠ No game log for {abbrev} ({year})")
            continue
        log["Team"] = abbrev
        log_frames.append(log)

    if not log_frames:
        print(f"  ⚠ No game log data for {year}")
        return schedule

    logs   = pd.concat(log_frames, ignore_index=True)
    print(f"  Game logs: {len(logs)} rows across {logs['Team'].nunique()} teams")

    merged = schedule.merge(logs, on=["Team", "Date"], how="left")
    merged["Season"] = year
    merged["RD"] = (
        pd.to_numeric(merged["R"], errors="coerce") -
        pd.to_numeric(merged["RA"], errors="coerce")
    )
    return merged


def scrape_training_seasons(years: list[int] = TRAINING_SEASONS) -> pd.DataFrame:
    all_frames = []
    for year in years:
        print(f"\nScraping {year}...")
        df = scrape_season(year, include_unfinished=False)
        if not df.empty:
            all_frames.append(df)
            print(f"  ✓ {year}: {len(df)} rows, {df['Team'].nunique()} teams")

    if not all_frames:
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)
    missing  = set(TEAM_ID_TO_ABBREV.values()) - set(combined["Team"].unique())
    if missing:
        print(f"\n⚠ Missing teams: {sorted(missing)}")
    else:
        print("\n✓ All 30 teams present")
    return combined


def scrape_current_season(year: int = CURRENT_SEASON) -> pd.DataFrame:
    print(f"\nScraping {year}...")
    df = scrape_season(year, include_unfinished=True)
    if not df.empty:
        completed = df["Rslt"].notna().sum() // 2
        scheduled = df["Rslt"].isna().sum()  // 2
        print(f"  ✓ {year}: {completed} completed, {scheduled} scheduled")
    return df


def generate_results(current_df: pd.DataFrame) -> None:
    if not os.getenv("MONGODB_URI"):
        return

    try:
        from mlb_database import log_results, get_db
        if get_db() is None:
            return
    except Exception as e:
        print(f"  ⚠ MongoDB connection failed: {e}")
        return

    try:
        from mlb_predictor import MLBPredictor
        predictor = MLBPredictor()
    except Exception as e:
        print(f"  ⚠ Could not load predictor: {e}")
        return

    completed = current_df.dropna(subset=["Rslt"]).copy()
    completed["Date"] = pd.to_datetime(completed["Date"], errors="coerce")
    completed = completed.dropna(subset=["Date"])

    dates = sorted(completed["Date"].dt.strftime("%Y-%m-%d").unique().tolist())
    print(f"\n  Generating results for {len(dates)} dates...")

    total = correct = 0
    for date_str in dates:
        preds = predictor.predict_date(date_str)
        if preds is None:
            continue

        pred_lookup = {
            "_".join(sorted([r["Home"], r["Away"]])): r
            for _, r in preds.iterrows()
        }

        day   = completed[completed["Date"].dt.strftime("%Y-%m-%d") == date_str]
        homes = day[day["Home_Away"] == "Home"]
        results = []

        for _, g in homes.iterrows():
            home, away, rslt = g["Team"], g["Opp"], g["Rslt"]
            if rslt not in ("W", "L"):
                continue
            key  = "_".join(sorted([home, away]))
            pred = pred_lookup.get(key)
            if pred is None:
                continue
            actual    = home if rslt == "W" else away
            predicted = pred["Predicted_Winner"]
            ok        = actual == predicted
            results.append({
                "Time":             g.get("Time", ""),
                "Status":           "Final",
                "Away":             away,
                "Home":             home,
                "Away_Score":       int(g["RA"]) if pd.notna(g.get("RA")) else 0,
                "Home_Score":       int(g["R"])  if pd.notna(g.get("R"))  else 0,
                "Actual_Winner":    actual,
                "Predicted_Winner": predicted,
                "Home_Win_Prob":    pred["Home_Win_Prob"],
                "Away_Win_Prob":    pred["Away_Win_Prob"],
                "Correct":          ok,
            })
            if ok:
                correct += 1
            total += 1

        if results:
            from mlb_database import log_results
            log_results(date_str, results)

    if total:
        print(f"  ✓ {total:,} games, {correct/total:.1%} accuracy")
    else:
        print("  ⚠ No results generated")


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"=== Scraping training seasons {TRAINING_SEASONS[0]}–{TRAINING_SEASONS[-1]} ===")
    training_df = scrape_training_seasons()

    if not training_df.empty:
        training_df.to_csv(TRAIN_CSV, index=False)
        print(f"\n✓ Saved {TRAIN_CSV} ({len(training_df):,} rows, {training_df['Team'].nunique()} teams)")
        if os.getenv("MONGODB_URI"):
            try:
                from mlb_database import upsert_games
                n = upsert_games(training_df)
                print(f"  ✓ MongoDB mlb_games: {n:,} rows upserted")
            except Exception as e:
                print(f"  ⚠ MongoDB write failed: {e}")

    print(f"\n=== Scraping current season ({CURRENT_SEASON}) ===")
    current_df = scrape_current_season()

    if not current_df.empty:
        current_df.to_csv(CURRENT_CSV, index=False)
        print(f"✓ Saved {CURRENT_CSV} ({len(current_df):,} rows, {current_df['Team'].nunique()} teams)")
        if os.getenv("MONGODB_URI"):
            try:
                from mlb_database import upsert_schedule
                n = upsert_schedule(current_df)
                print(f"  ✓ MongoDB mlb_schedule: {n:,} rows upserted")
            except Exception as e:
                print(f"  ⚠ MongoDB write failed: {e}")

    print("\n=== Generating results ===")
    generate_results(current_df)

    print("\n=== Done ===")
