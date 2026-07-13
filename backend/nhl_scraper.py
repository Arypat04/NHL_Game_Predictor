"""
NHL Scraper (official NHL API)
------------------------------
Collects NHL data from the official NHL API — no website scraping. (Replaced the
old Hockey-Reference scraper.) Produces two CSVs / MongoDB collections:

  - nhl_matches_2021_2025.csv  (training data, full per-game stats)
  - nhl_matches_2026.csv       (current season schedule + results)

Data sources (all free, no scraping):
  - api-web.nhle.com/v1/standings/{date}                  → teams in a season
  - api-web.nhle.com/v1/club-schedule-season/{team}/{sid} → games, scores, times
  - api-web.nhle.com/v1/gamecenter/{id}/right-rail        → SOG, PIM, PP, faceoffs
  - api-web.nhle.com/v1/gamecenter/{id}/play-by-play      → Corsi/Fenwick/oZS%/PDO

Direct stats (SOG, PIM, PPG, PPO, FOW, FO%) and box info (GF, GA, OT) come from
the schedule + right-rail endpoints. The advanced metrics Hockey-Reference
provided (CF/CA/CF%, FF/FA/FF%, oZS%, PDO) are reconstructed from play-by-play at
5v5 (situationCode "1551"), which matches HR's values almost exactly. SHG is
counted from goals scored while shorthanded (all situations).

Performance: games are fetched concurrently (MAX_WORKERS, env-overridable via
NHL_SCRAPER_WORKERS) and each season is checkpointed to data/nhl_cache/. A run
can be interrupted and resumed — already-scraped games are skipped, so completed
historical seasons cost nothing to re-run and weekly maintenance only fetches
the current season's new games.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from dotenv import load_dotenv

from nhl_teams import ABBREV_TO_NAME, RELOCATION_MAP

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "../data")
CACHE_DIR = os.path.join(DATA_DIR, "nhl_cache")   # per-season checkpoints for resume
load_dotenv(os.path.join(BASE_DIR, "../.env"))

API_WEB = "https://api-web.nhle.com/v1"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
ET = ZoneInfo("America/New_York")

TRAINING_SEASONS = list(range(2021, 2026))
CURRENT_SEASON   = 2026

# Concurrency / resilience. The NHL API is unauthenticated and tolerant; ~12
# workers cuts a full backfill from ~80 min to a few minutes. Lower if you ever
# see 429s.
MAX_WORKERS = int(os.getenv("NHL_SCRAPER_WORKERS", "12"))
FLUSH_EVERY = 200          # checkpoint a season's cache every N games
HTTP_RETRIES = 3

# Unblocked shot attempts (Fenwick). Corsi adds blocked-shot.
SHOT_EVENTS = {"shot-on-goal", "missed-shot", "goal"}
FIVE_ON_FIVE = "1551"   # away G, away skaters, home skaters, home G


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def api_get(path: str) -> dict | None:
    """GET with retries + backoff. Safe to call from worker threads."""
    url = f"{API_WEB}{path}"
    for attempt in range(HTTP_RETRIES):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(1.5 * (attempt + 1))   # transient — back off and retry
                continue
            print(f"  ⚠ {resp.status_code}: {path}")
            return None
        except Exception as e:
            if attempt == HTTP_RETRIES - 1:
                print(f"  ⚠ request error {path}: {e}")
            time.sleep(1.0 * (attempt + 1))
    return None


def season_id(label: int) -> str:
    """Season label (e.g. 2024 = the 2023-24 season) → API season id '20232024'."""
    return f"{label - 1}{label}"


def norm(abbrev: str) -> str:
    return RELOCATION_MAP.get(abbrev, abbrev)


def to_eastern(start_utc: str) -> str:
    if not start_utc:
        return ""
    try:
        dt = datetime.strptime(start_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=ZoneInfo("UTC"))
        return dt.astimezone(ET).strftime("%I:%M %p").lstrip("0")
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Game enumeration
# ---------------------------------------------------------------------------

def season_team_abbrevs(label: int) -> list[str]:
    js = api_get(f"/standings/{label}-04-01")
    if not js:
        return []
    return [t["teamAbbrev"]["default"] for t in js.get("standings", [])]


def enumerate_games(label: int) -> dict[int, dict]:
    """Return {gameId: schedule-summary} for all regular-season games in a season."""
    sid      = season_id(label)
    abbrevs  = season_team_abbrevs(label)
    games: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(api_get, f"/club-schedule-season/{ab}/{sid}") for ab in abbrevs]
        for fut in as_completed(futures):
            js = fut.result()
            if not js:
                continue
            for g in js.get("games", []):
                if g.get("gameType") == 2:    # regular season only
                    games[g["id"]] = g        # dedup across both teams' schedules
    return games


# ---------------------------------------------------------------------------
# Per-game reconstruction (training data)
# ---------------------------------------------------------------------------

def _right_rail_stats(rr: dict) -> dict[str, tuple]:
    """category -> (awayValue, homeValue)."""
    out = {}
    for entry in rr.get("teamGameStats", []):
        out[entry.get("category")] = (entry.get("awayValue"), entry.get("homeValue"))
    return out


def _parse_pp(value) -> tuple[int, int]:
    """'1/3' -> (ppg=1, ppo=3)."""
    try:
        g, o = str(value).split("/")
        return int(g), int(o)
    except Exception:
        return 0, 0


def _as_int(value) -> int:
    """right-rail values may arrive as ints or numeric strings."""
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return 0


def _pbp_advanced(pbp: dict, home_id: int, away_id: int) -> dict[int, dict]:
    """Reconstruct 5v5 Corsi/Fenwick/zone-starts + shorthanded goals per team."""
    acc = {home_id: dict(cf=0, ff=0, sog5=0, gf5=0, oz=0, dz=0, shg=0),
           away_id: dict(cf=0, ff=0, sog5=0, gf5=0, oz=0, dz=0, shg=0)}

    for p in pbp.get("plays", []):
        t   = p.get("typeDescKey")
        d   = p.get("details", {})
        own = d.get("eventOwnerTeamId")
        if own not in acc:
            continue
        even = p.get("situationCode") == FIVE_ON_FIVE

        # Corsi / Fenwick (5v5). Every shot attempt — including blocked — is
        # owned by the shooting team (verified against Hockey-Reference).
        if even and t in SHOT_EVENTS:
            acc[own]["cf"] += 1
            acc[own]["ff"] += 1
        elif even and t == "blocked-shot":
            acc[own]["cf"] += 1

        # 5v5 shots-on-goal (incl. goals) and goals — for PDO
        if even and t in ("shot-on-goal", "goal"):
            acc[own]["sog5"] += 1
        if even and t == "goal":
            acc[own]["gf5"] += 1

        # Offensive/defensive zone faceoffs (5v5). zoneCode is relative to the
        # faceoff winner (eventOwnerTeamId).
        if even and t == "faceoff":
            z = d.get("zoneCode")
            other = home_id if own == away_id else away_id
            if z == "O":
                acc[own]["oz"]   += 1
                acc[other]["dz"] += 1
            elif z == "D":
                acc[own]["dz"]   += 1
                acc[other]["oz"] += 1

        # Shorthanded goals (all situations)
        if t == "goal":
            code = p.get("situationCode", "")
            if len(code) == 4 and code.isdigit():
                away_sk, home_sk = int(code[1]), int(code[2])
                my_sk, opp_sk = (home_sk, away_sk) if own == home_id else (away_sk, home_sk)
                if my_sk < opp_sk:
                    acc[own]["shg"] += 1
    return acc


def reconstruct_game(gid: int, summary: dict | None = None) -> list[dict]:
    # The schedule summary (scores/date/outcome) is passed in during a season
    # scrape; when called standalone (e.g. tests) fall back to the boxscore,
    # which carries the same fields.
    if summary is None:
        summary = api_get(f"/gamecenter/{gid}/boxscore")
        if not summary:
            return []
    rr  = api_get(f"/gamecenter/{gid}/right-rail")
    pbp = api_get(f"/gamecenter/{gid}/play-by-play")
    if not rr or not pbp:
        return []

    home_id = pbp["homeTeam"]["id"]
    away_id = pbp["awayTeam"]["id"]
    home_ab = norm(pbp["homeTeam"]["abbrev"])
    away_ab = norm(pbp["awayTeam"]["abbrev"])

    home_score = summary["homeTeam"].get("score")
    away_score = summary["awayTeam"].get("score")
    if home_score is None or away_score is None:
        return []
    date    = summary.get("gameDate", "")
    season  = int(summary.get("season", season_id(CURRENT_SEASON)))
    ot_type = summary.get("gameOutcome", {}).get("lastPeriodType", "REG")
    ot      = ot_type if ot_type in ("OT", "SO") else None

    rrs = _right_rail_stats(rr)
    adv = _pbp_advanced(pbp, home_id, away_id)

    def side(idx):  # idx 0 = away, 1 = home in right-rail tuples
        sog = _as_int(rrs.get("sog", (0, 0))[idx])
        pim = _as_int(rrs.get("pim", (0, 0))[idx])
        ppg, ppo = _parse_pp(rrs.get("powerPlay", ("0/0", "0/0"))[idx])
        # faceoffWins arrives as "wins/total" (e.g. "26/60")
        fow, fototal = _parse_pp(rrs.get("faceoffWins", ("0/0", "0/0"))[idx])
        fol = fototal - fow
        return dict(sog=sog, pim=pim, ppg=ppg, ppo=ppo, fow=fow, fol=fol)

    away_rr, home_rr = side(0), side(1)

    def build(team_ab, opp_ab, tid, oid, ha, gf, ga, rr_self, rr_opp):
        a, o = adv[tid], adv[oid]
        cf, ca = a["cf"], o["cf"]
        ff, fa = a["ff"], o["ff"]
        fow, fol = rr_self["fow"], rr_self["fol"]
        sh = a["gf5"] / a["sog5"] if a["sog5"] else 0
        sv = 1 - o["gf5"] / o["sog5"] if o["sog5"] else 0
        return {
            "Date": date, "Home_Away": ha, "Opp": opp_ab, "Team": team_ab,
            "Rslt": "W" if gf > ga else "L", "GF": gf, "GA": ga, "OT": ot,
            "SOG": rr_self["sog"], "PIM": rr_self["pim"],
            "PPG": rr_self["ppg"], "PPO": rr_self["ppo"], "SHG": a["shg"],
            "SOG_OPP": rr_opp["sog"], "PIM_OPP": rr_opp["pim"],
            "PPG_OPP": rr_opp["ppg"], "PPO_OPP": rr_opp["ppo"], "SHG_OPP": o["shg"],
            "FOW": fow, "FOL": fol,
            "FO%": round(fow / (fow + fol) * 100, 1) if (fow + fol) else None,
            "CF": cf, "CA": ca, "CF%": round(cf / (cf + ca) * 100, 1) if (cf + ca) else None,
            "FF": ff, "FA": fa, "FF%": round(ff / (ff + fa) * 100, 1) if (ff + fa) else None,
            "oZS%": round(a["oz"] / (a["oz"] + a["dz"]) * 100, 1) if (a["oz"] + a["dz"]) else None,
            "PDO": round((sh + sv) * 100, 1),
            "Season": season,
        }

    rows = [
        build(home_ab, away_ab, home_id, away_id, "Home", home_score, away_score, home_rr, away_rr),
        build(away_ab, home_ab, away_id, home_id, "Away", away_score, home_score, away_rr, home_rr),
    ]
    for r in rows:
        r["GameId"] = gid     # used for resume/dedup; dropped from the final CSV
    return rows


# ---------------------------------------------------------------------------
# Per-season cache (checkpoint + resume)
# ---------------------------------------------------------------------------

def _cache_path(label: int) -> str:
    return os.path.join(CACHE_DIR, f"nhl_games_{label}.csv")


def _load_cache(label: int) -> tuple[list[dict], set[int]]:
    path = _cache_path(label)
    if not os.path.exists(path):
        return [], set()
    df = pd.read_csv(path)
    if df.empty or "GameId" not in df.columns:
        return [], set()
    rows = df.to_dict("records")
    return rows, set(df["GameId"].astype(int))


def _write_cache(label: int, rows: list[dict]) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(_cache_path(label), index=False)


# ---------------------------------------------------------------------------
# Public scrape functions
# ---------------------------------------------------------------------------

def scrape_season_gamelogs(label: int, max_games: int | None = None,
                           workers: int = MAX_WORKERS) -> pd.DataFrame:
    """
    Full per-game stat rows for one season (training schema).

    Fetches games concurrently and checkpoints to a per-season cache, so a run
    can be interrupted and resumed — already-scraped games are skipped, and
    completed historical seasons cost nothing to re-run.
    """
    games = enumerate_games(label)
    ids   = sorted(games)
    if max_games:
        ids = ids[:max_games]

    rows, done = _load_cache(label)
    todo = [gid for gid in ids if gid not in done]
    print(f"  {label}: {len(ids)} games — {len(done)} cached, {len(todo)} to fetch")

    if todo:
        completed = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(reconstruct_game, gid, games[gid]): gid for gid in todo}
            for fut in as_completed(futures):
                try:
                    rows.extend(fut.result())
                except Exception as e:
                    print(f"    ⚠ game {futures[fut]} failed: {e}")
                completed += 1
                if completed % FLUSH_EVERY == 0:
                    _write_cache(label, rows)
                    print(f"    {completed}/{len(todo)} fetched (checkpointed)")
        _write_cache(label, rows)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Team", "Date"]).reset_index(drop=True)
    df["Gtm"] = df.groupby("Team").cumcount() + 1
    return df


def scrape_training_seasons(years: list[int] = TRAINING_SEASONS) -> pd.DataFrame:
    frames = []
    for year in years:
        print(f"\nScraping {year}...")
        df = scrape_season_gamelogs(year)
        if not df.empty:
            frames.append(df)
            print(f"  ✓ {year}: {len(df)} rows, {df['Team'].nunique()} teams")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_current_season(label: int = CURRENT_SEASON) -> pd.DataFrame:
    """
    Current season with RICH per-game stats for completed games (so the model's
    possession features stay high quality in-season) PLUS schedule-only rows for
    upcoming games (for the prediction game list). Completed games are
    reconstructed exactly like training; unplayed games carry Date/Time/Opponent
    only. Replaces the old schedule-only current-season file.
    """
    games    = enumerate_games(label)
    time_map = {gid: to_eastern(g.get("startTimeUTC", "")) for gid, g in games.items()}

    rich = scrape_season_gamelogs(label)          # completed games, full stats
    completed_ids: set[int] = set()
    if not rich.empty:
        rich["Season"]   = label
        rich["Time"]     = rich["GameId"].map(time_map)
        rich["Opponent"] = rich["Opp"].map(ABBREV_TO_NAME).fillna(rich["Opp"])
        completed_ids    = set(rich["GameId"].unique())

    future_rows = []
    for gid, g in games.items():
        if gid in completed_ids or g.get("gameState") in ("OFF", "FINAL"):
            continue                              # played (rich) or unreconstructable
        date    = g.get("gameDate", "")
        time_et = time_map.get(gid, "")
        home_ab = norm(g["homeTeam"]["abbrev"])
        away_ab = norm(g["awayTeam"]["abbrev"])
        for team_ab, opp_ab, ha in [(home_ab, away_ab, "Home"), (away_ab, home_ab, "Away")]:
            future_rows.append({
                "Date": date, "Time": time_et, "Home_Away": ha, "Team": team_ab,
                "Opp": opp_ab, "Opponent": ABBREV_TO_NAME.get(opp_ab, opp_ab),
                "Rslt": None, "Season": label,
            })
    future = pd.DataFrame(future_rows)
    if not future.empty:
        future["Date"] = pd.to_datetime(future["Date"])

    combined = pd.concat([rich, future], ignore_index=True).drop(columns=["GameId"], errors="ignore")
    if not combined.empty:
        combined = combined.sort_values(["Team", "Date"]).reset_index(drop=True)
        combined.insert(0, "GP", combined.groupby("Team").cumcount() + 1)
    return combined


def build_schedule_df(label: int = CURRENT_SEASON) -> pd.DataFrame:
    """Current-season schedule + results (schedule schema, no play-by-play)."""
    games = enumerate_games(label)
    rows  = []
    for gid, g in games.items():
        date    = g.get("gameDate", "")
        time_et = to_eastern(g.get("startTimeUTC", ""))
        home_ab = norm(g["homeTeam"]["abbrev"])
        away_ab = norm(g["awayTeam"]["abbrev"])
        hs, as_ = g["homeTeam"].get("score"), g["awayTeam"].get("score")
        played  = g.get("gameState") in ("OFF", "FINAL") and hs is not None and as_ is not None

        for team_ab, opp_ab, ha, gf, ga in [
            (home_ab, away_ab, "Home", hs, as_),
            (away_ab, home_ab, "Away", as_, hs),
        ]:
            rslt = ("W" if gf > ga else "L") if played else None
            rows.append({
                "Date": date, "Time": time_et, "Home_Away": ha,
                "Opponent": ABBREV_TO_NAME.get(opp_ab, opp_ab),
                "GF": gf if played else None, "GA": ga if played else None,
                "Rslt": rslt, "Season": label, "Team": team_ab,
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values(["Team", "Date"]).reset_index(drop=True)
        df.insert(0, "GP", df.groupby("Team").cumcount() + 1)
    return df


# ---------------------------------------------------------------------------
# Result generation (predictions vs actual outcomes → MongoDB)
# ---------------------------------------------------------------------------

def generate_results(current_df: pd.DataFrame) -> None:
    if not os.getenv("MONGODB_URI"):
        return

    try:
        from nhl_database import log_results, get_db
        db = get_db()
        if db is None:
            return
    except Exception as e:
        print(f"  ⚠ Could not connect to MongoDB for results: {e}")
        return

    try:
        from nhl_predictor import NHLPredictor, TEAM_NAME_TO_ABBREV
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
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=== Scraping training seasons (2021–2025) via NHL API ===")
    training_df = scrape_training_seasons()
    if not training_df.empty:
        training_df = training_df.drop(columns=["GameId"], errors="ignore")  # internal only
        path = os.path.join(DATA_DIR, "nhl_matches_2021_2025.csv")
        training_df.to_csv(path, index=False)
        print(f"\n✓ Saved {path} ({len(training_df):,} rows, {training_df['Team'].nunique()} teams)")
        if os.getenv("MONGODB_URI"):
            try:
                from nhl_database import upsert_games
                print(f"  ✓ MongoDB games: {upsert_games(training_df):,} rows upserted")
            except Exception as e:
                print(f"  ⚠ MongoDB write failed: {e}")

    print("\n=== Scraping current season (2026) via NHL API ===")
    current_df = build_current_season(CURRENT_SEASON)
    if not current_df.empty:
        path = os.path.join(DATA_DIR, "nhl_matches_2026.csv")
        current_df.to_csv(path, index=False)
        completed = current_df["Rslt"].notna().sum()
        print(f"✓ Saved {path} ({len(current_df):,} rows, {completed} completed)")
        if os.getenv("MONGODB_URI"):
            try:
                from nhl_database import upsert_games, upsert_schedule
                # rich completed current-season games → games collection, so the
                # predictor's rolling features have real possession stats in-season
                done = current_df.dropna(subset=["Rslt"])
                if not done.empty:
                    print(f"  ✓ MongoDB games (current): {upsert_games(done):,} rows upserted")
                print(f"  ✓ MongoDB schedule: {upsert_schedule(current_df):,} rows upserted")
            except Exception as e:
                print(f"  ⚠ MongoDB write failed: {e}")

        print("\n=== Generating results ===")
        try:
            generate_results(current_df)
        except Exception as e:
            print(f"  ⚠ Result generation skipped: {e}")

    print("\n=== Done ===")
