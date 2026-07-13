"""
MLB starting-pitcher data collector.

Baseball outcomes are driven heavily by the starting pitcher, which the
team-level rolling stats can't capture. This module pulls per-pitcher game logs
from the MLB Stats API and builds leakage-safe rolling "form" stats over each
pitcher's prior N starts, then pivots to a per-game table with home_SP_* and
away_SP_* features.

Sources:
  - /stats?stats=season&group=pitching         → enumerate season's starters
  - /people/{id}/stats?stats=gameLog&group=pitching → per-start logs (joinable
    to games via gamePk; carries team/opponent ids + isHome)
  - /schedule?hydrate=probablePitcher          → probable starters for upcoming
    games (used at prediction time)

Rolling rate stats are computed from aggregated counting totals over the prior N
starts (shifted by one to exclude the current start) — i.e. ERA = ΣER/ΣIP*9 —
which is more stable than averaging single-game ratios.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

from mlb_teams import TEAM_ID_TO_ABBREV

API = "https://statsapi.mlb.com/api/v1"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

MAX_WORKERS  = int(os.getenv("MLB_SCRAPER_WORKERS", "12"))
HTTP_RETRIES = 3

# Rolling windows over a pitcher's prior starts.
SP_WINDOWS = [3, 5, 10]

# Counting stats accumulated per start; rates are derived from their sums.
COUNT_COLS = ["outs", "ER", "BB", "SO", "H", "HR", "BF"]


def api_get(path: str, params: dict | None = None) -> dict | None:
    url = f"{API}{path}"
    for attempt in range(HTTP_RETRIES):
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=20)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(1.5 * (attempt + 1))
                continue
            return None
        except Exception:
            time.sleep(1.0 * (attempt + 1))
    return None


def ip_to_outs(ip) -> int:
    """MLB innings-pitched '5.2' = 5 innings + 2 outs → 17 outs."""
    try:
        whole, _, frac = str(ip).partition(".")
        return int(whole) * 3 + (int(frac) if frac else 0)
    except (ValueError, TypeError):
        return 0


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def season_starter_ids(year: int) -> list[int]:
    """All pitcher ids with at least one start that season."""
    ids, offset = [], 0
    while True:
        js = api_get("/stats", {
            "stats": "season", "group": "pitching", "season": year,
            "sportId": 1, "gameType": "R", "playerPool": "all",
            "limit": 100, "offset": offset,
        })
        splits = js["stats"][0]["splits"] if js and js.get("stats") else []
        if not splits:
            break
        for sp in splits:
            if (sp["stat"].get("gamesStarted") or 0) > 0:
                ids.append(sp["player"]["id"])
        if len(splits) < 100:
            break
        offset += 100
    return ids


def pitcher_starts(pid: int, year: int) -> list[dict]:
    js = api_get(f"/people/{pid}/stats", {
        "stats": "gameLog", "group": "pitching", "season": year, "gameType": "R",
    })
    splits = js["stats"][0]["splits"] if js and js.get("stats") else []
    out = []
    for sp in splits:
        st = sp["stat"]
        if (st.get("gamesStarted") or 0) < 1:
            continue                      # bullpen appearance — skip
        out.append({
            "pid":      pid,
            "date":     sp.get("date"),
            "gamePk":   sp.get("game", {}).get("gamePk"),
            "team_id":  sp.get("team", {}).get("id"),
            "opp_id":   sp.get("opponent", {}).get("id"),
            "isHome":   sp.get("isHome"),
            "season":   year,
            "outs":     ip_to_outs(st.get("inningsPitched", "0")),
            "ER":       st.get("earnedRuns", 0) or 0,
            "BB":       st.get("baseOnBalls", 0) or 0,
            "SO":       st.get("strikeOuts", 0) or 0,
            "H":        st.get("hits", 0) or 0,
            "HR":       st.get("homeRuns", 0) or 0,
            "BF":       st.get("battersFaced", 0) or 0,
        })
    return out


def collect_season_starts(year: int, workers: int = MAX_WORKERS) -> pd.DataFrame:
    ids = season_starter_ids(year)
    print(f"  {year}: {len(ids)} starting pitchers")
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(pitcher_starts, pid, year): pid for pid in ids}
        for fut in as_completed(futures):
            try:
                rows.extend(fut.result())
            except Exception as e:
                print(f"    ⚠ pitcher {futures[fut]} failed: {e}")
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


# ---------------------------------------------------------------------------
# Rolling form (leakage-safe) + per-game pivot
# ---------------------------------------------------------------------------

def _derive_rates(df: pd.DataFrame, w: int) -> None:
    ip = df[f"s{w}_outs"] / 3.0
    df[f"SP{w}_ERA"]  = (df[f"s{w}_ER"] / ip * 9).where(ip > 0)
    df[f"SP{w}_WHIP"] = ((df[f"s{w}_BB"] + df[f"s{w}_H"]) / ip).where(ip > 0)
    df[f"SP{w}_K9"]   = (df[f"s{w}_SO"] / ip * 9).where(ip > 0)
    df[f"SP{w}_BB9"]  = (df[f"s{w}_BB"] / ip * 9).where(ip > 0)
    df[f"SP{w}_HR9"]  = (df[f"s{w}_HR"] / ip * 9).where(ip > 0)
    df[f"SP{w}_K_BB"] = (df[f"s{w}_SO"] / df[f"s{w}_BB"]).where(df[f"s{w}_BB"] > 0,
                                                                df[f"s{w}_SO"])
    df[f"SP{w}_IPps"] = (ip / w)          # avg innings per start (durability)


def add_rolling_form(df: pd.DataFrame, windows: list[int] = SP_WINDOWS) -> pd.DataFrame:
    """Per pitcher, rolling sums over the PRIOR `w` starts (shifted to exclude
    the current start), then derive rate stats from those sums."""
    df = df.sort_values(["pid", "date"]).reset_index(drop=True)
    g = df.groupby("pid")
    for w in windows:
        for col in COUNT_COLS:
            df[f"s{w}_{col}"] = g[col].transform(
                lambda s: s.shift(1).rolling(w, min_periods=1).sum()
            )
        _derive_rates(df, w)
    return df


def sp_feature_cols(windows: list[int] = SP_WINDOWS) -> list[str]:
    rates = ["ERA", "WHIP", "K9", "BB9", "HR9", "K_BB", "IPps"]
    return [f"SP{w}_{r}" for w in windows for r in rates]


def season_pitcher_form(year: int, workers: int = MAX_WORKERS) -> pd.DataFrame:
    """Per-pitcher start log with leakage-safe rolling form (+ team abbrev).
    This is the source for both the per-game pivot (training) and predict-time
    lookups (a probable pitcher's latest form as of a date)."""
    starts = collect_season_starts(year, workers=workers)
    if starts.empty:
        return starts
    starts = add_rolling_form(starts)
    starts["team"] = starts["team_id"].map(TEAM_ID_TO_ABBREV)
    return starts


def pivot_form_to_game(form: pd.DataFrame) -> pd.DataFrame:
    """Pitcher form rows → one row per game with home_SP_* / away_SP_*."""
    cols = ["gamePk", "date", "pid", "team"] + sp_feature_cols()
    home = form[form["isHome"] == True][cols].copy()    # noqa: E712
    away = form[form["isHome"] == False][cols].copy()   # noqa: E712
    home = home.rename(columns={c: f"home_{c}" for c in cols if c not in ("gamePk", "date")})
    away = away.rename(columns={c: f"away_{c}" for c in cols if c not in ("gamePk", "date")})
    home = home.drop_duplicates("gamePk")
    away = away.drop_duplicates("gamePk").drop(columns=["date"])
    return home.merge(away, on="gamePk", how="outer")


def per_game_sp_table(year: int, workers: int = MAX_WORKERS) -> pd.DataFrame:
    """One row per game with home_SP_* / away_SP_* rolling features."""
    form = season_pitcher_form(year, workers=workers)
    return pivot_form_to_game(form) if not form.empty else form


if __name__ == "__main__":
    import sys
    yr = int(sys.argv[1]) if len(sys.argv) > 1 else 2024
    g = per_game_sp_table(yr)
    print(f"games with SP features: {len(g)}")
    print(g.head().to_string())
