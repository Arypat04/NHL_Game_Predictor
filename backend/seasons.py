"""
Season configuration — one source of truth for which seasons to scrape/train.

`current_season(sport)` asks the league API which season is being played *right
now* (authoritative, to the day — no reliance on a hardcoded changeover date),
and falls back to a date heuristic when offline or for sports without an API
wired yet. The result is cached per process. Everything else (the rolling
training window) derives from it, so the year ranges roll forward automatically.

  MLB : single calendar year (starts ~March)  — label = that year
  NHL : Oct–Jun (spans two years)             — label = the END year (25-26 → 2026)
  NBA : Oct–Jun                               — label = the END year
  NFL : Sep–Feb                               — label = the START year (2025 season)

Note: predictors do NOT call this (they derive their window straight from the
data they load, so the backend/tests stay offline). Only the scrapers, which are
always online, resolve the current season from the API.
"""

from datetime import date

import requests

# start_month = first month of the season; "label" = which calendar year names it
_SPORT = {
    "nhl": {"start_month": 10, "label": "end"},
    "mlb": {"start_month": 3,  "label": "start"},
    "nba": {"start_month": 10, "label": "end"},
    "nfl": {"start_month": 9,  "label": "start"},
}

TRAIN_WINDOW = 5   # rolling number of completed seasons used for training
_HEADERS = {"User-Agent": "Mozilla/5.0"}
_cache: dict[str, int] = {}


def _heuristic_current_season(sport: str, today: date | None = None) -> int:
    cfg = _SPORT[sport]
    today = today or date.today()
    start_year = today.year if today.month >= cfg["start_month"] else today.year - 1
    return start_year + 1 if cfg["label"] == "end" else start_year


def _api_current_season(sport: str) -> int | None:
    """The season being played 'now' per the league API; None on any failure."""
    try:
        if sport == "nhl":
            r = requests.get("https://api-web.nhle.com/v1/standings/now",
                             headers=_HEADERS, timeout=5)
            sid = r.json()["standings"][0]["seasonId"]   # e.g. 20252026
            return int(str(sid)[4:])                       # end year → label
        if sport == "mlb":
            r = requests.get("https://statsapi.mlb.com/api/v1/seasons",
                             params={"sportId": 1}, headers=_HEADERS, timeout=5)
            return int(r.json()["seasons"][0]["seasonId"])
    except Exception:
        return None
    return None   # nba / nfl: no API wired yet → heuristic fallback


def current_season(sport: str, today: date | None = None) -> int:
    """Authoritative current season (API-backed, cached). Pass `today` to force
    the deterministic date heuristic — used by tests and offline callers."""
    if today is not None:
        return _heuristic_current_season(sport, today)
    if sport not in _cache:
        _cache[sport] = _api_current_season(sport) or _heuristic_current_season(sport)
    return _cache[sport]


def training_seasons(sport: str, window: int = TRAIN_WINDOW,
                     today: date | None = None) -> list[int]:
    cur = current_season(sport, today)
    return list(range(cur - window, cur))


def all_seasons(sport: str, window: int = TRAIN_WINDOW,
                today: date | None = None) -> list[int]:
    cur = current_season(sport, today)
    return list(range(cur - window, cur + 1))


if __name__ == "__main__":
    for s in _SPORT:
        api = _api_current_season(s)
        print(f"{s}: current={current_season(s)} (api={api}, "
              f"heuristic={_heuristic_current_season(s)})  train={training_seasons(s)}")
