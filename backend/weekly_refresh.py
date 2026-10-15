"""
Weekly maintenance for the Render cron (also runnable by hand).

Keeps MongoDB fresh so the web app stays fast and its numbers current:
  - NHL: rebuild the CURRENT season (rich per-game stats) → upsert games +
    schedule, then log results (season accuracy on /status).
  - MLB: rebuild the CURRENT season → upsert schedule, log results.

The models are Elo (see elo.py) — a cheap one-pass rating rebuild with no
training — so nothing here trains or caches a model. (The MLB starting-pitcher
collection was dropped when the model went to a clean team-only Elo; it returns
here if/when a starting-pitcher Elo adjustment is added.)

Historical team-game data is immutable and assumed already in MongoDB from the
initial full scrape (`nhl_scraper.py` / `mlb_scraper.py`). This script only
touches the current season.

Run:  python weekly_refresh.py
"""

import os


def refresh_nhl() -> None:
    print("\n=== NHL: current season → MongoDB ===")
    import nhl_scraper as S
    from nhl_database import get_game_keys, upsert_games, upsert_schedule
    from seasons import current_season

    label = current_season("nhl")

    # Schedule + results are cheap (one API call per team, no play-by-play), so
    # rebuild them in full every run — that keeps upcoming games and any
    # corrected scores current.
    sched = S.build_schedule_df(label)
    if sched.empty:
        print("  (no current-season games available yet)")
        return
    print(f"  schedule: {upsert_schedule(sched):,} upserted")

    # Rich per-game stats are expensive (~2 API calls per game), and a played
    # game never changes — so only reconstruct games MongoDB doesn't have yet.
    fresh = S.new_completed_games(label, get_game_keys())
    if fresh.empty:
        print("  games:    already up to date")
    else:
        print(f"  games:    {upsert_games(fresh):,} upserted")

    S.generate_results(sched)
    from nhl_database import set_refresh_time
    set_refresh_time()


def refresh_mlb() -> None:
    print("\n=== MLB: current season → MongoDB ===")
    import mlb_scraper as S
    from mlb_database import upsert_schedule

    cur = S.scrape_current_season()
    if not cur.empty:
        print(f"  schedule: {upsert_schedule(cur):,} upserted")
        S.generate_results(cur)
        from mlb_database import set_refresh_time
        set_refresh_time()


if __name__ == "__main__":
    if not os.getenv("MONGODB_URI"):
        print("MONGODB_URI not set — nothing to refresh.")
    else:
        refresh_nhl()
        refresh_mlb()
        print("\n=== Weekly refresh done ===")
