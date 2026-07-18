"""
Weekly maintenance for the Render cron (also runnable by hand).

Keeps MongoDB fresh so the web app stays fast and its numbers current:
  - NHL: rebuild the CURRENT season (rich per-game stats) → upsert games +
    schedule, then log results (season accuracy on /status).
  - MLB: rebuild the CURRENT season → upsert schedule, log results.
  - MLB pitcher starts: (re)collect every season in the training window into the
    `mlb_sp_starts` collection, so the web app READS pitcher form from MongoDB
    instead of re-collecting thousands of API calls on every startup.

Historical team-game data is immutable and assumed already in MongoDB from the
initial full scrape (`nhl_scraper.py` / `mlb_scraper.py`). This script only
touches the current season + the pitcher-start collection.

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


def refresh_mlb() -> None:
    print("\n=== MLB: current season + pitcher starts → MongoDB ===")
    import mlb_scraper as S
    from mlb_database import sp_start_counts, upsert_schedule, upsert_sp_starts
    from mlb_pitchers import collect_season_starts
    from seasons import all_seasons, current_season

    # Pitcher starts FIRST — so the predictor that generate_results loads reads
    # its pitcher form from MongoDB instead of re-collecting it from the API
    # (that would double the API work every run).
    this_year = current_season("mlb")
    stored    = sp_start_counts()
    total     = 0
    for year in all_seasons("mlb"):
        # A finished season's starts are immutable; re-collecting all six every
        # week was ~5/6 wasted work (and API calls) for identical data.
        if year != this_year and stored.get(year):
            print(f"  {year}: {stored[year]:,} starts already stored — skipping (season complete)")
            continue
        starts = collect_season_starts(year)
        if not starts.empty:
            total += upsert_sp_starts(starts)
    print(f"  mlb_sp_starts: {total:,} upserted")

    cur = S.scrape_current_season()
    if not cur.empty:
        print(f"  schedule: {upsert_schedule(cur):,} upserted")
        S.generate_results(cur)


if __name__ == "__main__":
    if not os.getenv("MONGODB_URI"):
        print("MONGODB_URI not set — nothing to refresh.")
    else:
        refresh_nhl()
        refresh_mlb()
        print("\n=== Weekly refresh done ===")
