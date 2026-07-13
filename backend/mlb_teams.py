"""
MLB team constants — single source of truth.

Imported by mlb_predictor.py, mlb_scraper.py and main.py so a team
abbreviation only ever needs to change in one place.

Team IDs are authoritative from the MLB Stats API (/teams?sportId=1).
Note: Athletics is "ATH" (team ID 133), not OAK.
"""

# MLB Stats API team id → abbreviation.
TEAM_ID_TO_ABBREV: dict[int, str] = {
    108: "LAA", 109: "ARI", 110: "BAL", 111: "BOS", 112: "CHC",
    113: "CIN", 114: "CLE", 115: "COL", 116: "DET", 117: "HOU",
    118: "KCR", 119: "LAD", 120: "WSN", 121: "NYM", 133: "ATH",
    134: "PIT", 135: "SDP", 136: "SEA", 137: "SFG", 138: "STL",
    139: "TBR", 140: "TEX", 141: "TOR", 142: "MIN", 143: "PHI",
    144: "ATL", 145: "CHW", 146: "MIA", 147: "NYY", 158: "MIL",
}

# Full team name (as returned by The Odds API) → abbreviation.
TEAM_NAME_TO_ABBREV: dict[str, str] = {
    "Los Angeles Angels":   "LAA", "Arizona Diamondbacks": "ARI",
    "Baltimore Orioles":    "BAL", "Boston Red Sox":       "BOS",
    "Chicago Cubs":         "CHC", "Cincinnati Reds":      "CIN",
    "Cleveland Guardians":  "CLE", "Colorado Rockies":     "COL",
    "Detroit Tigers":       "DET", "Houston Astros":       "HOU",
    "Kansas City Royals":   "KCR", "Los Angeles Dodgers":  "LAD",
    "Washington Nationals": "WSN", "New York Mets":        "NYM",
    "Athletics":            "ATH", "Pittsburgh Pirates":   "PIT",
    "San Diego Padres":     "SDP", "Seattle Mariners":     "SEA",
    "San Francisco Giants": "SFG", "St. Louis Cardinals":  "STL",
    "Tampa Bay Rays":       "TBR", "Texas Rangers":        "TEX",
    "Toronto Blue Jays":    "TOR", "Minnesota Twins":      "MIN",
    "Philadelphia Phillies": "PHI", "Atlanta Braves":      "ATL",
    "Chicago White Sox":    "CHW", "Miami Marlins":        "MIA",
    "New York Yankees":     "NYY", "Milwaukee Brewers":    "MIL",
}

# Used to filter out invalid team ids/abbrevs. Do NOT use startswith("T")
# to filter — it incorrectly drops valid teams TOR, TBR, TEX.
VALID_TEAM_IDS: set[int] = set(TEAM_ID_TO_ABBREV.keys())
VALID_ABBREVS:  set[str] = set(TEAM_ID_TO_ABBREV.values())
