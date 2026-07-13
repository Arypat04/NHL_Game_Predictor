"""
NHL team constants — single source of truth.

Imported by nhl_predictor.py, nhl_scraper.py and main.py so a team
abbreviation only ever needs to change in one place.
"""

# Full team name (as it appears on Hockey-Reference / the NHL API) → abbreviation.
# Some teams have multiple historical/accented names that map to the same abbrev
# (e.g. "Montréal Canadiens" and "Montreal Canadiens"; Utah's rename).
TEAM_NAME_TO_ABBREV: dict[str, str] = {
    "Anaheim Ducks":         "ANA",
    "Arizona Coyotes":       "ARI",
    "Boston Bruins":         "BOS",
    "Buffalo Sabres":        "BUF",
    "Calgary Flames":        "CGY",
    "Carolina Hurricanes":   "CAR",
    "Chicago Blackhawks":    "CHI",
    "Colorado Avalanche":    "COL",
    "Columbus Blue Jackets": "CBJ",
    "Dallas Stars":          "DAL",
    "Detroit Red Wings":     "DET",
    "Edmonton Oilers":       "EDM",
    "Florida Panthers":      "FLA",
    "Los Angeles Kings":     "LAK",
    "Minnesota Wild":        "MIN",
    "Montreal Canadiens":    "MTL",
    "Montréal Canadiens":    "MTL",
    "Nashville Predators":   "NSH",
    "New Jersey Devils":     "NJD",
    "New York Islanders":    "NYI",
    "New York Rangers":      "NYR",
    "Ottawa Senators":       "OTT",
    "Philadelphia Flyers":   "PHI",
    "Pittsburgh Penguins":   "PIT",
    "San Jose Sharks":       "SJS",
    "Seattle Kraken":        "SEA",
    "St. Louis Blues":       "STL",
    "Tampa Bay Lightning":   "TBL",
    "Toronto Maple Leafs":   "TOR",
    "Utah Hockey Club":      "UTA",
    "Utah Mammoth":          "UTA",
    "Vancouver Canucks":     "VAN",
    "Vegas Golden Knights":  "VGK",
    "Washington Capitals":   "WSH",
    "Winnipeg Jets":         "WPG",
}

# Hockey-Reference still serves some teams under their old URL abbreviation;
# normalize those to the current abbrev (Arizona → Utah, Vegas VEG → VGK).
RELOCATION_MAP: dict[str, str] = {
    "ARI": "UTA",
    "VEG": "VGK",
}

VALID_ABBREVS: set[str] = set(TEAM_NAME_TO_ABBREV.values())

# Reverse map (abbrev → a canonical full name). Used by the API scraper to write
# the schedule's "Opponent" column with a name that maps back via TEAM_NAME_TO_ABBREV.
ABBREV_TO_NAME: dict[str, str] = {}
for _name, _abbrev in TEAM_NAME_TO_ABBREV.items():
    ABBREV_TO_NAME.setdefault(_abbrev, _name)
