"""
FBRef data scraper.

Pulls match schedules, team match logs, and advanced stats directly
from FBRef HTML tables.  No external API wrapper needed — uses
requests + BeautifulSoup with automatic rate limiting.

FBRef quirk
-----------
Most stat tables are embedded inside HTML comments (<!-- ... -->) to
deter naive scrapers.  We extract them by searching comment nodes with
BeautifulSoup's Comment type.

Rate limiting
-------------
FBRef allows roughly 1 request per 3-4 seconds from a single IP.
All requests go through `_get_soup()` which enforces a minimum gap.
The default delay is 4s; use `delay=6` if you start seeing 429s.

Competitions supported
----------------------
Top domestic leagues (8):
  Premier League, La Liga, Bundesliga, Serie A, Ligue 1,
  Eredivisie, Primeira Liga, Super Lig

UEFA competitions (3):
  Champions League, Europa League, Conference League

Data returned
-------------
- `load_schedule(league, season)`     → match results + xG
- `load_team_match_log(team_id, season)` → per-team detailed stats
- `load_league_stats(league, season, stat_type)` → aggregate team stats

All outputs normalised to our schema:
  date, home_team, away_team, home_goals, away_goals
  + xg_home, xg_away (when available)
  + possession_home, possession_away, shots_home, shots_away, etc.
"""

from __future__ import annotations

import re
import time
import logging
from typing import Optional
from io import StringIO
from functools import lru_cache

import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup, Comment

logger = logging.getLogger(__name__)


# ── Competition registry ───────────────────────────────────────────────────────

COMPETITIONS: dict[str, dict] = {
    # Domestic leagues
    "ENG-PL":   {"name": "Premier League",       "fbref_id": 9,   "country": "England"},
    "ESP-LL":   {"name": "La Liga",               "fbref_id": 12,  "country": "Spain"},
    "GER-BL":   {"name": "Bundesliga",            "fbref_id": 20,  "country": "Germany"},
    "ITA-SA":   {"name": "Serie A",               "fbref_id": 11,  "country": "Italy"},
    "FRA-L1":   {"name": "Ligue 1",               "fbref_id": 13,  "country": "France"},
    "NED-ED":   {"name": "Eredivisie",            "fbref_id": 23,  "country": "Netherlands"},
    "POR-PL":   {"name": "Primeira Liga",         "fbref_id": 32,  "country": "Portugal"},
    "TUR-SL":   {"name": "Süper Lig",             "fbref_id": 26,  "country": "Turkey"},
    "BEL-PD":   {"name": "Belgian Pro League",    "fbref_id": 37,  "country": "Belgium"},
    "SCO-PL":   {"name": "Scottish Premiership",  "fbref_id": 40,  "country": "Scotland"},
    # UEFA competitions
    "UEFA-CL":  {"name": "Champions League",      "fbref_id": 8,   "country": "Europe"},
    "UEFA-EL":  {"name": "Europa League",          "fbref_id": 19,  "country": "Europe"},
    "UEFA-ECL": {"name": "Conference League",      "fbref_id": 882, "country": "Europe"},
}

# Mapping from our football-data.co.uk codes to FBRef codes
FDCO_TO_FBREF: dict[str, str] = {
    "E0": "ENG-PL", "SP1": "ESP-LL", "D1": "GER-BL", "I1": "ITA-SA",
    "F1": "FRA-L1", "N1": "NED-ED", "P1": "POR-PL",  "T1": "TUR-SL",
    "B1": "BEL-PD",
}

# Season formats: most FBRef leagues use "2023-24", a few use "2024"
_SPLIT_SEASON_LEAGUES = {9, 12, 20, 11, 13, 23, 32, 26, 37, 40, 8, 19, 882}

FBREF_BASE = "https://fbref.com"


# ── HTTP layer ─────────────────────────────────────────────────────────────────

class FBRefScraper:
    """
    FBRef data scraper with rate limiting, caching, and retry logic.

    Parameters
    ----------
    delay      : Seconds to wait between requests (default 4).
    cache_dir  : Directory to cache downloaded HTML (default None = no cache).
    user_agent : HTTP User-Agent string.
    """

    _HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; soccer-research-bot/1.0; "
            "+https://github.com/meteor21/MCQ-Capstone)"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    def __init__(
        self,
        delay: float = 4.0,
        cache_dir: Optional[str] = None,
        retries: int = 3,
        sr_username: Optional[str] = None,
        sr_password: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        delay        : Seconds between requests. Subscribers can use 2.0;
                       anonymous users should use 4.0+.
        cache_dir    : Path to cache downloaded HTML (strongly recommended
                       to avoid re-downloading on re-runs).
        sr_username  : sports-reference.com subscriber email.
        sr_password  : sports-reference.com subscriber password.
                       With an active subscription the server enforces softer
                       rate limits (~1 req/2s vs ~1 req/4s anonymous).
        """
        self.delay = delay
        self.cache_dir = cache_dir
        self.retries = retries
        self._last_request: float = 0.0
        self._session = requests.Session()
        self._session.headers.update(self._HEADERS)
        self._logged_in: bool = False

        if sr_username and sr_password:
            self._login(sr_username, sr_password)

        if cache_dir:
            import os
            os.makedirs(cache_dir, exist_ok=True)

    def _login(self, username: str, password: str) -> None:
        """
        Log into sports-reference.com to activate subscriber session.
        This stores the session cookie, enabling higher rate limits and
        access to subscriber-only data across all subsequent requests.
        """
        login_url = "https://stathead.com/users/login.cgi"
        payload = {
            "username": username,
            "password": password,
            "login":    "1",
        }
        try:
            resp = self._session.post(login_url, data=payload, timeout=20)
            if resp.status_code == 200 and "logout" in resp.text.lower():
                self._logged_in = True
                logger.info("Logged in to sports-reference.com")
            else:
                logger.warning(
                    "Login attempt made but could not confirm success. "
                    "Proceeding with anonymous session."
                )
        except requests.RequestException as e:
            logger.warning(f"Login failed: {e}. Proceeding anonymously.")

    def _cache_path(self, url: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        import os, hashlib
        key = hashlib.md5(url.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{key}.html")

    def _get_html(self, url: str) -> str:
        """Fetch URL with rate limiting, caching, and retries."""
        # Check cache
        cache_path = self._cache_path(url)
        if cache_path:
            import os
            if os.path.exists(cache_path):
                logger.debug(f"Cache hit: {url}")
                with open(cache_path, "r", encoding="utf-8") as f:
                    return f.read()

        # Rate limit
        elapsed = time.time() - self._last_request
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)

        for attempt in range(self.retries):
            try:
                resp = self._session.get(url, timeout=30)
                self._last_request = time.time()

                if resp.status_code == 429:
                    wait = 60 * (attempt + 1)
                    logger.warning(f"Rate limited. Waiting {wait}s …")
                    time.sleep(wait)
                    continue
                if resp.status_code == 404:
                    raise ValueError(f"FBRef 404: {url}")
                resp.raise_for_status()

                html = resp.text
                if cache_path:
                    with open(cache_path, "w", encoding="utf-8") as f:
                        f.write(html)
                return html

            except requests.RequestException as e:
                if attempt == self.retries - 1:
                    raise
                wait = 2 ** attempt
                logger.warning(f"Request failed ({e}). Retry in {wait}s …")
                time.sleep(wait)

        raise RuntimeError(f"All retries failed for {url}")

    def get_soup(self, url: str) -> BeautifulSoup:
        html = self._get_html(url)
        return BeautifulSoup(html, "html.parser")

    # ── Table parsing ────────────────────────────────────────────────────────

    @staticmethod
    def _extract_commented_tables(soup: BeautifulSoup) -> list[BeautifulSoup]:
        """
        FBRef hides many tables inside HTML comments.
        Extract all commented-out <table> elements.
        """
        commented = []
        for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
            inner = BeautifulSoup(comment, "html.parser")
            for table in inner.find_all("table"):
                commented.append(table)
        return commented

    @staticmethod
    def _table_to_df(table: BeautifulSoup) -> pd.DataFrame:
        """Convert a BeautifulSoup <table> element to a DataFrame."""
        html_str = str(table)
        dfs = pd.read_html(StringIO(html_str), header=0)
        if not dfs:
            return pd.DataFrame()
        df = dfs[0]
        # Flatten multi-level column headers
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(str(c) for c in col if "Unnamed" not in str(c)
                          ).strip("_") or f"col_{i}"
                          for i, col in enumerate(df.columns)]
        return df

    def find_table(
        self,
        soup: BeautifulSoup,
        table_id: str,
    ) -> Optional[pd.DataFrame]:
        """Find a table by its HTML id, including in comments."""
        # Try regular DOM first
        tbl = soup.find("table", {"id": table_id})
        if tbl is None:
            # Try commented tables
            for tbl in self._extract_commented_tables(soup):
                if tbl.get("id") == table_id:
                    break
            else:
                return None
        return self._table_to_df(tbl)

    # ── Season URL helpers ───────────────────────────────────────────────────

    @staticmethod
    def _season_str(season: str | int, fbref_id: int) -> str:
        """
        Convert season (e.g. '2324', '2023-24', 2024) to FBRef format.
        Most leagues: '2023-24'.  Some (MLS, etc.): '2024'.
        """
        s = str(season).strip()
        if re.match(r"^\d{4}-\d{2,4}$", s):
            return s  # already in correct format
        if re.match(r"^\d{4}$", s):
            y = int(s)
            if fbref_id in _SPLIT_SEASON_LEAGUES:
                return f"{y-1}-{str(y)[-2:]}"
            return s
        if re.match(r"^\d{2}\d{2}$", s):
            a, b = int("20" + s[:2]), int("20" + s[2:])
            return f"{a}-{str(b)[-2:]}"
        return s

    # ── Schedule / results ───────────────────────────────────────────────────

    def load_schedule(
        self,
        league_code: str,
        season: str | int = "2024-25",
        include_xg: bool = True,
    ) -> pd.DataFrame:
        """
        Load the full match schedule/results for a league season.

        Parameters
        ----------
        league_code : Key from COMPETITIONS dict (e.g. "ENG-PL") OR
                      football-data.co.uk code (e.g. "E0").
        season      : Season string ('2024-25', '2324', 2024, etc.).
        include_xg  : Include xG columns when available.

        Returns
        -------
        DataFrame with columns:
          date, home_team, away_team, home_goals, away_goals,
          xg_home, xg_away, venue, attendance, matchweek, comp
        """
        # Resolve league code
        if league_code in FDCO_TO_FBREF:
            league_code = FDCO_TO_FBREF[league_code]
        comp = COMPETITIONS.get(league_code)
        if comp is None:
            raise ValueError(
                f"Unknown league code '{league_code}'. "
                f"Use one of: {list(COMPETITIONS.keys())}"
            )

        fbref_id = comp["fbref_id"]
        season_str = self._season_str(season, fbref_id)
        comp_slug = comp["name"].replace(" ", "-")

        url = (
            f"{FBREF_BASE}/en/comps/{fbref_id}/{season_str}/schedule/"
            f"{season_str}-{comp_slug}-Scores-and-Fixtures"
        )
        logger.info(f"Fetching schedule: {url}")
        soup = self.get_soup(url)

        # Find the schedule table
        table_id = f"sched_{season_str}_{fbref_id}_1"
        df = self.find_table(soup, table_id)

        if df is None:
            # Try alternative table IDs
            for alt_id in [f"sched_{season_str}_{fbref_id}",
                           "sched_all", "schedule"]:
                df = self.find_table(soup, alt_id)
                if df is not None:
                    break

        if df is None:
            raise ValueError(
                f"Could not find schedule table for {league_code} {season_str}. "
                f"URL: {url}"
            )

        return self._normalise_schedule(df, comp["name"])

    def _normalise_schedule(self, df: pd.DataFrame, comp_name: str) -> pd.DataFrame:
        """Normalise a raw FBRef schedule DataFrame to our schema."""
        df = df.copy()

        # Drop separator rows (all NaN or 'Matchweek X' repeated)
        df = df.dropna(subset=["Date"], how="all")
        df = df[df["Date"].astype(str).str.match(r"\d{4}-\d{2}-\d{2}")]

        # Rename columns to standard names
        rename = {
            "Date":       "date",
            "Home":       "home_team",
            "Away":       "away_team",
            "Score":      "_score",
            "xG":         "xg_home",
            "xG.1":       "xg_away",
            "Attendance": "attendance",
            "Venue":      "venue",
            "Wk":         "matchweek",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

        # Parse score column "1–2" or "1-2"
        if "_score" in df.columns:
            score_split = (
                df["_score"].astype(str)
                .str.replace("–", "-", regex=False)
                .str.split("-", expand=True)
            )
            if score_split.shape[1] >= 2:
                df["home_goals"] = pd.to_numeric(score_split[0], errors="coerce")
                df["away_goals"] = pd.to_numeric(score_split[1], errors="coerce")
            df = df.drop(columns=["_score"], errors="ignore")

        # Parse date
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])

        # xG
        for col in ["xg_home", "xg_away"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Attendance
        if "attendance" in df.columns:
            df["attendance"] = (
                df["attendance"].astype(str)
                .str.replace(",", "", regex=False)
            )
            df["attendance"] = pd.to_numeric(df["attendance"], errors="coerce")

        # Add competition name
        df["comp"] = comp_name

        # Select and order columns
        base_cols = ["date", "home_team", "away_team", "home_goals", "away_goals"]
        extra_cols = [c for c in ["xg_home", "xg_away", "matchweek",
                                   "venue", "attendance", "comp"]
                      if c in df.columns]
        available = [c for c in base_cols + extra_cols if c in df.columns]

        df = df[available].reset_index(drop=True)

        # Drop rows without teams
        df = df.dropna(subset=["home_team", "away_team"])
        df = df[df["home_team"].astype(str).str.strip() != ""]
        df = df[df["away_team"].astype(str).str.strip() != ""]

        return df

    # ── Team match log ───────────────────────────────────────────────────────

    def load_team_match_log(
        self,
        team_id: str,
        team_name: str,
        season: str | int = "2024-25",
        comp_filter: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load per-match stats for a specific team from FBRef.

        Parameters
        ----------
        team_id   : FBRef team ID (e.g. "ecd11ca2" for Galatasaray).
                    Find it in any FBRef team URL.
        team_name : Human-readable team name (for URL construction).
        season    : Season string.
        comp_filter: Filter to a specific competition (e.g. "Champions Lg").

        Returns
        -------
        DataFrame with per-match detailed stats for the team.
        Columns include: date, opponent, venue, goals_for, goals_against,
                         xg_for, xg_against, shots, shots_on_target,
                         possession, progressive_carries, etc.
        """
        # Normalise team name for URL (spaces → hyphens)
        team_slug = re.sub(r"[^a-zA-Z0-9]+", "-", team_name).strip("-")

        url = (
            f"{FBREF_BASE}/en/squads/{team_id}/{season}/"
            f"matchlogs/all_comps/schedule/{team_slug}-Match-Logs"
        )
        logger.info(f"Fetching team match log: {url}")
        soup = self.get_soup(url)

        # Table ID for match log
        table_id = f"matchlogs_all"
        df = self.find_table(soup, table_id)

        if df is None:
            # Fallback: try any table on the page
            all_tables = soup.find_all("table")
            all_tables += self._extract_commented_tables(soup)
            for t in all_tables:
                candidate = self._table_to_df(t)
                if "GF" in candidate.columns or "Gls" in candidate.columns:
                    df = candidate
                    break

        if df is None:
            raise ValueError(
                f"Could not find match log table for team {team_id}. URL: {url}"
            )

        return self._normalise_team_log(df, team_name, team_id, comp_filter)

    def _normalise_team_log(
        self,
        df: pd.DataFrame,
        team_name: str,
        team_id: str,
        comp_filter: Optional[str],
    ) -> pd.DataFrame:
        df = df.copy()

        # Drop rows without dates
        if "Date" in df.columns:
            df = df.dropna(subset=["Date"])
            df = df[df["Date"].astype(str).str.match(r"\d{4}-\d{2}-\d{2}")]
            df = df.rename(columns={"Date": "date"})

        # Rename common columns
        rename_map = {
            "Comp":       "competition",
            "Round":      "round",
            "Venue":      "venue",
            "Result":     "result",
            "GF":         "goals_for",
            "GA":         "goals_against",
            "xG":         "xg_for",
            "xGA":        "xg_against",
            "Opponent":   "opponent",
            "Poss":       "possession",
            "Sh":         "shots",
            "SoT":        "shots_on_target",
            "Cmp":        "passes_completed",
            "Att":        "passes_attempted",
            "PrgP":       "progressive_passes",
            "PrgC":       "progressive_carries",
            "Tkl":        "tackles",
            "Int":        "interceptions",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        # Parse date
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])

        # Parse numeric columns
        for col in ["goals_for", "goals_against", "xg_for", "xg_against",
                    "possession", "shots", "shots_on_target",
                    "progressive_passes", "progressive_carries",
                    "tackles", "interceptions"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Add team identifier
        df["team"] = team_name
        df["team_id"] = team_id

        # Filter by competition
        if comp_filter and "competition" in df.columns:
            mask = df["competition"].astype(str).str.contains(
                comp_filter, case=False, na=False
            )
            df = df[mask]

        # Sort by date
        df = df.sort_values("date").reset_index(drop=True)
        return df

    # ── League-wide team stats ───────────────────────────────────────────────

    def load_league_stats(
        self,
        league_code: str,
        season: str | int = "2024-25",
        stat_type: str = "standard",
    ) -> pd.DataFrame:
        """
        Load aggregate team stats table for a league season.

        Parameters
        ----------
        stat_type : One of:
            'standard'   - Goals, assists, xG, xAG
            'shooting'   - Shots, SoT, xG per shot
            'passing'    - Passes, progressive passes, key passes
            'defense'    - Tackles, interceptions, blocks, pressures
            'possession' - Carries, progressive carries, touches
            'misc'       - Fouls, cards, offsides

        Returns
        -------
        DataFrame with one row per team.
        """
        if league_code in FDCO_TO_FBREF:
            league_code = FDCO_TO_FBREF[league_code]
        comp = COMPETITIONS.get(league_code)
        if comp is None:
            raise ValueError(f"Unknown league code '{league_code}'")

        fbref_id = comp["fbref_id"]
        season_str = self._season_str(season, fbref_id)
        comp_slug = comp["name"].replace(" ", "-")

        stat_slug = {
            "standard":   "stats",
            "shooting":   "shooting",
            "passing":    "passing",
            "defense":    "defense",
            "possession": "possession",
            "misc":       "misc",
        }.get(stat_type, stat_type)

        url = (
            f"{FBREF_BASE}/en/comps/{fbref_id}/{season_str}/{stat_slug}/"
            f"{season_str}-{comp_slug}-Stats"
        )
        logger.info(f"Fetching league stats ({stat_type}): {url}")
        soup = self.get_soup(url)

        # Try to find the "for" stats table (squads_standard_stats)
        table_id = f"stats_squads_{stat_type}_for"
        df = self.find_table(soup, table_id)

        if df is None:
            # Try alternate IDs
            for alt in [f"stats_squads_{stat_slug}_for",
                        f"stats_{stat_type}", "stats_squads"]:
                df = self.find_table(soup, alt)
                if df is not None:
                    break

        if df is None:
            raise ValueError(
                f"Could not find stats table ({stat_type}) for "
                f"{league_code} {season_str}. URL: {url}"
            )

        return self._normalise_league_stats(df, stat_type, comp["name"], season_str)

    def _normalise_league_stats(
        self,
        df: pd.DataFrame,
        stat_type: str,
        comp_name: str,
        season_str: str,
    ) -> pd.DataFrame:
        df = df.copy()

        # Drop totals/separator rows
        if "Squad" in df.columns:
            df = df[df["Squad"].notna()]
            df = df[df["Squad"].astype(str) != "Squad"]  # header repeat rows
            df = df.rename(columns={"Squad": "team"})

        # Add metadata
        df["competition"] = comp_name
        df["season"] = season_str
        df["stat_type"] = stat_type

        # Convert numeric columns
        skip_cols = {"team", "competition", "season", "stat_type",
                     "Country", "Pos", "Age", "MP", "Starts", "Min", "90s"}
        for col in df.columns:
            if col not in skip_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df.reset_index(drop=True)


# ── Convenience functions ──────────────────────────────────────────────────────

def download_fbref_seasons(
    leagues: list[str] | None = None,
    seasons: list[str] | None = None,
    cache_dir: str = "data/fbref_cache",
    delay: float = 4.0,
    verbose: bool = True,
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Batch download FBRef schedules for multiple leagues and seasons.

    Parameters
    ----------
    leagues    : List of league codes (e.g. ["ENG-PL", "ESP-LL"]).
                 Defaults to all 10 domestic + 3 UEFA comps.
    seasons    : List of season strings (e.g. ["2022-23", "2023-24", "2024-25"]).
                 Defaults to last 3 seasons.
    cache_dir  : HTML cache directory (avoids re-downloading).

    Returns
    -------
    Nested dict: {league_code: {season: DataFrame}}
    """
    if leagues is None:
        leagues = list(COMPETITIONS.keys())

    if seasons is None:
        seasons = ["2022-23", "2023-24", "2024-25"]

    scraper = FBRefScraper(delay=delay, cache_dir=cache_dir)
    results: dict[str, dict[str, pd.DataFrame]] = {}

    total = len(leagues) * len(seasons)
    done = 0

    for league in leagues:
        results[league] = {}
        for season in seasons:
            done += 1
            try:
                if verbose:
                    print(f"[{done:>3}/{total}] {league} {season} … ", end="", flush=True)
                df = scraper.load_schedule(league, season)
                results[league][season] = df
                if verbose:
                    n_complete = df["home_goals"].notna().sum()
                    print(f"OK ({n_complete}/{len(df)} completed)")
            except Exception as e:
                if verbose:
                    print(f"SKIP ({e})")
                results[league][season] = pd.DataFrame()

    return results


def combine_fbref_data(
    data: dict[str, dict[str, pd.DataFrame]],
    include_upcoming: bool = False,
) -> pd.DataFrame:
    """
    Flatten nested league×season dict into a single DataFrame.

    Parameters
    ----------
    include_upcoming : If False (default), drop rows without a score
                       (upcoming fixtures). Set True to keep them.

    Returns
    -------
    DataFrame with columns: date, home_team, away_team, home_goals,
    away_goals, xg_home, xg_away, league, season, comp
    """
    frames = []
    for league, seasons in data.items():
        for season, df in seasons.items():
            if df.empty:
                continue
            df = df.copy()
            df["league"] = league
            df["season"] = season
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    if not include_upcoming:
        combined = combined.dropna(subset=["home_goals", "away_goals"])

    combined["home_goals"] = combined["home_goals"].astype("Int64")
    combined["away_goals"] = combined["away_goals"].astype("Int64")
    combined = combined.sort_values(["date", "league"]).reset_index(drop=True)
    return combined


def build_xg_features(
    match_df: pd.DataFrame,
    n_games: int = 6,
) -> pd.DataFrame:
    """
    Compute rolling xG-based features for each match in match_df.

    These supplement the goal-based features from engineer.py with
    better signal from expected goals.

    Parameters
    ----------
    match_df : Combined FBRef DataFrame with columns date, home_team,
               away_team, home_goals, away_goals, xg_home, xg_away.
    n_games  : Rolling window for xG stats.

    Returns
    -------
    Original DataFrame with additional columns:
      h_xg_for_L{n}, h_xg_against_L{n}, h_xg_diff_L{n}
      a_xg_for_L{n}, a_xg_against_L{n}, a_xg_diff_L{n}
      h_xg_overperformance, a_xg_overperformance
      (actual goals vs expected over last N games)
    """
    if "xg_home" not in match_df.columns or match_df["xg_home"].isna().all():
        logger.warning("No xG data available — skipping xG feature engineering.")
        return match_df

    df = match_df.sort_values("date").reset_index(drop=True)

    # Expand to team records
    home_rec = pd.DataFrame({
        "date":        df["date"],
        "team":        df["home_team"],
        "venue":       "home",
        "xg_for":      df["xg_home"],
        "xg_against":  df["xg_away"],
        "goals_for":   df["home_goals"],
    })
    away_rec = pd.DataFrame({
        "date":        df["date"],
        "team":        df["away_team"],
        "venue":       "away",
        "xg_for":      df["xg_away"],
        "xg_against":  df["xg_home"],
        "goals_for":   df["away_goals"],
    })
    records = pd.concat([home_rec, away_rec]).sort_values(["team", "date"])
    records = records.reset_index(drop=True)

    def team_xg_stats(team: str, before_date: pd.Timestamp) -> dict:
        hist = records[
            (records["team"] == team) & (records["date"] < before_date)
        ].tail(n_games)
        if len(hist) < 1:
            return {}
        xg_for     = hist["xg_for"].dropna().mean()
        xg_against = hist["xg_against"].dropna().mean()
        goals_for  = hist["goals_for"].dropna().mean()
        xg_diff    = xg_for - xg_against
        # Overperformance: scoring more/fewer than xG suggests
        overperf   = goals_for - xg_for if not np.isnan(xg_for) else np.nan
        return {
            "xg_for":           round(xg_for, 4) if not np.isnan(xg_for) else np.nan,
            "xg_against":       round(xg_against, 4) if not np.isnan(xg_against) else np.nan,
            "xg_diff":          round(xg_diff, 4) if not np.isnan(xg_diff) else np.nan,
            "xg_overperf":      round(overperf, 4) if not np.isnan(overperf) else np.nan,
        }

    xg_rows = []
    for _, row in df.iterrows():
        h_stats = team_xg_stats(row["home_team"], row["date"])
        a_stats = team_xg_stats(row["away_team"], row["date"])
        xg_rows.append({
            f"h_xg_for":        h_stats.get("xg_for"),
            f"h_xg_against":    h_stats.get("xg_against"),
            f"h_xg_diff":       h_stats.get("xg_diff"),
            f"h_xg_overperf":   h_stats.get("xg_overperf"),
            f"a_xg_for":        a_stats.get("xg_for"),
            f"a_xg_against":    a_stats.get("xg_against"),
            f"a_xg_diff":       a_stats.get("xg_diff"),
            f"a_xg_overperf":   a_stats.get("xg_overperf"),
            "diff_xg_for":      (h_stats.get("xg_for", np.nan) or np.nan) -
                                 (a_stats.get("xg_for", np.nan) or np.nan),
            "diff_xg_against":  (h_stats.get("xg_against", np.nan) or np.nan) -
                                 (a_stats.get("xg_against", np.nan) or np.nan),
        })

    xg_feat_df = pd.DataFrame(xg_rows)
    result = pd.concat(
        [df.reset_index(drop=True), xg_feat_df.reset_index(drop=True)],
        axis=1,
    )
    return result
