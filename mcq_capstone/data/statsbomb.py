"""
StatsBomb Open Data downloader.

Pulls freely available event-level match data from the StatsBomb open-data
GitHub repository.  No API key or subscription required.

Available data (as of 2025):
  - La Liga        : 2004/05 – 2020/21  (~500 full matches)
  - Premier League : 2003/04, 2015/16   (418 matches)
  - Serie A        : 2015/16            (380 matches)
  - Ligue 1        : 2015/16–2022/23    (435 matches)
  - Bundesliga     : 2015/16, 2023/24   (340 matches)
  - Champions League: finals only
  - World Cup 2018/2022, Euro 2020/2024, Copa America 2024
  Total: ~3,464 matches, ~12M events

Output schema
-------------
matches.parquet
  match_id, date, kick_off, competition, season, matchweek,
  home_team, away_team, home_score, away_score, stadium, referee

events.parquet
  match_id, event_id, period, minute, second, type,
  team, player, x, y,
  -- goal fields --
  shot_outcome, shot_technique, shot_body_part, shot_first_time, shot_xg,
  -- card fields --
  bad_behaviour_card_type, foul_card_type,
  -- sub fields --
  substitution_replacement,
  -- extra --
  is_goal, is_card, is_penalty, is_own_goal,
  home_score_after, away_score_after      (running scoreline)

shots.parquet
  match_id, event_id, minute, second, period,
  team, player, shot_outcome, shot_technique,
  shot_body_part, shot_first_time, shot_xg,
  x, y, end_x, end_y, is_goal
"""

from __future__ import annotations

import json
import time
import logging
from typing import Optional
from pathlib import Path

import requests
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
_DELAY = 0.3   # seconds between requests (GitHub CDN, very permissive)


# ── Competition catalogue ──────────────────────────────────────────────────────

# Only competitions with meaningful match counts
GOOD_COMPETITIONS = {
    2:   "Premier League",
    11:  "La Liga",
    9:   "1. Bundesliga",
    12:  "Serie A",
    7:   "Ligue 1",
    16:  "Champions League",
    35:  "UEFA Europa League",
    43:  "FIFA World Cup",
    55:  "UEFA Euro",
    223: "Copa America",
    37:  "FA Women's Super League",
    49:  "NWSL",
    1267:"African Cup of Nations",
}


def _get(url: str, retries: int = 3) -> dict | list:
    """Fetch JSON from URL with retry."""
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


# ── Main downloader ────────────────────────────────────────────────────────────

class StatsBombLoader:
    """
    Download and flatten StatsBomb open data to Parquet files.

    Parameters
    ----------
    out_dir    : Directory to write parquet files (created if needed).
    min_matches: Skip competition-seasons with fewer than this many matches.
                 Set to 1 to download everything including finals-only datasets.
    competitions: List of competition IDs to include (None = all good ones).
    delay      : Seconds between HTTP requests.
    """

    def __init__(
        self,
        out_dir: str = "data/parquet",
        min_matches: int = 10,
        competitions: Optional[list[int]] = None,
        delay: float = _DELAY,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.min_matches = min_matches
        self.competitions = competitions or list(GOOD_COMPETITIONS.keys())
        self.delay = delay

    # ── Public entry point ─────────────────────────────────────────────────

    def download_all(self, verbose: bool = True) -> dict[str, pd.DataFrame]:
        """
        Download all available StatsBomb open data and save as Parquet.

        Returns dict with keys: 'matches', 'events', 'shots'
        """
        catalogue = _get(f"{_BASE}/competitions.json")
        todo = [
            c for c in catalogue
            if c["competition_id"] in self.competitions
        ]

        if verbose:
            print(f"\nStatsBomb Open Data Downloader")
            print(f"  Output dir   : {self.out_dir}")
            print(f"  Competition-seasons to check: {len(todo)}")

        all_matches: list[pd.DataFrame] = []
        all_events:  list[pd.DataFrame] = []
        all_shots:   list[pd.DataFrame] = []

        done = 0
        for ci, comp in enumerate(todo):
            cid = comp["competition_id"]
            sid = comp["season_id"]
            cname = comp["competition_name"]
            sname = comp["season_name"]

            url = f"{_BASE}/matches/{cid}/{sid}.json"
            try:
                matches_raw = _get(url)
                time.sleep(self.delay)
            except Exception as e:
                if verbose:
                    print(f"  [{ci+1:>2}/{len(todo)}] SKIP {cname} {sname}: {e}")
                continue

            if len(matches_raw) < self.min_matches:
                if verbose:
                    print(f"  [{ci+1:>2}/{len(todo)}] SKIP {cname} {sname} "
                          f"({len(matches_raw)} matches < min={self.min_matches})")
                continue

            if verbose:
                print(f"  [{ci+1:>2}/{len(todo)}] {cname:<30} {sname:<12} "
                      f"{len(matches_raw):>4} matches … ", end="", flush=True)

            matches_df = self._flatten_matches(matches_raw, cname, sname)
            all_matches.append(matches_df)

            ev_frames, sh_frames = [], []
            for match in matches_raw:
                mid = match["match_id"]
                try:
                    events_raw = _get(f"{_BASE}/events/{mid}.json")
                    time.sleep(self.delay)
                    ev, sh = self._flatten_events(events_raw, mid)
                    ev_frames.append(ev)
                    sh_frames.append(sh)
                    done += 1
                except Exception:
                    pass

            if ev_frames:
                all_events.append(pd.concat(ev_frames, ignore_index=True))
            if sh_frames:
                all_shots.append(pd.concat(sh_frames, ignore_index=True))

            if verbose:
                n_ev = sum(len(f) for f in ev_frames)
                n_sh = sum(len(f) for f in sh_frames)
                print(f"OK  ({n_ev:,} events, {n_sh:,} shots)")

        # Combine and save
        result = {}

        if all_matches:
            matches = pd.concat(all_matches, ignore_index=True)
            matches = matches.sort_values(["competition", "date"]).reset_index(drop=True)
            path = self.out_dir / "matches.parquet"
            matches.to_parquet(path, index=False)
            result["matches"] = matches
            if verbose:
                print(f"\n  matches.parquet  : {len(matches):>6,} rows → {path}")

        if all_events:
            events = pd.concat(all_events, ignore_index=True)
            path = self.out_dir / "events.parquet"
            events.to_parquet(path, index=False)
            result["events"] = events
            if verbose:
                print(f"  events.parquet   : {len(events):>6,} rows → {path}")

        if all_shots:
            shots = pd.concat(all_shots, ignore_index=True)
            path = self.out_dir / "shots.parquet"
            shots.to_parquet(path, index=False)
            result["shots"] = shots
            if verbose:
                print(f"  shots.parquet    : {len(shots):>6,} rows → {path}")

        return result

    # ── Flatten helpers ────────────────────────────────────────────────────

    @staticmethod
    def _flatten_matches(
        raw: list[dict],
        competition: str,
        season: str,
    ) -> pd.DataFrame:
        rows = []
        for m in raw:
            rows.append({
                "match_id":    m["match_id"],
                "date":        pd.Timestamp(m["match_date"]),
                "kick_off":    m.get("kick_off"),
                "competition": competition,
                "season":      season,
                "matchweek":   m.get("match_week"),
                "home_team":   m["home_team"]["home_team_name"],
                "away_team":   m["away_team"]["away_team_name"],
                "home_score":  m["home_score"],
                "away_score":  m["away_score"],
                "stadium":     m.get("stadium", {}).get("name") if m.get("stadium") else None,
                "referee":     m.get("referee", {}).get("name") if m.get("referee") else None,
                "home_managers": ", ".join(
                    mg.get("name","") for mg in m.get("home_team",{}).get("managers",[])
                ) or None,
                "away_managers": ", ".join(
                    mg.get("name","") for mg in m.get("away_team",{}).get("managers",[])
                ) or None,
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _flatten_events(
        raw: list[dict],
        match_id: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns (events_df, shots_df).

        events_df: filtered to key event types only:
          Shot (goals + non-goals), Card (yellow/red), Substitution,
          Penalty (if distinct type), Own Goal Against
        shots_df: all shots with xG
        """
        # Running score tracker
        home_score = 0
        away_score = 0

        # First pass — determine home team name from first event
        home_team = None
        for e in raw:
            if e.get("type", {}).get("name") == "Starting XI":
                home_team = e.get("team", {}).get("name")
                break

        ev_rows, sh_rows = [], []

        for e in raw:
            etype = e.get("type", {}).get("name", "")
            team  = e.get("team", {}).get("name", "")
            player = e.get("player", {}).get("name", "")
            minute = e.get("minute", 0)
            second = e.get("second", 0)
            period = e.get("period", 1)
            loc    = e.get("location") or [None, None]
            eid    = e.get("id", "")

            # ── Shots ──────────────────────────────────────────────────
            if etype == "Shot":
                shot = e.get("shot", {})
                outcome = shot.get("outcome", {}).get("name", "")
                is_goal = outcome == "Goal"
                is_og   = shot.get("type", {}).get("name") == "Own Goal For"

                if is_goal:
                    if team == home_team:
                        home_score += 1
                    else:
                        away_score += 1

                end_loc = shot.get("end_location") or [None, None, None]
                sh_rows.append({
                    "match_id":         match_id,
                    "event_id":         eid,
                    "period":           period,
                    "minute":           minute,
                    "second":           second,
                    "team":             team,
                    "player":           player,
                    "shot_outcome":     outcome,
                    "shot_technique":   shot.get("technique", {}).get("name"),
                    "shot_body_part":   shot.get("body_part", {}).get("name"),
                    "shot_first_time":  shot.get("first_time", False),
                    "shot_xg":          shot.get("statsbomb_xg"),
                    "shot_type":        shot.get("type", {}).get("name"),
                    "x":                loc[0],
                    "y":                loc[1],
                    "end_x":            end_loc[0],
                    "end_y":            end_loc[1],
                    "is_goal":          is_goal,
                    "is_own_goal":      is_og,
                    "home_score_after": home_score,
                    "away_score_after": away_score,
                })

                if is_goal:
                    ev_rows.append({
                        "match_id":         match_id,
                        "event_id":         eid,
                        "period":           period,
                        "minute":           minute,
                        "second":           second,
                        "type":             "Own Goal" if is_og else "Goal",
                        "team":             team,
                        "player":           player,
                        "x":                loc[0],
                        "y":                loc[1],
                        "shot_xg":          shot.get("statsbomb_xg"),
                        "shot_technique":   shot.get("technique", {}).get("name"),
                        "shot_body_part":   shot.get("body_part", {}).get("name"),
                        "shot_first_time":  shot.get("first_time", False),
                        "is_goal":          True,
                        "is_card":          False,
                        "is_penalty":       shot.get("type", {}).get("name") == "Penalty",
                        "is_own_goal":      is_og,
                        "home_score_after": home_score,
                        "away_score_after": away_score,
                        "card_type":        None,
                        "sub_replacement":  None,
                    })

            # ── Cards ──────────────────────────────────────────────────
            elif etype in ("Foul Committed", "Bad Behaviour"):
                card = None
                if etype == "Foul Committed":
                    card = e.get("foul_committed", {}).get("card", {}).get("name")
                else:
                    card = e.get("bad_behaviour", {}).get("card", {}).get("name")
                if card and card != "No Card":
                    ev_rows.append({
                        "match_id":         match_id,
                        "event_id":         eid,
                        "period":           period,
                        "minute":           minute,
                        "second":           second,
                        "type":             "Card",
                        "team":             team,
                        "player":           player,
                        "x":                loc[0],
                        "y":                loc[1],
                        "shot_xg":          None,
                        "shot_technique":   None,
                        "shot_body_part":   None,
                        "shot_first_time":  None,
                        "is_goal":          False,
                        "is_card":          True,
                        "is_penalty":       False,
                        "is_own_goal":      False,
                        "home_score_after": home_score,
                        "away_score_after": away_score,
                        "card_type":        card,
                        "sub_replacement":  None,
                    })

            # ── Substitutions ──────────────────────────────────────────
            elif etype == "Substitution":
                replacement = e.get("substitution", {}).get("replacement", {}).get("name")
                ev_rows.append({
                    "match_id":         match_id,
                    "event_id":         eid,
                    "period":           period,
                    "minute":           minute,
                    "second":           second,
                    "type":             "Substitution",
                    "team":             team,
                    "player":           player,
                    "x":                None,
                    "y":                None,
                    "shot_xg":          None,
                    "shot_technique":   None,
                    "shot_body_part":   None,
                    "shot_first_time":  None,
                    "is_goal":          False,
                    "is_card":          False,
                    "is_penalty":       False,
                    "is_own_goal":      False,
                    "home_score_after": home_score,
                    "away_score_after": away_score,
                    "card_type":        None,
                    "sub_replacement":  replacement,
                })

        events_df = pd.DataFrame(ev_rows)
        shots_df  = pd.DataFrame(sh_rows)
        return events_df, shots_df
