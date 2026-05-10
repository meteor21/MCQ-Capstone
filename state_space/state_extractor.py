"""
Stage 1 — State Extractor
=========================
Walk events chronologically and produce a state timeline per match.

The "state" at time t captures everything that has happened up to (but not
including) that minute window that affects future scoring rates. Downstream
stages consume (fixture_id, minute) → state_vector lookups.

Schema of output parquet
------------------------
fixture_id           int
minute               int   start of 5-minute window (0, 5, 10, ..., 90)
score_h              int   home goals scored so far
score_a              int   away goals scored so far
score_diff           int   score_h - score_a
total_goals          int   score_h + score_a
n_red_h              int   cumulative home red cards
n_red_a              int   cumulative away red cards
n_yellow_h           int   cumulative home yellow cards (informational)
n_yellow_a           int   cumulative away yellow cards
n_subs_h             int   cumulative home substitutions
n_subs_a             int   cumulative away substitutions
last_goal_minute     float NaN if no goal yet; otherwise the minute of last goal
minutes_since_last_goal float NaN if no goal yet
var_pending          bool  a VAR review event is open and unresolved
halftime_passed      bool  minute >= 45
in_added_time        bool  minute > 90
minutes_remaining    float 90 - minute (capped at 0; use 95 for added-time states)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


# ── Constants ─────────────────────────────────────────────────────────────────

WINDOW_MINUTES: int = 5
REGULATION_END: int = 90
ADDED_TIME_END: int = 95   # we simulate up to minute 95

# Event type strings as they appear in events_clean.parquet
EVENT_GOAL  = "goal"
EVENT_CARD  = "card"
EVENT_SUBST = "subst"
EVENT_VAR   = "var"

DETAIL_YELLOW      = "yellow card"
DETAIL_RED         = "red card"
DETAIL_SECOND_YEL  = "second yellow card"


# ── Core state extraction ──────────────────────────────────────────────────────

def extract_state_timeline(
    events_df: pd.DataFrame,
    fixture_id: int,
    home_team_id: str,
    away_team_id: str,
    window_minutes: int = WINDOW_MINUTES,
) -> pd.DataFrame:
    """
    Produce one row per window boundary for a single match.

    Parameters
    ----------
    events_df      : Full events dataframe (all matches). Filtered internally.
    fixture_id     : Match to process.
    home_team_id   : String team_id for the home side.
    away_team_id   : String team_id for the away side.
    window_minutes : Width of each window bucket (default 5).

    Returns
    -------
    DataFrame with one row per window start minute (0, 5, ..., 90).
    All state columns reflect what has happened BEFORE that minute.
    """
    ev = events_df[events_df["fixture_id"] == fixture_id].copy()
    ev["minute"] = pd.to_numeric(ev["minute"], errors="coerce").fillna(0).astype(int)
    ev["type"]   = ev["type"].fillna("").str.lower()
    ev["detail"] = ev["detail"].fillna("").str.lower()
    ev["team_id"] = ev["team_id"].astype(str)

    # Running state (mutated as we scan events in order)
    state: dict = {
        "score_h":   0,
        "score_a":   0,
        "n_red_h":   0,
        "n_red_a":   0,
        "n_yellow_h": 0,
        "n_yellow_a": 0,
        "n_subs_h":  0,
        "n_subs_a":  0,
        "last_goal_minute": np.nan,
        "var_pending": False,
    }

    ev_sorted = ev.sort_values(["minute", "extra"], na_position="last")

    # Accumulator: state snapshots at each event minute
    # We'll interpolate into windows after
    snapshots: list[dict] = []

    def snapshot_at(minute: int) -> dict:
        lgm = state["last_goal_minute"]
        return {
            "fixture_id":              fixture_id,
            "minute":                  minute,
            "score_h":                 state["score_h"],
            "score_a":                 state["score_a"],
            "score_diff":              state["score_h"] - state["score_a"],
            "total_goals":             state["score_h"] + state["score_a"],
            "n_red_h":                 state["n_red_h"],
            "n_red_a":                 state["n_red_a"],
            "n_yellow_h":              state["n_yellow_h"],
            "n_yellow_a":              state["n_yellow_a"],
            "n_subs_h":                state["n_subs_h"],
            "n_subs_a":                state["n_subs_a"],
            "last_goal_minute":        lgm,
            "minutes_since_last_goal": minute - lgm if not np.isnan(lgm) else np.nan,
            "var_pending":             state["var_pending"],
            "halftime_passed":         minute >= 45,
            "in_added_time":           minute > REGULATION_END,
            "minutes_remaining":       max(0, REGULATION_END - minute),
        }

    # Walk events, updating state, take snapshots at each window boundary
    window_boundaries = list(range(0, REGULATION_END + 1, window_minutes))
    boundary_idx = 0
    prev_minute = 0

    for _, ev_row in ev_sorted.iterrows():
        ev_minute = int(ev_row["minute"])
        is_home = ev_row["team_id"] == str(home_team_id)
        is_away = ev_row["team_id"] == str(away_team_id)

        # Flush any window boundaries that fall before this event
        while boundary_idx < len(window_boundaries) and \
              window_boundaries[boundary_idx] <= ev_minute:
            snapshots.append(snapshot_at(window_boundaries[boundary_idx]))
            boundary_idx += 1

        # Apply event to state
        etype  = ev_row["type"]
        detail = ev_row["detail"]

        if etype == EVENT_GOAL and "own goal" not in detail and "cancelled" not in detail:
            if is_home:
                state["score_h"] += 1
            elif is_away:
                state["score_a"] += 1
            state["last_goal_minute"] = ev_minute
            state["var_pending"] = False

        elif etype == EVENT_CARD:
            is_red = detail in {DETAIL_RED, DETAIL_SECOND_YEL}
            if is_red:
                if is_home:  state["n_red_h"] += 1
                elif is_away: state["n_red_a"] += 1
            else:
                if is_home:  state["n_yellow_h"] += 1
                elif is_away: state["n_yellow_a"] += 1

        elif etype == EVENT_SUBST:
            if is_home:  state["n_subs_h"] += 1
            elif is_away: state["n_subs_a"] += 1

        elif etype == EVENT_VAR:
            # TODO: parse VAR detail to determine if it opens or resolves a review
            # For now, mark as pending if detail suggests review in progress
            state["var_pending"] = "goal cancelled" not in detail

    # Flush remaining window boundaries after all events
    while boundary_idx < len(window_boundaries):
        snapshots.append(snapshot_at(window_boundaries[boundary_idx]))
        boundary_idx += 1

    return pd.DataFrame(snapshots)


# ── Batch builder ──────────────────────────────────────────────────────────────

def build_all_state_timelines(
    fixtures: pd.DataFrame,
    events: pd.DataFrame,
    window_minutes: int = WINDOW_MINUTES,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Run extract_state_timeline for every fixture and concatenate.

    Parameters
    ----------
    fixtures      : fixtures_usable.parquet — needs fixture_id, home_team_id, away_team_id.
    events        : events_clean.parquet.
    window_minutes: Passed through to extract_state_timeline.
    output_path   : If given, write state_timeline.parquet here.

    Returns
    -------
    Combined state timeline DataFrame.
    """
    all_timelines: list[pd.DataFrame] = []
    total = len(fixtures)

    for i, (_, fix) in enumerate(fixtures.iterrows()):
        fid      = fix["fixture_id"]
        home_id  = str(fix["home_team_id"])
        away_id  = str(fix["away_team_id"])

        try:
            tl = extract_state_timeline(events, fid, home_id, away_id, window_minutes)
            all_timelines.append(tl)
        except Exception as exc:
            # TODO: log failures to a separate error log instead of printing
            print(f"  [WARN] fixture {fid} failed: {exc}")

        if (i + 1) % 200 == 0:
            print(f"  processed {i+1}/{total} fixtures")

    combined = pd.concat(all_timelines, ignore_index=True)

    if output_path is not None:
        combined.to_parquet(output_path, index=False)
        print(f"Saved state_timeline.parquet → {output_path}  ({len(combined)} rows)")

    return combined


# ── Convenience lookup ─────────────────────────────────────────────────────────

def get_state_at(
    state_timeline: pd.DataFrame,
    fixture_id: int,
    minute: int,
    window_minutes: int = WINDOW_MINUTES,
) -> dict:
    """
    Return the state dict for (fixture_id, minute).

    Snaps minute down to the nearest window boundary.
    Raises KeyError if the fixture is not found.
    """
    boundary = (minute // window_minutes) * window_minutes
    row = state_timeline[
        (state_timeline["fixture_id"] == fixture_id) &
        (state_timeline["minute"] == boundary)
    ]
    if row.empty:
        raise KeyError(f"No state found for fixture={fixture_id} minute={boundary}")
    return row.iloc[0].to_dict()
