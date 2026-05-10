"""
Stage 6 — Replay Engine
=========================
For a given match, replay it window by window and emit the model's probability
and correlation output at each state transition. This is the core "what did the
model believe at each minute?" output.

The replay_results.parquet produced here is the primary deliverable:
a complete probabilistic state-trajectory for every test match, suitable for:
  - Backtesting in-play betting strategies (Stage 7)
  - Calibration analysis (do probabilities update correctly when a goal is scored?)
  - Building a live dashboard

Schema of replay_results.parquet
---------------------------------
fixture_id               int
minute                   int   window start (0, 5, ..., 90)
# State features (from Stage 1)
score_h                  int
score_a                  int
score_diff               int
total_goals              int
n_red_h                  int
n_red_a                  int
minutes_remaining        float
halftime_passed          bool
# Market probabilities (one per market)
p_home_win               float
p_draw                   float
p_away_win               float
p_btts                   float
p_over_25                float
... (all MARKET_NAMES prefixed with p_)
# Correlation compactly serialised
# Rather than storing M² values, store the top-k most useful off-diagonals:
corr_home_win_draw       float
corr_home_win_away_win   float
corr_over_25_btts        float
corr_home_win_over_25    float
... (configurable list)
# Cluster membership for each market (int cluster id)
cluster_home_win         int
cluster_draw             int
...
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from state_extractor import (
    get_state_at,
    build_all_state_timelines,
    WINDOW_MINUTES,
    REGULATION_END,
)
from forward_simulator import simulate_match, market_probabilities, MARKET_NAMES
from correlation_engine import full_correlation_analysis
from prior_integration import get_match_prior_lambdas


# ── Correlation pairs to store inline ────────────────────────────────────────

# These specific pairs are stored as individual columns in the output parquet
# for easy access without deserialising a full matrix.
STORED_CORR_PAIRS: list[tuple[str, str]] = [
    ("home_win",  "draw"),
    ("home_win",  "away_win"),
    ("home_win",  "over_25"),
    ("home_win",  "btts"),
    ("draw",      "btts"),
    ("over_25",   "btts"),
    ("away_win",  "over_25"),
    ("next_goal_home", "next_goal_away"),
    ("no_more_goals",  "draw"),
]


def _corr_col_name(m1: str, m2: str) -> str:
    return f"corr_{m1}__{m2}"


# ── Single match replay ────────────────────────────────────────────────────────

def replay_match(
    fixture_id: int,
    state_timeline: pd.DataFrame,
    home_hazard_model,
    away_hazard_model,
    gbm_predictions: pd.DataFrame,
    calibrated_lambdas: pd.DataFrame | None = None,
    n_sims: int = 5_000,
    window_minutes: int = WINDOW_MINUTES,
    corr_cluster_threshold: float = 0.6,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """
    Replay one match window by window, producing p̂ and ρ at each state.

    Parameters
    ----------
    fixture_id           : Match to replay.
    state_timeline       : Full output of build_all_state_timelines() (all matches).
    home/away_hazard_model: Fitted hazard models from Stage 2.
    gbm_predictions      : test_predictions.parquet — for prior calibration.
    calibrated_lambdas   : Pre-computed (fixture_id, lambda_h, lambda_a) table.
                           If None, calibration runs on-the-fly (slow).
    n_sims               : Simulations per window (5k default; trade off speed/accuracy).
    window_minutes       : Must match the step used in build_all_state_timelines().
    corr_cluster_threshold: Passed to identify_correlation_clusters().
    rng_seed             : Base seed; each window uses rng_seed + minute for independence.

    Returns
    -------
    DataFrame with one row per window boundary. Schema described in module docstring.
    """
    # Get calibrated base rates
    if calibrated_lambdas is not None:
        cl_row = calibrated_lambdas[calibrated_lambdas["fixture_id"] == fixture_id]
        if cl_row.empty:
            raise KeyError(f"fixture_id={fixture_id} not in calibrated_lambdas")
        lambda_h = float(cl_row.iloc[0]["lambda_h"])
        lambda_a = float(cl_row.iloc[0]["lambda_a"])
    else:
        # On-the-fly calibration (slow — use only for single-match inspection)
        lambda_h, lambda_a = get_match_prior_lambdas(
            fixture_id, gbm_predictions, home_hazard_model, away_hazard_model
        )

    window_starts = list(range(0, REGULATION_END + 1, window_minutes))
    rows: list[dict] = []

    for minute in window_starts:
        try:
            state = get_state_at(state_timeline, fixture_id, minute, window_minutes)
        except KeyError:
            # Match ended before this window (e.g. AET) or data gap
            continue

        rng = np.random.default_rng(rng_seed + minute)

        D = simulate_match(
            initial_state     = state,
            lambda_h_base     = lambda_h,
            lambda_a_base     = lambda_a,
            home_hazard_model = home_hazard_model,
            away_hazard_model = away_hazard_model,
            n_simulations     = n_sims,
            rng               = rng,
        )

        p_vec  = market_probabilities(D)
        ca     = full_correlation_analysis(D, p_vec, cluster_threshold=corr_cluster_threshold)
        corr   = ca["corr_matrix"]
        cluster_map = ca["cluster_map"]

        row: dict = {
            "fixture_id":       fixture_id,
            "minute":           minute,
            # State snapshot
            "score_h":          state["score_h"],
            "score_a":          state["score_a"],
            "score_diff":       state["score_diff"],
            "total_goals":      state["total_goals"],
            "n_red_h":          state["n_red_h"],
            "n_red_a":          state["n_red_a"],
            "minutes_remaining": state["minutes_remaining"],
            "halftime_passed":  state["halftime_passed"],
        }

        # Market probabilities
        for mkt in MARKET_NAMES:
            row[f"p_{mkt}"] = round(p_vec.get(mkt, float("nan")), 5)

        # Correlation pairs
        for m1, m2 in STORED_CORR_PAIRS:
            if m1 in corr.index and m2 in corr.columns:
                row[_corr_col_name(m1, m2)] = round(float(corr.loc[m1, m2]), 4)

        # Cluster membership
        for mkt in MARKET_NAMES:
            row[f"cluster_{mkt}"] = cluster_map.get(mkt, -1)

        rows.append(row)

    return pd.DataFrame(rows)


# ── Test-set batch replay ──────────────────────────────────────────────────────

def replay_test_set(
    test_fixtures: pd.DataFrame,
    state_timeline: pd.DataFrame,
    home_hazard_model,
    away_hazard_model,
    gbm_predictions: pd.DataFrame,
    calibrated_lambdas: pd.DataFrame,
    n_sims: int = 5_000,
    window_minutes: int = WINDOW_MINUTES,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Run replay_match for every fixture in test_fixtures and concatenate.

    Parameters
    ----------
    test_fixtures       : Subset of fixtures_usable for the test season.
    state_timeline      : Full timeline (all seasons; filtered per fixture internally).
    calibrated_lambdas  : Pre-computed lambda table (fixture_id, lambda_h, lambda_a).
    output_path         : If given, write replay_results.parquet here.

    Returns
    -------
    Combined replay DataFrame. Saved to output_path if provided.

    Notes
    -----
    - TODO: parallelise with joblib.Parallel — each fixture is independent.
    - Progress is printed every 10 fixtures.
    - Failures are skipped with a warning; partial output is saved on KeyboardInterrupt.
    """
    all_replays: list[pd.DataFrame] = []
    total = len(test_fixtures)

    try:
        for i, (_, fix) in enumerate(test_fixtures.iterrows()):
            fid = int(fix["fixture_id"])
            try:
                replay_df = replay_match(
                    fixture_id          = fid,
                    state_timeline      = state_timeline,
                    home_hazard_model   = home_hazard_model,
                    away_hazard_model   = away_hazard_model,
                    gbm_predictions     = gbm_predictions,
                    calibrated_lambdas  = calibrated_lambdas,
                    n_sims              = n_sims,
                    window_minutes      = window_minutes,
                )
                all_replays.append(replay_df)
            except Exception as exc:
                print(f"  [WARN] fixture {fid} replay failed: {exc}")

            if (i + 1) % 10 == 0:
                print(f"  replayed {i+1}/{total} fixtures")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted — saving partial results")

    if not all_replays:
        return pd.DataFrame()

    combined = pd.concat(all_replays, ignore_index=True)

    if output_path is not None:
        combined.to_parquet(output_path, index=False)
        print(f"Saved replay_results.parquet → {output_path}  ({len(combined)} rows)")

    return combined
