"""
Stage 3 — Forward Simulator
============================
Given a current match state and per-team base rates, simulate N continuations
to minute 95 using minute-by-minute Bernoulli sampling. This is the engine
that simultaneously produces:
  - p̂ (probability vector over M markets)
  - C (M × M market correlation matrix)

Both come from exactly the same N simulated paths, which is the key principle:
correlation is NOT estimated separately — it falls out of the joint distribution
of Bernoulli indicators observed across simulations.

Credit-risk analogy
-------------------
Each market is a Bernoulli indicator D_j ∈ {0, 1}.
The simulation produces an (N, M) matrix D where:
  E[D_j]      = p̂_j          (marginal probability)
  Cov(D_i,D_j) → correlation   (joint structure)

Both are computed from the same N paths without any independence assumption.

Simulation loop (per path)
--------------------------
For each minute t from current_minute to end_minute:
  1. Compute λ_h(state), λ_a(state) from hazard models.
  2. Sample home_goal ~ Bernoulli(λ_h).
  3. Sample away_goal ~ Bernoulli(λ_a).
  4. Update state (score, and note: red cards persist but are NOT re-drawn).
  5. At end_minute, read off final (score_h, score_a).

Markets evaluated
-----------------
  home_win        away_win        draw
  btts            no_btts
  over_05         over_15         over_25         over_35
  home_clean_sheet away_clean_sheet
  next_goal_home  next_goal_away  no_more_goals   (relative to current state)
  dc_1X           dc_X2           dc_12
  home_minus_15   away_minus_15   (win by 2+)
"""

from __future__ import annotations

from typing import NamedTuple
import numpy as np
import pandas as pd

from goal_hazard_model import predict_hazard, HAZARD_FEATURES


# ── Market registry ────────────────────────────────────────────────────────────

MARKET_NAMES: list[str] = [
    "home_win",
    "draw",
    "away_win",
    "btts",
    "no_btts",
    "over_05",
    "over_15",
    "over_25",
    "over_35",
    "home_clean_sheet",
    "away_clean_sheet",
    "next_goal_home",
    "next_goal_away",
    "no_more_goals",
    "dc_1X",   # double chance: home or draw
    "dc_X2",   # double chance: away or draw
    "dc_12",   # double chance: home or away (no draw)
    "home_minus_15",   # home wins by 2+
    "away_minus_15",   # away wins by 2+
]

M = len(MARKET_NAMES)
MARKET_INDEX: dict[str, int] = {name: i for i, name in enumerate(MARKET_NAMES)}


# ── State dataclass ────────────────────────────────────────────────────────────

class SimState(NamedTuple):
    """Minimal mutable state for one simulation path."""
    score_h: int
    score_a: int
    n_red_h: int
    n_red_a: int
    score_diff: int          # score_h - score_a
    total_goals: int
    halftime_passed: bool
    minutes_remaining: float

    @classmethod
    def from_state_dict(cls, s: dict) -> "SimState":
        return cls(
            score_h           = int(s["score_h"]),
            score_a           = int(s["score_a"]),
            n_red_h           = int(s["n_red_h"]),
            n_red_a           = int(s["n_red_a"]),
            score_diff        = int(s["score_diff"]),
            total_goals       = int(s["total_goals"]),
            halftime_passed   = bool(s["halftime_passed"]),
            minutes_remaining = float(s["minutes_remaining"]),
        )


# ── Market indicator computation ───────────────────────────────────────────────

def compute_market_indicators(
    final_score_h: np.ndarray,
    final_score_a: np.ndarray,
    goals_at_start_h: int,
    goals_at_start_a: int,
) -> np.ndarray:
    """
    Convert arrays of simulated final scores into binary market indicators.

    Parameters
    ----------
    final_score_h    : shape (N,) — home goals at end of simulation.
    final_score_a    : shape (N,) — away goals at end of simulation.
    goals_at_start_h : Home goals already scored when simulation began.
    goals_at_start_a : Away goals already scored when simulation began.

    Returns
    -------
    D : ndarray of shape (N, M), dtype int8. Each column is one Bernoulli.
    """
    N = len(final_score_h)
    D = np.zeros((N, M), dtype=np.int8)

    # Additional goals scored IN the simulation (not yet at start)
    new_h = final_score_h - goals_at_start_h
    new_a = final_score_a - goals_at_start_a
    total_final = final_score_h + final_score_a

    # --- Full-match markets (based on final score) ---
    D[:, MARKET_INDEX["home_win"]]        = (final_score_h > final_score_a).astype(np.int8)
    D[:, MARKET_INDEX["draw"]]            = (final_score_h == final_score_a).astype(np.int8)
    D[:, MARKET_INDEX["away_win"]]        = (final_score_h < final_score_a).astype(np.int8)
    D[:, MARKET_INDEX["btts"]]            = ((final_score_h >= 1) & (final_score_a >= 1)).astype(np.int8)
    D[:, MARKET_INDEX["no_btts"]]         = (~((final_score_h >= 1) & (final_score_a >= 1))).astype(np.int8)
    D[:, MARKET_INDEX["over_05"]]         = (total_final > 0).astype(np.int8)
    D[:, MARKET_INDEX["over_15"]]         = (total_final > 1).astype(np.int8)
    D[:, MARKET_INDEX["over_25"]]         = (total_final > 2).astype(np.int8)
    D[:, MARKET_INDEX["over_35"]]         = (total_final > 3).astype(np.int8)
    D[:, MARKET_INDEX["home_clean_sheet"]] = (final_score_a == 0).astype(np.int8)
    D[:, MARKET_INDEX["away_clean_sheet"]] = (final_score_h == 0).astype(np.int8)
    D[:, MARKET_INDEX["home_minus_15"]]   = (final_score_h - final_score_a >= 2).astype(np.int8)
    D[:, MARKET_INDEX["away_minus_15"]]   = (final_score_a - final_score_h >= 2).astype(np.int8)

    # Double chance
    D[:, MARKET_INDEX["dc_1X"]] = ((final_score_h >= final_score_a)).astype(np.int8)
    D[:, MARKET_INDEX["dc_X2"]] = ((final_score_a >= final_score_h)).astype(np.int8)
    D[:, MARKET_INDEX["dc_12"]] = ((final_score_h != final_score_a)).astype(np.int8)

    # --- In-play markets (relative to current state) ---
    # next_goal_home  = at least one more home goal scored in the simulation
    D[:, MARKET_INDEX["next_goal_home"]]  = (new_h >= 1).astype(np.int8)
    D[:, MARKET_INDEX["next_goal_away"]]  = (new_a >= 1).astype(np.int8)
    D[:, MARKET_INDEX["no_more_goals"]]   = ((new_h + new_a) == 0).astype(np.int8)

    return D


# ── Core simulation ────────────────────────────────────────────────────────────

def simulate_match(
    initial_state: dict,
    lambda_h_base: float,
    lambda_a_base: float,
    home_hazard_model,
    away_hazard_model,
    n_simulations: int = 10_000,
    end_minute: int = 95,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Run N forward simulations from initial_state to end_minute.

    Parameters
    ----------
    initial_state      : State dict from get_state_at() (Stage 1).
    lambda_h_base      : Calibrated home base rate (goals/minute), from Stage 5.
    lambda_a_base      : Calibrated away base rate (goals/minute), from Stage 5.
    home_hazard_model  : Fitted statsmodels result for home hazard (Stage 2).
    away_hazard_model  : Fitted statsmodels result for away hazard (Stage 2).
    n_simulations      : Number of Monte Carlo paths (10k default; 2k for calibration).
    end_minute         : Simulate up to this minute inclusive.
    rng                : Numpy random Generator (pass for reproducibility).

    Returns
    -------
    D : ndarray of shape (N, M), dtype int8.
        Each row is one simulation, each column is one binary market indicator.
        Use market_probabilities(D) to get p̂.

    Notes
    -----
    - Cards in remaining time are NOT re-sampled; current red counts persist.
    - TODO: model yellow→red progression (second yellow) for long remaining times.
    - TODO: added-time duration is fixed at 5 min; could sample from a distribution.
    - Performance: for 10k sims × 90 minutes this is ~900k Bernoulli draws.
      Pure Python loop is fine for Colab; vectorise if latency becomes an issue.
    """
    if rng is None:
        rng = np.random.default_rng()

    start_minute = int(initial_state["minute"])
    minutes_to_simulate = list(range(start_minute, end_minute + 1))

    # Pre-allocate output arrays
    final_score_h = np.full(n_simulations, initial_state["score_h"], dtype=np.int32)
    final_score_a = np.full(n_simulations, initial_state["score_a"], dtype=np.int32)

    # Each simulation path gets its own running state
    # We keep vectorised arrays (one value per sim) where possible
    sim_score_h   = np.full(n_simulations, initial_state["score_h"], dtype=np.int32)
    sim_score_a   = np.full(n_simulations, initial_state["score_a"], dtype=np.int32)

    # Cards: fixed per initial state (no new cards sampled)
    n_red_h = int(initial_state["n_red_h"])
    n_red_a = int(initial_state["n_red_a"])

    for minute in minutes_to_simulate:
        halftime_passed   = minute >= 45
        minutes_remaining = max(0, 90 - minute)

        # Build a representative state dict for hazard prediction
        # We use the MEAN score differential across simulations as an approximation.
        # TODO: for higher fidelity, bucket simulations by score and compute
        #       separate hazard rates per bucket (score-conditional simulation).
        mean_diff = float(np.mean(sim_score_h - sim_score_a))
        state_for_hazard = {
            "score_diff":        mean_diff,
            "minutes_remaining": minutes_remaining,
            "n_red_h":           n_red_h,
            "n_red_a":           n_red_a,
            "halftime_passed":   halftime_passed,
        }

        lam_h = predict_hazard(
            state_for_hazard, lambda_h_base, "home",
            home_hazard_model, away_hazard_model,
        )
        lam_a = predict_hazard(
            state_for_hazard, lambda_a_base, "away",
            home_hazard_model, away_hazard_model,
        )

        # Bernoulli draws for this minute across all simulations
        home_goal = rng.random(n_simulations) < lam_h
        away_goal = rng.random(n_simulations) < lam_a

        sim_score_h += home_goal.astype(np.int32)
        sim_score_a += away_goal.astype(np.int32)

    D = compute_market_indicators(
        sim_score_h,
        sim_score_a,
        goals_at_start_h = int(initial_state["score_h"]),
        goals_at_start_a = int(initial_state["score_a"]),
    )
    return D


# ── Probability extraction ─────────────────────────────────────────────────────

def market_probabilities(D: np.ndarray) -> dict[str, float]:
    """
    Compute E[D_j] = p̂_j for each market from the simulation matrix.

    Parameters
    ----------
    D : (N, M) binary indicator matrix from simulate_match().

    Returns
    -------
    Dict mapping market_name → probability in [0, 1].
    """
    means = D.mean(axis=0)
    return {name: float(means[i]) for i, name in enumerate(MARKET_NAMES)}
