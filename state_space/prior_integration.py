"""
Stage 5 — Prior Integration
=============================
Anchor the simulator's base rates to the GBM's pre-match predictions.

At minute 0, the forward simulator is parameterised by (λ_h_base, λ_a_base) —
the per-minute expected goal rates for home and away. We calibrate these so that
the simulator's marginal output (p_home_win, p_draw, p_away_win) exactly matches
the GBM's pre-match predictions.

This ensures:
  - At minute 0  : simulator output == GBM (principled anchor)
  - At minute t>0: simulator diverges from GBM as evidence accumulates
  - The divergence is state-conditional and event-driven, not heuristic

Calibration method
------------------
  Loss(λ_h, λ_a) = Σ_k (sim_prob_k(λ_h, λ_a) - gbm_prob_k)²

Minimised via scipy.optimize.minimize with L-BFGS-B, starting from a Dixon-
Coles or log5 warm start if available.

Practical note
--------------
Calibration is expensive (each evaluation runs n_sims forward simulations).
Pre-compute and cache (λ_h, λ_a) for every test match and store in a parquet,
rather than calibrating at inference time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from forward_simulator import simulate_match, market_probabilities, MARKET_NAMES
from state_extractor import REGULATION_END


# ── Initial state for minute-0 calibration ────────────────────────────────────

_INITIAL_STATE_TEMPLATE: dict = {
    "minute":               0,
    "score_h":              0,
    "score_a":              0,
    "score_diff":           0,
    "total_goals":          0,
    "n_red_h":              0,
    "n_red_a":              0,
    "n_yellow_h":           0,
    "n_yellow_a":           0,
    "n_subs_h":             0,
    "n_subs_a":             0,
    "last_goal_minute":     float("nan"),
    "minutes_since_last_goal": float("nan"),
    "var_pending":          False,
    "halftime_passed":      False,
    "in_added_time":        False,
    "minutes_remaining":    float(REGULATION_END),
}


# ── Core calibration ───────────────────────────────────────────────────────────

def calibrate_base_lambdas(
    gbm_p_home: float,
    gbm_p_draw: float,
    gbm_p_away: float,
    home_hazard_model,
    away_hazard_model,
    initial_state: dict | None = None,
    n_sims: int = 2_000,
    lambda_bounds: tuple[float, float] = (0.005, 0.06),
    rng_seed: int = 0,
    verbose: bool = False,
) -> tuple[float, float]:
    """
    Find (λ_h, λ_a) in goals/minute that reproduce GBM probabilities at minute 0.

    Parameters
    ----------
    gbm_p_home       : Pre-match home win probability from GBM.
    gbm_p_draw       : Pre-match draw probability from GBM.
    gbm_p_away       : Pre-match away win probability from GBM.
    home_hazard_model: Fitted hazard model (Stage 2).
    away_hazard_model: Fitted hazard model (Stage 2).
    initial_state    : Override the blank minute-0 state (usually not needed).
    n_sims           : Simulations per loss evaluation (2k balances speed/noise).
    lambda_bounds    : Search bounds in goals/minute.
                       Default: (0.005, 0.06) ≈ (0.45/90, 5.4/90) goals/match.
    rng_seed         : Seed for reproducibility across calibration runs.
    verbose          : Print optimiser progress.

    Returns
    -------
    (lambda_h, lambda_a) in goals per minute.

    Notes
    -----
    - Warm start: λ_h₀ ≈ gbm_p_home * 2.5 / 90 is a reasonable starting point
      based on the average of 2.5 total goals per match at typical home win rates.
    - TODO: if the optimiser fails (flat loss landscape), fall back to a lookup
      table mapping (p_home, p_away) → (λ_h, λ_a) pre-computed on a grid.
    - TODO: add a Poisson-consistency check: verify λ_h + λ_a is consistent with
      the implied total goals market (over/under 2.5).
    """
    rng = np.random.default_rng(rng_seed)
    state = initial_state or _INITIAL_STATE_TEMPLATE.copy()

    target = np.array([gbm_p_home, gbm_p_draw, gbm_p_away])

    def loss(params: np.ndarray) -> float:
        lam_h, lam_a = float(params[0]), float(params[1])
        D = simulate_match(
            initial_state    = state,
            lambda_h_base    = lam_h,
            lambda_a_base    = lam_a,
            home_hazard_model= home_hazard_model,
            away_hazard_model= away_hazard_model,
            n_simulations    = n_sims,
            rng              = rng,
        )
        p = market_probabilities(D)
        sim_probs = np.array([p["home_win"], p["draw"], p["away_win"]])
        return float(np.sum((sim_probs - target) ** 2))

    # Warm start: rough heuristic from GBM probs
    # Average 2.5 goals/match; home rate scales with home win prob
    avg_total = 2.5 / 90.0
    lam_h0 = avg_total * (0.4 + gbm_p_home * 0.6)
    lam_a0 = avg_total * (0.4 + gbm_p_away * 0.6)
    lam_h0 = np.clip(lam_h0, *lambda_bounds)
    lam_a0 = np.clip(lam_a0, *lambda_bounds)
    x0 = np.array([lam_h0, lam_a0])

    result = minimize(
        loss,
        x0,
        method="L-BFGS-B",
        bounds=[lambda_bounds, lambda_bounds],
        options={"maxiter": 50, "ftol": 1e-6, "disp": verbose},
    )

    lambda_h, lambda_a = float(result.x[0]), float(result.x[1])

    if verbose or not result.success:
        print(
            f"  Calibration: λ_h={lambda_h:.5f}  λ_a={lambda_a:.5f}  "
            f"loss={result.fun:.6f}  converged={result.success}"
        )

    return lambda_h, lambda_a


# ── Match-level wrapper ────────────────────────────────────────────────────────

def get_match_prior_lambdas(
    fixture_id: int,
    gbm_predictions: pd.DataFrame,
    home_hazard_model,
    away_hazard_model,
    n_sims: int = 2_000,
    verbose: bool = False,
) -> tuple[float, float]:
    """
    Pull GBM predictions for one fixture and return calibrated (λ_h, λ_a).

    Parameters
    ----------
    fixture_id      : Match to calibrate.
    gbm_predictions : DataFrame with columns (fixture_id, p_home, p_draw, p_away).
                      Typically test_predictions.parquet from the GBM stage.
    home/away_hazard_model: Stage 2 fitted models.
    n_sims          : Passed to calibrate_base_lambdas.

    Returns
    -------
    (lambda_h, lambda_a) in goals per minute.

    Raises
    ------
    KeyError if fixture_id is not in gbm_predictions.
    """
    row = gbm_predictions[gbm_predictions["fixture_id"] == fixture_id]
    if row.empty:
        raise KeyError(f"fixture_id={fixture_id} not found in gbm_predictions")

    r = row.iloc[0]
    # TODO: confirm column names match test_predictions.parquet schema
    p_home = float(r.get("p_home", r.get("prob_home_win", 0.40)))
    p_draw = float(r.get("p_draw", r.get("prob_draw",     0.27)))
    p_away = float(r.get("p_away", r.get("prob_away_win", 0.33)))

    # Normalise in case GBM probs don't sum exactly to 1
    total = p_home + p_draw + p_away
    p_home, p_draw, p_away = p_home / total, p_draw / total, p_away / total

    return calibrate_base_lambdas(
        gbm_p_home        = p_home,
        gbm_p_draw        = p_draw,
        gbm_p_away        = p_away,
        home_hazard_model = home_hazard_model,
        away_hazard_model = away_hazard_model,
        n_sims            = n_sims,
        verbose           = verbose,
    )


# ── Batch pre-computation ──────────────────────────────────────────────────────

def precompute_all_lambdas(
    gbm_predictions: pd.DataFrame,
    home_hazard_model,
    away_hazard_model,
    n_sims: int = 2_000,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Calibrate (λ_h, λ_a) for every fixture in gbm_predictions and cache results.

    Parameters
    ----------
    gbm_predictions : DataFrame with fixture_id + GBM probability columns.
    output_path     : If given, write calibrated_lambdas.parquet here.

    Returns
    -------
    DataFrame with columns: (fixture_id, lambda_h, lambda_a).

    Notes
    -----
    This is slow (O(n_fixtures × 50_iterations × n_sims) simulations).
    Run once, cache the parquet, and load at replay time.
    TODO: parallelise with joblib.Parallel across fixtures.
    """
    rows = []
    total = len(gbm_predictions)

    for i, (_, pred_row) in enumerate(gbm_predictions.iterrows()):
        fid = int(pred_row["fixture_id"])
        try:
            lam_h, lam_a = get_match_prior_lambdas(
                fid, gbm_predictions, home_hazard_model, away_hazard_model, n_sims
            )
            rows.append({"fixture_id": fid, "lambda_h": lam_h, "lambda_a": lam_a})
        except Exception as exc:
            print(f"  [WARN] fixture {fid} calibration failed: {exc}")
            rows.append({"fixture_id": fid, "lambda_h": float("nan"), "lambda_a": float("nan")})

        if (i + 1) % 20 == 0:
            print(f"  calibrated {i+1}/{total}")

    result = pd.DataFrame(rows)
    if output_path is not None:
        result.to_parquet(output_path, index=False)
        print(f"Saved calibrated_lambdas.parquet → {output_path}")

    return result
