"""
Stage 2 — Goal Hazard Model
============================
Fit a Poisson regression for instantaneous goal-scoring intensity as a
function of match state. Two separate models: one for the home side, one
for the away side.

Model form
----------
  log(λ_team_per_minute) = β₀ + β · state_features + log(base_rate)

where base_rate is the pre-match expected goals per minute for this team
(calibrated in Stage 5, or taken from the Dixon-Coles feature parquet).

The offset log(base_rate) lets the regression learn how state modulates a
match-specific prior rather than fitting a global mean — analogous to
exposure in actuarial Poisson models.

State features
--------------
  score_diff_perspective  : goals_for - goals_against from this team's POV
  minutes_remaining       : 90 - current_minute (pressure proxy)
  n_red_self              : own team red cards (reduces attacking)
  n_red_opp               : opponent red cards (boosts attacking)
  halftime_passed         : binary, slight structural break at HT
  is_home                 : universal home advantage on scoring rate

Training discipline
-------------------
ONLY seasons in `train_seasons` are used to fit the model.
The 2025-26 season (test set) is NEVER seen during training.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

# Optional heavy imports — wrapped so the file can be imported without crashing
# if statsmodels is not installed in the current environment.
try:
    import statsmodels.api as sm
    from statsmodels.genmod.families import Poisson
    _SM_AVAILABLE = True
except ImportError:
    _SM_AVAILABLE = False
    print("[WARN] statsmodels not found — fit_hazard_models() will raise ImportError")


# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_TRAIN_SEASONS = [2023, 2024]
MODEL_SAVE_PATH = Path("hazard_models.pkl")

# Feature columns used in the Poisson GLM
HAZARD_FEATURES = [
    "score_diff_perspective",
    "minutes_remaining",
    "n_red_self",
    "n_red_opp",
    "halftime_passed",
    "is_home",
]


# ── Training data preparation ──────────────────────────────────────────────────

def prepare_hazard_training_data(
    state_timeline: pd.DataFrame,
    events: pd.DataFrame,
    fixtures: pd.DataFrame,
    dc_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build a training dataframe for the Poisson GLM.

    For each (fixture_id, window_minute) row in state_timeline, determine:
      - Did the home team score in [minute, minute + window_size)?
      - Did the away team score in [minute, minute + window_size)?

    Then produce two rows per window (one per side) with the relevant state
    features and a binary outcome `goal_scored` (1/0).

    Parameters
    ----------
    state_timeline : Output of build_all_state_timelines().
    events         : events_clean.parquet — used to label outcomes.
    fixtures       : fixtures_usable.parquet — provides team IDs and seasons.
    dc_features    : Optional Dixon-Coles features parquet with columns
                     (fixture_id, dc_lambda_h, dc_lambda_a) for the base-rate
                     offset. If None, base_rate is set to a constant (1.2/90).

    Returns
    -------
    DataFrame with one row per (fixture_id, minute, side) triple.
    Columns: HAZARD_FEATURES + ['goal_scored', 'log_base_rate', 'season'].
    """
    # Merge season into timeline
    fix_slim = fixtures[["fixture_id", "home_team_id", "away_team_id", "season"]].copy()
    fix_slim["home_team_id"] = fix_slim["home_team_id"].astype(str)
    fix_slim["away_team_id"] = fix_slim["away_team_id"].astype(str)

    tl = state_timeline.merge(fix_slim, on="fixture_id", how="left")

    # Determine window size from the data (gap between consecutive minutes)
    # TODO: make this explicit instead of inferring
    minutes_sorted = tl["minute"].sort_values().unique()
    window_size = int(minutes_sorted[1] - minutes_sorted[0]) if len(minutes_sorted) > 1 else 5

    # Label goals: did a goal occur in [minute, minute + window_size)?
    goal_events = events[
        events["type"].str.lower() == "goal"
    ][["fixture_id", "minute", "team_id"]].copy()
    goal_events["minute"] = pd.to_numeric(goal_events["minute"], errors="coerce")
    goal_events["team_id"] = goal_events["team_id"].astype(str)

    rows: list[dict] = []

    for _, state in tl.iterrows():
        fid      = state["fixture_id"]
        t        = int(state["minute"])
        home_id  = str(state["home_team_id"])
        away_id  = str(state["away_team_id"])
        season   = state.get("season")

        window_goals = goal_events[
            (goal_events["fixture_id"] == fid) &
            (goal_events["minute"] >= t) &
            (goal_events["minute"] < t + window_size)
        ]

        home_scored = int((window_goals["team_id"] == home_id).any())
        away_scored = int((window_goals["team_id"] == away_id).any())

        # Base rate: default 1.2 expected goals per team per 90 min
        # TODO: pull from dc_features when available
        base_lambda_h = 1.2 / 90.0
        base_lambda_a = 1.0 / 90.0
        if dc_features is not None:
            dc_row = dc_features[dc_features["fixture_id"] == fid]
            if not dc_row.empty:
                base_lambda_h = float(dc_row.iloc[0].get("dc_lambda_h", 1.2)) / 90.0
                base_lambda_a = float(dc_row.iloc[0].get("dc_lambda_a", 1.0)) / 90.0

        for side, team_scored, score_diff_pov, n_red_self, n_red_opp, base_lambda in [
            ("home", home_scored,
             state["score_diff"],        # home POV: score_h - score_a
             state["n_red_h"],
             state["n_red_a"],
             base_lambda_h),
            ("away", away_scored,
             -state["score_diff"],       # away POV: score_a - score_h
             state["n_red_a"],
             state["n_red_h"],
             base_lambda_a),
        ]:
            rows.append({
                "fixture_id":            fid,
                "minute":                t,
                "side":                  side,
                "season":                season,
                # Features
                "score_diff_perspective": score_diff_pov,
                "minutes_remaining":      state["minutes_remaining"],
                "n_red_self":             n_red_self,
                "n_red_opp":              n_red_opp,
                "halftime_passed":        int(state["halftime_passed"]),
                "is_home":                int(side == "home"),
                # Outcome
                "goal_scored":            team_scored,
                # Offset for GLM: log(λ_base per minute × window_size)
                "log_base_rate":          np.log(max(base_lambda * window_size, 1e-9)),
            })

    return pd.DataFrame(rows)


# ── Model fitting ──────────────────────────────────────────────────────────────

def fit_hazard_models(
    training_df: pd.DataFrame,
    train_seasons: list[int] = DEFAULT_TRAIN_SEASONS,
    save_path: Path | None = MODEL_SAVE_PATH,
) -> tuple:
    """
    Fit two Poisson GLMs (home side, away side) on training seasons only.

    Parameters
    ----------
    training_df   : Output of prepare_hazard_training_data().
    train_seasons : Seasons to include. Test season MUST be excluded.
    save_path     : Pickle output path. Pass None to skip saving.

    Returns
    -------
    (home_result, away_result) — statsmodels GLMResultsWrapper objects.

    Notes
    -----
    - Uses an offset term (log_base_rate) so the GLM learns STATE adjustments
      rather than an absolute rate.
    - TODO: consider a regularised GLM (Ridge) if coefficients are unstable.
    - TODO: consider adding interaction terms, e.g. score_diff × minutes_remaining.
    """
    if not _SM_AVAILABLE:
        raise ImportError("statsmodels is required for fit_hazard_models()")

    train = training_df[training_df["season"].isin(train_seasons)].copy()
    print(f"Training rows (all seasons): {len(training_df)}")
    print(f"Training rows (train only ): {len(train)}")

    results = {}
    for side in ("home", "away"):
        subset = train[train["side"] == side].copy()
        X = sm.add_constant(subset[HAZARD_FEATURES].astype(float))
        y = subset["goal_scored"].astype(int)
        offset = subset["log_base_rate"].astype(float)

        # TODO: if any feature has near-zero variance, drop it to avoid rank deficiency
        glm = sm.GLM(y, X, family=Poisson(), offset=offset)
        result = glm.fit()
        results[side] = result
        print(f"\n[{side.upper()} hazard model]")
        print(result.summary2().tables[1])

    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump(results, f)
        print(f"\nSaved hazard models → {save_path}")

    return results["home"], results["away"]


def load_hazard_models(load_path: Path = MODEL_SAVE_PATH) -> tuple:
    """Load previously fitted hazard models from disk."""
    with open(load_path, "rb") as f:
        models = pickle.load(f)
    return models["home"], models["away"]


# ── Prediction ─────────────────────────────────────────────────────────────────

def predict_hazard(
    state: dict,
    base_rate_per_minute: float,
    side: Literal["home", "away"],
    home_model,
    away_model,
) -> float:
    """
    Predict goals-per-minute for one team given current state.

    Parameters
    ----------
    state                : Dict with keys matching HAZARD_FEATURES.
    base_rate_per_minute : Match-specific base rate (λ_base / 60), from prior_integration.
    side                 : 'home' or 'away'.
    home_model           : Fitted statsmodels result for the home side.
    away_model           : Fitted statsmodels result for the away side.

    Returns
    -------
    λ_team (goals per minute) — always positive, clamped to [1e-6, 0.20].
    """
    model = home_model if side == "home" else away_model

    score_diff_pov = state["score_diff"] if side == "home" else -state["score_diff"]
    n_red_self     = state["n_red_h"]    if side == "home" else state["n_red_a"]
    n_red_opp      = state["n_red_a"]    if side == "home" else state["n_red_h"]

    feature_row = {
        "const":                   1.0,
        "score_diff_perspective":  score_diff_pov,
        "minutes_remaining":       state["minutes_remaining"],
        "n_red_self":              n_red_self,
        "n_red_opp":               n_red_opp,
        "halftime_passed":         int(state["halftime_passed"]),
        "is_home":                 int(side == "home"),
    }

    X_row = pd.DataFrame([feature_row])
    log_offset = np.log(max(base_rate_per_minute, 1e-9))

    # TODO: statsmodels predict() doesn't accept an offset directly at inference —
    # compute manually: η = X @ β + offset, λ = exp(η)
    beta = model.params
    eta  = float(X_row[beta.index].values @ beta.values) + log_offset
    lam  = np.exp(eta)

    return float(np.clip(lam, 1e-6, 0.20))
