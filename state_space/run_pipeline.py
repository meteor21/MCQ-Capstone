"""
Run Pipeline — top-level orchestrator
=======================================
Drives the full state-space betting engine end-to-end.

Each stage is independently runnable and validated. The default flow is:
  1. Load existing parquet data
  2. Build state timelines for all fixtures (Stage 1) → state_timeline.parquet
  3. Fit hazard models on train seasons (Stage 2) → hazard_models.pkl
  4. Pre-compute (λ_h, λ_a) for every test fixture (Stage 5) → calibrated_lambdas.parquet
  5. Replay each test fixture window-by-window (Stage 6) → replay_results.parquet
  6. (Optional) Demo Stage 7 bet logic with proxied odds

Validation between stages: see stage{N}_validate.py modules. The user is
expected to call run_all_checks() after each stage and confirm pass before
proceeding.

Usage
-----
    # In Colab:
    !python run_pipeline.py

    # Or stage by stage:
    from run_pipeline import (
        load_data, run_stage1, run_stage2, run_stage5, run_stage6, run_stage7_demo,
    )
    data = load_data()
    state_timeline = run_stage1(data)
    home_model, away_model = run_stage2(state_timeline, data)
    calibrated = run_stage5(home_model, away_model, data)
    replay_df = run_stage6(state_timeline, home_model, away_model, calibrated, data)
"""

from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import pandas as pd

# Stage imports
from state_extractor      import build_all_state_timelines, get_state_at, WINDOW_MINUTES
from goal_hazard_model    import (
    prepare_hazard_training_data, fit_hazard_models, load_hazard_models,
    DEFAULT_TRAIN_SEASONS, MODEL_SAVE_PATH,
)
from forward_simulator    import simulate_match, market_probabilities, MARKET_NAMES
from correlation_engine   import full_correlation_analysis
from prior_integration    import precompute_all_lambdas
from replay_engine        import replay_test_set
from bet_logic            import recommend_bets, format_recommendations_text


# ── Configurable flags ─────────────────────────────────────────────────────────

DATA_DIR = Path("/content/drive/MyDrive/soccer-betting/data/processed/api_football")

WINDOW_MINS:        int   = WINDOW_MINUTES   # 5
N_SIMULATIONS:      int   = 5_000
N_CALIBRATION_SIMS: int   = 2_000
EDGE_THRESHOLD:     float = 0.03
CORR_THRESHOLD:     float = 0.6
KELLY_FRACTIONAL:   float = 0.25

TRAIN_SEASONS = [2023, 2024]
TEST_SEASON   = 2025


# ── Output paths ───────────────────────────────────────────────────────────────

STATE_TIMELINE_PATH      = DATA_DIR / "state_timeline.parquet"
HAZARD_MODELS_PATH       = DATA_DIR / "hazard_models.pkl"
CALIBRATED_LAMBDAS_PATH  = DATA_DIR / "calibrated_lambdas.parquet"
REPLAY_RESULTS_PATH      = DATA_DIR / "replay_results.parquet"


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data() -> dict:
    """Load all parquet inputs and return as a dict."""
    print("Loading inputs...")
    fixtures        = pd.read_parquet(DATA_DIR / "fixtures_usable.parquet")
    events          = pd.read_parquet(DATA_DIR / "events_clean.parquet")
    statistics      = pd.read_parquet(DATA_DIR / "statistics.parquet")

    # Optional features (some pipelines may skip generating these)
    dc_features         = _try_load(DATA_DIR / "dc_features.parquet")
    elo_features        = _try_load(DATA_DIR / "elo_features.parquet")
    standings_features  = _try_load(DATA_DIR / "standings_features.parquet")
    test_predictions    = pd.read_parquet(DATA_DIR / "test_predictions.parquet")

    print(f"  fixtures:        {fixtures.shape}")
    print(f"  events:          {events.shape}")
    print(f"  statistics:      {statistics.shape}")
    print(f"  test_predictions:{test_predictions.shape}")

    return {
        "fixtures":         fixtures,
        "events":           events,
        "statistics":       statistics,
        "dc_features":      dc_features,
        "elo_features":     elo_features,
        "standings":        standings_features,
        "gbm_predictions":  test_predictions,
    }


def _try_load(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_parquet(path)
    print(f"  [INFO] optional file not found, skipping: {path.name}")
    return None


# ── Stage runners ──────────────────────────────────────────────────────────────

def run_stage1(data: dict) -> pd.DataFrame:
    """
    Stage 1 — build state timelines for all matches.

    To run validation:
        from stage1_validate import run_all_checks
        run_all_checks(state_timeline, data['fixtures'], data['events'])
    """
    print("\n[Stage 1] Building state timelines...")
    return build_all_state_timelines(
        fixtures      = data["fixtures"],
        events        = data["events"],
        window_minutes= WINDOW_MINS,
        output_path   = STATE_TIMELINE_PATH,
    )


def run_stage2(state_timeline: pd.DataFrame, data: dict):
    """
    Stage 2 — fit Poisson hazard models on training seasons only.

    To run validation:
        from stage2_validate import run_all_checks
        run_all_checks(home_model, away_model, training_df, holdout_df)
    """
    print("\n[Stage 2] Preparing hazard training data...")
    training_df = prepare_hazard_training_data(
        state_timeline = state_timeline,
        events         = data["events"],
        fixtures       = data["fixtures"],
        dc_features    = data.get("dc_features"),
    )
    print(f"  training rows (all seasons): {len(training_df)}")

    print(f"\n[Stage 2] Fitting hazard models on seasons {TRAIN_SEASONS}...")
    home_model, away_model = fit_hazard_models(
        training_df, train_seasons=TRAIN_SEASONS, save_path=HAZARD_MODELS_PATH,
    )

    return home_model, away_model, training_df


def run_stage5(home_model, away_model, data: dict) -> pd.DataFrame:
    """
    Stage 5 — pre-compute calibrated (λ_h, λ_a) for every test fixture.

    To run validation:
        from stage5_validate import run_all_checks
        run_all_checks(calibrated_lambdas, data['gbm_predictions'], home_model, away_model)
    """
    print("\n[Stage 5] Calibrating base rates against GBM predictions...")
    return precompute_all_lambdas(
        gbm_predictions    = data["gbm_predictions"],
        home_hazard_model  = home_model,
        away_hazard_model  = away_model,
        n_sims             = N_CALIBRATION_SIMS,
        output_path        = CALIBRATED_LAMBDAS_PATH,
    )


def run_stage6(
    state_timeline: pd.DataFrame,
    home_model,
    away_model,
    calibrated_lambdas: pd.DataFrame,
    data: dict,
) -> pd.DataFrame:
    """
    Stage 6 — replay every test fixture window-by-window.

    To run validation:
        from stage6_validate import run_all_checks
        run_all_checks(replay_results, data['gbm_predictions'], data['fixtures'], data['events'])
    """
    test_fixtures = data["fixtures"][data["fixtures"]["season"] == TEST_SEASON]
    print(f"\n[Stage 6] Replaying {len(test_fixtures)} test fixtures...")

    return replay_test_set(
        test_fixtures        = test_fixtures,
        state_timeline       = state_timeline,
        home_hazard_model    = home_model,
        away_hazard_model    = away_model,
        gbm_predictions      = data["gbm_predictions"],
        calibrated_lambdas   = calibrated_lambdas,
        n_sims               = N_SIMULATIONS,
        window_minutes       = WINDOW_MINS,
        output_path          = REPLAY_RESULTS_PATH,
    )


def run_stage7_demo(
    replay_results: pd.DataFrame,
    sample_fixture_id: int | None = None,
    sample_minute: int = 30,
):
    """
    Stage 7 demo — produce a recommendation table for one (fixture, minute) using
    proxied odds. This is illustrative only; real backtest is future work.
    """
    print("\n[Stage 7] Demo: bet recommendations with proxied odds...")
    if sample_fixture_id is None:
        sample_fixture_id = int(replay_results["fixture_id"].iloc[0])

    row = replay_results[
        (replay_results["fixture_id"] == sample_fixture_id) &
        (replay_results["minute"] == sample_minute)
    ]
    if row.empty:
        print(f"  no replay row for fixture={sample_fixture_id}, minute={sample_minute}")
        return None
    r = row.iloc[0]

    # Build p_vector and cluster_map from replay row
    p_vector    = {m: float(r[f"p_{m}"]) for m in MARKET_NAMES if f"p_{m}" in r.index}
    cluster_map = {m: int(r.get(f"cluster_{m}", -1)) for m in MARKET_NAMES}

    # TODO: replace with real odds source. For demo, invert probabilities with 5% overround.
    odds_dict = {m: float(round(1.0 / max(p, 0.02) * 0.95, 2))
                 for m, p in p_vector.items()}

    recs = recommend_bets(
        p_vector=p_vector, cluster_map=cluster_map,
        market_odds_dict=odds_dict,
        edge_threshold=EDGE_THRESHOLD,
        corr_cluster_threshold=CORR_THRESHOLD,
        kelly_fractional=KELLY_FRACTIONAL,
    )

    print(format_recommendations_text(
        recs,
        fixture_label=f"fixture_{sample_fixture_id}",
        minute=sample_minute,
    ))
    return recs


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    """Run the complete pipeline end-to-end. Comment out stages already complete."""
    data = load_data()

    # Stage 1
    state_timeline = run_stage1(data)
    # state_timeline = pd.read_parquet(STATE_TIMELINE_PATH)   # ← skip Stage 1 if already done

    # Stage 2
    home_model, away_model, training_df = run_stage2(state_timeline, data)
    # home_model, away_model = load_hazard_models(HAZARD_MODELS_PATH)

    # Stage 3 has no batch step — used inside Stage 5/6.
    # Stage 4 has no batch step — used inside Stage 6.

    # Stage 5
    calibrated_lambdas = run_stage5(home_model, away_model, data)
    # calibrated_lambdas = pd.read_parquet(CALIBRATED_LAMBDAS_PATH)

    # Stage 6 — main deliverable
    replay_results = run_stage6(state_timeline, home_model, away_model, calibrated_lambdas, data)
    # replay_results = pd.read_parquet(REPLAY_RESULTS_PATH)

    # Stage 7 — demo
    run_stage7_demo(replay_results)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
