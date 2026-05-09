# State-Space Betting Engine

A 7-stage probabilistic pipeline for in-game soccer betting analysis, built on
credit-risk pricing principles.

## Core principle

Each betting market is a Bernoulli indicator `D_j ∈ {0, 1}`. The engine produces:

- **Probability vector**  `E[D] = (p_1, ..., p_M)` — the marginals.
- **Correlation matrix**  `C` — the M × M Pearson correlation between markets.

**Both come from the same Monte Carlo simulation.** We run `N` forward
simulations from the current match state to minute 95. The (N × M) binary
matrix `D` of indicator values gives us probabilities (column means) and
correlations (between columns) directly — no independence assumption, no
separate correlation estimation.

This is the credit-CDO pricing analogy: the probability and correlation of
joint losses are estimated from a single forward simulation of the latent
risk factors.

## Live odds, deliberately, are NOT in the model

Odds enter ONLY in Stage 7 (bet sizing / decision). The model produces
`(p̂, C)` purely from match state and fitted dynamics; the bet logic
translates `(p̂, C, odds)` into recommendations. This separation:

- Prevents odds-fitting (the model can't be biased toward market consensus).
- Lets us evaluate model quality without backtesting bets.
- Cleanly decouples "what's the probability?" from "should we bet?".

## File map

```
state_space/
├── README.md
├── run_pipeline.py            # Top-level orchestrator
├── state_extractor.py         # Stage 1 — events → state timeline
├── goal_hazard_model.py       # Stage 2 — Poisson regression on state
├── forward_simulator.py       # Stage 3 — Monte Carlo continuation simulator
├── correlation_engine.py      # Stage 4 — market correlation analysis
├── prior_integration.py       # Stage 5 — calibrate base rates to GBM
├── replay_engine.py           # Stage 6 — full match replay (DELIVERABLE)
├── bet_logic.py               # Stage 7 — Kelly sizing with correlation filter
├── stage1_validate.py         # Validation suites (one per stage)
├── stage2_validate.py
├── stage3_validate.py
├── stage4_validate.py
├── stage5_validate.py
├── stage6_validate.py
└── stage7_validate.py
```

## Pipeline stages

### Stage 1 — `state_extractor.py`
Walks events chronologically and produces a timeline of state vectors at
fixed window boundaries (default 5 min). Each state captures everything
that has happened to that point: scores, cards, subs, last-goal timing,
VAR pending, etc. Schema documented in module docstring.

Output: `state_timeline.parquet` — one row per `(fixture_id, window_minute)`.

### Stage 2 — `goal_hazard_model.py`
Two Poisson GLMs (home, away) for goal intensity per minute as a function of state:

```
log(λ) = β₀ + β · state_features + log(base_rate)
```

State features: `score_diff_perspective`, `minutes_remaining`, `n_red_self`,
`n_red_opp`, `halftime_passed`, `is_home`. The `base_rate` offset lets the GLM
learn STATE adjustments while keeping per-match scaling open for Stage 5.

Trained ONLY on `train_seasons` (default `[2023, 2024]`); test season `2025`
is locked away. Output: `hazard_models.pkl`.

### Stage 3 — `forward_simulator.py`
Given `(initial_state, λ_h_base, λ_a_base, hazard_models)`, runs N minute-
by-minute Bernoulli simulations to minute 95. Returns `(N, M)` int8 matrix `D`.

Markets evaluated: `home_win, draw, away_win, btts, no_btts, over_05/15/25/35,
home/away_clean_sheet, next_goal_home/away, no_more_goals, dc_1X/X2/12,
home/away_minus_15`.

`market_probabilities(D) → dict` returns the column means.

### Stage 4 — `correlation_engine.py`
Computes Pearson correlation across columns of `D`, identifies clusters of
related markets (single-linkage hierarchical clustering with `|ρ| > 0.6`
threshold by default), and produces a redundancy report.

Edge interpretation:
- `ρ = +1`: same bet, never stack.
- `ρ =  0`: independent, free to combine.
- `ρ = -1`: mutually exclusive, can't both win.

### Stage 5 — `prior_integration.py`
Calibrates `(λ_h, λ_a)` so that the simulator at minute 0 reproduces the
GBM's pre-match `(p_home, p_draw, p_away)`. Uses scipy L-BFGS-B with squared-
error loss. This anchors the simulator on the GBM at kickoff; from there the
simulator diverges in a state-conditional, event-driven way.

Output: `calibrated_lambdas.parquet` with `(fixture_id, λ_h, λ_a)` per match.

### Stage 6 — `replay_engine.py` — **PRIMARY DELIVERABLE**
For each test fixture, walks every window boundary, gets the state, runs the
forward simulator, and emits a row with state features, all market
probabilities, and key correlation entries.

Output: `replay_results.parquet`.

#### Schema of `replay_results.parquet`

| Column | Type | Notes |
| ------ | ---- | ----- |
| `fixture_id` | int | Match ID |
| `minute` | int | Window start (0, 5, ..., 90) |
| `score_h` | int | Home goals up to this point |
| `score_a` | int | Away goals up to this point |
| `score_diff` | int | `score_h - score_a` |
| `total_goals` | int | |
| `n_red_h`, `n_red_a` | int | Cumulative reds |
| `minutes_remaining` | float | `90 - minute` |
| `halftime_passed` | bool | |
| `p_<market>` | float | One column per market in `MARKET_NAMES` |
| `corr_<m1>__<m2>` | float | Selected correlation pairs (configurable) |
| `cluster_<market>` | int | Cluster ID at this state |

### Stage 7 — `bet_logic.py`
Pure decision layer. Takes `(p̂, cluster_map, market_odds_dict)` and produces
sized recommendations using fractional Kelly with a correlation-cluster filter
(at most one bet per cluster, the highest-edge one).

Future work: full mean-variance Kelly across the recommendation set (currently
greedy/clustered). Backtesting requires real odds time-series, not in scope
for the 2-day deliverable.

## Validation

**Run validation between stages.** Bugs at Stage 2 poison everything downstream.

Each `stageN_validate.py` exposes:

```python
from stageN_validate import run_all_checks
summary = run_all_checks(...)   # See module signatures
assert summary['all_passed']
```

Each suite runs:
- Algebraic / structural checks (passes/fails returned as dict).
- Visual diagnostics (saved to
  `/content/drive/MyDrive/soccer-betting/data/processed/api_football/diagnostics/stageN/`).

Highlights:
- **Stage 1**: window coverage, monotonic counters, score consistency vs fixtures, score-evolution plots.
- **Stage 2**: hazard realism (winning < losing, post-red-card drop), holdout calibration ratio, leakage assert.
- **Stage 3**: indicator logic, sim-count stability, final-score histogram.
- **Stage 4**: PSD, symmetric, diagonal-1; known correlations (`btts ⫫ over_25 > 0.5`, etc.); heatmap + dendrogram.
- **Stage 5**: GBM-vs-sim RMSE < 0.02 across (home, draw, away); λ_match ∈ [0.3, 3.5].
- **Stage 6**: minute-0 anchor matches GBM; probabilities all in [0, 1]; winner's prob rises in decisive matches; trajectory plot.
- **Stage 7**: edge calc, Kelly cap, cluster-filter logic, example output table.

## How to run

```python
# Colab
from run_pipeline import (
    load_data, run_stage1, run_stage2, run_stage5, run_stage6, run_stage7_demo,
)

data = load_data()

# Stage 1
state_timeline = run_stage1(data)
from stage1_validate import run_all_checks as v1
assert v1(state_timeline, data["fixtures"], data["events"])["all_passed"]

# Stage 2
home_model, away_model, training_df = run_stage2(state_timeline, data)
from stage2_validate import run_all_checks as v2
holdout_df = training_df[training_df["season"] == 2025]
assert v2(home_model, away_model, training_df, holdout_df)["all_passed"]

# Stage 5
calibrated = run_stage5(home_model, away_model, data)
from stage5_validate import run_all_checks as v5
assert v5(calibrated, data["gbm_predictions"], home_model, away_model)["all_passed"]

# Stage 6 — main deliverable
replay_results = run_stage6(state_timeline, home_model, away_model, calibrated, data)
from stage6_validate import run_all_checks as v6
v6(replay_results, data["gbm_predictions"], data["fixtures"], data["events"])

# Stage 7 — demo
run_stage7_demo(replay_results)
```

## Configuration

In `run_pipeline.py`:

```python
WINDOW_MINS:        int   = 5      # window granularity
N_SIMULATIONS:      int   = 5_000  # sims per replay window
N_CALIBRATION_SIMS: int   = 2_000  # sims per calibration eval
EDGE_THRESHOLD:     float = 0.03   # 3% min edge to bet
CORR_THRESHOLD:     float = 0.6    # |ρ| above this → same cluster
KELLY_FRACTIONAL:   float = 0.25   # quarter Kelly default
TRAIN_SEASONS:      list  = [2023, 2024]
TEST_SEASON:        int   = 2025
```

## Dependencies

```
numpy
pandas
scipy
statsmodels   # GLM Poisson
scikit-learn  # used in earlier feature pipeline
matplotlib
seaborn       # heatmaps in validation
```
