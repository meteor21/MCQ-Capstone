"""
Stage 3 Validation — forward_simulator.py
==========================================
Validation checks for the Monte Carlo forward simulator.

Run before moving to Stage 4:
    from stage3_validate import run_all_checks
    summary = run_all_checks(home_model, away_model, sample_state, fixtures, events)
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd

from forward_simulator import (
    simulate_match,
    market_probabilities,
    compute_market_indicators,
    MARKET_NAMES,
    MARKET_INDEX,
)


DIAGNOSTICS_DIR = Path(
    "/content/drive/MyDrive/soccer-betting/data/processed/api_football/diagnostics/stage3"
)


# ── Distribution sanity ───────────────────────────────────────────────────────

def check_simulation_distribution(
    home_model,
    away_model,
    initial_state: dict,
    lambda_h_base: float,
    lambda_a_base: float,
    n_sims: int = 10_000,
    tolerance: float = 0.20,
) -> dict:
    """
    Simulate from minute 0 with given base lambdas. Average final score
    should be close to 90 × λ_base (no systematic drift).
    """
    rng = np.random.default_rng(0)
    D = simulate_match(
        initial_state, lambda_h_base, lambda_a_base,
        home_model, away_model, n_simulations=n_sims, rng=rng,
    )

    # We can't recover scores from D directly; re-simulate to capture them
    # Quick re-derivation: over_05 + over_15 + over_25 + over_35 ≥ home + away count
    # Simpler: just compute expected total from probabilities and λ_base × 90
    p = market_probabilities(D)

    expected_total = (lambda_h_base + lambda_a_base) * 90.0

    # Use over/under probs to estimate mean total goals
    # E[total] ≈ Σ_k k × P(total = k); approximate with over_xy ladder
    p_over = [p["over_05"], p["over_15"], p["over_25"], p["over_35"]]
    # E[total] = Σ_n P(total > n)  (since totals are integer)
    estimated_mean_total = sum(p_over) + p["over_35"] * 1.5  # tail correction

    ratio = estimated_mean_total / max(expected_total, 1e-6)
    passed = abs(ratio - 1.0) <= tolerance

    return {
        "name":           "simulation_distribution",
        "passed":         passed,
        "expected_total": expected_total,
        "sim_total_est":  estimated_mean_total,
        "ratio":          ratio,
        "tolerance":      tolerance,
        "detail":         f"sim/expected total = {ratio:.3f} (target 1.0 ± {tolerance})",
    }


# ── Indicator logic ───────────────────────────────────────────────────────────

def check_market_indicator_logic() -> dict:
    """
    Hand-craft known final scores and verify the market indicator function
    produces the right binary values.
    """
    cases = [
        # (final_h, final_a, start_h, start_a, expected market values)
        (2, 1, 0, 0, {"home_win": 1, "draw": 0, "away_win": 0, "btts": 1,
                      "over_25": 0, "over_15": 1, "home_clean_sheet": 0,
                      "next_goal_home": 1}),
        (0, 0, 0, 0, {"home_win": 0, "draw": 1, "away_win": 0, "btts": 0,
                      "over_05": 0, "home_clean_sheet": 1, "no_more_goals": 1}),
        (3, 0, 1, 0, {"home_win": 1, "home_minus_15": 1, "btts": 0,
                      "next_goal_home": 1, "next_goal_away": 0,
                      "over_25": 1, "away_clean_sheet": 1}),
    ]

    failures = []
    for fh, fa, sh, sa, expected in cases:
        D = compute_market_indicators(
            np.array([fh]), np.array([fa]), goals_at_start_h=sh, goals_at_start_a=sa,
        )
        for mkt, want in expected.items():
            got = int(D[0, MARKET_INDEX[mkt]])
            if got != want:
                failures.append({
                    "final": (fh, fa), "start": (sh, sa),
                    "market": mkt, "want": want, "got": got,
                })

    return {
        "name":     "market_indicator_logic",
        "passed":   len(failures) == 0,
        "n_cases":  sum(len(c[4]) for c in cases),
        "failures": failures,
        "detail":   "All hand-crafted indicator cases match." if not failures
                    else f"{len(failures)} indicator mismatches: {failures[:5]}",
    }


# ── Sample row inspection ──────────────────────────────────────────────────────

def show_sample_simulation_rows(D: np.ndarray, n: int = 5) -> pd.DataFrame:
    """Print n sample simulation rows for human eyeballing."""
    sample_idx = np.random.default_rng(0).choice(D.shape[0], size=min(n, D.shape[0]), replace=False)
    return pd.DataFrame(D[sample_idx], columns=MARKET_NAMES)


# ── Stability under N ─────────────────────────────────────────────────────────

def check_sim_count_stability(
    home_model,
    away_model,
    initial_state: dict,
    lambda_h_base: float,
    lambda_a_base: float,
    n_sims_list: Sequence[int] = (1_000, 5_000, 10_000, 20_000),
    tolerance: float = 0.02,
) -> dict:
    """
    Run simulator at multiple n_sims values. p_home should converge.
    """
    results = []
    for n in n_sims_list:
        rng = np.random.default_rng(42)
        D = simulate_match(
            initial_state, lambda_h_base, lambda_a_base,
            home_model, away_model, n_simulations=n, rng=rng,
        )
        p = market_probabilities(D)
        results.append({"n_sims": n, "p_home": p["home_win"], "p_draw": p["draw"], "p_over_25": p["over_25"]})

    df = pd.DataFrame(results)
    spread = df["p_home"].max() - df["p_home"].min()
    return {
        "name":      "sim_count_stability",
        "passed":    spread <= tolerance,
        "tolerance": tolerance,
        "spread":    spread,
        "table":     df,
        "detail":    f"p_home spread across n_sims = {spread:.4f} (target ≤ {tolerance})",
    }


# ── Visual: final score histogram ──────────────────────────────────────────────

def plot_final_score_histogram(
    home_model,
    away_model,
    initial_state: dict,
    lambda_h_base: float,
    lambda_a_base: float,
    n_sims: int = 10_000,
    save_path: Path | None = None,
):
    """
    Histogram of (score_h, score_a) from N simulations. Should look bivariate Poisson.
    """
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    # Re-run a stripped simulator to capture raw scores
    sim_h = np.full(n_sims, initial_state["score_h"], dtype=np.int32)
    sim_a = np.full(n_sims, initial_state["score_a"], dtype=np.int32)
    n_red_h = int(initial_state["n_red_h"])
    n_red_a = int(initial_state["n_red_a"])
    start_minute = int(initial_state["minute"])

    from goal_hazard_model import predict_hazard
    for minute in range(start_minute, 96):
        state_for_hazard = {
            "score_diff":        float(np.mean(sim_h - sim_a)),
            "minutes_remaining": max(0, 90 - minute),
            "n_red_h":           n_red_h,
            "n_red_a":           n_red_a,
            "halftime_passed":   minute >= 45,
        }
        lam_h = predict_hazard(state_for_hazard, lambda_h_base, "home", home_model, away_model)
        lam_a = predict_hazard(state_for_hazard, lambda_a_base, "away", home_model, away_model)
        sim_h += (rng.random(n_sims) < lam_h).astype(np.int32)
        sim_a += (rng.random(n_sims) < lam_a).astype(np.int32)

    fig, ax = plt.subplots(figsize=(7, 6))
    h2d, x_edges, y_edges = np.histogram2d(
        sim_h, sim_a, bins=[range(0, 8), range(0, 8)],
    )
    im = ax.imshow(h2d.T / n_sims, origin="lower", cmap="Blues", aspect="equal")
    ax.set_xlabel("Home goals")
    ax.set_ylabel("Away goals")
    ax.set_title(f"Final score distribution (n={n_sims})\nλ_h={lambda_h_base*90:.2f}, λ_a={lambda_a_base*90:.2f} per match")
    ax.set_xticks(range(7)); ax.set_yticks(range(7))
    plt.colorbar(im, ax=ax, label="Probability")

    for i in range(7):
        for j in range(7):
            v = h2d[i, j] / n_sims
            if v > 0.01:
                ax.text(i, j, f"{v:.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120)
        print(f"  saved → {save_path}")
    plt.close(fig)
    return fig


# ── Orchestrator ───────────────────────────────────────────────────────────────

def run_all_checks(
    home_model,
    away_model,
    sample_initial_state: dict,
    lambda_h_base: float = 1.4 / 90,
    lambda_a_base: float = 1.1 / 90,
    n_sims: int = 10_000,
    diagnostics_dir: Path = DIAGNOSTICS_DIR,
) -> dict:
    print("=" * 60)
    print("  STAGE 3 VALIDATION — forward_simulator")
    print("=" * 60)

    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    checks = [
        check_market_indicator_logic(),
        check_simulation_distribution(
            home_model, away_model, sample_initial_state,
            lambda_h_base, lambda_a_base, n_sims,
        ),
        check_sim_count_stability(
            home_model, away_model, sample_initial_state,
            lambda_h_base, lambda_a_base,
        ),
    ]

    for c in checks:
        status = "PASS" if c["passed"] else "FAIL"
        print(f"  [{status}] {c['name']:<28} {c['detail']}")
        if "table" in c:
            print(c["table"].to_string(index=False))

    # Sample row inspection
    rng = np.random.default_rng(0)
    D = simulate_match(
        sample_initial_state, lambda_h_base, lambda_a_base,
        home_model, away_model, n_simulations=n_sims, rng=rng,
    )
    print("\n  Sample simulation rows (5 random):")
    print(show_sample_simulation_rows(D).to_string())

    plot_final_score_histogram(
        home_model, away_model, sample_initial_state,
        lambda_h_base, lambda_a_base, n_sims=n_sims,
        save_path=diagnostics_dir / "final_score_histogram.png",
    )

    summary = {c["name"]: c for c in checks}
    summary["all_passed"] = all(c["passed"] for c in checks)
    print(f"\n  ALL PASSED: {summary['all_passed']}")
    print("=" * 60)
    return summary
