"""
Stage 5 Validation — prior_integration.py
==========================================
Validation checks for the GBM-anchor calibration.

Run before moving to Stage 6:
    from stage5_validate import run_all_checks
    summary = run_all_checks(calibrated_lambdas, gbm_predictions, home_model, away_model)
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from forward_simulator import simulate_match, market_probabilities
from prior_integration import _INITIAL_STATE_TEMPLATE


DIAGNOSTICS_DIR = Path(
    "/content/drive/MyDrive/soccer-betting/data/processed/api_football/diagnostics/stage5"
)


# ── Calibration accuracy ──────────────────────────────────────────────────────

def check_calibration_quality(
    calibrated_lambdas: pd.DataFrame,
    gbm_predictions: pd.DataFrame,
    home_model,
    away_model,
    n_sims: int = 4_000,
    n_sample: int = 100,
    rmse_threshold: float = 0.02,
    rng_seed: int = 0,
) -> dict:
    """
    For up to n_sample test matches, simulate at minute 0 with calibrated lambdas
    and compare against GBM probabilities. RMSE across {p_home, p_draw, p_away}
    should be < rmse_threshold.
    """
    merged = gbm_predictions.merge(calibrated_lambdas, on="fixture_id", how="inner")
    if merged.empty:
        return {
            "name": "calibration_quality", "passed": False,
            "detail": "No overlap between gbm_predictions and calibrated_lambdas",
        }

    sample = merged.sample(min(n_sample, len(merged)), random_state=rng_seed)
    rows = []

    for _, row in sample.iterrows():
        fid    = int(row["fixture_id"])
        lam_h  = float(row["lambda_h"])
        lam_a  = float(row["lambda_a"])
        gbm_h  = float(row.get("p_home", row.get("prob_home_win", np.nan)))
        gbm_d  = float(row.get("p_draw", row.get("prob_draw",     np.nan)))
        gbm_a  = float(row.get("p_away", row.get("prob_away_win", np.nan)))

        if any(np.isnan([gbm_h, gbm_d, gbm_a, lam_h, lam_a])):
            continue

        rng = np.random.default_rng(fid)
        D = simulate_match(
            _INITIAL_STATE_TEMPLATE.copy(), lam_h, lam_a,
            home_model, away_model, n_simulations=n_sims, rng=rng,
        )
        p = market_probabilities(D)

        rows.append({
            "fixture_id": fid,
            "gbm_h": gbm_h, "sim_h": p["home_win"],
            "gbm_d": gbm_d, "sim_d": p["draw"],
            "gbm_a": gbm_a, "sim_a": p["away_win"],
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return {"name": "calibration_quality", "passed": False, "detail": "No rows evaluated"}

    rmse_h = float(np.sqrt(((df["sim_h"] - df["gbm_h"]) ** 2).mean()))
    rmse_d = float(np.sqrt(((df["sim_d"] - df["gbm_d"]) ** 2).mean()))
    rmse_a = float(np.sqrt(((df["sim_a"] - df["gbm_a"]) ** 2).mean()))
    rmse_avg = (rmse_h + rmse_d + rmse_a) / 3

    return {
        "name":      "calibration_quality",
        "passed":    rmse_avg <= rmse_threshold,
        "n_sample":  len(df),
        "rmse_home": rmse_h, "rmse_draw": rmse_d, "rmse_away": rmse_a,
        "rmse_avg":  rmse_avg,
        "threshold": rmse_threshold,
        "scatter_df": df,
        "detail":    f"avg RMSE = {rmse_avg:.4f} (target ≤ {rmse_threshold}) over n={len(df)} matches",
    }


# ── Lambda realism ────────────────────────────────────────────────────────────

def check_lambda_realism(
    calibrated_lambdas: pd.DataFrame,
    bounds_per_match: tuple[float, float] = (0.3, 3.5),
) -> dict:
    """
    Calibrated λ_match (=lambda_per_min × 90) should be in [0.3, 3.5] for
    realistic football. Outside that range likely indicates failed calibration.
    """
    df = calibrated_lambdas.copy()
    df["lh_match"] = df["lambda_h"] * 90.0
    df["la_match"] = df["lambda_a"] * 90.0

    lo, hi = bounds_per_match
    bad = df[
        (df["lh_match"] < lo) | (df["lh_match"] > hi) |
        (df["la_match"] < lo) | (df["la_match"] > hi)
    ]

    return {
        "name":         "lambda_realism",
        "passed":       len(bad) == 0,
        "n_total":      len(df),
        "n_bad":        len(bad),
        "bad_examples": bad[["fixture_id", "lh_match", "la_match"]].head(10).to_dict("records"),
        "lh_summary":   df["lh_match"].describe().to_dict(),
        "la_summary":   df["la_match"].describe().to_dict(),
        "detail":       (f"All λ_match in [{lo}, {hi}]." if len(bad) == 0
                         else f"{len(bad)}/{len(df)} matches have λ outside [{lo}, {hi}]"),
    }


# ── Visual: GBM vs sim scatter ────────────────────────────────────────────────

def plot_gbm_vs_sim_scatter(
    calibration_df: pd.DataFrame,
    save_path: Path | None = None,
):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    pairs = [("gbm_h", "sim_h", "Home win"),
             ("gbm_d", "sim_d", "Draw"),
             ("gbm_a", "sim_a", "Away win")]

    for ax, (gx, sx, label) in zip(axes, pairs):
        ax.scatter(calibration_df[gx], calibration_df[sx], alpha=0.5, s=15)
        ax.plot([0, 1], [0, 1], "r--", alpha=0.7)
        ax.set_xlabel(f"GBM {label}")
        ax.set_ylabel(f"Sim {label}")
        ax.set_title(label)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)

    fig.suptitle("Calibrated simulator vs GBM at minute 0")
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120)
        print(f"  saved → {save_path}")
    plt.close(fig)
    return fig


# ── Orchestrator ───────────────────────────────────────────────────────────────

def run_all_checks(
    calibrated_lambdas: pd.DataFrame,
    gbm_predictions: pd.DataFrame,
    home_model,
    away_model,
    n_sample: int = 100,
    n_sims: int = 4_000,
    diagnostics_dir: Path = DIAGNOSTICS_DIR,
) -> dict:
    print("=" * 60)
    print("  STAGE 5 VALIDATION — prior_integration")
    print("=" * 60)

    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    quality = check_calibration_quality(
        calibrated_lambdas, gbm_predictions, home_model, away_model,
        n_sims=n_sims, n_sample=n_sample,
    )
    realism = check_lambda_realism(calibrated_lambdas)

    for c in (quality, realism):
        status = "PASS" if c["passed"] else "FAIL"
        print(f"  [{status}] {c['name']:<28} {c['detail']}")

    if "scatter_df" in quality and not quality["scatter_df"].empty:
        plot_gbm_vs_sim_scatter(
            quality["scatter_df"],
            save_path=diagnostics_dir / "gbm_vs_sim.png",
        )

    summary = {"calibration_quality": quality, "lambda_realism": realism}
    summary["all_passed"] = quality["passed"] and realism["passed"]
    print(f"\n  ALL PASSED: {summary['all_passed']}")
    print("=" * 60)
    return summary
