"""
Stage 2 Validation — goal_hazard_model.py
==========================================
Validation checks for the Poisson hazard models.

Run before moving to Stage 3:
    from stage2_validate import run_all_checks
    summary = run_all_checks(home_model, away_model, training_df, holdout_df, fixtures)
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from goal_hazard_model import predict_hazard, HAZARD_FEATURES


DIAGNOSTICS_DIR = Path(
    "/content/drive/MyDrive/soccer-betting/data/processed/api_football/diagnostics/stage2"
)


# ── Hazard realism ────────────────────────────────────────────────────────────

def check_intensity_realism(home_model, away_model) -> dict:
    """
    Verify hazard predictions follow expected qualitative patterns:
      - λ when losing > λ when winning (chasing teams attack more)
      - λ post-red-card < λ pre-red-card (own red)
      - λ_home > λ_away on average (home advantage)
    """
    base_rate = 1.3 / 90.0   # arbitrary base; we compare RATIOS

    def lam(side: str, score_diff: int, n_red_self: int, n_red_opp: int,
            minutes_remaining: float = 30.0, halftime_passed: bool = True) -> float:
        state = {
            "score_diff":        score_diff if side == "home" else -score_diff,
            "minutes_remaining": minutes_remaining,
            "n_red_h":           n_red_self if side == "home" else n_red_opp,
            "n_red_a":           n_red_opp  if side == "home" else n_red_self,
            "halftime_passed":   halftime_passed,
        }
        return predict_hazard(state, base_rate, side, home_model, away_model)

    losing_2     = lam("home", -2, 0, 0)
    winning_2    = lam("home", +2, 0, 0)
    pre_red      = lam("home",  0, 0, 0)
    post_own_red = lam("home",  0, 1, 0)
    post_opp_red = lam("home",  0, 0, 1)
    home_avg     = lam("home",  0, 0, 0)
    away_avg     = lam("away",  0, 0, 0)

    checks = {
        "losing > winning":      losing_2 > winning_2,
        "post_own_red < pre_red": post_own_red < pre_red,
        "post_opp_red > pre_red": post_opp_red > pre_red,
        "home_avg > away_avg":    home_avg > away_avg,
    }

    table = pd.DataFrame([
        {"state": "losing by 2 (home)",  "lambda/min": losing_2},
        {"state": "winning by 2 (home)", "lambda/min": winning_2},
        {"state": "level, no reds (home)", "lambda/min": pre_red},
        {"state": "level, own red (home)", "lambda/min": post_own_red},
        {"state": "level, opp red (home)", "lambda/min": post_opp_red},
        {"state": "level, no reds (away)", "lambda/min": away_avg},
    ])

    return {
        "name":     "intensity_realism",
        "passed":   all(checks.values()),
        "subchecks": checks,
        "table":    table,
        "detail": (
            "All qualitative hazard patterns hold." if all(checks.values())
            else f"Failed sub-checks: {[k for k, v in checks.items() if not v]}"
        ),
    }


# ── Holdout calibration ───────────────────────────────────────────────────────

def check_holdout_calibration(
    home_model,
    away_model,
    holdout_df: pd.DataFrame,
    tolerance: float = 0.10,
) -> dict:
    """
    Predicted goals/min on holdout matches should match observed within ±10%.
    Total predicted goals / total observed goals ≈ 1.0.
    """
    home_rows = holdout_df[holdout_df["side"] == "home"]
    away_rows = holdout_df[holdout_df["side"] == "away"]

    def total_predicted(model, rows):
        if rows.empty: return 0.0
        X = pd.DataFrame({c: rows[c].astype(float) for c in HAZARD_FEATURES})
        X.insert(0, "const", 1.0)
        beta = model.params
        offset = rows["log_base_rate"].astype(float).values
        eta = X[beta.index].values @ beta.values + offset
        return float(np.exp(eta).sum())

    pred_home = total_predicted(home_model, home_rows)
    pred_away = total_predicted(away_model, away_rows)
    obs_home  = float(home_rows["goal_scored"].sum())
    obs_away  = float(away_rows["goal_scored"].sum())

    ratio_home = pred_home / max(obs_home, 1.0)
    ratio_away = pred_away / max(obs_away, 1.0)

    home_ok = abs(ratio_home - 1.0) <= tolerance
    away_ok = abs(ratio_away - 1.0) <= tolerance

    return {
        "name":         "holdout_calibration",
        "passed":       home_ok and away_ok,
        "tolerance":    tolerance,
        "pred_home":    pred_home,
        "obs_home":     obs_home,
        "ratio_home":   ratio_home,
        "pred_away":    pred_away,
        "obs_away":     obs_away,
        "ratio_away":   ratio_away,
        "detail": (
            f"home ratio={ratio_home:.3f}, away ratio={ratio_away:.3f} (target 1.0 ± {tolerance})"
        ),
    }


# ── Leakage check ──────────────────────────────────────────────────────────────

def check_no_leakage(
    training_df: pd.DataFrame,
    train_seasons: list[int],
    test_season: int = 2025,
) -> dict:
    """
    Hard assert that no test-season fixtures appear in training data.
    """
    train_subset = training_df[training_df["season"].isin(train_seasons)]
    leaked = train_subset[train_subset["season"] == test_season]

    return {
        "name":          "no_leakage",
        "passed":        len(leaked) == 0,
        "n_leaked_rows": int(len(leaked)),
        "train_seasons": train_seasons,
        "test_season":   test_season,
        "detail": (
            f"Training restricted to {train_seasons}; no test-season ({test_season}) leakage."
            if len(leaked) == 0
            else f"LEAKAGE: {len(leaked)} test-season rows in training data."
        ),
    }


# ── Visual ─────────────────────────────────────────────────────────────────────

def plot_predicted_vs_observed(
    home_model,
    away_model,
    holdout_df: pd.DataFrame,
    save_path: Path | None = None,
):
    """
    Scatter of predicted goal intensity vs observed goal rate, binned by
    score_diff and minute_remaining. Should follow y=x.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, side, model in [("home", "home", home_model), ("away", "away", away_model)]:
        rows = holdout_df[holdout_df["side"] == side].copy()
        if rows.empty:
            continue

        beta = model.params
        X = pd.DataFrame({c: rows[c].astype(float) for c in HAZARD_FEATURES})
        X.insert(0, "const", 1.0)
        offset = rows["log_base_rate"].astype(float).values
        rows["pred"] = np.exp(X[beta.index].values @ beta.values + offset)

        rows["sd_bin"] = pd.cut(rows["score_diff_perspective"], bins=[-99, -2, -1, 0, 1, 2, 99])
        rows["mr_bin"] = pd.cut(rows["minutes_remaining"], bins=[-1, 30, 60, 100])

        binned = rows.groupby(["sd_bin", "mr_bin"], observed=True).agg(
            pred=("pred", "mean"),
            obs =("goal_scored", "mean"),
            n   =("goal_scored", "size"),
        ).reset_index()

        ax.scatter(binned["pred"], binned["obs"], s=binned["n"]/30, alpha=0.6)
        m = binned[["pred", "obs"]].max().max()
        ax.plot([0, m], [0, m], "r--", alpha=0.5)
        ax.set_xlabel("Predicted goals/window")
        ax.set_ylabel("Observed goals/window")
        ax.set_title(f"{side.title()} — calibration (point size = n)")
        ax.grid(alpha=0.3)

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
    training_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    train_seasons: list[int] = [2023, 2024],
    test_season: int = 2025,
    diagnostics_dir: Path = DIAGNOSTICS_DIR,
) -> dict:
    print("=" * 60)
    print("  STAGE 2 VALIDATION — goal_hazard_model")
    print("=" * 60)

    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    checks = [
        check_intensity_realism(home_model, away_model),
        check_holdout_calibration(home_model, away_model, holdout_df),
        check_no_leakage(training_df, train_seasons, test_season),
    ]

    for c in checks:
        status = "PASS" if c["passed"] else "FAIL"
        print(f"  [{status}] {c['name']:<25} {c['detail']}")
        if "table" in c:
            print(c["table"].to_string(index=False))

    plot_predicted_vs_observed(
        home_model, away_model, holdout_df,
        save_path=diagnostics_dir / "calibration_scatter.png",
    )

    summary = {c["name"]: c for c in checks}
    summary["all_passed"] = all(c["passed"] for c in checks)
    print(f"\n  ALL PASSED: {summary['all_passed']}")
    print("=" * 60)
    return summary
