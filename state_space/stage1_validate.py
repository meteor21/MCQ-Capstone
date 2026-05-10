"""
Stage 1 Validation — state_extractor.py
=========================================
Validation checks for the state timeline extraction pipeline.

Run before moving to Stage 2:
    from stage1_validate import run_all_checks
    summary = run_all_checks(state_timeline, fixtures, events)
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from state_extractor import REGULATION_END, WINDOW_MINUTES


# ── Paths ─────────────────────────────────────────────────────────────────────

DIAGNOSTICS_DIR = Path(
    "/content/drive/MyDrive/soccer-betting/data/processed/api_football/diagnostics/stage1"
)


# ── Individual checks ─────────────────────────────────────────────────────────

def check_window_coverage(
    state_timeline: pd.DataFrame,
    window_minutes: int = WINDOW_MINUTES,
) -> dict:
    """
    Every match should have exactly (90 / window_minutes) + 1 windows.
    Flag matches with gaps or extra windows.
    """
    expected = (REGULATION_END // window_minutes) + 1   # 0, 5, ..., 90 → 19 with w=5

    counts = state_timeline.groupby("fixture_id")["minute"].nunique()
    bad_fids = counts[counts != expected].index.tolist()

    return {
        "name":         "window_coverage",
        "passed":       len(bad_fids) == 0,
        "expected":     expected,
        "n_matches":    int(counts.shape[0]),
        "n_bad":        len(bad_fids),
        "bad_examples": bad_fids[:10],
        "detail": (
            f"All {counts.shape[0]} matches have {expected} windows."
            if len(bad_fids) == 0
            else f"{len(bad_fids)} matches have wrong window count. First 10: {bad_fids[:10]}"
        ),
    }


def check_state_monotonicity(state_timeline: pd.DataFrame) -> dict:
    """
    Cumulative counters (n_red_*, n_subs_*, total_goals) must only increase
    or stay the same within a match — never decrease.
    """
    monotonic_cols = [
        "score_h", "score_a", "total_goals",
        "n_red_h", "n_red_a", "n_yellow_h", "n_yellow_a",
        "n_subs_h", "n_subs_a",
    ]
    monotonic_cols = [c for c in monotonic_cols if c in state_timeline.columns]

    violations: list[dict] = []
    for fid, grp in state_timeline.groupby("fixture_id"):
        grp = grp.sort_values("minute")
        for col in monotonic_cols:
            diffs = grp[col].diff().fillna(0)
            if (diffs < 0).any():
                violations.append({
                    "fixture_id": fid,
                    "column":     col,
                    "min_diff":   float(diffs.min()),
                })

    return {
        "name":         "state_monotonicity",
        "passed":       len(violations) == 0,
        "n_violations": len(violations),
        "examples":     violations[:10],
        "detail": (
            f"All cumulative counters monotonic across {state_timeline['fixture_id'].nunique()} matches."
            if len(violations) == 0
            else f"{len(violations)} monotonicity violations. First 10: {violations[:10]}"
        ),
    }


def check_score_consistency(
    state_timeline: pd.DataFrame,
    fixtures: pd.DataFrame,
    tolerance: int = 1,
) -> dict:
    """
    The final state's score should match fixtures.parquet's home_score/away_score
    within ±tolerance (own goals can credit the wrong team in events).
    """
    # Take the final window for each match
    final_state = (
        state_timeline.sort_values(["fixture_id", "minute"])
        .groupby("fixture_id")
        .last()
        .reset_index()[["fixture_id", "score_h", "score_a"]]
    )

    fix = fixtures[["fixture_id", "home_score", "away_score"]].copy()
    fix["home_score"] = pd.to_numeric(fix["home_score"], errors="coerce")
    fix["away_score"] = pd.to_numeric(fix["away_score"], errors="coerce")

    merged = final_state.merge(fix, on="fixture_id", how="inner")
    merged["home_diff"] = (merged["score_h"] - merged["home_score"]).abs()
    merged["away_diff"] = (merged["score_a"] - merged["away_score"]).abs()

    bad = merged[
        (merged["home_diff"] > tolerance) | (merged["away_diff"] > tolerance)
    ]

    return {
        "name":          "score_consistency",
        "passed":        len(bad) == 0,
        "tolerance":     tolerance,
        "n_checked":     len(merged),
        "n_mismatches":  len(bad),
        "bad_fids":      bad["fixture_id"].head(10).tolist(),
        "detail": (
            f"All {len(merged)} final scores match within ±{tolerance}."
            if len(bad) == 0
            else f"{len(bad)} matches exceed ±{tolerance} score tolerance."
        ),
    }


# ── Visual diagnostic ──────────────────────────────────────────────────────────

def plot_score_evolution(
    state_timeline: pd.DataFrame,
    events: pd.DataFrame,
    fixture_ids: list[int],
    save_path: Path | None = None,
):
    """
    Plot score_h and score_a over time as step functions for a list of matches,
    with red card events marked as vertical lines.
    """
    import matplotlib.pyplot as plt

    n = len(fixture_ids)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.6 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, fid in zip(axes, fixture_ids):
        tl = state_timeline[state_timeline["fixture_id"] == fid].sort_values("minute")
        if tl.empty:
            ax.set_title(f"Fixture {fid} — no timeline"); continue

        ax.step(tl["minute"], tl["score_h"], where="post", label="Home", linewidth=2)
        ax.step(tl["minute"], tl["score_a"], where="post", label="Away", linewidth=2)

        # Red card markers
        match_ev = events[events["fixture_id"] == fid]
        red_mask = (
            (match_ev["type"].str.lower() == "card") &
            (match_ev["detail"].fillna("").str.lower().isin({"red card", "second yellow card"}))
        )
        for _, ev_row in match_ev[red_mask].iterrows():
            ax.axvline(int(ev_row["minute"]), color="red", linestyle="--", alpha=0.5)

        ax.set_title(f"Fixture {fid}")
        ax.set_ylabel("Goals")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Minute")
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120)
        print(f"  saved → {save_path}")
    plt.close(fig)
    return fig


# ── Orchestrator ───────────────────────────────────────────────────────────────

def run_all_checks(
    state_timeline: pd.DataFrame,
    fixtures: pd.DataFrame,
    events: pd.DataFrame,
    diagnostics_dir: Path = DIAGNOSTICS_DIR,
    n_visual_samples: int = 5,
) -> dict:
    """
    Run all Stage 1 validation checks and produce visual diagnostics.

    Returns a summary dict with pass/fail per check.
    """
    print("=" * 60)
    print("  STAGE 1 VALIDATION — state_extractor")
    print("=" * 60)

    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    checks = [
        check_window_coverage(state_timeline),
        check_state_monotonicity(state_timeline),
        check_score_consistency(state_timeline, fixtures),
    ]

    for c in checks:
        status = "PASS" if c["passed"] else "FAIL"
        print(f"  [{status}] {c['name']:<25} {c['detail']}")

    # Visual diagnostic
    sample_ids = (
        fixtures.sample(min(n_visual_samples, len(fixtures)), random_state=42)
        ["fixture_id"].tolist()
    )
    print(f"\n  Generating score-evolution plot for {len(sample_ids)} sample matches...")
    plot_score_evolution(
        state_timeline, events, sample_ids,
        save_path=diagnostics_dir / "score_evolution_samples.png",
    )

    summary = {c["name"]: c for c in checks}
    summary["all_passed"] = all(c["passed"] for c in checks)
    print(f"\n  ALL PASSED: {summary['all_passed']}")
    print("=" * 60)
    return summary
