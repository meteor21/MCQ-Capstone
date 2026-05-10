"""
Stage 6 Validation — replay_engine.py
=======================================
Validation checks for the per-match replay output.

Run before moving to Stage 7:
    from stage6_validate import run_all_checks
    summary = run_all_checks(replay_results, gbm_predictions, fixtures, events)
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from forward_simulator import MARKET_NAMES


DIAGNOSTICS_DIR = Path(
    "/content/drive/MyDrive/soccer-betting/data/processed/api_football/diagnostics/stage6"
)


# ── Anchor at minute 0 ────────────────────────────────────────────────────────

def check_replay_anchoring(
    replay_results: pd.DataFrame,
    gbm_predictions: pd.DataFrame,
    rmse_threshold: float = 0.04,
) -> dict:
    """
    At minute 0, simulator probabilities should equal GBM probabilities
    (within calibration tolerance ~0.04 since calibration is approximate).
    """
    minute0 = replay_results[replay_results["minute"] == 0].copy()
    if minute0.empty:
        return {"name": "replay_anchoring", "passed": False,
                "detail": "No minute-0 rows in replay_results"}

    gbm = gbm_predictions.copy()
    p_home_col = "p_home" if "p_home" in gbm.columns else "prob_home_win"
    p_draw_col = "p_draw" if "p_draw" in gbm.columns else "prob_draw"
    p_away_col = "p_away" if "p_away" in gbm.columns else "prob_away_win"

    merged = minute0.merge(
        gbm[["fixture_id", p_home_col, p_draw_col, p_away_col]],
        on="fixture_id", how="inner",
    )

    rmse_h = float(np.sqrt(((merged["p_home_win"] - merged[p_home_col]) ** 2).mean()))
    rmse_d = float(np.sqrt(((merged["p_draw"]     - merged[p_draw_col]) ** 2).mean()))
    rmse_a = float(np.sqrt(((merged["p_away_win"] - merged[p_away_col]) ** 2).mean()))
    avg = (rmse_h + rmse_d + rmse_a) / 3

    return {
        "name":      "replay_anchoring",
        "passed":    avg <= rmse_threshold,
        "rmse_h":    rmse_h, "rmse_d": rmse_d, "rmse_a": rmse_a,
        "rmse_avg":  avg,
        "threshold": rmse_threshold,
        "n":         len(merged),
        "detail":    f"minute-0 RMSE vs GBM = {avg:.4f} (target ≤ {rmse_threshold}) over n={len(merged)}",
    }


# ── Probability bounds ─────────────────────────────────────────────────────────

def check_no_negative_no_over_one(replay_results: pd.DataFrame) -> dict:
    """
    All probability columns must be in (0, 1) at every state. No NaN, no negative, no >1.
    """
    p_cols = [f"p_{m}" for m in MARKET_NAMES if f"p_{m}" in replay_results.columns]
    bad = []

    for col in p_cols:
        s = replay_results[col]
        if s.isna().any():        bad.append((col, "NaN", int(s.isna().sum())))
        if (s < 0).any():         bad.append((col, "<0",  int((s < 0).sum())))
        if (s > 1).any():         bad.append((col, ">1",  int((s > 1).sum())))

    return {
        "name":     "no_negative_no_over_one",
        "passed":   len(bad) == 0,
        "n_cols":   len(p_cols),
        "violations": bad,
        "detail":   ("All probabilities in [0, 1], no NaN." if not bad
                     else f"Bound violations: {bad[:10]}"),
    }


# ── Probability-evolution sanity ───────────────────────────────────────────────

def check_probability_evolution(
    replay_results: pd.DataFrame,
    fixtures: pd.DataFrame,
    n_sample: int = 5,
) -> dict:
    """
    Pick matches with definitive outcomes (margin ≥ 2). Verify that
    the winner's probability rises monotonically (loosely) toward 1
    by minute 90 and the loser's drops toward 0.
    """
    fix = fixtures[["fixture_id", "home_score", "away_score"]].dropna().copy()
    fix["home_score"] = pd.to_numeric(fix["home_score"], errors="coerce")
    fix["away_score"] = pd.to_numeric(fix["away_score"], errors="coerce")
    fix["margin"]     = (fix["home_score"] - fix["away_score"]).abs()

    decisive = fix[fix["margin"] >= 2]
    if decisive.empty:
        return {"name": "probability_evolution", "passed": False,
                "detail": "No decisive matches found"}

    sample = decisive.sample(min(n_sample, len(decisive)), random_state=0)

    rows = []
    failures = []
    for _, fx in sample.iterrows():
        fid = int(fx["fixture_id"])
        rep = replay_results[replay_results["fixture_id"] == fid].sort_values("minute")
        if rep.empty:
            continue

        winner_col = "p_home_win" if fx["home_score"] > fx["away_score"] else "p_away_win"
        first_p = float(rep.iloc[0][winner_col])
        last_p  = float(rep.iloc[-1][winner_col])

        rose = last_p > first_p   # winner's prob should rise
        rows.append({
            "fixture_id": fid,
            "winner_col": winner_col,
            "first_p":    round(first_p, 3),
            "last_p":     round(last_p, 3),
            "delta":      round(last_p - first_p, 3),
            "rose":       rose,
        })
        if not rose:
            failures.append(fid)

    df = pd.DataFrame(rows)
    return {
        "name":     "probability_evolution",
        "passed":   bool(df["rose"].all()) if not df.empty else False,
        "table":    df,
        "n_sample": len(df),
        "n_failures": len(failures),
        "detail":   ("Winner's win-prob rose in all sampled matches."
                     if (not df.empty) and df["rose"].all()
                     else f"Failed in {len(failures)} matches"),
    }


# ── Showcase visual ────────────────────────────────────────────────────────────

def plot_match_probability_trajectory(
    replay_results: pd.DataFrame,
    fixture_id: int,
    fixtures: pd.DataFrame,
    events: pd.DataFrame,
    save_path: Path | None = None,
):
    """
    Plot p_home_win, p_draw, p_away_win, p_over_25, p_btts over the 90 minutes.
    Mark goal/red-card events with vertical lines.
    """
    import matplotlib.pyplot as plt

    rep = replay_results[replay_results["fixture_id"] == fixture_id].sort_values("minute")
    if rep.empty:
        print(f"  no replay rows for fixture {fixture_id}")
        return None

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(rep["minute"], rep["p_home_win"], label="P(home win)", linewidth=2, color="tab:blue")
    ax.plot(rep["minute"], rep["p_draw"],     label="P(draw)",     linewidth=2, color="tab:gray")
    ax.plot(rep["minute"], rep["p_away_win"], label="P(away win)", linewidth=2, color="tab:red")
    if "p_over_25" in rep.columns:
        ax.plot(rep["minute"], rep["p_over_25"], label="P(over 2.5)", linewidth=1.5,
                color="tab:green", linestyle="--")
    if "p_btts" in rep.columns:
        ax.plot(rep["minute"], rep["p_btts"], label="P(BTTS)", linewidth=1.5,
                color="tab:orange", linestyle="--")

    # Event markers
    fix_row = fixtures[fixtures["fixture_id"] == fixture_id].iloc[0]
    home_id = str(fix_row["home_team_id"])
    away_id = str(fix_row["away_team_id"])

    ev = events[events["fixture_id"] == fixture_id].copy()
    ev["team_id"] = ev["team_id"].astype(str)
    ev["minute"]  = pd.to_numeric(ev["minute"], errors="coerce")
    ev["type"]    = ev["type"].fillna("").str.lower()
    ev["detail"]  = ev["detail"].fillna("").str.lower()

    for _, e in ev.iterrows():
        m = float(e["minute"]) if pd.notna(e["minute"]) else None
        if m is None: continue
        if e["type"] == "goal" and "cancelled" not in e["detail"]:
            color = "blue" if e["team_id"] == home_id else "red"
            ax.axvline(m, color=color, alpha=0.35, linewidth=1)
        elif e["type"] == "card" and e["detail"] in {"red card", "second yellow card"}:
            color = "darkblue" if e["team_id"] == home_id else "darkred"
            ax.axvline(m, color=color, alpha=0.6, linewidth=1.5, linestyle=":")

    home_name = fix_row.get("home_team_name", "Home")
    away_name = fix_row.get("away_team_name", "Away")
    score = f"{int(fix_row['home_score'])}-{int(fix_row['away_score'])}"
    ax.set_title(f"Probability trajectory — {home_name} vs {away_name} ({score})")
    ax.set_xlabel("Minute"); ax.set_ylabel("Probability")
    ax.set_xlim(0, 90); ax.set_ylim(0, 1)
    ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120)
        print(f"  saved → {save_path}")
    plt.close(fig)
    return fig


# ── Orchestrator ───────────────────────────────────────────────────────────────

def run_all_checks(
    replay_results: pd.DataFrame,
    gbm_predictions: pd.DataFrame,
    fixtures: pd.DataFrame,
    events: pd.DataFrame,
    showcase_fixture_id: int | None = None,
    diagnostics_dir: Path = DIAGNOSTICS_DIR,
) -> dict:
    print("=" * 60)
    print("  STAGE 6 VALIDATION — replay_engine")
    print("=" * 60)

    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    checks = [
        check_no_negative_no_over_one(replay_results),
        check_replay_anchoring(replay_results, gbm_predictions),
        check_probability_evolution(replay_results, fixtures),
    ]

    for c in checks:
        status = "PASS" if c["passed"] else "FAIL"
        print(f"  [{status}] {c['name']:<28} {c['detail']}")
        if "table" in c:
            print(c["table"].to_string(index=False))

    # Showcase plot — pick a decisive match if not provided
    if showcase_fixture_id is None:
        fix = fixtures.copy()
        fix["margin"] = (
            pd.to_numeric(fix["home_score"], errors="coerce") -
            pd.to_numeric(fix["away_score"], errors="coerce")
        ).abs()
        candidates = fix[fix["margin"] >= 2]["fixture_id"].tolist()
        common = [f for f in candidates if f in replay_results["fixture_id"].values]
        if common:
            showcase_fixture_id = int(common[len(common) // 2])

    if showcase_fixture_id is not None:
        plot_match_probability_trajectory(
            replay_results, showcase_fixture_id, fixtures, events,
            save_path=diagnostics_dir / f"trajectory_fixture_{showcase_fixture_id}.png",
        )

    summary = {c["name"]: c for c in checks}
    summary["all_passed"] = all(c["passed"] for c in checks)
    print(f"\n  ALL PASSED: {summary['all_passed']}")
    print("=" * 60)
    return summary
