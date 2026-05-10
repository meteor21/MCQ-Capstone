"""
Stage 7 Validation — bet_logic.py
===================================
Validation checks for the bet sizing / decision layer.

Run before live deployment:
    from stage7_validate import run_all_checks
    summary = run_all_checks()
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from bet_logic import (
    single_bet_edge,
    kelly_fraction,
    filter_by_correlation,
    recommend_bets,
    format_recommendations_text,
    BetCandidate,
)


DIAGNOSTICS_DIR = Path(
    "/content/drive/MyDrive/soccer-betting/data/processed/api_football/diagnostics/stage7"
)


# ── Edge calculation ──────────────────────────────────────────────────────────

def check_edge_calculation(atol: float = 1e-6) -> dict:
    """
    Hard-coded edge cases:
      p=0.5,  odds=2.10 → +0.05
      p=0.5,  odds=1.90 → -0.05
      p=0.55, odds=2.00 → +0.10
      p=0.0,  odds=anything → -1.0
    """
    cases = [
        (0.50, 2.10,  0.05),
        (0.50, 1.90, -0.05),
        (0.55, 2.00,  0.10),
        (0.00, 3.00, -1.00),
        (1.00, 1.50,  0.50),
    ]
    failures = []
    for p, odds, expected in cases:
        got = single_bet_edge(p, odds)
        if abs(got - expected) > atol:
            failures.append({"p": p, "odds": odds, "expected": expected, "got": got})

    return {
        "name":     "edge_calculation",
        "passed":   len(failures) == 0,
        "n_cases":  len(cases),
        "failures": failures,
        "detail":   "All edge calculations correct." if not failures
                    else f"{len(failures)} mismatches: {failures}",
    }


# ── Kelly logic ───────────────────────────────────────────────────────────────

def check_kelly_logic() -> dict:
    """
    Verify:
      - Negative or zero edge → Kelly = 0
      - Positive edge → Kelly > 0
      - Kelly never exceeds 10% of bankroll (configured cap)
      - Quarter Kelly (0.25) is half of full Kelly
    """
    failures = []

    # 1. Zero/negative edge → 0
    if kelly_fraction(0.40, 2.00, fractional=0.25) > 0:  # edge = -0.20
        failures.append("Negative edge produced positive Kelly")

    # 2. Positive edge → positive Kelly
    k_pos = kelly_fraction(0.55, 2.00, fractional=0.25)
    if k_pos <= 0:
        failures.append("Positive edge produced zero/negative Kelly")

    # 3. Cap at 10%
    k_huge = kelly_fraction(0.99, 5.00, fractional=1.00)
    if k_huge > 0.10:
        failures.append(f"Kelly exceeded 10% cap: {k_huge}")

    # 4. Fractional scaling
    k_quarter = kelly_fraction(0.55, 2.00, fractional=0.25)
    k_half    = kelly_fraction(0.55, 2.00, fractional=0.50)
    if not (k_half >= 1.5 * k_quarter or k_quarter == 0):
        # half Kelly should be ~2× quarter (modulo cap)
        if k_quarter > 0 and k_half / k_quarter < 1.5:
            failures.append(f"Fractional scaling off: quarter={k_quarter}, half={k_half}")

    return {
        "name":     "kelly_logic",
        "passed":   len(failures) == 0,
        "failures": failures,
        "detail":   "All Kelly logic checks pass." if not failures else "; ".join(failures),
    }


# ── Correlation filter ────────────────────────────────────────────────────────

def check_correlation_filter() -> dict:
    """
    Construct synthetic candidates with known cluster IDs.
    Verify only the highest-edge candidate per cluster survives.
    """
    candidates = [
        BetCandidate("home_win", 0.55, 2.10, edge=0.155, kelly=0.05, cluster_id=0),
        BetCandidate("dc_1X",    0.78, 1.30, edge=0.014, kelly=0.01, cluster_id=0),
        BetCandidate("over_25",  0.62, 1.85, edge=0.147, kelly=0.04, cluster_id=1),
        BetCandidate("btts",     0.60, 1.90, edge=0.140, kelly=0.04, cluster_id=1),
        BetCandidate("away_win", 0.30, 4.00, edge=0.200, kelly=0.06, cluster_id=2),
    ]

    filtered = filter_by_correlation(candidates)
    cluster_ids = sorted(set(c.cluster_id for c in filtered))
    expected_cluster_count = 3
    survivors_per_cluster = {cid: [c.market for c in filtered if c.cluster_id == cid]
                             for cid in cluster_ids}

    # Within cluster 0, home_win (edge 0.155) > dc_1X (0.014)
    # Within cluster 1, over_25 (0.147) > btts (0.140)
    expected_winners = {0: "home_win", 1: "over_25", 2: "away_win"}
    actual_winners   = {c.cluster_id: c.market for c in filtered}

    passed = (
        len(filtered) == expected_cluster_count
        and actual_winners == expected_winners
    )

    return {
        "name":          "correlation_filter",
        "passed":        passed,
        "n_input":       len(candidates),
        "n_output":      len(filtered),
        "winners":       actual_winners,
        "expected":      expected_winners,
        "survivors":     survivors_per_cluster,
        "detail":        ("One bet per cluster, highest-edge survives."
                          if passed
                          else f"Got {actual_winners}, expected {expected_winners}"),
    }


# ── Visual: example output ─────────────────────────────────────────────────────

def make_example_recommendations_table(save_path: Path | None = None) -> pd.DataFrame:
    """
    Build a synthetic DataFrame showing what a recommendation list looks like.
    """
    p_vector = {
        "home_win":  0.52, "draw":     0.27, "away_win": 0.21,
        "btts":      0.58, "no_btts":  0.42,
        "over_05":   0.93, "over_15":  0.78, "over_25":  0.55, "over_35":  0.30,
        "home_clean_sheet": 0.32, "away_clean_sheet": 0.21,
        "next_goal_home":   0.55, "next_goal_away":   0.35, "no_more_goals": 0.10,
        "dc_1X": 0.79, "dc_X2": 0.48, "dc_12": 0.73,
        "home_minus_15": 0.27, "away_minus_15": 0.10,
    }
    cluster_map = {
        "home_win": 0, "dc_1X": 0, "home_minus_15": 0,
        "draw":     1, "no_btts": 1,
        "away_win": 2, "dc_X2": 2, "away_minus_15": 2,
        "btts":     3, "over_25": 3, "over_15": 3,
        "next_goal_home": 4, "next_goal_away": 4, "no_more_goals": 4,
    }
    market_odds = {
        "home_win": 2.10, "draw":   3.40, "away_win": 4.50,
        "btts":     1.85, "over_25": 1.95, "over_15":  1.30,
        "next_goal_home": 1.95,
    }

    recs = recommend_bets(
        p_vector=p_vector, cluster_map=cluster_map,
        market_odds_dict=market_odds, edge_threshold=0.03,
    )

    text = format_recommendations_text(recs, fixture_label="Mock Match", minute=0)
    print(text)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            f.write(text + "\n\n")
            f.write(recs.to_string(index=False))
        print(f"  saved → {save_path}")

    return recs


# ── Orchestrator ───────────────────────────────────────────────────────────────

def run_all_checks(diagnostics_dir: Path = DIAGNOSTICS_DIR) -> dict:
    print("=" * 60)
    print("  STAGE 7 VALIDATION — bet_logic")
    print("=" * 60)

    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    checks = [
        check_edge_calculation(),
        check_kelly_logic(),
        check_correlation_filter(),
    ]

    for c in checks:
        status = "PASS" if c["passed"] else "FAIL"
        print(f"  [{status}] {c['name']:<28} {c['detail']}")

    print("\n  Example recommendation output:")
    make_example_recommendations_table(
        save_path=diagnostics_dir / "example_recommendations.txt",
    )

    summary = {c["name"]: c for c in checks}
    summary["all_passed"] = all(c["passed"] for c in checks)
    print(f"\n  ALL PASSED: {summary['all_passed']}")
    print("=" * 60)
    return summary
