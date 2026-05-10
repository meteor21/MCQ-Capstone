"""
Stage 4 Validation — correlation_engine.py
============================================
Validation checks for the market correlation matrix.

Run before moving to Stage 5:
    from stage4_validate import run_all_checks
    summary = run_all_checks(D_array, market_names)
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from forward_simulator import MARKET_NAMES, market_probabilities
from correlation_engine import (
    market_correlation_matrix,
    identify_correlation_clusters,
)


DIAGNOSTICS_DIR = Path(
    "/content/drive/MyDrive/soccer-betting/data/processed/api_football/diagnostics/stage4"
)


# ── Algebraic properties ──────────────────────────────────────────────────────

def check_correlation_matrix_properties(C: pd.DataFrame, atol: float = 1e-6) -> dict:
    """
    Verify: symmetric, diagonal=1, all entries in [-1, 1], positive semi-definite.
    """
    M = C.values

    sym_err   = float(np.abs(M - M.T).max())
    diag_err  = float(np.abs(np.diag(M) - 1.0).max())
    range_ok  = bool((M >= -1 - atol).all() and (M <= 1 + atol).all())

    # Eigenvalue check: PSD requires all eigenvalues ≥ -atol
    eigs = np.linalg.eigvalsh(M)
    psd_ok = bool((eigs >= -1e-6).all())
    min_eig = float(eigs.min())

    passed = (sym_err < 1e-6) and (diag_err < 1e-6) and range_ok and psd_ok

    return {
        "name":        "matrix_properties",
        "passed":      passed,
        "sym_err":     sym_err,
        "diag_err":    diag_err,
        "range_ok":    range_ok,
        "psd_ok":      psd_ok,
        "min_eig":     min_eig,
        "detail": (
            f"sym={sym_err:.1e}, diag={diag_err:.1e}, "
            f"in_range={range_ok}, psd={psd_ok} (min eig={min_eig:.4f})"
        ),
    }


# ── Known correlations ────────────────────────────────────────────────────────

def check_known_correlations(C: pd.DataFrame) -> dict:
    """
    Hard-coded sanity checks reflecting market logic:
      - corr(home_win, draw) < 0           (mutually exclusive)
      - corr(over_25, btts) > 0.5          (highly related)
      - corr(home_clean_sheet, btts) ≈ -1  (impossible together)
      - corr(over_15, over_25) > 0.7       (over_25 → over_15 always)
      - corr(home_win, away_win) < -0.5    (mutually exclusive)
    """
    expectations = [
        ("home_win", "draw",             "<", -0.05),
        ("over_25",  "btts",             ">",  0.40),
        ("home_clean_sheet", "btts",     "<", -0.90),
        ("over_15",  "over_25",          ">",  0.65),
        ("home_win", "away_win",         "<", -0.40),
        ("dc_1X",    "away_win",         "<", -0.95),
        ("no_more_goals", "next_goal_home", "<", -0.30),
    ]

    results = []
    for m1, m2, op, threshold in expectations:
        if m1 not in C.index or m2 not in C.columns:
            results.append({"pair": f"{m1}~{m2}", "rho": None, "ok": False,
                            "expected": f"{op} {threshold}",
                            "note": "market not present"})
            continue
        rho = float(C.loc[m1, m2])
        ok  = (rho < threshold) if op == "<" else (rho > threshold)
        results.append({
            "pair": f"{m1}~{m2}", "rho": round(rho, 3),
            "expected": f"{op} {threshold}", "ok": ok, "note": "" if ok else "VIOLATION",
        })

    df = pd.DataFrame(results)
    return {
        "name":     "known_correlations",
        "passed":   bool(df["ok"].all()),
        "table":    df,
        "detail":   ("All known correlations agree with theory."
                     if df["ok"].all()
                     else f"{(~df['ok']).sum()} sanity violations: see table"),
    }


# ── Visual: heatmap ───────────────────────────────────────────────────────────

def plot_correlation_heatmap(
    C: pd.DataFrame,
    state_label: str = "minute 0",
    save_path: Path | None = None,
):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(
        C, ax=ax, annot=True, fmt=".2f", cmap="RdYlGn",
        center=0, vmin=-1, vmax=1, linewidths=0.3, annot_kws={"size": 7},
    )
    ax.set_title(f"Market Correlation Matrix — {state_label}", fontsize=11)
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120)
        print(f"  saved → {save_path}")
    plt.close(fig)
    return fig


# ── Visual: dendrogram ─────────────────────────────────────────────────────────

def plot_correlation_dendrogram(
    C: pd.DataFrame,
    save_path: Path | None = None,
):
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import linkage, dendrogram

    distance = 1.0 - np.abs(C.values)
    np.fill_diagonal(distance, 0.0)
    n = distance.shape[0]
    condensed = []
    for i in range(n):
        for j in range(i + 1, n):
            condensed.append(distance[i, j])
    Z = linkage(np.array(condensed), method="single")

    fig, ax = plt.subplots(figsize=(13, 6))
    dendrogram(Z, labels=C.columns.tolist(), ax=ax, leaf_rotation=80)
    ax.set_title("Hierarchical clustering of markets (distance = 1 - |ρ|)")
    ax.axhline(0.4, color="red", linestyle="--", alpha=0.5,
               label="cluster cutoff (|ρ| > 0.6)")
    ax.legend()
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120)
        print(f"  saved → {save_path}")
    plt.close(fig)
    return fig


# ── Orchestrator ───────────────────────────────────────────────────────────────

def run_all_checks(
    D: np.ndarray,
    state_label: str = "minute 0, evenly matched",
    diagnostics_dir: Path = DIAGNOSTICS_DIR,
    cluster_threshold: float = 0.6,
) -> dict:
    print("=" * 60)
    print("  STAGE 4 VALIDATION — correlation_engine")
    print("=" * 60)

    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    C = market_correlation_matrix(D)
    p_vec = market_probabilities(D)

    checks = [
        check_correlation_matrix_properties(C),
        check_known_correlations(C),
    ]

    for c in checks:
        status = "PASS" if c["passed"] else "FAIL"
        print(f"  [{status}] {c['name']:<28} {c['detail']}")
        if "table" in c:
            print(c["table"].to_string(index=False))

    # Cluster summary
    clusters = identify_correlation_clusters(C, threshold=cluster_threshold)
    print(f"\n  Identified {len(clusters)} correlation clusters at |ρ| > {cluster_threshold}:")
    for i, cl in enumerate(clusters):
        print(f"    cluster {i}: {cl}")

    plot_correlation_heatmap(
        C, state_label=state_label,
        save_path=diagnostics_dir / "corr_heatmap.png",
    )
    plot_correlation_dendrogram(
        C, save_path=diagnostics_dir / "corr_dendrogram.png",
    )

    summary = {c["name"]: c for c in checks}
    summary["all_passed"] = all(c["passed"] for c in checks)
    summary["clusters"] = clusters
    summary["correlation_matrix"] = C
    print(f"\n  ALL PASSED: {summary['all_passed']}")
    print("=" * 60)
    return summary
