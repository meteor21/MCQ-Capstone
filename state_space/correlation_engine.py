"""
Stage 4 — Correlation Engine
==============================
Compute and analyse market correlation from the simulation matrix D.

This is the credit-risk-pricing analogue: the N × M binary matrix D is
exactly the "loss indicator matrix" in a CDO pricing model. The Pearson
correlation ρ_ij between column i and column j quantifies how co-dependent
markets i and j are.

Key interpretation
------------------
  ρ =  1 : markets move perfectly together → same bet, never stack
  ρ =  0 : markets independent → free to combine
  ρ = -1 : markets perfectly anti-correlated → cannot both win

Because all columns come from the same simulated paths, ρ is path-consistent:
it reflects the actual joint probability structure, not a heuristic.

Downstream use (Stage 7)
------------------------
  - Bet stacking: only combine bets from different correlation clusters.
  - Kelly sizing: accounts for within-cluster correlation to avoid over-betting
    on what is effectively the same outcome.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

from forward_simulator import MARKET_NAMES, M


# ── Correlation matrix ─────────────────────────────────────────────────────────

def market_correlation_matrix(
    D: np.ndarray,
    market_names: list[str] = MARKET_NAMES,
) -> pd.DataFrame:
    """
    Compute Pearson correlation between all pairs of market indicators.

    Parameters
    ----------
    D            : (N, M) binary indicator array from simulate_match().
    market_names : Column labels (default: MARKET_NAMES from forward_simulator).

    Returns
    -------
    DataFrame of shape (M, M) — symmetric, diagonal = 1.0.

    Notes
    -----
    - Bernoulli columns with zero variance (p = 0 or p = 1) produce NaN in the
      correlation matrix. These are replaced with 0.0 (independent).
    - TODO: for small N, consider shrinkage (Ledoit-Wolf) to stabilise estimates.
    """
    corr = np.corrcoef(D.T)               # (M, M) numpy array
    corr = np.nan_to_num(corr, nan=0.0)   # handle zero-variance columns
    np.fill_diagonal(corr, 1.0)           # enforce exact diagonal
    return pd.DataFrame(corr, index=market_names, columns=market_names)


# ── Cluster analysis ───────────────────────────────────────────────────────────

def identify_correlation_clusters(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.6,
) -> list[list[str]]:
    """
    Group markets where pairwise |ρ| > threshold using hierarchical clustering.

    Parameters
    ----------
    corr_matrix : Output of market_correlation_matrix().
    threshold   : Correlation magnitude cutoff (default 0.6).

    Returns
    -------
    List of clusters, each a list of market name strings.
    Example: [['over_25', 'btts', 'home_win'], ['draw', 'no_btts'], ['away_win']]

    Algorithm
    ---------
    Converts correlation to a distance metric d = 1 - |ρ|, then applies
    agglomerative (single-linkage) clustering. Single linkage ensures that two
    markets end up in the same cluster if ANY pair between them is correlated
    above the threshold.

    TODO: experiment with average-linkage — single linkage can create long chains
    where distantly-related markets end up in the same cluster.
    """
    corr_vals = corr_matrix.values
    distance = 1.0 - np.abs(corr_vals)
    np.fill_diagonal(distance, 0.0)

    # Condense the square distance matrix to upper-triangle form for scipy
    n = distance.shape[0]
    condensed = []
    for i in range(n):
        for j in range(i + 1, n):
            condensed.append(distance[i, j])
    condensed = np.array(condensed)

    Z = linkage(condensed, method="single")
    labels = fcluster(Z, t=1.0 - threshold, criterion="distance")

    clusters: dict[int, list[str]] = {}
    for market_name, cluster_id in zip(corr_matrix.columns, labels):
        clusters.setdefault(int(cluster_id), []).append(market_name)

    return list(clusters.values())


# ── Redundancy report ──────────────────────────────────────────────────────────

def redundancy_report(
    corr_matrix: pd.DataFrame,
    p_vector: dict[str, float],
    high_threshold: float = 0.70,
    anti_threshold: float = -0.50,
) -> pd.DataFrame:
    """
    For each market, identify which other markets are redundant or anti-correlated.

    Parameters
    ----------
    corr_matrix     : (M, M) correlation DataFrame.
    p_vector        : Dict market_name → probability (from market_probabilities()).
    high_threshold  : ρ above this → redundant (same information).
    anti_threshold  : ρ below this → anti-correlated (mutually exclusive-ish).

    Returns
    -------
    DataFrame with columns:
        market, probability, redundant_with (comma-sep), anti_correlated_with (comma-sep)
    """
    rows = []
    markets = corr_matrix.columns.tolist()

    for m in markets:
        row_corr = corr_matrix.loc[m]
        redundant = [
            other for other in markets
            if other != m and row_corr[other] >= high_threshold
        ]
        anti = [
            other for other in markets
            if other != m and row_corr[other] <= anti_threshold
        ]
        rows.append({
            "market":              m,
            "probability":         round(p_vector.get(m, float("nan")), 4),
            "redundant_with":      ", ".join(redundant) if redundant else "—",
            "anti_correlated_with": ", ".join(anti)    if anti     else "—",
        })

    return pd.DataFrame(rows)


# ── Visualisation ──────────────────────────────────────────────────────────────

def visualize_state_correlation(
    state: dict,
    D: np.ndarray,
    save_path: Path,
    market_names: list[str] = MARKET_NAMES,
    title_extra: str = "",
) -> None:
    """
    Save a seaborn heatmap of the correlation matrix to a PNG file.

    Parameters
    ----------
    state        : State dict — used only for the plot title.
    D            : (N, M) simulation matrix.
    save_path    : Output path for the PNG.
    market_names : Labels (default MARKET_NAMES).
    title_extra  : Additional string appended to the title.
    """
    # Lazy imports to avoid loading matplotlib at module level
    import matplotlib.pyplot as plt
    import seaborn as sns

    corr = market_correlation_matrix(D, market_names)
    minute = state.get("minute", "?")
    score_h = state.get("score_h", "?")
    score_a = state.get("score_a", "?")
    title = (
        f"Market Correlation at Minute {minute} "
        f"(Score {score_h}–{score_a})"
        + (f" — {title_extra}" if title_extra else "")
    )

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.3,
        annot_kws={"size": 7},
    )
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Correlation heatmap saved → {save_path}")


# ── Convenience wrapper ────────────────────────────────────────────────────────

def full_correlation_analysis(
    D: np.ndarray,
    p_vector: dict[str, float],
    cluster_threshold: float = 0.6,
) -> dict:
    """
    Run the full correlation analysis pipeline and return everything.

    Returns
    -------
    Dict with keys:
        'corr_matrix'   : pd.DataFrame (M × M)
        'clusters'      : list[list[str]]
        'cluster_map'   : dict[market → cluster_id]
        'redundancy_df' : pd.DataFrame
    """
    corr = market_correlation_matrix(D)
    clusters = identify_correlation_clusters(corr, threshold=cluster_threshold)
    cluster_map = {
        market: cid
        for cid, cluster in enumerate(clusters)
        for market in cluster
    }
    redundancy = redundancy_report(corr, p_vector)

    return {
        "corr_matrix":   corr,
        "clusters":      clusters,
        "cluster_map":   cluster_map,
        "redundancy_df": redundancy,
    }
