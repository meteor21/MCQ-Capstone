"""
Stage 7 — Bet Logic (Sizing & Decision Layer)
===============================================
This file is ENTIRELY separated from the simulation model. It takes
(p̂, correlation_matrix, odds) as inputs and produces bet recommendations.
It never calls simulate_match(), knows nothing about states, and has no
access to events or statistics.

Separation principle
--------------------
The model's job: produce p̂ and C.
This file's job: translate (p̂, C, odds) → sized bet recommendations.

Sizing framework
----------------
  edge_j = p̂_j × decimal_odds_j - 1

Kelly fraction (without correlation adjustment):
  f_j = edge_j / (decimal_odds_j - 1)   (fractional applied after)

Correlation-adjusted Kelly (multi-bet):
  TODO: full mean-variance portfolio Kelly requires solving a quadratic program.
  For now, we use a greedy cluster filter: pick at most one bet per
  correlation cluster, then apply fractional Kelly independently.
  This is conservative and correct in the limit of independent clusters.

Edge threshold
--------------
  edge_threshold = 0.03 (3%) — only bet when model has meaningful edge.
  This is a tunable parameter, not a magic number. TODO: calibrate on val set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class BetCandidate:
    market:       str
    p_model:      float   # model probability
    decimal_odds: float   # bookmaker decimal odds
    edge:         float   # = p_model * decimal_odds - 1
    kelly:        float   # fractional Kelly stake (as fraction of bankroll)
    cluster_id:   int     # from correlation_engine


@dataclass
class BetRecommendation:
    candidates:   list[BetCandidate]
    total_kelly:  float  # sum of all stake fractions
    n_markets:    int    # number of bets recommended


# ── Single-market calculations ────────────────────────────────────────────────

def single_bet_edge(p_model: float, decimal_odds: float) -> float:
    """
    Compute expected edge for a single bet.

    Parameters
    ----------
    p_model      : Model probability in [0, 1].
    decimal_odds : Bookmaker decimal odds (e.g. 2.50 for evens+).

    Returns
    -------
    edge = p_model × decimal_odds - 1.
    Positive → model thinks bet is +EV.
    Negative → bookmaker has the edge.

    Notes
    -----
    This assumes the market price is fair (no bid-ask spread beyond the
    overround). In practice, use the best available odds and subtract
    a commission estimate.
    """
    return p_model * decimal_odds - 1.0


def kelly_fraction(
    p_model: float,
    decimal_odds: float,
    fractional: float = 0.25,
) -> float:
    """
    Compute fractional Kelly stake as a fraction of bankroll.

    Full Kelly: f* = (p × (b + 1) - 1) / b   where b = decimal_odds - 1
    Fractional Kelly: f = fractional × f*

    Parameters
    ----------
    p_model      : Model probability.
    decimal_odds : Decimal odds.
    fractional   : Kelly fraction (default 0.25 = quarter Kelly).
                   Quarter Kelly is standard for high-variance markets.

    Returns
    -------
    Stake as fraction of current bankroll. Clamped to [0, 0.10] — never bet
    more than 10% of bankroll on a single outcome regardless of Kelly.

    Notes
    -----
    - Full Kelly is theoretically optimal for a single bet but produces large
      variance swings in practice.
    - 0.25× Kelly (quarter Kelly) is a widely used conservative default.
    - TODO: use the correlation-adjusted Kelly for portfolios — the greedy
      cluster filter is a proxy, not the exact solution.
    """
    b = decimal_odds - 1.0
    if b <= 0 or p_model <= 0:
        return 0.0

    f_star = (p_model * (b + 1.0) - 1.0) / b
    f_star = max(f_star, 0.0)            # never negative Kelly (= don't bet)
    return float(min(fractional * f_star, 0.10))


# ── Correlation-aware filtering ────────────────────────────────────────────────

def filter_by_correlation(
    candidates: list[BetCandidate],
    threshold: float = 0.6,
) -> list[BetCandidate]:
    """
    Within each correlation cluster, keep only the highest-edge candidate.

    Parameters
    ----------
    candidates : List of BetCandidate (already edge-filtered).
    threshold  : Correlation threshold used when building clusters.
                 Passed here as documentation — clustering was already done
                 upstream in correlation_engine.

    Returns
    -------
    Filtered list: at most one bet per cluster, the highest-edge one.

    Rationale
    ---------
    If two markets are correlated at ρ > threshold, they are effectively
    betting on the same underlying outcome. Stacking both would:
      1. Over-size exposure to that outcome (ignoring the correlation).
      2. Double-count the edge if the edge source is the same model signal.
    Keeping only the highest-edge bet per cluster avoids both problems.

    TODO: for ρ between 0.3 and 0.6 (moderate correlation), apply a
    proportional stake reduction rather than a hard filter.
    """
    if not candidates:
        return []

    # Group by cluster_id
    clusters: dict[int, list[BetCandidate]] = {}
    for cand in candidates:
        clusters.setdefault(cand.cluster_id, []).append(cand)

    result = []
    for cluster_id, group in clusters.items():
        # Within each cluster, pick the one with maximum edge
        best = max(group, key=lambda c: c.edge)
        result.append(best)

    return sorted(result, key=lambda c: c.edge, reverse=True)


# ── Full recommendation pipeline ───────────────────────────────────────────────

def recommend_bets(
    p_vector: dict[str, float],
    cluster_map: dict[str, int],
    market_odds_dict: dict[str, float],
    edge_threshold: float = 0.03,
    corr_cluster_threshold: float = 0.6,
    kelly_fractional: float = 0.25,
) -> pd.DataFrame:
    """
    End-to-end pipeline: model probs + odds → sized, cluster-filtered recommendations.

    Parameters
    ----------
    p_vector          : Dict market_name → model probability (from forward_simulator).
    cluster_map       : Dict market_name → cluster_id (from correlation_engine).
    market_odds_dict  : Dict market_name → decimal odds. Markets not in this dict
                        are skipped (no odds available).
    edge_threshold    : Minimum edge to be included (default 3%).
    corr_cluster_threshold: For documentation only — clustering done upstream.
    kelly_fractional  : Fraction of full Kelly (default 0.25).

    Returns
    -------
    DataFrame with columns:
        market, p_model, decimal_odds, edge_pct, kelly_stake, cluster_id
    Sorted by edge_pct descending. Empty if no +EV bets found.

    Example
    -------
    p_vec  = {'home_win': 0.55, 'draw': 0.25, 'away_win': 0.20, 'over_25': 0.62, ...}
    odds   = {'home_win': 2.10, 'over_25': 1.85}
    recs   = recommend_bets(p_vec, cluster_map, odds)
    """
    candidates: list[BetCandidate] = []

    for market, decimal_odds in market_odds_dict.items():
        p_model = p_vector.get(market)
        if p_model is None:
            continue

        edge = single_bet_edge(p_model, decimal_odds)
        if edge < edge_threshold:
            continue

        k = kelly_fraction(p_model, decimal_odds, fractional=kelly_fractional)
        if k <= 0:
            continue

        cluster_id = cluster_map.get(market, -1)
        candidates.append(BetCandidate(
            market       = market,
            p_model      = p_model,
            decimal_odds = decimal_odds,
            edge         = edge,
            kelly        = k,
            cluster_id   = cluster_id,
        ))

    # Filter to at most one bet per correlation cluster
    filtered = filter_by_correlation(candidates, threshold=corr_cluster_threshold)

    if not filtered:
        return pd.DataFrame(columns=[
            "market", "p_model", "decimal_odds", "edge_pct", "kelly_stake", "cluster_id"
        ])

    return pd.DataFrame([{
        "market":       c.market,
        "p_model":      round(c.p_model, 4),
        "decimal_odds": round(c.decimal_odds, 3),
        "edge_pct":     round(c.edge * 100, 2),
        "kelly_stake":  round(c.kelly, 4),
        "cluster_id":   c.cluster_id,
    } for c in filtered]).sort_values("edge_pct", ascending=False).reset_index(drop=True)


# ── Formatting ─────────────────────────────────────────────────────────────────

def format_recommendations_text(
    recs: pd.DataFrame,
    fixture_label: str = "",
    minute: int | None = None,
) -> str:
    """
    Pretty-print a recommendations DataFrame for human reading.

    Parameters
    ----------
    recs          : Output of recommend_bets().
    fixture_label : E.g. "Galatasaray vs Liverpool" for the header.
    minute        : If given, shown in the header.

    Returns
    -------
    Multi-line string ready for print() or logging.
    """
    if recs.empty:
        return f"[{fixture_label}] No +EV bets found."

    header_parts = ["BET RECOMMENDATIONS"]
    if fixture_label:
        header_parts.append(fixture_label)
    if minute is not None:
        header_parts.append(f"Minute {minute}")
    header = " | ".join(header_parts)
    sep = "─" * 65

    lines = [sep, header, sep]
    lines.append(
        f"{'Market':<22} {'p_model':>7} {'Odds':>7} {'Edge%':>7} "
        f"{'Kelly':>7} {'Cluster':>7}"
    )
    lines.append("─" * 65)

    for _, row in recs.iterrows():
        lines.append(
            f"{row['market']:<22} {row['p_model']:>7.3f} {row['decimal_odds']:>7.2f} "
            f"{row['edge_pct']:>6.1f}% {row['kelly_stake']:>7.4f} "
            f"{int(row['cluster_id']):>7}"
        )

    total_kelly = recs["kelly_stake"].sum()
    lines.append(sep)
    lines.append(f"Total Kelly exposure: {total_kelly:.4f} ({total_kelly*100:.2f}% of bankroll)")
    lines.append(sep)

    return "\n".join(lines)
