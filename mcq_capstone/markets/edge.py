"""
Edge calculation: compare fair probability to market-implied probability.

Functions are stateless helpers — apply them to any (prob, odds) pair.
"""

from __future__ import annotations
import numpy as np
from typing import Optional


def expected_value(model_prob: float, book_odds: float) -> float:
    """
    Expected value (edge) for a single bet.

    EV = P_model × decimal_odds − 1

    Interpretation:
        +0.10 → model expects 10% profit per unit staked
         0.00 → break-even
        -0.05 → model expects 5% loss
    """
    return model_prob * book_odds - 1.0


def implied_probability(book_odds: float) -> float:
    """Book's implied probability (includes margin): 1 / decimal_odds."""
    return 1.0 / book_odds


def overround(odds_home: float, odds_draw: float, odds_away: float) -> float:
    """
    Bookmaker overround (margin).

    overround = Σ implied_probs − 1
    A 5% overround means the book earns 5 units per 100 staked.
    """
    return implied_probability(odds_home) + implied_probability(odds_draw) + implied_probability(odds_away) - 1.0


def remove_margin_multiplicative(odds_home: float, odds_draw: float, odds_away: float) -> tuple[float, float, float]:
    """
    Remove the bookmaker margin multiplicatively to get fair probabilities.

    Distributes the overround proportionally across all outcomes.
    """
    raw = [implied_probability(o) for o in [odds_home, odds_draw, odds_away]]
    total = sum(raw)
    fair_probs = [p / total for p in raw]
    return tuple(fair_probs)  # (p_home, p_draw, p_away)


def edge_vs_market(model_prob: float, book_odds: float) -> float:
    """
    Edge = model probability − market implied probability.

    Positive means model is more bullish than market.
    Equivalent to EV / book_odds.
    """
    return model_prob - implied_probability(book_odds)


def kelly_fraction(model_prob: float, book_odds: float,
                   fractional: float = 0.25,
                   max_f: float = 0.04) -> float:
    """
    Fractional Kelly criterion.

    Full Kelly: f* = (b·p − q) / b
      where b = decimal_odds − 1, p = win prob, q = 1 − p

    Returns fraction of bankroll to stake (0 if -EV).
    """
    b = book_odds - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - model_prob
    f_full = (b * model_prob - q) / b
    return float(np.clip(f_full * fractional, 0.0, max_f))
