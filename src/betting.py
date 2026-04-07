"""
Betting logic: expected value, Kelly criterion, bet selection.

Key insight from initial backtest:
  - Draws at EV=4% had 17% win rate — barely better than random
  - H/W outcomes at EV=4% had ~45% win rate — still not sharp enough
  - The fix: higher EV threshold + per-market confidence floor + overround filter

Overround (book margin) check:
    overround = 1/odds_home + 1/odds_draw + 1/odds_away - 1
    A 5% overround means the book earns 5% on average.
    When overround > 12%, there's very little room for our edge to survive.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional


# ── Core math ──────────────────────────────────────────────────────────────────

def expected_value(prob: float, odds: float) -> float:
    """EV = P_model × decimal_odds − 1.  Positive = edge over the book."""
    return prob * odds - 1.0


def overround(odds_home: float, odds_draw: float, odds_away: float) -> float:
    """Bookmaker margin (overround).  0.05 = 5%."""
    return 1 / odds_home + 1 / odds_draw + 1 / odds_away - 1.0


def kelly_stake(prob: float, odds: float,
                fraction: float = 0.25,
                max_fraction: float = 0.04) -> float:
    """
    Fractional Kelly criterion.
    Returns fraction of bankroll to stake. Returns 0 if bet is -EV.
    """
    b = odds - 1.0
    q = 1.0 - prob
    full_kelly = (b * prob - q) / b
    return float(np.clip(full_kelly * fraction, 0.0, max_fraction))


# ── Bet scanner ────────────────────────────────────────────────────────────────

def scan_match(
    home_team: str,
    away_team: str,
    probs: dict,
    odds_home: Optional[float],
    odds_draw: Optional[float],
    odds_away: Optional[float],
    # thresholds
    ev_threshold: float = 0.08,
    draw_ev_threshold: float = 0.15,
    min_model_prob: float = 0.40,
    max_book_overround: float = 0.12,
    kelly_fraction: float = 0.25,
    max_bet_fraction: float = 0.04,
    min_odds: float = 1.50,
    max_odds: float = 5.00,
) -> list[dict]:
    """
    Evaluate all three outcomes and return qualifying bets.

    Filtering pipeline (applied in order):
      1. Odds must be within [min_odds, max_odds]
      2. All three odds must be available for overround check
      3. Book overround must be ≤ max_book_overround
      4. Model probability must be ≥ min_model_prob
      5. EV must exceed threshold (draw uses draw_ev_threshold)
      6. Kelly fraction > 0

    Returns list of bet dicts (empty if no value found).
    """
    # Need all three odds to compute overround
    if any(x is None or (isinstance(x, float) and np.isnan(x))
           for x in [odds_home, odds_draw, odds_away]):
        return []

    # Overround filter — applied once for the whole market
    or_ = overround(odds_home, odds_draw, odds_away)
    if or_ > max_book_overround:
        return []

    candidates = [
        ("home", probs.get("home"), odds_home),
        ("draw", probs.get("draw"), odds_draw),
        ("away", probs.get("away"), odds_away),
    ]
    bets = []
    for market, prob, odds in candidates:
        if prob is None or odds is None:
            continue
        if np.isnan(odds) or odds < min_odds or odds > max_odds:
            continue

        # Per-market EV threshold — draws need much more edge
        threshold = draw_ev_threshold if market == "draw" else ev_threshold

        # Minimum model confidence
        if prob < min_model_prob:
            continue

        ev = expected_value(prob, odds)
        if ev < threshold:
            continue

        kelly = kelly_stake(prob, odds, fraction=kelly_fraction,
                            max_fraction=max_bet_fraction)
        if kelly <= 0:
            continue

        bets.append({
            "home_team": home_team,
            "away_team": away_team,
            "market": market,
            "prob": round(prob, 4),
            "fair_odds": round(1 / prob, 3),
            "book_odds": odds,
            "ev": round(ev, 4),
            "overround": round(or_, 4),
            "kelly": round(kelly, 5),
        })
    return bets


def evaluate_bet(bet: dict, actual_home: int, actual_away: int) -> float:
    """Returns (odds − 1) if won, −1 if lost."""
    market = bet["market"]
    if market == "home":
        won = actual_home > actual_away
    elif market == "draw":
        won = actual_home == actual_away
    else:
        won = actual_away > actual_home
    return (bet["book_odds"] - 1.0) if won else -1.0


def bets_to_dataframe(bets: list[dict]) -> pd.DataFrame:
    if not bets:
        return pd.DataFrame(columns=["home_team", "away_team", "market",
                                     "prob", "fair_odds", "book_odds",
                                     "ev", "overround", "kelly"])
    return (pd.DataFrame(bets)
              .sort_values("ev", ascending=False)
              .reset_index(drop=True))
