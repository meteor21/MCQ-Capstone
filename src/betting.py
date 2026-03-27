"""
Betting logic: expected value calculation, Kelly criterion, bet selection.

The core workflow:
  1. Model produces (P_home, P_draw, P_away)
  2. For each outcome, calculate EV = P_model * decimal_odds - 1
  3. Only bet when EV > MIN_EV_THRESHOLD  (edge over the book)
  4. Size stake with fractional Kelly criterion
  5. Enforce max-bet-fraction cap as a safety guard
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional


# ── Core math ──────────────────────────────────────────────────────────────────

def expected_value(prob: float, odds: float) -> float:
    """
    Expected value of a single bet.

    EV = probability_of_winning * decimal_odds - 1

    Interpretation:
        EV  > 0  →  +EV bet (expected profit per unit staked)
        EV == 0  →  break-even
        EV  < 0  →  -EV bet (house has edge)

    Parameters
    ----------
    prob  : Model's estimated probability of the outcome occurring.
    odds  : Bookmaker decimal odds (e.g. 2.50 = win 1.50 per 1.00 staked).
    """
    return prob * odds - 1.0


def kelly_stake(prob: float, odds: float,
                fraction: float = 0.25,
                max_fraction: float = 0.05) -> float:
    """
    Fractional Kelly criterion for bet sizing.

    Full Kelly:  f* = (b·p - q) / b
        where b = decimal_odds - 1, p = win prob, q = 1 - p

    We apply `fraction` (typically 0.25 = quarter Kelly) to reduce variance.
    The result is additionally capped at `max_fraction` of bankroll.

    Returns fraction of current bankroll to stake (0 if bet is -EV).
    """
    b = odds - 1.0
    q = 1.0 - prob
    full_kelly = (b * prob - q) / b
    fractional = full_kelly * fraction
    return float(np.clip(fractional, 0.0, max_fraction))


# ── Bet scanner ────────────────────────────────────────────────────────────────

def scan_match(home_team: str,
               away_team: str,
               probs: dict,
               odds_home: Optional[float],
               odds_draw: Optional[float],
               odds_away: Optional[float],
               ev_threshold: float = 0.04,
               kelly_fraction: float = 0.25,
               max_bet_fraction: float = 0.05,
               min_odds: float = 1.30,
               max_odds: float = 6.00) -> list[dict]:
    """
    Evaluate all three outcomes for a match and return qualifying bets.

    Parameters
    ----------
    probs         : dict with keys 'home', 'draw', 'away'
    odds_*        : bookmaker decimal odds (None = not available)
    ev_threshold  : minimum EV required to include a bet
    kelly_fraction: fractional Kelly multiplier
    max_bet_fraction: hard cap on stake as fraction of bankroll
    min_odds / max_odds: filter out lines outside this range

    Returns
    -------
    List of bet dicts (may be empty if no value found).
    Each dict contains:
        home_team, away_team, market, prob, odds, ev, kelly
    """
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
        ev = expected_value(prob, odds)
        if ev < ev_threshold:
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
            "kelly": round(kelly, 5),
        })
    return bets


def evaluate_bet(bet: dict, actual_home: int, actual_away: int) -> float:
    """
    Given an actual scoreline, return the P&L for a bet staked at stake=1.

    Returns  (odds - 1)  if won,  -1  if lost.
    """
    market = bet["market"]
    if market == "home":
        won = actual_home > actual_away
    elif market == "draw":
        won = actual_home == actual_away
    else:  # away
        won = actual_away > actual_home
    return (bet["book_odds"] - 1.0) if won else -1.0


# ── Pretty-printing ────────────────────────────────────────────────────────────

def bets_to_dataframe(bets: list[dict]) -> pd.DataFrame:
    """Convert a list of bet dicts to a tidy DataFrame."""
    if not bets:
        return pd.DataFrame(columns=["home_team", "away_team", "market",
                                     "prob", "fair_odds", "book_odds", "ev", "kelly"])
    df = pd.DataFrame(bets)
    df = df.sort_values("ev", ascending=False).reset_index(drop=True)
    return df
