"""
Market pricer: convert model predictions to fair odds for all markets.

Bridges the pre-match Dixon-Coles model and the in-play simulator,
providing a single interface regardless of match state.
"""

from __future__ import annotations
from typing import Dict, Optional

import numpy as np

from ..pre_match.dixon_coles import DixonColesModel
from ..in_play.state import MatchState
from ..in_play.simulator import MonteCarloSimulator, SimulationResult


class MarketPricer:
    """
    Compute fair probabilities (and fair decimal odds) for a match.

    Supports two modes:
      - Pre-match: uses the analytic Dixon-Coles score matrix (fast, exact).
      - In-play  : uses Monte Carlo simulation from the current MatchState.

    Parameters
    ----------
    model        : Fitted DixonColesModel.
    simulator    : MonteCarloSimulator (for in-play pricing).
    n_simulations: Fallback sim count if no simulator provided.
    """

    def __init__(self,
                 model: DixonColesModel,
                 simulator: Optional[MonteCarloSimulator] = None,
                 n_simulations: int = 50_000):
        self.model = model
        self.simulator = simulator or MonteCarloSimulator(n_simulations)

    # ── Pre-match ──────────────────────────────────────────────────────────────

    def price_pre_match(self,
                        home_team: str,
                        away_team: str) -> Dict[str, float]:
        """
        Return fair probability dict for a pre-match 1X2 market.

        Uses the analytic Dixon-Coles Poisson score matrix.
        Also returns expected goals (lam, mu) for downstream use.
        """
        return self.model.predict_probs(home_team, away_team)

    def price_pre_match_all_markets(self,
                                    home_team: str,
                                    away_team: str) -> SimulationResult:
        """
        Return SimulationResult for a pre-match state (minute=0, 0-0).
        Uses Monte Carlo for all market types (over/under, BTTS, AH, etc.).
        """
        lam, mu = self.model.predict_goals(home_team, away_team)
        state = MatchState.pre_match(home_lambda=lam, away_lambda=mu)
        return self.simulator.simulate(state)

    # ── In-play ────────────────────────────────────────────────────────────────

    def price_in_play(self,
                      home_team: str,
                      away_team: str,
                      minute: int,
                      home_goals: int,
                      away_goals: int) -> SimulationResult:
        """
        Return full SimulationResult from an in-play state.

        Parameters
        ----------
        minute      : Current match minute (1-90).
        home_goals  : Goals scored by home team so far.
        away_goals  : Goals scored by away team so far.
        """
        lam, mu = self.model.predict_goals(home_team, away_team)
        state = MatchState(
            minute=minute,
            home_goals=home_goals,
            away_goals=away_goals,
            home_lambda=lam,
            away_lambda=mu,
        )
        return self.simulator.simulate(state)

    def price_state(self, state: MatchState) -> SimulationResult:
        """Price any arbitrary MatchState directly."""
        return self.simulator.simulate(state)

    # ── Probability path ───────────────────────────────────────────────────────

    def probability_path(self,
                         home_team: str,
                         away_team: str,
                         step_minutes: int = 5) -> list[dict]:
        """
        Show how 1X2 probabilities evolve over time for a 0-0 match.
        Useful for identifying windows where the model disagrees with live odds.
        """
        lam, mu = self.model.predict_goals(home_team, away_team)
        return self.simulator.probability_path(lam, mu, step_minutes)

    # ── Fair odds conversion ───────────────────────────────────────────────────

    @staticmethod
    def to_decimal_odds(prob: float, min_prob: float = 0.01) -> float:
        """Convert probability to fair decimal odds (no margin)."""
        return 1.0 / max(prob, min_prob)

    def fair_odds_1x2(self, home_team: str, away_team: str) -> Dict[str, float]:
        """Return fair decimal odds for 1X2 (no bookmaker margin)."""
        probs = self.price_pre_match(home_team, away_team)
        return {
            "home": self.to_decimal_odds(probs["home"]),
            "draw": self.to_decimal_odds(probs["draw"]),
            "away": self.to_decimal_odds(probs["away"]),
        }
