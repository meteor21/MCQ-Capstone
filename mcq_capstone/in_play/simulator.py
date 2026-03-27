"""
Monte Carlo match simulator.

Given a MatchState, simulate the remaining match N times and return
outcome probability distributions for all standard markets.

Why Monte Carlo over analytic Poisson?
  For the 1X2 market, analytic Poisson is exact and fast.
  But for more complex markets (e.g. "home scores next", "over 2.5 goals",
  "both teams score") and for in-play states where remaining rates are
  non-trivially adjusted, simulation generalises better and is easier
  to maintain.

  We use N=50,000 by default.  At that sample size, standard error on a
  50% probability estimate is ~0.22% — precise enough for any practical use.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict

from .state import MatchState


@dataclass(frozen=True)
class SimulationResult:
    """All market probabilities computed from a simulation run."""
    # 1X2
    p_home_win: float
    p_draw: float
    p_away_win: float

    # Over/Under (total goals in full match including goals already scored)
    p_over_15: float
    p_over_25: float
    p_over_35: float
    p_under_25: float

    # Both teams to score
    p_btts: float

    # Asian handicap (whole-ball)
    p_home_ah_minus1: float   # Home −1 (home wins by ≥2)
    p_away_ah_plus1: float    # Away +1 (away wins or draws or loses by 1)

    # Next goal scorer
    p_next_home: float
    p_next_away: float
    p_no_more_goals: float

    # Diagnostics
    n_simulations: int
    state_minute: int

    @property
    def result_1x2(self) -> Dict[str, float]:
        return {"home": self.p_home_win, "draw": self.p_draw, "away": self.p_away_win}

    def __str__(self) -> str:
        lines = [
            f"SimResult @ t={self.state_minute}'  (N={self.n_simulations:,})",
            f"  1X2     home={self.p_home_win:.3f}  draw={self.p_draw:.3f}  away={self.p_away_win:.3f}",
            f"  O/U 2.5 over={self.p_over_25:.3f}  under={self.p_under_25:.3f}",
            f"  BTTS    yes={self.p_btts:.3f}",
            f"  Next    home={self.p_next_home:.3f}  away={self.p_next_away:.3f}  none={self.p_no_more_goals:.3f}",
        ]
        return "\n".join(lines)


class MonteCarloSimulator:
    """
    Simulate soccer matches from a MatchState using Poisson random variates.

    Usage
    -----
    sim = MonteCarloSimulator(n_simulations=50_000, seed=42)
    state = MatchState.pre_match(home_lambda=1.5, away_lambda=1.1)
    result = sim.simulate(state)
    print(result.p_home_win)
    """

    def __init__(self, n_simulations: int = 50_000, seed: Optional[int] = None):
        self.n_simulations = n_simulations
        self._rng = np.random.default_rng(seed)

    def simulate(self, state: MatchState) -> SimulationResult:
        """
        Run N simulations from the given MatchState.

        Samples remaining goals for home and away independently from
        Poisson(lambda_remaining) and Poisson(mu_remaining).
        Combines with already-scored goals to get final scores.
        """
        n = self.n_simulations
        lam = state.lambda_remaining
        mu = state.mu_remaining

        # Sample remaining goals
        rem_h = self._rng.poisson(lam, n)
        rem_a = self._rng.poisson(mu, n)

        # Final score
        final_h = state.home_goals + rem_h
        final_a = state.away_goals + rem_a
        total_goals = final_h + final_a

        # ── 1X2 ───────────────────────────────────────────────────────────────
        home_wins = final_h > final_a
        draws     = final_h == final_a
        away_wins = final_a > final_h

        # ── Over/Under ────────────────────────────────────────────────────────
        over_15 = total_goals > 1
        over_25 = total_goals > 2
        over_35 = total_goals > 3

        # ── BTTS ──────────────────────────────────────────────────────────────
        btts = (final_h >= 1) & (final_a >= 1)

        # ── AH ─1 / +1 ────────────────────────────────────────────────────────
        home_ah_m1 = (final_h - final_a) >= 2        # home wins by 2+
        away_ah_p1 = (final_a - final_h) >= 0        # away wins, draws, or loses by ≤1

        # ── Next goal ─────────────────────────────────────────────────────────
        # Approximate: first remaining goal is home with probability lam/(lam+mu)
        # if any goals remain, or neither if lam+mu ≈ 0
        total_rem = lam + mu
        if total_rem < 1e-8:
            p_next_home = 0.0
            p_next_away = 0.0
            p_no_more   = 1.0
        else:
            has_any_goal = (rem_h + rem_a) > 0
            p_no_more   = float(np.mean(~has_any_goal))
            # Among simulations with at least one more goal,
            # first goal is home with prob lam/(lam+mu)
            p_goal      = 1.0 - p_no_more
            p_next_home = p_goal * (lam / total_rem)
            p_next_away = p_goal * (mu  / total_rem)

        return SimulationResult(
            p_home_win=float(np.mean(home_wins)),
            p_draw=float(np.mean(draws)),
            p_away_win=float(np.mean(away_wins)),
            p_over_15=float(np.mean(over_15)),
            p_over_25=float(np.mean(over_25)),
            p_over_35=float(np.mean(over_35)),
            p_under_25=float(np.mean(~over_25)),
            p_btts=float(np.mean(btts)),
            p_home_ah_minus1=float(np.mean(home_ah_m1)),
            p_away_ah_plus1=float(np.mean(away_ah_p1)),
            p_next_home=p_next_home,
            p_next_away=p_next_away,
            p_no_more_goals=p_no_more,
            n_simulations=n,
            state_minute=state.minute,
        )

    def probability_path(self,
                         home_lambda: float,
                         away_lambda: float,
                         step_minutes: int = 5) -> list[dict]:
        """
        Simulate probability evolution for a 0-0 match from kick-off to
        full time in steps of `step_minutes`.

        Useful for understanding how bet value changes as time passes.

        Returns list of dicts: {minute, p_home, p_draw, p_away, p_over_25}.
        """
        from .state import MatchState
        path = []
        for minute in range(0, 91, step_minutes):
            state = MatchState(
                minute=minute,
                home_goals=0,
                away_goals=0,
                home_lambda=home_lambda,
                away_lambda=away_lambda,
            )
            r = self.simulate(state)
            path.append({
                "minute": minute,
                "p_home": r.p_home_win,
                "p_draw": r.p_draw,
                "p_away": r.p_away_win,
                "p_over_25": r.p_over_25,
            })
        return path


# Optional[int] needs to be imported
from typing import Optional
