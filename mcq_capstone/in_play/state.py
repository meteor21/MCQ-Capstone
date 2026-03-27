"""
Match state model.

Tracks the state of a soccer match at any discrete time step (minute),
and provides adjusted Poisson rates for remaining goals.

The core insight:
    If pre-match expected goals are (λ, μ), then at minute t with
    score (g_h, g_a), remaining expected goals are:

        λ_rem = λ × f(t) × ψ(g_h − g_a)
        μ_rem = μ × f(t) × ψ(g_a − g_h)

    where:
        f(t)             = (90 − t) / 90     time fraction remaining
        ψ(lead)          = goal-rate modifier as a function of current goal difference
                           (trailing teams attack more; leading teams defend)

    ψ values are taken from Dixon & Robinson (1998) empirical estimates:
        lead = -2 (trailing by 2)  → ψ ≈ 1.20
        lead = -1 (trailing by 1)  → ψ ≈ 1.10
        lead =  0 (level)          → ψ ≈ 1.00
        lead = +1 (leading by 1)   → ψ ≈ 0.87
        lead = +2 (leading by 2)   → ψ ≈ 0.80

Reference:
    Dixon, M.J. & Robinson, M.E. (1998). A birth process model for
    association football matches. Journal of the Royal Statistical Society:
    Series D (The Statistician), 47(3), 523-538.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# ── Dixon-Robinson lead-rate table ────────────────────────────────────────────
# Maps goal lead to scoring rate multiplier.  Clamped outside ±2.
_LEAD_RATE: dict[int, float] = {
    -2: 1.20,
    -1: 1.10,
     0: 1.00,
     1: 0.87,
     2: 0.80,
}


def _lead_modifier(lead: int) -> float:
    """
    Goal-rate modifier for a team that currently leads by `lead` goals.
    Negative lead = trailing, positive = ahead.
    """
    clamped = max(-2, min(2, lead))
    return _LEAD_RATE[clamped]


@dataclass
class MatchState:
    """
    Complete snapshot of a match at a given minute.

    Parameters
    ----------
    minute       : Match time in minutes (0 = pre-match kick-off, 90 = full time).
    home_goals   : Current home goals.
    away_goals   : Current away goals.
    home_lambda  : Pre-match expected goals for home team (from Dixon-Coles).
    away_lambda  : Pre-match expected goals for away team (from Dixon-Coles).
    total_minutes: Full match duration (default 90; use 120 for extra time).
    """
    minute: int
    home_goals: int
    away_goals: int
    home_lambda: float
    away_lambda: float
    total_minutes: int = 90

    # ── Derived properties ─────────────────────────────────────────────────────

    @property
    def time_remaining(self) -> float:
        """Minutes remaining in the match."""
        return float(max(self.total_minutes - self.minute, 0))

    @property
    def time_fraction_remaining(self) -> float:
        """Fraction of match time remaining (1.0 at kick-off, 0.0 at FT)."""
        return self.time_remaining / self.total_minutes

    @property
    def goal_diff(self) -> int:
        """Home goals minus away goals."""
        return self.home_goals - self.away_goals

    @property
    def lambda_remaining(self) -> float:
        """
        Adjusted expected remaining goals for the HOME team.
        Incorporates time fraction + lead-based rate modifier.
        """
        lead = self.goal_diff          # home lead
        return (self.home_lambda
                * self.time_fraction_remaining
                * _lead_modifier(lead))

    @property
    def mu_remaining(self) -> float:
        """
        Adjusted expected remaining goals for the AWAY team.
        """
        lead = -self.goal_diff         # away lead = negative of home lead
        return (self.away_lambda
                * self.time_fraction_remaining
                * _lead_modifier(lead))

    # ── Factory methods ────────────────────────────────────────────────────────

    @classmethod
    def pre_match(cls, home_lambda: float, away_lambda: float) -> "MatchState":
        """Create a pre-match state (minute=0, score 0-0)."""
        return cls(minute=0, home_goals=0, away_goals=0,
                   home_lambda=home_lambda, away_lambda=away_lambda)

    def after_goal(self, scorer: str, minute: int) -> "MatchState":
        """Return a new MatchState after a goal is scored."""
        new_hg = self.home_goals + (1 if scorer == "home" else 0)
        new_ag = self.away_goals + (1 if scorer == "away" else 0)
        return MatchState(
            minute=minute,
            home_goals=new_hg,
            away_goals=new_ag,
            home_lambda=self.home_lambda,
            away_lambda=self.away_lambda,
            total_minutes=self.total_minutes,
        )

    def at_minute(self, minute: int) -> "MatchState":
        """Return a copy at a different minute (same score)."""
        return MatchState(
            minute=minute,
            home_goals=self.home_goals,
            away_goals=self.away_goals,
            home_lambda=self.home_lambda,
            away_lambda=self.away_lambda,
            total_minutes=self.total_minutes,
        )

    # ── Display ────────────────────────────────────────────────────────────────

    def __str__(self) -> str:
        return (f"MatchState(t={self.minute}', "
                f"score={self.home_goals}-{self.away_goals}, "
                f"λ_rem={self.lambda_remaining:.3f}, "
                f"μ_rem={self.mu_remaining:.3f})")
