"""
Exposure tracker: monitors open risk and enforces position limits.

Prevents over-betting on a single league, team, or market type.
"""

from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExposureLimits:
    """
    Hard limits on open risk.  All values are expressed as fraction of current bankroll.
    """
    max_per_bet: float = 0.04           # max stake on a single bet
    max_per_match: float = 0.06         # max total at risk on one match
    max_per_league: float = 0.15        # max total at risk in one league
    max_per_market_type: float = 0.20   # max across all draws, all homes, etc.
    max_total: float = 0.30             # max total open exposure


class ExposureTracker:
    """
    Tracks open exposure and enforces ExposureLimits.

    Usage
    -----
    tracker = ExposureTracker(initial_bankroll=1000.0, limits=ExposureLimits())
    ok, reason = tracker.can_bet("E0", "Arsenal", "Chelsea", "home", stake=25.0)
    if ok:
        tracker.open_bet("E0", "Arsenal", "Chelsea", "home", stake=25.0)

    # When a bet settles:
    tracker.close_bet("E0", "Arsenal", "Chelsea", "home")
    """

    def __init__(self,
                 initial_bankroll: float = 1000.0,
                 limits: Optional[ExposureLimits] = None):
        self.bankroll = initial_bankroll
        self.limits = limits or ExposureLimits()
        # {(league, home, away, market): stake}
        self._open: dict[tuple, float] = {}

    def update_bankroll(self, new_bankroll: float):
        self.bankroll = new_bankroll

    @property
    def total_open(self) -> float:
        return sum(self._open.values())

    def open_by_league(self) -> dict[str, float]:
        result: dict[str, float] = defaultdict(float)
        for (league, *_), stake in self._open.items():
            result[league] += stake
        return dict(result)

    def open_by_market(self) -> dict[str, float]:
        result: dict[str, float] = defaultdict(float)
        for (_, _h, _a, market), stake in self._open.items():
            result[market] += stake
        return dict(result)

    def open_on_match(self, league: str, home: str, away: str) -> float:
        return sum(s for (l, h, a, _), s in self._open.items()
                   if l == league and h == home and a == away)

    def can_bet(self, league: str, home_team: str, away_team: str,
                market: str, stake: float) -> tuple[bool, str]:
        """
        Check whether placing this bet would violate any exposure limit.

        Returns (allowed: bool, reason: str).
        """
        br = self.bankroll

        if stake / br > self.limits.max_per_bet:
            return False, f"Stake {stake/br:.1%} > max_per_bet {self.limits.max_per_bet:.1%}"

        match_open = self.open_on_match(league, home_team, away_team)
        if (match_open + stake) / br > self.limits.max_per_match:
            return False, f"Match exposure {(match_open+stake)/br:.1%} > max_per_match"

        league_open = self.open_by_league().get(league, 0.0)
        if (league_open + stake) / br > self.limits.max_per_league:
            return False, f"League exposure {(league_open+stake)/br:.1%} > max_per_league"

        mkt_open = self.open_by_market().get(market, 0.0)
        if (mkt_open + stake) / br > self.limits.max_per_market_type:
            return False, f"Market-type exposure {(mkt_open+stake)/br:.1%} > max_per_market_type"

        if (self.total_open + stake) / br > self.limits.max_total:
            return False, f"Total exposure {(self.total_open+stake)/br:.1%} > max_total"

        return True, "ok"

    def open_bet(self, league: str, home_team: str, away_team: str,
                 market: str, stake: float):
        key = (league, home_team, away_team, market)
        self._open[key] = self._open.get(key, 0.0) + stake

    def close_bet(self, league: str, home_team: str, away_team: str, market: str):
        key = (league, home_team, away_team, market)
        self._open.pop(key, None)

    def print_exposure(self):
        br = self.bankroll
        print(f"  Total open exposure: {self.total_open:.2f} ({self.total_open/br:.1%} of bankroll)")
        for league, amt in self.open_by_league().items():
            print(f"    {league}: {amt:.2f} ({amt/br:.1%})")
