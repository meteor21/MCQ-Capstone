"""
Rule-based trade execution.

All trading decisions are driven by explicit, auditable rules — never
by black-box ML outputs.  Each rule answers one question:

    Given this market state, should I place a trade?

Rules are composable:  CombinedRule(rule_a, rule_b) requires both.

Built-in rules
--------------
ValueBetRule     — bet when EV > threshold and odds are in range
DrawFilterRule   — require higher EV for draw markets (draws are noisier)
OverroundRule    — skip markets where book margin exceeds a cap
ConfidenceRule   — require model probability ≥ floor
PreMatchOnlyRule — only trade before kick-off (minute == 0)
InPlayRule       — only trade after a goal (minute > 0, score changed)
"""

from __future__ import annotations
import abc
from dataclasses import dataclass
from typing import Optional, List

from ..in_play.state import MatchState
from ..markets.edge import expected_value, overround, kelly_fraction


# ── Trade dataclass ────────────────────────────────────────────────────────────

@dataclass
class Trade:
    """A single recommended trade."""
    market: str              # 'home', 'draw', 'away', 'over_25', 'btts', etc.
    model_prob: float        # model's probability
    book_odds: float         # bookmaker decimal odds
    ev: float                # expected value
    kelly: float             # recommended fraction of bankroll
    trigger_rule: str        # which rule generated this trade
    state: Optional[MatchState] = None  # snapshot at trade time

    @property
    def fair_odds(self) -> float:
        return 1.0 / self.model_prob if self.model_prob > 0 else float("inf")

    def __str__(self) -> str:
        return (f"Trade({self.market}, prob={self.model_prob:.3f}, "
                f"odds={self.book_odds}, ev={self.ev:+.3f}, kelly={self.kelly:.4f})")


# ── Rule ABC ───────────────────────────────────────────────────────────────────

class Rule(abc.ABC):
    """
    Abstract base class for all trade rules.

    Subclass and override `evaluate` to define custom logic.
    """

    @abc.abstractmethod
    def evaluate(self,
                 market: str,
                 model_prob: float,
                 book_odds: float,
                 state: Optional[MatchState] = None,
                 all_odds: Optional[dict] = None) -> Optional[Trade]:
        """
        Evaluate whether to trade this market.

        Parameters
        ----------
        market      : 'home' | 'draw' | 'away'
        model_prob  : Model's estimated probability.
        book_odds   : Bookmaker decimal odds.
        state       : Current MatchState (None = pre-match).
        all_odds    : dict with keys 'home', 'draw', 'away' for overround calc.

        Returns None if no trade, or a Trade object if criteria are met.
        """

    def __and__(self, other: "Rule") -> "CombinedRule":
        return CombinedRule([self, other])

    def __repr__(self) -> str:
        return self.__class__.__name__


class CombinedRule(Rule):
    """Logical AND: all constituent rules must pass."""

    def __init__(self, rules: List[Rule]):
        self.rules = rules

    def evaluate(self, market, model_prob, book_odds,
                 state=None, all_odds=None) -> Optional[Trade]:
        result = None
        for rule in self.rules:
            result = rule.evaluate(market, model_prob, book_odds, state, all_odds)
            if result is None:
                return None   # any rule rejecting kills the trade
        return result

    def __repr__(self) -> str:
        return " & ".join(repr(r) for r in self.rules)


# ── Built-in rules ─────────────────────────────────────────────────────────────

class ValueBetRule(Rule):
    """
    Bet when EV exceeds a threshold and odds are within range.

    This is the core selection rule.  Combine with DrawFilterRule or
    OverroundRule for stricter filtering.
    """

    def __init__(self,
                 min_ev: float = 0.10,
                 min_odds: float = 1.50,
                 max_odds: float = 5.00,
                 kelly_fraction_: float = 0.25,
                 max_kelly: float = 0.04):
        self.min_ev = min_ev
        self.min_odds = min_odds
        self.max_odds = max_odds
        self.kelly_fraction_ = kelly_fraction_
        self.max_kelly = max_kelly

    def evaluate(self, market, model_prob, book_odds,
                 state=None, all_odds=None) -> Optional[Trade]:
        if book_odds < self.min_odds or book_odds > self.max_odds:
            return None
        ev = expected_value(model_prob, book_odds)
        if ev < self.min_ev:
            return None
        k = kelly_fraction(model_prob, book_odds,
                           fractional=self.kelly_fraction_,
                           max_f=self.max_kelly)
        if k <= 0:
            return None
        return Trade(market=market, model_prob=model_prob,
                     book_odds=book_odds, ev=ev, kelly=k,
                     trigger_rule=repr(self), state=state)


class DrawFilterRule(Rule):
    """
    Apply a stricter EV threshold to the draw market.
    Pass-through (no filter) for non-draw markets.

    Used in conjunction with ValueBetRule.
    """

    def __init__(self, draw_min_ev: float = 0.15):
        self.draw_min_ev = draw_min_ev

    def evaluate(self, market, model_prob, book_odds,
                 state=None, all_odds=None) -> Optional[Trade]:
        if market == "draw":
            ev = expected_value(model_prob, book_odds)
            if ev < self.draw_min_ev:
                return None
        # For non-draw markets, this rule always passes — pair with ValueBetRule
        return Trade(market=market, model_prob=model_prob,
                     book_odds=book_odds,
                     ev=expected_value(model_prob, book_odds),
                     kelly=0.0, trigger_rule=repr(self), state=state)


class OverroundRule(Rule):
    """
    Reject the entire market if the bookmaker's margin is too high.
    Requires all_odds dict to compute overround.
    """

    def __init__(self, max_overround: float = 0.12):
        self.max_overround = max_overround

    def evaluate(self, market, model_prob, book_odds,
                 state=None, all_odds=None) -> Optional[Trade]:
        if all_odds is None:
            return None   # can't check without all three lines
        try:
            or_ = overround(all_odds["home"], all_odds["draw"], all_odds["away"])
        except (KeyError, TypeError, ZeroDivisionError):
            return None
        if or_ > self.max_overround:
            return None
        return Trade(market=market, model_prob=model_prob,
                     book_odds=book_odds,
                     ev=expected_value(model_prob, book_odds),
                     kelly=0.0, trigger_rule=repr(self), state=state)


class ConfidenceRule(Rule):
    """Only trade when the model assigns at least min_prob to the outcome."""

    def __init__(self, min_prob: float = 0.40):
        self.min_prob = min_prob

    def evaluate(self, market, model_prob, book_odds,
                 state=None, all_odds=None) -> Optional[Trade]:
        if model_prob < self.min_prob:
            return None
        return Trade(market=market, model_prob=model_prob,
                     book_odds=book_odds,
                     ev=expected_value(model_prob, book_odds),
                     kelly=0.0, trigger_rule=repr(self), state=state)


class PreMatchOnlyRule(Rule):
    """Only trade at minute == 0 (pre-match)."""

    def evaluate(self, market, model_prob, book_odds,
                 state=None, all_odds=None) -> Optional[Trade]:
        if state is not None and state.minute > 0:
            return None
        return Trade(market=market, model_prob=model_prob,
                     book_odds=book_odds,
                     ev=expected_value(model_prob, book_odds),
                     kelly=0.0, trigger_rule=repr(self), state=state)


class InPlayGoalRule(Rule):
    """
    Only trade in-play, and only immediately after a goal event.

    The idea: after a goal, in-play odds may be slow to update — this is
    when the model has the most edge.  Requires state.minute > 0.
    """

    def __init__(self, max_minutes_after_goal: int = 3):
        self.max_minutes_after_goal = max_minutes_after_goal
        self._last_total_goals: int = 0
        self._goal_minute: int = 0

    def notify_goal(self, state: MatchState):
        """Call this whenever a goal is scored to arm the rule."""
        total = state.home_goals + state.away_goals
        if total > self._last_total_goals:
            self._last_total_goals = total
            self._goal_minute = state.minute

    def evaluate(self, market, model_prob, book_odds,
                 state=None, all_odds=None) -> Optional[Trade]:
        if state is None or state.minute == 0:
            return None
        minutes_since_goal = state.minute - self._goal_minute
        if minutes_since_goal > self.max_minutes_after_goal:
            return None
        return Trade(market=market, model_prob=model_prob,
                     book_odds=book_odds,
                     ev=expected_value(model_prob, book_odds),
                     kelly=0.0, trigger_rule=repr(self), state=state)


# ── Rule factory ───────────────────────────────────────────────────────────────

def default_pre_match_rule(
    min_ev: float = 0.10,
    draw_min_ev: float = 0.15,
    min_prob: float = 0.40,
    max_overround: float = 0.12,
    min_odds: float = 1.50,
    max_odds: float = 5.00,
) -> CombinedRule:
    """
    Standard pre-match rule stack (recommended starting point).

    Passes only if ALL of:
      1. Odds within [min_odds, max_odds]
      2. EV > min_ev for H/A, EV > draw_min_ev for draw
      3. Model prob ≥ min_prob
      4. Market overround ≤ max_overround
    """
    return CombinedRule([
        OverroundRule(max_overround),
        ConfidenceRule(min_prob),
        DrawFilterRule(draw_min_ev),
        ValueBetRule(min_ev, min_odds, max_odds),
    ])


# ── Scan helper ────────────────────────────────────────────────────────────────

def scan_match(home_team: str,
               away_team: str,
               model_probs: dict,
               book_odds: dict,
               rule: Rule,
               state: Optional[MatchState] = None) -> list[Trade]:
    """
    Apply a rule to all three outcomes of a match.

    Parameters
    ----------
    model_probs : {home, draw, away} → float
    book_odds   : {home, draw, away} → float (None values are skipped)
    rule        : Rule instance to apply

    Returns list of Trade objects (may be empty).
    """
    trades = []
    for market in ["home", "draw", "away"]:
        prob = model_probs.get(market)
        odds = book_odds.get(market)
        if prob is None or odds is None or (isinstance(odds, float) and odds != odds):
            continue
        trade = rule.evaluate(
            market=market,
            model_prob=prob,
            book_odds=float(odds),
            state=state,
            all_odds={k: v for k, v in book_odds.items() if v is not None},
        )
        if trade is not None:
            trade.market = market   # ensure market is set
            trades.append(trade)
    return trades
