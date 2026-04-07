"""
P&L logger: tracks all trades and computes running statistics.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd


@dataclass
class BetRecord:
    """A single settled bet."""
    bet_id: int
    date: object
    league: str
    home_team: str
    away_team: str
    market: str
    model_prob: float
    book_odds: float
    ev: float
    kelly: float
    stake: float
    pnl: float
    won: bool
    trigger_rule: str = ""


class PnLLogger:
    """
    Accumulates bet records and provides running P&L statistics.

    Usage
    -----
    logger = PnLLogger(initial_bankroll=1000.0)
    logger.record(date, "E0", "Arsenal", "Chelsea", "home",
                  model_prob=0.52, book_odds=2.10, ev=0.092,
                  kelly=0.025, stake=25.0, actual_home=2, actual_away=1)
    df = logger.to_dataframe()
    print(logger.summary())
    """

    def __init__(self, initial_bankroll: float = 1000.0):
        self.initial_bankroll = initial_bankroll
        self.bankroll = initial_bankroll
        self._records: list[BetRecord] = []
        self._next_id = 1

    def record(self,
               date,
               league: str,
               home_team: str,
               away_team: str,
               market: str,
               model_prob: float,
               book_odds: float,
               ev: float,
               kelly: float,
               stake: float,
               actual_home: int,
               actual_away: int,
               trigger_rule: str = "") -> BetRecord:
        """
        Record a bet and settle it based on the actual result.
        Updates running bankroll.
        """
        won = _did_win(market, actual_home, actual_away)
        pnl = stake * (book_odds - 1.0) if won else -stake
        self.bankroll += pnl

        rec = BetRecord(
            bet_id=self._next_id,
            date=date,
            league=league,
            home_team=home_team,
            away_team=away_team,
            market=market,
            model_prob=model_prob,
            book_odds=book_odds,
            ev=ev,
            kelly=kelly,
            stake=stake,
            pnl=pnl,
            won=won,
            trigger_rule=trigger_rule,
        )
        self._records.append(rec)
        self._next_id += 1
        return rec

    def to_dataframe(self) -> pd.DataFrame:
        if not self._records:
            return pd.DataFrame()
        return pd.DataFrame([vars(r) for r in self._records])

    def summary(self) -> dict:
        df = self.to_dataframe()
        if df.empty:
            return {"total_bets": 0, "bankroll": self.bankroll}

        n = len(df)
        total_staked = df["stake"].sum()
        total_pnl = df["pnl"].sum()

        per_bet = df["pnl"] / df["stake"]
        sharpe = (float(per_bet.mean() / per_bet.std())
                  if len(per_bet) > 1 and per_bet.std() > 0 else float("nan"))

        # Max drawdown
        bankroll_curve = self.initial_bankroll + df["pnl"].cumsum()
        peak = bankroll_curve.cummax()
        max_dd = float(((bankroll_curve - peak) / peak).min())

        return {
            "total_bets": n,
            "wins": int(df["won"].sum()),
            "win_rate": float(df["won"].mean()),
            "total_staked": round(total_staked, 2),
            "total_pnl": round(total_pnl, 2),
            "roi": round(total_pnl / total_staked, 4) if total_staked > 0 else 0.0,
            "bankroll": round(self.bankroll, 2),
            "max_drawdown": round(max_dd, 4),
            "sharpe": round(sharpe, 3),
        }

    def print_summary(self):
        s = self.summary()
        print(f"  Bets:      {s['total_bets']}")
        print(f"  Win rate:  {s.get('win_rate', 0):.1%}")
        print(f"  ROI:       {s.get('roi', 0):+.2%}")
        print(f"  P&L:       {s.get('total_pnl', 0):+.2f}")
        print(f"  Bankroll:  {s['bankroll']:.2f}")
        print(f"  Max DD:    {s.get('max_drawdown', 0):.2%}")
        print(f"  Sharpe:    {s.get('sharpe', float('nan')):.3f}")


def _did_win(market: str, home: int, away: int) -> bool:
    if market == "home":   return home > away
    if market == "draw":   return home == away
    if market == "away":   return away > home
    if market == "over_25": return (home + away) > 2
    if market == "under_25": return (home + away) <= 2
    if market == "btts":   return home >= 1 and away >= 1
    return False
