"""
Walk-forward backtest engine (production-style).

Wires together:
  - DixonColesModel (parameter estimation)
  - MarketPricer    (fair probability computation)
  - Rule            (trade execution logic)
  - PnLLogger       (trade recording and P&L)
  - CalibrationAnalyzer (probability calibration)
  - ExposureTracker (position limit enforcement)

The engine processes matches in chronological order.  For each match:
  1. Build training window (rolling, time-decay weighted).
  2. Fit the model on training data.
  3. Price the match (compute fair probabilities).
  4. Apply the Rule to generate Trade candidates.
  5. Check ExposureTracker limits.
  6. Log trades with PnLLogger.
  7. Update CalibrationAnalyzer with model predictions vs outcomes.
"""

from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..pre_match.dixon_coles import DixonColesModel
from ..markets.pricer import MarketPricer
from ..in_play.simulator import MonteCarloSimulator
from ..execution.rules import Rule, scan_match, default_pre_match_rule
from ..logging.pnl import PnLLogger
from ..logging.calibration import CalibrationAnalyzer
from ..logging.exposure import ExposureTracker, ExposureLimits


class BacktestEngine:
    """
    Walk-forward backtest with full logging.

    Parameters
    ----------
    rule             : Trade rule (default: default_pre_match_rule()).
    initial_bankroll : Starting bankroll.
    time_decay       : Dixon-Coles time-decay constant.
    min_matches      : Warm-up: minimum historical matches before first bet.
    train_days       : Rolling training window in days.
    min_games_team   : Min games per team in window (skip if fewer).
    n_simulations    : Monte Carlo paths for in-play pricing (not used pre-match).
    exposure_limits  : ExposureLimits object (None = use defaults).
    """

    def __init__(
        self,
        rule: Optional[Rule] = None,
        initial_bankroll: float = 1000.0,
        time_decay: float = 0.0065,
        min_matches: int = 60,
        train_days: int = 365,
        min_games_team: int = 5,
        n_simulations: int = 50_000,
        exposure_limits: Optional[ExposureLimits] = None,
    ):
        self.rule = rule or default_pre_match_rule()
        self.initial_bankroll = initial_bankroll
        self.time_decay = time_decay
        self.min_matches = min_matches
        self.train_days = train_days
        self.min_games_team = min_games_team
        self.n_simulations = n_simulations

        self.logger = PnLLogger(initial_bankroll)
        self.calibration = CalibrationAnalyzer()
        self.exposure = ExposureTracker(initial_bankroll, exposure_limits)

    def run(self,
            df: pd.DataFrame,
            league: str = "",
            verbose: bool = True) -> tuple[pd.DataFrame, dict]:
        """
        Run the walk-forward backtest on a single league DataFrame.

        Parameters
        ----------
        df      : Match DataFrame with columns:
                  date, home_team, away_team, home_goals, away_goals,
                  odds_home, odds_draw, odds_away
        league  : League code label for logging.
        verbose : Show tqdm progress bar.

        Returns
        -------
        (bet_log_df, summary_dict)
        """
        has_odds = all(c in df.columns for c in ["odds_home", "odds_draw", "odds_away"])
        if not has_odds:
            raise ValueError("Backtest requires odds_home, odds_draw, odds_away columns")

        df = df.sort_values("date").reset_index(drop=True)
        iterator = range(self.min_matches, len(df))
        if verbose:
            label = f"Backtest {league}" if league else "Backtest"
            iterator = tqdm(iterator, desc=label, unit="match")

        for i in iterator:
            row = df.iloc[i]
            match_date = row["date"]
            home_team  = str(row["home_team"])
            away_team  = str(row["away_team"])

            # ── Training window ───────────────────────────────────────────────
            window_start = match_date - pd.Timedelta(days=self.train_days)
            train = df.iloc[:i]
            train = train[train["date"] >= window_start]
            if len(train) < self.min_matches:
                continue

            known = set(train["home_team"]) | set(train["away_team"])
            if home_team not in known or away_team not in known:
                continue

            if self.min_games_team > 0:
                gh = ((train["home_team"] == home_team) | (train["away_team"] == home_team)).sum()
                ga = ((train["home_team"] == away_team) | (train["away_team"] == away_team)).sum()
                if gh < self.min_games_team or ga < self.min_games_team:
                    continue

            # ── Fit model ─────────────────────────────────────────────────────
            model = DixonColesModel(time_decay=self.time_decay)
            try:
                model.fit(train, current_date=match_date)
            except Exception:
                continue

            # ── Price ─────────────────────────────────────────────────────────
            try:
                pricer = MarketPricer(model, MonteCarloSimulator(50, seed=None))  # fast for pre-match 1X2
                model_probs = pricer.price_pre_match(home_team, away_team)
            except Exception:
                continue

            # ── Calibration: record ALL predictions (not just bets) ───────────
            result_home = int(row["home_goals"]) > int(row["away_goals"])
            self.calibration.add(model_probs["home"], result_home)

            # ── Book odds ─────────────────────────────────────────────────────
            def _safe(col):
                v = row.get(col)
                return float(v) if v is not None and not pd.isna(v) else None

            book = {
                "home": _safe("odds_home"),
                "draw": _safe("odds_draw"),
                "away": _safe("odds_away"),
            }
            if any(v is None for v in book.values()):
                continue

            # ── Apply rule ────────────────────────────────────────────────────
            trades = scan_match(home_team, away_team,
                                model_probs, book,
                                self.rule)

            # ── Execute ───────────────────────────────────────────────────────
            for trade in trades:
                stake = trade.kelly * self.logger.bankroll
                if stake < 0.01:
                    continue

                ok, reason = self.exposure.can_bet(
                    league, home_team, away_team, trade.market, stake)
                if not ok:
                    continue

                self.exposure.open_bet(league, home_team, away_team, trade.market, stake)
                rec = self.logger.record(
                    date=match_date,
                    league=league,
                    home_team=home_team,
                    away_team=away_team,
                    market=trade.market,
                    model_prob=trade.model_prob,
                    book_odds=trade.book_odds,
                    ev=trade.ev,
                    kelly=trade.kelly,
                    stake=stake,
                    actual_home=int(row["home_goals"]),
                    actual_away=int(row["away_goals"]),
                    trigger_rule=trade.trigger_rule,
                )
                self.exposure.close_bet(league, home_team, away_team, trade.market)
                self.exposure.update_bankroll(self.logger.bankroll)

        bet_df = self.logger.to_dataframe()
        summary = self.logger.summary()
        summary["calibration"] = self.calibration.summary()
        return bet_df, summary

    def print_report(self, bet_df: pd.DataFrame, summary: dict, league: str = ""):
        title = f"BACKTEST REPORT — {league}" if league else "BACKTEST REPORT"
        print("\n" + "=" * 64)
        print(f"  {title}")
        print("=" * 64)
        if summary.get("total_bets", 0) == 0:
            print("  No bets placed.")
        else:
            self.logger.print_summary()
            print()
            self.calibration.print_summary()
            if "by_market" not in summary:
                df_ = bet_df
                if not df_.empty:
                    bm = (df_.groupby("market")
                          .agg(bets=("pnl","count"), staked=("stake","sum"),
                               pnl=("pnl","sum"), wins=("won","sum"))
                          .assign(roi=lambda d: d["pnl"]/d["staked"],
                                  win_rate=lambda d: d["wins"]/d["bets"]))
                    print("\n  ── By market ──────────────────────────────────────────")
                    print(bm.to_string())
        print("=" * 64)
