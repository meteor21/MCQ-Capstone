"""
Walk-forward backtesting engine.

Methodology
-----------
1. Sort matches by date.
2. Wait until MIN_MATCHES_TO_FIT historical matches are available.
3. For each subsequent match:
   a. Train a fresh Dixon-Coles model on all matches in the rolling training
      window (up to TRAIN_WINDOW_DAYS days before the match date).
   b. Predict H/D/A probabilities.
   c. Compare with available bookmaker odds → identify +EV bets.
   d. Size bets with fractional Kelly on current bankroll.
   e. Settle bets and update bankroll.
4. Aggregate results into a performance report.

This is an "out-of-sample" test: the model never sees the match it is
predicting when it makes its prediction.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional

from .dixon_coles import DixonColesModel
from .betting import scan_match, evaluate_bet


def run_backtest(
    df: pd.DataFrame,
    initial_bankroll: float = 1000.0,
    ev_threshold: float = 0.04,
    kelly_fraction: float = 0.25,
    max_bet_fraction: float = 0.05,
    min_odds: float = 1.30,
    max_odds: float = 6.00,
    time_decay: float = 0.0065,
    min_matches_to_fit: int = 60,
    train_window_days: int = 365,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Run a walk-forward backtest.

    Parameters
    ----------
    df : full match DataFrame (date, home_team, away_team, home_goals, away_goals,
         odds_home, odds_draw, odds_away)
    initial_bankroll : starting bankroll
    ev_threshold     : minimum EV to bet
    kelly_fraction   : fractional Kelly multiplier
    max_bet_fraction : max stake as fraction of bankroll
    min_odds / max_odds : odds filter
    time_decay       : Dixon-Coles decay constant (per day)
    min_matches_to_fit : warm-up period (matches before we start betting)
    train_window_days  : rolling window for training data
    verbose          : show tqdm progress bar

    Returns
    -------
    (bet_log, metrics)
    bet_log : DataFrame of all bets placed with settlement info
    metrics : dict of performance statistics
    """
    df = df.sort_values("date").reset_index(drop=True)

    has_odds = all(c in df.columns for c in ["odds_home", "odds_draw", "odds_away"])
    if not has_odds:
        raise ValueError("Backtest requires bookmaker odds columns: "
                         "odds_home, odds_draw, odds_away")

    bankroll = initial_bankroll
    bankroll_series = [bankroll]
    bet_log: list[dict] = []

    iterator = range(min_matches_to_fit, len(df))
    if verbose:
        iterator = tqdm(iterator, desc="Backtesting", unit="match")

    for i in iterator:
        row = df.iloc[i]
        match_date = row["date"]

        # ── Build training set ────────────────────────────────────────────────
        window_start = match_date - pd.Timedelta(days=train_window_days)
        train = df.iloc[:i]
        train = train[train["date"] >= window_start]

        if len(train) < min_matches_to_fit:
            bankroll_series.append(bankroll)
            continue

        # Check both teams exist in training data
        home_team = row["home_team"]
        away_team = row["away_team"]
        all_train_teams = set(train["home_team"]) | set(train["away_team"])
        if home_team not in all_train_teams or away_team not in all_train_teams:
            bankroll_series.append(bankroll)
            continue

        # ── Fit model ────────────────────────────────────────────────────────
        model = DixonColesModel(time_decay=time_decay)
        try:
            model.fit(train, current_date=match_date)
        except Exception:
            bankroll_series.append(bankroll)
            continue

        # ── Predict ──────────────────────────────────────────────────────────
        try:
            probs = model.predict_probs(home_team, away_team)
        except Exception:
            bankroll_series.append(bankroll)
            continue

        # ── Scan for value bets ───────────────────────────────────────────────
        oh = row["odds_home"] if not pd.isna(row["odds_home"]) else None
        od = row["odds_draw"] if not pd.isna(row["odds_draw"]) else None
        oa = row["odds_away"] if not pd.isna(row["odds_away"]) else None

        bets = scan_match(
            home_team, away_team, probs,
            oh, od, oa,
            ev_threshold=ev_threshold,
            kelly_fraction=kelly_fraction,
            max_bet_fraction=max_bet_fraction,
            min_odds=min_odds,
            max_odds=max_odds,
        )

        # ── Place and settle bets ─────────────────────────────────────────────
        for bet in bets:
            stake = bet["kelly"] * bankroll
            if stake < 0.01:
                continue

            pnl_per_unit = evaluate_bet(bet, int(row["home_goals"]), int(row["away_goals"]))
            pnl = stake * pnl_per_unit
            bankroll += pnl

            bet_log.append({
                "date": match_date,
                "home_team": home_team,
                "away_team": away_team,
                "market": bet["market"],
                "prob": bet["prob"],
                "fair_odds": bet["fair_odds"],
                "book_odds": bet["book_odds"],
                "ev": bet["ev"],
                "kelly": bet["kelly"],
                "stake": round(stake, 4),
                "pnl": round(pnl, 4),
                "bankroll_after": round(bankroll, 4),
                "won": pnl > 0,
            })

        bankroll_series.append(bankroll)

    # ── Compute metrics ───────────────────────────────────────────────────────
    bet_df = pd.DataFrame(bet_log) if bet_log else pd.DataFrame()
    metrics = _compute_metrics(bet_df, initial_bankroll, bankroll, bankroll_series)
    return bet_df, metrics


def _compute_metrics(bet_df: pd.DataFrame,
                     initial_bankroll: float,
                     final_bankroll: float,
                     bankroll_series: list) -> dict:
    if bet_df.empty:
        return {
            "total_bets": 0, "win_rate": None, "roi": None,
            "total_pnl": 0, "final_bankroll": final_bankroll,
            "max_drawdown": 0, "sharpe": None,
        }

    n = len(bet_df)
    wins = int(bet_df["won"].sum())
    total_staked = bet_df["stake"].sum()
    total_pnl = bet_df["pnl"].sum()
    roi = total_pnl / total_staked if total_staked > 0 else 0.0

    # Sharpe ratio on per-bet returns (pnl / stake)
    per_bet_returns = bet_df["pnl"] / bet_df["stake"]
    sharpe = (per_bet_returns.mean() / per_bet_returns.std()
              if per_bet_returns.std() > 0 else float("nan"))

    # Max drawdown on bankroll curve
    br = np.array(bankroll_series)
    peak = np.maximum.accumulate(br)
    drawdown = (br - peak) / peak
    max_dd = float(drawdown.min())

    # Market breakdown
    by_market = (
        bet_df.groupby("market")
        .agg(bets=("pnl", "count"),
             staked=("stake", "sum"),
             pnl=("pnl", "sum"),
             wins=("won", "sum"))
        .assign(roi=lambda d: d["pnl"] / d["staked"],
                win_rate=lambda d: d["wins"] / d["bets"])
    )

    return {
        "total_bets": n,
        "wins": wins,
        "win_rate": wins / n,
        "total_staked": round(total_staked, 2),
        "total_pnl": round(total_pnl, 2),
        "roi": round(roi, 4),
        "final_bankroll": round(final_bankroll, 2),
        "max_drawdown": round(max_dd, 4),
        "sharpe": round(sharpe, 3),
        "by_market": by_market,
    }


def print_report(metrics: dict, bet_df: pd.DataFrame):
    """Pretty-print a backtest performance report."""
    print("\n" + "=" * 60)
    print("  BACKTEST PERFORMANCE REPORT")
    print("=" * 60)

    if metrics["total_bets"] == 0:
        print("  No bets placed — try lowering ev_threshold or checking data.")
        return

    print(f"  Total bets placed : {metrics['total_bets']}")
    print(f"  Win rate          : {metrics['win_rate']:.1%}")
    print(f"  Total staked      : {metrics['total_staked']:.2f}")
    print(f"  Total P&L         : {metrics['total_pnl']:+.2f}")
    print(f"  ROI               : {metrics['roi']:+.2%}")
    print(f"  Final bankroll    : {metrics['final_bankroll']:.2f}")
    print(f"  Max drawdown      : {metrics['max_drawdown']:.2%}")
    print(f"  Sharpe ratio      : {metrics['sharpe']:.3f}")

    if "by_market" in metrics:
        print("\n  ── By market ─────────────────────────────────────────")
        print(metrics["by_market"].to_string())

    print("\n  ── Top 10 bets by EV ─────────────────────────────────")
    if not bet_df.empty:
        top = (bet_df.sort_values("ev", ascending=False)
                     .head(10)[["date", "home_team", "away_team",
                                "market", "prob", "book_odds",
                                "ev", "stake", "pnl", "won"]]
                     .to_string(index=False))
        print(top)

    print("=" * 60)
