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
   c. Compare with bookmaker odds → identify bets passing all filters.
   d. Size bets with fractional Kelly on current bankroll.
   e. Settle bets and update bankroll.
4. Aggregate results into a performance report.

The model never sees the match it is predicting when it predicts it.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from tqdm import tqdm

from .dixon_coles import DixonColesModel
from .betting import scan_match, evaluate_bet


def run_backtest(
    df: pd.DataFrame,
    initial_bankroll: float = 1000.0,
    # bet-selection thresholds
    ev_threshold: float = 0.10,
    draw_ev_threshold: float = 0.15,
    min_model_prob: float = 0.40,
    max_book_overround: float = 0.12,
    # sizing
    kelly_fraction: float = 0.25,
    max_bet_fraction: float = 0.04,
    # odds filter
    min_odds: float = 1.50,
    max_odds: float = 5.00,
    # model
    time_decay: float = 0.0065,
    # window
    min_matches_to_fit: int = 60,
    train_window_days: int = 365,
    min_games_per_team: int = 5,   # skip matches where either team has < N games in window
    league_label: str = "",
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Run a walk-forward backtest on a single league dataset.

    Returns (bet_log_df, metrics_dict).
    """
    df = df.sort_values("date").reset_index(drop=True)

    has_odds = all(c in df.columns for c in ["odds_home", "odds_draw", "odds_away"])
    if not has_odds:
        raise ValueError("Backtest requires columns: odds_home, odds_draw, odds_away")

    bankroll = initial_bankroll
    bankroll_series = [bankroll]
    bet_log: list[dict] = []

    iterator = range(min_matches_to_fit, len(df))
    if verbose:
        label = f"Backtesting {league_label}" if league_label else "Backtesting"
        iterator = tqdm(iterator, desc=label, unit="match")

    for i in iterator:
        row = df.iloc[i]
        match_date = row["date"]

        # ── Training window ───────────────────────────────────────────────────
        window_start = match_date - pd.Timedelta(days=train_window_days)
        train = df.iloc[:i]
        train = train[train["date"] >= window_start]

        if len(train) < min_matches_to_fit:
            bankroll_series.append(bankroll)
            continue

        home_team = row["home_team"]
        away_team = row["away_team"]
        known = set(train["home_team"]) | set(train["away_team"])
        if home_team not in known or away_team not in known:
            bankroll_series.append(bankroll)
            continue

        # Skip if either team has too few games in window (unreliable parameters)
        if min_games_per_team > 0:
            games_home = ((train["home_team"] == home_team) |
                          (train["away_team"] == home_team)).sum()
            games_away = ((train["home_team"] == away_team) |
                          (train["away_team"] == away_team)).sum()
            if games_home < min_games_per_team or games_away < min_games_per_team:
                bankroll_series.append(bankroll)
                continue

        # ── Fit ───────────────────────────────────────────────────────────────
        model = DixonColesModel(time_decay=time_decay)
        try:
            model.fit(train, current_date=match_date)
        except Exception:
            bankroll_series.append(bankroll)
            continue

        # ── Predict ───────────────────────────────────────────────────────────
        try:
            probs = model.predict_probs(home_team, away_team)
        except Exception:
            bankroll_series.append(bankroll)
            continue

        # ── Scan ──────────────────────────────────────────────────────────────
        def _safe(x):
            return float(x) if x and not pd.isna(x) else None

        bets = scan_match(
            home_team, away_team, probs,
            _safe(row["odds_home"]),
            _safe(row["odds_draw"]),
            _safe(row["odds_away"]),
            ev_threshold=ev_threshold,
            draw_ev_threshold=draw_ev_threshold,
            min_model_prob=min_model_prob,
            max_book_overround=max_book_overround,
            kelly_fraction=kelly_fraction,
            max_bet_fraction=max_bet_fraction,
            min_odds=min_odds,
            max_odds=max_odds,
        )

        # ── Settle ────────────────────────────────────────────────────────────
        for bet in bets:
            stake = bet["kelly"] * bankroll
            if stake < 0.01:
                continue
            pnl = stake * evaluate_bet(bet, int(row["home_goals"]), int(row["away_goals"]))
            bankroll += pnl
            bet_log.append({
                "date": match_date,
                "league": league_label,
                "home_team": home_team,
                "away_team": away_team,
                "market": bet["market"],
                "prob": bet["prob"],
                "fair_odds": bet["fair_odds"],
                "book_odds": bet["book_odds"],
                "ev": bet["ev"],
                "overround": bet["overround"],
                "kelly": bet["kelly"],
                "stake": round(stake, 4),
                "pnl": round(pnl, 4),
                "bankroll_after": round(bankroll, 4),
                "won": pnl > 0,
            })

        bankroll_series.append(bankroll)

    bet_df = pd.DataFrame(bet_log) if bet_log else pd.DataFrame()
    metrics = _compute_metrics(bet_df, initial_bankroll, bankroll, bankroll_series)
    return bet_df, metrics


def _compute_metrics(bet_df, initial_bankroll, final_bankroll, bankroll_series) -> dict:
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

    per_bet = bet_df["pnl"] / bet_df["stake"]
    sharpe = float(per_bet.mean() / per_bet.std()) if per_bet.std() > 0 else float("nan")

    br = np.array(bankroll_series)
    peak = np.maximum.accumulate(br)
    max_dd = float(((br - peak) / peak).min())

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


def print_report(metrics: dict, bet_df: pd.DataFrame, title: str = "BACKTEST PERFORMANCE REPORT"):
    print("\n" + "=" * 62)
    print(f"  {title}")
    print("=" * 62)

    if metrics["total_bets"] == 0:
        print("  No bets placed — try lowering thresholds or adding more data.")
        print("=" * 62)
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
        print("\n  ── By market ───────────────────────────────────────────")
        print(metrics["by_market"].to_string())

    if not bet_df.empty:
        print("\n  ── Top 10 bets by EV ───────────────────────────────────")
        cols = ["date", "league", "home_team", "away_team",
                "market", "prob", "book_odds", "ev", "stake", "pnl", "won"]
        cols = [c for c in cols if c in bet_df.columns]
        print(bet_df.sort_values("ev", ascending=False)
                    .head(10)[cols]
                    .to_string(index=False))

    print("=" * 62)
