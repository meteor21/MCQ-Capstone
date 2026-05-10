"""
Multi-league backtest runner.

Creates one BacktestEngine per league, runs them in sequence,
then aggregates results.
"""

from __future__ import annotations
from pathlib import Path
import urllib.request

import pandas as pd

from ..data.loader import load_csv
from ..execution.rules import Rule, default_pre_match_rule
from ..logging.exposure import ExposureLimits
from .engine import BacktestEngine


DOMESTIC_LEAGUES = {
    "E0":  "Premier League (England)",
    "SP1": "La Liga (Spain)",
    "D1":  "Bundesliga (Germany)",
    "I1":  "Serie A (Italy)",
    "F1":  "Ligue 1 (France)",
    "N1":  "Eredivisie (Netherlands)",
    "P1":  "Primeira Liga (Portugal)",
    "B1":  "Belgian Pro League",
    "T1":  "Süper Lig (Turkey)",
    "G1":  "Super League (Greece)",
}

DEFAULT_SEASONS = ["2122", "2223", "2324"]


def download_all(data_dir: Path,
                 leagues: dict = None,
                 seasons: list = None,
                 force: bool = False) -> list[str]:
    leagues = leagues or DOMESTIC_LEAGUES
    seasons = seasons or DEFAULT_SEASONS
    data_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for code in leagues:
        for season in seasons:
            dest = data_dir / f"{code}_{season}.csv"
            if dest.exists() and not force:
                downloaded.append(str(dest))
                continue
            url = f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
            try:
                urllib.request.urlretrieve(url, dest)
                downloaded.append(str(dest))
            except Exception as e:
                print(f"  Warning: {code}/{season}: {e}")
    return downloaded


def load_all(data_dir: Path,
             leagues: dict = None,
             seasons: list = None) -> dict[str, pd.DataFrame]:
    leagues = leagues or DOMESTIC_LEAGUES
    seasons = seasons or DEFAULT_SEASONS
    result = {}
    for code, name in leagues.items():
        frames = []
        for season in seasons:
            path = data_dir / f"{code}_{season}.csv"
            if not path.exists():
                continue
            try:
                df = load_csv(path)
                df["league"] = code
                frames.append(df)
            except Exception as e:
                print(f"  Warning: {path.name}: {e}")
        if frames:
            combined = pd.concat(frames).sort_values("date").reset_index(drop=True)
            result[code] = combined
            print(f"  {code:4s}  {name:<34s}  {len(combined):4d} matches")
    return result


def run_multi_league(
    league_data: dict[str, pd.DataFrame],
    rule: Rule = None,
    initial_bankroll: float = 1000.0,
    time_decay: float = 0.0065,
    min_matches: int = 60,
    train_days: int = 365,
    min_games_team: int = 5,
    exposure_limits: ExposureLimits = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Run BacktestEngine on each league and aggregate.

    Returns (combined_bet_log, aggregate_metrics).
    """
    rule = rule or default_pre_match_rule()
    all_bets = []

    for code, df in league_data.items():
        if not all(c in df.columns for c in ["odds_home", "odds_draw", "odds_away"]):
            continue

        engine = BacktestEngine(
            rule=rule,
            initial_bankroll=initial_bankroll,
            time_decay=time_decay,
            min_matches=min_matches,
            train_days=train_days,
            min_games_team=min_games_team,
            exposure_limits=exposure_limits,
        )
        bet_df, metrics = engine.run(df, league=code, verbose=False)

        n = metrics.get("total_bets", 0)
        roi = metrics.get("roi", 0) or 0
        pnl = metrics.get("total_pnl", 0) or 0
        wr  = metrics.get("win_rate", 0) or 0
        cal = metrics.get("calibration", {})
        bs  = cal.get("brier_score", float("nan"))
        print(f"  {code:4s}  bets={n:4d}  win={wr:.1%}  ROI={roi:+.2%}  "
              f"P&L={pnl:+.0f}  Brier={bs:.4f}")

        if not bet_df.empty:
            all_bets.append(bet_df)

    if not all_bets:
        return pd.DataFrame(), {"total_bets": 0}

    combined = pd.concat(all_bets).sort_values("date").reset_index(drop=True)

    total_staked = combined["stake"].sum()
    total_pnl    = combined["pnl"].sum()
    n            = len(combined)
    wins         = int(combined["won"].sum())
    roi          = total_pnl / total_staked if total_staked > 0 else 0.0
    per_bet      = combined["pnl"] / combined["stake"]

    import numpy as np
    sharpe = (float(per_bet.mean() / per_bet.std())
              if len(per_bet) > 1 and per_bet.std() > 0 else float("nan"))

    by_market = (
        combined.groupby("market")
        .agg(bets=("pnl","count"), staked=("stake","sum"),
             pnl=("pnl","sum"), wins=("won","sum"))
        .assign(roi=lambda d: d["pnl"]/d["staked"],
                win_rate=lambda d: d["wins"]/d["bets"])
    )
    by_league = (
        combined.groupby("league")
        .agg(bets=("pnl","count"), staked=("stake","sum"),
             pnl=("pnl","sum"), wins=("won","sum"))
        .assign(roi=lambda d: d["pnl"]/d["staked"],
                win_rate=lambda d: d["wins"]/d["bets"])
        .sort_values("roi", ascending=False)
    )

    return combined, {
        "total_bets": n, "wins": wins,
        "win_rate": wins / n,
        "total_staked": round(total_staked, 2),
        "total_pnl": round(total_pnl, 2),
        "roi": round(roi, 4),
        "sharpe": round(sharpe, 3),
        "by_market": by_market,
        "by_league": by_league,
    }


def print_multi_report(metrics: dict, bet_df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("  MULTI-LEAGUE BACKTEST REPORT")
    print("=" * 70)
    if metrics.get("total_bets", 0) == 0:
        print("  No bets placed.")
        print("=" * 70)
        return
    print(f"  Total bets   : {metrics['total_bets']}")
    print(f"  Win rate     : {metrics['win_rate']:.1%}")
    print(f"  Total staked : {metrics['total_staked']:.2f}")
    print(f"  Total P&L    : {metrics['total_pnl']:+.2f}")
    print(f"  ROI          : {metrics['roi']:+.2%}")
    print(f"  Sharpe       : {metrics['sharpe']:.3f}")
    if "by_market" in metrics:
        print("\n  ── By market ─────────────────────────────────────────────────")
        print(metrics["by_market"].to_string())
    if "by_league" in metrics:
        print("\n  ── By league ─────────────────────────────────────────────────")
        print(metrics["by_league"].to_string())
    if not bet_df.empty:
        print("\n  ── Top 10 bets by EV ─────────────────────────────────────────")
        cols = ["date","league","home_team","away_team","market",
                "model_prob","book_odds","ev","stake","pnl","won"]
        cols = [c for c in cols if c in bet_df.columns]
        print(bet_df.sort_values("ev", ascending=False).head(10)[cols].to_string(index=False))
    print("=" * 70)
