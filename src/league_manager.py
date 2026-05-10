"""
Multi-league manager: download, load, backtest, and scan across all leagues.

Supported out of the box:
  10 domestic leagues via football-data.co.uk (see config.DOMESTIC_LEAGUES)

UEFA competitions (Champions League, Europa League, Conference League):
  football-data.co.uk does not provide this data in a compatible CSV format.
  To add UEFA:
    1. Export a CSV with columns: Date, HomeTeam, AwayTeam, FTHG, FTAG,
       B365H, B365D, B365A
    2. Place it in data/sample/UEFA_<season>.csv
    3. It will be picked up automatically when running multi-league commands.
"""

from __future__ import annotations
import urllib.request
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from .data_loader import load_csv
from .backtest import run_backtest, print_report, _compute_metrics
from .betting import scan_match, bets_to_dataframe
from .dixon_coles import DixonColesModel


# ── Download ───────────────────────────────────────────────────────────────────

def download_all(data_dir: Path,
                 leagues: dict,
                 seasons: list[str],
                 force: bool = False) -> list[str]:
    """
    Download league CSVs from football-data.co.uk.

    Returns list of successfully written file paths.
    """
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
                print(f"  Warning: could not download {code}/{season}: {e}")
    return downloaded


# ── Load ───────────────────────────────────────────────────────────────────────

def load_league_data(data_dir: Path,
                     leagues: dict,
                     seasons: list[str]) -> dict[str, pd.DataFrame]:
    """
    Load all available CSVs, concatenate by league code.

    Returns {league_code: DataFrame}.
    """
    result: dict[str, pd.DataFrame] = {}
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
                print(f"  Warning: could not load {path.name}: {e}")
        if frames:
            combined = pd.concat(frames).sort_values("date").reset_index(drop=True)
            result[code] = combined
            print(f"  {code:4s}  {name:<34s}  {len(combined):4d} matches "
                  f"({combined['date'].min().date()} → {combined['date'].max().date()})")
    return result


# ── Backtest ───────────────────────────────────────────────────────────────────

def backtest_all_leagues(
    league_data: dict[str, pd.DataFrame],
    verbose_per_league: bool = False,
    min_games_per_team: int = 5,
    **backtest_kwargs,
) -> tuple[pd.DataFrame, dict]:
    """
    Run walk-forward backtest for every loaded league.
    Returns combined (bet_log_df, aggregate_metrics).
    """
    all_bets: list[pd.DataFrame] = []
    final_bankroll_sum = 0.0
    initial_bankroll = backtest_kwargs.get("initial_bankroll", 1000.0)

    for code, df in league_data.items():
        has_odds = all(c in df.columns for c in ["odds_home", "odds_draw", "odds_away"])
        if not has_odds:
            print(f"  Skipping {code}: no odds columns")
            continue
        bet_df, metrics = run_backtest(
            df,
            league_label=code,
            verbose=verbose_per_league,
            min_games_per_team=min_games_per_team,
            **backtest_kwargs,
        )
        if not bet_df.empty:
            all_bets.append(bet_df)
        final_bankroll_sum += metrics["final_bankroll"]
        if metrics["total_bets"] > 0:
            print(f"  {code:4s}  bets={metrics['total_bets']:4d}  "
                  f"win={metrics['win_rate']:.1%}  "
                  f"ROI={metrics['roi']:+.2%}  "
                  f"P&L={metrics['total_pnl']:+.0f}")
        else:
            print(f"  {code:4s}  no bets")

    if not all_bets:
        return pd.DataFrame(), {"total_bets": 0}

    combined = pd.concat(all_bets).sort_values("date").reset_index(drop=True)

    # Compute aggregate metrics as if bankrolls were independent (per-league)
    total_staked = combined["stake"].sum()
    total_pnl = combined["pnl"].sum()
    n = len(combined)
    wins = int(combined["won"].sum())
    roi = total_pnl / total_staked if total_staked > 0 else 0.0

    per_bet = combined["pnl"] / combined["stake"]
    import numpy as np
    sharpe = float(per_bet.mean() / per_bet.std()) if per_bet.std() > 0 else float("nan")

    by_market = (
        combined.groupby("market")
        .agg(bets=("pnl", "count"),
             staked=("stake", "sum"),
             pnl=("pnl", "sum"),
             wins=("won", "sum"))
        .assign(roi=lambda d: d["pnl"] / d["staked"],
                win_rate=lambda d: d["wins"] / d["bets"])
    )

    by_league = (
        combined.groupby("league")
        .agg(bets=("pnl", "count"),
             staked=("stake", "sum"),
             pnl=("pnl", "sum"),
             wins=("won", "sum"))
        .assign(roi=lambda d: d["pnl"] / d["staked"],
                win_rate=lambda d: d["wins"] / d["bets"])
        .sort_values("roi", ascending=False)
    )

    metrics = {
        "total_bets": n,
        "wins": wins,
        "win_rate": wins / n,
        "total_staked": round(total_staked, 2),
        "total_pnl": round(total_pnl, 2),
        "roi": round(roi, 4),
        "final_bankroll": round(final_bankroll_sum, 2),
        "max_drawdown": None,   # not meaningful across independent bankrolls
        "sharpe": round(sharpe, 3),
        "by_market": by_market,
        "by_league": by_league,
    }
    return combined, metrics


def print_multi_report(metrics: dict, bet_df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("  MULTI-LEAGUE BACKTEST REPORT")
    print("=" * 70)

    if metrics.get("total_bets", 0) == 0:
        print("  No bets placed.")
        print("=" * 70)
        return

    print(f"  Total bets        : {metrics['total_bets']}")
    print(f"  Win rate          : {metrics['win_rate']:.1%}")
    print(f"  Total staked      : {metrics['total_staked']:.2f}")
    print(f"  Total P&L         : {metrics['total_pnl']:+.2f}")
    print(f"  ROI (all leagues) : {metrics['roi']:+.2%}")
    print(f"  Sharpe ratio      : {metrics['sharpe']:.3f}")

    if "by_market" in metrics:
        print("\n  ── By market ───────────────────────────────────────────────────")
        print(metrics["by_market"].to_string())

    if "by_league" in metrics:
        print("\n  ── By league (sorted by ROI) ───────────────────────────────────")
        print(metrics["by_league"].to_string())

    if not bet_df.empty:
        print("\n  ── Top 15 bets by EV ───────────────────────────────────────────")
        cols = ["date", "league", "home_team", "away_team",
                "market", "prob", "book_odds", "ev", "stake", "pnl", "won"]
        cols = [c for c in cols if c in bet_df.columns]
        print(bet_df.sort_values("ev", ascending=False)
                    .head(15)[cols]
                    .to_string(index=False))

    print("=" * 70)


# ── Live scan ──────────────────────────────────────────────────────────────────

def scan_all_leagues(
    league_data: dict[str, pd.DataFrame],
    fixtures_df: Optional[pd.DataFrame] = None,
    **bet_kwargs,
) -> pd.DataFrame:
    """
    Fit a model per league on all available data, then scan fixtures for value.

    If fixtures_df is None, no odds are available and only probabilities are shown.

    fixtures_df columns: league, home_team, away_team, [odds_home, odds_draw, odds_away]
    """
    all_bets = []

    for code, df in league_data.items():
        model = DixonColesModel(time_decay=bet_kwargs.get("time_decay", 0.0065))
        try:
            model.fit(df)
        except Exception as e:
            print(f"  Warning: could not fit {code}: {e}")
            continue

        if fixtures_df is not None:
            league_fixtures = fixtures_df[fixtures_df["league"] == code]
        else:
            # Predict every possible pairing
            teams = sorted(model.team_idx_.keys())
            league_fixtures = pd.DataFrame(
                [(code, h, a) for h in teams for a in teams if h != a],
                columns=["league", "home_team", "away_team"],
            )

        for _, row in league_fixtures.iterrows():
            home, away = row["home_team"], row["away_team"]
            if home not in model.team_idx_ or away not in model.team_idx_:
                continue
            probs = model.predict_probs(home, away)

            def _safe(col):
                v = row.get(col)
                return float(v) if v is not None and not pd.isna(v) else None

            bets = scan_match(
                home, away, probs,
                _safe("odds_home"), _safe("odds_draw"), _safe("odds_away"),
                **{k: v for k, v in bet_kwargs.items()
                   if k in ("ev_threshold", "draw_ev_threshold", "min_model_prob",
                            "max_book_overround", "kelly_fraction",
                            "max_bet_fraction", "min_odds", "max_odds")},
            )
            for b in bets:
                b["league"] = code
            all_bets.extend(bets)

    return bets_to_dataframe(all_bets)
