#!/usr/bin/env python3
"""
MCQ-Capstone Soccer Betting Algorithm — CLI Entry Point

Usage
-----
# Run a full walk-forward backtest on synthetic data (default)
python src/main.py backtest

# Backtest on your own CSV (football-data.co.uk format)
python src/main.py backtest --data path/to/matches.csv

# Predict a specific match
python src/main.py predict --home "Arsenal" --away "Chelsea"
python src/main.py predict --home "Arsenal" --away "Chelsea" --odds-home 2.10 --odds-draw 3.40 --odds-away 3.60

# Show team ratings
python src/main.py ratings

# Scan for today's value bets from a fixtures CSV
python src/main.py scan --fixtures fixtures.csv --data historical.csv
"""

import argparse
import sys
from pathlib import Path

# Make sure the project root is on the path when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config import (
    TIME_DECAY, MIN_EV_THRESHOLD, KELLY_FRACTION,
    MAX_BET_FRACTION, INITIAL_BANKROLL, MIN_MATCHES_TO_FIT,
    TRAIN_WINDOW_DAYS, MIN_ODDS, MAX_ODDS, SAMPLE_DATA_DIR,
)
from src.data_loader import load_or_generate
from src.dixon_coles import DixonColesModel
from src.betting import scan_match, bets_to_dataframe
from src.backtest import run_backtest, print_report


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _load_data(data_path: str | None) -> pd.DataFrame:
    return load_or_generate(data_path)


def _fit_model(df: pd.DataFrame, current_date=None) -> DixonColesModel:
    model = DixonColesModel(time_decay=TIME_DECAY)
    print(f"Fitting Dixon-Coles model on {len(df)} matches …")
    model.fit(df, current_date=current_date)
    print(f"  Home advantage : {model.params_[-2]:.3f}")
    print(f"  Rho (DC corr)  : {model.params_[-1]:.3f}")
    return model


# ── Sub-commands ───────────────────────────────────────────────────────────────

def cmd_backtest(args):
    df = _load_data(args.data)
    print(f"Loaded {len(df)} matches  ({df['date'].min().date()} → {df['date'].max().date()})")

    bet_df, metrics = run_backtest(
        df,
        initial_bankroll=INITIAL_BANKROLL,
        ev_threshold=args.ev_threshold,
        kelly_fraction=KELLY_FRACTION,
        max_bet_fraction=MAX_BET_FRACTION,
        min_odds=MIN_ODDS,
        max_odds=MAX_ODDS,
        time_decay=TIME_DECAY,
        min_matches_to_fit=MIN_MATCHES_TO_FIT,
        train_window_days=TRAIN_WINDOW_DAYS,
        verbose=True,
    )

    print_report(metrics, bet_df)

    if args.output and not bet_df.empty:
        bet_df.to_csv(args.output, index=False)
        print(f"\nBet log saved to {args.output}")


def cmd_predict(args):
    df = _load_data(args.data)
    model = _fit_model(df)

    home, away = args.home, args.away
    if home not in model.team_idx_:
        print(f"ERROR: '{home}' not found. Known teams:\n  "
              + "\n  ".join(sorted(model.team_idx_.keys())))
        sys.exit(1)
    if away not in model.team_idx_:
        print(f"ERROR: '{away}' not found. Known teams:\n  "
              + "\n  ".join(sorted(model.team_idx_.keys())))
        sys.exit(1)

    probs = model.predict_probs(home, away)
    lam, mu = probs["lam"], probs["mu"]

    print(f"\n{'─'*50}")
    print(f"  {home}  vs  {away}")
    print(f"{'─'*50}")
    print(f"  Expected goals : {home} {lam:.2f} — {mu:.2f} {away}")
    print(f"  {'Outcome':<10}  {'Model prob':>11}  {'Fair odds':>10}")
    print(f"  {'-'*38}")
    for outcome, key in [("Home win", "home"), ("Draw", "draw"), ("Away win", "away")]:
        p = probs[key]
        fair = 1 / p if p > 0 else float("inf")
        print(f"  {outcome:<10}  {p:>10.1%}  {fair:>10.3f}")

    # Score matrix
    print("\n  Most likely scorelines:")
    sm = model.score_matrix(home, away, max_goals=4)
    flat = sm.stack()
    flat.index = [f"{h}–{a}" for h, a in flat.index]
    top5 = flat.nlargest(6)
    for score, p in top5.items():
        print(f"    {score:>5}  {p:.2%}")

    # EV check if odds provided
    if any(x is not None for x in [args.odds_home, args.odds_draw, args.odds_away]):
        print(f"\n  ── Value bet scan ───────────────────────────────────")
        bets = scan_match(
            home, away, probs,
            args.odds_home, args.odds_draw, args.odds_away,
            ev_threshold=MIN_EV_THRESHOLD,
            kelly_fraction=KELLY_FRACTION,
            max_bet_fraction=MAX_BET_FRACTION,
            min_odds=MIN_ODDS,
            max_odds=MAX_ODDS,
        )
        if bets:
            print(bets_to_dataframe(bets).to_string(index=False))
        else:
            print(f"  No +EV bets found at EV threshold {MIN_EV_THRESHOLD:.0%}")

    print(f"{'─'*50}")


def cmd_ratings(args):
    df = _load_data(args.data)
    model = _fit_model(df)
    ratings = model.get_ratings()
    print(f"\n{'─'*60}")
    print(f"  TEAM RATINGS  (attack: higher = more goals scored)")
    print(f"  (defense: lower = fewer goals conceded)")
    print(f"{'─'*60}")
    print(ratings.to_string(index=False, float_format="{:.3f}".format))
    print(f"{'─'*60}")


def cmd_scan(args):
    """Scan a fixtures file for value bets using a model trained on historical data."""
    df = _load_data(args.data)
    model = _fit_model(df)

    if args.fixtures:
        fixtures = pd.read_csv(args.fixtures)
    else:
        # Default: generate upcoming fixtures from last known teams
        print("No fixtures file provided — showing predictions for all team combinations.")
        teams = sorted(model.team_idx_.keys())
        pairs = [(h, a) for h in teams for a in teams if h != a]
        fixtures = pd.DataFrame(pairs, columns=["home_team", "away_team"])
        for col in ["odds_home", "odds_draw", "odds_away"]:
            fixtures[col] = np.nan

    all_bets = []
    for _, row in fixtures.iterrows():
        home, away = row["home_team"], row["away_team"]
        if home not in model.team_idx_ or away not in model.team_idx_:
            continue
        probs = model.predict_probs(home, away)
        oh = row.get("odds_home", None)
        od = row.get("odds_draw", None)
        oa = row.get("odds_away", None)
        bets = scan_match(
            home, away, probs,
            (float(oh) if oh and not pd.isna(oh) else None),
            (float(od) if od and not pd.isna(od) else None),
            (float(oa) if oa and not pd.isna(oa) else None),
            ev_threshold=MIN_EV_THRESHOLD,
            kelly_fraction=KELLY_FRACTION,
            max_bet_fraction=MAX_BET_FRACTION,
            min_odds=MIN_ODDS,
            max_odds=MAX_ODDS,
        )
        all_bets.extend(bets)

    if all_bets:
        out = bets_to_dataframe(all_bets)
        print("\n  VALUE BETS FOUND")
        print(out.to_string(index=False))
    else:
        print("No value bets found with current thresholds.")


# ── Argument parsing ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mcq-capstone",
        description="Soccer betting algorithm using Dixon-Coles model",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ── backtest ──────────────────────────────────────────────────────────────
    bt = sub.add_parser("backtest", help="Run walk-forward backtest")
    bt.add_argument("--data", default=None,
                    help="Path to CSV (default: use synthetic data)")
    bt.add_argument("--ev-threshold", type=float, default=MIN_EV_THRESHOLD,
                    help=f"Minimum EV to bet (default {MIN_EV_THRESHOLD})")
    bt.add_argument("--output", default=None,
                    help="Save bet log to this CSV path")

    # ── predict ───────────────────────────────────────────────────────────────
    pr = sub.add_parser("predict", help="Predict a specific match")
    pr.add_argument("--home", required=True, help="Home team name")
    pr.add_argument("--away", required=True, help="Away team name")
    pr.add_argument("--data", default=None, help="Historical CSV path")
    pr.add_argument("--odds-home", type=float, default=None)
    pr.add_argument("--odds-draw", type=float, default=None)
    pr.add_argument("--odds-away", type=float, default=None)

    # ── ratings ───────────────────────────────────────────────────────────────
    rt = sub.add_parser("ratings", help="Show team attack/defense ratings")
    rt.add_argument("--data", default=None, help="Historical CSV path")

    # ── scan ──────────────────────────────────────────────────────────────────
    sc = sub.add_parser("scan", help="Scan fixtures for value bets")
    sc.add_argument("--data", default=None, help="Historical CSV path")
    sc.add_argument("--fixtures", default=None,
                    help="CSV with upcoming fixtures (home_team, away_team, "
                         "odds_home, odds_draw, odds_away)")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "backtest": cmd_backtest,
        "predict": cmd_predict,
        "ratings": cmd_ratings,
        "scan": cmd_scan,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
