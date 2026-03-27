#!/usr/bin/env python3
"""
MCQ-Capstone Soccer Betting Algorithm — CLI Entry Point

Usage
-----
# Single-league backtest (Premier League, default synthetic data)
python src/main.py backtest

# Single-league on real CSV
python src/main.py backtest --data data/sample/E0_2324.csv

# Multi-league backtest (all 10 leagues, 3 seasons)
python src/main.py multi-backtest

# Download all league data first
python src/main.py download

# Predict a specific match
python src/main.py predict --home "Arsenal" --away "Chelsea"
python src/main.py predict --home "Arsenal" --away "Chelsea" \\
    --odds-home 2.10 --odds-draw 3.40 --odds-away 3.60

# Show team ratings for a league
python src/main.py ratings --data data/sample/E0_2324.csv

# Scan upcoming fixtures for value bets
python src/main.py scan --fixtures fixtures.csv --league E0
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from config import (
    TIME_DECAY, MIN_EV_THRESHOLD, DRAW_EV_THRESHOLD,
    MIN_MODEL_PROB, MAX_BOOK_OVERROUND,
    KELLY_FRACTION, MAX_BET_FRACTION,
    INITIAL_BANKROLL, MIN_MATCHES_TO_FIT, TRAIN_WINDOW_DAYS,
    MIN_ODDS, MAX_ODDS, DATA_DIR, SAMPLE_DATA_DIR,
    DOMESTIC_LEAGUES, DOWNLOAD_SEASONS,
)
from src.data_loader import load_or_generate
from src.dixon_coles import DixonColesModel
from src.betting import scan_match, bets_to_dataframe
from src.backtest import run_backtest, print_report
from src.league_manager import (
    download_all, load_league_data,
    backtest_all_leagues, print_multi_report,
    scan_all_leagues,
)

_BET_KWARGS = dict(
    ev_threshold=MIN_EV_THRESHOLD,
    draw_ev_threshold=DRAW_EV_THRESHOLD,
    min_model_prob=MIN_MODEL_PROB,
    max_book_overround=MAX_BOOK_OVERROUND,
    kelly_fraction=KELLY_FRACTION,
    max_bet_fraction=MAX_BET_FRACTION,
    min_odds=MIN_ODDS,
    max_odds=MAX_ODDS,
)


# ── download ───────────────────────────────────────────────────────────────────

def cmd_download(args):
    print(f"Downloading {len(DOMESTIC_LEAGUES)} leagues × "
          f"{len(DOWNLOAD_SEASONS)} seasons …")
    paths = download_all(SAMPLE_DATA_DIR, DOMESTIC_LEAGUES, DOWNLOAD_SEASONS,
                         force=args.force)
    print(f"Done. {len(paths)} files available in {SAMPLE_DATA_DIR}")


# ── backtest (single league) ───────────────────────────────────────────────────

def cmd_backtest(args):
    df = load_or_generate(args.data)
    print(f"Loaded {len(df)} matches  "
          f"({df['date'].min().date()} → {df['date'].max().date()})")

    bet_df, metrics = run_backtest(
        df,
        initial_bankroll=INITIAL_BANKROLL,
        time_decay=TIME_DECAY,
        min_matches_to_fit=MIN_MATCHES_TO_FIT,
        train_window_days=TRAIN_WINDOW_DAYS,
        verbose=True,
        **{k: v for k, v in _BET_KWARGS.items()},
    )
    print_report(metrics, bet_df)

    if args.output and not bet_df.empty:
        bet_df.to_csv(args.output, index=False)
        print(f"\nBet log → {args.output}")


# ── multi-backtest ─────────────────────────────────────────────────────────────

def cmd_multi_backtest(args):
    print("Loading league data …")
    league_data = load_league_data(SAMPLE_DATA_DIR, DOMESTIC_LEAGUES, DOWNLOAD_SEASONS)

    if not league_data:
        print("No data found. Run:  python src/main.py download")
        sys.exit(1)

    print(f"\nRunning walk-forward backtest on {len(league_data)} leagues …\n")
    combined_bet_df, metrics = backtest_all_leagues(
        league_data,
        initial_bankroll=INITIAL_BANKROLL,
        time_decay=TIME_DECAY,
        min_matches_to_fit=MIN_MATCHES_TO_FIT,
        train_window_days=TRAIN_WINDOW_DAYS,
        verbose_per_league=False,
        **{k: v for k, v in _BET_KWARGS.items()},
    )
    print_multi_report(metrics, combined_bet_df)

    if args.output and not combined_bet_df.empty:
        combined_bet_df.to_csv(args.output, index=False)
        print(f"\nBet log → {args.output}")


# ── predict ────────────────────────────────────────────────────────────────────

def cmd_predict(args):
    df = load_or_generate(args.data)
    model = DixonColesModel(time_decay=TIME_DECAY)
    print(f"Fitting model on {len(df)} matches …")
    model.fit(df)

    home, away = args.home, args.away
    for team in [home, away]:
        if team not in model.team_idx_:
            print(f"ERROR: '{team}' not in training data. Known teams:")
            for t in sorted(model.team_idx_):
                print(f"  {t}")
            sys.exit(1)

    probs = model.predict_probs(home, away)
    lam, mu = probs["lam"], probs["mu"]

    print(f"\n{'─'*52}")
    print(f"  {home}  vs  {away}")
    print(f"{'─'*52}")
    print(f"  Expected goals : {home} {lam:.2f}  —  {mu:.2f} {away}")
    print(f"\n  {'Outcome':<12}  {'Model prob':>11}  {'Fair odds':>10}")
    print(f"  {'─'*38}")
    for label, key in [("Home win", "home"), ("Draw", "draw"), ("Away win", "away")]:
        p = probs[key]
        print(f"  {label:<12}  {p:>10.1%}  {1/p:>10.3f}")

    print("\n  Most likely scorelines:")
    sm = model.score_matrix(home, away, max_goals=4)
    flat = sm.stack()
    flat.index = [f"{h}–{a}" for h, a in flat.index]
    for score, p in flat.nlargest(6).items():
        print(f"    {score:>5}  {p:.2%}")

    if any(x is not None for x in [args.odds_home, args.odds_draw, args.odds_away]):
        print(f"\n  ── Value bet scan ───────────────────────────────────────")
        bets = scan_match(
            home, away, probs,
            args.odds_home, args.odds_draw, args.odds_away,
            **_BET_KWARGS,
        )
        if bets:
            print(bets_to_dataframe(bets).to_string(index=False))
        else:
            print(f"  No value bets found at current thresholds.")
    print(f"{'─'*52}")


# ── ratings ────────────────────────────────────────────────────────────────────

def cmd_ratings(args):
    df = load_or_generate(args.data)
    model = DixonColesModel(time_decay=TIME_DECAY)
    print(f"Fitting model on {len(df)} matches …")
    model.fit(df)
    ratings = model.get_ratings()
    print(f"\n{'─'*62}")
    print(f"  TEAM RATINGS")
    print(f"{'─'*62}")
    print(ratings.to_string(index=False, float_format="{:.3f}".format))
    print(f"{'─'*62}")


# ── scan ───────────────────────────────────────────────────────────────────────

def cmd_scan(args):
    if args.league:
        df = load_or_generate(args.data)
        model = DixonColesModel(time_decay=TIME_DECAY)
        model.fit(df)
        league_data = {args.league: df}
    else:
        print("Loading all leagues …")
        league_data = load_league_data(SAMPLE_DATA_DIR, DOMESTIC_LEAGUES, DOWNLOAD_SEASONS)

    fixtures = None
    if args.fixtures:
        fixtures = pd.read_csv(args.fixtures)
        if "league" not in fixtures.columns and args.league:
            fixtures["league"] = args.league

    result = scan_all_leagues(league_data, fixtures_df=fixtures, **_BET_KWARGS,
                              time_decay=TIME_DECAY)

    if result.empty:
        print("No value bets found at current thresholds.")
    else:
        print("\nVALUE BETS:")
        print(result.to_string(index=False))


# ── parser ─────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mcq-capstone",
        description="Soccer betting algorithm — Dixon-Coles model",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # download
    dl = sub.add_parser("download", help="Download all league CSVs")
    dl.add_argument("--force", action="store_true", help="Re-download existing files")

    # backtest
    bt = sub.add_parser("backtest", help="Single-league walk-forward backtest")
    bt.add_argument("--data", default=None)
    bt.add_argument("--output", default=None, help="Save bet log CSV")

    # multi-backtest
    mb = sub.add_parser("multi-backtest", help="Backtest all 10 leagues")
    mb.add_argument("--output", default=None, help="Save combined bet log CSV")

    # predict
    pr = sub.add_parser("predict", help="Predict a match")
    pr.add_argument("--home", required=True)
    pr.add_argument("--away", required=True)
    pr.add_argument("--data", default=None)
    pr.add_argument("--odds-home", type=float, default=None)
    pr.add_argument("--odds-draw", type=float, default=None)
    pr.add_argument("--odds-away", type=float, default=None)

    # ratings
    rt = sub.add_parser("ratings", help="Show team ratings")
    rt.add_argument("--data", default=None)

    # scan
    sc = sub.add_parser("scan", help="Scan for value bets")
    sc.add_argument("--data", default=None)
    sc.add_argument("--league", default=None, help="League code (e.g. E0)")
    sc.add_argument("--fixtures", default=None,
                    help="CSV: league,home_team,away_team,odds_home,odds_draw,odds_away")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    {
        "download": cmd_download,
        "backtest": cmd_backtest,
        "multi-backtest": cmd_multi_backtest,
        "predict": cmd_predict,
        "ratings": cmd_ratings,
        "scan": cmd_scan,
    }[args.command](args)


if __name__ == "__main__":
    main()
