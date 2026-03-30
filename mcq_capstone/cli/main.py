#!/usr/bin/env python3
"""
MCQ-Capstone CLI — production entry point.

Commands
--------
  download         Download all league CSVs from football-data.co.uk
  backtest         Single-league walk-forward backtest
  multi-backtest   Backtest all 10 leagues
  predict          Predict match + optional value scan
  simulate         Show probability evolution over match time
  ratings          Team attack/defense ratings
  calibration      Show calibration report for a league
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Allow running as  python mcq_capstone/cli/main.py
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd

from mcq_capstone.data.loader import load_or_generate
from mcq_capstone.pre_match.dixon_coles import DixonColesModel
from mcq_capstone.features.engineer import build_feature_matrix
from mcq_capstone.models.goal_model import PoissonGoalModel, RidgeGoalModel
from mcq_capstone.models.outcome_model import LogisticOutcomeModel, GradientBoostModel
from mcq_capstone.models.market_analysis import MarketPricingModel
from mcq_capstone.models.evaluator import walk_forward_eval, compare_models, print_comparison
from mcq_capstone.in_play.state import MatchState
from mcq_capstone.in_play.simulator import MonteCarloSimulator
from mcq_capstone.markets.pricer import MarketPricer
from mcq_capstone.execution.rules import (
    default_pre_match_rule, scan_match as rule_scan, Trade,
)
from mcq_capstone.logging.calibration import CalibrationAnalyzer
from mcq_capstone.backtest.engine import BacktestEngine
from mcq_capstone.backtest.multi_league import (
    DOMESTIC_LEAGUES, DEFAULT_SEASONS,
    download_all, load_all, run_multi_league, print_multi_report,
)

# ── Defaults ───────────────────────────────────────────────────────────────────
_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "sample"
_RULE = default_pre_match_rule(
    min_ev=0.10, draw_min_ev=0.15, min_prob=0.40,
    max_overround=0.12, min_odds=1.50, max_odds=5.00,
)
_ENGINE_KWARGS = dict(
    initial_bankroll=1000.0,
    time_decay=0.0065,
    min_matches=60,
    train_days=365,
    min_games_team=5,
)


# ── download ───────────────────────────────────────────────────────────────────

def cmd_download(args):
    print(f"Downloading {len(DOMESTIC_LEAGUES)} leagues × {len(DEFAULT_SEASONS)} seasons …")
    paths = download_all(_DATA_DIR, force=args.force)
    print(f"Done. {len(paths)} files in {_DATA_DIR}")


# ── backtest ───────────────────────────────────────────────────────────────────

def cmd_backtest(args):
    df = load_or_generate(args.data)
    print(f"Loaded {len(df)} matches  "
          f"({df['date'].min().date()} → {df['date'].max().date()})")
    engine = BacktestEngine(rule=_RULE, **_ENGINE_KWARGS)
    bet_df, summary = engine.run(df, league=args.league or "", verbose=True)
    engine.print_report(bet_df, summary, league=args.league or "")

    if args.output and not bet_df.empty:
        bet_df.to_csv(args.output, index=False)
        print(f"Bet log → {args.output}")


# ── multi-backtest ─────────────────────────────────────────────────────────────

def cmd_multi_backtest(args):
    print("Loading league data …")
    league_data = load_all(_DATA_DIR)
    if not league_data:
        print("No data found. Run:  python mcq_capstone/cli/main.py download")
        sys.exit(1)

    print(f"\nRunning backtest on {len(league_data)} leagues …\n")
    combined, metrics = run_multi_league(league_data, rule=_RULE, **_ENGINE_KWARGS)
    print_multi_report(metrics, combined)

    if args.output and not combined.empty:
        combined.to_csv(args.output, index=False)
        print(f"Bet log → {args.output}")


# ── predict ────────────────────────────────────────────────────────────────────

def cmd_predict(args):
    df = load_or_generate(args.data)
    model = DixonColesModel(time_decay=0.0065)
    print(f"Fitting model on {len(df)} matches …")
    model.fit(df)

    home, away = args.home, args.away
    for t in [home, away]:
        if t not in model.team_idx_:
            print(f"Unknown team: '{t}'. Available:")
            for x in sorted(model.team_idx_):
                print(f"  {x}")
            sys.exit(1)

    pricer = MarketPricer(model, MonteCarloSimulator(100_000, seed=0))
    probs = pricer.price_pre_match(home, away)
    lam, mu = probs["lam"], probs["mu"]

    print(f"\n{'─'*56}")
    print(f"  {home}  vs  {away}")
    print(f"{'─'*56}")
    print(f"  Expected goals: {home} {lam:.2f} — {mu:.2f} {away}")
    print(f"\n  {'Outcome':<12}  {'Model P':>9}  {'Fair odds':>10}")
    print(f"  {'─'*36}")
    for label, key in [("Home win","home"),("Draw","draw"),("Away win","away")]:
        p = probs[key]
        print(f"  {label:<12}  {p:>9.1%}  {1/p:>10.3f}")

    # All-markets simulation
    result = pricer.price_pre_match_all_markets(home, away)
    print(f"\n  All-market simulation (N=100k):")
    print(f"  O/U 2.5  over={result.p_over_25:.3f}  under={result.p_under_25:.3f}")
    print(f"  BTTS     yes={result.p_btts:.3f}")
    print(f"  AH −1    home wins by 2+: {result.p_home_ah_minus1:.3f}")

    print("\n  Likely scorelines:")
    sm = model.score_matrix(home, away, max_goals=4)
    flat = sm.stack()
    flat.index = [f"{h}–{a}" for h, a in flat.index]
    for score, p in flat.nlargest(6).items():
        print(f"    {score:>5}  {p:.2%}")

    # EV scan if odds provided
    if any(x is not None for x in [args.odds_home, args.odds_draw, args.odds_away]):
        book = {"home": args.odds_home, "draw": args.odds_draw, "away": args.odds_away}
        trades = rule_scan(home, away, probs, book, _RULE)
        print(f"\n  ── Value scan ───────────────────────────────────────────")
        if trades:
            for t in trades:
                print(f"    BET {t.market:5s}  prob={t.model_prob:.3f}  "
                      f"odds={t.book_odds}  EV={t.ev:+.3f}  Kelly={t.kelly:.4f}")
        else:
            print(f"  No value bets at current thresholds.")

    print(f"{'─'*56}")


# ── simulate ───────────────────────────────────────────────────────────────────

def cmd_simulate(args):
    """Show how probabilities evolve during a match."""
    df = load_or_generate(args.data)
    model = DixonColesModel(time_decay=0.0065)
    model.fit(df)

    home, away = args.home, args.away
    for t in [home, away]:
        if t not in model.team_idx_:
            print(f"Unknown team: '{t}'"); sys.exit(1)

    sim = MonteCarloSimulator(n_simulations=50_000, seed=42)
    pricer = MarketPricer(model, sim)

    # Show probability path for a 0-0 match
    path = pricer.probability_path(home, away, step_minutes=10)

    # Also simulate from a specific state if requested
    state = MatchState(
        minute=args.minute,
        home_goals=args.home_goals,
        away_goals=args.away_goals,
        home_lambda=model.predict_goals(home, away)[0],
        away_lambda=model.predict_goals(home, away)[1],
    )
    result = pricer.price_state(state)

    print(f"\n{'─'*64}")
    print(f"  {home} vs {away} — in-play simulation")
    print(f"  State: t={args.minute}', score {args.home_goals}–{args.away_goals}")
    print(f"{'─'*64}")
    print(result)
    print(f"\n  Probability path (0-0 scoreline):")
    print(f"  {'Min':>4}  {'Home':>7}  {'Draw':>7}  {'Away':>7}  {'O2.5':>7}")
    for row in path:
        print(f"  {row['minute']:>4}'  {row['p_home']:>7.1%}  "
              f"{row['p_draw']:>7.1%}  {row['p_away']:>7.1%}  {row['p_over_25']:>7.1%}")
    print(f"{'─'*64}")


# ── ratings ────────────────────────────────────────────────────────────────────

def cmd_ratings(args):
    df = load_or_generate(args.data)
    model = DixonColesModel(time_decay=0.0065)
    model.fit(df)
    ratings = model.get_ratings()
    print(f"\n  TEAM RATINGS")
    print(ratings.to_string(index=False, float_format="{:.3f}".format))


# ── calibration ────────────────────────────────────────────────────────────────

def cmd_calibration(args):
    df = load_or_generate(args.data)
    print(f"Fitting model and collecting calibration data ({len(df)} matches) …")

    df = df.sort_values("date").reset_index(drop=True)
    cal = CalibrationAnalyzer()
    min_m = 60
    train_days = 365

    for i in range(min_m, len(df)):
        row = df.iloc[i]
        match_date = row["date"]
        win_start = match_date - pd.Timedelta(days=train_days)
        train = df.iloc[:i]
        train = train[train["date"] >= win_start]
        if len(train) < min_m:
            continue
        home, away = str(row["home_team"]), str(row["away_team"])
        known = set(train["home_team"]) | set(train["away_team"])
        if home not in known or away not in known:
            continue
        model = DixonColesModel(time_decay=0.0065)
        try:
            model.fit(train, current_date=match_date)
            probs = model.predict_probs(home, away)
        except Exception:
            continue
        outcome = int(row["home_goals"]) > int(row["away_goals"])
        cal.add(probs["home"], outcome)

    print("\n")
    cal.print_summary()
    print("\n  Reliability table:")
    print(cal.reliability_table().to_string(index=False))
    if args.plot:
        cal.plot(save_path=args.plot)
        print(f"  Plot saved → {args.plot}")


# ── train ──────────────────────────────────────────────────────────────────────

def cmd_train(args):
    """
    Build feature matrix, train all models, compare them, run market analysis.
    This is the core a priori development command.
    """
    df = load_or_generate(args.data)
    print(f"Loaded {len(df)} matches.  Building feature matrix …")
    feat_df = build_feature_matrix(df)
    print(f"Feature matrix: {len(feat_df)} rows × {feat_df.shape[1]} columns "
          f"({feat_df.shape[1] - 8} engineered features)")

    if args.feature_matrix:
        feat_df.to_csv(args.feature_matrix, index=False)
        print(f"Feature matrix saved → {args.feature_matrix}")

    # ── Model comparison ──────────────────────────────────────────────────────
    print("\nRunning walk-forward model comparison …")
    models = {
        "PoissonGLM":   PoissonGoalModel(alpha=1.0),
        "RidgeGoal":    RidgeGoalModel(alpha=1.0),
        "Logistic":     LogisticOutcomeModel(C=1.0),
        "GradientBoost": GradientBoostModel(n_estimators=100),
    }
    eval_results = walk_forward_eval(
        feat_df,
        models,
        min_train=args.min_train,
        train_days=args.train_days,
        step=args.step,
        verbose=True,
    )
    summary = compare_models(eval_results)
    print_comparison(eval_results, summary)

    # ── Feature importance from best linear model ─────────────────────────────
    print("\nFitting RidgeGoal on full dataset for feature importance …")
    ridge = RidgeGoalModel(alpha=1.0)
    ridge.fit(feat_df)
    fi = ridge.feature_importance()
    print("\n  Top 20 features by |coef_home − coef_away|  (most predictive for outcome):")
    print(fi.head(20).to_string(index=False))

    # ── GradientBoost feature importance ─────────────────────────────────────
    print("\nFitting GradientBoost for non-linear feature importance …")
    gb = GradientBoostModel(n_estimators=200)
    gb.fit(feat_df)
    fi_gb = gb.feature_importance()
    print("\n  Top 20 features (GradientBoost gain):")
    print(fi_gb.head(20).to_string(index=False))

    # ── Market pricing analysis ───────────────────────────────────────────────
    has_odds = all(c in feat_df.columns for c in ["odds_home", "odds_draw", "odds_away"])
    if has_odds:
        print("\nFitting market pricing model …")
        mpm = MarketPricingModel()
        mpm.fit(feat_df.dropna(subset=["odds_home", "odds_draw", "odds_away"]))
        mpm.print_report()
    else:
        print("\nNo odds columns — skipping market pricing analysis.")

    if args.output:
        summary.to_csv(args.output, index=False)
        print(f"\nModel comparison saved → {args.output}")


# ── market-analysis ────────────────────────────────────────────────────────────

def cmd_market_analysis(args):
    """
    Deep-dive into what drives bookmaker pricing and where it deviates
    from fundamental team statistics.
    """
    df = load_or_generate(args.data)
    required = ["odds_home", "odds_draw", "odds_away"]
    if not all(c in df.columns for c in required):
        print("ERROR: This command requires odds columns (odds_home, odds_draw, odds_away).")
        print("Use a real data CSV, e.g.:  --data data/sample/E0_2324.csv")
        sys.exit(1)

    print(f"Building feature matrix for {len(df)} matches …")
    feat_df = build_feature_matrix(df)
    feat_df = feat_df.dropna(subset=required)
    print(f"  {len(feat_df)} matches with features + odds")

    mpm = MarketPricingModel()
    mpm.fit(feat_df)
    mpm.print_report()

    if args.output:
        report = mpm.market_inefficiency_report(top_n=50)
        report.to_csv(args.output, index=False)
        print(f"\nMispricing report saved → {args.output}")


# ── parser ─────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(prog="mcq-capstone",
                                description="MCQ-Capstone soccer betting algorithm")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("download", help="Download league CSVs").add_argument(
        "--force", action="store_true")

    bt = sub.add_parser("backtest", help="Single-league backtest")
    bt.add_argument("--data"); bt.add_argument("--league", default="")
    bt.add_argument("--output", default=None)

    mb = sub.add_parser("multi-backtest", help="Backtest all leagues")
    mb.add_argument("--output", default=None)

    pr = sub.add_parser("predict", help="Predict a match")
    pr.add_argument("--home", required=True); pr.add_argument("--away", required=True)
    pr.add_argument("--data", default=None)
    pr.add_argument("--odds-home", type=float); pr.add_argument("--odds-draw", type=float)
    pr.add_argument("--odds-away", type=float)

    sim = sub.add_parser("simulate", help="In-play probability path")
    sim.add_argument("--home", required=True); sim.add_argument("--away", required=True)
    sim.add_argument("--data", default=None)
    sim.add_argument("--minute", type=int, default=0)
    sim.add_argument("--home-goals", type=int, default=0)
    sim.add_argument("--away-goals", type=int, default=0)

    rt = sub.add_parser("ratings", help="Team ratings")
    rt.add_argument("--data", default=None)

    cb = sub.add_parser("calibration", help="Calibration analysis")
    cb.add_argument("--data", default=None)
    cb.add_argument("--plot", default=None, help="Save plot to this path")

    tr = sub.add_parser("train", help="Feature engineering + model comparison + market analysis")
    tr.add_argument("--data", default=None)
    tr.add_argument("--min-train", type=int, default=80)
    tr.add_argument("--train-days", type=int, default=365)
    tr.add_argument("--step", type=int, default=5,
                    help="Evaluate every N matches (1=all, slower)")
    tr.add_argument("--feature-matrix", default=None, help="Save feature matrix CSV")
    tr.add_argument("--output", default=None, help="Save model comparison CSV")

    ma = sub.add_parser("market-analysis", help="Reverse-engineer bookmaker pricing")
    ma.add_argument("--data", default=None)
    ma.add_argument("--output", default=None, help="Save mispricing report CSV")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    {
        "download": cmd_download,
        "backtest": cmd_backtest,
        "multi-backtest": cmd_multi_backtest,
        "predict": cmd_predict,
        "simulate": cmd_simulate,
        "ratings": cmd_ratings,
        "calibration": cmd_calibration,
        "train": cmd_train,
        "market-analysis": cmd_market_analysis,
    }[args.command](args)


if __name__ == "__main__":
    main()
