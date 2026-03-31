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
  fbref-download   Download FBRef match data (requires sports-reference subscription)
  fbref-team       Download a specific team's full match log from FBRef
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
from mcq_capstone.models.extended_models import ElasticNetGoalModel, NegBinGoalModel, StackingEnsemble
from mcq_capstone.models.market_analysis import MarketPricingModel
from mcq_capstone.models.evaluator import walk_forward_eval, compare_models, print_comparison
from mcq_capstone.models.splitter import TemporalSplitter
from mcq_capstone.models.leakage_audit import audit_features
from mcq_capstone.models.tuner import HyperparameterTuner, tune_all_models, PARAM_GRIDS
from mcq_capstone.pipeline import DeployablePipeline, MultiLeaguePipeline
from mcq_capstone.data.fbref import (
    FBRefScraper, COMPETITIONS, FDCO_TO_FBREF,
    download_fbref_seasons, combine_fbref_data, build_xg_features,
)
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


# ── tune ───────────────────────────────────────────────────────────────────────

def cmd_tune(args):
    """
    Hyperparameter tuning using TimeSeriesSplit CV on train+val only.
    Reports best params per model family and saves a summary CSV.
    """
    df = load_or_generate(args.data)
    print(f"Loaded {len(df)} matches. Building feature matrix …")
    feat_df = build_feature_matrix(df)
    print(f"Feature matrix: {len(feat_df)} rows × {feat_df.shape[1]} columns")

    # Leakage audit first
    print("\nRunning leakage audit …")
    audit_features(feat_df, verbose=True)

    # Temporal split — tuning only on train+val
    splitter = TemporalSplitter(test_months=4, val_months=3)
    split = splitter.split(feat_df)
    splitter.print_split_info(feat_df)

    import numpy as np
    train_val_df = feat_df.iloc[
        np.concatenate([split.train_idx, split.val_idx])
    ].reset_index(drop=True)

    print(f"\nTuning on {len(train_val_df)} rows (train+val). "
          f"Test set ({len(split.test_idx)} rows) is LOCKED.")

    # Select which models to tune
    model_map = {
        "logistic":   (LogisticOutcomeModel, None),
        "gbm":        (GradientBoostModel,   None),
        "poisson":    (PoissonGoalModel,      None),
        "ridge":      (RidgeGoalModel,        None),
        "elasticnet": (ElasticNetGoalModel,   None),
        "negbin":     (NegBinGoalModel,       None),
    }
    if args.models:
        selected = [(model_map[m][0], model_map[m][1])
                    for m in args.models.split(",") if m in model_map]
    else:
        # Default: fast models only unless --all
        if args.all_models:
            selected = list(model_map.values())
        else:
            selected = [
                (LogisticOutcomeModel, None),
                (PoissonGoalModel,     None),
                (RidgeGoalModel,       None),
                (ElasticNetGoalModel,  None),
            ]

    tuner = HyperparameterTuner(n_cv_folds=args.cv_folds, verbose=True)
    all_results = {}
    for cls, grid in selected:
        try:
            res = tuner.tune(cls, train_val_df, param_grid=grid)
            all_results[cls.__name__] = res
        except Exception as e:
            print(f"  ERROR tuning {cls.__name__}: {e}")

    # Summary table
    print(f"\n{'='*70}")
    print("  TUNING SUMMARY")
    print(f"{'='*70}")
    rows = []
    for name, res in all_results.items():
        rows.append({
            "model":       name,
            "best_brier":  res.best_brier,
            "best_logloss": res.best_log_loss,
            "best_params": str(res.best_params),
        })
    if rows:
        summary_df = pd.DataFrame(rows).sort_values("best_brier")
        print(summary_df.to_string(index=False))

    if args.output and rows:
        summary_df.to_csv(args.output, index=False)
        print(f"\nTuning summary saved → {args.output}")


# ── build-pipeline ─────────────────────────────────────────────────────────────

def cmd_build_pipeline(args):
    """
    Build, validate, and serialise a deployable prediction pipeline.

    Steps:
      1. Load data + engineer features
      2. Leakage audit
      3. Temporal train/val/test split
      4. (Optionally) tune hyperparameters on train+val
      5. Fit best model on train+val
      6. Final evaluation on locked test set
      7. Save pipeline to disk
    """
    df = load_or_generate(args.data)
    print(f"Loaded {len(df)} matches.")

    tune_params = None
    if args.tune:
        print("\nRunning hyperparameter tuning first …")
        feat_df = build_feature_matrix(df)
        splitter = TemporalSplitter(test_months=4, val_months=3)
        split = splitter.split(feat_df)
        import numpy as np
        train_val_df = feat_df.iloc[
            np.concatenate([split.train_idx, split.val_idx])
        ].reset_index(drop=True)

        model_cls_map = {
            "GradientBoostModel":   GradientBoostModel,
            "LogisticOutcomeModel": LogisticOutcomeModel,
            "PoissonGoalModel":     PoissonGoalModel,
            "RidgeGoalModel":       RidgeGoalModel,
            "ElasticNetGoalModel":  ElasticNetGoalModel,
            "NegBinGoalModel":      NegBinGoalModel,
        }
        model_cls = model_cls_map.get(args.model, GradientBoostModel)
        tuner = HyperparameterTuner(n_cv_folds=5, verbose=True)
        try:
            res = tuner.tune(model_cls, train_val_df)
            tune_params = res.best_params
            print(f"\nBest params: {tune_params}")
        except Exception as e:
            print(f"Tuning failed ({e}), using defaults.")

    pipeline = DeployablePipeline.build_and_fit(
        raw_df=df,
        league=args.league or "unknown",
        model_type=args.model,
        model_params=tune_params,
        test_holdout_months=4,
        verbose=True,
    )

    save_path = args.output or f"models/{args.league or 'model'}_pipeline.pkl"
    pipeline.save(save_path)
    print(pipeline.summary())


# ── deploy ─────────────────────────────────────────────────────────────────────

def cmd_deploy(args):
    """
    Load a saved pipeline and predict bets for upcoming fixtures.
    """
    pipeline = DeployablePipeline.load(args.pipeline)

    if args.fixtures:
        fixtures_df = pd.read_csv(args.fixtures, parse_dates=["date"])
    else:
        # Fall back to loading historical data (pipeline will use last matches
        # as context and filter to upcoming)
        fixtures_df = load_or_generate(args.data)

    bets = pipeline.predict_bets(
        fixtures_df,
        bankroll=args.bankroll,
        verbose=True,
    )

    if args.output and bets:
        import dataclasses
        bet_rows = [dataclasses.asdict(b) for b in bets]
        pd.DataFrame(bet_rows).to_csv(args.output, index=False)
        print(f"Bet recommendations saved → {args.output}")


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


# ── fbref-download ─────────────────────────────────────────────────────────────

def cmd_fbref_download(args):
    """
    Download FBRef match schedules for multiple leagues and seasons.

    With your sports-reference.com subscription:
      - Pass --username and --password to get subscriber rate limits (~2s/req)
      - Results cached to --cache-dir to avoid re-downloading

    Example
    -------
    python -m mcq_capstone.cli.main fbref-download \\
        --leagues ENG-PL ESP-LL GER-BL ITA-SA FRA-L1 NED-ED POR-PL TUR-SL \\
                  UEFA-CL UEFA-EL UEFA-ECL \\
        --seasons 2022-23 2023-24 2024-25 \\
        --username your@email.com --password yourpass \\
        --output data/fbref/all_leagues.csv
    """
    leagues  = args.leagues or list(COMPETITIONS.keys())
    seasons  = args.seasons or ["2022-23", "2023-24", "2024-25"]
    delay    = 2.0 if (args.username and args.password) else 4.0

    print(f"Downloading FBRef data:")
    print(f"  Leagues  : {', '.join(leagues)}")
    print(f"  Seasons  : {', '.join(seasons)}")
    print(f"  Delay    : {delay}s per request")
    print(f"  Cache dir: {args.cache_dir}")

    data = download_fbref_seasons(
        leagues=leagues,
        seasons=seasons,
        cache_dir=args.cache_dir,
        delay=delay,
        verbose=True,
    )

    combined = combine_fbref_data(data, include_upcoming=args.include_upcoming)
    print(f"\nCombined: {len(combined)} completed matches across "
          f"{combined['league'].nunique() if 'league' in combined.columns else '?'} leagues")

    if "xg_home" in combined.columns:
        xg_avail = combined["xg_home"].notna().mean()
        print(f"xG data available: {xg_avail:.0%} of matches")

    if args.output:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        combined.to_csv(args.output, index=False)
        print(f"Saved → {args.output}")
    else:
        print("\nSample output:")
        print(combined.head(10).to_string())

    # Per-league summary
    if "league" in combined.columns:
        print("\nPer-league match counts:")
        summary = (
            combined.groupby("league")
            .agg(
                matches=("date", "count"),
                date_from=("date", "min"),
                date_to=("date", "max"),
            )
            .sort_values("matches", ascending=False)
        )
        print(summary.to_string())


def cmd_fbref_team(args):
    """
    Download per-match stats for a specific team from FBRef.

    The team_id is found in any FBRef team URL, e.g.:
      fbref.com/en/squads/ecd11ca2/Galatasaray-Stats
      → team_id = ecd11ca2

    Example
    -------
    python -m mcq_capstone.cli.main fbref-team \\
        --team-id ecd11ca2 --team-name Galatasaray \\
        --season 2024-25 \\
        --username your@email.com --password yourpass \\
        --output data/fbref/galatasaray_2024.csv
    """
    delay = 2.0 if (args.username and args.password) else 4.0
    scraper = FBRefScraper(
        delay=delay,
        cache_dir=args.cache_dir,
        sr_username=args.username,
        sr_password=args.password,
    )

    seasons = args.seasons or [args.season or "2024-25"]
    all_logs = []

    for season in seasons:
        try:
            print(f"  Fetching {args.team_name} ({args.team_id}) — {season} … ", end="", flush=True)
            df = scraper.load_team_match_log(
                team_id=args.team_id,
                team_name=args.team_name,
                season=season,
                comp_filter=args.comp_filter,
            )
            df["season"] = season
            all_logs.append(df)
            print(f"OK ({len(df)} matches)")
        except Exception as e:
            print(f"SKIP ({e})")

    if not all_logs:
        print("No data retrieved.")
        return

    combined = pd.concat(all_logs, ignore_index=True)
    print(f"\nTotal: {len(combined)} match records")

    cols_to_show = [c for c in ["date", "competition", "venue", "opponent",
                                 "result", "goals_for", "goals_against",
                                 "xg_for", "xg_against", "possession",
                                 "shots", "shots_on_target"] if c in combined.columns]
    print(combined[cols_to_show].head(10).to_string(index=False))

    if args.output:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        combined.to_csv(args.output, index=False)
        print(f"\nSaved → {args.output}")


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

    tn = sub.add_parser("tune", help="Hyperparameter tuning with TimeSeriesSplit CV")
    tn.add_argument("--data", default=None)
    tn.add_argument("--models", default=None,
                    help="Comma-separated: logistic,gbm,poisson,ridge,elasticnet,negbin")
    tn.add_argument("--all-models", action="store_true",
                    help="Tune all model families (slow)")
    tn.add_argument("--cv-folds", type=int, default=5)
    tn.add_argument("--output", default=None, help="Save tuning summary CSV")

    bp = sub.add_parser("build-pipeline",
                        help="Build + validate + serialise a deployable pipeline")
    bp.add_argument("--data", default=None)
    bp.add_argument("--league", default="E0")
    bp.add_argument("--model", default="GradientBoostModel",
                    choices=["GradientBoostModel", "LogisticOutcomeModel",
                             "PoissonGoalModel", "RidgeGoalModel",
                             "ElasticNetGoalModel", "NegBinGoalModel",
                             "StackingEnsemble"])
    bp.add_argument("--tune", action="store_true",
                    help="Run hyperparameter tuning before fitting")
    bp.add_argument("--output", default=None, help="Path to save .pkl pipeline")

    dp = sub.add_parser("deploy", help="Load saved pipeline + predict bets")
    dp.add_argument("--pipeline", required=True, help="Path to .pkl pipeline file")
    dp.add_argument("--fixtures", default=None,
                    help="CSV of upcoming fixtures with odds columns")
    dp.add_argument("--data", default=None,
                    help="Historical data CSV (used if --fixtures not provided)")
    dp.add_argument("--bankroll", type=float, default=1000.0)
    dp.add_argument("--output", default=None, help="Save bet recommendations CSV")

    fd = sub.add_parser(
        "fbref-download",
        help="Download FBRef match data for multiple leagues/seasons",
    )
    fd.add_argument(
        "--leagues", nargs="+", default=None,
        metavar="LEAGUE",
        help=(
            "League codes: ENG-PL ESP-LL GER-BL ITA-SA FRA-L1 NED-ED POR-PL "
            "TUR-SL BEL-PD SCO-PL UEFA-CL UEFA-EL UEFA-ECL"
        ),
    )
    fd.add_argument(
        "--seasons", nargs="+", default=None,
        metavar="SEASON",
        help="Season strings e.g. 2022-23 2023-24 2024-25",
    )
    fd.add_argument("--username", default=None, help="sports-reference.com email")
    fd.add_argument("--password", default=None, help="sports-reference.com password")
    fd.add_argument("--cache-dir", default="data/fbref_cache",
                    help="Directory to cache downloaded HTML (default: data/fbref_cache)")
    fd.add_argument("--output", default=None, help="Save combined CSV to this path")
    fd.add_argument("--include-upcoming", action="store_true",
                    help="Include upcoming fixtures (no score yet)")

    ft = sub.add_parser(
        "fbref-team",
        help="Download full match log for a specific team from FBRef",
    )
    ft.add_argument("--team-id",   required=True,
                    help="FBRef team ID from URL (e.g. ecd11ca2 for Galatasaray)")
    ft.add_argument("--team-name", required=True,
                    help="Team name (e.g. 'Galatasaray')")
    ft.add_argument("--season",    default="2024-25")
    ft.add_argument("--seasons",   nargs="+", default=None,
                    help="Multiple seasons (overrides --season)")
    ft.add_argument("--comp-filter", default=None,
                    help="Filter to competition name substring (e.g. 'Champions')")
    ft.add_argument("--username",  default=None)
    ft.add_argument("--password",  default=None)
    ft.add_argument("--cache-dir", default="data/fbref_cache")
    ft.add_argument("--output",    default=None)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    {
        "download":        cmd_download,
        "backtest":        cmd_backtest,
        "multi-backtest":  cmd_multi_backtest,
        "predict":         cmd_predict,
        "simulate":        cmd_simulate,
        "ratings":         cmd_ratings,
        "calibration":     cmd_calibration,
        "train":           cmd_train,
        "market-analysis": cmd_market_analysis,
        "tune":            cmd_tune,
        "build-pipeline":  cmd_build_pipeline,
        "deploy":          cmd_deploy,
        "fbref-download":  cmd_fbref_download,
        "fbref-team":      cmd_fbref_team,
    }[args.command](args)


if __name__ == "__main__":
    main()
