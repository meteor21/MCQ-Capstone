"""
Deployable inference pipeline.

This module provides end-to-end prediction for new fixtures:
  1. Load fitted model from disk (joblib serialisation)
  2. Compute features from raw match history
  3. Predict outcome probabilities
  4. Apply betting rules (EV threshold, overround, Kelly)
  5. Output recommended bets

Usage
-----
# --- Training / serialisation (run once) ---
from mcq_capstone.pipeline import DeployablePipeline

pipeline = DeployablePipeline.build_and_fit(
    raw_df=full_match_history_df,
    league="E0",
    model_type="GradientBoostModel",   # or "PoissonGoalModel" etc
    model_params={},                   # override defaults
)
pipeline.save("models/pl_model_2024.pkl")

# --- Inference (run daily) ---
pipeline = DeployablePipeline.load("models/pl_model_2024.pkl")
bets = pipeline.predict_bets(
    new_fixtures_df=upcoming_matches,   # needs home_team, away_team, date, odds_*
    bankroll=10_000,
    verbose=True,
)
print(bets)
"""

from __future__ import annotations
import os
import copy
import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Optional

from .features.engineer import build_feature_matrix


# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_MODEL_PARAMS = {
    "GradientBoostModel":    {"n_estimators": 200, "max_depth": 4,
                              "learning_rate": 0.05, "subsample": 0.8},
    "PoissonGoalModel":      {"alpha": 1.0},
    "RidgeGoalModel":        {"alpha": 10.0},
    "LogisticOutcomeModel":  {"C": 1.0},
    "ElasticNetGoalModel":   {"alpha": 1.0, "l1_ratio": 0.5},
    "NegBinGoalModel":       {"alpha": 1.0},
    "StackingEnsemble":      {},
}


def _get_model_class(name: str):
    from .models.outcome_model import LogisticOutcomeModel, GradientBoostModel
    from .models.goal_model import PoissonGoalModel, RidgeGoalModel
    from .models.extended_models import (
        ElasticNetGoalModel, NegBinGoalModel, StackingEnsemble
    )
    registry = {
        "LogisticOutcomeModel": LogisticOutcomeModel,
        "GradientBoostModel":   GradientBoostModel,
        "PoissonGoalModel":     PoissonGoalModel,
        "RidgeGoalModel":       RidgeGoalModel,
        "ElasticNetGoalModel":  ElasticNetGoalModel,
        "NegBinGoalModel":      NegBinGoalModel,
        "StackingEnsemble":     StackingEnsemble,
    }
    if name not in registry:
        raise ValueError(
            f"Unknown model type '{name}'. "
            f"Choose from: {list(registry.keys())}"
        )
    return registry[name]


# ── Bet output ─────────────────────────────────────────────────────────────────

@dataclass
class BetRecommendation:
    date: Any
    home_team: str
    away_team: str
    market: str         # "home" | "draw" | "away"
    model_prob: float
    book_odds: float
    ev: float
    kelly_stake: float   # as fraction of bankroll
    stake_amount: float  # in currency units (bankroll × kelly_stake)
    expected_profit: float

    def __str__(self):
        return (
            f"{self.date} | {self.home_team:>20} vs {self.away_team:<20} | "
            f"{self.market:>4} | odds={self.book_odds:.2f} | "
            f"p={self.model_prob:.3f} | EV={self.ev:+.3f} | "
            f"stake={self.stake_amount:.0f}"
        )


# ── Core pipeline ──────────────────────────────────────────────────────────────

class DeployablePipeline:
    """
    Serialisable end-to-end prediction pipeline.

    Attributes
    ----------
    model_type     : Name of the model class (for reconstruction).
    model          : Fitted model instance.
    train_cutoff   : Date up to which training data was used.
    league         : League code (e.g. "E0" for Premier League).
    feature_params : kwargs forwarded to build_feature_matrix.
    bet_params     : EV threshold, Kelly fraction, overround limit.
    """

    # Betting defaults
    MIN_EV:          float = 0.08
    MIN_PROB:        float = 0.38
    MAX_OVERROUND:   float = 0.12
    KELLY_FRACTION:  float = 0.25
    MAX_BET_FRAC:    float = 0.04
    MIN_ODDS:        float = 1.40
    MAX_ODDS:        float = 6.00
    DRAW_MIN_EV:     float = 0.13

    def __init__(
        self,
        model_type: str = "GradientBoostModel",
        model_params: dict | None = None,
        feature_params: dict | None = None,
        bet_params: dict | None = None,
    ):
        self.model_type = model_type
        self.model_params = model_params or DEFAULT_MODEL_PARAMS.get(model_type, {})
        self.feature_params = feature_params or {}
        self.bet_params = bet_params or {}

        # Update defaults with user overrides
        for k, v in self.bet_params.items():
            setattr(self, k.upper(), v)

        self.model = None
        self.train_cutoff: Optional[pd.Timestamp] = None
        self.league: str = "unknown"
        self.metadata: dict = {}

    @classmethod
    def build_and_fit(
        cls,
        raw_df: pd.DataFrame,
        league: str = "E0",
        model_type: str = "GradientBoostModel",
        model_params: dict | None = None,
        feature_params: dict | None = None,
        bet_params: dict | None = None,
        test_holdout_months: int = 4,
        verbose: bool = True,
    ) -> "DeployablePipeline":
        """
        Build features, train/val/test split, fit model on train+val.

        The test set is evaluated and the results stored in pipeline.metadata
        but the pipeline is fitted on ALL data except the last `test_holdout_months`.

        Parameters
        ----------
        raw_df              : Full match history DataFrame (must have date,
                              home_team, away_team, home_goals, away_goals).
        league              : League identifier (informational only).
        model_type          : One of the keys in DEFAULT_MODEL_PARAMS.
        test_holdout_months : Months to reserve for final evaluation.
        verbose             : Print progress.
        """
        from .models.splitter import TemporalSplitter
        from .models.leakage_audit import audit_features
        from .models.evaluator import compare_models

        if verbose:
            print(f"\n{'='*60}")
            print(f"  Building DeployablePipeline [{model_type}] | {league}")
            print(f"{'='*60}")

        # Step 1: Feature engineering
        if verbose:
            print("\n[1/5] Engineering features...")
        feat_df = build_feature_matrix(raw_df, **feature_params or {})
        if verbose:
            print(f"      {len(feat_df)} rows × {len(feat_df.columns)} columns")

        # Step 2: Leakage audit
        if verbose:
            print("\n[2/5] Leakage audit...")
        issues = audit_features(feat_df, verbose=verbose)
        if issues:
            print(f"  WARNING: {len(issues)} potential leakage issues found!")

        # Step 3: Temporal split
        if verbose:
            print(f"\n[3/5] Temporal split (test holdout = {test_holdout_months}m)...")
        splitter = TemporalSplitter(test_months=test_holdout_months, val_months=3)
        split = splitter.split(feat_df)
        splitter.print_split_info(feat_df)

        train_val_df = feat_df.iloc[
            np.concatenate([split.train_idx, split.val_idx])
        ].reset_index(drop=True)
        test_df = feat_df.iloc[split.test_idx].reset_index(drop=True)

        # Step 4: Fit model on train+val
        if verbose:
            print(f"\n[4/5] Fitting {model_type} on train+val "
                  f"({len(train_val_df)} rows)...")
        model_cls = _get_model_class(model_type)
        params = model_params or DEFAULT_MODEL_PARAMS.get(model_type, {})
        model = model_cls(**params)
        model.fit(train_val_df)
        if verbose:
            print(f"      Done.")

        # Step 5: Evaluate on locked test set
        if verbose:
            print(f"\n[5/5] Final evaluation on test set ({len(test_df)} rows)...")

        test_metrics = _evaluate_on_test(model, test_df, verbose=verbose)

        # Build pipeline
        pipe = cls(
            model_type=model_type,
            model_params=params,
            feature_params=feature_params,
            bet_params=bet_params,
        )
        pipe.model = model
        pipe.train_cutoff = train_val_df["date"].max()
        pipe.league = league
        pipe.metadata = {
            "league":          league,
            "model_type":      model_type,
            "model_params":    params,
            "n_train_val":     len(train_val_df),
            "n_test":          len(test_df),
            "train_cutoff":    str(pipe.train_cutoff),
            "test_start":      str(test_df["date"].min()),
            "test_end":        str(test_df["date"].max()),
            "leakage_issues":  issues,
            **test_metrics,
        }

        if verbose:
            print(f"\n  Pipeline ready. train_cutoff = {pipe.train_cutoff.date()}")
            print(f"{'='*60}")

        return pipe

    def predict_probs(self, new_fixtures_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict outcome probabilities for new fixtures.

        new_fixtures_df must have history rows mixed in (to compute features)
        OR have pre-computed feature columns.

        If 'h_ewma_gf' is not present, will attempt to build features using
        historical match data embedded in the DataFrame.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call build_and_fit() first.")

        # Check if features already present
        if "h_ewma_gf" not in new_fixtures_df.columns:
            new_fixtures_df = build_feature_matrix(
                new_fixtures_df, **self.feature_params
            )

        return self.model.predict_probs(new_fixtures_df)

    def predict_bets(
        self,
        new_fixtures_df: pd.DataFrame,
        bankroll: float = 1000.0,
        verbose: bool = True,
    ) -> list[BetRecommendation]:
        """
        Full prediction + betting filter pipeline.

        new_fixtures_df must have columns:
            date, home_team, away_team, odds_home, odds_draw, odds_away
        Plus enough historical context to compute features.

        Returns list of BetRecommendation objects, sorted by EV descending.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call build_and_fit() first.")

        # Build features
        if "h_ewma_gf" not in new_fixtures_df.columns:
            feat_df = build_feature_matrix(
                new_fixtures_df, **self.feature_params
            )
        else:
            feat_df = new_fixtures_df.copy()

        # Only predict on upcoming matches (no result yet)
        if "home_goals" in feat_df.columns:
            predict_mask = feat_df["home_goals"].isna()
            if predict_mask.sum() == 0:
                predict_mask = feat_df["date"] >= self.train_cutoff
        else:
            predict_mask = pd.Series([True] * len(feat_df))

        predict_df = feat_df[predict_mask].reset_index(drop=True)
        if len(predict_df) == 0:
            if verbose:
                print("No upcoming fixtures to predict.")
            return []

        probs_df = self.model.predict_probs(predict_df)

        bets = []
        for i, (_, row) in enumerate(predict_df.iterrows()):
            if i >= len(probs_df):
                break

            odds_home = row.get("odds_home", np.nan)
            odds_draw = row.get("odds_draw", np.nan)
            odds_away = row.get("odds_away", np.nan)

            if any(pd.isna(x) for x in [odds_home, odds_draw, odds_away]):
                continue

            # Overround filter
            overround = (1/odds_home + 1/odds_draw + 1/odds_away) - 1.0
            if overround > self.MAX_OVERROUND:
                continue

            p_home = float(probs_df.iloc[i]["p_home"])
            p_draw = float(probs_df.iloc[i]["p_draw"])
            p_away = float(probs_df.iloc[i]["p_away"])

            for market, prob, book_odds, min_ev in [
                ("home", p_home, odds_home, self.MIN_EV),
                ("draw", p_draw, odds_draw, self.DRAW_MIN_EV),
                ("away", p_away, odds_away, self.MIN_EV),
            ]:
                ev = prob * book_odds - 1.0
                if (ev < min_ev
                        or prob < self.MIN_PROB
                        or book_odds < self.MIN_ODDS
                        or book_odds > self.MAX_ODDS):
                    continue

                # Fractional Kelly
                b = book_odds - 1.0
                q = 1.0 - prob
                kelly_full = (b * prob - q) / b
                kelly_frac = max(0.0, kelly_full * self.KELLY_FRACTION)
                kelly_capped = min(kelly_frac, self.MAX_BET_FRAC)

                stake = bankroll * kelly_capped
                if stake < 1.0:
                    continue

                bets.append(BetRecommendation(
                    date=row.get("date"),
                    home_team=str(row.get("home_team", "")),
                    away_team=str(row.get("away_team", "")),
                    market=market,
                    model_prob=prob,
                    book_odds=float(book_odds),
                    ev=float(ev),
                    kelly_stake=float(kelly_capped),
                    stake_amount=float(stake),
                    expected_profit=float(stake * ev),
                ))

        bets.sort(key=lambda x: x.ev, reverse=True)

        if verbose:
            print(f"\n{'='*80}")
            print(f"  BET RECOMMENDATIONS [{self.league}]")
            print(f"{'='*80}")
            if bets:
                for b in bets:
                    print(f"  {b}")
                total_stake = sum(b.stake_amount for b in bets)
                total_exp   = sum(b.expected_profit for b in bets)
                print(f"\n  Total stake: {total_stake:.0f} | "
                      f"Expected profit: {total_exp:+.0f} | "
                      f"Bets: {len(bets)}")
            else:
                print("  No bets meet the threshold criteria.")
            print(f"{'='*80}")

        return bets

    def save(self, path: str) -> None:
        """Serialise the fitted pipeline to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump(self, path, compress=3)
        print(f"  Pipeline saved → {path}")

    @classmethod
    def load(cls, path: str) -> "DeployablePipeline":
        """Load a previously saved pipeline."""
        pipe = joblib.load(path)
        print(f"  Pipeline loaded ← {path}")
        print(f"  League: {pipe.league} | Model: {pipe.model_type} | "
              f"Cutoff: {pipe.train_cutoff}")
        return pipe

    def summary(self) -> str:
        meta = self.metadata
        lines = [
            f"\n{'='*60}",
            f"  PIPELINE SUMMARY",
            f"{'='*60}",
            f"  League        : {self.league}",
            f"  Model         : {self.model_type}",
            f"  Train cutoff  : {self.train_cutoff}",
            f"  Train+val rows: {meta.get('n_train_val', '?')}",
            f"  Test rows     : {meta.get('n_test', '?')}",
            f"  Test period   : {meta.get('test_start','?')} → {meta.get('test_end','?')}",
            f"  Test Brier    : {meta.get('test_brier', '?')}",
            f"  Test LogLoss  : {meta.get('test_log_loss', '?')}",
            f"  Leakage issues: {len(meta.get('leakage_issues', []))}",
            f"{'='*60}",
        ]
        return "\n".join(lines)

    def __repr__(self):
        return (f"DeployablePipeline(model={self.model_type}, "
                f"league={self.league}, cutoff={self.train_cutoff})")


# ── Test set evaluation helper ─────────────────────────────────────────────────

def _evaluate_on_test(
    model,
    test_df: pd.DataFrame,
    verbose: bool = True,
) -> dict:
    """Evaluate model on the locked test set. Called once."""
    if len(test_df) == 0:
        return {}

    try:
        probs = model.predict_probs(test_df)
        hg = test_df["home_goals"].values.astype(int)
        ag = test_df["away_goals"].values.astype(int)
        actual = np.where(hg > ag, 0, np.where(hg == ag, 1, 2))

        p = probs[["p_home", "p_draw", "p_away"]].values
        n = len(actual)
        one_hot = np.zeros((n, 3))
        one_hot[np.arange(n), actual] = 1.0

        brier = float(((p - one_hot) ** 2).sum(axis=1).mean())
        p_actual = p[np.arange(n), actual]
        ll = float(-np.log(np.maximum(p_actual, 1e-10)).mean())

        # Accuracy: predicted class = argmax prob
        pred_class = np.argmax(p, axis=1)
        acc = float((pred_class == actual).mean())

        if verbose:
            print(f"      Test Brier  : {brier:.5f}  (baseline=0.667)")
            print(f"      Test LogLoss: {ll:.5f}")
            print(f"      Test Acc    : {acc:.3f}")

        return {
            "test_brier":    round(brier, 5),
            "test_log_loss": round(ll, 5),
            "test_accuracy": round(acc, 3),
        }
    except Exception as e:
        if verbose:
            print(f"      Evaluation failed: {e}")
        return {}


# ── Multi-league deployment ────────────────────────────────────────────────────

class MultiLeaguePipeline:
    """
    Container for pipelines trained per league.

    Usage
    -----
    mlp = MultiLeaguePipeline()
    mlp.fit_leagues(
        data_by_league={"E0": pl_df, "SP1": la_liga_df},
        model_type="GradientBoostModel",
    )
    mlp.save_all("models/")

    bets = mlp.predict_all_leagues(fixtures_by_league, bankroll=10_000)
    """

    def __init__(self):
        self._pipelines: dict[str, DeployablePipeline] = {}

    def fit_leagues(
        self,
        data_by_league: dict[str, pd.DataFrame],
        model_type: str = "GradientBoostModel",
        model_params: dict | None = None,
        verbose: bool = True,
    ) -> "MultiLeaguePipeline":
        for league, df in data_by_league.items():
            if verbose:
                print(f"\n--- Fitting {league} ---")
            try:
                pipe = DeployablePipeline.build_and_fit(
                    raw_df=df,
                    league=league,
                    model_type=model_type,
                    model_params=model_params,
                    verbose=verbose,
                )
                self._pipelines[league] = pipe
            except Exception as e:
                print(f"  ERROR fitting {league}: {e}")
        return self

    def save_all(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        for league, pipe in self._pipelines.items():
            pipe.save(os.path.join(directory, f"{league}_pipeline.pkl"))

    @classmethod
    def load_all(cls, directory: str) -> "MultiLeaguePipeline":
        mlp = cls()
        for fname in os.listdir(directory):
            if fname.endswith("_pipeline.pkl"):
                league = fname.replace("_pipeline.pkl", "")
                path = os.path.join(directory, fname)
                mlp._pipelines[league] = DeployablePipeline.load(path)
        return mlp

    def predict_all_leagues(
        self,
        fixtures_by_league: dict[str, pd.DataFrame],
        bankroll: float = 1000.0,
        verbose: bool = True,
    ) -> dict[str, list[BetRecommendation]]:
        all_bets = {}
        for league, fixtures in fixtures_by_league.items():
            if league not in self._pipelines:
                continue
            bets = self._pipelines[league].predict_bets(
                fixtures, bankroll=bankroll, verbose=verbose
            )
            all_bets[league] = bets
        return all_bets

    def summary_table(self) -> pd.DataFrame:
        rows = []
        for league, pipe in self._pipelines.items():
            meta = pipe.metadata
            rows.append({
                "league":      league,
                "model":       pipe.model_type,
                "train_cutoff": str(pipe.train_cutoff)[:10],
                "n_train_val": meta.get("n_train_val"),
                "n_test":      meta.get("n_test"),
                "test_brier":  meta.get("test_brier"),
                "test_logloss": meta.get("test_log_loss"),
            })
        return pd.DataFrame(rows)

    def __repr__(self):
        return f"MultiLeaguePipeline(leagues={list(self._pipelines.keys())})"
