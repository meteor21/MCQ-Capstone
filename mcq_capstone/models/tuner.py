"""
Hyperparameter tuning using TimeSeriesSplit cross-validation.

Key principles
--------------
- TimeSeriesSplit (not random k-fold) — future data never leaks into training.
- Imputer and scaler are re-fit inside each fold on TRAIN data only.
- Scoring metric: Brier score (lower = better for probability calibration).
- Search: Grid search over explicit param grids (no random — reproducible).
- TEST set is NEVER touched during tuning.

Supported models
----------------
- LogisticOutcomeModel   (C, time_decay)
- GradientBoostModel     (n_estimators, max_depth, learning_rate, subsample)
- PoissonGoalModel       (alpha, time_decay)
- RidgeGoalModel         (alpha, time_decay)
- ElasticNetGoalModel    (alpha, l1_ratio, time_decay)

Usage
-----
from mcq_capstone.models.tuner import HyperparameterTuner
from mcq_capstone.models.outcome_model import LogisticOutcomeModel

tuner = HyperparameterTuner(n_cv_folds=5, verbose=True)
best_params, cv_results = tuner.tune(
    model_class=LogisticOutcomeModel,
    param_grid={"C": [0.1, 1.0, 10.0], "time_decay": [0.003, 0.006]},
    feat_df=train_val_df,   # NEVER pass test data here
)
best_model = LogisticOutcomeModel(**best_params)
best_model.fit(train_val_df)
"""

from __future__ import annotations
import copy
import itertools
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Type
from sklearn.model_selection import TimeSeriesSplit


# ── Default param grids ────────────────────────────────────────────────────────

PARAM_GRIDS: dict[str, dict] = {
    "LogisticOutcomeModel": {
        "C":          [0.01, 0.1, 1.0, 5.0, 10.0],
        "time_decay": [0.003, 0.0065, 0.010],
    },
    "GradientBoostModel": {
        "n_estimators":  [100, 200, 400],
        "max_depth":     [3, 4, 5],
        "learning_rate": [0.03, 0.05, 0.10],
        "subsample":     [0.7, 0.8],
    },
    "PoissonGoalModel": {
        "alpha":      [0.01, 0.1, 1.0, 10.0],
        "time_decay": [0.003, 0.0065, 0.010],
    },
    "RidgeGoalModel": {
        "alpha":      [0.01, 0.1, 1.0, 10.0, 100.0],
        "time_decay": [0.003, 0.0065, 0.010],
    },
    "ElasticNetGoalModel": {
        "alpha":      [0.01, 0.1, 1.0, 10.0],
        "l1_ratio":   [0.1, 0.5, 0.9],
        "time_decay": [0.003, 0.0065, 0.010],
    },
    "NegBinGoalModel": {
        "alpha":      [0.5, 1.0, 2.0, 5.0],
        "time_decay": [0.003, 0.0065, 0.010],
    },
}


# ── Scoring helpers ────────────────────────────────────────────────────────────

def _outcome_code(hg: np.ndarray, ag: np.ndarray) -> np.ndarray:
    return np.where(hg > ag, 0, np.where(hg == ag, 1, 2))


def _brier_score(probs_df: pd.DataFrame, actual: np.ndarray) -> float:
    """
    Multi-class Brier score.
    probs_df: columns p_home, p_draw, p_away (n rows)
    actual  : array of 0/1/2
    """
    p = probs_df[["p_home", "p_draw", "p_away"]].values
    n = len(actual)
    one_hot = np.zeros((n, 3))
    one_hot[np.arange(n), actual] = 1.0
    return float(((p - one_hot) ** 2).sum(axis=1).mean())


def _log_loss(probs_df: pd.DataFrame, actual: np.ndarray, eps: float = 1e-10) -> float:
    p = probs_df[["p_home", "p_draw", "p_away"]].values
    p_actual = p[np.arange(len(actual)), actual]
    return float(-np.log(np.maximum(p_actual, eps)).mean())


# ── CV fold evaluation ─────────────────────────────────────────────────────────

def _eval_fold(
    model_class: type,
    params: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> tuple[float, float]:
    """
    Fit model on train_df, score on val_df.
    Returns (brier, log_loss).  Returns (inf, inf) on any error.
    """
    try:
        model = model_class(**params)
        model.fit(train_df)
        probs = model.predict_probs(val_df)
        actual = _outcome_code(
            val_df["home_goals"].values.astype(int),
            val_df["away_goals"].values.astype(int),
        )
        return _brier_score(probs, actual), _log_loss(probs, actual)
    except Exception as e:
        return float("inf"), float("inf")


# ── Main tuner ─────────────────────────────────────────────────────────────────

@dataclass
class TuningResult:
    best_params: dict
    best_brier: float
    best_log_loss: float
    cv_results: pd.DataFrame   # all param combos × folds
    model_name: str

    def summary(self) -> str:
        lines = [
            f"\n{'='*55}",
            f"  TUNING RESULTS: {self.model_name}",
            f"{'='*55}",
            f"  Best params  : {self.best_params}",
            f"  Best Brier   : {self.best_brier:.5f}",
            f"  Best LogLoss : {self.best_log_loss:.5f}",
            f"  Param combos : {len(self.cv_results['param_hash'].unique())}",
            f"  CV folds     : {self.cv_results['fold'].max() + 1}",
            f"{'='*55}",
        ]
        return "\n".join(lines)


class HyperparameterTuner:
    """
    Grid search with TimeSeriesSplit CV.

    Parameters
    ----------
    n_cv_folds : Number of TimeSeriesSplit folds (default 5).
    verbose    : Print progress.
    metric     : 'brier' or 'log_loss' (default 'brier').
    """

    def __init__(
        self,
        n_cv_folds: int = 5,
        verbose: bool = True,
        metric: str = "brier",
    ):
        self.n_cv_folds = n_cv_folds
        self.verbose = verbose
        self.metric = metric

    def tune(
        self,
        model_class: type,
        feat_df: pd.DataFrame,
        param_grid: dict | None = None,
        min_train_size: int = 60,
    ) -> TuningResult:
        """
        Run grid search over param_grid using TimeSeriesSplit CV.

        Parameters
        ----------
        model_class    : Class (not instance) of the model.
        feat_df        : Feature DataFrame for train+val (NEVER include test).
        param_grid     : Dict of param_name → list of values.
                         If None, uses PARAM_GRIDS[model_class.__name__].
        min_train_size : Minimum rows in a CV training fold.

        Returns
        -------
        TuningResult with best_params and full cv_results.
        """
        name = model_class.__name__
        if param_grid is None:
            param_grid = PARAM_GRIDS.get(name, {})
            if not param_grid:
                raise ValueError(
                    f"No default param grid for {name}. "
                    f"Provide param_grid explicitly."
                )

        feat_df = feat_df.sort_values("date").reset_index(drop=True)

        # Build param combinations
        keys = list(param_grid.keys())
        combos = list(itertools.product(*[param_grid[k] for k in keys]))
        n_combos = len(combos)

        if self.verbose:
            print(f"\nTuning {name}: {n_combos} param combos × {self.n_cv_folds} folds")

        # TimeSeriesSplit indices
        tscv = TimeSeriesSplit(n_splits=self.n_cv_folds)
        folds = list(tscv.split(feat_df))

        # Filter out tiny folds
        folds = [
            (tr, va) for tr, va in folds
            if len(tr) >= min_train_size and len(va) >= 5
        ]
        if not folds:
            raise ValueError("No valid CV folds — dataset too small.")

        rows = []
        for ci, combo in enumerate(combos):
            params = dict(zip(keys, combo))
            param_hash = str(sorted(params.items()))

            fold_briers, fold_lls = [], []
            for fi, (tr_idx, va_idx) in enumerate(folds):
                train_fold = feat_df.iloc[tr_idx]
                val_fold   = feat_df.iloc[va_idx]
                b, ll = _eval_fold(model_class, params, train_fold, val_fold)
                fold_briers.append(b)
                fold_lls.append(ll)
                rows.append({
                    "param_hash":  param_hash,
                    "fold":        fi,
                    "brier":       b,
                    "log_loss":    ll,
                    **params,
                })

            mean_b = np.mean([x for x in fold_briers if np.isfinite(x)] or [np.inf])
            if self.verbose:
                print(
                    f"  [{ci+1:>3}/{n_combos}] {str(params):<55} "
                    f"Brier={mean_b:.4f}"
                )

        cv_results = pd.DataFrame(rows)

        # Aggregate by param combo
        agg = (
            cv_results
            .groupby("param_hash")[["brier", "log_loss"]]
            .mean()
            .reset_index()
        )

        # Find best params
        sort_col = "brier" if self.metric == "brier" else "log_loss"
        best_row = agg.sort_values(sort_col).iloc[0]
        best_hash = best_row["param_hash"]

        # Recover original params from the hash
        match_rows = cv_results[cv_results["param_hash"] == best_hash].iloc[0]
        best_params = {k: match_rows[k] for k in keys}

        result = TuningResult(
            best_params  = best_params,
            best_brier   = float(best_row["brier"]),
            best_log_loss= float(best_row["log_loss"]),
            cv_results   = cv_results,
            model_name   = name,
        )
        if self.verbose:
            print(result.summary())

        return result


def tune_all_models(
    feat_df: pd.DataFrame,
    models_to_tune: list[tuple] | None = None,
    n_cv_folds: int = 5,
    verbose: bool = True,
) -> dict[str, TuningResult]:
    """
    Convenience wrapper: tune multiple model families and return all results.

    Parameters
    ----------
    feat_df         : train+val feature DataFrame.
    models_to_tune  : List of (model_class, param_grid|None).
                      If None, tunes all models with default grids.

    Returns
    -------
    Dict of {model_name: TuningResult}
    """
    from .outcome_model import LogisticOutcomeModel, GradientBoostModel
    from .goal_model import PoissonGoalModel, RidgeGoalModel
    from .extended_models import ElasticNetGoalModel, NegBinGoalModel

    if models_to_tune is None:
        models_to_tune = [
            (LogisticOutcomeModel, None),
            (GradientBoostModel,   None),
            (PoissonGoalModel,     None),
            (RidgeGoalModel,       None),
            (ElasticNetGoalModel,  None),
            (NegBinGoalModel,      None),
        ]

    tuner = HyperparameterTuner(n_cv_folds=n_cv_folds, verbose=verbose)
    results = {}
    for cls, grid in models_to_tune:
        try:
            res = tuner.tune(cls, feat_df, param_grid=grid)
            results[cls.__name__] = res
        except Exception as e:
            if verbose:
                print(f"  ERROR tuning {cls.__name__}: {e}")
    return results
