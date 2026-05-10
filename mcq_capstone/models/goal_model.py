"""
Goal prediction models.

Two approaches:
  1. PoissonGoalModel   — Two Poisson GLMs (home goals, away goals) with
                          engineered features and time-decay weights.
                          Generalises Dixon-Coles with richer inputs.
  2. RidgeGoalModel     — WLS Ridge regression on goal difference.
                          Simpler, very fast, useful as baseline.

Both expose the same interface:
    .fit(X, y_home, y_away, weights)
    .predict_goals(X)         → (lambda_home, lambda_away)
    .predict_probs(X)         → DataFrame with p_home, p_draw, p_away
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import Ridge, PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from typing import Optional


def _outcome_probs_from_lambdas(lam: np.ndarray,
                                mu: np.ndarray,
                                max_goals: int = 10) -> pd.DataFrame:
    """
    Convert (lambda_home, lambda_away) arrays to H/D/A probabilities
    using independent Poisson distributions.
    """
    g = np.arange(max_goals + 1)
    rows = []
    for l, m in zip(lam, mu):
        p_h = poisson.pmf(g, max(l, 1e-6))
        p_a = poisson.pmf(g, max(m, 1e-6))
        matrix = np.outer(p_h, p_a)
        total = matrix.sum()
        rows.append({
            "p_home": float(np.tril(matrix, -1).sum() / total),
            "p_draw": float(np.trace(matrix) / total),
            "p_away": float(np.triu(matrix, 1).sum() / total),
            "lambda_home": float(l),
            "lambda_away": float(m),
        })
    return pd.DataFrame(rows)


def _time_weights(dates: pd.Series, decay: float = 0.0065) -> np.ndarray:
    """Exponential time-decay weights. Most recent date gets weight 1.0."""
    ref = dates.max()
    days_ago = (ref - dates).dt.days.values.astype(float)
    w = np.exp(-decay * days_ago)
    return w / w.sum() * len(w)   # normalise so mean weight = 1


def _feature_cols(df: pd.DataFrame) -> list[str]:
    """Return engineered feature column names (excludes match metadata)."""
    meta = {"date", "home_team", "away_team", "home_goals", "away_goals",
            "odds_home", "odds_draw", "odds_away", "league", "div",
            "result", "ftr"}
    return [c for c in df.columns if c.lower() not in meta
            and df[c].dtype in [np.float64, np.int64, float, int]]


# ── Poisson GLM ────────────────────────────────────────────────────────────────

class PoissonGoalModel:
    """
    Two Poisson regression models (sklearn PoissonRegressor):
      - model_home: predicts λ_home (expected home goals)
      - model_away: predicts λ_away (expected away goals)

    Features are standardised and missing values imputed with column median.
    Time-decay weights are applied during fit.

    Parameters
    ----------
    alpha      : L2 regularisation strength (higher = smoother / less overfit).
    time_decay : Per-day exponential decay for sample weights.
    max_iter   : Solver iterations.
    """

    def __init__(self, alpha: float = 1.0, time_decay: float = 0.0065,
                 max_iter: int = 500):
        self.alpha = alpha
        self.time_decay = time_decay
        self.max_iter = max_iter
        self._pipe_home: Optional[Pipeline] = None
        self._pipe_away: Optional[Pipeline] = None
        self.feature_names_: list[str] = []
        self.is_fitted: bool = False

    def _make_pipe(self) -> Pipeline:
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale",  StandardScaler()),
            ("model",  PoissonRegressor(alpha=self.alpha, max_iter=self.max_iter)),
        ])

    def fit(self, feat_df: pd.DataFrame) -> "PoissonGoalModel":
        """
        Fit on a feature DataFrame that includes home_goals, away_goals, date.
        """
        cols = _feature_cols(feat_df)
        self.feature_names_ = cols
        X = feat_df[cols].values
        y_h = feat_df["home_goals"].values.astype(float)
        y_a = feat_df["away_goals"].values.astype(float)
        w   = _time_weights(feat_df["date"], self.time_decay)

        self._pipe_home = self._make_pipe()
        self._pipe_away = self._make_pipe()
        self._pipe_home.fit(X, y_h, model__sample_weight=w)
        self._pipe_away.fit(X, y_a, model__sample_weight=w)
        self.is_fitted = True
        return self

    def predict_goals(self, feat_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (lambda_home, lambda_away) arrays."""
        X = feat_df[self.feature_names_].values
        lam = np.maximum(self._pipe_home.predict(X), 1e-4)
        mu  = np.maximum(self._pipe_away.predict(X), 1e-4)
        return lam, mu

    def predict_probs(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame with p_home, p_draw, p_away, lambda_home, lambda_away."""
        lam, mu = self.predict_goals(feat_df)
        return _outcome_probs_from_lambdas(lam, mu)

    def __repr__(self):
        return f"PoissonGoalModel(alpha={self.alpha}, fitted={self.is_fitted})"


# ── Ridge WLS ─────────────────────────────────────────────────────────────────

class RidgeGoalModel:
    """
    WLS Ridge regression predicting home goals and away goals separately.

    Simpler than Poisson GLM — normal linear model — but still effective
    and very fast.  Good as a baseline and for feature importance.

    Parameters
    ----------
    alpha      : Ridge L2 regularisation strength.
    time_decay : Per-day exponential decay weight.
    """

    def __init__(self, alpha: float = 1.0, time_decay: float = 0.0065):
        self.alpha = alpha
        self.time_decay = time_decay
        self._pipe_home: Optional[Pipeline] = None
        self._pipe_away: Optional[Pipeline] = None
        self.feature_names_: list[str] = []
        self.coef_home_: Optional[np.ndarray] = None
        self.coef_away_: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def _make_pipe(self) -> Pipeline:
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale",  StandardScaler()),
            ("model",  Ridge(alpha=self.alpha)),
        ])

    def fit(self, feat_df: pd.DataFrame) -> "RidgeGoalModel":
        cols = _feature_cols(feat_df)
        self.feature_names_ = cols
        X = feat_df[cols].values
        y_h = feat_df["home_goals"].values.astype(float)
        y_a = feat_df["away_goals"].values.astype(float)
        w   = _time_weights(feat_df["date"], self.time_decay)

        self._pipe_home = self._make_pipe()
        self._pipe_away = self._make_pipe()
        self._pipe_home.fit(X, y_h, model__sample_weight=w)
        self._pipe_away.fit(X, y_a, model__sample_weight=w)

        self.coef_home_ = self._pipe_home.named_steps["model"].coef_
        self.coef_away_ = self._pipe_away.named_steps["model"].coef_
        self.is_fitted = True
        return self

    def predict_goals(self, feat_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = feat_df[self.feature_names_].values
        lam = np.maximum(self._pipe_home.predict(X), 1e-4)
        mu  = np.maximum(self._pipe_away.predict(X), 1e-4)
        return lam, mu

    def predict_probs(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        lam, mu = self.predict_goals(feat_df)
        return _outcome_probs_from_lambdas(lam, mu)

    def feature_importance(self) -> pd.DataFrame:
        """Return feature coefficients for home and away goal models."""
        return pd.DataFrame({
            "feature":   self.feature_names_,
            "coef_home": self.coef_home_,
            "coef_away": self.coef_away_,
            "coef_diff": self.coef_home_ - self.coef_away_,
        }).assign(abs_diff=lambda d: d["coef_diff"].abs()
        ).sort_values("abs_diff", ascending=False).reset_index(drop=True)

    def __repr__(self):
        return f"RidgeGoalModel(alpha={self.alpha}, fitted={self.is_fitted})"
