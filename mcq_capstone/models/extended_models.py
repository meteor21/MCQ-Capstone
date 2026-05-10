"""
Extended model families for soccer outcome prediction.

Adds:
  - ElasticNetGoalModel   : L1+L2 regularised goal regression (feature selection)
  - NegBinGoalModel       : Negative Binomial regression for overdispersed goals
  - StackingEnsemble      : Meta-model combining base model predictions

ElasticNet advantage
--------------------
Combines Ridge (L2) and Lasso (L1) penalties.  L1 drives low-signal features
toward zero, L2 prevents any single feature from dominating.  Better than
plain Ridge when many features are near-irrelevant.

Negative Binomial advantage
---------------------------
Poisson assumes Var[goals] = E[goals].  In practice, Var > E (overdispersion)
because some matches are consistently open/defensive.  NegBin adds a dispersion
parameter α such that Var = E + αE².  This produces better-calibrated tail
probabilities (e.g. 4-0 scores).

Stacking Ensemble advantage
----------------------------
Trains a meta-model (logistic regression) on the probability predictions of
multiple base models.  The meta-model learns which base models to trust most
for which regions of the feature space.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, List

from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from .goal_model import _time_weights, _feature_cols, _outcome_probs_from_lambdas
from .outcome_model import _encode_outcome


# ── ElasticNet goal model ──────────────────────────────────────────────────────

class ElasticNetGoalModel:
    """
    ElasticNet (L1+L2) regression for home and away goals.

    Parameters
    ----------
    alpha      : Overall regularisation strength.
    l1_ratio   : Mix between L1 (1.0) and L2 (0.0).  0.5 = equal mix.
    time_decay : Exponential weight decay rate (per day).
    max_iter   : Solver iterations.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        time_decay: float = 0.0065,
        max_iter: int = 2000,
    ):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.time_decay = time_decay
        self.max_iter = max_iter
        self._pipe_home: Optional[Pipeline] = None
        self._pipe_away: Optional[Pipeline] = None
        self.feature_names_: list[str] = []
        self.coef_home_: Optional[np.ndarray] = None
        self.coef_away_: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def fit(self, feat_df: pd.DataFrame) -> "ElasticNetGoalModel":
        cols = _feature_cols(feat_df)
        self.feature_names_ = cols
        X = feat_df[cols].values
        y_home = feat_df["home_goals"].values.astype(float)
        y_away = feat_df["away_goals"].values.astype(float)
        w = _time_weights(feat_df["date"], self.time_decay)

        def make_pipe():
            return Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale",  StandardScaler()),
                ("model",  ElasticNet(
                    alpha=self.alpha,
                    l1_ratio=self.l1_ratio,
                    max_iter=self.max_iter,
                    random_state=42,
                )),
            ])

        self._pipe_home = make_pipe()
        self._pipe_away = make_pipe()
        self._pipe_home.fit(X, y_home, model__sample_weight=w)
        self._pipe_away.fit(X, y_away, model__sample_weight=w)

        self.coef_home_ = self._pipe_home.named_steps["model"].coef_
        self.coef_away_ = self._pipe_away.named_steps["model"].coef_
        self.is_fitted = True
        return self

    def predict_goals(self, feat_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = feat_df[self.feature_names_].values
        lam = np.maximum(self._pipe_home.predict(X), 0.1)
        mu  = np.maximum(self._pipe_away.predict(X), 0.1)
        return lam, mu

    def predict_probs(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        lam, mu = self.predict_goals(feat_df)
        return _outcome_probs_from_lambdas(lam, mu)

    def feature_importance(self) -> pd.DataFrame:
        """Features with non-zero ElasticNet coefficients (sparse selection)."""
        diff = self.coef_home_ - self.coef_away_
        return pd.DataFrame({
            "feature":   self.feature_names_,
            "coef_home": self.coef_home_,
            "coef_away": self.coef_away_,
            "coef_diff": diff,
            "nonzero":   (np.abs(self.coef_home_) + np.abs(self.coef_away_)) > 0,
        }).sort_values("coef_diff", ascending=False).reset_index(drop=True)

    def n_selected_features(self) -> int:
        """How many features survived L1 shrinkage (non-zero coef)."""
        return int(
            (np.abs(self.coef_home_) > 0).sum() +
            (np.abs(self.coef_away_) > 0).sum()
        )

    def __repr__(self):
        n = self.n_selected_features() if self.is_fitted else "?"
        return (f"ElasticNetGoalModel(alpha={self.alpha}, l1_ratio={self.l1_ratio}, "
                f"selected_features={n}, fitted={self.is_fitted})")


# ── Negative Binomial goal model ───────────────────────────────────────────────

class NegBinGoalModel:
    """
    Negative Binomial regression for goals (handles overdispersion).

    Uses statsmodels NegativeBinomial.  Falls back to PoissonGoalModel
    if statsmodels is not available.

    Why Negative Binomial?
    ----------------------
    In the Premier League (2018-24), the variance of home goals is ~1.62
    while the mean is ~1.53.  Poisson assumes these are equal.  The
    overdispersion is small but meaningful for tail calibration (e.g. 4-0,
    5-1 results).

    Parameters
    ----------
    alpha      : Dispersion parameter regularisation (passed to statsmodels).
    time_decay : Exponential weight decay rate.
    """

    def __init__(self, alpha: float = 1.0, time_decay: float = 0.0065):
        self.alpha = alpha
        self.time_decay = time_decay
        self._model_home = None
        self._model_away = None
        self._imputer_home = SimpleImputer(strategy="median")
        self._imputer_away = SimpleImputer(strategy="median")
        self._scaler_home = StandardScaler()
        self._scaler_away = StandardScaler()
        self.feature_names_: list[str] = []
        self._use_statsmodels: bool = False
        self.is_fitted: bool = False

    def fit(self, feat_df: pd.DataFrame) -> "NegBinGoalModel":
        try:
            import statsmodels.api as sm
            self._use_statsmodels = True
        except ImportError:
            self._use_statsmodels = False

        cols = _feature_cols(feat_df)
        self.feature_names_ = cols
        X_raw = feat_df[cols].values
        y_home = feat_df["home_goals"].values.astype(float)
        y_away = feat_df["away_goals"].values.astype(float)
        w = _time_weights(feat_df["date"], self.time_decay)

        X_h = self._scaler_home.fit_transform(
            self._imputer_home.fit_transform(X_raw)
        )
        X_a = self._scaler_away.fit_transform(
            self._imputer_away.fit_transform(X_raw)
        )

        if self._use_statsmodels:
            import statsmodels.api as sm
            X_h_c = sm.add_constant(X_h)
            X_a_c = sm.add_constant(X_a)
            try:
                self._model_home = sm.NegativeBinomial(
                    y_home, X_h_c, loglike_method="nb2"
                ).fit(
                    freq_weights=w,
                    disp=0,
                    maxiter=200,
                    method="bfgs",
                    full_output=False,
                    disp_score=False,
                )
                self._model_away = sm.NegativeBinomial(
                    y_away, X_a_c, loglike_method="nb2"
                ).fit(
                    freq_weights=w,
                    disp=0,
                    maxiter=200,
                    method="bfgs",
                    full_output=False,
                    disp_score=False,
                )
            except Exception:
                # NegBin failed to converge — fall back to Poisson
                self._use_statsmodels = False
                self._fit_poisson_fallback(feat_df, cols, w)
        else:
            self._fit_poisson_fallback(feat_df, cols, w)

        self.is_fitted = True
        return self

    def _fit_poisson_fallback(self, feat_df, cols, w):
        """Fall back to Poisson when NegBin fails."""
        from sklearn.linear_model import PoissonRegressor
        X_h = self._scaler_home.transform(self._imputer_home.transform(
            feat_df[cols].values
        ))
        X_a = self._scaler_away.transform(self._imputer_away.transform(
            feat_df[cols].values
        ))
        self._model_home = PoissonRegressor(alpha=self.alpha, max_iter=300)
        self._model_away = PoissonRegressor(alpha=self.alpha, max_iter=300)
        self._model_home.fit(X_h, feat_df["home_goals"].values.astype(float),
                             sample_weight=w)
        self._model_away.fit(X_a, feat_df["away_goals"].values.astype(float),
                             sample_weight=w)

    def predict_goals(self, feat_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X_raw = feat_df[self.feature_names_].values
        X_h = self._scaler_home.transform(self._imputer_home.transform(X_raw))
        X_a = self._scaler_away.transform(self._imputer_away.transform(X_raw))

        if self._use_statsmodels:
            import statsmodels.api as sm
            X_h_c = sm.add_constant(X_h, has_constant="add")
            X_a_c = sm.add_constant(X_a, has_constant="add")
            lam = np.exp(X_h_c @ self._model_home.params)
            mu  = np.exp(X_a_c @ self._model_away.params)
        else:
            lam = self._model_home.predict(X_h)
            mu  = self._model_away.predict(X_a)

        return np.maximum(lam, 0.1), np.maximum(mu, 0.1)

    def predict_probs(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        lam, mu = self.predict_goals(feat_df)
        return _outcome_probs_from_lambdas(lam, mu)

    def __repr__(self):
        backend = "NegBin(statsmodels)" if self._use_statsmodels else "Poisson(fallback)"
        return f"NegBinGoalModel({backend}, fitted={self.is_fitted})"


# ── Stacking Ensemble ──────────────────────────────────────────────────────────

class StackingEnsemble:
    """
    Two-level stacking ensemble.

    Level-0 (base models): PoissonGoalModel, RidgeGoalModel,
                           LogisticOutcomeModel, GradientBoostModel,
                           ElasticNetGoalModel, NegBinGoalModel
    Level-1 (meta-model) : Logistic regression on stacked probabilities.

    Key anti-leakage design
    -----------------------
    Base models are fit on OOF (out-of-fold) training data via
    TimeSeriesSplit.  Meta-model trains on the OOF predictions.
    At inference, base models fitted on full training data produce
    predictions for the meta-model.

    Parameters
    ----------
    base_models    : List of (name, unfitted_model_instance) pairs.
    n_meta_folds   : Folds for generating OOF predictions (default 5).
    meta_C         : Regularisation for meta logistic regression.
    """

    def __init__(
        self,
        base_models: list[tuple[str, Any]] | None = None,
        n_meta_folds: int = 5,
        meta_C: float = 1.0,
    ):
        self.n_meta_folds = n_meta_folds
        self.meta_C = meta_C
        self._base_models: list[tuple[str, Any]] = base_models or []
        self._fitted_base: list[tuple[str, Any]] = []
        self._meta: Optional[LogisticRegression] = None
        self.is_fitted: bool = False

    def _default_base_models(self):
        from .goal_model import PoissonGoalModel, RidgeGoalModel
        from .outcome_model import LogisticOutcomeModel, GradientBoostModel
        return [
            ("poisson",   PoissonGoalModel()),
            ("ridge",     RidgeGoalModel()),
            ("logistic",  LogisticOutcomeModel()),
            ("gbm",       GradientBoostModel(n_estimators=100)),
            ("elasticnet", ElasticNetGoalModel()),
        ]

    def fit(self, feat_df: pd.DataFrame) -> "StackingEnsemble":
        import copy
        from sklearn.model_selection import TimeSeriesSplit

        if not self._base_models:
            self._base_models = self._default_base_models()

        feat_df = feat_df.sort_values("date").reset_index(drop=True)
        n = len(feat_df)
        n_models = len(self._base_models)

        # --- Generate OOF predictions for meta-model training ---
        oof_preds = np.full((n, n_models * 3), np.nan)
        tscv = TimeSeriesSplit(n_splits=self.n_meta_folds)

        for fold_i, (tr_idx, va_idx) in enumerate(tscv.split(feat_df)):
            if len(tr_idx) < 50 or len(va_idx) < 5:
                continue
            train_fold = feat_df.iloc[tr_idx]
            val_fold   = feat_df.iloc[va_idx]

            for mi, (name, model_proto) in enumerate(self._base_models):
                model = copy.deepcopy(model_proto)
                try:
                    model.fit(train_fold)
                    probs = model.predict_probs(val_fold)
                    col_start = mi * 3
                    oof_preds[va_idx, col_start:col_start+3] = probs[
                        ["p_home", "p_draw", "p_away"]
                    ].values
                except Exception:
                    pass

        # --- Fit meta-model on OOF predictions ---
        valid_mask = ~np.isnan(oof_preds).any(axis=1)
        X_meta = oof_preds[valid_mask]
        y_meta = _encode_outcome(feat_df.iloc[valid_mask])

        self._meta = LogisticRegression(
            C=self.meta_C,
            solver="lbfgs", max_iter=500,
        )
        self._meta.fit(X_meta, y_meta)

        # --- Fit base models on FULL training data ---
        self._fitted_base = []
        for name, model_proto in self._base_models:
            model = copy.deepcopy(model_proto)
            try:
                model.fit(feat_df)
                self._fitted_base.append((name, model))
            except Exception as e:
                pass

        self.is_fitted = True
        return self

    def predict_probs(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Call .fit() first.")

        n = len(feat_df)
        n_models = len(self._fitted_base)
        X_meta = np.full((n, n_models * 3), np.nan)

        for mi, (name, model) in enumerate(self._fitted_base):
            try:
                probs = model.predict_probs(feat_df)
                X_meta[:, mi*3:(mi+1)*3] = probs[
                    ["p_home", "p_draw", "p_away"]
                ].values
            except Exception:
                pass

        # Fill NaN columns with uniform prior
        col_nan = np.isnan(X_meta).any(axis=0)
        X_meta[:, col_nan] = 1/3

        proba = self._meta.predict_proba(X_meta)
        return pd.DataFrame(proba, columns=["p_home", "p_draw", "p_away"])

    def __repr__(self):
        base_names = [n for n, _ in (self._fitted_base or self._base_models)]
        return (f"StackingEnsemble(base={base_names}, "
                f"meta_C={self.meta_C}, fitted={self.is_fitted})")
