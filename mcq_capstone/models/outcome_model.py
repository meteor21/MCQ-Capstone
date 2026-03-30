"""
Direct outcome classification models.

Instead of going through goal prediction → Poisson distribution, these models
predict P(home win), P(draw), P(away win) directly from features.

Models
------
LogisticOutcomeModel  — multinomial logistic regression (WLS via sample_weight)
GradientBoostModel    — XGBoost / sklearn GradientBoosting (non-linear)
                        Captures feature interactions and non-linear effects.

Both expose .fit() / .predict_probs() like the goal models.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.multiclass import OneVsRestClassifier
from typing import Optional

from .goal_model import _time_weights, _feature_cols


def _encode_outcome(df: pd.DataFrame) -> np.ndarray:
    """Encode H/D/A as 0/1/2."""
    hg = df["home_goals"].values
    ag = df["away_goals"].values
    return np.where(hg > ag, 0, np.where(hg == ag, 1, 2))


# ── Logistic regression ────────────────────────────────────────────────────────

class LogisticOutcomeModel:
    """
    Multinomial logistic regression for H/D/A prediction.

    Time-decay weights applied via sample_weight.
    L2 regularisation (C = 1/alpha) prevents overfitting on small leagues.
    """

    def __init__(self, C: float = 1.0, time_decay: float = 0.0065,
                 max_iter: int = 1000):
        self.C = C
        self.time_decay = time_decay
        self.max_iter = max_iter
        self._pipe: Optional[Pipeline] = None
        self.feature_names_: list[str] = []
        self.classes_  = ["home", "draw", "away"]
        self.is_fitted: bool = False

    def fit(self, feat_df: pd.DataFrame) -> "LogisticOutcomeModel":
        cols = _feature_cols(feat_df)
        self.feature_names_ = cols
        X = feat_df[cols].values
        y = _encode_outcome(feat_df)
        w = _time_weights(feat_df["date"], self.time_decay)

        self._pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale",  StandardScaler()),
            ("model",  LogisticRegression(
                C=self.C, multi_class="multinomial",
                solver="lbfgs", max_iter=self.max_iter,
            )),
        ])
        self._pipe.fit(X, y, model__sample_weight=w)
        self.is_fitted = True
        return self

    def predict_probs(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        X = feat_df[self.feature_names_].values
        proba = self._pipe.predict_proba(X)   # (n, 3): [home, draw, away]
        return pd.DataFrame(
            proba, columns=["p_home", "p_draw", "p_away"]
        )

    def __repr__(self):
        return f"LogisticOutcomeModel(C={self.C}, fitted={self.is_fitted})"


# ── Gradient boosting (non-linear) ────────────────────────────────────────────

class GradientBoostModel:
    """
    Gradient boosting for outcome prediction.

    Tries XGBoost first; falls back to sklearn GradientBoostingClassifier
    if xgboost is not installed.

    Non-linear model — captures:
      - Feature interactions (e.g. attack × opponent defence)
      - Non-linear thresholds (form cliff effects)
      - Asymmetric effects (winning streak vs losing streak)

    SHAP values available via .explain(X) if shap is installed.
    """

    def __init__(self, n_estimators: int = 200, learning_rate: float = 0.05,
                 max_depth: int = 4, time_decay: float = 0.0065,
                 subsample: float = 0.8):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.time_decay = time_decay
        self.subsample = subsample
        self._imputer = SimpleImputer(strategy="median")
        self._model = None
        self._use_xgb: bool = False
        self.feature_names_: list[str] = []
        self.is_fitted: bool = False

    def _build_model(self):
        try:
            from xgboost import XGBClassifier
            self._use_xgb = True
            return XGBClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                subsample=self.subsample,
                use_label_encoder=False,
                eval_metric="mlogloss",
                verbosity=0,
                random_state=42,
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.multiclass import OneVsRestClassifier
            self._use_xgb = False
            return OneVsRestClassifier(GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                subsample=self.subsample,
                random_state=42,
            ))

    def fit(self, feat_df: pd.DataFrame) -> "GradientBoostModel":
        cols = _feature_cols(feat_df)
        self.feature_names_ = cols
        X = self._imputer.fit_transform(feat_df[cols].values)
        y = _encode_outcome(feat_df)
        w = _time_weights(feat_df["date"], self.time_decay)

        self._model = self._build_model()
        if self._use_xgb:
            self._model.fit(X, y, sample_weight=w)
        else:
            self._model.fit(X, y, sample_weight=w)
        self.is_fitted = True
        return self

    def predict_probs(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        X = self._imputer.transform(feat_df[self.feature_names_].values)
        proba = self._model.predict_proba(X)
        if proba.shape[1] == 3:
            return pd.DataFrame(proba, columns=["p_home", "p_draw", "p_away"])
        # Fallback if OneVsRest doesn't produce calibrated probabilities
        proba = proba / proba.sum(axis=1, keepdims=True)
        return pd.DataFrame(proba, columns=["p_home", "p_draw", "p_away"])

    def feature_importance(self) -> pd.DataFrame:
        """Return feature importances (gain for XGBoost, mean impurity for sklearn)."""
        if self._use_xgb:
            imp = self._model.feature_importances_
        else:
            # Average across OvR estimators
            imp = np.mean([e.feature_importances_
                           for e in self._model.estimators_], axis=0)
        return pd.DataFrame({
            "feature":    self.feature_names_,
            "importance": imp,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

    def explain(self, feat_df: pd.DataFrame) -> Optional[object]:
        """
        Return SHAP explanation object (requires shap package).
        Use: shap.summary_plot(explain.values, feat_df[feature_names])
        """
        try:
            import shap
        except ImportError:
            print("shap not installed. Run: pip install shap")
            return None
        if not self._use_xgb:
            print("SHAP explanation only supported for XGBoost backend.")
            return None
        X = self._imputer.transform(feat_df[self.feature_names_].values)
        explainer = shap.TreeExplainer(self._model)
        return explainer(X)

    def __repr__(self):
        backend = "XGBoost" if self._use_xgb else "GradientBoosting"
        return (f"GradientBoostModel({backend}, n={self.n_estimators}, "
                f"fitted={self.is_fitted})")
