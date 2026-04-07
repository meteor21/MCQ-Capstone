"""
Market pricing analysis.

Goal: reverse-engineer what drives bookmaker prices using fundamental
team statistics, then identify systematic deviations.

Method
------
1. Target variable: logit of market-implied home win probability
     y_i = log( p_market_home / (1 - p_market_home) )

2. Features: engineered team stats (no odds used as inputs)

3. Model: WLS Ridge regression with time-decay weights
     y_hat = X @ beta

4. Residuals: y_i - y_hat_i
     Large positive residual → model expects higher home win prob than market
       (market undervaluing home team — potential value on home)
     Large negative residual → market prices home team higher than model
       (potential value on away)

5. Feature coefficients reveal WHAT the market is most sensitive to.
   If a coefficient for "h_form_trend" is near zero, the market ignores
   recent momentum — you can exploit this if form_trend is genuinely predictive.

This is the intelligence layer: find the systematic blind spots in pricing.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import warnings

from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from .goal_model import _time_weights, _feature_cols


def _market_implied_probs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract fair market-implied probabilities by removing overround
    multiplicatively from B365 lines.
    """
    oh = df["odds_home"].values.astype(float)
    od = df["odds_draw"].values.astype(float)
    oa = df["odds_away"].values.astype(float)

    raw_h = 1.0 / oh
    raw_d = 1.0 / od
    raw_a = 1.0 / oa
    total = raw_h + raw_d + raw_a

    return pd.DataFrame({
        "market_p_home": raw_h / total,
        "market_p_draw": raw_d / total,
        "market_p_away": raw_a / total,
        "overround":     total - 1.0,
    })


def _logit(p: np.ndarray, clip: float = 1e-4) -> np.ndarray:
    p = np.clip(p, clip, 1 - clip)
    return np.log(p / (1 - p))


class MarketPricingModel:
    """
    Reverse-engineer bookmaker pricing and expose where the market deviates
    from fundamentals.

    Usage
    -----
    mpm = MarketPricingModel()
    mpm.fit(feat_df_with_odds)   # needs odds_home, odds_draw, odds_away columns
    report = mpm.market_inefficiency_report()
    print(report)
    """

    def __init__(self, time_decay: float = 0.0065,
                 alphas: tuple = (0.01, 0.1, 1.0, 10.0, 100.0)):
        self.time_decay = time_decay
        self.alphas = alphas
        self._pipe: Pipeline | None = None
        self.feature_names_: list[str] = []
        self.coef_: np.ndarray | None = None
        self.coef_df_: pd.DataFrame | None = None
        self.residuals_: np.ndarray | None = None
        self._feat_df_: pd.DataFrame | None = None
        self.is_fitted: bool = False

    def fit(self, feat_df: pd.DataFrame) -> "MarketPricingModel":
        """
        Fit on a feature DataFrame that includes odds columns.
        """
        required = ["odds_home", "odds_draw", "odds_away"]
        missing = [c for c in required if c not in feat_df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Drop rows with missing odds
        mask = feat_df[required].notna().all(axis=1)
        df = feat_df[mask].copy()

        implied = _market_implied_probs(df)
        y = _logit(implied["market_p_home"].values)

        cols = _feature_cols(df)
        self.feature_names_ = cols
        X = df[cols].values
        w = _time_weights(df["date"], self.time_decay)

        # RidgeCV picks best alpha
        self._pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale",  StandardScaler()),
            ("model",  RidgeCV(alphas=self.alphas, cv=5)),
        ])
        self._pipe.fit(X, y, model__sample_weight=w)

        # Compute residuals on training data
        y_hat = self._pipe.predict(X)
        self.residuals_ = y - y_hat

        # Store coefficients with feature names
        coef = self._pipe.named_steps["model"].coef_
        self.coef_ = coef
        self.coef_df_ = pd.DataFrame({
            "feature":     cols,
            "coef":        coef,
            "abs_coef":    np.abs(coef),
        }).sort_values("abs_coef", ascending=False).reset_index(drop=True)

        # Cache for report generation
        self._feat_df_ = df.copy()
        self._feat_df_["market_p_home"] = implied["market_p_home"].values
        self._feat_df_["y_logit"]       = y
        self._feat_df_["y_hat_logit"]   = y_hat
        self._feat_df_["residual"]      = self.residuals_

        self.is_fitted = True
        return self

    def predict_market_logit(self, feat_df: pd.DataFrame) -> np.ndarray:
        """Predict the market's logit home prob for new fixtures."""
        X = feat_df[self.feature_names_].values
        return self._pipe.predict(X)

    def score_fixtures(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        """
        For a set of upcoming fixtures (no odds required), predict what the
        market SHOULD price based on fundamentals, expressed as fair prob.

        Returns DataFrame with model_p_home plus fixture info.
        """
        y_logit_hat = self.predict_market_logit(feat_df)
        p_home = 1.0 / (1.0 + np.exp(-y_logit_hat))
        p_away = 1.0 - p_home   # two-outcome simplification

        out = feat_df[["date", "home_team", "away_team"]].copy().reset_index(drop=True)
        out["model_p_home_vs_not"] = p_home
        out["model_p_away_vs_not"] = p_away
        return out

    def market_inefficiency_report(self, top_n: int = 20) -> pd.DataFrame:
        """
        Return the top-N matches where model's implied price most disagrees
        with the market.  Large positive residual = model expects home to win
        more often than market implies.
        """
        if self._feat_df_ is None:
            raise RuntimeError("Call .fit() first.")
        df = self._feat_df_.copy()
        df["edge_home"] = df["residual"]   # positive = home underpriced
        return (df[["date", "home_team", "away_team",
                     "market_p_home", "residual", "edge_home"]]
                .assign(market_odds_home=lambda d: 1.0 / d["market_p_home"].clip(1e-4))
                .sort_values("residual", ascending=False)
                .head(top_n)
                .reset_index(drop=True))

    def feature_sensitivity(self) -> pd.DataFrame:
        """
        Which features most influence bookmaker pricing?
        High absolute coefficient = market is highly sensitive to this feature.
        Near-zero coefficient = market ignores this feature (potential edge if it predicts outcomes).
        """
        return self.coef_df_.copy()

    def print_report(self):
        if not self.is_fitted:
            print("Not fitted yet.")
            return
        alpha = self._pipe.named_steps["model"].alpha_
        r2_train = self._pipe.score(
            self._feat_df_[self.feature_names_].values,
            self._feat_df_["y_logit"].values
        )
        print(f"\n{'='*60}")
        print(f"  MARKET PRICING MODEL REPORT")
        print(f"{'='*60}")
        print(f"  Best alpha (RidgeCV)  : {alpha:.4f}")
        print(f"  R² on logit(p_home)  : {r2_train:.4f}")
        print(f"  Residual std         : {self.residuals_.std():.4f}")
        print(f"\n  Top features by market sensitivity:")
        print(self.coef_df_.head(15).to_string(index=False))
        print(f"\n  Top mispriced matches (model says home undervalued):")
        print(self.market_inefficiency_report(10).to_string(index=False))
        print(f"{'='*60}")
