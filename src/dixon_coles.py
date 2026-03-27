"""
Dixon-Coles Model for Soccer Match Prediction.

Reference:
    Dixon, M.J. & Coles, S.G. (1997). Modelling Association Football Scores
    and Inefficiencies in the Football Betting Market.
    Applied Statistics, 46(2), 265-280.

The model assumes:
    - Home goals ~ Poisson(λ)  where log λ = α_h + β_a + γ
    - Away goals ~ Poisson(μ)  where log μ = α_a + β_h
    - α_i = attack strength of team i
    - β_i = defense strength of team i  (negative = strong)
    - γ   = home advantage (global constant)
    - ρ   = low-score correction (Dixon-Coles τ factor)

Parameters are estimated by maximum likelihood with exponential time-decay
weighting so that recent matches count more than old ones.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson
from typing import Dict, Tuple, Optional


class DixonColesModel:
    """Fitted Dixon-Coles Poisson model for soccer score prediction."""

    def __init__(self, time_decay: float = 0.0065):
        """
        Parameters
        ----------
        time_decay : float
            Exponential decay constant (per day).  0.0065 ≈ half-life ~107 days.
            Set to 0 for uniform weighting (no decay).
        """
        self.time_decay = time_decay
        self.params_: Optional[np.ndarray] = None
        self.teams_: Optional[list] = None
        self.team_idx_: Optional[Dict[str, int]] = None
        self.is_fitted: bool = False
        self._log_lik_final: float = float("nan")

    # ── Core probability mechanics ─────────────────────────────────────────────

    @staticmethod
    def _tau(x: np.ndarray, y: np.ndarray,
             lam: np.ndarray, mu: np.ndarray, rho: float) -> np.ndarray:
        """
        Vectorised Dixon-Coles τ correction for low-scoring scorelines.
        Adjusts joint Poisson mass for (0,0), (1,0), (0,1), (1,1).
        """
        tau = np.ones(len(x), dtype=float)
        m00 = (x == 0) & (y == 0)
        m01 = (x == 0) & (y == 1)
        m10 = (x == 1) & (y == 0)
        m11 = (x == 1) & (y == 1)
        tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
        tau[m01] = 1.0 + lam[m01] * rho
        tau[m10] = 1.0 + mu[m10] * rho
        tau[m11] = 1.0 - rho
        return np.clip(tau, 1e-10, None)

    # ── Log-likelihood ─────────────────────────────────────────────────────────

    def _neg_log_lik(self,
                     params: np.ndarray,
                     hidx: np.ndarray,
                     aidx: np.ndarray,
                     hg: np.ndarray,
                     ag: np.ndarray,
                     w: np.ndarray) -> float:
        n = len(self.teams_)
        atk = params[:n]
        dfn = params[n:2 * n]
        home_adv = params[-2]
        rho = params[-1]

        lam = np.exp(atk[hidx] + dfn[aidx] + home_adv)
        mu = np.exp(atk[aidx] + dfn[hidx])

        # Poisson log-PMF (vectorised, avoids scipy overhead)
        log_p_h = hg * np.log(np.maximum(lam, 1e-10)) - lam - gammaln(hg + 1)
        log_p_a = ag * np.log(np.maximum(mu, 1e-10)) - mu - gammaln(ag + 1)

        tau = self._tau(hg, ag, lam, mu, rho)

        ll = w * (np.log(tau) + log_p_h + log_p_a)
        return -np.sum(ll)

    # ── Fitting ────────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame, current_date=None) -> "DixonColesModel":
        """
        Fit the model to historical match data.

        Parameters
        ----------
        df : DataFrame with columns
             date (datetime64), home_team, away_team, home_goals, away_goals
        current_date : reference date for time decay  (default: df['date'].max())

        Returns self for chaining.
        """
        if current_date is None:
            current_date = df["date"].max()

        self.teams_ = sorted(
            set(df["home_team"].tolist()) | set(df["away_team"].tolist())
        )
        self.team_idx_ = {t: i for i, t in enumerate(self.teams_)}
        n = len(self.teams_)

        hidx = df["home_team"].map(self.team_idx_).values
        aidx = df["away_team"].map(self.team_idx_).values
        hg = df["home_goals"].values.astype(int)
        ag = df["away_goals"].values.astype(int)

        days_ago = (current_date - df["date"]).dt.days.values
        w = np.exp(-self.time_decay * days_ago.astype(float))

        # Initial parameter vector: [attacks×n, defenses×n, home_adv, rho]
        x0 = np.zeros(2 * n + 2)
        x0[-2] = 0.30   # home advantage start
        x0[-1] = -0.10  # rho start

        # Bounds: rho ∈ (-1, 1), home_adv > 0
        bounds = [(None, None)] * (2 * n) + [(0.0, 2.0), (-0.99, 0.99)]

        result = minimize(
            self._neg_log_lik,
            x0,
            args=(hidx, aidx, hg, ag, w),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
        )

        self.params_ = result.x.copy()
        # Centre attack parameters (identifiability)
        mean_atk = self.params_[:n].mean()
        self.params_[:n] -= mean_atk
        self.params_[n:2 * n] += mean_atk  # compensate in defence

        self._log_lik_final = -result.fun
        self.is_fitted = True
        return self

    # ── Prediction ─────────────────────────────────────────────────────────────

    def predict_goals(self, home_team: str, away_team: str) -> Tuple[float, float]:
        """Return (λ, μ): expected goals for home and away team."""
        self._check_fitted()
        n = len(self.teams_)
        h = self.team_idx_[home_team]
        a = self.team_idx_[away_team]
        lam = np.exp(self.params_[h] + self.params_[n + a] + self.params_[-2])
        mu = np.exp(self.params_[a] + self.params_[n + h])
        return float(lam), float(mu)

    def predict_probs(self, home_team: str, away_team: str,
                      max_goals: int = 10) -> Dict[str, float]:
        """
        Return dict with keys 'home', 'draw', 'away', 'lam', 'mu'.
        Probabilities are derived from the full score matrix with DC correction.
        """
        self._check_fitted()
        lam, mu = self.predict_goals(home_team, away_team)
        rho = float(self.params_[-1])

        g = np.arange(max_goals + 1)
        # Outer product of Poisson PMFs
        p_h = poisson.pmf(g, lam)   # shape (max_goals+1,)
        p_a = poisson.pmf(g, mu)
        matrix = np.outer(p_h, p_a)  # matrix[i,j] = P(home=i, away=j)

        # Apply DC correction to low-scoring cells
        for x, y in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            if x <= max_goals and y <= max_goals:
                xarr = np.array([x])
                yarr = np.array([y])
                larr = np.array([lam])
                marr = np.array([mu])
                tau = float(self._tau(xarr, yarr, larr, marr, rho)[0])
                matrix[x, y] *= tau

        total = matrix.sum()
        home_win = float(np.tril(matrix, -1).sum()) / total
        draw = float(np.trace(matrix)) / total
        away_win = float(np.triu(matrix, 1).sum()) / total

        return {"home": home_win, "draw": draw, "away": away_win,
                "lam": lam, "mu": mu}

    def score_matrix(self, home_team: str, away_team: str,
                     max_goals: int = 5) -> pd.DataFrame:
        """
        Return a DataFrame showing P(score = i–j) for i,j in 0..max_goals.
        Rows = home goals, columns = away goals.
        """
        self._check_fitted()
        lam, mu = self.predict_goals(home_team, away_team)
        rho = float(self.params_[-1])
        g = np.arange(max_goals + 1)
        p_h = poisson.pmf(g, lam)
        p_a = poisson.pmf(g, mu)
        matrix = np.outer(p_h, p_a)
        for x, y in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            if x <= max_goals and y <= max_goals:
                xarr = np.array([x])
                yarr = np.array([y])
                tau = float(self._tau(xarr, yarr, np.array([lam]), np.array([mu]), rho)[0])
                matrix[x, y] *= tau
        matrix /= matrix.sum()
        return pd.DataFrame(
            matrix,
            index=pd.Index(range(max_goals + 1), name=f"{home_team} goals"),
            columns=pd.Index(range(max_goals + 1), name=f"{away_team} goals"),
        )

    def get_ratings(self) -> pd.DataFrame:
        """Return attack / defense ratings for all teams, sorted by attack."""
        self._check_fitted()
        n = len(self.teams_)
        return (
            pd.DataFrame(
                {
                    "team": self.teams_,
                    "attack": self.params_[:n],
                    "defense": self.params_[n:2 * n],
                    "overall": self.params_[:n] - self.params_[n:2 * n],
                }
            )
            .sort_values("overall", ascending=False)
            .reset_index(drop=True)
        )

    # ── Internals ──────────────────────────────────────────────────────────────

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call .fit() first.")

    def __repr__(self):
        if self.is_fitted:
            return (f"DixonColesModel(teams={len(self.teams_)}, "
                    f"decay={self.time_decay}, "
                    f"log_lik={self._log_lik_final:.1f})")
        return f"DixonColesModel(decay={self.time_decay}, unfitted)"
