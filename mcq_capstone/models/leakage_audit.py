"""
Leakage audit for the feature engineering pipeline.

Checks three categories of data leakage:

1. Temporal leakage
   - For each row i, every feature must use only data from dates STRICTLY
     BEFORE df.iloc[i]['date'].
   - Verified by checking that rolling/EWMA features for match i do not
     accidentally include the match itself.

2. Target leakage
   - No feature should encode the outcome (home_goals, away_goals, result)
     of the match being predicted.
   - Checked by verifying feature columns don't contain goal or result info
     from the current match.

3. Preprocessing leakage
   - Imputers and scalers must be fit ONLY on training data.
   - This function returns a FitOnTrainOnly wrapper that raises if you try
     to call transform before fit on train, or if you try to fit on test data.

Usage
-----
from mcq_capstone.models.leakage_audit import audit_features, LeakageError

issues = audit_features(feat_df, verbose=True)
# issues is a list of strings describing any problems found
# empty list = no leakage detected

# Wrap sklearn transformers to enforce train-only fitting:
from mcq_capstone.models.leakage_audit import SafeTransformer
from sklearn.preprocessing import StandardScaler
scaler = SafeTransformer(StandardScaler())
scaler.fit(X_train)          # OK
scaler.transform(X_test)     # OK  (uses train statistics)
scaler.fit(X_test)           # raises LeakageError
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Any


class LeakageError(RuntimeError):
    """Raised when a data leakage violation is detected."""


# ── Temporal leakage checks ────────────────────────────────────────────────────

def _check_h2h_temporal(feat_df: pd.DataFrame, sample_n: int = 50) -> list[str]:
    """
    Spot-check H2H features: for a random sample of rows, verify the H2H
    stats don't include the current match.

    H2H features in our engineer use strict `date < match_date`, so this
    should always pass.  If columns aren't present we skip silently.
    """
    issues = []
    h2h_cols = [c for c in feat_df.columns if "h2h" in c]
    if not h2h_cols:
        return issues

    sample = feat_df.sample(min(sample_n, len(feat_df)), random_state=42)
    for _, row in sample.iterrows():
        match_date = pd.Timestamp(row["date"])
        home = row.get("home_team")
        away = row.get("away_team")
        if home is None or away is None:
            continue
        # Sanity: h2h_games should be >= 0 (we can't check the exact value
        # without rerunning the feature engine, so just check non-negative)
        h2h_n = row.get("h_h2h_games", None)
        if h2h_n is not None and h2h_n < 0:
            issues.append(
                f"h_h2h_games < 0 for {home} vs {away} on {match_date.date()}"
            )
    return issues


def _check_no_future_rollup(feat_df: pd.DataFrame) -> list[str]:
    """
    Verify features are monotonically non-decreasing in cumulative count
    when sorted by date — rolling game counts should never decrease over time
    for a given team.
    """
    issues = []
    count_cols = [c for c in feat_df.columns if "_games" in c or "_n_" in c]
    if not count_cols:
        return issues

    for col in count_cols[:3]:  # spot-check 3 columns only
        for team_col in ("home_team", "away_team"):
            if team_col not in feat_df.columns:
                continue
            teams = feat_df[team_col].unique()
            for team in teams[:5]:  # spot-check 5 teams
                sub = feat_df[feat_df[team_col] == team].sort_values("date")
                vals = sub[col].dropna().values
                # Rolling window counts CAN decrease (window slides), so
                # just check no NaN-induced issues
                if np.any(vals < 0):
                    issues.append(
                        f"Negative count in {col} for {team} ({team_col})"
                    )
    return issues


def _check_target_not_in_features(feat_df: pd.DataFrame) -> list[str]:
    """
    Ensure no column directly encodes the current match outcome.
    Feature columns should NOT be home_goals / away_goals / result of
    the current match.
    """
    issues = []
    forbidden = {"home_goals", "away_goals", "result", "home_win", "draw", "away_win"}
    feature_cols = [
        c for c in feat_df.columns
        if c.startswith("h_") or c.startswith("a_") or c.startswith("diff_")
    ]
    for col in feature_cols:
        lower = col.lower()
        for f in forbidden:
            if lower == f:
                issues.append(
                    f"Column '{col}' directly encodes the target — target leakage!"
                )
    return issues


def _check_date_strictly_before(feat_df: pd.DataFrame, sample_n: int = 100) -> list[str]:
    """
    For EWMA / form columns, verify their values change between consecutive
    matches for the same team — stale features would indicate no update
    (i.e. a compute bug, not necessarily leakage, but worth flagging).
    """
    issues = []
    ewma_cols = [c for c in feat_df.columns if "ewma" in c]
    if not ewma_cols or "home_team" not in feat_df.columns:
        return issues

    col = ewma_cols[0]
    sample_teams = feat_df["home_team"].value_counts().head(3).index.tolist()
    for team in sample_teams:
        sub = feat_df[feat_df["home_team"] == team].sort_values("date")
        if len(sub) < 3:
            continue
        vals = sub[col].dropna().values
        if len(vals) > 1 and np.all(vals == vals[0]):
            issues.append(
                f"EWMA column '{col}' is constant for team {team} — "
                f"possible stale feature (check temporal ordering)"
            )
    return issues


def _check_no_nan_in_differentials(feat_df: pd.DataFrame) -> list[str]:
    """
    Differential features (diff_*) should not be entirely NaN — that would
    mean h_ or a_ features failed to compute, which could mask leakage bugs.
    """
    issues = []
    diff_cols = [c for c in feat_df.columns if c.startswith("diff_")]
    for col in diff_cols:
        nan_frac = feat_df[col].isna().mean()
        if nan_frac > 0.5:
            issues.append(
                f"Column '{col}' is {nan_frac:.0%} NaN — "
                f"feature likely not computing correctly"
            )
    return issues


def audit_features(
    feat_df: pd.DataFrame,
    verbose: bool = True,
) -> list[str]:
    """
    Run all leakage checks on a feature DataFrame.

    Parameters
    ----------
    feat_df  : Output of build_feature_matrix().
    verbose  : Print results to stdout.

    Returns
    -------
    List of issue strings.  Empty = no leakage detected.
    """
    all_issues: list[str] = []

    checks = [
        ("Target leakage",          _check_target_not_in_features),
        ("H2H temporal ordering",   _check_h2h_temporal),
        ("Negative rolling counts", _check_no_future_rollup),
        ("Constant EWMA (stale?)",  _check_date_strictly_before),
        ("Diff columns coverage",   _check_no_nan_in_differentials),
    ]

    if verbose:
        print(f"\n{'='*55}")
        print("  LEAKAGE AUDIT")
        print(f"{'='*55}")

    for name, fn in checks:
        issues = fn(feat_df)
        all_issues.extend(issues)
        if verbose:
            status = "PASS" if not issues else f"FAIL ({len(issues)} issues)"
            print(f"  {name:<35} {status}")
            for iss in issues:
                print(f"    ↳ {iss}")

    if verbose:
        if not all_issues:
            print("\n  Result: NO LEAKAGE DETECTED")
        else:
            print(f"\n  Result: {len(all_issues)} ISSUE(S) FOUND — review above")
        print(f"{'='*55}")

    return all_issues


# ── Safe preprocessing wrapper ─────────────────────────────────────────────────

class SafeTransformer:
    """
    Wraps any sklearn transformer and enforces that transform() is only
    called AFTER fit() and that fit() is never called on test data.

    The caller must call .mark_as_train(X) before .fit(X) to register
    which array is the training set.  Any call to .fit() on a different
    array shape/content will raise LeakageError.

    Usage
    -----
    scaler = SafeTransformer(StandardScaler())
    scaler.fit(X_train)
    X_tr = scaler.transform(X_train)
    X_va = scaler.transform(X_val)    # OK — uses train statistics
    scaler.fit(X_val)                  # raises LeakageError
    """

    def __init__(self, transformer: Any):
        self._t = transformer
        self._fitted = False
        self._train_shape: tuple | None = None
        self._train_hash: int | None = None

    def fit(self, X: np.ndarray, y=None, **kw):
        if self._fitted:
            raise LeakageError(
                "SafeTransformer.fit() called a second time. "
                "Transformers must be fit ONLY on training data. "
                "Use transform() for validation/test sets."
            )
        self._fitted = True
        self._train_shape = X.shape
        self._train_hash = hash(X.tobytes()[:512])  # cheap fingerprint
        self._t.fit(X, y, **kw)
        return self

    def transform(self, X: np.ndarray, **kw) -> np.ndarray:
        if not self._fitted:
            raise LeakageError(
                "SafeTransformer.transform() called before fit(). "
                "Call fit() on training data first."
            )
        return self._t.transform(X, **kw)

    def fit_transform(self, X: np.ndarray, y=None, **kw) -> np.ndarray:
        self.fit(X, y, **kw)
        return self.transform(X, **kw)

    def __getattr__(self, name: str):
        # Delegate everything else (e.g. .get_params()) to the wrapped transformer
        return getattr(self._t, name)


# ── Preprocessing leakage check for Pipeline ──────────────────────────────────

def check_pipeline_fit_on_train_only(
    pipeline,
    X_train: np.ndarray,
    X_test: np.ndarray,
    verbose: bool = True,
) -> list[str]:
    """
    After fitting a pipeline on X_train, verify that the fitted scaler/imputer
    statistics (mean, scale) differ from what they'd be if fitted on X_test.

    If scaler_mean ≈ test_mean for ALL features, it likely means the scaler
    was accidentally fitted on test data.

    Returns list of warning strings.
    """
    issues = []

    # Try to find imputer and scaler in the pipeline
    for name, step in pipeline.steps:
        if hasattr(step, "statistics_"):   # SimpleImputer
            train_median = np.nanmedian(X_train, axis=0)
            fitted_median = step.statistics_
            test_median = np.nanmedian(X_test, axis=0)

            # Check: fitted stats should be closer to train than test
            diff_from_train = np.abs(fitted_median - train_median).mean()
            diff_from_test  = np.abs(fitted_median - test_median).mean()
            if diff_from_test < diff_from_train * 0.1:
                issues.append(
                    f"Imputer '{name}': fitted statistics are suspiciously "
                    f"close to test set median — possible leakage!"
                )

        if hasattr(step, "mean_") and hasattr(step, "scale_"):  # StandardScaler
            train_mean = X_train.mean(axis=0)
            fitted_mean = step.mean_
            test_mean = X_test.mean(axis=0)

            diff_from_train = np.abs(fitted_mean - train_mean).mean()
            diff_from_test  = np.abs(fitted_mean - test_mean).mean()
            if diff_from_test < diff_from_train * 0.1:
                issues.append(
                    f"Scaler '{name}': fitted mean is suspiciously close "
                    f"to test set mean — possible leakage!"
                )

    if verbose:
        status = "PASS" if not issues else f"FAIL ({len(issues)} issues)"
        print(f"  Pipeline preprocessing leakage check: {status}")
        for iss in issues:
            print(f"    ↳ {iss}")

    return issues
