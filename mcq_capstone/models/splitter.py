"""
Temporal data splitting for time-series model validation.

Enforces strict temporal ordering to prevent leakage:
  - TRAIN  : oldest data, used to fit models and imputer/scaler
  - VAL    : middle period, used for hyperparameter tuning only
  - TEST   : most recent data, locked away — evaluated ONCE at the end

TimeSeriesSplit folds are constructed only from TRAIN+VAL to tune HPs.
The TEST set is NEVER used during tuning.

Leakage controls
----------------
1. Hard date cutoffs — no random shuffling.
2. Imputer / scaler fitted on train fold only, then applied to val/test.
3. Feature matrix pre-computed with strict `date < match_date` — see engineer.py.
4. `get_cv_splits` yields (train_idx, val_idx) pairs where val_idx > train_idx
   in time, never the other way.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import TimeSeriesSplit
from typing import Iterator


@dataclass
class TemporalSplit:
    """Indices and date ranges for a single train/val/test split."""
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def summary(self) -> str:
        return (
            f"Train : {self.train_start.date()} → {self.train_end.date()} "
            f"({len(self.train_idx)} rows)\n"
            f"Val   : {self.val_start.date()} → {self.val_end.date()} "
            f"({len(self.val_idx)} rows)\n"
            f"Test  : {self.test_start.date()} → {self.test_end.date()} "
            f"({len(self.test_idx)} rows)"
        )


class TemporalSplitter:
    """
    Hard temporal split into train / val / test sets.

    Parameters
    ----------
    test_months   : Number of months to hold out as test set (default 4).
    val_months    : Number of months for validation (default 3).
    n_cv_folds    : Number of TimeSeriesSplit folds within train+val (default 5).

    Usage
    -----
    splitter = TemporalSplitter(test_months=4, val_months=3)
    split = splitter.split(feat_df)
    train_df  = feat_df.iloc[split.train_idx]
    val_df    = feat_df.iloc[split.val_idx]
    test_df   = feat_df.iloc[split.test_idx]   # ← LOCK UNTIL FINAL EVAL

    cv_folds = splitter.get_cv_splits(feat_df)  # for HP tuning
    """

    def __init__(
        self,
        test_months: int = 4,
        val_months: int = 3,
        n_cv_folds: int = 5,
    ):
        self.test_months = test_months
        self.val_months = val_months
        self.n_cv_folds = n_cv_folds

    def split(self, df: pd.DataFrame) -> TemporalSplit:
        """
        Create a single hard train/val/test split.

        df must have a 'date' column.  Rows must be sorted (or will be).
        """
        df = df.sort_values("date").reset_index(drop=True)
        dates = pd.to_datetime(df["date"])

        max_date = dates.max()
        test_start  = max_date - pd.DateOffset(months=self.test_months)
        val_start   = test_start - pd.DateOffset(months=self.val_months)

        train_mask = dates < val_start
        val_mask   = (dates >= val_start) & (dates < test_start)
        test_mask  = dates >= test_start

        train_idx = np.where(train_mask)[0]
        val_idx   = np.where(val_mask)[0]
        test_idx  = np.where(test_mask)[0]

        return TemporalSplit(
            train_idx   = train_idx,
            val_idx     = val_idx,
            test_idx    = test_idx,
            train_start = dates[train_idx[0]]  if len(train_idx) else pd.NaT,
            train_end   = dates[train_idx[-1]] if len(train_idx) else pd.NaT,
            val_start   = dates[val_idx[0]]    if len(val_idx)   else pd.NaT,
            val_end     = dates[val_idx[-1]]   if len(val_idx)   else pd.NaT,
            test_start  = dates[test_idx[0]]   if len(test_idx)  else pd.NaT,
            test_end    = dates[test_idx[-1]]  if len(test_idx)  else pd.NaT,
        )

    def get_cv_splits(
        self,
        df: pd.DataFrame,
        use_trainval: bool = True,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Return (train_idx, val_idx) pairs for TimeSeriesSplit cross-validation.

        By default operates on TRAIN+VAL rows only (never touches TEST).
        All indices are relative to the ORIGINAL df.

        Parameters
        ----------
        use_trainval : If True (default), exclude the test period from CV entirely.
        """
        df = df.sort_values("date").reset_index(drop=True)
        split = self.split(df)

        if use_trainval:
            eligible = np.concatenate([split.train_idx, split.val_idx])
        else:
            eligible = np.arange(len(df))

        tscv = TimeSeriesSplit(n_splits=self.n_cv_folds)
        folds = []
        for tr, va in tscv.split(eligible):
            folds.append((eligible[tr], eligible[va]))
        return folds

    def print_split_info(self, df: pd.DataFrame) -> None:
        split = self.split(df)
        print(split.summary())
        total = len(df)
        print(f"\nSplit fractions:")
        print(f"  Train : {len(split.train_idx)/total:.1%}")
        print(f"  Val   : {len(split.val_idx)/total:.1%}")
        print(f"  Test  : {len(split.test_idx)/total:.1%}  ← LOCKED")
