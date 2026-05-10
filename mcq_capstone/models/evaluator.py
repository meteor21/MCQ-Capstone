"""
Model comparison and walk-forward evaluation.

Compares all models head-to-head on the same walk-forward splits:
  - PoissonGoalModel
  - RidgeGoalModel
  - LogisticOutcomeModel
  - GradientBoostModel

Metrics per model:
  - Brier score (lower = better)
  - Log loss
  - ROI when used as betting signal (with fixed EV threshold)
  - Feature importance (where available)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional
from tqdm import tqdm

from ..features.engineer import build_feature_matrix
from ..logging.calibration import CalibrationAnalyzer
from .goal_model import _time_weights


def walk_forward_eval(
    df: pd.DataFrame,
    models: dict,           # {name: model_instance}
    min_train: int = 80,
    train_days: int = 365,
    step: int = 10,         # evaluate every N matches (for speed)
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Walk-forward evaluation of multiple models on the same splits.

    For each test match:
      1. Build feature matrix for all matches before it.
      2. Fit each model on the training window.
      3. Predict outcome probs for the test match.
      4. Record predicted prob and actual outcome.

    Parameters
    ----------
    df      : Feature DataFrame (output of build_feature_matrix).
    models  : Dict of {name: unfitted_model_instance}.
    min_train, train_days : Training window parameters.
    step    : Evaluate every `step` matches (1 = every match, slow).

    Returns
    -------
    Dict of {model_name: DataFrame with columns
             [date, home_team, away_team, p_home, p_draw, p_away,
              actual, brier, log_loss]}
    """
    df = df.sort_values("date").reset_index(drop=True)
    results = {name: [] for name in models}

    test_indices = range(min_train, len(df), step)
    if verbose:
        test_indices = tqdm(test_indices, desc="Walk-forward eval", unit="match")

    for i in test_indices:
        row = df.iloc[i]
        match_date = row["date"]

        # Training window
        window_start = match_date - pd.Timedelta(days=train_days)
        train = df[(df["date"] >= window_start) & (df["date"] < match_date)]
        if len(train) < min_train:
            continue

        actual = _outcome_code(int(row["home_goals"]), int(row["away_goals"]))

        for name, model_proto in models.items():
            # Fresh copy per fold
            import copy
            model = copy.deepcopy(model_proto)
            try:
                model.fit(train)
                probs = model.predict_probs(df.iloc[[i]])
            except Exception:
                continue

            p_home = float(probs["p_home"].iloc[0])
            p_draw = float(probs["p_draw"].iloc[0])
            p_away = float(probs["p_away"].iloc[0])

            p_actual = [p_home, p_draw, p_away][actual]
            brier = (p_home - (actual == 0)) ** 2 + \
                    (p_draw - (actual == 1)) ** 2 + \
                    (p_away - (actual == 2)) ** 2
            ll = float(np.log(max(p_actual, 1e-10)))

            results[name].append({
                "date": match_date,
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "p_home": p_home,
                "p_draw": p_draw,
                "p_away": p_away,
                "actual": actual,
                "brier": brier,
                "log_loss": -ll,
            })

    return {name: pd.DataFrame(v) for name, v in results.items()}


def compare_models(eval_results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Summarise walk-forward results across all models.
    Returns a comparison DataFrame sorted by Brier score.
    """
    rows = []
    for name, df in eval_results.items():
        if df.empty:
            continue
        rows.append({
            "model":        name,
            "n_predictions": len(df),
            "brier_score":  round(float(df["brier"].mean()), 5),
            "log_loss":     round(float(df["log_loss"].mean()), 5),
            "home_accuracy": round(float(((df["actual"] == 0) ==
                                          (df["p_home"] > df[["p_draw","p_away"]].max(axis=1))).mean()), 3),
        })
    return pd.DataFrame(rows).sort_values("brier_score").reset_index(drop=True)


def print_comparison(eval_results: dict, df_summary: Optional[pd.DataFrame] = None):
    if df_summary is None:
        df_summary = compare_models(eval_results)
    print(f"\n{'='*60}")
    print("  MODEL COMPARISON  (walk-forward out-of-sample)")
    print(f"{'='*60}")
    print(df_summary.to_string(index=False))
    print(f"  Note: Brier lower = better  |  baseline (uniform 1/3) = 0.667")
    print(f"{'='*60}")


def _outcome_code(hg: int, ag: int) -> int:
    if hg > ag: return 0   # home
    if hg == ag: return 1  # draw
    return 2               # away
