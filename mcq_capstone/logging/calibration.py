"""
Calibration analysis for probability predictions.

A well-calibrated model: when it predicts 60%, outcomes occur ~60% of the time.
Overconfident model: predicts 60%, outcomes occur 40%.
Underconfident model: predicts 60%, outcomes occur 80%.

Metrics
-------
Brier Score   = mean((p_i - o_i)^2)          lower is better, 0 = perfect
Log Score     = mean(log(p_i))                higher is better, 0 = perfect
Reliability   = mean((p_bucket - freq)^2)     lower is better

Reliability diagram: plot predicted probability vs actual frequency per bucket.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional


class CalibrationAnalyzer:
    """
    Accumulates (predicted probability, actual outcome) pairs and computes
    calibration metrics.

    Usage
    -----
    cal = CalibrationAnalyzer()
    cal.add(0.65, True)   # predicted 65%, outcome was win
    cal.add(0.30, False)
    ...
    print(cal.brier_score())
    df = cal.reliability_table()
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self._probs: list[float] = []
        self._outcomes: list[int] = []   # 1 = correct, 0 = wrong

    def add(self, prob: float, outcome: bool):
        """Record a single prediction."""
        self._probs.append(float(prob))
        self._outcomes.append(int(outcome))

    def add_batch(self, probs, outcomes):
        """Add arrays of predictions and outcomes."""
        for p, o in zip(probs, outcomes):
            self.add(p, bool(o))

    @property
    def _arrays(self):
        return np.array(self._probs), np.array(self._outcomes, dtype=float)

    def brier_score(self) -> float:
        """
        Brier score: mean squared error of probability forecasts.
        Range [0, 1]; lower = better.  Naive baseline (always predict 0.5) = 0.25.
        """
        p, o = self._arrays
        if len(p) == 0:
            return float("nan")
        return float(np.mean((p - o) ** 2))

    def log_score(self) -> float:
        """
        Mean log loss.  Range (−∞, 0]; 0 = perfect.
        """
        p, o = self._arrays
        if len(p) == 0:
            return float("nan")
        p_clipped = np.clip(p, 1e-10, 1.0 - 1e-10)
        return float(np.mean(o * np.log(p_clipped) + (1 - o) * np.log(1 - p_clipped)))

    def reliability_table(self) -> pd.DataFrame:
        """
        Bin predictions into `n_bins` equally-spaced buckets.
        Returns a DataFrame with columns:
            bin_centre, n_predictions, mean_pred_prob, actual_freq, reliability
        """
        p, o = self._arrays
        if len(p) == 0:
            return pd.DataFrame()

        bins = np.linspace(0.0, 1.0, self.n_bins + 1)
        rows = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (p >= lo) & (p < hi)
            if lo == bins[-2]:
                mask = (p >= lo) & (p <= hi)   # include 1.0 in last bin
            n = mask.sum()
            if n == 0:
                continue
            mean_p = float(p[mask].mean())
            freq   = float(o[mask].mean())
            rows.append({
                "bin_centre": round((lo + hi) / 2, 2),
                "n": int(n),
                "mean_pred": round(mean_p, 4),
                "actual_freq": round(freq, 4),
                "error": round(mean_p - freq, 4),
            })
        return pd.DataFrame(rows)

    def summary(self) -> dict:
        tbl = self.reliability_table()
        rel = float(np.average(
            (tbl["mean_pred"] - tbl["actual_freq"]) ** 2,
            weights=tbl["n"]
        )) if not tbl.empty else float("nan")

        return {
            "n_predictions": len(self._probs),
            "brier_score": round(self.brier_score(), 5),
            "log_score": round(self.log_score(), 5),
            "reliability": round(rel, 6),
        }

    def print_summary(self):
        s = self.summary()
        print(f"  Calibration ({s['n_predictions']} predictions)")
        print(f"    Brier score  : {s['brier_score']:.4f}  (lower = better; baseline ~0.25)")
        print(f"    Log score    : {s['log_score']:.4f}  (higher = better; max = 0)")
        print(f"    Reliability  : {s['reliability']:.6f}  (lower = better; 0 = perfect)")

    def plot(self, title: str = "Calibration Plot", save_path: Optional[str] = None):
        """
        Draw reliability diagram (requires matplotlib).
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed — skipping calibration plot")
            return

        tbl = self.reliability_table()
        if tbl.empty:
            return

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", lw=1)
        ax.scatter(tbl["mean_pred"], tbl["actual_freq"],
                   s=tbl["n"].clip(5, 500), alpha=0.7, label="Actual")
        ax.plot(tbl["mean_pred"], tbl["actual_freq"], "b-", alpha=0.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Actual frequency")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
        else:
            plt.show()
        plt.close(fig)
