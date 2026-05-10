"""
Feature engineering from raw match data.

All features are computed using ONLY data available BEFORE each match —
no lookahead bias.  The function build_feature_matrix() is the main entry point.

Feature groups
--------------
overall_form      Rolling stats over last N games (all venues)
venue_form        Rolling stats over last N home/away games only
recent_form       Last 5 games (for trend / momentum detection)
ewma              Exponentially weighted averages of goals and points
fatigue           Days since last match, matches in last 14/30 days
h2h               Head-to-head record between specific team pair

Each feature is prefixed:
    h_  →  home team context
    a_  →  away team context
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional


# ── Internal helpers ───────────────────────────────────────────────────────────

def _expand_to_team_records(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert match-level DataFrame to team-level records.
    Each match → two rows (one per team).

    Columns: date, team, opponent, venue, goals_for, goals_against,
             result (W/D/L), points
    """
    home = pd.DataFrame({
        "date":          df["date"].values,
        "team":          df["home_team"].values,
        "opponent":      df["away_team"].values,
        "venue":         "home",
        "goals_for":     df["home_goals"].values,
        "goals_against": df["away_goals"].values,
    })
    away = pd.DataFrame({
        "date":          df["date"].values,
        "team":          df["away_team"].values,
        "opponent":      df["home_team"].values,
        "venue":         "away",
        "goals_for":     df["away_goals"].values,
        "goals_against": df["home_goals"].values,
    })
    records = pd.concat([home, away], ignore_index=True)

    gf, ga = records["goals_for"].values, records["goals_against"].values
    records["result"] = np.where(gf > ga, "W", np.where(gf == ga, "D", "L"))
    records["points"] = np.where(gf > ga, 3, np.where(gf == ga, 1, 0))
    records["goal_diff"] = gf - ga

    return records.sort_values(["team", "date"]).reset_index(drop=True)


def _team_stats(team_history: pd.DataFrame,
                before_date,
                n: int,
                venue_filter: Optional[str] = None,
                alpha: float = 0.3) -> dict:
    """
    Compute rolling statistics for one team from their history up to before_date.

    Parameters
    ----------
    team_history   : All records for this team (sorted by date).
    before_date    : Exclude matches on or after this date.
    n              : Number of recent games to use.
    venue_filter   : 'home' | 'away' | None (all venues).
    alpha          : EWMA decay factor (only used when venue_filter is None).
    """
    hist = team_history[team_history["date"] < before_date]
    if venue_filter:
        hist = hist[hist["venue"] == venue_filter]
    hist = hist.tail(n)

    if len(hist) == 0:
        return {}   # caller will fill with NaN / league average

    gf = hist["goals_for"].values
    ga = hist["goals_against"].values
    pts = hist["points"].values
    wins = (hist["result"] == "W").values.astype(float)
    draws = (hist["result"] == "D").values.astype(float)

    stats = {
        "avg_gf":   float(gf.mean()),
        "avg_ga":   float(ga.mean()),
        "avg_gd":   float((gf - ga).mean()),
        "win_rate": float(wins.mean()),
        "draw_rate": float(draws.mean()),
        "loss_rate": float(1 - wins.mean() - draws.mean()),
        "ppg":      float(pts.mean()),
        "n_games":  len(hist),
    }

    # Clean sheet rate and failure to score
    stats["clean_sheet_rate"] = float((ga == 0).mean())
    stats["scored_rate"]      = float((gf > 0).mean())

    return stats


def _ewma_stats(team_history: pd.DataFrame, before_date, alpha: float = 0.3) -> dict:
    """Exponentially weighted averages (most recent has highest weight)."""
    hist = team_history[team_history["date"] < before_date]
    if len(hist) == 0:
        return {"ewma_gf": np.nan, "ewma_ga": np.nan, "ewma_pts": np.nan}

    gf  = hist["goals_for"].values.astype(float)
    ga  = hist["goals_against"].values.astype(float)
    pts = hist["points"].values.astype(float)

    def _ewma(arr):
        w = np.array([(1 - alpha) ** i for i in range(len(arr) - 1, -1, -1)])
        w /= w.sum()
        return float((arr * w).sum())

    return {
        "ewma_gf":  _ewma(gf),
        "ewma_ga":  _ewma(ga),
        "ewma_pts": _ewma(pts),
    }


def _form_trend(team_history: pd.DataFrame, before_date,
                n_recent: int = 5, n_long: int = 10) -> dict:
    """Trend = recent_ppg − long_ppg.  Positive = improving form."""
    hist = team_history[team_history["date"] < before_date]
    recent = hist.tail(n_recent)["points"].mean() if len(hist) >= 1 else np.nan
    long_  = hist.tail(n_long)["points"].mean()   if len(hist) >= 1 else np.nan
    return {"form_trend": float(recent - long_) if not np.isnan(recent) and not np.isnan(long_) else np.nan}


def _fatigue(team_history: pd.DataFrame, before_date) -> dict:
    """Days since last match + matches in last 14 / 30 days."""
    hist = team_history[team_history["date"] < before_date]
    if len(hist) == 0:
        return {"days_rest": np.nan, "games_l14": np.nan, "games_l30": np.nan}

    last_match = hist["date"].iloc[-1]
    days_rest  = (before_date - last_match).days

    cutoff14 = before_date - pd.Timedelta(days=14)
    cutoff30 = before_date - pd.Timedelta(days=30)
    games_l14 = int((hist["date"] >= cutoff14).sum())
    games_l30 = int((hist["date"] >= cutoff30).sum())

    return {"days_rest": days_rest, "games_l14": games_l14, "games_l30": games_l30}


def _h2h_stats(team_a: str, team_b: str,
               all_records: pd.DataFrame,
               before_date,
               n: int = 8) -> dict:
    """
    Head-to-head record between team_a (as home) and team_b.
    team_a_win_rate = fraction of past meetings won by team_a.
    """
    h2h = all_records[
        ((all_records["team"] == team_a) & (all_records["opponent"] == team_b)) &
        (all_records["date"] < before_date)
    ].tail(n)

    if len(h2h) == 0:
        return {"h2h_win_rate": np.nan, "h2h_avg_gf": np.nan, "h2h_avg_ga": np.nan}

    return {
        "h2h_win_rate": float((h2h["result"] == "W").mean()),
        "h2h_avg_gf":   float(h2h["goals_for"].mean()),
        "h2h_avg_ga":   float(h2h["goals_against"].mean()),
    }


# ── Public API ─────────────────────────────────────────────────────────────────

def build_feature_matrix(
    df: pd.DataFrame,
    n_overall: int = 10,
    n_venue: int = 6,
    n_recent: int = 5,
    ewma_alpha: float = 0.3,
    min_games: int = 3,
) -> pd.DataFrame:
    """
    Build the full feature matrix for all matches in df.

    For each match, computes features using ONLY prior matches (no lookahead).
    Rows with insufficient history (< min_games for either team) are dropped.

    Returns a DataFrame indexed to match df.index, with feature columns
    prefixed h_ (home team) and a_ (away team).

    Feature groups
    --------------
    h_/a_ + avg_gf_10, avg_ga_10, win_rate_10, ppg_10       overall last 10
    h_/a_ + avg_gf_6v, avg_ga_6v, win_rate_6v, ppg_6v       venue-specific last 6
    h_/a_ + avg_gf_5,  avg_ga_5,  win_rate_5,  ppg_5        recent last 5
    h_/a_ + ewma_gf, ewma_ga, ewma_pts                      EWMA
    h_/a_ + form_trend                                       trend
    h_/a_ + days_rest, games_l14, games_l30                  fatigue
    h2h_win_rate, h2h_avg_gf, h2h_avg_ga                    head-to-head
    """
    df = df.sort_values("date").reset_index(drop=True)
    records = _expand_to_team_records(df)

    # Per-team lookup table
    by_team = {t: grp.reset_index(drop=True)
               for t, grp in records.groupby("team")}

    rows = []
    valid_idx = []

    for idx, row in df.iterrows():
        date      = row["date"]
        home_team = row["home_team"]
        away_team = row["away_team"]

        h_rec = by_team.get(home_team, pd.DataFrame())
        a_rec = by_team.get(away_team, pd.DataFrame())

        h_prior = h_rec[h_rec["date"] < date] if len(h_rec) else pd.DataFrame()
        a_prior = a_rec[a_rec["date"] < date] if len(a_rec) else pd.DataFrame()

        if len(h_prior) < min_games or len(a_prior) < min_games:
            continue   # not enough history

        feats = {}

        # ── Overall form (last n_overall) ─────────────────────────────────────
        for prefix, team_h in [("h_", h_rec), ("a_", a_rec)]:
            s = _team_stats(team_h, date, n_overall)
            for k, v in s.items():
                feats[f"{prefix}{k}_{n_overall}"] = v

        # ── Venue-specific form (last n_venue home/away games) ────────────────
        h_venue = "home"
        a_venue = "away"
        for prefix, team_h, venue in [("h_", h_rec, h_venue), ("a_", a_rec, a_venue)]:
            s = _team_stats(team_h, date, n_venue, venue_filter=venue)
            for k, v in s.items():
                feats[f"{prefix}{k}_{n_venue}v"] = v

        # ── Recent form (last n_recent) ───────────────────────────────────────
        for prefix, team_h in [("h_", h_rec), ("a_", a_rec)]:
            s = _team_stats(team_h, date, n_recent)
            for k, v in s.items():
                feats[f"{prefix}{k}_{n_recent}"] = v

        # ── EWMA ──────────────────────────────────────────────────────────────
        for prefix, team_h in [("h_", h_rec), ("a_", a_rec)]:
            s = _ewma_stats(team_h, date, alpha=ewma_alpha)
            for k, v in s.items():
                feats[f"{prefix}{k}"] = v

        # ── Form trend ────────────────────────────────────────────────────────
        for prefix, team_h in [("h_", h_rec), ("a_", a_rec)]:
            s = _form_trend(team_h, date, n_recent, n_overall)
            for k, v in s.items():
                feats[f"{prefix}{k}"] = v

        # ── Fatigue ───────────────────────────────────────────────────────────
        for prefix, team_h in [("h_", h_rec), ("a_", a_rec)]:
            s = _fatigue(team_h, date)
            for k, v in s.items():
                feats[f"{prefix}{k}"] = v

        # ── Head-to-head ──────────────────────────────────────────────────────
        h2h = _h2h_stats(home_team, away_team, records, date)
        feats.update(h2h)

        # ── Differential features (home − away) ───────────────────────────────
        # These directly encode relative strength and are often the most predictive
        feats["diff_avg_gf"]     = feats.get(f"h_avg_gf_{n_overall}", np.nan) - feats.get(f"a_avg_gf_{n_overall}", np.nan)
        feats["diff_avg_ga"]     = feats.get(f"h_avg_ga_{n_overall}", np.nan) - feats.get(f"a_avg_ga_{n_overall}", np.nan)
        feats["diff_ppg"]        = feats.get(f"h_ppg_{n_overall}", np.nan)    - feats.get(f"a_ppg_{n_overall}", np.nan)
        feats["diff_ewma_gf"]    = feats.get("h_ewma_gf", np.nan) - feats.get("a_ewma_gf", np.nan)
        feats["diff_form_trend"] = feats.get("h_form_trend", np.nan) - feats.get("a_form_trend", np.nan)

        rows.append(feats)
        valid_idx.append(idx)

    feat_df = pd.DataFrame(rows, index=valid_idx)

    # Align with original df
    result = df.loc[valid_idx].copy().reset_index(drop=True)
    feat_df = feat_df.reset_index(drop=True)
    return pd.concat([result, feat_df], axis=1)
