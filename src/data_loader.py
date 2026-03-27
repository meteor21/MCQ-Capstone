"""
Data loading and preprocessing for soccer match data.

Supports the football-data.co.uk CSV format which provides free historical
data for major European leagues.

Download data from:  https://www.football-data.co.uk/data.php
e.g. Premier League:  https://www.football-data.co.uk/mmz4281/2324/E0.csv

Expected CSV columns (minimum required):
    Date, HomeTeam, AwayTeam, FTHG, FTAG
Optional (for backtesting with odds):
    B365H, B365D, B365A  (Bet365 decimal odds for H/D/A)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from datetime import date, timedelta
import random


# ── Column name aliases ────────────────────────────────────────────────────────
_DATE_COLS = ["Date", "date"]
_HOME_COLS = ["HomeTeam", "home_team", "Home", "home"]
_AWAY_COLS = ["AwayTeam", "away_team", "Away", "away"]
_HGOAL_COLS = ["FTHG", "home_goals", "HG", "hg"]
_AGOAL_COLS = ["FTAG", "away_goals", "AG", "ag"]
_B365H_COLS = ["B365H", "b365h", "OddsH", "odds_home"]
_B365D_COLS = ["B365D", "b365d", "OddsD", "odds_draw"]
_B365A_COLS = ["B365A", "b365a", "OddsA", "odds_away"]


def _pick_col(df: pd.DataFrame, candidates: list) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_csv(path: str | Path) -> pd.DataFrame:
    """
    Load a football-data.co.uk style CSV and return a normalised DataFrame
    with columns:
        date (datetime64), home_team, away_team,
        home_goals (int), away_goals (int),
        odds_home (float, optional), odds_draw (float, optional),
        odds_away (float, optional)
    """
    df = pd.read_csv(path)

    col_map = {}

    date_col = _pick_col(df, _DATE_COLS)
    if date_col is None:
        raise ValueError("No date column found in CSV")
    col_map[date_col] = "date"

    for candidates, target in [
        (_HOME_COLS, "home_team"),
        (_AWAY_COLS, "away_team"),
        (_HGOAL_COLS, "home_goals"),
        (_AGOAL_COLS, "away_goals"),
    ]:
        col = _pick_col(df, candidates)
        if col is None:
            raise ValueError(f"Cannot find column for '{target}' in CSV. "
                             f"Tried: {candidates}")
        col_map[col] = target

    # Optional odds columns
    for candidates, target in [
        (_B365H_COLS, "odds_home"),
        (_B365D_COLS, "odds_draw"),
        (_B365A_COLS, "odds_away"),
    ]:
        col = _pick_col(df, candidates)
        if col:
            col_map[col] = target

    df = df.rename(columns=col_map)
    keep = [c for c in ["date", "home_team", "away_team", "home_goals", "away_goals",
                         "odds_home", "odds_draw", "odds_away"] if c in df.columns]
    df = df[keep].copy()

    # Parse dates — football-data uses DD/MM/YY or DD/MM/YYYY
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals"])

    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)
    df = df.sort_values("date").reset_index(drop=True)

    return df


# ── Synthetic data generator ───────────────────────────────────────────────────

_PL_TEAMS = [
    "Arsenal", "Aston Villa", "Brentford", "Brighton",
    "Burnley", "Chelsea", "Crystal Palace", "Everton",
    "Fulham", "Leeds", "Leicester", "Liverpool",
    "Man City", "Man United", "Newcastle", "Nottm Forest",
    "Southampton", "Tottenham", "West Ham", "Wolves",
]

# Ground-truth latent strengths used to generate synthetic data
_ATTACK_TRUE = {
    "Man City":      0.55, "Arsenal":       0.48, "Liverpool":     0.50,
    "Tottenham":     0.35, "Chelsea":        0.30, "Man United":    0.28,
    "Newcastle":     0.25, "Brighton":       0.22, "West Ham":      0.18,
    "Aston Villa":   0.20, "Brentford":     0.15, "Fulham":        0.10,
    "Crystal Palace":0.05, "Wolves":        0.02, "Leeds":        -0.05,
    "Everton":      -0.08, "Nottm Forest": -0.10, "Leicester":    -0.12,
    "Burnley":      -0.20, "Southampton":  -0.25,
}
_DEFENSE_TRUE = {
    "Man City":     -0.40, "Arsenal":      -0.35, "Liverpool":    -0.32,
    "Chelsea":      -0.28, "Newcastle":    -0.25, "Brighton":     -0.22,
    "Tottenham":    -0.18, "Man United":   -0.16, "Brentford":    -0.15,
    "Fulham":       -0.10, "Aston Villa":  -0.08, "West Ham":     -0.06,
    "Wolves":       -0.04, "Crystal Palace":0.02, "Leeds":         0.05,
    "Everton":       0.08, "Nottm Forest":  0.10, "Leicester":     0.14,
    "Burnley":       0.20, "Southampton":   0.25,
}
_HOME_ADV_TRUE = 0.30
_RHO_TRUE = -0.13


def _poisson_goal(lam: float) -> int:
    return int(np.random.poisson(max(lam, 0.01)))


def _dc_tau(x, y, lam, mu, rho):
    if x == 0 and y == 0:
        return max(1 - lam * mu * rho, 1e-8)
    if x == 0 and y == 1:
        return max(1 + lam * rho, 1e-8)
    if x == 1 and y == 0:
        return max(1 + mu * rho, 1e-8)
    if x == 1 and y == 1:
        return max(1 - rho, 1e-8)
    return 1.0


def _fair_probs(home, away) -> tuple[float, float, float]:
    """Compute exact H/D/A probabilities from the synthetic ground-truth model."""
    lam = np.exp(_ATTACK_TRUE[home] + _DEFENSE_TRUE[away] + _HOME_ADV_TRUE)
    mu = np.exp(_ATTACK_TRUE[away] + _DEFENSE_TRUE[home])
    from scipy.stats import poisson as _pois
    max_g = 10
    h_win = d = a_win = 0.0
    for x in range(max_g + 1):
        for y in range(max_g + 1):
            p = _dc_tau(x, y, lam, mu, _RHO_TRUE) * _pois.pmf(x, lam) * _pois.pmf(y, mu)
            if x > y:
                h_win += p
            elif x == y:
                d += p
            else:
                a_win += p
    total = h_win + d + a_win
    return h_win / total, d / total, a_win / total


def _bookmaker_odds(ph, pd_, pa, margin=0.05, noise=0.06, rng=None):
    """
    Convert true probabilities into bookmaker odds with margin + random noise.
    Some bets will have positive EV (noise swings past fair value).
    """
    if rng is None:
        rng = np.random
    # Apply overround (shrink probs so they sum > 1)
    scale = 1 + margin
    ph_ = ph * scale + rng.uniform(-noise, noise)
    pd__ = pd_ * scale + rng.uniform(-noise, noise)
    pa_ = pa * scale + rng.uniform(-noise, noise)
    # Clip to avoid odds below 1.01
    oh = max(1 / max(ph_, 1e-4), 1.01)
    od = max(1 / max(pd__, 1e-4), 1.01)
    oa = max(1 / max(pa_, 1e-4), 1.01)
    return round(oh, 2), round(od, 2), round(oa, 2)


def generate_sample_data(n_seasons: int = 2, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic Premier League-style match data.

    Each season: 20 teams × 19 home × 2 = 380 matches.
    Scores are drawn from the Dixon-Coles process using the latent strengths
    defined in _ATTACK_TRUE / _DEFENSE_TRUE.
    Bookmaker odds include a ~5% margin plus noise so realistic +EV spots exist.

    Parameters
    ----------
    n_seasons : Number of full seasons to generate.
    seed      : Random seed for reproducibility.

    Returns
    -------
    DataFrame with columns:
        date, home_team, away_team, home_goals, away_goals,
        odds_home, odds_draw, odds_away
    """
    np.random.seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)

    rows = []
    start = date(2022, 8, 5)

    for season in range(n_seasons):
        season_start = start + timedelta(days=season * 365)
        teams = _PL_TEAMS.copy()
        random.shuffle(teams)

        # Build fixture list (each pair plays home & away once)
        fixtures = [(h, a) for h in teams for a in teams if h != a]
        # Spread over ~38 gameweeks (10 matches/gw)
        gw_dates = [season_start + timedelta(days=7 * gw) for gw in range(38)]
        gw_fixtures = [fixtures[i * 10:(i + 1) * 10] for i in range(38)]

        for gw, (gw_date, gw_f) in enumerate(zip(gw_dates, gw_fixtures)):
            for home, away in gw_f:
                lam = np.exp(_ATTACK_TRUE[home] + _DEFENSE_TRUE[away] + _HOME_ADV_TRUE)
                mu = np.exp(_ATTACK_TRUE[away] + _DEFENSE_TRUE[home])
                hg = _poisson_goal(lam)
                ag = _poisson_goal(mu)
                ph, pd_, pa = _fair_probs(home, away)
                oh, od, oa = _bookmaker_odds(ph, pd_, pa, rng=rng)
                # Slight day variation within gameweek
                match_date = gw_date + timedelta(days=int(rng.integers(0, 3)))
                rows.append({
                    "date": pd.Timestamp(match_date),
                    "home_team": home,
                    "away_team": away,
                    "home_goals": hg,
                    "away_goals": ag,
                    "odds_home": oh,
                    "odds_draw": od,
                    "odds_away": oa,
                })

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df


def load_or_generate(path: Optional[str | Path] = None) -> pd.DataFrame:
    """
    Load data from path if provided and file exists, otherwise generate synthetic data.
    """
    if path and Path(path).exists():
        print(f"Loading data from {path}")
        return load_csv(path)
    print("No data file found — generating synthetic 2-season dataset.")
    return generate_sample_data(n_seasons=2)
