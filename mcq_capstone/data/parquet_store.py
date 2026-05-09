"""
Unified Parquet data store for all soccer data sources.

Sources
-------
1. StatsBomb Open Data  (free, event-level: goals/cards/subs/shots with minute)
2. football-data.co.uk  (free CSVs: results + bookmaker odds for 10+ leagues)

Output files
------------
data/parquet/
  matches.parquet     — one row per match (from StatsBomb)
  events.parquet      — goals, cards, subs with exact minute (StatsBomb)
  shots.parquet       — every shot with xG, location, technique (StatsBomb)
  results.parquet     — match results + odds from football-data.co.uk (all leagues)
  features.parquet    — engineered feature matrix ready for modelling

Quick load example
------------------
import pandas as pd
matches = pd.read_parquet('data/parquet/matches.parquet')
events  = pd.read_parquet('data/parquet/events.parquet')
shots   = pd.read_parquet('data/parquet/shots.parquet')
results = pd.read_parquet('data/parquet/results.parquet')
"""

from __future__ import annotations

import io
import time
import logging
from pathlib import Path
from typing import Optional

import requests
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ── football-data.co.uk result scraper ────────────────────────────────────────

# Leagues available on football-data.co.uk  (code → (name, url_prefix))
FDCO_LEAGUES: dict[str, tuple[str, str]] = {
    "E0":  ("Premier League",      "E0"),
    "E1":  ("Championship",        "E1"),
    "SP1": ("La Liga",             "SP1"),
    "SP2": ("La Liga 2",           "SP2"),
    "D1":  ("Bundesliga",          "D1"),
    "D2":  ("2. Bundesliga",       "D2"),
    "I1":  ("Serie A",             "I1"),
    "I2":  ("Serie B",             "I2"),
    "F1":  ("Ligue 1",             "F1"),
    "F2":  ("Ligue 2",             "F2"),
    "N1":  ("Eredivisie",          "N1"),
    "P1":  ("Primeira Liga",       "P1"),
    "T1":  ("Süper Lig",           "T1"),
    "G1":  ("Super League Greece", "G1"),
    "B1":  ("Belgian Pro League",  "B1"),
    "SC0": ("Scottish Premiership","SC0"),
}

FDCO_SEASONS = [
    "1819", "1920", "2021", "2122", "2223", "2324", "2425",
]

FDCO_BASE = "https://www.football-data.co.uk/mmz4281"


def _download_fdco_csv(league: str, season: str) -> Optional[pd.DataFrame]:
    """Download one CSV from football-data.co.uk."""
    url = f"{FDCO_BASE}/{season}/{league}.csv"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), encoding="latin-1", on_bad_lines="skip")
        if df.empty or "Date" not in df.columns:
            return None
        return df
    except Exception as e:
        logger.debug(f"  FDCO {league} {season}: {e}")
        return None


def _normalise_fdco(df: pd.DataFrame, league: str, season: str) -> pd.DataFrame:
    """Normalise a raw football-data.co.uk CSV to a clean schema."""
    df = df.copy()

    # Date parsing — FDCO uses both DD/MM/YY and DD/MM/YYYY
    for fmt in ["%d/%m/%Y", "%d/%m/%y"]:
        try:
            df["date"] = pd.to_datetime(df["Date"], format=fmt, errors="raise")
            break
        except (ValueError, KeyError):
            continue
    else:
        df["date"] = pd.to_datetime(df.get("Date"), dayfirst=True, errors="coerce")

    # Rename core columns
    rename = {
        "HomeTeam": "home_team", "AwayTeam": "away_team",
        "FTHG": "home_goals",   "FTAG": "away_goals",
        "HTHG": "ht_home_goals","HTAG": "ht_away_goals",
        "FTR":  "result",       "HTR":  "ht_result",
        "HS":   "home_shots",   "AS":   "away_shots",
        "HST":  "home_shots_target", "AST": "away_shots_target",
        "HC":   "home_corners", "AC":   "away_corners",
        "HF":   "home_fouls",   "AF":   "away_fouls",
        "HY":   "home_yellows", "AY":   "away_yellows",
        "HR":   "home_reds",    "AR":   "away_reds",
        # Odds (B365, market)
        "B365H": "odds_home", "B365D": "odds_draw", "B365A": "odds_away",
        "BbMxH": "odds_home_max", "BbMxD": "odds_draw_max", "BbMxA": "odds_away_max",
        "BbAvH": "odds_home_avg", "BbAvD": "odds_draw_avg", "BbAvA": "odds_away_avg",
        # xG (newer seasons)
        "B365>2.5": "odds_over25", "B365<2.5": "odds_under25",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Use B365 odds as fallback if primary odds cols missing
    for col in ["odds_home", "odds_draw", "odds_away"]:
        if col not in df.columns:
            # try other bookmaker columns
            alt_map = {
                "odds_home": ["BWH", "IWH", "LBH", "WHH", "VCH"],
                "odds_draw": ["BWD", "IWD", "LBD", "WHD", "VCD"],
                "odds_away": ["BWA", "IWA", "LBA", "WHA", "VCA"],
            }
            for alt in alt_map.get(col, []):
                if alt in df.columns:
                    df[col] = df[alt]
                    break

    df["league"] = league
    df["season"] = season

    # Numeric coercion
    numeric_cols = [
        "home_goals", "away_goals", "ht_home_goals", "ht_away_goals",
        "home_shots", "away_shots", "home_shots_target", "away_shots_target",
        "home_corners", "away_corners", "home_fouls", "away_fouls",
        "home_yellows", "away_yellows", "home_reds", "away_reds",
        "odds_home", "odds_draw", "odds_away",
        "odds_home_max", "odds_draw_max", "odds_away_max",
        "odds_home_avg", "odds_draw_avg", "odds_away_avg",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows missing core fields
    df = df.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals"])
    df = df[df["home_team"].astype(str).str.strip() != ""]

    # Select columns present
    keep = ["date", "home_team", "away_team", "home_goals", "away_goals",
            "ht_home_goals", "ht_away_goals", "result", "ht_result",
            "home_shots", "away_shots", "home_shots_target", "away_shots_target",
            "home_corners", "away_corners", "home_fouls", "away_fouls",
            "home_yellows", "away_yellows", "home_reds", "away_reds",
            "odds_home", "odds_draw", "odds_away",
            "odds_home_max", "odds_draw_max", "odds_away_max",
            "odds_home_avg", "odds_draw_avg", "odds_away_avg",
            "odds_over25", "odds_under25",
            "league", "season"]
    present = [c for c in keep if c in df.columns]
    return df[present].sort_values("date").reset_index(drop=True)


def download_fdco_results(
    leagues: Optional[list[str]] = None,
    seasons: Optional[list[str]] = None,
    out_dir: str = "data/parquet",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Download all football-data.co.uk CSVs and save as results.parquet.

    Parameters
    ----------
    leagues : List of league codes (default: all 16 leagues).
    seasons : List of season codes like '2324' (default: 2018/19 → 2024/25).
    out_dir : Directory to save parquet.

    Returns
    -------
    Combined DataFrame saved to {out_dir}/results.parquet
    """
    leagues = leagues or list(FDCO_LEAGUES.keys())
    seasons = seasons or FDCO_SEASONS
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    frames = []
    total = len(leagues) * len(seasons)
    done  = 0

    if verbose:
        print(f"\nDownloading football-data.co.uk results")
        print(f"  Leagues: {len(leagues)}  Seasons: {len(seasons)}  "
              f"({total} combinations)")

    for league in leagues:
        league_name = FDCO_LEAGUES.get(league, (league,))[0]
        for season in seasons:
            done += 1
            df = _download_fdco_csv(league, season)
            if df is not None:
                df = _normalise_fdco(df, league, season)
                frames.append(df)
                if verbose:
                    print(f"  [{done:>3}/{total}] {league_name:<25} {season}  "
                          f"{len(df):>4} matches", flush=True)
            else:
                if verbose:
                    print(f"  [{done:>3}/{total}] {league_name:<25} {season}  "
                          f"  — not available", flush=True)
            time.sleep(0.5)  # be polite

    if not frames:
        print("No data downloaded.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["league", "date"]).reset_index(drop=True)
    out_path = Path(out_dir) / "results.parquet"
    combined.to_parquet(out_path, index=False)

    if verbose:
        print(f"\n  results.parquet : {len(combined):>6,} matches → {out_path}")
        print(f"  Leagues         : {combined['league'].nunique()}")
        print(f"  Date range      : {combined['date'].min().date()} "
              f"→ {combined['date'].max().date()}")
        if "odds_home" in combined.columns:
            pct = combined["odds_home"].notna().mean()
            print(f"  Odds coverage   : {pct:.0%}")

    return combined


# ── Feature matrix builder ─────────────────────────────────────────────────────

def build_and_save_features(
    results_df: pd.DataFrame,
    out_dir: str = "data/parquet",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build the 82-feature engineering matrix from results_df and save as parquet.
    Uses the existing engineer.py pipeline.
    """
    from mcq_capstone.features.engineer import build_feature_matrix

    if verbose:
        print(f"\nBuilding feature matrix for {len(results_df):,} matches …")

    feat_df = build_feature_matrix(results_df)

    out_path = Path(out_dir) / "features.parquet"
    feat_df.to_parquet(out_path, index=False)

    if verbose:
        print(f"  features.parquet: {feat_df.shape[0]:>6,} rows × "
              f"{feat_df.shape[1]} cols → {out_path}")

    return feat_df


# ── Convenience summary ────────────────────────────────────────────────────────

def parquet_summary(out_dir: str = "data/parquet") -> None:
    """Print a summary of all parquet files in out_dir."""
    p = Path(out_dir)
    files = sorted(p.glob("*.parquet"))
    if not files:
        print(f"No parquet files found in {out_dir}")
        return

    print(f"\n{'='*60}")
    print(f"  PARQUET STORE: {out_dir}")
    print(f"{'='*60}")
    for f in files:
        try:
            df = pd.read_parquet(f)
            size_mb = f.stat().st_size / 1e6
            print(f"  {f.name:<25} {len(df):>8,} rows  "
                  f"{df.shape[1]:>3} cols  {size_mb:.1f} MB")
            if "date" in df.columns:
                print(f"    date range : {df['date'].min().date()} "
                      f"→ {df['date'].max().date()}")
            if "competition" in df.columns or "league" in df.columns:
                col = "competition" if "competition" in df.columns else "league"
                comps = df[col].value_counts().head(6)
                for comp, cnt in comps.items():
                    print(f"    {comp:<30} {cnt:>6,}")
        except Exception as e:
            print(f"  {f.name}: error reading ({e})")
    print(f"{'='*60}")
