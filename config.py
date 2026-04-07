"""Configuration for the MCQ-Capstone soccer betting algorithm."""
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
MODELS_DIR = BASE_DIR / "models"

# ── Model parameters ──────────────────────────────────────────────────────────
TIME_DECAY = 0.0065     # Per-day exponential decay for match weights
MAX_GOALS = 10          # Max goals considered in probability matrix

# ── Bet selection — the core edge filters ─────────────────────────────────────
#
# The old 4% threshold produced 34% win rate because it picked up noise.
# Higher thresholds mean fewer bets but each one carries real edge.
#
MIN_EV_THRESHOLD = 0.10     # Minimum edge: H/W or A/W outcome (10%)
DRAW_EV_THRESHOLD = 0.15    # Draws require much higher edge (15%)
                            # Draws have ~17% historical win rate at low EV —
                            # DC model is weakest predicting draw probability.
MIN_MODEL_PROB = 0.40       # Model must assign ≥40% to the outcome we're betting
MAX_BOOK_OVERROUND = 0.12   # Skip markets where book margin > 12% (too juicy)

# ── Bet sizing ────────────────────────────────────────────────────────────────
KELLY_FRACTION = 0.25       # Quarter Kelly (reduces variance at cost of EV)
MAX_BET_FRACTION = 0.04     # Hard cap: never risk >4% of bankroll per bet

# ── Odds range ────────────────────────────────────────────────────────────────
MIN_ODDS = 1.50             # Skip heavy favourites (margin eats the edge)
MAX_ODDS = 5.00             # Skip long shots (variance too high)

# ── Backtest parameters ───────────────────────────────────────────────────────
MIN_MATCHES_TO_FIT = 60     # Warm-up: matches before we start betting
TRAIN_WINDOW_DAYS = 365     # Rolling training window
INITIAL_BANKROLL = 1000.0

# ── Leagues ───────────────────────────────────────────────────────────────────
# football-data.co.uk league codes
DOMESTIC_LEAGUES = {
    "E0":  "Premier League (England)",
    "SP1": "La Liga (Spain)",
    "D1":  "Bundesliga (Germany)",
    "I1":  "Serie A (Italy)",
    "F1":  "Ligue 1 (France)",
    "N1":  "Eredivisie (Netherlands)",
    "P1":  "Primeira Liga (Portugal)",
    "B1":  "Belgian Pro League",
    "T1":  "Süper Lig (Turkey)",
    "G1":  "Super League (Greece)",
}

# Seasons to download (most recent 3 = enough data, no stale teams)
DOWNLOAD_SEASONS = ["2122", "2223", "2324"]

# UEFA note: football-data.co.uk doesn't provide CL/EL in the same CSV format.
# To add UEFA competitions, export your data as a CSV with columns:
#   Date, HomeTeam, AwayTeam, FTHG, FTAG, B365H, B365D, B365A
# and pass it with --league UEFA to the CLI.
