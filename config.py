"""Configuration for the MCQ-Capstone soccer betting algorithm."""
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
MODELS_DIR = BASE_DIR / "models"

# ── Model parameters ──────────────────────────────────────────────────────────
TIME_DECAY = 0.0065     # Per-day exponential decay for match weights
                        # ~0.0065 → matches 1 year ago weighted ~10%
MAX_GOALS = 10          # Max goals considered in probability matrix

# ── Betting parameters ────────────────────────────────────────────────────────
MIN_EV_THRESHOLD = 0.04     # Minimum edge to trigger a bet (4%)
KELLY_FRACTION = 0.25       # Fractional Kelly multiplier (quarter Kelly = safer)
MAX_BET_FRACTION = 0.05     # Hard cap: never risk >5% of bankroll on one bet
INITIAL_BANKROLL = 1000.0   # Starting bankroll for backtesting (any currency unit)

# ── Backtest parameters ───────────────────────────────────────────────────────
MIN_MATCHES_TO_FIT = 60     # Minimum historical matches needed before first bet
TRAIN_WINDOW_DAYS = 365     # Rolling training window (1 year of data)
MIN_ODDS = 1.30             # Ignore lines below this (too heavy favourite juice)
MAX_ODDS = 6.00             # Ignore long shots above this
