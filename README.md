# MCQ-Capstone — Soccer Prediction Betting Algorithm

A priori soccer prediction algorithm for trading on prediction markets.

## Model

**Dixon-Coles Poisson model** — the industry-standard approach for soccer score prediction.

- Models home goals as **Poisson(λ)** and away goals as **Poisson(μ)**
- Each team has a latent **attack** and **defense** strength parameter
- Global **home advantage** parameter
- **Dixon-Coles τ correction** adjusts probabilities for low-scoring scorelines (0–0, 1–0, 0–1, 1–1) which are correlated
- Parameters estimated by **maximum likelihood** with **exponential time-decay** weighting (recent matches count more)

## Betting Strategy

1. Model outputs P(home win), P(draw), P(away win)
2. **Expected value**: `EV = P_model × decimal_odds − 1`
3. Only bet when `EV > threshold` (default 4% edge)
4. **Fractional Kelly criterion** for stake sizing: `f = (b·p − q) / b × 0.25`
5. Hard cap: never risk more than 5% of bankroll on a single bet

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Backtest on synthetic data (works out of the box)
```bash
python src/main.py backtest
```

### Backtest on real data (football-data.co.uk CSV format)
```bash
# Download from: https://www.football-data.co.uk/data.php
python src/main.py backtest --data path/to/E0.csv
```

### Predict a match
```bash
python src/main.py predict --home "Arsenal" --away "Chelsea"

# With bookmaker odds to check for value:
python src/main.py predict \
  --home "Arsenal" --away "Chelsea" \
  --odds-home 2.10 --odds-draw 3.40 --odds-away 3.60
```

### View team ratings
```bash
python src/main.py ratings
```

### Scan fixtures for value bets
```bash
# With a fixtures CSV (columns: home_team, away_team, odds_home, odds_draw, odds_away)
python src/main.py scan --fixtures fixtures.csv
```

## Real Data Source

Download free historical CSV data from **football-data.co.uk**:

| League | URL |
|--------|-----|
| Premier League | `https://www.football-data.co.uk/mmz4281/2324/E0.csv` |
| La Liga | `https://www.football-data.co.uk/mmz4281/2324/SP1.csv` |
| Bundesliga | `https://www.football-data.co.uk/mmz4281/2324/D1.csv` |
| Serie A | `https://www.football-data.co.uk/mmz4281/2324/I1.csv` |
| Ligue 1 | `https://www.football-data.co.uk/mmz4281/2324/F1.csv` |

The CSV must contain at minimum: `Date, HomeTeam, AwayTeam, FTHG, FTAG`
For backtesting with odds: `B365H, B365D, B365A`

Multiple seasons can be concatenated:
```python
import pandas as pd
from src.data_loader import load_csv

df = pd.concat([load_csv("E0_2223.csv"), load_csv("E0_2324.csv")]).reset_index(drop=True)
```

## Configuration

Edit `config.py` to tune the algorithm:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TIME_DECAY` | 0.0065 | Per-day decay weight (~1 year half-life) |
| `MIN_EV_THRESHOLD` | 0.04 | Minimum edge to bet (4%) |
| `KELLY_FRACTION` | 0.25 | Quarter Kelly (conservative sizing) |
| `MAX_BET_FRACTION` | 0.05 | Max 5% of bankroll per bet |
| `MIN_ODDS / MAX_ODDS` | 1.30 / 6.00 | Ignore extreme odds |

## Reference

> Dixon, M.J. & Coles, S.G. (1997). *Modelling Association Football Scores
> and Inefficiencies in the Football Betting Market.*
> Applied Statistics, 46(2), 265–280.
