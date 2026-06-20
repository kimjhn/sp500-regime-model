"""Shared constants for the STEP 2 deep-learning regime model.

The architecture skeleton (two-track GRU + MLP + attention late-fusion, the
date splits, the loss family) is fixed by the project proposal (see CLAUDE.md).
Only values that the proposal left open are tuned here.
"""
from pathlib import Path

# --- paths ---
CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent.parent
DATA_CSV = CODE_DIR.parent / "data" / "sp500_regime_dataset_final.csv"
ARTIFACT_DIR = CODE_DIR / "artifacts"

# --- features ---
# Track 1 (GRU): past price/technical sequence. Use log_vol (not raw vol).
TRACK1_FEATS = [
    "log_return", "log_vol_5d", "log_vol_20d", "log_vol_60d",
    "vol_ratio_5_20", "vol_ratio_5_60",
    "sma_gap_20d", "sma_gap_60d", "sma_gap_200d",
    "rsi_14", "macd_pct", "macd_hist",
]
# Track 2 (MLP): macro snapshot. Non-stationary features use the *_diff version.
TRACK2_FEATS = [
    "M2_YOY", "NFCI", "YIELD_SPREAD_diff",
    "CPI_YOY", "DEBT_EQUITY_diff", "LOAN_DEPOSIT",
]
SOFT_LABEL_COLS = [f"regime_prob_{i}" for i in range(5)]
LABEL_COL = "regime_label"
CLOSE_COL = "close"

# --- regime semantics (sorted ascending by return in STEP 1) ---
N_REGIMES = 5
REGIME_NAMES = {0: "Bear", 1: "Weak Bear", 2: "Neutral", 3: "Weak Bull", 4: "Bull"}
RISK_OFF_REGIMES = (0, 1)  # Bear + Weak Bear -> the regimes whose recall matters most

# --- sequence / forecast geometry ---
SEQ_LEN = 60       # GRU lookback (proposal-fixed)
HORIZON = 20       # label looks forward 20 trading days -> embargo length

# --- chronological splits (proposal-fixed; no random split) ---
TRAIN_END = "2018-12-31"
VAL_START, VAL_END = "2019-01-01", "2021-12-31"
TEST_START = "2022-01-01"

# --- backtest: regime -> equity exposure ladder (defensive long/cash) ---
EXPOSURE_LADDER = {0: 0.00, 1: 0.25, 2: 0.50, 3: 0.75, 4: 1.00}
TRADING_DAYS = 252
COST_PER_TURNOVER = 0.0005  # 5 bps per unit of weight change

SEED = 42
