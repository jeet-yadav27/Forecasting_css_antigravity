"""
forecasting/config.py
---------------------
Central configuration for the Warranty Forecasting System.
Edit values here; every other module imports from this file.
"""

import os
import random

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
FORECASTS_DIR = os.path.join(OUTPUT_DIR, "forecasts")
# Locked hyperparameters from a prior upload/train run (reused unless retuned)
BEST_PARAMS_PATH = os.path.join(OUTPUT_DIR, "best_params.json")

DATA_FILES = [
    os.path.join(DATA_DIR, "Test_All.csv"),
]

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Forecasting parameters
# ---------------------------------------------------------------------------
PRODUCTION_PER_MONTH: int = 25_000   # starting monthly production (synthetic)
PRODUCTION_GROWTH_RATE: float = 0.02  # +2% every month when production is synthetic
FORECAST_HORIZON: int = 12           # months to forecast forward
LOOKBACK_WINDOW: int = 12            # sliding-window history length
WARRANTY_MONTHS: int = 36            # 3-year warranty lifecycle

# ---------------------------------------------------------------------------
# Walk-forward cross-validation & hyperparameter tuning
# ---------------------------------------------------------------------------
# Reserve the last N_CV_FOLDS months as the rolling test window.
# Fold k trains on history up to that month, forecasts 1 step, then rolls forward.
N_CV_FOLDS: int = 6

# When True, a small hyperparameter grid is searched on training data only.
HP_TUNE: bool = True

# Composite ranking weights for automatic best-model selection
# (lower is better for error metrics; R² is inverted in the ranker).
METRIC_RANK_WEIGHTS: dict = {
    "RMSE": 0.30,
    "MAE":  0.25,
    "MAPE": 0.25,
    "R2":   0.20,
}

# ---------------------------------------------------------------------------
# Countermeasure config (static / config-file based — still supported)
# ---------------------------------------------------------------------------
# Format: {"Part Name": {"month": "YYYY-MM", "effectiveness": 0.0–1.0}}
# effectiveness = fraction of claims PREVENTED at the CM month.
# Decays exponentially with CM_DECAY_HALF_LIFE months.
COUNTERMEASURES: dict = {
    # Example (uncomment & edit to activate):
    # "Part 1": {"month": "2022-01", "effectiveness": 0.70},
    # "Part 5": {"month": "2021-09", "effectiveness": 0.55},
}

CM_DECAY_HALF_LIFE: int = 3          # months for countermeasure half-life decay

# ---------------------------------------------------------------------------
# Visual / dashboard
# ---------------------------------------------------------------------------
PALETTE = [
    "#6C63FF", "#FF6584", "#43D9AD", "#FFBB35", "#4FC3F7",
    "#FF8A65", "#A5D6A7", "#CE93D8", "#80DEEA", "#FFCC80",
]

# Active models: deep-learning forecasters + SARIMA baseline only.
# Tree-based ML (XGBoost / LightGBM / GradBoost) is intentionally excluded.
MODEL_COLORS: dict = {
    "CNN-LSTM":    "#6C63FF",
    "N-BEATS":     "#FF6584",
    "Transformer": "#43D9AD",
    "SARIMA":      "#A5D6A7",
}

MODEL_NAMES = list(MODEL_COLORS.keys())
