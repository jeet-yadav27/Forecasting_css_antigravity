"""
forecasting/models/ml_models.py
--------------------------------
Classical time-series baseline (SARIMA) and shared hyperparameter grids
for the deep-learning forecasters.

Tree-based ML models (XGBoost / LightGBM / GradientBoosting) are not used.
"""

import itertools
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error


# ---------------------------------------------------------------------------
# Hyperparameter search grids (DL + SARIMA only)
# ---------------------------------------------------------------------------

HP_GRIDS: dict = {
    "SARIMA": [
        {"order": order, "seasonal_order": so}
        for order, so in itertools.product(
            [(1, 1, 1), (1, 1, 0), (0, 1, 1)],
            [(1, 1, 0, 12), (0, 1, 1, 12)],
        )
    ],
    # CNN-LSTM is intentionally NOT grid-searched (too slow on CPU/NumPy).
    # Fixed defaults are applied in runner.py: lr=1e-2, epochs=40.
    "N-BEATS": [
        {"lr": lr, "epochs": ep}
        for lr, ep in itertools.product([5e-3, 1e-3], [50, 80])
    ],
    "Transformer": [
        {"d_model": dm, "lr": lr, "epochs": ep}
        for dm, lr, ep in itertools.product([16, 32], [3e-3, 1e-3], [40, 70])
    ],
}


# ---------------------------------------------------------------------------
# SARIMA baseline
# ---------------------------------------------------------------------------

def fit_sarima(series: np.ndarray, horizon: int,
               order=(1, 1, 1), seasonal_order=(1, 1, 0, 12)) -> np.ndarray:
    """
    Fit a SARIMA model and return *horizon*-step forecast.

    Falls back to a seasonal naive forecast if fitting fails.
    """
    try:
        mod = sm.tsa.statespace.SARIMAX(
            series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = mod.fit(disp=False, maxiter=100)
        return np.clip(res.forecast(steps=horizon), 0, None)
    except Exception:
        if len(series) >= 12:
            return np.array(
                [series[-12 + i % 12] for i in range(horizon)], dtype=float
            )
        return np.full(horizon, series.mean())


def grid_search_sarima(series: np.ndarray, true_next: np.ndarray) -> dict:
    """
    Try each SARIMA config in HP_GRIDS['SARIMA'], return the params that
    produce the lowest 1-step-ahead MAE on *true_next*.
    """
    best_mae = float("inf")
    best_p = HP_GRIDS["SARIMA"][0]
    H = len(true_next)

    for p in HP_GRIDS["SARIMA"]:
        try:
            fc = fit_sarima(series, H, p["order"], p["seasonal_order"])
            mae = mean_absolute_error(true_next, fc)
            if mae < best_mae:
                best_mae = mae
                best_p = p
        except Exception:
            continue

    return best_p
