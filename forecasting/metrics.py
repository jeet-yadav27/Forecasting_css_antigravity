"""
forecasting/metrics.py
----------------------
Walk-forward evaluation metrics, ranking, and best-model selection.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

from forecasting.config import METRIC_RANK_WEIGHTS

logger = logging.getLogger(__name__)


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE ignoring near-zero actuals to avoid division blow-ups."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mask = np.abs(y_true) > 1e-8
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def compute_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict:
    """
    Compute RMSE, MAE, MAPE (%), and R² for a pair of series.

    Parameters
    ----------
    y_true, y_pred : array-like
        Actual and predicted values (same length).

    Returns
    -------
    dict
        Keys: RMSE, MAE, MAPE, R2.
    """
    yt = np.asarray(list(y_true), dtype=float).ravel()
    yp = np.asarray(list(y_pred), dtype=float).ravel()
    if len(yt) == 0 or len(yp) == 0 or len(yt) != len(yp):
        return {"RMSE": float("nan"), "MAE": float("nan"),
                "MAPE": float("nan"), "R2": float("nan")}

    err = yt - yp
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae  = float(np.mean(np.abs(err)))
    mape = _safe_mape(yt, yp)
    r2   = _safe_r2(yt, yp)
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}


def build_ranking_table(metrics_by_model: dict[str, dict]) -> pd.DataFrame:
    """
    Build a ranked comparison table from per-model metric dicts.

    Ranking score = weighted sum of min-max normalised ranks where
    lower RMSE/MAE/MAPE is better and higher R² is better.

    Parameters
    ----------
    metrics_by_model : dict
        ``{model_name: {"RMSE": ..., "MAE": ..., "MAPE": ..., "R2": ...}}``

    Returns
    -------
    pd.DataFrame
        Sorted by Rank ascending (1 = best).
    """
    if not metrics_by_model:
        return pd.DataFrame(columns=[
            "Rank", "Model", "RMSE", "MAE", "MAPE", "R2", "Score",
        ])

    rows = []
    for model, m in metrics_by_model.items():
        rows.append({
            "Model": model,
            "RMSE":  m.get("RMSE", float("nan")),
            "MAE":   m.get("MAE",  float("nan")),
            "MAPE":  m.get("MAPE", float("nan")),
            "R2":    m.get("R2",   float("nan")),
        })
    df = pd.DataFrame(rows)

    # Min-max normalise each error metric (0=best, 1=worst); invert R²
    score = np.zeros(len(df))
    for metric, weight in METRIC_RANK_WEIGHTS.items():
        vals = df[metric].astype(float).values
        if metric == "R2":
            # higher R² is better → convert to cost = 1 - R2 (clipped)
            cost = 1.0 - np.nan_to_num(vals, nan=0.0)
        else:
            cost = np.nan_to_num(vals, nan=np.nanmax(vals) if np.isfinite(vals).any() else 0.0)

        cmin, cmax = float(np.nanmin(cost)), float(np.nanmax(cost))
        if cmax - cmin < 1e-12:
            norm = np.zeros_like(cost)
        else:
            norm = (cost - cmin) / (cmax - cmin)
        score += weight * norm

    df["Score"] = score
    df = df.sort_values("Score", ascending=True).reset_index(drop=True)
    df.insert(0, "Rank", np.arange(1, len(df) + 1))
    # Round for display
    for col in ["RMSE", "MAE", "MAPE", "R2", "Score"]:
        df[col] = df[col].round(4)
    return df


def select_best_model(ranking_df: pd.DataFrame) -> str:
    """Return the top-ranked model name (falls back to 'Ensemble' logic upstream)."""
    if ranking_df is None or ranking_df.empty:
        logger.warning("Empty ranking table — defaulting to SARIMA")
        return "SARIMA"
    return str(ranking_df.iloc[0]["Model"])
