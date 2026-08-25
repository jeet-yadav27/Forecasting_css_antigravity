"""
forecasting/pipeline/annotated_forecast.py
-----------------------------------------
Reference-style data feeding + last-6-month walk-forward evaluation.

Mirrors the annotated CNN-LSTM workflow:
  Part_Failure, Production, Warranty_Days, Countermeasure, FCOK_Jan_Aug
with enhancements for multiple models (Keras CNN-LSTM, NumPy CNN-LSTM,
N-BEATS, Transformer, SARIMA).
"""

from __future__ import annotations

import logging
import os
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from forecasting.config import (
    N_CV_FOLDS,
    OUTPUT_DIR,
    PRODUCTION_PER_MONTH,
    PRODUCTION_GROWTH_RATE,
    RANDOM_SEED,
)

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "Part_Failure",
    "Production",
    "Warranty_Days",
    "Countermeasure",
    "FCOK_Jan_Aug",
]

try:
    pd.tseries.frequencies.to_offset("ME")
    _MONTH_FREQ = "ME"
except (ValueError, KeyError):
    _MONTH_FREQ = "M"

# Optional Keras
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, LSTM, Flatten, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
    HAS_TF = True
except Exception:  # pragma: no cover
    HAS_TF = False
    logger.info("TensorFlow not available — Keras CNN-LSTM disabled; NumPy models still work.")


def _resolve_part_col(df: pd.DataFrame) -> str:
    for c in ("Part Name", "Part_Name+", "Part Name+", "PART_NAME", "Part"):
        if c in df.columns:
            return c
    raise KeyError("Part name column not found.")


def _resolve_month_col(df: pd.DataFrame) -> str:
    for c in (
        "Wty_Month", "Wty Month", "PROCESSING_DATE", "Process Date",
        "PROC_MONTH", "Claim Month",
    ):
        if c in df.columns:
            return c
    raise KeyError("Warranty / process month column not found.")


def build_annotated_monthly(
    raw: pd.DataFrame,
    part_name: str,
    *,
    countermeasure_start: str | None = None,
    use_synthetic_production: bool = False,
) -> pd.DataFrame:
    """
    Build reference-style monthly feature frame for one part.

    Columns: Part_Failure, Production, Warranty_Days, Countermeasure, FCOK_Jan_Aug
    Index: month-end DatetimeIndex
    """
    df = raw.copy()
    part_col = _resolve_part_col(df)
    month_col = _resolve_month_col(df)

    df[month_col] = pd.to_datetime(df[month_col], errors="coerce")
    part_df = df[df[part_col] == part_name].dropna(subset=[month_col]).copy()
    if part_df.empty:
        return pd.DataFrame(columns=FEATURE_COLS)

    # Warranty days
    wcol = None
    for candidate in ("Warranty Days", "WarrantyDays", "Warranty_Days", "VEHICLE_AGE_MONTHS"):
        if candidate in part_df.columns:
            wcol = candidate
            break
    if wcol is None and "VEHICLE_AGE_MONTHS" not in part_df.columns:
        # derive from FCOK if present
        if "FCOK_DATE" in part_df.columns:
            part_df["_wdays"] = (
                (part_df[month_col] - pd.to_datetime(part_df["FCOK_DATE"], errors="coerce"))
                .dt.days.clip(lower=0)
            )
            wcol = "_wdays"
        else:
            part_df["_wdays"] = 0.0
            wcol = "_wdays"

    failures = (
        part_df.groupby(pd.Grouper(key=month_col, freq=_MONTH_FREQ))
        .agg(Part_Failure=(part_col, "count"))
    )

    if "Production" in part_df.columns:
        production = (
            part_df.groupby(pd.Grouper(key=month_col, freq=_MONTH_FREQ))
            .agg(Production=("Production", "sum"))
        )
    else:
        production = pd.DataFrame(index=failures.index)
        production["Production"] = 0.0
        if use_synthetic_production and len(failures):
            logger.warning(
                "Synthetic production requested but disabled by policy — using zeros."
            )

    warranty = (
        part_df.groupby(pd.Grouper(key=month_col, freq=_MONTH_FREQ))
        .agg(Warranty_Days=(wcol, "mean"))
    )
    warranty["Warranty_Days"] = warranty["Warranty_Days"].fillna(0.0)

    monthly = (
        failures.join(production, how="outer")
                .join(warranty, how="outer")
                .fillna(0)
                .sort_index()
    )

    # Countermeasure flag (config start or default far past → all zeros unless set)
    if countermeasure_start:
        cm_start = pd.to_datetime(countermeasure_start)
        monthly["Countermeasure"] = (monthly.index >= cm_start).astype(float)
    else:
        monthly["Countermeasure"] = 0.0

    # Seasonal indicator (Jan / Aug) — same as reference FCOK_Jan_Aug
    monthly["FCOK_Jan_Aug"] = monthly.index.month.isin([1, 8]).astype(int)

    return monthly[FEATURE_COLS]


def _make_sequences(scaled: np.ndarray, time_step: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(scaled) - time_step):
        X.append(scaled[i:i + time_step])
        y.append(scaled[i + time_step, 0])
    if not X:
        return np.empty((0, time_step, scaled.shape[1])), np.empty((0,))
    return np.asarray(X), np.asarray(y)


def _build_keras_cnn_lstm(time_step: int, n_features: int, lr: float = 1e-3,
                          dropout: float = 0.2):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation="relu",
                     input_shape=(time_step, n_features)))
    model.add(Dropout(dropout))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=lr), loss="mean_squared_error")
    return model


def _predict_keras(model, last_window: np.ndarray, scaler: MinMaxScaler, n_features: int) -> float:
    next_scaled = model.predict(last_window, verbose=0)
    pad = np.concatenate([next_scaled, np.zeros((1, n_features - 1))], axis=1)
    return float(max(0.0, scaler.inverse_transform(pad)[0, 0]))


def _predict_numpy_model(name: str, X_tr: np.ndarray, y_tr: np.ndarray,
                         last_window: np.ndarray, scaler: MinMaxScaler,
                         time_step: int, n_features: int) -> float:
    """Train a NumPy DL model on sequences and predict next Part_Failure."""
    from forecasting.models.cnn_lstm import CnnLstmForecaster
    from forecasting.models.nbeats import NBeatsForecaster
    from forecasting.models.transformer import TransformerForecaster

    # Direct multi-step horizon=1 targets already in y_tr shaped (N,)
    y2 = y_tr.reshape(-1, 1)
    if name == "CNN-LSTM":
        mdl = CnnLstmForecaster(
            lookback=time_step, n_features=n_features, horizon=1,
            lr=1e-2, epochs=40, seed=RANDOM_SEED,
        )
    elif name == "N-BEATS":
        mdl = NBeatsForecaster(
            lookback=time_step, n_features=n_features, horizon=1,
            lr=5e-3, epochs=50, seed=RANDOM_SEED,
        )
    elif name == "Transformer":
        mdl = TransformerForecaster(
            lookback=time_step, n_features=n_features, horizon=1,
            d_model=16, n_heads=2, lr=3e-3, epochs=40, seed=RANDOM_SEED,
        )
    else:
        raise ValueError(name)

    mdl.fit(X_tr, y2)
    pred_sc = float(mdl.predict(last_window)[0, 0])
    pad = np.array([[pred_sc] + [0.0] * (n_features - 1)])
    return float(max(0.0, scaler.inverse_transform(pad)[0, 0]))


def _predict_sarima(history_failures: np.ndarray) -> float:
    from forecasting.models.ml_models import fit_sarima
    fc = fit_sarima(history_failures.astype(float), horizon=1)
    return float(max(0.0, fc[0]))


def forecast_last_n_months_annotated(
    raw: pd.DataFrame,
    part_name: str,
    *,
    models: Iterable[str] | None = None,
    time_step: int = 4,
    epochs: int = 75,
    n_test_months: int = N_CV_FOLDS,
    countermeasure_start: str | None = None,
) -> pd.DataFrame | None:
    """
    Walk-forward forecast of the last *n_test_months* using reference data feeding.

    For each test month:
      - Fit scaler ONLY on history before that month (no leakage)
      - Build sequences, train selected model(s) on that history
      - Predict the test month Part_Failure

    Returns a results DataFrame with Actuals, per-model forecasts, Error / Error% / Accuracy%.
    """
    monthly = build_annotated_monthly(
        raw, part_name, countermeasure_start=countermeasure_start
    )
    if len(monthly) < time_step + n_test_months + 2:
        logger.warning(
            "Not enough months for %s (have %d, need ~%d)",
            part_name, len(monthly), time_step + n_test_months + 2,
        )
        return None

    available = ["CNN-LSTM", "N-BEATS", "Transformer", "SARIMA"]
    if HAS_TF:
        available = ["Keras-CNN-LSTM"] + available
    selected = [m for m in (models or available) if m in available]
    if not selected:
        selected = ["CNN-LSTM"]

    test_months = monthly.index[-n_test_months:]
    rows = []

    for month in test_months:
        hist = monthly[monthly.index < month]
        if len(hist) <= time_step:
            continue

        actual = float(monthly.loc[month, "Part_Failure"])
        row = {"Date": month, "Actuals": actual}

        feats = hist[FEATURE_COLS].values.astype(float)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(feats)
        X, y = _make_sequences(scaled, time_step)
        n_features = feats.shape[1]
        if len(X) == 0:
            continue
        last_window = X[-1:].astype(float)

        for model_name in selected:
            try:
                if model_name == "Keras-CNN-LSTM" and HAS_TF:
                    mdl = _build_keras_cnn_lstm(time_step, n_features, dropout=0.2)
                    def lr_schedule(epoch, initial_lr=0.001, drop=0.5, epochs_drop=10):
                        return initial_lr * (drop ** (epoch // epochs_drop))
                    # Drop remaining epochs when val_loss stops decreasing
                    early = EarlyStopping(
                        monitor="val_loss" if (len(X) >= 10) else "loss",
                        patience=5,
                        min_delta=1e-6,
                        restore_best_weights=True,
                    )
                    callbacks = [LearningRateScheduler(lr_schedule), early]
                    val_split = 0.1 if len(X) >= 10 else 0.0
                    mdl.fit(
                        X, y, epochs=epochs, batch_size=1, verbose=0,
                        callbacks=callbacks,
                        validation_split=val_split,
                    )
                    pred = _predict_keras(mdl, last_window, scaler, n_features)
                elif model_name == "SARIMA":
                    pred = _predict_sarima(hist["Part_Failure"].values)
                else:
                    # Map Keras alias away; NumPy CNN-LSTM for "CNN-LSTM"
                    pred = _predict_numpy_model(
                        model_name, X, y, last_window, scaler, time_step, n_features
                    )
            except Exception as exc:
                logger.warning("%s failed for %s @ %s: %s", model_name, part_name, month, exc)
                pred = float("nan")

            row[f"{model_name} Forecast"] = round(pred, 1) if np.isfinite(pred) else np.nan
            err = abs(actual - pred) if np.isfinite(pred) else np.nan
            row[f"Error {model_name}"] = round(err, 1) if np.isfinite(err) else np.nan
            if actual == 0 or not np.isfinite(err):
                row[f"Error % {model_name}"] = np.nan
                row[f"Accuracy % {model_name}"] = np.nan
            else:
                row[f"Error % {model_name}"] = round((err / actual) * 100, 1)
                row[f"Accuracy % {model_name}"] = round((1 - err / actual) * 100, 1)

        rows.append(row)

    if not rows:
        return None

    results_df = pd.DataFrame(rows)
    # Save CSV like reference
    os.makedirs(os.path.join(OUTPUT_DIR, "Forecast"), exist_ok=True)
    last_date = pd.Timestamp(results_df["Date"].max()).strftime("%Y-%m-%d")
    safe_part = "".join(c if c.isalnum() or c in "-_" else "_" for c in part_name)
    path = os.path.join(
        OUTPUT_DIR, "Forecast",
        f"Forecasting_result_warranty_{safe_part}_{last_date}_annotated.csv",
    )
    results_df.to_csv(path, index=False)
    logger.info("Annotated forecast saved → %s", path)
    results_df.attrs["save_path"] = path
    results_df.attrs["monthly"] = monthly
    return results_df
