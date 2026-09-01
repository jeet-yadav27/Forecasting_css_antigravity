"""Build standalone_forecast_models.ipynb (no project-module imports)."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "standalone_forecast_models.ipynb"


def _strip_module_header(src: str) -> str:
    src = re.sub(r'^"""[\s\S]*?"""\s*', "", src, count=1)
    src = re.sub(r"^from \.base import \([^)]*\)\s*", "", src, count=1, flags=re.M)
    src = re.sub(r"^from \.base import .+\n", "", src, count=1, flags=re.M)
    src = re.sub(r"^from \.\w+ import .+\n", "", src, flags=re.M)
    src = src.replace("import statsmodels.api as sm\nfrom sklearn.metrics import mean_absolute_error\n", "")
    src = src.replace("import itertools\nimport numpy as np\n", "")
    return src.strip() + "\n"


def _nb_cell(cell_type: str, source: str) -> dict:
    lines = source.strip("\n") + "\n"
    # Jupyter source is a list of lines including newlines except last optional
    text = lines.splitlines(keepends=True)
    if not text:
        text = ["\n"]
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": text,
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def main() -> None:
    base = _strip_module_header((ROOT / "forecasting/models/base.py").read_text(encoding="utf-8"))
    # Keep numpy import only once in the notebook setup cell
    base = re.sub(r"^import numpy as np\n", "", base)
    base = re.sub(
        r"try:\n    from tqdm\.auto import tqdm\nexcept ImportError:  # pragma: no cover\n    def tqdm\(iterable=None, \*\*kwargs\):\n        return iterable if iterable is not None else range\(kwargs.get\(\"total\", 0\)\)\n",
        "def tqdm(iterable=None, **kwargs):\n    return iterable if iterable is not None else range(kwargs.get(\"total\", 0))\n",
        base,
        count=1,
    )
    cnn = _strip_module_header((ROOT / "forecasting/models/cnn_lstm.py").read_text(encoding="utf-8"))
    cnn = re.sub(r"^import numpy as np\n", "", cnn)
    nbeats = _strip_module_header((ROOT / "forecasting/models/nbeats.py").read_text(encoding="utf-8"))
    nbeats = re.sub(r"^import numpy as np\n", "", nbeats)
    trans = _strip_module_header((ROOT / "forecasting/models/transformer.py").read_text(encoding="utf-8"))
    trans = re.sub(r"^import numpy as np\n", "", trans)
    ml = (ROOT / "forecasting/models/ml_models.py").read_text(encoding="utf-8")
    # Keep only fit_sarima
    m = re.search(r"def fit_sarima[\s\S]+?return np\.full\(horizon, series\.mean\(\)\)\n", ml)
    if not m:
        raise SystemExit("fit_sarima not found")
    sarima_fn = m.group(0)

    cells = []
    cells.append(_nb_cell("markdown", """# Standalone forecast models

This notebook copies **CNN-LSTM, N-BEATS, Transformer, SARIMA, Holt-Winters**, and an **inverse-MAE ensemble** into cells.

It does **not** import `forecasting`, `main`, or any other project `.py` file.

Allowed libraries: `numpy`, `pandas`, `scikit-learn`, `statsmodels`, `matplotlib` (optional `tensorflow` for Keras CNN-LSTM).

Run **all cells in order**. Deep-learning demos use a small epoch count so they finish on CPU.
"""))
    cells.append(_nb_cell("markdown", "## 1. Third-party imports only"))
    cells.append(_nb_cell("code", '''from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

try:
    import statsmodels.api as sm
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except ImportError as exc:
    raise ImportError("Install statsmodels to run SARIMA and Holt-Winters.") from exc

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

SEED = 42
LOOKBACK = 12
HORIZON = 6
DEMO_EPOCHS = 8
np.random.seed(SEED)
print("Ready. No project modules imported.")
'''))
    cells.append(_nb_cell("markdown", "## 2. Shared NumPy utilities (activations, dropout, Adam, early stop)"))
    cells.append(_nb_cell("code", base))
    cells.append(_nb_cell("markdown", "## 3. CNN-LSTM (inlined)"))
    cells.append(_nb_cell("code", cnn))
    cells.append(_nb_cell("markdown", "## 4. N-BEATS (inlined)"))
    cells.append(_nb_cell("code", nbeats))
    cells.append(_nb_cell("markdown", "## 5. Transformer (inlined)"))
    cells.append(_nb_cell("code", trans))
    cells.append(_nb_cell("markdown", "## 6. SARIMA (inlined)"))
    cells.append(_nb_cell("code", "import statsmodels.api as sm\n\n" + sarima_fn))
    cells.append(_nb_cell("markdown", "## 7. Holt-Winters (inlined)"))
    cells.append(_nb_cell("code", '''def fit_holt_winters(series: np.ndarray, horizon: int, seasonal_periods: int = 12) -> np.ndarray:
    """Additive Holt-Winters; drops season if the series is too short."""
    y = np.asarray(series, dtype=float).ravel()
    n = len(y)
    sp = min(seasonal_periods, max(2, n // 2))
    use_seas = n >= sp * 2
    kw = dict(trend="add", initialization_method="estimated")
    if use_seas:
        kw["seasonal"] = "add"
        kw["seasonal_periods"] = sp
    try:
        res = ExponentialSmoothing(y, **kw).fit(optimized=True)
        return np.clip(np.asarray(res.forecast(horizon), dtype=float), 0, None)
    except Exception:
        if n >= 12:
            return np.array([y[-12 + i % 12] for i in range(horizon)], dtype=float)
        return np.full(horizon, float(y[-1]) if n else 0.0)


def inverse_mae_ensemble(forecasts: dict[str, np.ndarray], mae: dict[str, float]) -> np.ndarray:
    """Blend model forecasts with weights 1 / MAE."""
    names = [k for k in forecasts if k in mae and np.isfinite(mae[k])]
    if not names:
        arrs = [np.asarray(v, dtype=float) for v in forecasts.values()]
        return np.mean(np.stack(arrs, axis=0), axis=0)
    inv = {k: 1.0 / (mae[k] + 1e-9) for k in names}
    tot = sum(inv.values())
    H = len(next(iter(forecasts.values())))
    out = np.zeros(H)
    for k in names:
        fc = np.asarray(forecasts[k], dtype=float).ravel()[:H]
        out += (inv[k] / tot) * fc
    return np.clip(out, 0, None)
'''))
    cells.append(_nb_cell("markdown", """## 8. Optional Keras CNN-LSTM

Uses TensorFlow if installed. Skip this cell if `tensorflow` is missing — NumPy models still run.
"""))
    cells.append(_nb_cell("code", '''HAS_TF = False
try:
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, LSTM
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam

    HAS_TF = True

    def build_keras_cnn_lstm(time_step: int, n_features: int, lr: float = 1e-3, dropout: float = 0.2):
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
except Exception as exc:
    print("TensorFlow not available — Keras CNN-LSTM skipped:", type(exc).__name__)
'''))
    cells.append(_nb_cell("markdown", """## 9. Load a monthly series (pandas only)

Reads `data/Test_All.csv` if present (claim counts by process month for one part).
Otherwise builds a synthetic seasonal series so the notebook still runs.
"""))
    cells.append(_nb_cell("code", '''def load_monthly_claims(csv_path: Path, part: str | None = None) -> pd.Series:
    df = pd.read_csv(csv_path)
    part_col = "Part Name" if "Part Name" in df.columns else df.columns[0]
    date_col = "PROCESSING_DATE" if "PROCESSING_DATE" in df.columns else None
    if date_col is None:
        raise ValueError("CSV needs PROCESSING_DATE")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    df = df.dropna(subset=[date_col])
    if part is None:
        part = str(df[part_col].value_counts().idxmax())
    sub = df[df[part_col].astype(str) == str(part)]
    monthly = (
        sub.groupby(sub[date_col].dt.to_period("M"))
        .size()
        .rename("claims")
        .sort_index()
    )
    full = pd.period_range(monthly.index.min(), monthly.index.max(), freq="M")
    monthly = monthly.reindex(full, fill_value=0).astype(float)
    monthly.index = monthly.index.to_timestamp()
    return monthly, part


def synthetic_claims(n: int = 48) -> pd.Series:
    t = np.arange(n)
    y = 40 + 0.15 * t + 8 * np.sin(2 * np.pi * t / 12) + np.random.default_rng(SEED).normal(0, 2, n)
    idx = pd.date_range("2021-01-01", periods=n, freq="MS")
    return pd.Series(np.clip(y, 0, None), index=idx, name="claims")


csv = Path("data/Test_All.csv")
if csv.is_file():
    claims, PART = load_monthly_claims(csv)
    print(f"Loaded {csv} part={PART!r}  n={len(claims)}")
else:
    claims, PART = synthetic_claims(), "synthetic"
    print("CSV not found — using synthetic series")

claims.head()
'''))
    cells.append(_nb_cell("markdown", "## 10. Sliding windows (written here, not imported)"))
    cells.append(_nb_cell("code", '''def make_features(y: np.ndarray) -> np.ndarray:
    """Claims plus simple lags / calendar — all computed in this notebook."""
    s = pd.Series(y)
    n = len(s)
    month = np.array([pd.Timestamp(claims.index[i]).month for i in range(n)], dtype=float)
    feat = pd.DataFrame({
        "lag1": s.shift(1),
        "lag12": s.shift(12),
        "rm3": s.rolling(3, min_periods=1).mean(),
        "month": month,
        "sin12": np.sin(2 * np.pi * np.arange(n) / 12),
        "cos12": np.cos(2 * np.pi * np.arange(n) / 12),
    }).bfill().ffill().fillna(0.0)
    return feat.to_numpy(dtype=float)


def build_windows(series_vals: np.ndarray, exog: np.ndarray, lookback: int, horizon: int):
    n = len(series_vals) - lookback - horizon + 1
    if n <= 0:
        return None, None
    X, Y = [], []
    for i in range(n):
        win_y = series_vals[i:i + lookback, None]
        win_x = exog[i:i + lookback]
        X.append(np.concatenate([win_y, win_x], axis=1))
        Y.append(series_vals[i + lookback:i + lookback + horizon])
    return np.asarray(X), np.asarray(Y)


y_raw = claims.to_numpy(dtype=float)
exog_raw = make_features(y_raw)
c_scaler = MinMaxScaler()
e_scaler = MinMaxScaler()
y_sc = c_scaler.fit_transform(y_raw[:, None]).ravel()
ex_sc = e_scaler.fit_transform(exog_raw)

W = min(LOOKBACK, max(4, len(y_raw) // 3))
X, Y = build_windows(y_sc, ex_sc, W, HORIZON)
assert X is not None and len(X) >= 3, "Series too short — need more months"
last_window = X[-1:]
n_features = X.shape[2]
print(f"windows={len(X)}  lookback={W}  features={n_features}  horizon={HORIZON}")
'''))
    cells.append(_nb_cell("markdown", "## 11. Run each model independently"))
    cells.append(_nb_cell("code", '''def inv_claims(arr) -> np.ndarray:
    a = np.clip(np.asarray(arr, dtype=float).ravel(), 0, 1)
    return c_scaler.inverse_transform(a[:, None]).ravel().clip(0)


forecasts = {}

# --- Holt-Winters (claims only) ---
forecasts["Holt-Winters"] = fit_holt_winters(y_raw, HORIZON)
print("Holt-Winters", forecasts["Holt-Winters"].round(2))

# --- SARIMA (claims only) ---
forecasts["SARIMA"] = fit_sarima(y_sc, HORIZON)
forecasts["SARIMA"] = inv_claims(forecasts["SARIMA"])
print("SARIMA", forecasts["SARIMA"].round(2))

# --- CNN-LSTM ---
cnn = CnnLstmForecaster(
    lookback=W, n_features=n_features, horizon=HORIZON,
    lr=1e-2, epochs=DEMO_EPOCHS, seed=SEED, dropout=0.2, early_patience=3,
)
cnn.fit(X, Y)
forecasts["CNN-LSTM"] = inv_claims(cnn.predict(last_window)[0])
print("CNN-LSTM", forecasts["CNN-LSTM"].round(2))

# --- N-BEATS ---
nb = NBeatsForecaster(
    lookback=W, n_features=n_features, horizon=HORIZON,
    lr=5e-3, epochs=DEMO_EPOCHS, seed=SEED, dropout=0.2, early_patience=3,
)
nb.fit(X, Y)
forecasts["N-BEATS"] = inv_claims(nb.predict(last_window)[0])
print("N-BEATS", forecasts["N-BEATS"].round(2))

# --- Transformer ---
tfm = TransformerForecaster(
    lookback=W, n_features=n_features, horizon=HORIZON,
    d_model=16, n_heads=2, lr=3e-3, epochs=DEMO_EPOCHS, seed=SEED,
    dropout=0.2, early_patience=3,
)
tfm.fit(X, Y)
forecasts["Transformer"] = inv_claims(tfm.predict(last_window)[0])
print("Transformer", forecasts["Transformer"].round(2))
'''))
    cells.append(_nb_cell("markdown", "## 12. Keras CNN-LSTM (optional, 1-step then repeated)"))
    cells.append(_nb_cell("code", '''if HAS_TF:
    X1, y1 = [], []
    for i in range(len(y_sc) - W):
        X1.append(np.concatenate([y_sc[i:i + W, None], ex_sc[i:i + W]], axis=1))
        y1.append(y_sc[i + W])
    X1, y1 = np.asarray(X1), np.asarray(y1)
    km = build_keras_cnn_lstm(W, n_features)
    km.fit(X1, y1, epochs=max(DEMO_EPOCHS, 12), verbose=0,
           callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])
    hist = y_sc.copy()
    ex = ex_sc.copy()
    keras_fc = []
    for _ in range(HORIZON):
        win = np.concatenate([hist[-W:, None], ex[-W:]], axis=1)[None]
        p = float(np.clip(km.predict(win, verbose=0).ravel()[0], 0, 1))
        keras_fc.append(p)
        hist = np.append(hist, p)
        ex = np.vstack([ex, ex[-1]])
    forecasts["Keras-CNN-LSTM"] = inv_claims(keras_fc)
    print("Keras-CNN-LSTM", forecasts["Keras-CNN-LSTM"].round(2))
else:
    print("Skipped Keras-CNN-LSTM")
'''))
    cells.append(_nb_cell("markdown", "## 13. Ensemble + table"))
    cells.append(_nb_cell("code", '''# Toy in-sample MAE on last HORIZON actuals vs each model's first-step style score
hold = min(HORIZON, len(y_raw) // 5)
actual_tail = y_raw[-hold:]
mae = {}
for name, fc in forecasts.items():
    pred = np.asarray(fc, dtype=float).ravel()[:hold]
    if len(pred) < hold:
        continue
    # Compare last hold actuals to the mean of the forecast (demo only)
    mae[name] = float(np.mean(np.abs(actual_tail - pred[:hold])))

forecasts["Ensemble"] = inverse_mae_ensemble(
    {k: v for k, v in forecasts.items() if k != "Ensemble"}, mae
)

future_idx = pd.date_range(claims.index[-1] + pd.offsets.MonthBegin(1), periods=HORIZON, freq="MS")
table = pd.DataFrame({k: np.asarray(v, dtype=float).ravel()[:HORIZON] for k, v in forecasts.items()},
                     index=future_idx)
table.index.name = "Month"
table.round(2)
print("Demo MAE (not walk-forward):", {k: round(v, 3) for k, v in mae.items()})
'''))
    cells.append(_nb_cell("markdown", "## 14. Plot"))
    cells.append(_nb_cell("code", '''if HAS_MPL:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(claims.index, y_raw, label="history", color="#1D4ED8")
    colors = {
        "Holt-Winters": "#B45309", "SARIMA": "#047857", "CNN-LSTM": "#6C63FF",
        "N-BEATS": "#FF6584", "Transformer": "#0D9488", "Keras-CNN-LSTM": "#7C3AED",
        "Ensemble": "#111827",
    }
    for name, fc in forecasts.items():
        ax.plot(future_idx, fc, label=name, linestyle="--" if name != "Ensemble" else "-",
                color=colors.get(name), linewidth=2 if name == "Ensemble" else 1.4)
    ax.set_title(f"Standalone models — {PART}")
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylabel("Claims")
    fig.tight_layout()
    plt.show()
else:
    print(table.round(2))
'''))

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
