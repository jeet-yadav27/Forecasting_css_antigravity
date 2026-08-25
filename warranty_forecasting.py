"""
=============================================================================
  AUTOMOTIVE WARRANTY CLAIMS FORECASTING SYSTEM
  Models: CNN-LSTM · N-BEATS · Transformer · XGBoost · LightGBM · Ensemble
  Output: Interactive HTML Dashboard
=============================================================================
"""

import os, warnings, json
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import statsmodels.api as sm

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
PRODUCTION_PER_MONTH = 25_000
FORECAST_HORIZON     = 12
LOOKBACK_WINDOW      = 12
DATA_FILES = [
    "Test 1.csv", "Test 2.csv", "Test 3.csv",
    "Test 4.csv", "Test 5.csv"
]

# Countermeasure config:
# {"Part Name": {"month": "YYYY-MM", "effectiveness": 0.0-1.0}}
# effectiveness = fraction of claims PREVENTED at the CM month.
# Decays exponentially with CM_DECAY_HALF_LIFE months.
COUNTERMEASURES = {
    # Example (uncomment & edit to activate):
    # "Part 1": {"month": "2022-01", "effectiveness": 0.70},
    # "Part 5": {"month": "2021-09", "effectiveness": 0.55},
}
CM_DECAY_HALF_LIFE = 3

OUTPUT_HTML = "warranty_forecast_dashboard.html"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def tanh(x):
    return np.tanh(np.clip(x, -30, 30))


class AdamOptimizer:
    """Minimal Adam optimizer (per-key state)."""
    def __init__(self, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.t = 0
        self.m = {}
        self.v = {}

    def update(self, key, param, grad):
        if key not in self.m:
            self.m[key] = np.zeros_like(param)
            self.v[key] = np.zeros_like(param)
        self.t += 1
        self.m[key] = self.b1 * self.m[key] + (1 - self.b1) * grad
        self.v[key] = self.b2 * self.v[key] + (1 - self.b2) * grad ** 2
        m_hat = self.m[key] / (1 - self.b1 ** self.t)
        v_hat = self.v[key] / (1 - self.b2 ** self.t)
        return param - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# =============================================================================
# 1. DATA LOADING & FEATURE ENGINEERING
# =============================================================================

def load_and_prepare(files):
    dfs = []
    for f in files:
        dfs.append(pd.read_csv(f))
    raw = pd.concat(dfs, ignore_index=True)

    date_cols = ["FCOK_DATE", "REGD_DATE", "REPAIR_DATE", "PROCESSING_DATE"]
    for c in date_cols:
        raw[c] = pd.to_datetime(raw[c], dayfirst=True, errors="coerce")

    raw["PROC_MONTH"] = raw["PROCESSING_DATE"].dt.to_period("M")
    raw["VEHICLE_AGE_MONTHS"] = (
        (raw["REPAIR_DATE"] - raw["REGD_DATE"]).dt.days / 30.44
    ).clip(lower=0)
    raw["MFCTR_AGE_MONTHS"] = (
        (raw["REPAIR_DATE"] - raw["FCOK_DATE"]).dt.days / 30.44
    ).clip(lower=0)
    return raw


def build_monthly_series(raw, part):
    sub = raw[raw["Part Name"] == part].copy()
    monthly = (
        sub.groupby("PROC_MONTH")
        .agg(
            claim_count     = ("Part Name",           "count"),
            avg_odometer    = ("ODOMETER",             "mean"),
            avg_vehicle_age = ("VEHICLE_AGE_MONTHS",   "mean"),
            avg_mfctr_age   = ("MFCTR_AGE_MONTHS",     "mean"),
        )
        .reset_index()
        .rename(columns={"PROC_MONTH": "period"})
        .sort_values("period")
    )

    all_periods = pd.period_range(
        start=monthly["period"].min(),
        end  =monthly["period"].max(),
        freq ="M"
    )
    monthly = (
        monthly.set_index("period")
               .reindex(all_periods)
               .reset_index()
               .rename(columns={"index": "period"})
    )
    monthly["claim_count"]     = monthly["claim_count"].fillna(0)
    monthly["avg_odometer"]    = monthly["avg_odometer"].interpolate()
    monthly["avg_vehicle_age"] = monthly["avg_vehicle_age"].interpolate()
    monthly["avg_mfctr_age"]   = monthly["avg_mfctr_age"].interpolate()
    monthly["production"]      = PRODUCTION_PER_MONTH
    monthly["claim_rate"]      = monthly["claim_count"] / (PRODUCTION_PER_MONTH / 1000)
    monthly["t"]               = np.arange(len(monthly))
    monthly["sin_12"]          = np.sin(2 * np.pi * monthly["t"] / 12)
    monthly["cos_12"]          = np.cos(2 * np.pi * monthly["t"] / 12)
    monthly["sin_6"]           = np.sin(2 * np.pi * monthly["t"] / 6)
    monthly["cos_6"]           = np.cos(2 * np.pi * monthly["t"] / 6)
    return monthly.reset_index(drop=True)


def apply_countermeasure(part, future_periods):
    multipliers = np.ones(FORECAST_HORIZON)
    if part not in COUNTERMEASURES:
        return multipliers
    cm   = COUNTERMEASURES[part]
    cm_p = pd.Period(cm["month"], freq="M")
    eff  = float(cm.get("effectiveness", 0.6))
    hl   = CM_DECAY_HALF_LIFE
    for i, fp in enumerate(future_periods):
        t_after = (fp - cm_p).n
        if t_after >= 0:
            reduction = eff * (0.5 ** (t_after / hl))
            multipliers[i] = 1.0 - reduction
    return multipliers


# =============================================================================
# 2. DEEP-LEARNING MODELS (Pure NumPy)
# =============================================================================

# ---- 2a. CNN-LSTM ------------------------------------------------------------

class CnnLstmForecaster:
    """1-D CNN feature extractor + LSTM memory → Dense output."""

    def __init__(self, lookback=12, n_features=1, horizon=12,
                 lr=1e-2, epochs=80, seed=42):
        np.random.seed(seed)
        self.W = lookback
        self.F = n_features
        self.H = horizon
        self.lr = lr
        self.epochs = epochs
        self.H_lstm = 32
        self._init_weights()

    def _init_weights(self):
        # Conv1: kernel=3, F_in=F, F_out=16
        self.Wc1 = np.random.randn(3, self.F, 16) * 0.1
        self.bc1 = np.zeros(16)
        # Conv2: kernel=3, F_in=16, F_out=8
        self.Wc2 = np.random.randn(3, 16, 8) * 0.1
        self.bc2 = np.zeros(8)
        # LSTM: input_size=8, hidden=32
        H, L_in = self.H_lstm, 8
        self.Wlstm = np.random.randn(L_in + H, 4 * H) * 0.05
        self.blstm = np.zeros(4 * H)
        self.blstm[H:2*H] = 1.0  # forget bias = 1
        # Dense: H -> horizon
        self.Wd = np.random.randn(H, self.H) * 0.1
        self.bd = np.zeros(self.H)

    def _conv1d(self, x, W, b):
        k, _, C_out = W.shape
        T = x.shape[0]
        out = np.zeros((T - k + 1, C_out))
        for i in range(T - k + 1):
            out[i] = x[i:i+k].reshape(-1) @ W.reshape(-1, C_out) + b
        return out

    def _lstm_step(self, x_seq):
        H = self.H_lstm
        h, c = np.zeros(H), np.zeros(H)
        for xt in x_seq:
            combined = np.concatenate([xt, h])
            gates = combined @ self.Wlstm + self.blstm
            ig = sigmoid(gates[:H])
            fg = sigmoid(gates[H:2*H])
            g  = tanh   (gates[2*H:3*H])
            og = sigmoid(gates[3*H:])
            c  = fg * c + ig * g
            h  = og * tanh(c)
        return h

    def _forward(self, x):
        c1 = relu(self._conv1d(x, self.Wc1, self.bc1))
        c2 = relu(self._conv1d(c1, self.Wc2, self.bc2))
        h  = self._lstm_step(c2)
        return h @ self.Wd + self.bd

    def fit(self, X, y):
        opt = AdamOptimizer(self.lr)
        N   = X.shape[0]
        for ep in range(self.epochs):
            for i in np.random.permutation(N):
                xi, yi = X[i], y[i]
                pred   = self._forward(xi)
                dL     = 2 * (pred - yi) / self.H
                # Analytic dense grad
                c1 = relu(self._conv1d(xi, self.Wc1, self.bc1))
                c2 = relu(self._conv1d(c1, self.Wc2, self.bc2))
                h  = self._lstm_step(c2)
                self.Wd = opt.update("Wd", self.Wd, np.outer(h, dL))
                self.bd = opt.update("bd", self.bd, dL)
                # Numerical grad for LSTM (partial)
                eps = 1e-3
                for name in ["Wlstm", "blstm"]:
                    W_ = getattr(self, name)
                    grad = np.zeros_like(W_)
                    n_upd = min(W_.size, 128)
                    idx2  = np.random.choice(W_.size, n_upd, replace=False)
                    flat  = W_.ravel()
                    for j in idx2:
                        orig   = flat[j]
                        flat[j] = orig + eps
                        setattr(self, name, flat.reshape(W_.shape))
                        pp = self._forward(xi)
                        flat[j] = orig - eps
                        setattr(self, name, flat.reshape(W_.shape))
                        pm = self._forward(xi)
                        flat[j] = orig
                        setattr(self, name, flat.reshape(W_.shape))
                        grad.ravel()[j] = (np.mean((pp-yi)**2) - np.mean((pm-yi)**2)) / (2*eps)
                    setattr(self, name, opt.update(name, getattr(self, name), grad))

    def predict(self, X):
        return np.array([self._forward(xi) for xi in X])


# ---- 2b. N-BEATS -------------------------------------------------------------

class NBeatsBlock:
    def __init__(self, in_dim, theta_dim, H, hidden=64, seed=0):
        np.random.seed(seed)
        self.W1  = np.random.randn(in_dim,  hidden) * 0.05
        self.b1  = np.zeros(hidden)
        self.W2  = np.random.randn(hidden,  hidden) * 0.05
        self.b2  = np.zeros(hidden)
        self.W3  = np.random.randn(hidden,  hidden) * 0.05
        self.b3  = np.zeros(hidden)
        self.Wtb = np.random.randn(hidden, theta_dim) * 0.05
        self.Wtf = np.random.randn(hidden, theta_dim) * 0.05
        self.btb = np.zeros(theta_dim)
        self.btf = np.zeros(theta_dim)
        t_b = np.linspace(-1, 0, in_dim)
        t_f = np.linspace( 0, 1, H)
        self.Vb = np.vstack([t_b**k for k in range(theta_dim)]).T
        self.Vf = np.vstack([t_f**k for k in range(theta_dim)]).T

    def forward(self, x):
        h = relu(x @ self.W1 + self.b1)
        h = relu(h @ self.W2 + self.b2)
        h = relu(h @ self.W3 + self.b3)
        bc = (h @ self.Wtb + self.btb) @ self.Vb.T
        fc = (h @ self.Wtf + self.btf) @ self.Vf.T
        return bc, fc


class NBeatsForecaster:
    def __init__(self, lookback=12, n_features=1, horizon=12,
                 lr=5e-3, epochs=100, seed=42):
        np.random.seed(seed)
        self.W       = lookback * n_features
        self.H       = horizon
        self.lr      = lr
        self.epochs  = epochs
        self.blocks  = [
            NBeatsBlock(self.W, 8, horizon, hidden=64, seed=s*10+b)
            for s in range(2) for b in range(3)
        ]

    def _forward(self, x):
        residual = x.copy()
        forecast = np.zeros(self.H)
        for blk in self.blocks:
            bc, fc = blk.forward(residual)
            residual = residual - bc
            forecast = forecast + fc
        return forecast

    def fit(self, X, y):
        N      = X.shape[0]
        Xf     = X.reshape(N, -1)
        opt    = AdamOptimizer(self.lr)
        eps    = 1e-3
        for ep in range(self.epochs):
            for i in np.random.permutation(N):
                xi, yi = Xf[i], y[i]
                pred   = self._forward(xi)
                for bi, blk in enumerate(self.blocks):
                    for attr in ["Wtf", "btf"]:
                        W_   = getattr(blk, attr)
                        grad = np.zeros_like(W_)
                        n_up = min(W_.size, 40)
                        idxs = np.random.choice(W_.size, n_up, replace=False)
                        flat = W_.ravel()
                        for j in idxs:
                            orig   = flat[j]
                            flat[j] = orig + eps
                            setattr(blk, attr, flat.reshape(W_.shape))
                            pp = self._forward(xi)
                            flat[j] = orig - eps
                            setattr(blk, attr, flat.reshape(W_.shape))
                            pm = self._forward(xi)
                            flat[j] = orig
                            setattr(blk, attr, flat.reshape(W_.shape))
                            grad.ravel()[j] = (np.mean((pp-yi)**2) - np.mean((pm-yi)**2)) / (2*eps)
                        setattr(blk, attr, opt.update(f"b{bi}_{attr}", getattr(blk, attr), grad))

    def predict(self, X):
        Xf = X.reshape(X.shape[0], -1)
        return np.array([self._forward(xi) for xi in Xf])


# ---- 2c. Transformer ---------------------------------------------------------

class TransformerForecaster:
    def __init__(self, lookback=12, n_features=1, horizon=12,
                 d_model=16, n_heads=2, lr=3e-3, epochs=80, seed=42):
        np.random.seed(seed)
        self.W  = lookback
        self.F  = n_features
        self.H  = horizon
        self.dm = d_model
        self.nh = n_heads
        self.dh = d_model // n_heads
        self.lr = lr
        self.epochs = epochs
        self._init_weights()

    def _init_weights(self):
        dm = self.dm
        self.We  = np.random.randn(self.F, dm) * 0.1
        self.be  = np.zeros(dm)
        pos = np.arange(self.W)[:, None]
        div = np.exp(np.arange(0, dm, 2) * (-np.log(10000) / dm))
        self.PE       = np.zeros((self.W, dm))
        self.PE[:, 0::2] = np.sin(pos * div)
        cols_cos = np.arange(1, dm, 2)
        self.PE[:, cols_cos] = np.cos(pos * div[:len(cols_cos)])
        self.Wq = np.random.randn(self.nh, dm, self.dh) * 0.05
        self.Wk = np.random.randn(self.nh, dm, self.dh) * 0.05
        self.Wv = np.random.randn(self.nh, dm, self.dh) * 0.05
        self.Wo = np.random.randn(self.nh * self.dh, dm) * 0.05
        self.Wff1 = np.random.randn(dm, 32) * 0.05
        self.bff1 = np.zeros(32)
        self.Wff2 = np.random.randn(32, dm) * 0.05
        self.bff2 = np.zeros(dm)
        self.Wd = np.random.randn(dm, self.H) * 0.1
        self.bd = np.zeros(self.H)

    def _attn(self, x):
        heads = []
        for h in range(self.nh):
            Q = x @ self.Wq[h]
            K = x @ self.Wk[h]
            V = x @ self.Wv[h]
            sc = Q @ K.T / np.sqrt(self.dh)
            sc -= sc.max(axis=-1, keepdims=True)
            A = np.exp(sc) / (np.exp(sc).sum(axis=-1, keepdims=True) + 1e-9)
            heads.append(A @ V)
        return np.concatenate(heads, axis=-1) @ self.Wo

    def _forward(self, x):
        e  = x @ self.We + self.be + self.PE
        e  = e + self._attn(e)
        ff = relu(e @ self.Wff1 + self.bff1) @ self.Wff2 + self.bff2
        e  = e + ff
        e  = e + self._attn(e)
        return e.mean(axis=0) @ self.Wd + self.bd

    def fit(self, X, y):
        opt = AdamOptimizer(self.lr)
        N   = X.shape[0]
        eps = 1e-3
        for ep in range(self.epochs):
            for i in np.random.permutation(N):
                xi, yi = X[i], y[i]
                pred   = self._forward(xi)
                dL     = 2 * (pred - yi) / self.H
                pooled = (xi @ self.We + self.be + self.PE).mean(axis=0)
                self.Wd = opt.update("Wd", self.Wd, np.outer(pooled, dL))
                self.bd = opt.update("bd", self.bd, dL)
                for name in ["Wo", "Wff2", "We"]:
                    W_   = getattr(self, name)
                    grad = np.zeros_like(W_)
                    n_up = min(W_.size, 50)
                    idxs = np.random.choice(W_.size, n_up, replace=False)
                    flat = W_.ravel()
                    for j in idxs:
                        orig   = flat[j]
                        flat[j] = orig + eps
                        setattr(self, name, flat.reshape(W_.shape))
                        pp = self._forward(xi)
                        flat[j] = orig - eps
                        setattr(self, name, flat.reshape(W_.shape))
                        pm = self._forward(xi)
                        flat[j] = orig
                        setattr(self, name, flat.reshape(W_.shape))
                        grad.ravel()[j] = (np.mean((pp-yi)**2) - np.mean((pm-yi)**2)) / (2*eps)
                    setattr(self, name, opt.update(name, getattr(self, name), grad))

    def predict(self, X):
        return np.array([self._forward(xi) for xi in X])


# =============================================================================
# 3. ML MODELS (XGBoost, LightGBM, GradientBoosting)
# =============================================================================

def build_supervised_flat(series_vals, exog, lookback, horizon):
    N = len(series_vals) - lookback - horizon + 1
    if N <= 0:
        return None, None
    X_list, y_list = [], []
    for i in range(N):
        x_row = np.concatenate([series_vals[i:i+lookback], exog[i:i+lookback].ravel()])
        y_row = series_vals[i+lookback:i+lookback+horizon]
        X_list.append(x_row)
        y_list.append(y_row)
    return np.array(X_list), np.array(y_list)


def train_ml_models(X_train, y_train):
    models = {}
    H = y_train.shape[1]
    for h in range(H):
        yt = y_train[:, h]
        models[f"xgb_{h}"] = xgb.XGBRegressor(
            n_estimators=150, max_depth=4, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
        ).fit(X_train, yt)
        models[f"lgb_{h}"] = lgb.LGBMRegressor(
            n_estimators=150, max_depth=4, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
        ).fit(X_train, yt)
        models[f"gbr_{h}"] = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.08, random_state=42
        ).fit(X_train, yt)
    return models


def predict_ml_models(models, X, H):
    return {
        "XGBoost":   np.array([models[f"xgb_{h}"].predict(X) for h in range(H)]).T,
        "LightGBM":  np.array([models[f"lgb_{h}"].predict(X) for h in range(H)]).T,
        "GradBoost": np.array([models[f"gbr_{h}"].predict(X) for h in range(H)]).T,
    }


# =============================================================================
# 4. SARIMA BASELINE
# =============================================================================

def fit_sarima(series, horizon):
    try:
        mod = sm.tsa.statespace.SARIMAX(
            series, order=(1,1,1), seasonal_order=(1,1,0,12),
            enforce_stationarity=False, enforce_invertibility=False
        )
        res = mod.fit(disp=False, maxiter=100)
        return np.clip(res.forecast(steps=horizon), 0, None)
    except Exception:
        if len(series) >= 12:
            return np.array([series[-12+i%12] for i in range(horizon)], dtype=float)
        return np.full(horizon, series.mean())


# =============================================================================
# 5. PER-PART PIPELINE
# =============================================================================

def build_window_dataset(series_vals, exog, lookback, horizon):
    N = len(series_vals) - lookback - horizon + 1
    if N <= 0:
        return None, None
    X_list, y_list = [], []
    for i in range(N):
        win_y  = series_vals[i:i+lookback, None]
        win_ex = exog[i:i+lookback]
        X_list.append(np.concatenate([win_y, win_ex], axis=1))
        y_list.append(series_vals[i+lookback:i+lookback+horizon])
    return np.array(X_list), np.array(y_list)


def run_pipeline_for_part(monthly, part):
    print(f"  [{part}] {len(monthly)} months of data")

    claim_vals = monthly["claim_count"].values.astype(float)
    if claim_vals.sum() == 0:
        print(f"    -> SKIP (no claims)")
        return None

    scaler = MinMaxScaler()
    claim_scaled = scaler.fit_transform(claim_vals[:, None]).ravel()

    exog_cols = ["avg_odometer", "avg_vehicle_age", "avg_mfctr_age",
                 "production", "sin_12", "cos_12", "sin_6", "cos_6"]
    exog_raw = np.nan_to_num(monthly[exog_cols].values.astype(float), nan=0.0)
    exog_sc  = MinMaxScaler().fit_transform(exog_raw)

    W = min(LOOKBACK_WINDOW, len(claim_vals) - FORECAST_HORIZON - 2)
    if W < 4:
        print(f"    -> SKIP (too short)")
        return None

    X_dl, y_dl = build_window_dataset(claim_scaled, exog_sc, W, FORECAST_HORIZON)
    X_ml, y_ml = build_supervised_flat(claim_scaled, exog_sc, W, FORECAST_HORIZON)
    if X_dl is None or len(X_dl) < 4:
        print(f"    -> SKIP (insufficient samples)")
        return None

    n_val   = max(1, min(6, len(X_dl) // 4))
    X_tr_dl = X_dl[:-n_val];  X_va_dl = X_dl[-n_val:]
    y_tr_dl = y_dl[:-n_val];  y_va_dl = y_dl[-n_val:]
    X_tr_ml = X_ml[:-n_val];  X_va_ml = X_ml[-n_val:]
    y_tr_ml = y_ml[:-n_val];  y_va_ml = y_ml[-n_val:]

    F = X_dl.shape[2]

    # --- Train models
    print(f"    CNN-LSTM ...")
    cnn_lstm = CnnLstmForecaster(lookback=W, n_features=F, horizon=FORECAST_HORIZON, lr=1e-2, epochs=80)
    cnn_lstm.fit(X_tr_dl, y_tr_dl)

    print(f"    N-BEATS ...")
    nbeats = NBeatsForecaster(lookback=W, n_features=F, horizon=FORECAST_HORIZON, lr=5e-3, epochs=100)
    nbeats.fit(X_tr_dl, y_tr_dl)

    print(f"    Transformer ...")
    transformer = TransformerForecaster(lookback=W, n_features=F, horizon=FORECAST_HORIZON, d_model=16, n_heads=2, lr=3e-3, epochs=80)
    transformer.fit(X_tr_dl, y_tr_dl)

    print(f"    ML models ...")
    ml_models = train_ml_models(X_tr_ml, y_tr_ml)

    print(f"    SARIMA ...")
    sarima_fc_sc = fit_sarima(claim_scaled[:-(n_val + FORECAST_HORIZON)], FORECAST_HORIZON)

    # --- Validation MAE -> ensemble weights
    def _mae(pred, true):
        return mean_absolute_error(true.ravel(), pred.ravel())

    val_mae = {}
    try: val_mae["CNN-LSTM"]    = _mae(cnn_lstm.predict(X_va_dl),    y_va_dl)
    except: val_mae["CNN-LSTM"] = 1.0
    try: val_mae["N-BEATS"]     = _mae(nbeats.predict(X_va_dl),      y_va_dl)
    except: val_mae["N-BEATS"]  = 1.0
    try: val_mae["Transformer"] = _mae(transformer.predict(X_va_dl), y_va_dl)
    except: val_mae["Transformer"] = 1.0

    try:
        ml_val = predict_ml_models(ml_models, X_va_ml, FORECAST_HORIZON)
        for k, v in ml_val.items():
            val_mae[k] = _mae(v, y_va_ml)
    except:
        for k in ["XGBoost","LightGBM","GradBoost"]:
            val_mae[k] = 1.0

    val_mae["SARIMA"] = np.mean(list(val_mae.values()))  # neutral

    inv_w   = {k: 1.0 / (v + 1e-9) for k, v in val_mae.items()}
    total_w = sum(inv_w.values())
    weights = {k: v / total_w for k, v in inv_w.items()}

    # --- Forecast from last window
    lx_dl = X_dl[-1:]
    lx_ml = X_ml[-1:]

    fc_sc = {}
    fc_sc["CNN-LSTM"]    = cnn_lstm.predict(lx_dl)[0]
    fc_sc["N-BEATS"]     = nbeats.predict(lx_dl)[0]
    fc_sc["Transformer"] = transformer.predict(lx_dl)[0]
    ml_fc = predict_ml_models(ml_models, lx_ml, FORECAST_HORIZON)
    for k, v in ml_fc.items():
        fc_sc[k] = v[0]
    fc_sc["SARIMA"] = sarima_fc_sc

    ens_sc = sum(weights[k] * fc_sc[k] for k in fc_sc)

    def inv_sc(arr):
        return scaler.inverse_transform(np.clip(arr, 0, 1)[:, None]).ravel().clip(0)

    fc_raw = {k: inv_sc(v) for k, v in fc_sc.items()}
    ens_raw = inv_sc(ens_sc)

    last_period   = monthly["period"].iloc[-1]
    future_pds    = [last_period + i + 1 for i in range(FORECAST_HORIZON)]
    cm_mults      = apply_countermeasure(part, future_pds)
    ens_cm        = ens_raw * cm_mults
    for k in fc_raw:
        fc_raw[k] = fc_raw[k] * cm_mults

    all_fc_arr = np.array(list(fc_raw.values()))
    ci_low  = (ens_cm - 1.5 * all_fc_arr.std(axis=0)).clip(0)
    ci_high = ens_cm + 1.5 * all_fc_arr.std(axis=0)

    return {
        "part":           part,
        "monthly":        monthly,
        "claim_vals":     claim_vals,
        "hist_rate":      claim_vals / (PRODUCTION_PER_MONTH / 1000),
        "future_periods": future_pds,
        "ensemble_raw":   ens_cm,
        "forecasts_raw":  fc_raw,
        "ci_low":         ci_low,
        "ci_high":        ci_high,
        "forecast_rate":  ens_cm / (PRODUCTION_PER_MONTH / 1000),
        "weights":        weights,
        "val_mae":        val_mae,
        "cm_mults":       cm_mults,
        "has_cm":         part in COUNTERMEASURES,
    }


# =============================================================================
# 6. DASHBOARD
# =============================================================================

PALETTE = ["#6C63FF","#FF6584","#43D9AD","#FFBB35","#4FC3F7",
           "#FF8A65","#A5D6A7","#CE93D8","#80DEEA","#FFCC80"]

MODEL_COLORS = {
    "CNN-LSTM":    "#6C63FF",
    "N-BEATS":     "#FF6584",
    "Transformer": "#43D9AD",
    "XGBoost":     "#FFBB35",
    "LightGBM":    "#4FC3F7",
    "GradBoost":   "#FF8A65",
    "SARIMA":      "#A5D6A7",
}


def p2s(p):
    return str(p)


def make_part_figure(r):
    part      = r["part"]
    monthly   = r["monthly"]
    hist_x    = [p2s(p) for p in monthly["period"]]
    fut_x     = [p2s(p) for p in r["future_periods"]]

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Claims: Historical + Ensemble Forecast",
            "Model-by-Model Forecasts",
            "Claim Rate (per 1,000 vehicles/month)",
            "Ensemble Model Weights",
            "Monthly Claims Distribution",
            "Countermeasure Multiplier",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    # R1C1 — history + CI + ensemble
    fig.add_trace(go.Scatter(x=hist_x, y=r["claim_vals"],
        mode="lines+markers", name="Historical",
        line=dict(color="#6C63FF", width=2), marker=dict(size=3)), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=fut_x + fut_x[::-1],
        y=list(r["ci_high"]) + list(r["ci_low"][::-1]),
        fill="toself", fillcolor="rgba(255,187,53,0.12)",
        line=dict(color="rgba(0,0,0,0)"), name="95% CI"), row=1, col=1)
    fig.add_trace(go.Scatter(x=fut_x, y=r["ensemble_raw"],
        mode="lines+markers", name="Ensemble Forecast",
        line=dict(color="#FFBB35", width=3, dash="dot"),
        marker=dict(size=7, symbol="diamond")), row=1, col=1)

    # R1C2 — all model forecasts
    for mn, fc in r["forecasts_raw"].items():
        fig.add_trace(go.Scatter(x=fut_x, y=fc, mode="lines+markers",
            name=mn, line=dict(color=MODEL_COLORS.get(mn, "#999"), width=1.5),
            marker=dict(size=4)), row=1, col=2)
    fig.add_trace(go.Scatter(x=fut_x, y=r["ensemble_raw"], mode="lines+markers",
        name="Ensemble", line=dict(color="#FFFFFF", width=3),
        marker=dict(size=8, symbol="star")), row=1, col=2)

    # R2C1 — claim rate
    fig.add_trace(go.Scatter(x=hist_x, y=r["hist_rate"], mode="lines",
        name="Historical Rate", line=dict(color="#6C63FF", width=2),
        fill="tozeroy", fillcolor="rgba(108,99,255,0.18)"), row=2, col=1)
    fig.add_trace(go.Scatter(x=fut_x, y=r["forecast_rate"], mode="lines",
        name="Forecast Rate", line=dict(color="#FF6584", width=2, dash="dot"),
        fill="tozeroy", fillcolor="rgba(255,101,132,0.12)"), row=2, col=1)

    # R2C2 — weights bar
    wn = list(r["weights"].keys())
    wv = [r["weights"][k] for k in wn]
    fig.add_trace(go.Bar(x=wn, y=wv,
        marker_color=[MODEL_COLORS.get(k, "#888") for k in wn],
        text=[f"{v:.2f}" for v in wv], textposition="outside",
        name="Weights"), row=2, col=2)

    # R3C1 — histogram
    fig.add_trace(go.Histogram(x=r["claim_vals"], nbinsx=20,
        marker_color="#6C63FF", opacity=0.8, name="Claim Dist."), row=3, col=1)

    # R3C2 — CM multiplier
    fig.add_trace(go.Scatter(x=fut_x, y=r["cm_mults"], mode="lines+markers",
        name="CM Factor", line=dict(color="#43D9AD", width=2),
        marker=dict(size=6), fill="tozeroy",
        fillcolor="rgba(67,217,173,0.12)"), row=3, col=2)
    fig.add_hline(y=1.0, line_dash="dash", line_color="#FF6584",
                  annotation_text="No CM", row=3, col=2)

    fig.update_layout(
        height=920,
        paper_bgcolor="#0F0F1A",
        plot_bgcolor="#1A1A2E",
        font=dict(color="#E0E0E0", size=11, family="Inter, sans-serif"),
        title_text=f"<b>Warranty Forecast — {part}</b>",
        title_font_size=20,
        showlegend=True,
        legend=dict(bgcolor="rgba(255,255,255,0.04)",
                    bordercolor="rgba(255,255,255,0.08)",
                    font_size=9),
        margin=dict(l=40, r=40, t=80, b=40),
    )
    fig.update_xaxes(showgrid=False, tickangle=-30)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)")
    return fig


def build_dashboard(results, raw):
    parts = [r["part"] for r in results]

    # --- Summary charts ---
    donut_fig = go.Figure(go.Pie(
        labels=[r["part"] for r in results],
        values=[r["claim_vals"].sum() for r in results],
        hole=0.55, textinfo="label+percent",
        marker=dict(colors=PALETTE * 5), textfont_size=10,
    ))
    donut_fig.update_layout(
        title_text="Total Historical Claims by Part", title_font_size=16,
        paper_bgcolor="#0F0F1A", plot_bgcolor="#0F0F1A",
        font_color="#E0E0E0", height=400, margin=dict(l=20,r=20,t=50,b=20),
    )

    bar_fig = go.Figure(go.Bar(
        x=[r["part"] for r in results],
        y=[r["ensemble_raw"].sum() for r in results],
        marker_color=PALETTE[:len(results)],
        text=[f"{r['ensemble_raw'].sum():.0f}" for r in results],
        textposition="outside",
    ))
    bar_fig.update_layout(
        title_text="Forecasted Claims (Next 12 Months)", title_font_size=16,
        xaxis_title="Part", yaxis_title="Count",
        paper_bgcolor="#0F0F1A", plot_bgcolor="#1A1A2E",
        font_color="#E0E0E0", height=400,
        margin=dict(l=20,r=20,t=50,b=80), xaxis=dict(tickangle=-30),
    )

    model_names = ["CNN-LSTM","N-BEATS","Transformer","XGBoost","LightGBM","GradBoost","SARIMA"]
    wmat = [[r["weights"].get(m,0) for m in model_names] for r in results]
    heat_fig = go.Figure(go.Heatmap(
        z=wmat, x=model_names, y=[r["part"] for r in results],
        colorscale="Viridis",
        text=[[f"{v:.2f}" for v in row] for row in wmat],
        texttemplate="%{text}",
        colorbar=dict(title="Weight"),
    ))
    heat_fig.update_layout(
        title_text="Ensemble Weights — Model × Part", title_font_size=16,
        paper_bgcolor="#0F0F1A", plot_bgcolor="#1A1A2E",
        font_color="#E0E0E0", height=420,
        margin=dict(l=100,r=20,t=50,b=80),
    )

    # ---- helper: fig -> html div
    def to_div(fig, div_id):
        return pio.to_html(fig, full_html=False, include_plotlyjs=False,
                           div_id=div_id, config={"responsive": True})

    # ---- metrics
    total_hist = sum(r["claim_vals"].sum() for r in results)
    total_fc   = sum(r["ensemble_raw"].sum() for r in results)
    pct_chg    = (total_fc - total_hist) / (total_hist + 1e-9) * 100
    n_cm       = sum(1 for r in results if r["has_cm"])

    metrics_html = f"""
<div class="metrics-grid">
  <div class="metric-card">
    <div class="metric-icon">📦</div>
    <div class="metric-value">{int(total_hist):,}</div>
    <div class="metric-label">Historical Claims</div>
  </div>
  <div class="metric-card">
    <div class="metric-icon">🔮</div>
    <div class="metric-value">{int(total_fc):,}</div>
    <div class="metric-label">Forecasted (12mo)</div>
  </div>
  <div class="metric-card {'metric-card--up' if pct_chg>0 else 'metric-card--down'}">
    <div class="metric-icon">{'📈' if pct_chg>0 else '📉'}</div>
    <div class="metric-value">{pct_chg:+.1f}%</div>
    <div class="metric-label">Trend vs History</div>
  </div>
  <div class="metric-card">
    <div class="metric-icon">🛡️</div>
    <div class="metric-value">{n_cm}</div>
    <div class="metric-label">Active Countermeasures</div>
  </div>
  <div class="metric-card">
    <div class="metric-icon">🔩</div>
    <div class="metric-value">{len(results)}</div>
    <div class="metric-label">Parts Analysed</div>
  </div>
  <div class="metric-card">
    <div class="metric-icon">🏭</div>
    <div class="metric-value">{PRODUCTION_PER_MONTH:,}</div>
    <div class="metric-label">Vehicles/Month</div>
  </div>
</div>"""

    # ---- tab nav
    tab_nav = '<div class="tab-nav"><button class="tab-btn active" onclick="showTab(\'summary\',this)">📊 Summary</button>'
    for r in results:
        pid = r["part"].replace(" ", "_")
        cm_badge = ' <span class="cm-badge">CM</span>' if r["has_cm"] else ""
        tab_nav += f'<button class="tab-btn" onclick="showTab(\'{pid}\',this)">🔩 {r["part"]}{cm_badge}</button>'
    tab_nav += "</div>"

    # ---- summary tab
    tabs_html = f"""
<div id="tab-summary" class="tab-content active">
  {metrics_html}
  <div class="chart-grid">
    <div class="chart-box">{to_div(donut_fig,"donut")}</div>
    <div class="chart-box">{to_div(bar_fig,"bar")}</div>
  </div>
  <div class="chart-box chart-full">{to_div(heat_fig,"heat")}</div>
</div>"""

    # ---- per-part tabs
    for r in results:
        pid  = r["part"].replace(" ","_")
        part = r["part"]
        fig  = make_part_figure(r)
        fut_x = [p2s(p) for p in r["future_periods"]]
        rows = ""
        for i, fp in enumerate(r["future_periods"]):
            mid  = int(r["ensemble_raw"][i])
            lo   = max(0, int(r["ci_low"][i]))
            hi   = int(r["ci_high"][i])
            rate = f"{r['forecast_rate'][i]:.2f}"
            cm_s = f"{r['cm_mults'][i]*100:.0f}%" if r["has_cm"] else "—"
            rows += f"""<tr>
              <td>{p2s(fp)}</td>
              <td class="td-r">{mid:,}</td>
              <td class="td-r td-lo">{lo:,}</td>
              <td class="td-r td-hi">{hi:,}</td>
              <td class="td-r">{rate}</td>
              <td class="td-r">{cm_s}</td>
            </tr>"""
        tabs_html += f"""
<div id="tab-{pid}" class="tab-content">
  <div class="part-header">
    <h2>🔩 {part}</h2>
    {'<span class="cm-active">✅ Countermeasure Active</span>' if r["has_cm"] else ""}
  </div>
  <div class="chart-box chart-full">{to_div(fig, f"fig_{pid}")}</div>
  <div class="fc-table-wrap">
    <h3>📅 12-Month Forecast Detail</h3>
    <table class="fc-table">
      <thead><tr>
        <th>Month</th><th>Forecast</th><th>CI Low</th>
        <th>CI High</th><th>Rate/1k</th><th>CM Factor</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
</div>"""

    # ---- assemble full page
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Automotive Warranty Forecasting Dashboard</title>
<meta name="description" content="Automotive warranty claims forecasting using CNN-LSTM, N-BEATS, Transformer, XGBoost, LightGBM and ensemble methods.">
<script src="https://cdn.plot.ly/plotly-3.0.0.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{{
  --bg0:#0F0F1A; --bg1:#1A1A2E; --bg2:#16213E;
  --a1:#6C63FF; --a2:#43D9AD; --a3:#FFBB35; --a4:#FF6584;
  --txt:#E0E0E0; --muted:#888;
  --border:rgba(255,255,255,0.07);
  --r:12px;
}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Inter',sans-serif;background:var(--bg0);color:var(--txt);min-height:100vh}}

/* header */
.hdr{{
  background:linear-gradient(135deg,var(--bg0) 0%,var(--bg1) 60%,var(--bg2) 100%);
  border-bottom:1px solid var(--border);
  padding:22px 36px;display:flex;align-items:center;gap:18px
}}
.hdr-logo{{
  width:46px;height:46px;border-radius:var(--r);
  background:linear-gradient(135deg,var(--a1),var(--a2));
  display:flex;align-items:center;justify-content:center;font-size:22px
}}
.hdr-title h1{{
  font-size:20px;font-weight:700;
  background:linear-gradient(135deg,var(--a1),var(--a2));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent
}}
.hdr-title p{{font-size:11px;color:var(--muted);margin-top:3px}}
.hdr-meta{{margin-left:auto;text-align:right;font-size:11px;color:var(--muted)}}
.hdr-meta span{{display:block;color:var(--a2);font-weight:500}}

/* tabs */
.tab-nav{{
  display:flex;flex-wrap:wrap;gap:5px;
  padding:14px 36px;background:var(--bg1);
  border-bottom:1px solid var(--border);
  position:sticky;top:0;z-index:100
}}
.tab-btn{{
  padding:7px 14px;border-radius:7px;
  border:1px solid var(--border);
  background:transparent;color:var(--muted);
  cursor:pointer;font-family:inherit;font-size:12px;font-weight:500;
  transition:all .18s ease;white-space:nowrap
}}
.tab-btn:hover{{background:rgba(108,99,255,.14);color:var(--txt);border-color:var(--a1)}}
.tab-btn.active{{
  background:linear-gradient(135deg,var(--a1),rgba(108,99,255,.65));
  color:#fff;border-color:transparent;font-weight:600
}}
.cm-badge{{
  display:inline-block;background:var(--a2);color:#000;
  font-size:8px;padding:1px 4px;border-radius:3px;font-weight:700;margin-left:3px
}}

/* main */
.main{{padding:28px 36px}}
.tab-content{{display:none}}
.tab-content.active{{display:block}}

/* metrics */
.metrics-grid{{
  display:grid;grid-template-columns:repeat(auto-fill,minmax(155px,1fr));
  gap:14px;margin-bottom:24px
}}
.metric-card{{
  background:var(--bg2);border:1px solid var(--border);
  border-radius:var(--r);padding:18px;text-align:center;
  position:relative;overflow:hidden;
  transition:transform .18s,box-shadow .18s
}}
.metric-card::before{{
  content:'';position:absolute;top:0;left:0;right:0;height:3px;
  background:linear-gradient(90deg,var(--a1),var(--a2))
}}
.metric-card--up::before{{background:linear-gradient(90deg,var(--a4),#FF8A65)}}
.metric-card--down::before{{background:linear-gradient(90deg,var(--a2),#2ECC71)}}
.metric-card:hover{{transform:translateY(-3px);box-shadow:0 8px 24px rgba(108,99,255,.18)}}
.metric-icon{{font-size:22px;margin-bottom:7px}}
.metric-value{{
  font-size:24px;font-weight:700;
  background:linear-gradient(135deg,var(--a1),var(--a2));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent
}}
.metric-label{{font-size:10px;color:var(--muted);margin-top:3px}}

/* charts */
.chart-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}}
.chart-box{{background:var(--bg2);border:1px solid var(--border);border-radius:var(--r);overflow:hidden}}
.chart-full{{margin-bottom:16px}}

/* part header */
.part-header{{display:flex;align-items:center;gap:14px;margin-bottom:16px}}
.part-header h2{{font-size:20px;font-weight:700}}
.cm-active{{
  background:linear-gradient(135deg,var(--a2),#2ECC71);color:#000;
  padding:4px 12px;border-radius:20px;font-size:11px;font-weight:600
}}

/* forecast table */
.fc-table-wrap{{
  margin-top:20px;background:var(--bg2);
  border:1px solid var(--border);border-radius:var(--r);padding:18px
}}
.fc-table-wrap h3{{font-size:13px;font-weight:600;margin-bottom:14px;color:var(--a2)}}
.fc-table{{width:100%;border-collapse:collapse;font-size:13px}}
.fc-table th{{
  background:rgba(108,99,255,.14);padding:9px 13px;text-align:left;
  font-weight:600;font-size:10px;text-transform:uppercase;
  letter-spacing:.5px;color:var(--a1);border-bottom:1px solid var(--border)
}}
.fc-table td{{padding:8px 13px;border-bottom:1px solid rgba(255,255,255,.03)}}
.fc-table tr:hover td{{background:rgba(255,255,255,.02)}}
.td-r{{text-align:right;font-variant-numeric:tabular-nums}}
.td-lo{{color:var(--a2)}} .td-hi{{color:var(--a4)}}

/* footer */
.footer{{
  text-align:center;padding:20px 36px;
  font-size:10px;color:var(--muted);
  border-top:1px solid var(--border);margin-top:36px
}}

@media(max-width:768px){{
  .hdr,.tab-nav,.main{{padding:14px}}
  .chart-grid{{grid-template-columns:1fr}}
}}
</style>
</head>
<body>
<header class="hdr">
  <div class="hdr-logo">🚗</div>
  <div class="hdr-title">
    <h1>Automotive Warranty Claims Forecasting</h1>
    <p>CNN-LSTM &middot; N-BEATS &middot; Transformer &middot; XGBoost &middot; LightGBM &middot; GradBoost &middot; SARIMA &middot; Ensemble</p>
  </div>
  <div class="hdr-meta">
    Generated: {now_str}<br>
    <span>{len(results)} Parts &middot; {FORECAST_HORIZON}-Month Horizon</span>
  </div>
</header>
{tab_nav}
<main class="main">{tabs_html}</main>
<footer class="footer">
  Automotive Warranty Forecasting &mdash; Production baseline {PRODUCTION_PER_MONTH:,} vehicles/month &mdash;
  CM decay half-life {CM_DECAY_HALF_LIFE} months
</footer>
<script>
function showTab(id, btn){{
  document.querySelectorAll('.tab-content').forEach(el=>el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el=>el.classList.remove('active'));
  document.getElementById('tab-'+id).classList.add('active');
  btn.classList.add('active');
  window.dispatchEvent(new Event('resize'));
}}
</script>
</body>
</html>"""
    return html


# =============================================================================
# 7. MAIN
# =============================================================================

def main():
    print("=" * 68)
    print("  AUTOMOTIVE WARRANTY CLAIMS FORECASTING SYSTEM")
    print("=" * 68)

    print("\n[1/4] Loading data ...")
    raw   = load_and_prepare(DATA_FILES)
    parts = sorted(raw["Part Name"].dropna().unique())
    print(f"  Records: {len(raw):,}  |  Parts: {len(parts)}")

    print("\n[2/4] Training models per part ...")
    results = []
    for part in parts:
        monthly = build_monthly_series(raw, part)
        res = run_pipeline_for_part(monthly, part)
        if res is not None:
            results.append(res)

    print(f"\n  Done: {len(results)}/{len(parts)} parts processed")

    print("\n[3/4] Building interactive dashboard ...")
    html = build_dashboard(results, raw)

    print(f"\n[4/4] Saving → {OUTPUT_HTML}")
    with open(OUTPUT_HTML, "w", encoding="utf-8") as fh:
        fh.write(html)

    print("\n" + "=" * 68)
    print(f"  ✅  Dashboard saved: {OUTPUT_HTML}")
    print(f"  Parts : {len(results)}   Horizon : {FORECAST_HORIZON} months")
    print(f"  Countermeasures active : {len(COUNTERMEASURES)}")
    print("=" * 68)


if __name__ == "__main__":
    main()
