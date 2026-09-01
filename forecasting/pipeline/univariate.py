"""
forecasting/pipeline/univariate.py
----------------------------------
Independent claims-only univariate analysis + forecasting.

Does not use production or other multivariate exogenous drivers.
Does not modify the multivariate pipeline in runner.py.
"""

from __future__ import annotations

import itertools
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from forecasting.config import FORECAST_HORIZON, LOOKBACK_WINDOW, N_CV_FOLDS, RANDOM_SEED
from forecasting.data.loader import build_monthly_series
from forecasting.metrics import compute_metrics
from forecasting.models.cnn_lstm import CnnLstmForecaster
from forecasting.models.ml_models import fit_sarima
from forecasting.models.nbeats import NBeatsForecaster
from forecasting.models.transformer import TransformerForecaster
from forecasting.pipeline.runner import build_window_dataset

logger = logging.getLogger(__name__)

UNI_MODELS = ["Holt-Winters", "SARIMA", "CNN-LSTM", "Transformer", "N-BEATS"]

UNI_FEATURE_COLS = [
    "Lag_1", "Lag_2", "Lag_3", "Lag_12",
    "Rolling_Mean_3", "Rolling_Mean_6", "Rolling_Mean_12",
    "Rolling_Std_3", "Rolling_Std_6",
    "Month", "Quarter", "Year",
    "Trend_Index", "Seasonality_Index",
]


def _period_list(monthly: pd.DataFrame) -> list:
    return list(monthly["period"])


def build_univariate_features(claims: np.ndarray, periods) -> pd.DataFrame:
    """Lag / rolling / calendar / trend / seasonality features from claims only."""
    s = pd.Series(np.asarray(claims, dtype=float).ravel())
    n = len(s)
    periods = list(periods)
    months = np.array([int(pd.Period(p, freq="M").month) for p in periods], dtype=float)
    years = np.array([int(pd.Period(p, freq="M").year) for p in periods], dtype=float)
    quarters = np.array([int(pd.Period(p, freq="M").quarter) for p in periods], dtype=float)

    df = pd.DataFrame({
        "claims": s.values,
        "Lag_1": s.shift(1),
        "Lag_2": s.shift(2),
        "Lag_3": s.shift(3),
        "Lag_12": s.shift(12),
        "Rolling_Mean_3": s.rolling(3, min_periods=1).mean(),
        "Rolling_Mean_6": s.rolling(6, min_periods=1).mean(),
        "Rolling_Mean_12": s.rolling(12, min_periods=1).mean(),
        "Rolling_Std_3": s.rolling(3, min_periods=2).std(),
        "Rolling_Std_6": s.rolling(6, min_periods=2).std(),
        "Month": months,
        "Quarter": quarters,
        "Year": years,
        "Trend_Index": np.arange(n, dtype=float),
    })
    seas = s.groupby(months).transform("mean") / (float(s.mean()) + 1e-9)
    df["Seasonality_Index"] = seas.values
    df = df.bfill().ffill().fillna(0.0)
    return df


def _data_overview(monthly: pd.DataFrame, part: str) -> dict:
    periods = _period_list(monthly)
    if not periods:
        return {
            "Part Name": part, "Total Records": 0, "Start Month": "—",
            "End Month": "—", "Missing Months": 0, "Data Quality Status": "Empty",
        }
    start, end = periods[0], periods[-1]
    expected = pd.period_range(start, end, freq="M")
    have = set(pd.Period(p, freq="M") for p in periods)
    missing = [str(p) for p in expected if p not in have]
    claims = monthly["claim_count"].to_numpy(dtype=float)
    status = "Good"
    if missing:
        status = "Gaps filled"
    if len(claims) < 18:
        status = "Short series"
    if float(np.std(claims)) < 1e-9:
        status = "Flat / no variation"
    return {
        "Part Name": part,
        "Total Records": int(len(claims)),
        "Start Month": str(start),
        "End Month": str(end),
        "Missing Months": int(len(missing)),
        "Data Quality Status": status,
        "missing_list": missing,
    }


def _stats_summary(claims: np.ndarray) -> dict:
    x = np.asarray(claims, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {}
    ser = pd.Series(x)
    modes = ser.mode()
    mode_v = float(modes.iloc[0]) if len(modes) else float("nan")
    try:
        from scipy import stats
        skew = float(stats.skew(x, bias=False)) if len(x) > 2 else float("nan")
        kurt = float(stats.kurtosis(x, bias=False, fisher=True)) if len(x) > 3 else float("nan")
    except Exception:
        skew = float(ser.skew())
        kurt = float(ser.kurt())
    return {
        "Mean": float(np.mean(x)),
        "Median": float(np.median(x)),
        "Mode": mode_v,
        "Std Deviation": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "Variance": float(np.var(x, ddof=1)) if len(x) > 1 else 0.0,
        "Min": float(np.min(x)),
        "Max": float(np.max(x)),
        "Skewness": skew,
        "Kurtosis": kurt,
    }


def _trend_stats(claims: np.ndarray) -> dict:
    x = np.asarray(claims, dtype=float).ravel()
    n = len(x)
    if n < 3:
        return {"growth_pct": 0.0, "decline_pct": 0.0, "direction": "Insufficient data",
                "vol_dir": "—", "chg_6m": 0.0}
    t = np.arange(n, dtype=float)
    slope = float(np.polyfit(t, x, 1)[0])
    diffs = np.diff(x)
    up = float(np.sum(np.clip(diffs, 0, None)))
    down = float(np.sum(np.clip(-diffs, 0, None)))
    tot = up + down + 1e-9
    if slope > 0.05 * (np.mean(x) + 1e-9):
        direction = "Increasing"
    elif slope < -0.05 * (np.mean(x) + 1e-9):
        direction = "Decreasing"
    else:
        direction = "Stable"
    k = min(6, n)
    chg_6 = (float(np.mean(x[-k:])) - float(np.mean(x[:k]))) / (abs(float(np.mean(x[:k]))) + 1e-9) * 100.0
    if n >= 8:
        vol_recent = float(np.std(x[-6:]))
        vol_early = float(np.std(x[:6]))
        vol_dir = "increasing" if vol_recent > vol_early * 1.1 else (
            "decreasing" if vol_recent < vol_early * 0.9 else "stable"
        )
    else:
        vol_dir = "—"
    return {
        "growth_pct": up / tot * 100.0,
        "decline_pct": down / tot * 100.0,
        "direction": direction,
        "vol_dir": vol_dir,
        "chg_6m": chg_6,
        "slope": slope,
    }


def _holt_winters_forecast(train: np.ndarray, horizon: int, seasonal_periods: int = 12):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    y = np.asarray(train, dtype=float).ravel()
    n = len(y)
    sp = min(seasonal_periods, max(2, n // 2))
    seasonal = "add" if n >= sp * 2 else None
    alphas = [0.1, 0.3, 0.5, 0.8]
    betas = [0.05, 0.2, 0.4]
    gammas = [0.1, 0.3, 0.5] if seasonal else [None]
    hold = max(3, min(6, n // 4))
    tr, va = y[:-hold], y[-hold:]

    def _make_hw(series, a, b, g):
        use_seas = seasonal and len(series) >= sp * 2
        kw = dict(trend="add", initialization_method="estimated")
        if use_seas:
            kw["seasonal"] = "add"
            kw["seasonal_periods"] = sp
        mod = ExponentialSmoothing(series, **kw)
        fit_kw = {"smoothing_level": a, "smoothing_trend": b, "optimized": False}
        if use_seas and g is not None:
            fit_kw["smoothing_seasonal"] = g
        return mod.fit(**fit_kw)

    best = {"alpha": 0.3, "beta": 0.1, "gamma": 0.1, "rmse": float("inf")}
    for a, b, g in itertools.product(alphas, betas, gammas):
        try:
            res = _make_hw(tr, a, b, g)
            pred = np.asarray(res.forecast(hold), dtype=float)
            rmse = float(np.sqrt(np.mean((va - pred[:len(va)]) ** 2)))
            if rmse < best["rmse"]:
                best = {
                    "alpha": a, "beta": b,
                    "gamma": float(g) if g is not None else 0.0,
                    "rmse": rmse,
                }
        except Exception:
            continue

    try:
        res = _make_hw(y, best["alpha"], best["beta"],
                       best["gamma"] if seasonal else None)
        fc = np.clip(np.asarray(res.forecast(horizon), dtype=float), 0, None)
        fitted = np.asarray(res.fittedvalues, dtype=float)
        m = min(len(fitted), len(y))
        sigma = float(np.nanstd(fitted[:m] - y[:m])) if m else float(np.std(y))
    except Exception:
        fc = np.full(horizon, float(y[-1]) if len(y) else 0.0)
        sigma = float(np.std(y)) if len(y) else 1.0

    if not np.isfinite(best.get("rmse", np.inf)):
        best["rmse"] = float("nan")
    return fc, best, max(sigma, 1e-6)


def _sarima_univariate(train: np.ndarray, horizon: int, true_hold: np.ndarray | None = None):
    orders = [(1, 1, 1), (1, 1, 0), (0, 1, 1), (2, 1, 1)]
    seas = [(1, 1, 0, 12), (0, 1, 1, 12), (1, 0, 1, 12)]
    y = np.asarray(train, dtype=float).ravel()
    hold = true_hold if true_hold is not None and len(true_hold) else y[-min(4, len(y)):]
    best_p = {"order": (1, 1, 1), "seasonal_order": (1, 1, 0, 12)}
    best_rmse = float("inf")
    split = max(8, len(y) - len(hold))
    for order, so in itertools.product(orders, seas):
        try:
            fc = fit_sarima(y[:split], len(y) - split, order=order, seasonal_order=so)
            actual = y[split:]
            m = min(len(fc), len(actual))
            if m == 0:
                continue
            rmse = float(np.sqrt(np.mean((actual[:m] - fc[:m]) ** 2)))
            if rmse < best_rmse:
                best_rmse = rmse
                best_p = {"order": order, "seasonal_order": so}
        except Exception:
            continue
    fc = fit_sarima(y, horizon, order=best_p["order"], seasonal_order=best_p["seasonal_order"])
    return np.clip(np.asarray(fc, dtype=float).ravel()[:horizon], 0, None), best_p


def _resolve_uni_models(selected: list[str] | None) -> list[str]:
    if not selected:
        return list(UNI_MODELS)
    picked = [m for m in UNI_MODELS if m in selected]
    return picked or list(UNI_MODELS)


def _walk_forward_1step(claims: np.ndarray, feats: np.ndarray, n_folds: int,
                        model_names: list[str] | None = None) -> dict:
    """1-step OOS preds per selected model for ranking (up to 6 test months).

    DL models (CNN-LSTM, Transformer, N-BEATS) are pre-filled with a naive
    fallback at the start of every fold so ``len(oos[m]) == len(actual)``
    always holds.  A successful training run overwrites the fallback in-place.
    """
    y = np.asarray(claims, dtype=float).ravel()
    T = len(y)
    names = _resolve_uni_models(model_names)
    n_folds = min(6, max(2, T - 8))
    oos = {m: [] for m in names}
    actual = []
    dl_names = [m for m in ("CNN-LSTM", "Transformer", "N-BEATS") if m in names]

    for k in range(n_folds):
        val_idx = T - n_folds + k
        train_end = val_idx
        if train_end < 8:
            continue
        actual.append(float(y[val_idx]))
        tr = y[:train_end]

        # ── Statistical models ───────────────────────────────────────────
        if "Holt-Winters" in names:
            try:
                hw_fc, _, _ = _holt_winters_forecast(tr, 1)
                oos["Holt-Winters"].append(float(hw_fc[0]))
            except Exception:
                oos["Holt-Winters"].append(float(tr[-1]))

        if "SARIMA" in names:
            try:
                s_fc, _ = _sarima_univariate(tr, 1)
                oos["SARIMA"].append(float(s_fc[0]))
            except Exception:
                oos["SARIMA"].append(float(tr[-1]))

        # ── Deep-learning models: pre-fill fallback first ────────────────
        # This guarantees len(oos[name]) == len(actual) regardless of
        # whether training succeeds or data is too short.
        fold_idx = len(actual) - 1   # index of the value just appended
        for name in dl_names:
            oos[name].append(float(tr[-1]))

        if not dl_names:
            continue

        try:
            csc = MinMaxScaler()
            esc = MinMaxScaler()
            ys = csc.fit_transform(tr[:, None]).ravel()
            es = esc.fit_transform(feats[:train_end])
            W = min(LOOKBACK_WINDOW, max(4, train_end // 3))
            X, Y = build_window_dataset(ys, es, W, 1)
            last = np.concatenate([ys[-W:, None], es[-W:]], axis=1)[None, ...]
            nf = last.shape[-1]
            ctors = {
                "CNN-LSTM": CnnLstmForecaster,
                "Transformer": TransformerForecaster,
                "N-BEATS": NBeatsForecaster,
            }
            if X is not None and len(X) >= 4:
                # ── Parallel DL model training per fold ─────────────────
                n_workers = min(len(dl_names), os.cpu_count() or 1)

                def _train_one_dl(name: str) -> tuple[str, float]:
                    mdl = ctors[name](lookback=W, n_features=nf, horizon=1, epochs=12, seed=RANDOM_SEED)
                    mdl.fit(X, Y)
                    p = float(mdl.predict(last)[0, 0])
                    val = float(csc.inverse_transform([[np.clip(p, 0, 1)]])[0, 0])
                    return name, val

                with ThreadPoolExecutor(max_workers=n_workers) as pool:
                    futures = {pool.submit(_train_one_dl, n): n for n in dl_names}
                    for fut in as_completed(futures):
                        try:
                            dl_name, val = fut.result()
                            # Overwrite the pre-filled fallback with the real prediction
                            oos[dl_name][fold_idx] = val
                        except Exception:
                            pass  # keep the pre-filled fallback
            else:
                pass # fallbacks already pre-filled above
        except Exception:
            pass  # fallbacks already pre-filled above

    actual = np.asarray(actual, dtype=float)
    metrics = {}
    for m in names:
        pred = np.asarray(oos[m][:len(actual)], dtype=float)
        if len(pred) != len(actual) or len(actual) == 0:
            metrics[m] = {"RMSE": float("nan"), "MAE": float("nan"), "MAPE": float("nan"), "R2": float("nan")}
        else:
            metrics[m] = compute_metrics(actual, pred)
    return {"actual": actual, "preds": oos, "metrics": metrics}


def _refit_dl_full(claims: np.ndarray, feats: np.ndarray, horizon: int,
                   model_names: list[str] | None = None) -> dict:
    y = np.asarray(claims, dtype=float).ravel()
    csc = MinMaxScaler()
    esc = MinMaxScaler()
    ys = csc.fit_transform(y[:, None]).ravel()
    es = esc.fit_transform(feats)
    W = min(LOOKBACK_WINDOW, max(4, len(y) // 3))
    X, Yh = build_window_dataset(ys, es, W, min(horizon, max(1, len(y) - W - 1)))
    H = min(horizon, Yh.shape[1] if Yh is not None else 1)
    last = np.concatenate([ys[-W:, None], es[-W:]], axis=1)[None, ...]
    nf = last.shape[-1]
    out = {}
    if X is None or len(X) < 3:
        naive = np.full(horizon, float(y[-1]))
        names = _resolve_uni_models(model_names)
        return {m: naive.copy() for m in names if m in ("CNN-LSTM", "Transformer", "N-BEATS")}

    Y = Yh[:, :H]
    names = _resolve_uni_models(model_names)
    specs = [
        spec for spec in (
            ("CNN-LSTM", CnnLstmForecaster, 22),
            ("Transformer", TransformerForecaster, 22),
            ("N-BEATS", NBeatsForecaster, 22),
        )
        if spec[0] in names
    ]
    if not specs:
        return {}

    # ── Parallel full-refit of all DL models ─────────────────────────────
    n_workers = min(len(specs), os.cpu_count() or 1)

    def _refit_one(spec: tuple) -> tuple[str, np.ndarray]:
        name, ctor, ep = spec
        try:
            mdl = ctor(lookback=W, n_features=nf, horizon=H, epochs=ep, seed=RANDOM_SEED)
            mdl.fit(X, Y)
            pred_sc = np.clip(mdl.predict(last)[0], 0, 1)
            pred = csc.inverse_transform(pred_sc.reshape(-1, 1)).ravel()
            if len(pred) < horizon:
                extra = np.full(horizon - len(pred), float(pred[-1]))
                pred = np.concatenate([pred, extra])
            return name, np.clip(pred[:horizon], 0, None)
        except Exception as exc:
            logger.warning("Univariate %s refit failed: %s", name, exc)
            return name, np.full(horizon, float(y[-1]))

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_refit_one, spec): spec[0] for spec in specs}
        for fut in as_completed(futures):
            name, pred = fut.result()
            out[name] = pred
    return out


def generate_insights(overview: dict, stats: dict, trend: dict, best_model: str,
                      horizon: int, ranking: pd.DataFrame) -> list[str]:
    lines = []
    d = trend.get("direction", "Stable")
    if d == "Increasing":
        lines.append("Increasing trend detected.")
    elif d == "Decreasing":
        lines.append("Decreasing trend detected.")
    else:
        lines.append("Stable trend detected.")

    seas = float(stats.get("Std Deviation", 0) or 0) / (abs(float(stats.get("Mean", 1) or 1)) + 1e-9)
    if seas > 0.35:
        lines.append("Seasonal / high-variation behavior detected.")
    else:
        lines.append("Seasonal swings appear moderate.")

    chg = trend.get("chg_6m", 0.0)
    lines.append(f"Claims changed by {chg:+.1f}% comparing the first vs last 6 months.")
    lines.append(f"Volatility is {trend.get('vol_dir', '—')}.")
    lines.append(f"Best performing univariate model based on RMSE: **{best_model}**.")
    lines.append(
        f"Forecast indicates expected claims trend for the next {horizon} months "
        f"(claims-only, no production features)."
    )
    if overview.get("Data Quality Status") not in ("Good",):
        lines.append(f"Data quality note: {overview.get('Data Quality Status')}.")
    if ranking is not None and not ranking.empty:
        top = ranking.iloc[0]
        lines.append(
            f"{top['Model']} RMSE={top['RMSE']:.2f}, MAE={top['MAE']:.2f}, MAPE={top['MAPE']:.1f}%."
        )
    return lines


def monthly_claims_template_csv(path: str | None = None) -> str:
    """Write a CSV template: Part Name, Month, Claims (one row per part-month)."""
    import os
    import tempfile

    months = [f"{y}-{m:02d}" for y in (2022, 2023) for m in range(1, 13)]
    p1 = [42, 38, 45, 51, 47, 40, 36, 41, 44, 48, 52, 49,
          46, 41, 48, 54, 50, 43, 39, 44, 47, 51, 55, 52]
    p2 = [22, 25, 21, 28, 24, 20, 19, 23, 26, 27, 30, 29,
          24, 27, 23, 30, 26, 22, 21, 25, 28, 29, 32, 31]
    sample = pd.DataFrame({
        "Part Name": ["Part_1"] * 24 + ["Part_2"] * 24,
        "Month": months + months,
        "Claims": p1 + p2,
    })
    if path is None:
        fd, path = tempfile.mkstemp(suffix=".csv", prefix="univariate_monthly_claims_")
        os.close(fd)
    sample.to_csv(path, index=False)
    return path


def load_monthly_claims_csv(path: str) -> pd.DataFrame:
    """
    Load Part Name + monthly claims CSV.

    Required columns (aliases allowed): Part Name, Month, Claims.
    """
    path_l = str(path).lower()
    if path_l.endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Uploaded monthly claims file is empty.")
    col_map = {str(c).lower().strip(): c for c in df.columns}
    rename = {}
    for canon, aliases in {
        "Part Name": ["part name", "part", "part_name", "partname"],
        "Month": ["month", "period", "date", "claim_month", "year_month"],
        "Claims": ["claims", "claim_count", "claim", "count", "volume", "part_failure"],
    }.items():
        if canon in df.columns:
            continue
        for a in aliases:
            if a in col_map:
                rename[col_map[a]] = canon
                break
    if rename:
        df = df.rename(columns=rename)
    missing = [c for c in ("Part Name", "Month", "Claims") if c not in df.columns]
    if missing:
        raise ValueError(
            "CSV must include Part Name, Month, and Claims "
            f"(missing: {missing}). Download the template, fill it, then upload."
        )
    out = df[["Part Name", "Month", "Claims"]].copy()
    out["Part Name"] = out["Part Name"].astype(str).str.strip()
    out["Month"] = out["Month"].astype(str).str.strip()
    out["Claims"] = pd.to_numeric(out["Claims"], errors="coerce").fillna(0.0)
    out = out[out["Part Name"].ne("") & out["Part Name"].ne("nan")]
    if out.empty:
        raise ValueError("No valid Part Name / Month / Claims rows found.")
    return out.reset_index(drop=True)


def monthly_from_claims_sheet(sheet: pd.DataFrame, part: str) -> pd.DataFrame:
    """Build a contiguous monthly frame (period, claim_count) for one part."""
    sub = sheet[sheet["Part Name"].astype(str) == str(part)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["period", "claim_count"])
    parsed = []
    for m in sub["Month"]:
        try:
            parsed.append(pd.Period(pd.to_datetime(str(m)), freq="M"))
        except Exception:
            try:
                parsed.append(pd.Period(str(m), freq="M"))
            except Exception:
                parsed.append(None)
    sub["period"] = parsed
    sub = sub[sub["period"].notna()]
    if sub.empty:
        return pd.DataFrame(columns=["period", "claim_count"])
    agg = sub.groupby("period", as_index=False)["Claims"].sum()
    agg = agg.rename(columns={"Claims": "claim_count"}).sort_values("period")
    full = pd.period_range(agg["period"].min(), agg["period"].max(), freq="M")
    monthly = agg.set_index("period").reindex(full)
    monthly.index.name = "period"
    monthly = monthly.reset_index()
    monthly["claim_count"] = monthly["claim_count"].fillna(0.0)
    return monthly


def run_univariate_from_monthly(monthly: pd.DataFrame, part: str,
                                horizon: int | None = None,
                                selected_models: list[str] | None = None) -> dict | None:
    """Run the univariate pipeline from an already-aggregated monthly series."""
    horizon = int(horizon or FORECAST_HORIZON)
    names = _resolve_uni_models(selected_models)
    if monthly is None or monthly.empty:
        return None
    claims = monthly["claim_count"].to_numpy(dtype=float)
    if claims.sum() <= 0 or len(claims) < 10:
        return None

    periods = _period_list(monthly)
    feat_df = build_univariate_features(claims, periods)
    feats = feat_df[UNI_FEATURE_COLS].to_numpy(dtype=float)

    overview = _data_overview(monthly, part)
    stats = _stats_summary(claims)
    trend = _trend_stats(claims)

    logger.info("Univariate analysis for %s (%d months) models=%s", part, len(claims), names)
    cv = _walk_forward_1step(claims, feats, N_CV_FOLDS, model_names=names)
    metrics = cv["metrics"]

    rows = []
    for m in names:
        met = metrics.get(m, {})
        rows.append({
            "Model": m,
            "RMSE": met.get("RMSE", float("nan")),
            "MAE": met.get("MAE", float("nan")),
            "MAPE": met.get("MAPE", float("nan")),
        })
    ranking = pd.DataFrame(rows)
    ranking = ranking.sort_values("RMSE", ascending=True, na_position="last").reset_index(drop=True)
    ranking.insert(0, "Rank", np.arange(1, len(ranking) + 1))
    ranking["Best"] = ["★" if i == 0 else "" for i in range(len(ranking))]
    for col in ("RMSE", "MAE", "MAPE"):
        ranking[col] = ranking[col].round(4)
    if len(names) == 1:
        best_model = names[0]
    else:
        best_model = str(ranking.iloc[0]["Model"])

    forecasts: dict[str, np.ndarray] = {}
    hw_best, hw_sigma, sar_p = {}, 1.0, {}
    if "Holt-Winters" in names:
        hw_fc, hw_best, hw_sigma = _holt_winters_forecast(claims, horizon)
        forecasts["Holt-Winters"] = hw_fc
    if "SARIMA" in names:
        sar_fc, sar_p = _sarima_univariate(claims, horizon)
        forecasts["SARIMA"] = sar_fc
    dl_fc = _refit_dl_full(claims, feats, horizon, model_names=names)
    forecasts.update(dl_fc)

    fallback = next(iter(forecasts.values())) if forecasts else np.full(horizon, float(claims[-1]))
    primary = np.asarray(forecasts.get(best_model, fallback), dtype=float).ravel()[:horizon]
    sigma = max(float(np.std(claims[-12:])) * 0.35, float(hw_sigma) * 0.5, 1.0)
    ci_low = np.clip(primary - 1.96 * sigma, 0, None)
    ci_high = primary + 1.96 * sigma

    last_p = pd.Period(periods[-1], freq="M")
    future = [last_p + i for i in range(1, horizon + 1)]

    insights = generate_insights(overview, stats, trend, best_model, horizon, ranking)

    month_avg = (
        pd.Series(claims, index=[int(pd.Period(p, freq="M").month) for p in periods])
        .groupby(level=0).mean()
    )
    seasonality = pd.DataFrame({
        "Month": list(range(1, 13)),
        "Avg Claims": [float(month_avg.get(m, 0.0)) for m in range(1, 13)],
    })

    return {
        "part": part,
        "claims": claims,
        "periods": [str(p) for p in periods],
        "features": feat_df,
        "overview": overview,
        "stats": stats,
        "trend": trend,
        "ranking": ranking,
        "best_model": best_model,
        "forecasts": {k: np.asarray(v, dtype=float).ravel()[:horizon] for k, v in forecasts.items()},
        "best_forecast": primary,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "future_periods": [str(p) for p in future],
        "holt_params": {
            "alpha": hw_best.get("alpha"),
            "beta": hw_best.get("beta"),
            "gamma": hw_best.get("gamma"),
            "holdout_rmse": hw_best.get("rmse"),
        },
        "sarima_params": sar_p,
        "insights": insights,
        "seasonality": seasonality,
        "rolling_mean_3": feat_df["Rolling_Mean_3"].to_numpy(dtype=float),
        "rolling_mean_12": feat_df["Rolling_Mean_12"].to_numpy(dtype=float),
        "oos_actual": cv["actual"],
        "oos_preds": {k: np.asarray(v[:len(cv["actual"])], dtype=float) for k, v in cv["preds"].items()},
        "oos_periods": [str(p) for p in periods[-len(cv["actual"]):]] if len(cv["actual"]) else [],
        "horizon": horizon,
        "selected_models": names,
    }


def run_univariate_analysis(raw: pd.DataFrame, part: str,
                            horizon: int | None = None,
                            selected_models: list[str] | None = None) -> dict | None:
    """
    Claims-only univariate pipeline for one part.

    Returns a dict consumed by the Univariate Analysis tab.
    """
    monthly = build_monthly_series(raw, part, production=None, require_production=False)
    return run_univariate_from_monthly(
        monthly, part, horizon=horizon, selected_models=selected_models,
    )
