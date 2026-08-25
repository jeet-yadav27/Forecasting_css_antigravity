"""
forecasting/pipeline/runner.py
------------------------------
Per-part training pipeline with walk-forward cross-validation,
hyperparameter tuning, automatic best-model selection, and main().

Walk-Forward CV Scheme (zero leakage)
--------------------------------------
Reserve the last N_CV_FOLDS months as the rolling test window:

  Fold 1: train=[0 .. T-N_FOLDS],    forecast month T-N_FOLDS, add actual
  Fold 2: train=[0 .. T-N_FOLDS+1],  forecast next month, …
  Fold N: train=[0 .. T-1],          forecast final test month

Each fold's MinMaxScaler is fit ONLY on the training slice.
After CV, metrics (RMSE/MAE/MAPE/R²) are computed in original claim units,
models are ranked, the best model is selected, then refit on FULL history
for the final 12-month forecast.  The inverse-MAE ensemble is retained.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import forecasting.config as cfg
from forecasting.config import (
    PRODUCTION_PER_MONTH,
    FORECAST_HORIZON,
    LOOKBACK_WINDOW,
    DATA_FILES,
    COUNTERMEASURES,
    N_CV_FOLDS,
    RANDOM_SEED,
    MODELS_DIR,
    FORECASTS_DIR,
    OUTPUT_DIR,
    MODEL_NAMES,
)
from forecasting.data.loader import (
    load_and_prepare,
    build_monthly_series,
    apply_countermeasure,
    EXOG_COLS,
)
from forecasting.metrics import compute_metrics, build_ranking_table, select_best_model
from forecasting.models.cnn_lstm import CnnLstmForecaster
from forecasting.models.nbeats import NBeatsForecaster
from forecasting.models.transformer import TransformerForecaster
from forecasting.models.ml_models import (
    HP_GRIDS,
    fit_sarima,
    grid_search_sarima,
)
from forecasting.dashboard.builder import build_gradio_app

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Window dataset builder (3-D, for DL models)
# ---------------------------------------------------------------------------

def build_window_dataset(series_vals: np.ndarray, exog: np.ndarray,
                          lookback: int, horizon: int):
    """Build a 3-D sliding-window dataset for sequence-to-sequence DL models."""
    N = len(series_vals) - lookback - horizon + 1
    if N <= 0:
        return None, None

    X_list, y_list = [], []
    for i in range(N):
        win_y  = series_vals[i:i + lookback, None]
        win_ex = exog[i:i + lookback]
        X_list.append(np.concatenate([win_y, win_ex], axis=1))
        y_list.append(series_vals[i + lookback:i + lookback + horizon])

    return np.array(X_list), np.array(y_list)


# ---------------------------------------------------------------------------
# Leak-free scaler helpers
# ---------------------------------------------------------------------------

def _scale_fold(claim_vals: np.ndarray, exog_raw: np.ndarray, train_end: int):
    """Fit scalers on [:train_end] only; transform the full arrays."""
    c_scaler = MinMaxScaler()
    e_scaler = MinMaxScaler()
    c_scaler.fit(claim_vals[:train_end, None])
    e_scaler.fit(exog_raw[:train_end])
    claim_sc_full = c_scaler.transform(claim_vals[:, None]).ravel()
    exog_sc_full  = e_scaler.transform(exog_raw)
    return claim_sc_full, exog_sc_full, c_scaler


def _inv_scale(scaler: MinMaxScaler, arr: np.ndarray) -> np.ndarray:
    return scaler.inverse_transform(
        np.clip(np.asarray(arr, dtype=float).ravel(), 0, 1)[:, None]
    ).ravel().clip(0)


# ---------------------------------------------------------------------------
# DL model factory
# ---------------------------------------------------------------------------

def _make_dl_model(name: str, lookback: int, n_features: int,
                   horizon: int, hp: dict):
    """Return an instantiated (untrained) DL forecaster with given *hp*."""
    if name == "CNN-LSTM":
        # Fixed defaults only — no hyperparameter grid search for CNN-LSTM
        return CnnLstmForecaster(
            lookback=lookback, n_features=n_features, horizon=horizon,
            lr=hp.get("lr", 1e-2), epochs=hp.get("epochs", 40),
        )
    if name == "N-BEATS":
        return NBeatsForecaster(
            lookback=lookback, n_features=n_features, horizon=horizon,
            lr=hp.get("lr", 5e-3), epochs=hp.get("epochs", 100),
        )
    if name == "Transformer":
        return TransformerForecaster(
            lookback=lookback, n_features=n_features, horizon=horizon,
            d_model=hp.get("d_model", 16), n_heads=2,
            lr=hp.get("lr", 3e-3), epochs=hp.get("epochs", 80),
        )
    raise ValueError(f"Unknown DL model: {name}")


def _resolve_selected_models(selected: list[str] | None) -> list[str]:
    """Validate user model choices; default to all active models."""
    allowed = list(MODEL_NAMES)
    if not selected:
        return allowed
    picked = [m for m in selected if m in allowed]
    if not picked:
        logger.warning("No valid models in %s — using all %s", selected, allowed)
        return allowed
    # Preserve canonical order
    return [m for m in allowed if m in picked]


# ---------------------------------------------------------------------------
# Walk-forward CV + hyperparameter tuning
# ---------------------------------------------------------------------------

def _walk_forward_cv(claim_vals: np.ndarray, exog_raw: np.ndarray,
                     W: int, n_folds: int, hp_tune: bool,
                     preset_params: dict | None = None,
                     model_names: list[str] | None = None):
    """
    Expanding-window walk-forward CV over the last *n_folds* months.

    Only *model_names* are trained / scored. If *preset_params* is provided
    and *hp_tune* is False, locked hyperparameters are used.
    """
    T = len(claim_vals)
    model_names = _resolve_selected_models(model_names)
    dl_names = [m for m in model_names if m in ("CNN-LSTM", "N-BEATS", "Transformer")]
    use_sarima = "SARIMA" in model_names

    claim_vals = np.asarray(claim_vals, dtype=float).ravel()
    exog_raw = np.asarray(exog_raw, dtype=float)
    if exog_raw.ndim == 1:
        exog_raw = exog_raw.reshape(-1, 1)
    T = int(len(claim_vals))
    W = int(W)
    n_folds = int(min(max(1, int(n_folds)), max(1, T - W - 2)))

    cv_mae: dict = {k: [] for k in model_names}
    best_params: dict = {k: {} for k in model_names}
    if preset_params:
        for k, v in _restore_param_types(preset_params).items():
            if k in best_params:
                best_params[k] = v
    if "CNN-LSTM" in model_names:
        best_params["CNN-LSTM"] = {"lr": 1e-2, "epochs": 40}

    oos_actual: list[float] = []
    oos_preds: dict[str, list[float]] = {k: [] for k in model_names}

    fold1_train_end = T - n_folds
    cs1, es1, _ = _scale_fold(claim_vals, exog_raw, fold1_train_end)
    val_idx1 = fold1_train_end

    if hp_tune:
        logger.info(
            "Hyperparameter tuning on fold-1 (models=%s) …", ", ".join(model_names)
        )
        if "CNN-LSTM" in model_names:
            best_params["CNN-LSTM"] = {"lr": 1e-2, "epochs": 40}
            logger.info("CNN-LSTM: skipping HP tune — fixed defaults")

        if use_sarima:
            best_params["SARIMA"] = grid_search_sarima(
                cs1[:fold1_train_end], cs1[val_idx1:val_idx1 + 1]
            )

        tune_dl = [m for m in dl_names if m != "CNN-LSTM"]
        if tune_dl:
            X_dl_tr, y_dl_tr = build_window_dataset(
                cs1[:fold1_train_end], es1[:fold1_train_end], W, 1
            )
            if X_dl_tr is not None and len(X_dl_tr) >= 4:
                F = X_dl_tr.shape[2]

                # ── Parallel HP grid search across DL models ─────────────
                def _hp_search_one(dl_name: str) -> tuple[str, dict]:
                    best_mae = float("inf")
                    best_hp = HP_GRIDS[dl_name][0]
                    for hp in HP_GRIDS[dl_name]:
                        try:
                            mdl = _make_dl_model(dl_name, W, F, 1, hp)
                            mdl.fit(X_dl_tr, y_dl_tr)
                            pred = mdl.predict(X_dl_tr[-1:])[0]
                            mae = float(np.abs(pred[0] - cs1[val_idx1]))
                            if mae < best_mae:
                                best_mae = mae
                                best_hp = hp
                        except Exception:
                            continue
                    return dl_name, best_hp

                n_hp_workers = min(len(tune_dl), os.cpu_count() or 1)
                with ThreadPoolExecutor(max_workers=n_hp_workers) as pool:
                    hp_futures = {pool.submit(_hp_search_one, n): n for n in tune_dl}
                    for fut in as_completed(hp_futures):
                        try:
                            dl_name, hp_result = fut.result()
                            best_params[dl_name] = hp_result
                        except Exception as exc:
                            dl_name = hp_futures[fut]
                            logger.warning("HP tuning failed for %s: %s", dl_name, exc)
                            best_params[dl_name] = HP_GRIDS[dl_name][0]
    else:
        if "CNN-LSTM" in model_names and not best_params.get("CNN-LSTM"):
            best_params["CNN-LSTM"] = {"lr": 1e-2, "epochs": 40}

    # ── Walk-forward folds ───────────────────────────────────────────────
    # Iterate a plain range so nested epoch tqdm bars (Gradio track_tqdm)
    # cannot inflate the fold index past T-1.
    from forecasting.models.base import tqdm as _tqdm
    fold_bar = _tqdm(
        total=n_folds,
        desc=f"Walk-forward ({len(model_names)} models)",
        unit="fold",
        leave=True,
        dynamic_ncols=True,
    )
    for k in range(n_folds):
        train_end = T - n_folds + k
        val_idx = train_end
        if train_end < W + 2 or val_idx >= T:
            fold_bar.update(1)
            continue

        cs, es, c_scaler = _scale_fold(claim_vals, exog_raw, train_end)
        true_orig = float(claim_vals[val_idx])
        fold_preds_orig: dict[str, float] = {}

        if use_sarima:
            if hasattr(fold_bar, "set_postfix_str"):
                fold_bar.set_postfix_str("SARIMA")
            sarima_p = best_params.get("SARIMA", {})
            sarima_fc = fit_sarima(
                cs[:train_end], 1,
                sarima_p.get("order", (1, 1, 1)),
                sarima_p.get("seasonal_order", (1, 1, 0, 12)),
            )
            pred_sc = float(sarima_fc[0])
            fold_preds_orig["SARIMA"] = float(_inv_scale(c_scaler, [pred_sc])[0])
            cv_mae["SARIMA"].append(float(np.abs(pred_sc - cs[val_idx])))

        if dl_names:
            X_dl_tr, y_dl_tr = build_window_dataset(
                cs[:train_end], es[:train_end], W, 1
            )
            if X_dl_tr is not None and len(X_dl_tr) >= 4:
                F = X_dl_tr.shape[2]
                lx = np.concatenate([
                    cs[val_idx - W:val_idx, None],
                    es[val_idx - W:val_idx],
                ], axis=1)[None]

                # ── Parallel DL training within fold ─────────────────────
                def _fold_dl_one(dl_name: str) -> tuple[str, float, float]:
                    hp = best_params.get(dl_name, {})
                    mdl = _make_dl_model(dl_name, W, F, 1, hp)
                    mdl.fit(X_dl_tr, y_dl_tr)
                    pred_sc = float(mdl.predict(lx)[0, 0])
                    pred_orig = float(_inv_scale(c_scaler, [pred_sc])[0])
                    mae_sc = float(np.abs(pred_sc - cs[val_idx]))
                    return dl_name, pred_orig, mae_sc

                n_dl_workers = min(len(dl_names), os.cpu_count() or 1)
                with ThreadPoolExecutor(max_workers=n_dl_workers) as pool:
                    dl_futures = {pool.submit(_fold_dl_one, n): n for n in dl_names}
                    for fut in as_completed(dl_futures):
                        try:
                            dl_name, pred_orig, mae_sc = fut.result()
                            fold_preds_orig[dl_name] = pred_orig
                            cv_mae[dl_name].append(mae_sc)
                        except Exception:
                            dl_name = dl_futures[fut]
                            fold_preds_orig[dl_name] = true_orig
                            cv_mae[dl_name].append(1.0)

        if len(fold_preds_orig) == len(model_names):
            oos_actual.append(true_orig)
            for mname in model_names:
                oos_preds[mname].append(fold_preds_orig[mname])
        fold_bar.update(1)
    fold_bar.close()

    oos_actual_arr = np.asarray(oos_actual, dtype=float)
    oos_preds_arr = {k: np.asarray(v, dtype=float) for k, v in oos_preds.items()}
    return cv_mae, best_params, oos_actual_arr, oos_preds_arr


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _ensure_output_dirs() -> None:
    for d in (OUTPUT_DIR, MODELS_DIR, FORECASTS_DIR):
        os.makedirs(d, exist_ok=True)


def _safe_part_name(part: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in part)


def save_part_artifacts(result: dict, trained_bundle: dict) -> None:
    """Persist trained models (pickle) and forecast/metrics CSVs for one part."""
    _ensure_output_dirs()
    safe = _safe_part_name(result["part"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_path = os.path.join(MODELS_DIR, f"{safe}_{stamp}.pkl")
    try:
        with open(model_path, "wb") as f:
            pickle.dump(trained_bundle, f)
        result["model_path"] = model_path
        logger.info("Saved models → %s", model_path)
    except Exception as exc:
        logger.warning("Could not pickle models for %s: %s", result["part"], exc)

    # Forecast CSV
    rows = []
    for i, fp in enumerate(result["future_periods"]):
        row = {
            "Part": result["part"],
            "Month": str(fp),
            "Best_Model": result["best_model"],
            "Best_Forecast": float(result["best_forecast"][i]),
            "Ensemble_Forecast": float(result["ensemble_raw"][i]),
            "CI_Low": float(result["ci_low"][i]),
            "CI_High": float(result["ci_high"][i]),
        }
        for mn, fc in result["forecasts_raw"].items():
            row[mn] = float(fc[i])
        rows.append(row)
    fc_path = os.path.join(FORECASTS_DIR, f"{safe}_forecast_{stamp}.csv")
    pd.DataFrame(rows).to_csv(fc_path, index=False)
    result["forecast_path"] = fc_path

    # Metrics / ranking CSV
    if result.get("ranking_df") is not None:
        met_path = os.path.join(FORECASTS_DIR, f"{safe}_metrics_{stamp}.csv")
        result["ranking_df"].to_csv(met_path, index=False)
        result["metrics_path"] = met_path

    # JSON summary
    summary = {
        "part": result["part"],
        "best_model": result["best_model"],
        "best_params": {k: {kk: str(vv) for kk, vv in v.items()}
                        for k, v in result.get("best_params", {}).items()},
        "metrics": result.get("metrics_by_model", {}),
        "generated_at": stamp,
    }
    js_path = os.path.join(FORECASTS_DIR, f"{safe}_summary_{stamp}.json")
    with open(js_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Per-part pipeline
# ---------------------------------------------------------------------------

def load_fixed_best_params(part: str | None = None) -> dict:
    """Load locked hyperparameters from outputs/best_params.json."""
    path = cfg.BEST_PARAMS_PATH if hasattr(cfg, "BEST_PARAMS_PATH") else os.path.join(
        OUTPUT_DIR, "best_params.json"
    )
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            store = json.load(f)
        if part is None:
            return store
        return store.get(part, {})
    except Exception as exc:
        logger.warning("Could not load fixed params: %s", exc)
        return {}


def save_fixed_best_params(part: str, best_params: dict) -> str:
    """Persist best hyperparameters for *part* so later runs can reuse them."""
    _ensure_output_dirs()
    path = getattr(cfg, "BEST_PARAMS_PATH", os.path.join(OUTPUT_DIR, "best_params.json"))
    store: dict = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                store = json.load(f)
        except Exception:
            store = {}
    # JSON-serialise nested tuples (SARIMA orders)
    serialisable = {}
    for model, params in (best_params or {}).items():
        serialisable[model] = {
            k: list(v) if isinstance(v, tuple) else v
            for k, v in (params or {}).items()
        }
    store[part] = serialisable
    with open(path, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2)
    logger.info("Locked best params for %s → %s", part, path)
    return path


def _restore_param_types(best_params: dict) -> dict:
    """Convert JSON lists back to SARIMA tuples where needed."""
    out = {}
    for model, params in (best_params or {}).items():
        p = dict(params or {})
        if "order" in p and isinstance(p["order"], list):
            p["order"] = tuple(p["order"])
        if "seasonal_order" in p and isinstance(p["seasonal_order"], list):
            p["seasonal_order"] = tuple(p["seasonal_order"])
        out[model] = p
    return out


def run_pipeline_for_part(
    monthly,
    part: str,
    raw: pd.DataFrame | None = None,
    *,
    hp_tune: bool | None = None,
    fixed_params: dict | None = None,
    lock_params: bool = True,
    selected_models: list[str] | None = None,
) -> dict | None:
    """
    Full end-to-end pipeline for a single part.

    *selected_models*: if one model → forecast with that model only;
    if several → train/compare those and build an ensemble of the selection.
    """
    model_names = _resolve_selected_models(selected_models)
    dl_names = [m for m in model_names if m in ("CNN-LSTM", "N-BEATS", "Transformer")]
    use_sarima = "SARIMA" in model_names

    print(f"  [{part}] {len(monthly)} months | models: {', '.join(model_names)}")
    logger.info("=== Part: %s (%d months) models=%s ===", part, len(monthly), model_names)

    claim_vals = monthly["claim_count"].values.astype(float)
    if claim_vals.sum() == 0:
        print("    -> SKIP (no claims)")
        return None

    use_cols = [c for c in EXOG_COLS if c in monthly.columns]
    exog_raw = np.nan_to_num(monthly[use_cols].values.astype(float), nan=0.0)

    T = len(claim_vals)
    W = min(LOOKBACK_WINDOW, T - FORECAST_HORIZON - N_CV_FOLDS - 2)
    if W < 4:
        print(f"    -> SKIP (too short for {N_CV_FOLDS}-fold CV with lookback)")
        return None

    do_tune = cfg.HP_TUNE if hp_tune is None else bool(hp_tune)
    preset = fixed_params
    if preset is None and not do_tune:
        preset = load_fixed_best_params(part)
    if preset:
        preset = _restore_param_types(preset)
        print(f"    Using locked hyperparameters (tune={do_tune})")
    else:
        print(f"    Walk-forward CV ({N_CV_FOLDS} folds) + HP tuning={do_tune} ...")

    cv_mae, best_params, oos_actual, oos_preds = _walk_forward_cv(
        claim_vals, exog_raw, W, N_CV_FOLDS, do_tune,
        preset_params=preset, model_names=model_names,
    )

    mean_cv_mae = {
        k: float(np.mean(v)) if v else 1.0
        for k, v in cv_mae.items()
    }

    metrics_by_model = {}
    for mname in model_names:
        if mname in oos_preds and len(oos_preds[mname]) == len(oos_actual) and len(oos_actual):
            metrics_by_model[mname] = compute_metrics(oos_actual, oos_preds[mname])
        else:
            metrics_by_model[mname] = {
                "RMSE": float("nan"), "MAE": mean_cv_mae.get(mname, float("nan")),
                "MAPE": float("nan"), "R2": float("nan"),
            }

    ranking_df = build_ranking_table(metrics_by_model)
    # Single selection → that model is the forecast; multi → auto-pick best
    if len(model_names) == 1:
        best_model_name = model_names[0]
    else:
        best_model_name = select_best_model(ranking_df)
    print(f"    Forecast model: {best_model_name} "
          f"({'single' if len(model_names) == 1 else 'best of ' + str(len(model_names))})")
    logger.info("Ranking:\n%s", ranking_df.to_string(index=False))

    for mname, mae in mean_cv_mae.items():
        m = metrics_by_model.get(mname, {})
        print(
            f"      {mname:<12} CV MAE={mae:.5f}  "
            f"RMSE={m.get('RMSE', float('nan')):.2f}  "
            f"MAPE={m.get('MAPE', float('nan')):.1f}%  "
            f"R²={m.get('R2', float('nan')):.3f}"
        )

    # ── Refit all models on FULL dataset in parallel ─────────────────────
    print("    Refitting selected models on full dataset (parallel) ...")
    full_c_scaler = MinMaxScaler()
    full_e_scaler = MinMaxScaler()
    claim_sc = full_c_scaler.fit_transform(claim_vals[:, None]).ravel()
    exog_sc = full_e_scaler.fit_transform(exog_raw)

    X_dl_full, y_dl_full = build_window_dataset(claim_sc, exog_sc, W, FORECAST_HORIZON)
    if dl_names and (X_dl_full is None or len(X_dl_full) < 2):
        print("    -> SKIP (insufficient samples on full dataset)")
        return None

    fc_sc: dict = {}
    dl_bundle: dict = {}

    def _refit_dl(dl_name: str) -> tuple[str, np.ndarray, object]:
        """Refit one DL model on the full dataset and return its forecast."""
        F = X_dl_full.shape[2]
        lx_dl = X_dl_full[-1:]
        print(f"    Fitting {dl_name} (parallel) ...")
        mdl = _make_dl_model(
            dl_name, W, F, FORECAST_HORIZON, best_params.get(dl_name, {})
        )
        mdl.fit(X_dl_full, y_dl_full)
        fc = mdl.predict(lx_dl)[0]
        return dl_name, fc, mdl

    def _refit_sarima() -> tuple[str, np.ndarray]:
        """Refit SARIMA on the full scaled dataset and return its forecast."""
        print("    Fitting SARIMA (parallel) ...")
        sarima_p = best_params.get("SARIMA", {})
        fc = fit_sarima(
            claim_sc, FORECAST_HORIZON,
            sarima_p.get("order", (1, 1, 1)),
            sarima_p.get("seasonal_order", (1, 1, 0, 12)),
        )
        return "SARIMA", fc

    # Build task list — all DL models + SARIMA run concurrently
    refit_tasks: list = []
    if dl_names and X_dl_full is not None:
        refit_tasks.extend([("dl", n) for n in dl_names])
    if use_sarima:
        refit_tasks.append(("sarima", None))

    n_refit_workers = min(len(refit_tasks), os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=n_refit_workers) as pool:
        refit_futures = []
        for kind, name in refit_tasks:
            if kind == "dl":
                refit_futures.append(pool.submit(_refit_dl, name))
            else:
                refit_futures.append(pool.submit(_refit_sarima))

        for fut in as_completed(refit_futures):
            try:
                result = fut.result()
                model_id = result[0]
                if len(result) == 3:          # DL: (name, fc, model_obj)
                    fc_sc[model_id] = result[1]
                    dl_bundle[model_id] = result[2]
                else:                          # SARIMA: (name, fc)
                    fc_sc[model_id] = result[1]
            except Exception as exc:
                logger.warning("Refit task failed: %s", exc)

    inv_w = {k: 1.0 / (v + 1e-9) for k, v in mean_cv_mae.items()}
    total_w = sum(inv_w.values()) or 1.0
    weights = {k: v / total_w for k, v in inv_w.items()}

    ens_sc = sum(weights[k] * fc_sc[k] for k in fc_sc)

    def inv_sc(arr: np.ndarray) -> np.ndarray:
        return _inv_scale(full_c_scaler, arr)

    fc_raw = {k: inv_sc(v) for k, v in fc_sc.items()}
    ens_raw = inv_sc(ens_sc)
    # One model → primary forecast is that model; multi → best + ensemble
    best_forecast = fc_raw[best_model_name].copy()
    if len(model_names) == 1:
        ens_raw = best_forecast.copy()
        weights = {best_model_name: 1.0}

    last_period = monthly["period"].iloc[-1]
    future_pds = [last_period + i + 1 for i in range(FORECAST_HORIZON)]
    cm_mults = apply_countermeasure(part, future_pds)
    ens_cm = ens_raw * cm_mults
    best_cm = best_forecast * cm_mults
    fc_raw = {k: v * cm_mults for k, v in fc_raw.items()}

    all_fc_arr = np.array(list(fc_raw.values()))
    if len(fc_raw) == 1:
        # Single model: CI from mild historical residual scale
        spread = max(float(np.std(claim_vals[-12:])) * 0.25, 1.0)
        ci_low = (best_cm - 1.5 * spread).clip(0)
        ci_high = best_cm + 1.5 * spread
    else:
        ci_low = (best_cm - 1.5 * all_fc_arr.std(axis=0)).clip(0)
        ci_high = best_cm + 1.5 * all_fc_arr.std(axis=0)

    prod_vals = monthly["production"].values.astype(float) if "production" in monthly else (
        np.full(T, np.nan, dtype=float)
    )
    # Prefer last known production for rate (never invent synthetic volumes)
    last_prod = float(np.nanmean(prod_vals[-3:])) if len(prod_vals) else np.nan
    if not np.isfinite(last_prod) or last_prod <= 0:
        last_prod = float(np.nanmean(prod_vals)) if np.any(np.isfinite(prod_vals)) else 1.0

    result = {
        "part": part,
        "monthly": monthly,
        "claim_vals": claim_vals,
        "production": prod_vals,
        "hist_rate": claim_vals / (prod_vals / 1_000 + 1e-9),
        "future_periods": future_pds,
        "ensemble_raw": ens_cm,
        "best_forecast": best_cm,
        "best_model": best_model_name,
        "selected_models": model_names,
        "forecasts_raw": fc_raw,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "forecast_rate": best_cm / (last_prod / 1_000),
        "weights": weights,
        "val_mae": mean_cv_mae,
        "cv_mae_folds": cv_mae,
        "best_params": best_params,
        "cm_mults": cm_mults,
        "has_cm": part in COUNTERMEASURES,
        "metrics_by_model": metrics_by_model,
        "ranking_df": ranking_df,
        "oos_actual": oos_actual,
        "oos_preds": oos_preds,
        "oos_periods": [str(p) for p in monthly["period"].tolist()[-len(oos_actual):]]
        if len(oos_actual) else [],
        "exog_cols": use_cols,
        "raw_ref": raw,
    }

    trained_bundle = {
        "part": part,
        "best_model": best_model_name,
        "selected_models": model_names,
        "best_params": best_params,
        "lookback": W,
        "horizon": FORECAST_HORIZON,
        "exog_cols": use_cols,
        "claim_scaler": full_c_scaler,
        "exog_scaler": full_e_scaler,
        "dl": dl_bundle,
        "weights": weights,
        "metrics": metrics_by_model,
    }
    try:
        save_part_artifacts(result, trained_bundle)
    except Exception as exc:
        logger.warning("Artifact save failed for %s: %s", part, exc)

    if lock_params and best_params:
        try:
            save_fixed_best_params(part, best_params)
            result["params_locked_path"] = cfg.BEST_PARAMS_PATH
        except Exception as exc:
            logger.warning("Could not lock params for %s: %s", part, exc)

    return result


def train_uploaded_part(
    raw: pd.DataFrame,
    part: str,
    *,
    retune: bool = True,
    use_locked_params: bool = True,
    selected_models: list[str] | None = None,
    production: np.ndarray | None = None,
    production_cost: np.ndarray | None = None,
    cm_enabled: bool = False,
    cm_month: str | None = None,
    cm_reduction_pct: float | None = 20.0,
    cm_reduction_known: bool = True,
) -> dict | None:
    """
    Train + forecast a single part from an already-loaded upload dataframe.

    *production* is optional (claims-only runs use unit production for rates).
    Optional *production_cost* enables CPV metrics.
    Optional countermeasure reduces forecast after the CM date.
    When *cm_reduction_known* is False, reduction % is estimated from data.
    """
    from forecasting.economics import (
        apply_cm_to_forecast,
        compute_cpv_claim_ratio,
        estimate_cm_reduction_pct,
        forecast_economics,
    )

    user_prod = production is not None and len(np.asarray(production).ravel()) > 0
    monthly = build_monthly_series(
        raw, part,
        production=production if user_prod else None,
        require_production=False,
    )
    if monthly.empty:
        logger.warning("No monthly series for part=%s", part)
        return None

    # Claims-only: unit production so exog / rates stay finite (not synthetic growth)
    if not user_prod or np.any(~np.isfinite(monthly["production"].to_numpy(dtype=float))):
        monthly["production"] = 1.0
        monthly["claim_rate"] = monthly["claim_count"] / (monthly["production"] / 1_000)
        user_prod = False

    costs = None
    if production_cost is not None and user_prod:
        costs = np.asarray(production_cost, dtype=float).ravel()
        if len(costs) < len(monthly):
            pad = np.full(len(monthly) - len(costs), np.nan)
            costs = np.concatenate([costs, pad])
        costs = costs[:len(monthly)]
        monthly["production_cost"] = costs
        monthly["cpv"] = monthly["production_cost"] / monthly["production"]
        monthly["claim_ratio"] = monthly["claim_count"] / monthly["production"]

    fixed = None
    if not retune and use_locked_params:
        fixed = load_fixed_best_params(part) or None

    result = run_pipeline_for_part(
        monthly, part, raw=raw,
        hp_tune=retune,
        fixed_params=fixed,
        lock_params=True,
        selected_models=selected_models,
    )
    if result is None:
        return None

    claims = result["claim_vals"]
    prod = result["production"]
    econ = compute_cpv_claim_ratio(claims, prod, costs if user_prod else None)
    result["economics"] = econ
    result["production_cost"] = econ["production_cost"]
    result["cpv"] = econ["cpv"]
    result["claim_ratio"] = econ["claim_ratio"]
    result["claim_ratio_per_1k"] = econ["claim_ratio_per_1k"]
    result["production_user_provided"] = bool(user_prod)

    last_prod = float(np.nanmean(prod[-3:])) if len(prod) else np.nan
    if not np.isfinite(last_prod) or last_prod <= 0:
        last_prod = float(np.nanmean(prod)) if len(prod) else 1.0
    fut_prod = np.full(len(result["future_periods"]), last_prod)
    result["forecast_rate"] = result["best_forecast"] / (last_prod / 1_000.0)

    # Resolve reduction % (optional — estimate when unknown)
    red_meta = {
        "reduction_pct": float(cm_reduction_pct) if cm_reduction_pct is not None else 0.0,
        "source": "user",
        "note": "",
    }
    if cm_enabled:
        if not cm_reduction_known or cm_reduction_pct is None:
            red_meta = estimate_cm_reduction_pct(
                raw, part, cm_month,
                hist_claims=claims,
                hist_periods=result["monthly"]["period"],
                hist_production=prod if user_prod else None,
            )
        else:
            red_meta = {
                "reduction_pct": float(cm_reduction_pct),
                "source": "user",
                "note": f"User-specified reduction {float(cm_reduction_pct):.0f}%.",
            }

    baseline = np.asarray(result.get("best_forecast", result["ensemble_raw"]), dtype=float).copy()
    cm_sim = apply_cm_to_forecast(
        baseline,
        raw,
        part,
        cm_enabled=cm_enabled,
        cm_month=cm_month,
        reduction_pct=float(red_meta["reduction_pct"]),
        future_periods=result["future_periods"],
        future_production=fut_prod if user_prod else None,
        hist_production=prod if user_prod else None,
        use_peak_fcok=True,
        sensitivity=1.25,
    )
    cm_sim["reduction_source"] = red_meta.get("source")
    cm_sim["reduction_note"] = red_meta.get("note", "")
    result["cm_sim"] = cm_sim
    result["has_cm"] = bool(cm_enabled and cm_sim.get("cm_enabled"))
    result["best_forecast_raw"] = baseline.copy()
    if result["has_cm"]:
        result["best_forecast"] = cm_sim["adjusted"]
        result["ensemble_raw"] = cm_sim["adjusted"]
        result["cm_mults"] = np.where(
            baseline > 0, cm_sim["adjusted"] / baseline, 1.0
        )
        result["forecast_rate"] = result["best_forecast"] / (last_prod / 1_000.0)
    else:
        result["cm_mults"] = np.ones(len(baseline))

    result["forecast_economics"] = forecast_economics(
        result["best_forecast"], econ,
        future_production=fut_prod,
    )
    return result


# ---------------------------------------------------------------------------
# Main entry point — launch UI only (no auto-training on bundled data)
# ---------------------------------------------------------------------------

def main(
    parts_filter: list[str] | None = None,
    port: int = 7860,
    share: bool = False,
    inbrowser: bool = True,
    data_files: list[str] | None = None,
) -> None:
    """
    Launch the Gradio app immediately.

    Training starts only after the user uploads a dataset and selects a part
    in the UI. Bundled ``data/Test *.csv`` files are NOT auto-trained on startup.
    """
    _ensure_output_dirs()

    print("=" * 68)
    print("  AUTOMOTIVE WARRANTY CLAIMS FORECASTING SYSTEM")
    print("  Mode: Upload → Select Part → Train → Forecast")
    print(f"  Seed: {RANDOM_SEED}  |  Horizon: {FORECAST_HORIZON} mo")
    print(f"  Locked params file: {cfg.BEST_PARAMS_PATH}")
    print("=" * 68)
    print("\n[OK] Opening Gradio — no training until you upload data.")
    print(f"     http://localhost:{port}\n")

    # Optional: preload paths passed via CLI (still no training until user clicks)
    preload_files = data_files
    app = build_gradio_app(
        results=None,
        raw=None,
        preload_files=preload_files,
        parts_filter=parts_filter,
    )
    # Long DL training needs a queued, durable request so plots return to the UI
    try:
        app.queue(default_concurrency_limit=1)
    except Exception:
        pass
    app.launch(
        server_port=port,
        share=share,
        inbrowser=inbrowser,
        show_error=True,
    )
