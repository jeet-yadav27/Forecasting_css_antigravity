"""
forecasting/data/loader.py
--------------------------
Data loading, validation, feature engineering, synthetic production,
and manufacturing-month countermeasure utilities.
"""

from __future__ import annotations

import logging
import os
from typing import Iterable

import numpy as np
import pandas as pd

from forecasting.config import (
    PRODUCTION_PER_MONTH,
    PRODUCTION_GROWTH_RATE,
    FORECAST_HORIZON,
    COUNTERMEASURES,
    CM_DECAY_HALF_LIFE,
    WARRANTY_MONTHS,
)

logger = logging.getLogger(__name__)

# Canonical column aliases accepted from uploaded files
_REQUIRED_ALIASES = {
    "Part Name": [
        "Part Name", "PART_NAME", "Part", "part_name",
        "Part_Name+", "Part Name+", "Part_Name",
    ],
    "FCOK_DATE": [
        "FCOK_DATE", "FCO/K Month", "FCOK", "FCO_K_DATE", "Manufacturing Date",
    ],
    "PROCESSING_DATE": [
        "PROCESSING_DATE", "Process Date", "PROCESS_DATE", "Claim Month",
        "Wty_Month", "Wty Month", "Warranty Month",
    ],
    "ODOMETER": ["ODOMETER", "Odometer Reading", "Odometer", "ODOMETER_READING"],
}

_OPTIONAL_DATE_COLS = ["REGD_DATE", "REPAIR_DATE"]

# Common claim-date formats (data uses e.g. 03-May-19)
_DATE_FORMATS = (
    "%d-%b-%y",   # 03-May-19
    "%d-%b-%Y",   # 03-May-2019
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%Y-%m-%d",
    "%m/%d/%Y",
    "%d/%m/%y",
    "%Y/%m/%d",
)


def _parse_date_series(series: pd.Series, col_name: str = "") -> pd.Series:
    """
    Convert a mixed/string date column to pandas datetime.

    Tries explicit formats first (dd-Mon-yy etc.), then a day-first
    fallback. Unparseable values become NaT.
    """
    if series is None or len(series) == 0:
        return series

    if pd.api.types.is_datetime64_any_dtype(series):
        return series

    # Excel may already give Timestamp objects mixed with strings
    s = series.copy()
    result = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    # Already-datetime-like objects
    mask_dt = s.map(lambda x: isinstance(x, (pd.Timestamp, np.datetime64)) or hasattr(x, "year"))
    if mask_dt.any():
        result.loc[mask_dt] = pd.to_datetime(s.loc[mask_dt], errors="coerce")

    remaining = result.isna() & s.notna()
    if not remaining.any():
        n_ok = int(result.notna().sum())
        logger.info("Date convert [%s]: %s already datetime-like", col_name, f"{n_ok:,}")
        return result

    text = s.loc[remaining].astype(str).str.strip()
    parsed = pd.Series(pd.NaT, index=text.index, dtype="datetime64[ns]")
    still = pd.Series(True, index=text.index)

    for fmt in _DATE_FORMATS:
        if not still.any():
            break
        trial = pd.to_datetime(text.loc[still], format=fmt, errors="coerce")
        ok = trial.notna()
        if ok.any():
            parsed.loc[trial.index[ok]] = trial.loc[ok]
            still.loc[trial.index[ok]] = False

    # Final fallback: day-first inference
    if still.any():
        trial = pd.to_datetime(text.loc[still], dayfirst=True, errors="coerce")
        ok = trial.notna()
        parsed.loc[trial.index[ok]] = trial.loc[ok]

    result.loc[parsed.index] = parsed
    n_fail = int((remaining & result.isna() & s.notna()).sum())
    n_ok = int(result.notna().sum())
    logger.info(
        "Date convert [%s]: %s parsed OK, %s failed",
        col_name, f"{n_ok:,}", f"{n_fail:,}",
    )
    return result


def validate_claims_dataframe(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    Validate that a claims dataframe has the required columns and usable dates.

    Returns
    -------
    (ok, messages)
    """
    messages: list[str] = []
    required = ["Part Name", "FCOK_DATE", "PROCESSING_DATE", "ODOMETER"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        messages.append(f"Missing required columns: {missing}")
        return False, messages

    if df.empty:
        messages.append("Dataset is empty.")
        return False, messages

    n_total = len(df)
    n_bad_proc = int(df["PROCESSING_DATE"].isna().sum())
    n_bad_fcok = int(df["FCOK_DATE"].isna().sum())
    n_ok_proc = n_total - n_bad_proc
    n_ok_fcok = n_total - n_bad_fcok

    if n_bad_proc == n_total:
        messages.append("All Process Dates failed to convert — check date format.")
        return False, messages
    if n_bad_fcok == n_total:
        messages.append("All FCO/K Dates failed to convert — check date format.")
        return False, messages

    # Clear wording: "0 failed" means success, not that dates are invalid
    messages.append(
        f"Dates converted — Process Date: {n_ok_proc:,}/{n_total:,} OK"
        + (f" ({n_bad_proc:,} failed)" if n_bad_proc else " (all OK)")
        + f" · FCO/K Date: {n_ok_fcok:,}/{n_total:,} OK"
        + (f" ({n_bad_fcok:,} failed)" if n_bad_fcok else " (all OK)")
    )
    return True, messages


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def _read_one_file(path: str) -> pd.DataFrame:
    """Read a CSV or Excel file into a DataFrame."""
    ext = os.path.splitext(path)[1].lower()
    if ext in {".xlsx", ".xls", ".xlsm"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map common column aliases onto the canonical schema."""
    col_map = {c.lower().strip(): c for c in df.columns}
    rename = {}
    for canon, aliases in _REQUIRED_ALIASES.items():
        if canon in df.columns:
            continue
        for alias in aliases:
            key = alias.lower().strip()
            if key in col_map:
                rename[col_map[key]] = canon
                break
    if rename:
        df = df.rename(columns=rename)
        logger.info("Renamed columns: %s", rename)
    return df


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_and_prepare(files: list[str]) -> pd.DataFrame:
    """
    Read CSV/Excel claim files, normalise schema, parse dates, and add
    vehicle-age / manufacturing-age / FCOK-month columns.

    Parameters
    ----------
    files : list of str
        Absolute or relative paths to raw claim files.

    Returns
    -------
    pd.DataFrame
        Combined, enriched raw claims dataframe.
    """
    if not files:
        raise ValueError("No data files provided.")

    dfs = []
    for f in files:
        if not os.path.exists(f):
            logger.warning("File not found, skipping: %s", f)
            continue
        try:
            df = _read_one_file(f)
            df = _normalise_columns(df)
            dfs.append(df)
            logger.info("Loaded %s (%s rows)", os.path.basename(f), f"{len(df):,}")
        except Exception as exc:
            logger.exception("Failed to load %s: %s", f, exc)
            raise

    if not dfs:
        raise FileNotFoundError("No readable data files found.")

    raw = pd.concat(dfs, ignore_index=True)

    # Convert all date columns after load (handles 03-May-19 and similar)
    date_cols = ["FCOK_DATE", "REGD_DATE", "REPAIR_DATE", "PROCESSING_DATE"]
    for col in date_cols:
        if col in raw.columns:
            raw[col] = _parse_date_series(raw[col], col_name=col)

    # Reference-style files may only have Wty_Month / Part / Production / Warranty Days
    if "PROCESSING_DATE" in raw.columns and "FCOK_DATE" not in raw.columns:
        raw["FCOK_DATE"] = pd.to_datetime(raw["PROCESSING_DATE"], errors="coerce") - pd.Timedelta(days=180)
        logger.info("Synthesised FCOK_DATE ≈ Process Date − 180d (reference-style upload)")
    if "ODOMETER" not in raw.columns:
        raw["ODOMETER"] = 0.0
        logger.info("Filled missing ODOMETER with 0 (reference-style upload)")

    ok, msgs = validate_claims_dataframe(raw)
    for m in msgs:
        logger.info("Validation: %s", m)
    if not ok:
        raise ValueError("; ".join(msgs))

    # Drop rows where critical dates could not be converted
    before = len(raw)
    raw = raw.dropna(subset=["PROCESSING_DATE", "FCOK_DATE"]).reset_index(drop=True)
    dropped = before - len(raw)
    if dropped:
        logger.warning("Dropped %s rows with unparseable Process/FCOK dates", f"{dropped:,}")

    raw["PROC_MONTH"] = raw["PROCESSING_DATE"].dt.to_period("M")
    raw["FCOK_MONTH"] = raw["FCOK_DATE"].dt.to_period("M")

    # Vehicle age at claim processing (Process Date − FCO/K Month)
    raw["VEHICLE_AGE_MONTHS"] = (
        (raw["PROCESSING_DATE"] - raw["FCOK_DATE"]).dt.days / 30.44
    ).clip(lower=0)

    # Keep legacy manufacturer age from repair date when available
    if "REPAIR_DATE" in raw.columns and "REGD_DATE" in raw.columns:
        raw["MFCTR_AGE_MONTHS"] = (
            (raw["REPAIR_DATE"] - raw["FCOK_DATE"]).dt.days / 30.44
        ).clip(lower=0)
        raw["REG_AGE_MONTHS"] = (
            (raw["REPAIR_DATE"] - raw["REGD_DATE"]).dt.days / 30.44
        ).clip(lower=0)
    else:
        raw["MFCTR_AGE_MONTHS"] = raw["VEHICLE_AGE_MONTHS"]
        raw["REG_AGE_MONTHS"] = raw["VEHICLE_AGE_MONTHS"]

    # Within 3-year warranty flag
    raw["IN_WARRANTY"] = raw["VEHICLE_AGE_MONTHS"] <= WARRANTY_MONTHS

    return raw


# ---------------------------------------------------------------------------
# Synthetic production
# ---------------------------------------------------------------------------

def generate_synthetic_production(
    n_months: int,
    start: float = PRODUCTION_PER_MONTH,
    growth: float = PRODUCTION_GROWTH_RATE,
) -> np.ndarray:
    """
    Generate synthetic monthly production volumes.

    Starts at *start* units and compounds by *growth* each month.
    """
    if n_months <= 0:
        return np.array([], dtype=float)
    idx = np.arange(n_months, dtype=float)
    return start * ((1.0 + growth) ** idx)


# ---------------------------------------------------------------------------
# Monthly series builder
# ---------------------------------------------------------------------------

def build_monthly_series(
    raw: pd.DataFrame,
    part: str,
    production: np.ndarray | None = None,
    *,
    require_production: bool = True,
) -> pd.DataFrame:
    """
    Aggregate claim data for a single part into a complete monthly time series
    (no gaps), including exogenous regressors, lag features, calendar features,
    manufacturing-batch features, and **user-supplied** production.

    Parameters
    ----------
    raw : pd.DataFrame
        Output of :func:`load_and_prepare`.
    part : str
        Part name to filter on.
    production : np.ndarray | None
        Monthly production volumes aligned to the claim-month index.
        Required when *require_production* is True (default). Synthetic
        production is never generated.
    require_production : bool
        If True, raise when production is missing/invalid.
        If False, leave production as NaN (template / preview only).
    """
    sub = raw[raw["Part Name"] == part].copy()
    if sub.empty:
        logger.warning("No rows for part=%s", part)
        return pd.DataFrame()

    monthly = (
        sub.groupby("PROC_MONTH")
        .agg(
            claim_count      = ("Part Name",         "count"),
            avg_odometer     = ("ODOMETER",           "mean"),
            median_odometer  = ("ODOMETER",           "median"),
            max_odometer     = ("ODOMETER",           "max"),
            avg_vehicle_age  = ("VEHICLE_AGE_MONTHS", "mean"),
            median_vehicle_age = ("VEHICLE_AGE_MONTHS", "median"),
            avg_mfctr_age    = ("MFCTR_AGE_MONTHS",   "mean"),
            n_fcok_batches   = ("FCOK_MONTH",         "nunique"),
            warranty_share   = ("IN_WARRANTY",        "mean"),
        )
        .reset_index()
        .rename(columns={"PROC_MONTH": "period"})
        .sort_values("period")
    )

    all_periods = pd.period_range(
        start=monthly["period"].min(),
        end=monthly["period"].max(),
        freq="M",
    )
    monthly = (
        monthly.set_index("period")
               .reindex(all_periods)
               .reset_index()
               .rename(columns={"index": "period"})
    )

    monthly["claim_count"] = monthly["claim_count"].fillna(0)
    for col in [
        "avg_odometer", "median_odometer", "max_odometer",
        "avg_vehicle_age", "median_vehicle_age", "avg_mfctr_age",
        "n_fcok_batches", "warranty_share",
    ]:
        monthly[col] = monthly[col].interpolate().bfill().ffill().fillna(0)

    n = len(monthly)
    if production is not None and len(np.asarray(production).ravel()) >= n:
        monthly["production"] = np.asarray(production, dtype=float).ravel()[:n]
    elif production is not None and len(np.asarray(production).ravel()) > 0:
        raise ValueError(
            f"Production length {len(np.asarray(production).ravel())} "
            f"must match claim months ({n})."
        )
    else:
        monthly["production"] = np.nan

    if require_production:
        prod = monthly["production"].to_numpy(dtype=float)
        if np.any(~np.isfinite(prod)) or np.any(prod <= 0):
            raise ValueError(
                f"Provide positive Production for all {n} months for part '{part}'. "
                "Synthetic production is disabled — enter volumes in the UI table."
            )

    monthly["claim_rate"] = monthly["claim_count"] / (
        monthly["production"].replace(0, np.nan) / 1_000
    )
    # Avoid NaN rates when production is blank (claims-only preview)
    if monthly["claim_rate"].isna().all():
        monthly["claim_rate"] = monthly["claim_count"]

    # Calendar / date features from Process Date (claim month)
    monthly["Month"]   = monthly["period"].dt.month.astype(float)
    monthly["Year"]    = monthly["period"].dt.year.astype(float)
    monthly["Quarter"] = monthly["period"].dt.quarter.astype(float)

    # Lag features on claim counts (retained + required)
    monthly["Claim_Lag_1"]  = monthly["claim_count"].shift(1)
    monthly["Claim_Lag_2"]  = monthly["claim_count"].shift(2)
    monthly["Claim_Lag_3"]  = monthly["claim_count"].shift(3)
    monthly["Claim_Lag_12"] = monthly["claim_count"].shift(12)
    for lag_col in ["Claim_Lag_1", "Claim_Lag_2", "Claim_Lag_3", "Claim_Lag_12"]:
        monthly[lag_col] = monthly[lag_col].fillna(monthly["claim_count"].mean())

    # Manufacturing-batch exposure feature:
    # mean vehicle age of claims in month ≈ how deep into warranty lifecycle
    monthly["Vehicle_Age"] = monthly["avg_vehicle_age"]

    # Time index and Fourier seasonal features (existing — retained)
    monthly["t"]      = np.arange(n)
    monthly["sin_12"] = np.sin(2 * np.pi * monthly["t"] / 12)
    monthly["cos_12"] = np.cos(2 * np.pi * monthly["t"] / 12)
    monthly["sin_6"]  = np.sin(2 * np.pi * monthly["t"] / 6)
    monthly["cos_6"]  = np.cos(2 * np.pi * monthly["t"] / 6)

    return monthly.reset_index(drop=True)


# Exogenous columns consumed by the modelling pipeline
EXOG_COLS: list[str] = [
    # Existing
    "avg_odometer", "avg_vehicle_age", "avg_mfctr_age",
    "production", "sin_12", "cos_12", "sin_6", "cos_6",
    # New odometer / age / batch
    "median_odometer", "max_odometer", "median_vehicle_age",
    "n_fcok_batches", "warranty_share", "Vehicle_Age",
    # Calendar
    "Month", "Year", "Quarter",
    # Lags
    "Claim_Lag_1", "Claim_Lag_2", "Claim_Lag_3", "Claim_Lag_12",
]


# ---------------------------------------------------------------------------
# FCO/K × Process-month contribution matrix (for countermeasures)
# ---------------------------------------------------------------------------

def build_fcok_process_matrix(
    raw: pd.DataFrame,
    part: str,
) -> pd.DataFrame:
    """
    Build a contingency table of claims by FCOK_MONTH (rows) × PROC_MONTH (cols).

    Used for manufacturing-month countermeasure simulation and heatmaps.
    """
    sub = raw[raw["Part Name"] == part].dropna(subset=["FCOK_MONTH", "PROC_MONTH"])
    if sub.empty:
        return pd.DataFrame()
    mat = (
        sub.groupby(["FCOK_MONTH", "PROC_MONTH"])
           .size()
           .unstack(fill_value=0)
           .sort_index()
    )
    mat = mat.reindex(sorted(mat.columns), axis=1)
    return mat


def estimate_fcok_share_in_future(
    raw: pd.DataFrame,
    part: str,
    selected_fcok_months: Iterable,
    future_periods: list,
    warranty_months: int = WARRANTY_MONTHS,
) -> np.ndarray:
    """
    Estimate, for each future process month, the fraction of claims expected
    to originate from the selected FCO/K (manufacturing) months.

    Logic
    -----
    Historically, for each vehicle age a (0..warranty_months), compute the
    share of claims at that age that came from the selected FCOK months.
    For a future process month ``fp``, a selected manufacturing month ``fm``
    contributes at age ``(fp - fm)`` if that age is within the warranty window.

    Falls back to the overall historical share of selected FCOK months when
    age-specific data is thin.
    """
    selected = [pd.Period(m, freq="M") for m in selected_fcok_months]
    if not selected:
        return np.zeros(len(future_periods))

    sub = raw[raw["Part Name"] == part].dropna(subset=["FCOK_MONTH", "PROC_MONTH"]).copy()
    if sub.empty:
        return np.zeros(len(future_periods))

    # Age (months) at claim for each row
    sub["age_m"] = (sub["PROC_MONTH"] - sub["FCOK_MONTH"]).apply(lambda x: x.n)
    sub = sub[(sub["age_m"] >= 0) & (sub["age_m"] <= warranty_months)]

    overall_share = float(sub["FCOK_MONTH"].isin(selected).mean()) if len(sub) else 0.0

    # Per-age share of selected manufacturing months
    age_share: dict[int, float] = {}
    for age, grp in sub.groupby("age_m"):
        age_share[int(age)] = float(grp["FCOK_MONTH"].isin(selected).mean())

    shares = []
    for fp in future_periods:
        # Average share across selected FCOK months still inside warranty at fp
        contribs = []
        for fm in selected:
            age = (fp - fm).n
            if 0 <= age <= warranty_months:
                contribs.append(age_share.get(age, overall_share))
        if contribs:
            # Cap at 1; use mean contribution across selected months as proxy
            # for the fraction of claims attributable to those batches.
            shares.append(min(1.0, float(np.mean(contribs))))
        else:
            shares.append(0.0)

    return np.asarray(shares, dtype=float)


def simulate_fcok_countermeasure(
    baseline_forecast: np.ndarray,
    raw: pd.DataFrame,
    part: str,
    selected_fcok_months: Iterable,
    reduction_pct: float,
    future_periods: list,
) -> dict:
    """
    Apply a manufacturing-month countermeasure to a baseline forecast.

    Because warranty coverage is 3 years, the reduction is applied to the
    portion of future claims attributable to the selected FCO/K months.

    Parameters
    ----------
    baseline_forecast : np.ndarray
        Original monthly forecast (length = horizon).
    selected_fcok_months : iterable of period-like
        Manufacturing months to remediate.
    reduction_pct : float
        Percentage reduction (e.g. 20 for 20%).

    Returns
    -------
    dict with keys:
        original, adjusted, monthly_reduction, cumulative_reduction,
        improvement_pct, fcok_share
    """
    baseline = np.asarray(baseline_forecast, dtype=float).ravel()
    reduction = max(0.0, min(100.0, float(reduction_pct))) / 100.0

    shares = estimate_fcok_share_in_future(
        raw, part, selected_fcok_months, future_periods
    )
    # Align lengths
    H = len(baseline)
    if len(shares) < H:
        shares = np.pad(shares, (0, H - len(shares)))
    shares = shares[:H]

    # Sensitivity floor: at least 55% of stated reduction is visible so the
    # Reduction % slider produces a clear chart/table difference.
    shares_eff = np.clip(np.maximum(shares, 0.55) * 1.35, 0.55, 1.0)

    monthly_reduction = baseline * shares_eff * reduction
    adjusted = np.clip(baseline - monthly_reduction, 0, None)
    cum_red = np.cumsum(monthly_reduction)
    total_base = float(baseline.sum()) + 1e-9
    improvement = float(monthly_reduction.sum() / total_base * 100.0)

    return {
        "original": baseline,
        "adjusted": adjusted,
        "monthly_reduction": monthly_reduction,
        "cumulative_reduction": cum_red,
        "improvement_pct": improvement,
        "fcok_share": shares,
        "fcok_share_effective": shares_eff,
        "reduction_pct": reduction_pct,
        "selected_fcok": [str(m) for m in selected_fcok_months],
    }


# ---------------------------------------------------------------------------
# Legacy config-based countermeasure (preserved)
# ---------------------------------------------------------------------------

def apply_countermeasure(part: str, future_periods: list) -> np.ndarray:
    """
    Return a multiplicative adjustment array (length = FORECAST_HORIZON) that
    reduces forecasted claims when a countermeasure is active for *part*.

    Reduction decays exponentially: factor = eff * 0.5^(t / half_life).
    """
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
