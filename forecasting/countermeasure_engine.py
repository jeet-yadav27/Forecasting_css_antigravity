"""
forecasting/countermeasure_engine.py
--------------------------------------
Standalone countermeasure (CM) logic engine for automotive warranty
claim forecasting.

Business Logic Summary
----------------------
1.  Check whether a CM exists for the selected part.
    → No CM   : return baseline forecast unchanged.
    → CM exists: execute steps 2–7.

2.  Analyse historical warranty data grouped by FCOK month.
3.  Identify the FCOK months with the highest claims.
4.  Calculate the average production volume of those peak months.
5.  Derive the production-adjusted baseline:
        adj_prod[t] = avg_peak_prod - cumulative_cm_claims[t]
    This captures the net exposure reduction caused by the CM.
6.  Feed adjusted production into a production-weighted claim-rate
    model to project future claims under the CM scenario.
    Factor = 1.0 means the full stated reduction is applied with no
    softening (known / confirmed CM effectiveness).
7.  Compare with-CM vs without-CM forecasts and compute savings.

Warranty window
---------------
Claims are considered only within 36 months (WARRANTY_MONTHS) of the
FCO/K manufacturing month. The CM impact is therefore evaluated across
three years from the CM implementation date.

Statistical note
----------------
The claim rate (claims per 1 000 production units) is used as the
proportionality bridge between production volume and future claims.
This avoids the dimensional mismatch of subtracting claims directly
from production counts. The net-exposure approach (adj_prod = avg_prod
- avg_claims) is retained as the *offset* that reduces the exposure
denominator, giving higher weight to production changes post-CM.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

from forecasting.config import WARRANTY_MONTHS, FORECAST_HORIZON

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Countermeasure presence check
# ---------------------------------------------------------------------------

def check_countermeasure_exists(
    part: str,
    cm_registry: dict | None = None,
    *,
    cm_enabled_flag: bool = False,
) -> bool:
    """
    Return True when a countermeasure is registered for *part*.

    Parameters
    ----------
    part : str
        Part name as it appears in the claims data.
    cm_registry : dict, optional
        Mapping ``{part_name: {month, effectiveness, …}}``.
        Defaults to the config-level COUNTERMEASURES dict.
    cm_enabled_flag : bool
        UI-level override — when True, treat CM as active regardless
        of the registry (user has explicitly enabled it in the UI).
    """
    if cm_enabled_flag:
        return True
    if cm_registry is None:
        from forecasting.config import COUNTERMEASURES
        cm_registry = COUNTERMEASURES
    return part in cm_registry


# ---------------------------------------------------------------------------
# 2 & 3. Peak FCOK-month identification
# ---------------------------------------------------------------------------

def identify_peak_fcok_months(
    raw: pd.DataFrame,
    part: str,
    top_n: int = 3,
) -> pd.DataFrame:
    """
    Rank FCOK (manufacturing) months by total claim count for *part*.

    Returns
    -------
    pd.DataFrame
        Columns: fcok_month, claim_count, rank
        Sorted descending by claim_count (rank 1 = highest).
    """
    sub = raw[raw["Part Name"] == part].dropna(subset=["FCOK_MONTH"])
    if sub.empty:
        return pd.DataFrame(columns=["fcok_month", "claim_count", "rank"])

    by_fcok = (
        sub.groupby("FCOK_MONTH")
           .size()
           .reset_index(name="claim_count")
           .rename(columns={"FCOK_MONTH": "fcok_month"})
           .sort_values("claim_count", ascending=False)
           .reset_index(drop=True)
    )
    by_fcok["rank"] = by_fcok.index + 1
    return by_fcok.head(top_n)


# ---------------------------------------------------------------------------
# 4 & 5. Adjusted production baseline
# ---------------------------------------------------------------------------

def compute_adjusted_baseline_production(
    raw: pd.DataFrame,
    part: str,
    production_series: np.ndarray,
    monthly_periods,
    top_n: int = 3,
) -> dict:
    """
    Compute the production-adjusted baseline from peak FCOK months.

    Logic
    -----
    For the top-N FCOK months (by claim count):
      avg_peak_prod   = mean production of those months
      avg_peak_claims = mean claim count of those months
      adj_prod        = avg_peak_prod - avg_peak_claims

    The *adj_prod* figure is used as the starting point for future
    production. Each subsequent month's production is decremented by
    the average CM-period claims, simulating a systematic reduction
    in exposure.

    Parameters
    ----------
    raw : pd.DataFrame
        Output of load_and_prepare — must contain FCOK_MONTH column.
    part : str
        Part name to filter on.
    production_series : np.ndarray
        Historical production volumes aligned to *monthly_periods*.
    monthly_periods : array-like of pd.Period
        Monthly periods corresponding to *production_series*.

    Returns
    -------
    dict
        avg_peak_prod, avg_peak_claims, adj_prod, peak_fcok_df,
        monthly_fcok_prod (Series: FCOK_MONTH → production)
    """
    peak_df = identify_peak_fcok_months(raw, part, top_n=top_n)
    if peak_df.empty:
        avg_prod = float(np.nanmean(production_series)) if len(production_series) else 0.0
        return {
            "avg_peak_prod": avg_prod,
            "avg_peak_claims": 0.0,
            "adj_prod": avg_prod,
            "peak_fcok_df": peak_df,
            "monthly_fcok_prod": pd.Series(dtype=float),
        }

    periods = list(monthly_periods)
    prod_arr = np.asarray(production_series, dtype=float)

    # Map each FCOK month to its production value (process-month lookup)
    period_to_prod: dict = {}
    for i, p in enumerate(periods):
        if i < len(prod_arr):
            period_to_prod[str(p)] = float(prod_arr[i])

    # Production for each peak FCOK month
    peak_prod_vals: list[float] = []
    for fm in peak_df["fcok_month"]:
        pv = period_to_prod.get(str(fm), np.nan)
        if np.isfinite(pv) and pv > 0:
            peak_prod_vals.append(pv)

    avg_peak_prod = (
        float(np.mean(peak_prod_vals))
        if peak_prod_vals
        else float(np.nanmean(prod_arr)) if np.any(np.isfinite(prod_arr)) else 25_000.0
    )
    avg_peak_claims = float(peak_df["claim_count"].mean())
    # Net exposure offset: reduce production by average peak-month claims
    adj_prod = max(avg_peak_prod - avg_peak_claims, 1.0)

    monthly_fcok_prod = pd.Series(period_to_prod, name="production")

    return {
        "avg_peak_prod": avg_peak_prod,
        "avg_peak_claims": avg_peak_claims,
        "adj_prod": adj_prod,
        "peak_fcok_df": peak_df,
        "monthly_fcok_prod": monthly_fcok_prod,
    }


# ---------------------------------------------------------------------------
# 6. Future production sequence (CM-adjusted)
# ---------------------------------------------------------------------------

def build_cm_adjusted_production(
    adj_prod_start: float,
    avg_peak_claims: float,
    n_months: int,
    warranty_months: int = WARRANTY_MONTHS,
) -> np.ndarray:
    """
    Build a future production sequence that decreases from the adjusted
    baseline, reflecting the ongoing exposure reduction post-CM.

    Rule (from spec)
    ----------------
    Month t production = adj_prod_start − t × avg_peak_claims
    where t ∈ [0, min(n_months, warranty_months) − 1].

    After the warranty window, production returns to adj_prod_start
    (new manufacturing batches are unaffected by the CM).

    Parameters
    ----------
    adj_prod_start : float
        Starting adjusted production (avg_peak_prod - avg_peak_claims).
    avg_peak_claims : float
        Average claims from the peak FCOK months (the decrement step).
    n_months : int
        Number of future forecast months.
    warranty_months : int
        Warranty duration in months; default 36.

    Returns
    -------
    np.ndarray, shape (n_months,)
        Future production volumes (all positive, floor at 1).
    """
    result = np.empty(n_months, dtype=float)
    for t in range(n_months):
        if t < warranty_months:
            result[t] = adj_prod_start - t * avg_peak_claims
        else:
            result[t] = adj_prod_start  # warranty expired, new batch baseline
    return np.clip(result, 1.0, None)


# ---------------------------------------------------------------------------
# 6. CM-adjusted forecast (production-weighted claim rate model)
# ---------------------------------------------------------------------------

def compute_cm_forecast(
    baseline_forecast: np.ndarray,
    cm_adjusted_production: np.ndarray,
    hist_production: np.ndarray,
    hist_claims: np.ndarray,
    *,
    warranty_months: int = WARRANTY_MONTHS,
) -> np.ndarray:
    """
    Project future claims under the CM scenario using a pure production-
    weighted claim-rate approach.

    Formula
    -------
    hist_rate      = mean(hist_claims / hist_production)   [per unit]
    cm_forecast[t] = hist_rate x cm_adjusted_production[t]

    There is **no factor and no blending with the baseline**.
    The adjusted production sequence fully determines the forecast
    because we do not know the exact magnitude of reduction caused by
    the countermeasure; only the production exposure change is known.
    Any factor or blending would introduce an unquantified assumption.

    Parameters
    ----------
    baseline_forecast : np.ndarray
        Original monthly claim forecast (used for length reference only).
    cm_adjusted_production : np.ndarray
        Post-CM production sequence from build_cm_adjusted_production().
    hist_production : np.ndarray
        Historical monthly production volumes.
    hist_claims : np.ndarray
        Historical monthly claim counts.
    warranty_months : int
        Warranty window (for length alignment only).

    Returns
    -------
    np.ndarray
        CM-adjusted monthly claim forecast, same length as baseline.
    """
    baseline = np.asarray(baseline_forecast, dtype=float).ravel()
    cm_prod = np.asarray(cm_adjusted_production, dtype=float).ravel()
    H = len(baseline)

    hist_prod = np.asarray(hist_production, dtype=float).ravel()
    hist_cl = np.asarray(hist_claims, dtype=float).ravel()

    # Historical claim rate (claims per production unit)
    valid_mask = np.isfinite(hist_prod) & (hist_prod > 0) & np.isfinite(hist_cl)
    if valid_mask.any():
        hist_rate = float(np.nanmean(hist_cl[valid_mask] / hist_prod[valid_mask]))
    else:
        hist_rate = float(np.nanmean(hist_cl)) / max(
            float(np.nanmean(hist_prod)), 1.0
        )
    hist_rate = max(hist_rate, 1e-10)

    # Align cm_prod to forecast horizon
    cm_prod_aligned = cm_prod[:H] if len(cm_prod) >= H else np.pad(
        cm_prod, (0, H - len(cm_prod)), mode="edge"
    )

    # Pure production-rate forecast: no factor, no blend with baseline
    cm_fc = hist_rate * cm_prod_aligned
    return np.clip(cm_fc, 0.0, None)


# ---------------------------------------------------------------------------
# 7. Comparison: with vs without CM
# ---------------------------------------------------------------------------

def compare_forecasts(
    baseline: np.ndarray,
    cm_forecast: np.ndarray,
    future_periods,
    cost_per_claim: float | None = None,
) -> dict:
    """
    Build a structured comparison of with-CM vs without-CM forecasts.

    Returns
    -------
    dict with:
        baseline, cm_forecast, monthly_reduction, cumulative_reduction,
        pct_reduction (array and total), cost_savings (when available),
        total_baseline_claims, total_cm_claims, total_reduction,
        total_pct_reduction, total_cost_savings, comparison_df
    """
    baseline = np.asarray(baseline, dtype=float).ravel()
    cm_fc = np.asarray(cm_forecast, dtype=float).ravel()
    H = min(len(baseline), len(cm_fc))
    baseline, cm_fc = baseline[:H], cm_fc[:H]

    monthly_red = np.clip(baseline - cm_fc, 0.0, None)
    cum_red = np.cumsum(monthly_red)
    pct_red = np.where(baseline > 0, monthly_red / baseline * 100.0, 0.0)

    total_base = float(baseline.sum())
    total_cm = float(cm_fc.sum())
    total_red = float(monthly_red.sum())
    total_pct = float(total_red / (total_base + 1e-9) * 100.0)

    cost_sav: np.ndarray | None = None
    total_cost_sav: float | None = None
    if cost_per_claim is not None and np.isfinite(cost_per_claim) and cost_per_claim > 0:
        cost_sav = monthly_red * float(cost_per_claim)
        total_cost_sav = float(cost_sav.sum())

    periods_str = [str(p) for p in list(future_periods)[:H]]
    comp_df = pd.DataFrame({
        "Month": periods_str,
        "Baseline_Claims": baseline,
        "CM_Claims": cm_fc,
        "Monthly_Reduction": monthly_red,
        "Cumulative_Reduction": cum_red,
        "Reduction_%": pct_red,
    })
    if cost_sav is not None:
        comp_df["Cost_Savings"] = cost_sav

    return {
        "baseline": baseline,
        "cm_forecast": cm_fc,
        "monthly_reduction": monthly_red,
        "cumulative_reduction": cum_red,
        "pct_reduction": pct_red,
        "cost_savings": cost_sav,
        "total_baseline_claims": total_base,
        "total_cm_claims": total_cm,
        "total_reduction": total_red,
        "total_pct_reduction": total_pct,
        "total_cost_savings": total_cost_sav,
        "cost_per_claim": cost_per_claim,
        "comparison_df": comp_df,
    }


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def run_cm_analysis(
    raw: pd.DataFrame,
    part: str,
    baseline_forecast: np.ndarray,
    future_periods: list,
    production_series: np.ndarray,
    monthly_periods,
    hist_claims: np.ndarray,
    *,
    cm_enabled: bool = False,
    cm_registry: dict | None = None,
    cm_date: str | None = None,
    top_n_fcok: int = 3,
    cost_per_claim: float | None = None,
    warranty_months: int = WARRANTY_MONTHS,
) -> dict:
    """
    Full countermeasure analysis orchestrator.

    Workflow
    --------
    1. Check CM presence -> return baseline if no CM.
    2. Identify peak FCOK months.
    3. Compute adjusted production baseline.
    4. Build CM-adjusted future production sequence.
    5. Compute CM-adjusted forecast (pure rate x production, no factor).
    6. Compare with-CM vs without-CM.

    Parameters
    ----------
    raw : pd.DataFrame
        Full claims dataframe (output of load_and_prepare).
    part : str
        Part name.
    baseline_forecast : np.ndarray
        12-month (or FORECAST_HORIZON-month) baseline forecast.
    future_periods : list
        Future monthly periods (pd.Period or str) aligned to forecast.
    production_series : np.ndarray
        Historical monthly production volumes.
    monthly_periods : array-like
        Monthly periods aligned to production_series.
    hist_claims : np.ndarray
        Historical monthly claim counts.
    cm_enabled : bool
        UI-level CM enable flag.
    cm_registry : dict, optional
        Config-level CM registry. Falls back to COUNTERMEASURES.
    cm_date : str, optional
        CM implementation date string (e.g. "2024-03").
    top_n_fcok : int
        How many peak FCOK months to use for baseline calculation.
    cost_per_claim : float, optional
        Average warranty repair cost per claim (for savings estimate).
    warranty_months : int
        Warranty window in months.

    Returns
    -------
    dict
        ``cm_active`` (bool), ``cm_production``, ``cm_forecast``,
        ``comparison``, ``peak_fcok_df``, ``avg_peak_prod``,
        ``avg_peak_claims``, ``adj_prod``, ``cm_date``,
        ``warranty_months``.
    """
    # ── Step 1: presence check ───────────────────────────────────────────
    has_cm = check_countermeasure_exists(
        part, cm_registry, cm_enabled_flag=cm_enabled
    )
    no_cm_result = {
        "cm_active": False,
        "baseline": np.asarray(baseline_forecast, dtype=float).ravel(),
        "cm_forecast": np.asarray(baseline_forecast, dtype=float).ravel(),
        "peak_fcok_df": pd.DataFrame(),
        "avg_peak_prod": float(np.nanmean(production_series)) if len(production_series) else 0.0,
        "avg_peak_claims": 0.0,
        "adj_prod": float(np.nanmean(production_series)) if len(production_series) else 0.0,
        "cm_production": np.asarray(
            [float(np.nanmean(production_series))] * len(baseline_forecast)
        ),
        "cm_date": cm_date,
        "warranty_months": warranty_months,
        "comparison": compare_forecasts(
            baseline_forecast, baseline_forecast, future_periods, cost_per_claim
        ),
        "message": "No countermeasure active — baseline forecast returned.",
    }
    if not has_cm:
        logger.info("[CM] No CM found for part=%s — returning baseline.", part)
        return no_cm_result

    logger.info("[CM] Running CM analysis for part=%s (pure rate x production, no factor)", part)

    # ── Step 2 & 3: peak FCOK + adjusted baseline ────────────────────────
    prod_baseline = compute_adjusted_baseline_production(
        raw, part, production_series, monthly_periods, top_n=top_n_fcok
    )
    avg_peak_prod = prod_baseline["avg_peak_prod"]
    avg_peak_claims = prod_baseline["avg_peak_claims"]
    adj_prod = prod_baseline["adj_prod"]
    peak_df = prod_baseline["peak_fcok_df"]

    logger.info(
        "[CM] avg_peak_prod=%.0f  avg_peak_claims=%.1f  adj_prod=%.0f",
        avg_peak_prod, avg_peak_claims, adj_prod,
    )

    # ── Step 4: future production sequence ───────────────────────────────
    n_future = len(future_periods)
    cm_prod = build_cm_adjusted_production(
        adj_prod_start=adj_prod,
        avg_peak_claims=avg_peak_claims,
        n_months=n_future,
        warranty_months=warranty_months,
    )

    # ── Step 5: CM-adjusted forecast (pure rate x production) ────────────
    cm_fc = compute_cm_forecast(
        baseline_forecast=baseline_forecast,
        cm_adjusted_production=cm_prod,
        hist_production=production_series,
        hist_claims=hist_claims,
        warranty_months=warranty_months,
    )

    # ── Step 6: comparison ────────────────────────────────────────────────
    comparison = compare_forecasts(
        baseline_forecast, cm_fc, future_periods, cost_per_claim
    )

    logger.info(
        "[CM] Total reduction: %.0f claims (%.1f%%)  Cost savings: %s",
        comparison["total_reduction"],
        comparison["total_pct_reduction"],
        f"{comparison['total_cost_savings']:,.0f}" if comparison["total_cost_savings"] is not None else "N/A",
    )

    return {
        "cm_active": True,
        "baseline": np.asarray(baseline_forecast, dtype=float).ravel(),
        "cm_forecast": cm_fc,
        "peak_fcok_df": peak_df,
        "avg_peak_prod": avg_peak_prod,
        "avg_peak_claims": avg_peak_claims,
        "adj_prod": adj_prod,
        "cm_production": cm_prod,
        "cm_date": cm_date,
        "warranty_months": warranty_months,
        "comparison": comparison,
        "message": (
            f"CM active — {comparison['total_pct_reduction']:.1f}% claim reduction "
            f"({comparison['total_reduction']:.0f} claims over {n_future} months)."
        ),
    }


# ---------------------------------------------------------------------------
# Standalone script entry-point (for use outside Gradio app)
# ---------------------------------------------------------------------------

def demo_cm_analysis(
    csv_path: str,
    part: str,
    baseline_claims: list[float],
    future_months: list[str],
    production_values: list[float],
    *,
    factor: float = 1.0,
    cost_per_claim: float | None = None,
    top_n: int = 3,
) -> None:
    """
    Demonstration entry-point — run CM analysis from a CSV file without
    the Gradio UI. Prints a summary table and key metrics.

    Parameters
    ----------
    csv_path : str
        Path to the raw claims CSV/Excel file.
    part : str
        Part name to analyse.
    baseline_claims : list of float
        Baseline forecast claims (e.g. from a prior model run).
    future_months : list of str
        Future month strings aligned to baseline_claims.
    production_values : list of float
        Historical production volumes (same length as claim history).
    factor : float
        CM effectiveness factor (1.0 = fully known).
    cost_per_claim : float, optional
        Warranty cost per claim in currency units.
    top_n : int
        Number of peak FCOK months to use.
    """
    from forecasting.data.loader import load_and_prepare, build_monthly_series

    print("\n" + "=" * 64)
    print("  COUNTERMEASURE ANALYSIS — Standalone Demo")
    print("=" * 64)

    raw = load_and_prepare([csv_path])
    prod = np.asarray(production_values, dtype=float)
    monthly = build_monthly_series(raw, part, production=prod, require_production=False)

    if monthly.empty:
        print(f"[!] No data found for part: {part}")
        return

    hist_claims = monthly["claim_count"].values.astype(float)
    hist_prod = monthly["production"].values.astype(float)
    hist_periods = monthly["period"].tolist()

    future_periods = [pd.Period(m, freq="M") for m in future_months]
    baseline = np.asarray(baseline_claims, dtype=float)

    result = run_cm_analysis(
        raw=raw,
        part=part,
        baseline_forecast=baseline,
        future_periods=future_periods,
        production_series=hist_prod,
        monthly_periods=hist_periods,
        hist_claims=hist_claims,
        cm_enabled=True,
        factor=factor,
        top_n_fcok=top_n,
        cost_per_claim=cost_per_claim,
    )

    print(f"\nPart        : {part}")
    print(f"CM Active   : {result['cm_active']}")
    print(f"Factor      : {result['factor']}")
    print(f"Avg Peak Prod    : {result['avg_peak_prod']:,.0f}")
    print(f"Avg Peak Claims  : {result['avg_peak_claims']:,.1f}")
    print(f"Adj. Prod Baseline: {result['adj_prod']:,.0f}")
    print()
    print("── Peak FCOK Months ──")
    print(result["peak_fcok_df"].to_string(index=False))
    print()
    print("── Forecast Comparison ──")
    comp = result["comparison"]
    print(comp["comparison_df"].to_string(index=False, float_format="{:.1f}".format))
    print()
    print(f"Total Baseline Claims : {comp['total_baseline_claims']:,.0f}")
    print(f"Total CM Claims       : {comp['total_cm_claims']:,.0f}")
    print(f"Total Reduction       : {comp['total_reduction']:,.0f}  "
          f"({comp['total_pct_reduction']:.1f}%)")
    if comp["total_cost_savings"] is not None:
        print(f"Estimated Cost Savings: {comp['total_cost_savings']:,.2f}")
    print()
    print(result["message"])
    print("=" * 64)
