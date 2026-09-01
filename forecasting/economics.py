"""
forecasting/economics.py
------------------------
Production, CPV (cost per vehicle), claim-ratio, and countermeasure helpers.
"""

from __future__ import annotations

import logging
import os
from typing import Iterable

import numpy as np
import pandas as pd

from forecasting.config import FORECAST_HORIZON, WARRANTY_MONTHS

logger = logging.getLogger(__name__)


def month_input_template(raw: pd.DataFrame, part: str) -> pd.DataFrame:
    """
    Build an editable template: one row per claim month for *part*.

    Columns: Month, Production, Production_Cost
    Production / cost start empty (NaN) — user must fill them (no synthetic).
    """
    from forecasting.data.loader import build_monthly_series

    # Build calendar only; production left empty for the user
    monthly = build_monthly_series(
        raw, part, production=None, require_production=False,
    )
    if monthly.empty:
        return pd.DataFrame(columns=["Month", "Production", "Production_Cost"])

    return pd.DataFrame({
        "Month": [str(p) for p in monthly["period"]],
        "Production": [np.nan] * len(monthly),
        "Production_Cost": [np.nan] * len(monthly),
    })


def production_sheet_to_csv(df: pd.DataFrame, path: str | None = None) -> str:
    """Write the Month / Production / Production_Cost sheet to a CSV file."""
    import tempfile

    work = df.copy() if df is not None else pd.DataFrame(
        columns=["Month", "Production", "Production_Cost"]
    )
    # Ensure canonical columns
    for col in ("Month", "Production", "Production_Cost"):
        if col not in work.columns:
            work[col] = np.nan
    work = work[["Month", "Production", "Production_Cost"]]
    if path is None:
        fd, path = tempfile.mkstemp(suffix=".csv", prefix="production_cost_sheet_")
        os.close(fd)
    work.to_csv(path, index=False)
    return path


def load_production_sheet_csv(path: str) -> pd.DataFrame:
    """
    Load a user-filled production/cost CSV or Excel into the canonical table.

    Accepts columns named Month / Production / Production_Cost
    (case-insensitive; common aliases allowed).
    """
    path_l = str(path).lower()
    if path_l.endswith((".xlsx", ".xls")):
        # Prefer Production sheet when present
        try:
            xl = pd.ExcelFile(path)
            sheet = "Production" if "Production" in xl.sheet_names else xl.sheet_names[0]
            df = pd.read_excel(path, sheet_name=sheet)
        except Exception:
            df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Uploaded production sheet is empty.")
    col_map = {c.lower().strip(): c for c in df.columns}
    rename = {}
    for canon, aliases in {
        "Month": ["month", "period", "wty_month", "claim month"],
        "Production": ["production", "prod", "volume", "units"],
        "Production_Cost": [
            "production_cost", "production cost", "cost", "prod_cost",
        ],
    }.items():
        if canon in df.columns:
            continue
        for a in aliases:
            if a in col_map:
                rename[col_map[a]] = canon
                break
    if rename:
        df = df.rename(columns=rename)
    missing = [c for c in ("Month", "Production") if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV/Excel must include columns Month and Production (missing: {missing}). "
            "Download the template, fill it, then upload."
        )
    if "Production_Cost" not in df.columns:
        df["Production_Cost"] = np.nan
    out = df[["Month", "Production", "Production_Cost"]].copy()
    out["Month"] = out["Month"].astype(str)
    out["Production"] = pd.to_numeric(out["Production"], errors="coerce")
    out["Production_Cost"] = pd.to_numeric(out["Production_Cost"], errors="coerce")
    return out.reset_index(drop=True)


def parse_month_inputs(
    df: pd.DataFrame,
    *,
    require_production: bool = False,
) -> tuple[np.ndarray | None, np.ndarray | None, list[str]]:
    """
    Parse user production + cost table.

    When *require_production* is False (default), empty / missing production
    is allowed and returns ``(None, None, months)`` so claims-only runs work.
    """
    if df is None or len(df) == 0:
        if require_production:
            raise ValueError(
                "Enter monthly Production (and optionally Production_Cost) "
                "for every claim month before training."
            )
        return None, None, []

    work = df.copy()
    cols = {c.lower().strip(): c for c in work.columns}
    prod_col = cols.get("production")
    cost_col = cols.get("production_cost") or cols.get("production cost") or cols.get("cost")
    month_col = cols.get("month")
    months = (
        work[month_col].astype(str).tolist()
        if month_col else [str(i) for i in range(len(work))]
    )

    if prod_col is None:
        if require_production:
            raise ValueError("Production column is required in the monthly input table.")
        return None, None, months

    production = pd.to_numeric(work[prod_col], errors="coerce").to_numpy(dtype=float)
    all_missing = bool(np.all(~np.isfinite(production) | (production <= 0)))
    if all_missing:
        if require_production:
            raise ValueError(
                "Fill Production with positive values for every month."
            )
        return None, None, months

    if np.any(~np.isfinite(production)) or np.any(production <= 0):
        bad = [months[i] for i, v in enumerate(production) if not (np.isfinite(v) and v > 0)]
        if require_production:
            raise ValueError(
                "Fill Production with positive values for every month. "
                f"Missing/invalid: {', '.join(bad[:8])}{'…' if len(bad) > 8 else ''}."
            )
        # Partial fill → treat as no production (claims-only) rather than fail
        return None, None, months

    if cost_col is not None:
        costs = pd.to_numeric(work[cost_col], errors="coerce").to_numpy(dtype=float)
        costs = np.where(np.isfinite(costs) & (costs >= 0), costs, np.nan)
    else:
        costs = np.full(len(production), np.nan, dtype=float)

    return production, costs, months


def build_upload_templates(path: str | None = None) -> str:
    """
    Write an Excel workbook with Claims + Production template sheets
    (and matching CSVs alongside when path ends with .xlsx).

    Returns absolute path to the Excel file.
    """
    import tempfile

    if path is None:
        fd, path = tempfile.mkstemp(suffix=".xlsx", prefix="claims_production_template_")
        os.close(fd)

    claims = pd.DataFrame({
        "Part Name": ["Part_1", "Part_1", "Part_1"],
        "FCOK_DATE": ["2022-01-15", "2022-02-10", "2022-03-05"],
        "PROCESSING_DATE": ["2023-06-20", "2023-07-12", "2023-08-01"],
        "ODOMETER": [12000, 18500, 22000],
    })
    production = pd.DataFrame({
        "Month": ["2023-06", "2023-07", "2023-08"],
        "Production": [25000, 26000, 25500],
        "Production_Cost": [1_250_000, 1_300_000, 1_275_000],
    })
    monthly_combo = pd.DataFrame({
        "Month": ["2023-06", "2023-07", "2023-08"],
        "Part Name": ["Part_1", "Part_1", "Part_1"],
        "Claims": [42, 38, 45],
        "Production": [25000, 26000, 25500],
        "Production_Cost": [1_250_000, 1_300_000, 1_275_000],
    })

    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            claims.to_excel(writer, sheet_name="Claims", index=False)
            production.to_excel(writer, sheet_name="Production", index=False)
            monthly_combo.to_excel(writer, sheet_name="Monthly_Optional", index=False)
            notes = pd.DataFrame({
                "Notes": [
                    "Sheet Claims: upload transactional warranty rows (required columns shown).",
                    "Sheet Production: optional — Month + Production (+ Production_Cost for CPV).",
                    "Sheet Monthly_Optional: example combined monthly view (reference only).",
                    "You may upload claims alone; production is optional for forecasting.",
                    "Countermeasure date is optional; expected reduction % is optional "
                    "(app estimates from data when unknown).",
                ]
            })
            notes.to_excel(writer, sheet_name="README", index=False)
        return path
    except Exception as exc:
        logger.warning("Excel template failed (%s); writing CSV fallbacks.", exc)
        import zipfile
        base = os.path.splitext(path)[0]
        claims_path = base + "_claims.csv"
        prod_path = base + "_production.csv"
        claims.to_csv(claims_path, index=False)
        production.to_csv(prod_path, index=False)
        zip_path = base + "_templates.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(claims_path, arcname="Claims_template.csv")
            zf.write(prod_path, arcname="Production_template.csv")
        return zip_path


def estimate_cm_reduction_pct(
    raw: pd.DataFrame,
    part: str,
    cm_month: str | None,
    hist_claims: np.ndarray,
    hist_periods,
    hist_production: np.ndarray | None = None,
) -> dict:
    """
    Estimate expected claim reduction % when the user does not supply one.

    Prefer observed pre/post CM claim-rate change (production-normalized when
    production exists). Otherwise forecast a reduction from recent trend and
    peak FCO/K share.
    """
    claims = np.asarray(hist_claims, dtype=float).ravel()
    periods = [
        p if isinstance(p, pd.Period) else pd.Period(p, freq="M")
        for p in list(hist_periods)[: len(claims)]
    ]
    cm_p = parse_cm_period(cm_month)

    if hist_production is not None and len(hist_production) >= len(claims):
        prod = np.asarray(hist_production, dtype=float).ravel()[: len(claims)]
        if np.any(np.isfinite(prod) & (prod > 0)):
            rates = claims / np.where(prod > 0, prod, np.nan)
        else:
            rates = claims.copy()
    else:
        rates = claims.copy()

    note = ""
    source = "forecasted"

    if cm_p is not None and len(periods) == len(rates):
        pre_idx = [i for i, p in enumerate(periods) if p < cm_p]
        post_idx = [i for i, p in enumerate(periods) if p >= cm_p]
        if len(pre_idx) >= 2 and len(post_idx) >= 1:
            pre = float(np.nanmean(rates[pre_idx]))
            post = float(np.nanmean(rates[post_idx]))
            if np.isfinite(pre) and pre > 0 and np.isfinite(post):
                drop = (pre - post) / pre * 100.0
                if drop >= 2.0:
                    pct = float(np.clip(drop, 5.0, 85.0))
                    return {
                        "reduction_pct": pct,
                        "source": "observed_post_cm",
                        "note": (
                            f"Estimated {pct:.0f}% from observed claim-rate drop "
                            f"after CM date {cm_p}."
                        ),
                    }
                note = "Little/no drop observed after CM date; using trend + FCO/K estimate."

    # Forecasted reduction (no usable post-CM history)
    estimated = 25.0
    peak, peak_n = peak_fcok_month(raw, part)
    if peak is not None and raw is not None and len(raw):
        sub = raw[raw["Part Name"] == part]
        if not sub.empty and "FCOK_MONTH" in sub.columns:
            share = float((sub["FCOK_MONTH"].astype(str) == str(peak)).mean())
            estimated = float(np.clip(share * 55.0, 15.0, 55.0))
            note = (
                f"Forecasted ~{estimated:.0f}% from peak FCO/K `{peak}` "
                f"share ({share:.0%}, {peak_n:,} claims)."
            )

    if len(rates) >= 8:
        recent = float(np.nanmean(rates[-3:]))
        earlier = float(np.nanmean(rates[-9:-3]))
        if np.isfinite(earlier) and earlier > 0 and np.isfinite(recent) and recent < earlier:
            trend_drop = (earlier - recent) / earlier * 100.0
            estimated = float(np.clip(max(estimated, trend_drop), 10.0, 70.0))
            note = (
                f"Forecasted ~{estimated:.0f}% from recent claim-rate decline "
                f"(+ FCO/K share)."
            )
            source = "forecasted_trend"

    return {
        "reduction_pct": float(np.clip(estimated, 5.0, 80.0)),
        "source": source,
        "note": note or f"Forecasted default reduction ~{estimated:.0f}%.",
    }


def compute_cpv_claim_ratio(
    claims: np.ndarray,
    production: np.ndarray,
    costs: np.ndarray | None = None,
) -> dict:
    """
    claim_ratio = claims / production
    CPV = production_cost / production  (Cost Per Vehicle)
    """
    claims = np.asarray(claims, dtype=float).ravel()
    production = np.asarray(production, dtype=float).ravel()
    n = min(len(claims), len(production))
    claims, production = claims[:n], production[:n]
    prod_safe = np.where(production > 0, production, np.nan)
    claim_ratio = claims / prod_safe
    claim_ratio_per_1k = claim_ratio * 1000.0

    if costs is None:
        costs = np.full(n, np.nan)
    else:
        costs = np.asarray(costs, dtype=float).ravel()[:n]
    cpv = costs / prod_safe

    return {
        "claim_ratio": claim_ratio,
        "claim_ratio_per_1k": claim_ratio_per_1k,
        "cpv": cpv,
        "production_cost": costs,
        "avg_claim_ratio": float(np.nanmean(claim_ratio)),
        "avg_cpv": float(np.nanmean(cpv)) if np.any(np.isfinite(cpv)) else float("nan"),
        "total_production": float(np.nansum(production)),
        "total_cost": float(np.nansum(costs)) if np.any(np.isfinite(costs)) else float("nan"),
        "total_claims": float(np.nansum(claims)),
    }


def peak_fcok_month(raw: pd.DataFrame, part: str, future_periods: list | None = None):
    """
    FCO/K manufacturing month with the highest claim count for *part*.

    When *future_periods* is provided, prefer the peak among FCO/K months
    that still fall inside the 3-year warranty window for the forecast horizon
    (so the countermeasure can actually affect future claims).
    """
    sub = raw[raw["Part Name"] == part]
    if sub.empty or "FCOK_MONTH" not in sub.columns:
        return None, 0
    counts = sub.groupby("FCOK_MONTH").size().sort_values(ascending=False)
    if counts.empty:
        return None, 0

    if future_periods:
        last_fp = future_periods[-1]
        if not isinstance(last_fp, pd.Period):
            last_fp = pd.Period(last_fp, freq="M")
        eligible = []
        for fm, n in counts.items():
            try:
                age = (last_fp - pd.Period(fm, freq="M")).n
            except Exception:
                continue
            if 0 <= age <= WARRANTY_MONTHS:
                eligible.append((fm, int(n)))
        if eligible:
            eligible.sort(key=lambda x: x[1], reverse=True)
            return eligible[0][0], eligible[0][1]

    return counts.index[0], int(counts.iloc[0])


def list_fcok_months(raw: pd.DataFrame, part: str) -> list[str]:
    sub = raw[raw["Part Name"] == part]
    if sub.empty or "FCOK_MONTH" not in sub.columns:
        return []
    return sorted({str(p) for p in sub["FCOK_MONTH"].dropna().unique()})


def parse_cm_period(cm_month_or_date: str | None) -> pd.Period | None:
    """Parse a countermeasure date / month string to a monthly Period."""
    if cm_month_or_date is None:
        return None
    s = str(cm_month_or_date).strip()
    if not s:
        return None
    for parser in (
        lambda x: pd.Period(pd.to_datetime(x), freq="M"),
        lambda x: pd.Period(x, freq="M"),
    ):
        try:
            return parser(s)
        except Exception:
            continue
    return None


def apply_cm_to_forecast(
    baseline_forecast: np.ndarray,
    raw: pd.DataFrame,
    part: str,
    *,
    cm_enabled: bool,
    cm_month: str | None,
    reduction_pct: float,
    future_periods: list,
    future_production: np.ndarray | None = None,
    hist_production: np.ndarray | None = None,
    hist_claims: np.ndarray | None = None,
    hist_periods=None,
    use_peak_fcok: bool = True,
    sensitivity: float = 1.0,
    prod_adjustment_mode: str = "share",
    engine_factor: float = 1.0,
    cost_per_claim: float | None = None,
    cm_registry: dict | None = None,
) -> dict:
    """
    Reduce future claims after a countermeasure date, weighted by production.

    Two modes are supported via *prod_adjustment_mode*:

    ``'share'`` (default / legacy)
        Reduce claims by ``reduction_pct`` scaled by FCO/K share and
        relative future production weights. Original behaviour preserved.

    ``'offset'`` (new — production-adjusted baseline engine)
        Delegates to :mod:`forecasting.countermeasure_engine` which:
        • Identifies peak FCOK months by claim count.
        • Computes avg_prod of those months and derives adj_prod =
          avg_prod − avg_peak_claims.
        • Builds a declining future-production sequence within the
          3-year warranty window.
        • Uses a production-weighted claim-rate model (factor=engine_factor).
        • Returns a full comparison dict with savings estimate.

    After the CM implementation month, forecast claims are cut by
    ``reduction_pct``, scaled by relative future production so high-volume
    months show a clearer absolute drop. Peak FCO/K share is blended in so
    manufacturing-batch targeting still participates.
    """
    from forecasting.data.loader import estimate_fcok_share_in_future

    baseline = np.asarray(baseline_forecast, dtype=float).ravel()
    H = len(baseline)
    empty = {
        "original": baseline,
        "adjusted": baseline.copy(),
        "monthly_reduction": np.zeros(H, dtype=float),
        "cumulative_reduction": np.zeros(H, dtype=float),
        "improvement_pct": 0.0,
        "fcok_share": np.zeros(H, dtype=float),
        "reduction_pct": float(reduction_pct or 0.0),
        "selected_fcok": [],
        "cm_enabled": False,
        "cm_month": cm_month,
        "peak_fcok": None,
        "warranty_months": WARRANTY_MONTHS,
        "post_cm_mask": np.zeros(H, dtype=float),
        "prod_weight": np.ones(H, dtype=float),
        "engine_result": None,
    }
    if not cm_enabled:
        return empty

    # ── New offset mode: delegate to countermeasure_engine ───────────────
    if prod_adjustment_mode == "offset":
        from forecasting.countermeasure_engine import run_cm_analysis

        prod_series = (
            hist_production
            if hist_production is not None and len(hist_production)
            else np.ones(1)
        )
        claims_series = (
            hist_claims
            if hist_claims is not None and len(hist_claims)
            else np.zeros(len(prod_series))
        )
        periods = hist_periods if hist_periods is not None else pd.period_range(
            end=pd.Period.now("M"), periods=len(prod_series), freq="M"
        )

        eng = run_cm_analysis(
            raw=raw,
            part=part,
            baseline_forecast=baseline,
            future_periods=future_periods,
            production_series=prod_series,
            monthly_periods=periods,
            hist_claims=claims_series,
            cm_enabled=cm_enabled,
            cm_registry=cm_registry,
            cm_date=cm_month,
            cost_per_claim=cost_per_claim,
        )

        comp = eng["comparison"]
        cm_adj = eng["cm_forecast"]
        monthly_red = comp["monthly_reduction"]
        return {
            "original": baseline,
            "adjusted": cm_adj,
            "monthly_reduction": monthly_red,
            "cumulative_reduction": comp["cumulative_reduction"],
            "improvement_pct": comp["total_pct_reduction"],
            "fcok_share": np.zeros(H, dtype=float),
            "reduction_pct": comp["total_pct_reduction"],
            "selected_fcok": [],
            "cm_enabled": True,
            "cm_month": cm_month,
            "peak_fcok": (
                str(eng["peak_fcok_df"]["fcok_month"].iloc[0])
                if not eng["peak_fcok_df"].empty else None
            ),
            "warranty_months": WARRANTY_MONTHS,
            "post_cm_mask": np.ones(H, dtype=float),
            "prod_weight": eng["cm_production"][:H] / max(
                float(np.nanmean(eng["cm_production"])), 1.0
            ),
            "engine_result": eng,
            "cost_savings": comp["cost_savings"],
            "total_cost_savings": comp["total_cost_savings"],
            "avg_peak_prod": eng["avg_peak_prod"],
            "avg_peak_claims": eng["avg_peak_claims"],
            "adj_prod": eng["adj_prod"],
            "cm_production": eng["cm_production"],
            "peak_fcok_df": eng["peak_fcok_df"],
            "engine_message": eng["message"],
        }

    red = max(0.0, min(100.0, float(reduction_pct))) / 100.0
    sens = max(0.5, float(sensitivity))

    # Future production weights (relative to mean) → clearer absolute reductions
    if future_production is not None and len(np.asarray(future_production).ravel()) >= H:
        prod = np.asarray(future_production, dtype=float).ravel()[:H]
    elif hist_production is not None and len(hist_production):
        last = float(np.nanmean(np.asarray(hist_production, dtype=float)[-3:]))
        if not np.isfinite(last) or last <= 0:
            last = float(np.nanmean(hist_production)) if np.any(np.isfinite(hist_production)) else 1.0
        prod = np.full(H, max(last, 1.0))
    else:
        prod = np.ones(H)
    mean_p = float(np.nanmean(prod)) if np.any(np.isfinite(prod)) else 1.0
    mean_p = max(mean_p, 1.0)
    prod_w = np.clip(np.nan_to_num(prod, nan=mean_p) / mean_p, 0.85, 1.75)

    cm_period = parse_cm_period(cm_month)
    post_mask = np.ones(H, dtype=float)
    if cm_period is not None:
        post_mask = np.array(
            [
                1.0 if (pd.Period(fp, freq="M") if not isinstance(fp, pd.Period) else fp) >= cm_period
                else 0.0
                for fp in future_periods[:H]
            ],
            dtype=float,
        )
        # If every month is before CM date, still apply from month 0 so UI reacts
        if float(post_mask.sum()) < 1e-9:
            post_mask = np.ones(H, dtype=float)

    peak, peak_n = peak_fcok_month(raw, part, future_periods=future_periods)
    selected: list = []
    if use_peak_fcok and peak is not None:
        selected = [peak]
    elif cm_month:
        selected = [cm_month]

    if selected:
        shares = estimate_fcok_share_in_future(
            raw, part, selected, future_periods[:H]
        )
        if len(shares) < H:
            shares = np.pad(shares, (0, H - len(shares)))
        shares = shares[:H]
    else:
        shares = np.zeros(H, dtype=float)

    # Visible floor: after CM date, at least 60% of the stated reduction applies,
    # blended upward with FCO/K share so manufacturing targeting still matters.
    intensity = np.maximum(shares, 0.60) * post_mask * prod_w * sens
    intensity = np.clip(intensity, 0.0, 1.0)

    monthly_reduction = baseline * red * intensity
    adjusted = np.clip(baseline - monthly_reduction, 0.0, None)
    improvement = float(monthly_reduction.sum() / (float(baseline.sum()) + 1e-9) * 100.0)

    return {
        "original": baseline,
        "adjusted": adjusted,
        "monthly_reduction": monthly_reduction,
        "cumulative_reduction": np.cumsum(monthly_reduction),
        "improvement_pct": improvement,
        "fcok_share": shares,
        "reduction_pct": float(reduction_pct),
        "selected_fcok": [str(m) for m in selected],
        "cm_enabled": True,
        "cm_month": cm_month,
        "peak_fcok": str(peak) if peak is not None else None,
        "peak_fcok_claims": peak_n,
        "warranty_months": WARRANTY_MONTHS,
        "post_cm_mask": post_mask,
        "prod_weight": prod_w,
        "floor_applied": True,
    }


def forecast_economics(
    forecast_claims: np.ndarray,
    hist_econ: dict,
    future_production: np.ndarray | None = None,
) -> dict:
    """
    Project claim ratio and cost metrics onto the forecast horizon.

    Future production defaults to the last historical production level
    (mean of last 3 months when available).
    """
    fc = np.asarray(forecast_claims, dtype=float).ravel()
    H = len(fc)
    hist_prod_total = hist_econ.get("total_production", 0.0) or 0.0
    hist_claims = hist_econ.get("total_claims", 0.0) or 0.0
    avg_ratio = hist_econ.get("avg_claim_ratio", np.nan)
    avg_cpv = hist_econ.get("avg_cpv", np.nan)

    if future_production is None or len(future_production) < H:
        # Use average monthly production from history
        n_hist = max(1, int(round(hist_claims / (avg_ratio + 1e-12)))) if np.isfinite(avg_ratio) and avg_ratio > 0 else 1
        # Prefer total_production / n months from claim_ratio length
        cr = hist_econ.get("claim_ratio", np.array([]))
        n_mo = max(len(cr), 1)
        avg_prod = hist_prod_total / n_mo if hist_prod_total > 0 else np.nan
        fut_prod = np.full(H, avg_prod, dtype=float)
    else:
        fut_prod = np.asarray(future_production, dtype=float).ravel()[:H]

    fut_ratio = fc / np.where(fut_prod > 0, fut_prod, np.nan)
    # Expected warranty cost proxy: forecast claims × (avg CPV is production cost;
    # claim-linked cost uses avg cost per claim if we have total_cost & claims)
    total_cost = hist_econ.get("total_cost", np.nan)
    cost_per_claim = (
        float(total_cost) / hist_claims
        if np.isfinite(total_cost) and hist_claims > 0
        else float("nan")
    )
    forecast_claim_cost = fc * cost_per_claim if np.isfinite(cost_per_claim) else np.full(H, np.nan)

    return {
        "future_production": fut_prod,
        "forecast_claim_ratio": fut_ratio,
        "forecast_claim_ratio_per_1k": fut_ratio * 1000.0,
        "avg_hist_cpv": avg_cpv,
        "cost_per_claim": cost_per_claim,
        "forecast_claim_cost": forecast_claim_cost,
        "horizon": H,
    }


# ---------------------------------------------------------------------------
# Production-weighted CM reduction helper (standalone)
# ---------------------------------------------------------------------------

def production_weighted_cm_reduction(
    baseline_forecast: np.ndarray,
    future_production: np.ndarray,
    hist_claims: np.ndarray,
    hist_production: np.ndarray,
    *,
    warranty_months: int = WARRANTY_MONTHS,
) -> dict:
    """
    Apply a production-weighted claim reduction to the baseline forecast.

    Claims drop in proportion to the adjusted production exposure —
    no factor is applied, because the actual reduction magnitude is unknown.
    The production change alone drives the post-CM claim projection.

    Parameters
    ----------
    baseline_forecast : np.ndarray
        Original monthly claim forecast.
    future_production : np.ndarray
        Post-CM adjusted future production volumes.
    hist_claims : np.ndarray
        Historical monthly claim counts.
    hist_production : np.ndarray
        Historical monthly production volumes.
    warranty_months : int
        Warranty window (for length alignment).

    Returns
    -------
    dict
        ``adjusted`` (np.ndarray), ``monthly_reduction``, ``improvement_pct``,
        ``hist_rate``, ``prod_weights``.
    """
    from forecasting.countermeasure_engine import compute_cm_forecast

    adjusted = compute_cm_forecast(
        baseline_forecast=baseline_forecast,
        cm_adjusted_production=future_production,
        hist_production=hist_production,
        hist_claims=hist_claims,
        warranty_months=warranty_months,
    )
    baseline = np.asarray(baseline_forecast, dtype=float).ravel()
    H = min(len(baseline), len(adjusted))
    monthly_red = np.clip(baseline[:H] - adjusted[:H], 0.0, None)
    total = float(baseline[:H].sum()) + 1e-9

    # Historical rate
    hp = np.asarray(hist_production, dtype=float).ravel()
    hc = np.asarray(hist_claims, dtype=float).ravel()
    valid = np.isfinite(hp) & (hp > 0) & np.isfinite(hc)
    hist_rate = float(np.nanmean(hc[valid] / hp[valid])) if valid.any() else 0.0

    fp = np.asarray(future_production, dtype=float).ravel()[:H]
    mean_fp = float(np.nanmean(fp)) if np.any(np.isfinite(fp)) else 1.0
    prod_weights = np.clip(fp / max(mean_fp, 1.0), 0.5, 2.0)

    return {
        "adjusted": adjusted[:H],
        "monthly_reduction": monthly_red,
        "improvement_pct": float(monthly_red.sum() / total * 100.0),
        "hist_rate": hist_rate,
        "prod_weights": prod_weights,
    }
