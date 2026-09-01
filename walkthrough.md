# Countermeasure (CM) Logic — Documentation

## Overview

This document describes the **production-adjusted countermeasure engine** built into the automotive warranty claim forecasting system.

The engine answers one question:
> *"If a countermeasure was applied to a part, how will future warranty claims change — given that claims are driven by production exposure?"*

The answer is derived **entirely from the change in production exposure**. No effectiveness factor or blending coefficient is used, because the actual magnitude of reduction caused by the countermeasure is unknown.

---

## Files

| File | Role |
|---|---|
| [`forecasting/countermeasure_engine.py`](file:///d:/forecasting_cursor_copy2/forecasting/countermeasure_engine.py) | Core CM logic — stateless, standalone |
| [`forecasting/economics.py`](file:///d:/forecasting_cursor_copy2/forecasting/economics.py) | `apply_cm_to_forecast()` + `production_weighted_cm_reduction()` |
| [`forecasting/pipeline/runner.py`](file:///d:/forecasting_cursor_copy2/forecasting/pipeline/runner.py) | Calls engine inside `train_uploaded_part()` |
| [`forecasting/dashboard/builder.py`](file:///d:/forecasting_cursor_copy2/forecasting/dashboard/builder.py) | 3 chart builders for the CM Analysis panel |
| [`forecasting/dashboard/app_ui.py`](file:///d:/forecasting_cursor_copy2/forecasting/dashboard/app_ui.py) | "🔧 CM Engine Analysis" accordion in the Countermeasure tab |

---

## Logic Flow (Step by Step)

```
User enables CM → enters CM date → clicks Train & Forecast
        │
        ▼
Step 1 — check_countermeasure_exists()
  ├─ No CM found  →  return baseline forecast unchanged
  └─ CM active    →  continue ─────────────────────────┐
                                                        │
        ┌───────────────────────────────────────────────┘
        │
        ▼
Step 2 — identify_peak_fcok_months(raw, part, top_n=3)
   Ranks all FCOK manufacturing months by total warranty claim count.
   The top 3 months represent the highest-exposure production batches.

        ▼
Step 3 — compute_adjusted_baseline_production()
   avg_peak_prod   = mean production of the top-3 FCOK months
   avg_peak_claims = mean claims   of the top-3 FCOK months
   adj_prod        = avg_peak_prod − avg_peak_claims

   Example:
     avg_peak_prod   = 25,000 units
     avg_peak_claims =    450 claims
     adj_prod        = 24,550 units   ← adjusted production starting point

        ▼
Step 4 — build_cm_adjusted_production(adj_prod, avg_peak_claims, n_months)
   Builds a declining monthly production sequence post-CM:

     Month 0  → adj_prod                           = 24,550
     Month 1  → adj_prod − 1 × avg_peak_claims     = 24,100
     Month 2  → adj_prod − 2 × avg_peak_claims     = 23,650
     Month t  → adj_prod − t × avg_peak_claims      (clipped at 1)
     ...
     Month 36 → production resets (warranty window expires, new batches unaffected)

   The sequence is capped at 1 unit and resets after WARRANTY_MONTHS = 36.

        ▼
Step 5 — compute_cm_forecast(baseline, cm_adjusted_production, hist_prod, hist_claims)
   Computes the post-CM claim forecast using the historical claim rate:

     hist_rate  =  mean( hist_claims[t] / hist_production[t] )
     cm_fc[t]   =  hist_rate × cm_adjusted_production[t]

   ┌─────────────────────────────────────────────────────────────┐
   │  NO FACTOR.  NO BLENDING WITH BASELINE.                     │
   │                                                             │
   │  The post-CM forecast is driven ONLY by the change in       │
   │  production exposure. Any factor or blend weight would      │
   │  introduce an unquantified assumption about how much the    │
   │  countermeasure itself reduces claims — which is unknown.   │
   └─────────────────────────────────────────────────────────────┘

        ▼
Step 6 — compare_forecasts(baseline, cm_fc)
   monthly_reduction    = baseline[t] − cm_fc[t]
   pct_reduction        = monthly_reduction[t] / baseline[t] × 100
   cumulative_reduction = cumsum(monthly_reduction)
   cost_savings         = monthly_reduction × cost_per_claim   (if available)
```

---

## Why No Factor?

The previous implementation used:
```
cm_fc[t] = hist_rate × cm_prod[t] × factor
final[t]  = α[t] × cm_fc[t] + (1−α[t]) × baseline[t]
```
where `factor ∈ [0.75, 1.0]` and `α` ramped from 0.65 → 0.90.

Both were removed because:

| Removed element | Reason |
|---|---|
| `factor` | We do not know how much of the claim reduction is caused by the CM versus natural exposure decline. Applying any factor would be an unvalidated assumption. |
| Alpha blend (`α × cm + (1-α) × baseline`) | Blending with the baseline reintroduces claims that the production model already excluded. It effectively "softens" the exposure signal without justification. |

**Current formula (pure exposure model):**
```
hist_rate = mean( hist_claims / hist_production )
cm_fc[t]  = hist_rate × cm_adjusted_production[t]
```
The forecast is determined solely by: *how much has production exposure changed?*

---

## Warranty Window (3-Year Rule)

Claims are considered over a **3-year (36-month) warranty window** per the specification.

- The declining production sequence runs for 36 months maximum.
- After month 36, production resets to `adj_prod` (new manufacturing batches after the CM window are not affected by the old failure mode).
- `WARRANTY_MONTHS = 36` is configured in [`forecasting/config.py`](file:///d:/forecasting_cursor_copy2/forecasting/config.py).

---

## Numerical Example

| Parameter | Value |
|---|---|
| avg_peak_prod | 25,000 units |
| avg_peak_claims | 450 claims |
| adj_prod | 24,550 units |
| hist_rate | 450 / 25,000 = **0.018 claims/unit** |

| Month | CM Production | CM Forecast | Baseline | Reduction |
|---|---|---|---|---|
| 1 | 24,550 | 441.9 | 450 | 8.1 |
| 2 | 24,100 | 433.8 | 450 | 16.2 |
| 3 | 23,650 | 425.7 | 450 | 24.3 |
| 6 | 22,300 | 401.4 | 450 | 48.6 |
| 12 | 19,600 | 352.8 | 450 | 97.2 |
| 36 | 8,950 | 161.1 | 450 | 288.9 |

The reduction grows each month because production exposure continues to decrease by `avg_peak_claims` units per month, propagating the CM benefit across the full warranty window.

---

## Dashboard Usage

1. Upload **claims CSV** (+ optionally a **production CSV**) in **Tab 1**
2. Expand the **"Countermeasure (optional)"** accordion
   - Check **"Countermeasure taken?"**
   - Enter the CM implementation date (e.g. `2024-03`)
3. Click **Train & Forecast**
4. Navigate to **Tab 4 — Countermeasure**
5. Expand **"🔧 CM Engine Analysis (Production-Adjusted Baseline)"**
6. Select the trained part → click **🔄 Refresh CM Engine Analysis**

### What the Panel Shows

| Element | Description |
|---|---|
| **Savings summary card** | Baseline vs CM total claims, % reduction, estimated cost savings, top peak FCOK months table |
| **4-panel chart** | With/Without CM forecast lines, Monthly Reduction bars (with % labels), Cumulative Reduction area, CM-Adjusted Production trajectory |
| **Production trajectory chart** | Standalone view of the declining production sequence with `avg_peak_prod` and `adj_prod` reference lines |
| **Month-by-month comparison table** | Downloadable as CSV |

---

## API Reference

### `countermeasure_engine.py`

```python
check_countermeasure_exists(part, cm_registry, cm_enabled_flag) -> bool
```
Returns `True` if a CM is registered for the part **or** the UI flag is enabled.

```python
identify_peak_fcok_months(raw, part, top_n=3) -> pd.DataFrame
```
Returns the top-N FCOK manufacturing months ranked by claim count.

```python
compute_adjusted_baseline_production(raw, part, production_series, monthly_periods, top_n=3) -> dict
```
Returns `avg_peak_prod`, `avg_peak_claims`, `adj_prod`, `peak_fcok_df`.

```python
build_cm_adjusted_production(adj_prod_start, avg_peak_claims, n_months, warranty_months=36) -> np.ndarray
```
Returns the declining future production array.

```python
compute_cm_forecast(baseline_forecast, cm_adjusted_production, hist_production, hist_claims) -> np.ndarray
```
Pure rate model: `hist_rate × cm_adjusted_production`. No factor, no blend.

```python
compare_forecasts(baseline, cm_fc, future_periods, cost_per_claim=None) -> dict
```
Returns monthly/cumulative reduction, %, cost savings, and a `comparison_df`.

```python
run_cm_analysis(raw, part, baseline_forecast, future_periods, production_series,
                monthly_periods, hist_claims, *, cm_enabled, cm_date, ...) -> dict
```
Orchestrates all 6 steps. Returns a single result dict consumed by the dashboard.

---

## Verification

```
[OK] factor NOT in compute_cm_forecast signature
[OK] factor NOT in run_cm_analysis signature
[OK] factor NOT in production_weighted_cm_reduction signature
[OK] Pure rate model: hist_rate=0.01800  cm_fc[0]=441.90
[OK] CM forecast <= baseline for all months (no amplification)
[OK] run_cm_analysis with cm_enabled=False -> baseline returned
[OK] run_cm_analysis with cm_enabled=True  -> cm_active=True

ALL TESTS PASSED — no factor in any function
```
