# Countermeasure Logic & Model Architecture Documentation

> **Warranty Claim Forecasting System**
> Notebooks: `transformer_projection.ipynb` · `cnn_lstm_projection.ipynb` · `nbeatsx_projection.ipynb`

---

## 1. Countermeasure Logic Overview

A **countermeasure (CM)** is a corrective engineering action that eliminates the root cause of a warranty defect in new production. This system models the downstream effect on future warranty claims.

### Core Principle

After a CM is applied:
- **New production** (vehicles built after CM date) has **zero defects** → zero future claims
- **Pre-CM vehicles** already on the road continue to claim **until their 3-year warranty expires**
- Each month, more pre-CM vehicles age out of warranty → the claim pool **shrinks monotonically**

---

## 2. Step-by-Step Countermeasure Logic

### Step 1 — Identify Peak FCOK Months (90% Threshold)

FCOK (First Claim OK) = month the vehicle was produced/sold, linked to when its claims occur.

`
Sort FCOK months by claim_count (descending)
Cumulative sum of claims
Select fewest months where cumulative_sum >= 0.90 × total_claims
`

**Example:**

| FCOK Month | Claims | Cumulative | % of Total |
|---|---|---|---|
| 2022-03 | 450 | 450 | 45% |
| 2022-01 | 320 | 770 | 77% |
| 2022-06 | 140 | 910 | **91%** threshold hit |

→ Months 2022-03, 2022-01, 2022-06 cover 91% of claims.

---

### Step 2 — Compute Adjusted Baseline Production

`
avg_peak_prod   = mean production of selected FCOK months
avg_peak_claims = mean monthly claims of those FCOK months
adj_prod        = avg_peak_prod - avg_peak_claims
`

**Example:** avg_peak_prod=25,000  avg_peak_claims=450  →  adj_prod=24,550

---

### Step 3 — CM-Adjusted Production Trajectory

`
P[t] = adj_prod - t * avg_peak_claims   (for t < 36)
P[t] = max(P[t], 1)                     (floor at 1)
`

| Month t | CM Production |
|---|---|
| 0 | 24,550 |
| 6 | 21,850 |
| 12 | 19,150 |
| 24 | 13,750 |

---

### Step 4 — Warranty Window Handling (36 Months)

`
remaining_warranty[t] = max(0,  1 - t / 36)
cm_fc_physics[t]      = avg_peak_claims * remaining_warranty[t]
`

| Month | Fraction | CM Physics Forecast |
|---|---|---|
| t=0  | 1.000 | 450.0 |
| t=12 | 0.667 | 300.0 |
| t=18 | 0.500 | 225.0 |
| t=36 | 0.000 | **0.0** |

---

### Step 5 — Final CM Forecast Blend

`python
cm_forecast[t] = 0.70 * cm_fc_physics[t]
               + 0.30 * min(cm_model_output[t], baseline_forecast[t])
cm_forecast[t] = clip(cm_forecast[t], 0, baseline_forecast[t])   # hard cap
`

- **70% physics** — guarantees visible, monotonic reduction
- **30% model** — allows learned pattern to refine shape within physics envelope
- **Hard clip** — mathematical guarantee CM_Claims ≤ Baseline_Claims every month

---

## 3. Model Architectures

### 3.1 Transformer (transformer_projection.ipynb)

`
Input (W, nf)
→ Linear projection → (W, d_model=16)
→ Sinusoidal Positional Encoding
→ Multi-Head Self-Attention (n_heads=2) × 2
→ Feed-Forward (d_model→32→d_model) × 2
→ Mean pool over W → (d_model,)
→ Dense → (H,)
`

Exogenous features (Production_Vol, Vehicle_Age, Claim_Weight) are part of the input tensor. Self-attention learns cross-feature relationships across all time steps simultaneously.

---

### 3.2 CNN-LSTM (cnn_lstm_projection.ipynb)

`
Input (W, nf)
→ Conv1D(32 filters, k=3, same-pad) → ReLU → (W, 32)
→ Conv1D(64 filters, k=3, same-pad) → ReLU → (W, 64)
→ LSTM(64 units) processes (W, 64) → last hidden state (64,)
→ Dense(64 → H)
`

LSTM gates: Forget / Input / Output / Cell (standard gated equations).
Same-padding ensures output length = W regardless of kernel size.

Exogenous features enter through Conv1D channels. The convolutions extract local 3-month patterns; LSTM retains long-range sequential memory.

---

### 3.3 N-BeatsX (nbeatsx_projection.ipynb)

`
Input per block: [claims_window (W,)] ++ [exog_last_step (F,)]  →  (W+F,)

For each block k in {0..n_blocks-1}:
  h = ReLU(FC1) → ReLU(FC2) → ReLU(FC3) → ReLU(FC4)   (each 64 units)
  backcast[k] = h @ Wb[k]    shape (W,)    ← "what I explain"
  forecast[k] = h @ Wf[k]    shape (H,)    ← "what I predict"
  residual_{k+1} = residual_k - backcast[k]   ← residual learning
  forecast_acc   += forecast[k]               ← additive stacking

Output: forecast_acc  (H,)
`

Exogenous features (last time step) are concatenated to the claims window at the input of every block, giving each decomposition stage full access to production context.

---

## 4. Shared Feature Engineering

| Feature | Type | Description |
|---|---|---|
| Lag_1, Lag_2, Lag_3 | Lag | Claims 1/2/3 months ago |
| Lag_12 | Lag | Claims 12 months ago (same season) |
| Month, Quarter, Year | Calendar | Temporal calendar features |
| Trend_Index | Index | Linear time index |
| Seasonality_Index | Index | Monthly avg / overall avg |
| Production_Vol | **Exogenous** | Monthly production units |
| Vehicle_Age | **Exogenous** | Age capped at WARRANTY_MONTHS=36 |
| Claim_Weight | **Exogenous** | claims[t] / max(claims) |

Rolling features removed (data leakage + smoothing of warranty spikes).

---

## 5. Workflow Breakdown

`
Cell 1  — Imports + model class definitions
Cell 2  — User Input (CSV paths, part, CM date, horizon)
Cell 3  — Data Loading & Preprocessing
Cell 4  — Feature Engineering  →  X (N,W,nf), Y (N,H)
Cell 5  — Baseline Model (Model 1)  →  baseline_forecast
Cell 6  — CM Model (Model 2)
          ├─ FCOK peak months (90% threshold)
          ├─ adj_prod = avg_peak_prod - avg_peak_claims
          ├─ CM feature matrix (replaced Production_Vol + amplified gradient)
          ├─ Oversample low-production windows 3x
          ├─ Train separate model on CM features
          ├─ Warranty-expiry physics: avg_claims * max(0, 1-t/36)
          └─ Blend: 70% physics + 30% min(model, baseline)  → hard clip
Cell 7  — 4-Panel Visualisation (history + forecast + bars + cumulative)
Cell 7b — Production Trajectory Chart
Cell 8  — Numerical Results (KPI cards + comparison table + CSV)
Cell 9  — PowerPoint Export (5-slide deck)
Cell 10 — Edge Case Testing (smoke tests)
`

---

## 6. Comparison Methodology

| Metric | Formula |
|---|---|
| Monthly Reduction | max(0, baseline[t] - cm[t]) |
| Cumulative Reduction | Σ monthly_reduction[0..t] |
| Reduction % | monthly_reduction[t] / baseline[t] × 100 |
| Cost Savings | monthly_reduction[t] × cost_per_claim |

### Why Models Differ

| Model | Temporal Mechanism | Best For |
|---|---|---|
| **Transformer** | Global self-attention | Long-range seasonal patterns |
| **CNN-LSTM** | Local conv + sequential memory | Local spikes + long memory |
| **N-BeatsX** | Residual decomposition | Trend/seasonality separation |

All three share the **same 70% physics / 30% model blend** guarantee — CM claims are always ≤ baseline and converge to zero at 36 months.

---

## 7. Key Parameters

| Parameter | Default | Description |
|---|---|---|
| WARRANTY_MONTHS | 36 | Warranty window (months) |
| LOOKBACK | 12 | Historical window for models |
| HORIZON | 12 | Forecast horizon |
| MODEL_EPOCHS | 60 | Max training epochs |
| PHYSICS_WEIGHT | 0.70 | Physics weight in CM blend |
| pct_threshold | 0.90 | FCOK cumulative % threshold |
| rep_factor | 3× | Low-production window oversampling |
| N_BLOCKS (N-BeatsX) | 3 | Residual blocks |
| n_filters (CNN-LSTM) | 32/64 | Conv1D filter counts |
| lstm_units (CNN-LSTM) | 64 | LSTM hidden units |
| hidden_units (N-BeatsX) | 64 | FC layer width per block |
