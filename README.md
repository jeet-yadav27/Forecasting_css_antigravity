# Automotive Warranty Claims Forecasting System

End-to-end multi-model warranty claims forecasting with walk-forward validation,
automatic best-model selection, manufacturing-month countermeasure simulation,
and an interactive **Gradio** dashboard.

## Models

| Model | Type |
|---|---|
| CNN-LSTM | Pure-NumPy deep learning |
| N-BEATS | Pure-NumPy deep learning |
| Transformer | Pure-NumPy deep learning |
| SARIMA | Classical time-series baseline |
| Ensemble | Inverse-MAE weighted combination (retained) |

Tree-based ML models (XGBoost, LightGBM, Gradient Boosting) are **not used**.
The pipeline ranks CNN-LSTM / N-BEATS / Transformer / SARIMA by
**RMSE / MAE / MAPE / R²** and auto-selects the best model for the final
12-month forecast.

---

## Project Structure

```
Forecastingv2cursor/
├── data/                        # Raw CSV claim files
├── outputs/
│   ├── models/                  # Pickled trained model bundles
│   └── forecasts/               # Forecast / metrics CSV + JSON
├── forecasting/
│   ├── config.py                # Tuneable constants + random seed
│   ├── metrics.py               # RMSE/MAE/MAPE/R² + ranking
│   ├── data/loader.py           # Load, validate, feature engineer, CM sim
│   ├── models/                  # CNN-LSTM, N-BEATS, Transformer, ML, SARIMA
│   ├── pipeline/runner.py       # Walk-forward CV + training + persistence
│   └── dashboard/builder.py     # Plotly figures + Gradio UI
├── main.py
├── requirements.txt
└── README.md
```

---

## Quick Start

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

Dashboard: **http://localhost:7860**

---

## Feature Engineering

Claims are aggregated by **Process Date** (claim month). Features include:

- **FCO/K manufacturing month** batch intensity + warranty-share (3-year lifecycle)
- **Odometer**: average, median, max by claim month
- **Vehicle Age** = Process Date − FCO/K Month
- **Lags**: `Claim_Lag_1`, `Claim_Lag_2`, `Claim_Lag_3`, `Claim_Lag_12`
- **Calendar**: Month, Year, Quarter
- **Production**: synthetic series starting at 25,000 units, **+2% per month**
  (or supply your own production array)
- Existing Fourier seasonality and age features are retained

---

## Walk-Forward Validation

1. Reserve the **last 6 months** as the rolling test window  
2. Train on remaining history → forecast next month  
3. Append actual → retrain → forecast next month  
4. Repeat for all 6 test months  
5. Tune hyperparameters on **training data only**  
6. Rank models; select best; refit on **full history**; forecast **12 months**

---

## Countermeasure Simulation

In the **Countermeasure** tab:

1. Choose a part  
2. Select one or more **FCO/K manufacturing months**  
3. Set a reduction % (e.g. 10 / 20 / 30)  
4. Reduction is applied across the **3-year warranty** window to the share of
   future claims attributable to those manufacturing batches  

Outputs: Original Forecast, Adjusted Forecast, Monthly Reduction,
Cumulative Reduction, Improvement %

---

## Dashboard Tabs

| Tab | Contents |
|---|---|
| Summary | KPI cards, donut, forecast bar, weights heatmap, summary table |
| Model Ranking | Global leaderboard + per-part RMSE/MAE/MAPE/R² ranking |
| Overview | Claims, production, age, odometer, FCO/K heatmap |
| Part N | Forecast panels, actual vs predicted, CV folds, HP table |
| Countermeasure | Interactive FCO/K manufacturing-month simulation |
| Upload Data | CSV / Excel upload + validation |
| Export | CSV + Excel workbook (forecasts, metrics, ranking) |

---

## Configuration

Edit [`forecasting/config.py`](forecasting/config.py):

| Parameter | Default | Description |
|---|---|---|
| `PRODUCTION_PER_MONTH` | 25,000 | Starting synthetic production |
| `PRODUCTION_GROWTH_RATE` | 0.02 | +2% / month |
| `FORECAST_HORIZON` | 12 | Forecast months |
| `LOOKBACK_WINDOW` | 12 | Model lookback |
| `N_CV_FOLDS` | 6 | Walk-forward test months |
| `HP_TUNE` | True | Grid search on training data |
| `WARRANTY_MONTHS` | 36 | 3-year warranty |
| `RANDOM_SEED` | 42 | Reproducibility |
| `COUNTERMEASURES` | `{}` | Static config-based CM (still supported) |

---

## CLI

```powershell
python main.py
python main.py --horizon 6
python main.py --parts "Part 1" "Part 3"
python main.py --port 8080 --no-browser
python main.py --share
```
