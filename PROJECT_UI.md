# Project UI — Output Meaning Guide

This document explains what each conceptual area of the Gradio app shows, how data is processed, and how to interpret results.

**Run the app:** `python main.py` → http://localhost:7860

**Actual Gradio tabs ↔ this guide**

| Guide section | Gradio tab(s) |
|---|---|
| Data Input Tab | `1. Upload & Train` (Steps 1–3) |
| Forecast Tab | `1. Upload & Train` (Results) + `2. Diagnostics` |
| Cost Analysis Tab | History CPV / claim-ratio table + forecast economics in results / PPT |
| PPT Export Tab | `5. Summary & Export` |
| Model Testing Tab | `4. Annotated Walk-Forward` (+ ranking on Tab 1) |

Countermeasure simulation also lives on **`3. Countermeasure`** (interactive what-if after training).

---

## Data Input Tab

**Where:** Gradio → **1. Upload & Train** (upload, part select, production/cost sheet, CM prompts).

### What is shown
- File upload for claims CSV/Excel.
- Load status (row counts, date conversion quality).
- Part dropdown.
- Monthly **Production / Production_Cost** table (one row per claim month).
- Optional CSV **download / upload** for that sheet.
- Optional countermeasure: yes/no, FCO/K month, reduction %.
- Model multi-select (CNN-LSTM, N-BEATS, Transformer, SARIMA).

### How it is processed
1. Claims are normalized (`Part Name`, `FCOK_DATE`, `PROCESSING_DATE`, `ODOMETER`), dates parsed, ages and warranty flags derived.
2. Claims are aggregated to a contiguous monthly series by process month.
3. **User-entered production** is required (synthetic production is disabled).
4. Optional **Production_Cost** enables CPV.
5. If CM is enabled, the peak-claim FCO/K manufacturing month (within the 3-year warranty window) is used to adjust later forecasts.

### How to interpret
- Green load messages mean schema/dates are usable.
- Every month must have **positive Production** before Train.
- Fill **Production_Cost** if you need CPV; leave blank only if cost metrics are not needed.
- CM month is when the fix was implemented; remediation targets the **highest-claim FCO/K batch**, not every month equally.

---

## Forecast Tab

**Where:** Gradio → **1. Upload & Train** results + **2. Diagnostics**.

### What is shown
- Status line (best model, avg CPV, claim ratio, CM note).
- Part forecast plot (history + best model + ensemble/CI when available).
- 12-month forecast chart and table.
- Actual vs predicted (walk-forward OOS).
- Model ranking (RMSE / MAE / MAPE / R²) and hyperparameters.
- Diagnostics: historical claims, production, vehicle age, odometer, FCO/K heatmap, CV folds, model comparison.

### How it is processed
1. Walk-forward CV over the last `N_CV_FOLDS` months (default 6).
2. Selected models train without leakage (scalers fit only on history before each fold).
3. Models are ranked; best model is refit on full history for a 12-month horizon.
4. If CM is on, baseline forecast is reduced by the share attributable to the peak FCO/K month inside the warranty window.
5. Forecast rate uses **user production** (e.g. claims per 1,000 vehicles), not a synthetic constant.

### How to interpret
- Prefer **low RMSE/MAE/MAPE** and sensible R² on the ranking table.
- Walk-forward Actual vs Predicted is the honest holdout view — large gaps mean poor generalization.
- CI bands widen when models disagree or residuals are large.
- Diagnostics heatmaps show which manufacturing months drive process-month claims.

---

## Cost Analysis Tab

**Where:** Gradio → Tab 1 **History: CPV & Claim Ratio** table; also forecast economics columns in the forecast table and PPT slides.

### What is shown
- Per historical month: Claims, Production, Production_Cost, **CPV**, **Claim_Ratio**, Claim_Ratio_per_1k.
- On forecast rows (when costs were provided): future production proxy, forecast claim ratio / 1k, estimated claim cost.

### How it is processed
- **CPV** = `Production_Cost / Production` (cost per vehicle manufactured).
- **Claim ratio** = `Claims / Production` (also shown ×1000 as per-1k).
- Forecast economics project ratios onto the horizon using recent average production and historical cost-per-claim when costs exist.

### How to interpret
- Rising claim ratio with flat production ⇒ quality / warranty pressure.
- CPV is a manufacturing cost intensity metric; combine with claim ratio to discuss warranty burden relative to build volume.
- Estimated claim cost on the forecast is a **proxy** (claims × historical cost/claim), not a full actuarial reserve.

---

## PPT Export Tab

**Where:** Gradio → **5. Summary & Export** → **Download PowerPoint (.pptx)**.

### What is shown
- Downloadable briefing deck after at least one successful train (and optional annotated run).
- Also CSV downloads for forecasts and rankings; KPI / summary tables on Refresh.

### How it is processed
1. `python-pptx` builds a widescreen deck from in-memory results.
2. Slides include: title, portfolio KPIs, model ranking, per-part forecast tables, **production/CPV/claim-ratio**, **CM-adjusted forecast**, forecast economics, optional annotated walk-forward, notes.
3. Chart images embed when `kaleido` is installed; tables always export.

### How to interpret
- Use the deck for stakeholder review: model choice, 12-mo outlook, CM impact %, and cost intensity.
- If PPT fails with `No module named 'pptx'`, install into the **same** Python that runs `main.py`:
  `python -m pip install python-pptx`

---

## Model Testing Tab

**Where:** Gradio → **4. Annotated Walk-Forward** (reference-style last-N evaluation) and ranking on Tab 1.

### What is shown
- Checkbox of models (Keras-CNN-LSTM if TensorFlow is present, plus NumPy CNN-LSTM / N-BEATS / Transformer / SARIMA).
- Epochs (Keras), optional CM start date.
- Actuals vs forecasts chart.
- Results table: Actuals, Forecast, Error, Error %, Accuracy % per model.
- Downloadable annotated CSV under `outputs/Forecast/`.

### How it is processed
1. Builds monthly features: Part_Failure, Production, Warranty_Days, Countermeasure, FCOK_Jan_Aug.
2. For each of the last 6 test months: fit scaler **only** on prior history, train selected model(s), predict that month.
3. Early stopping + dropout apply on DL models when loss plateaus.

### How to interpret
- Compare **Accuracy %** / Error % across months — unstable accuracy ⇒ fragile model.
- Run **one model first** for speed; multi-model annotated runs are slow (especially NumPy DL).
- Use this tab to stress-test data feeding and model choice before relying on the 12-month Forecast Tab.

---

## Related: Countermeasure what-if (Tab 3)

After training, pick FCO/K month(s) and a reduction % to simulate manufacturing-batch remediation over the warranty life. Compare original vs adjusted forecast and cumulative reduction. This is interactive exploration; Tab 1 CM applies a similar adjustment at train time from the peak-claim FCO/K logic.

---

## Quick workflow

1. **Data Input** — upload claims → fill or CSV-upload production/cost → optional CM.
2. **Forecast** — Train & Forecast → read ranking + 12-mo chart → Diagnostics Refresh.
3. **Cost Analysis** — review CPV / claim ratio table.
4. **Model Testing** — Annotated Walk-Forward for holdout Accuracy %.
5. **PPT Export** — Summary & Export → Download PowerPoint.

For scripted / edge-case testing, see [`notebook.ipynb`](notebook.ipynb).
