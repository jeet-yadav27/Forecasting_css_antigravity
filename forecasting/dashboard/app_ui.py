"""
forecasting/dashboard/app_ui.py
--------------------------------
Upload-first Gradio UI.

Flow
----
1. ``python main.py`` opens the dashboard immediately (no training).
2. User uploads claims + production → auto-process & forecast.
3. Optional countermeasure date + reduction % adjusts post-CM claims.
4. Results, ranking, charts, countermeasure, and exports update.
"""

from __future__ import annotations

import logging
import os
import tempfile
import traceback
from datetime import datetime

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from forecasting.config import (
    FORECAST_HORIZON,
    N_CV_FOLDS,
    WARRANTY_MONTHS,
    BEST_PARAMS_PATH,
    CM_DECAY_HALF_LIFE,
    PRODUCTION_PER_MONTH,
    MODEL_NAMES,
)
from forecasting.data.loader import (
    load_and_prepare,
    validate_claims_dataframe,
    simulate_fcok_countermeasure,
)
from forecasting.dashboard.builder import (
    _CUSTOM_CSS,
    make_part_figure,
    make_production_figure,
    make_fcok_heatmap,
    make_vehicle_age_figure,
    make_odometer_figure,
    make_actual_vs_predicted_figure,
    make_cv_fold_figure,
    make_model_comparison_figure,
    make_12m_forecast_figure,
    make_countermeasure_impact_figure,
    make_baseline_vs_adjusted_figure,
    make_cm_analysis_figure,
    make_cm_production_figure,
    make_cm_savings_html,
    make_annotated_results_figure,
    make_forecast_df,
    make_best_params_df,
    make_summary_df,
    make_global_ranking_df,
    make_kpi_html,
    p2s,
)
from forecasting.pipeline.annotated_forecast import (
    HAS_TF,
    forecast_last_n_months_annotated,
)
from forecasting.dashboard.univariate_views import (
    uni_overview_df,
    uni_forecast_df,
    make_uni_trend_figure,
    make_uni_season_figure,
    make_uni_forecast_figure,
)
from forecasting.economics import (
    month_input_template,
    parse_month_inputs,
    list_fcok_months,
    peak_fcok_month,
    production_sheet_to_csv,
    load_production_sheet_csv,
    build_upload_templates,
)

logger = logging.getLogger(__name__)


def _empty_fig(msg: str = "Upload data and train a part to see results") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(size=15, color="#000000"))
    fig.update_layout(
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        height=360, margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        font=dict(color="#000000"),
    )
    return fig


def _slim_result(res: dict) -> dict:
    """Drop heavy / non-UI fields so Gradio can safely hold results in memory."""
    skip = {"raw_ref"}
    out = {}
    for k, v in res.items():
        if k in skip:
            continue
        if isinstance(v, np.ndarray):
            out[k] = v
        else:
            out[k] = v
    return out


def _safe_fig(fn, *args, fallback: str = "Chart error"):
    try:
        fig = fn(*args)
        if fig is None:
            return _empty_fig(fallback)
        fig.update_layout(
            template="plotly_white",
            font=dict(color="#000000", size=12, family="DM Sans, Inter, sans-serif"),
            title_font=dict(color="#000000"),
            legend=dict(font=dict(color="#000000")),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
        )
        fig.update_xaxes(tickfont=dict(color="#000000"), title_font=dict(color="#000000"))
        fig.update_yaxes(tickfont=dict(color="#000000"), title_font=dict(color="#000000"))
        fig.update_annotations(font=dict(color="#000000"))
        return fig
    except Exception as exc:
        logger.exception("Figure build failed: %s", exc)
        return _empty_fig(f"{fallback}: {exc}")


def build_interactive_app(
    results=None,
    raw=None,
    preload_files=None,
    parts_filter=None,
) -> gr.Blocks:
    """Build Gradio Blocks that train only after upload + part selection."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Session store (single-user local app)
    state = {
        "raw": raw,
        "results": {r["part"]: _slim_result(r) for r in (results or [])},
        "paths": list(preload_files or []),
        "annotated_df": None,
        "univariate": {},
        "uni_monthly_sheet": None,
    }

    with gr.Blocks(
        title="Automotive Warranty Forecasting",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="emerald",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("DM Sans"),
        ).set(
            body_background_fill="#E8EEF7",
            block_background_fill="#FFFFFF",
            border_color_primary="#94A3B8",
            body_text_color="#000000",
            block_label_text_color="#000000",
            button_primary_background_fill="#BFDBFE",
            button_primary_text_color="#000000",
        ),
        css=_CUSTOM_CSS,
    ) as demo:

        gr.HTML(f"""
<div class="aw-hero">
  <h1>Automotive Warranty Claims Forecasting</h1>
  <p><strong>Upload → configure → forecast</strong>
     &nbsp;·&nbsp; Optional production &amp; countermeasure
     &nbsp;·&nbsp; Horizon {FORECAST_HORIZON} mo
     &nbsp;·&nbsp; Walk-forward {N_CV_FOLDS} folds
     &nbsp;·&nbsp; {now_str}</p>
  <span class="aw-chip">Interactive Gradio</span>
  <span class="aw-chip">CM-adjusted forecast</span>
  <span class="aw-chip">PPT export</span>
  <div class="aw-steps">
    <div class="aw-step"><b>1</b> Upload claims (+ production)</div>
    <div class="aw-step"><b>2</b> Set part / models / CM</div>
    <div class="aw-step"><b>3</b> Process &amp; Forecast</div>
    <div class="aw-step"><b>4</b> Explore charts · export PPT</div>
  </div>
</div>
        """)

        with gr.Tabs(elem_id="aw-main-tabs"):

            # ── 1. Run ───────────────────────────────────────────────────
            with gr.Tab("1. Upload & Forecast"):
                with gr.Accordion("① Data upload", open=True):
                    gr.Markdown(
                        "Upload **claims** (required) and **production** (optional). "
                        "Both together → Tab 1 multivariate **plus** Tab 3 univariate analysis."
                    )
                    with gr.Row():
                        dl_tmpl_btn = gr.Button(
                            "Download Excel template",
                            variant="secondary",
                        )
                        dl_tmpl_file = gr.File(label="Template workbook")
                    tmpl_status = gr.Markdown(
                        "*Template: Claims + Production + README sheets.*"
                    )
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("#### Claims *(required)*")
                            upload_box = gr.File(
                                label="Drop claims CSV / Excel",
                                file_count="multiple",
                                file_types=[".csv", ".xlsx", ".xls"],
                                height=140,
                            )
                            gr.Markdown(
                                "*Part Name · FCOK_DATE · PROCESSING_DATE · ODOMETER*"
                            )
                        with gr.Column(scale=1):
                            gr.Markdown("#### Production *(optional)*")
                            up_prod_file = gr.File(
                                label="Drop production / cost file",
                                file_types=[".csv", ".xlsx", ".xls"],
                                file_count="single",
                                height=140,
                            )
                            gr.Markdown(
                                "*Month · Production · Production_Cost*"
                            )
                    with gr.Row():
                        load_btn = gr.Button("Load & Validate", variant="secondary")
                        auto_btn = gr.Button(
                            "Process & Forecast", variant="primary",
                        )
                    load_status = gr.Markdown("*Waiting for upload…*")

                with gr.Accordion("② Part, production sheet & models", open=True):
                    with gr.Row():
                        part_dd = gr.Dropdown(
                            choices=[], label="Part Name", interactive=True,
                        )
                        retune_ck = gr.Checkbox(
                            value=False,
                            label="Tune hyperparameters & lock best params",
                            info="Leave unchecked for faster runs.",
                        )
                    prod_hint = gr.Markdown(
                        "*Production table fills after claims load (optional).*"
                    )
                    prod_table = gr.Dataframe(
                        headers=["Month", "Production", "Production_Cost"],
                        datatype=["str", "number", "number"],
                        label="Monthly Production & Cost (optional)",
                        interactive=True,
                        wrap=True,
                        elem_id="prod-cost-table",
                    )
                    with gr.Row():
                        dl_prod_btn = gr.Button(
                            "Download production CSV", variant="secondary",
                        )
                        dl_prod_file = gr.File(label="Production CSV")
                        up_prod_btn = gr.Button(
                            "Apply production file to table", variant="secondary",
                        )
                    prod_csv_status = gr.Markdown("")
                    model_ck = gr.CheckboxGroup(
                        choices=list(MODEL_NAMES),
                        value=[],
                        label="Models for forecasting",
                        info=(
                            "Select at least one model. "
                            "ONE model → use that model directly. "
                            "MULTIPLE → compare, rank, and use the best."
                        ),
                    )

                with gr.Accordion("③ Countermeasure (optional)", open=False):
                    cm_yes = gr.Checkbox(
                        value=False,
                        label="Countermeasure taken?",
                        info="Turn on to reveal date / reduction controls.",
                    )
                    with gr.Group(visible=False) as cm_panel:
                        with gr.Row():
                            cm_date = gr.Textbox(
                                label="Countermeasure date (YYYY-MM or YYYY-MM-DD)",
                                placeholder="e.g. 2023-06",
                                value="",
                            )
                            cm_month_dd = gr.Dropdown(
                                choices=[],
                                label="Peak FCO/K reference (auto)",
                                interactive=True,
                                allow_custom_value=True,
                            )
                        with gr.Row():
                            cm_know_red = gr.Checkbox(
                                value=False,
                                label="I know the expected claim reduction %",
                                info="Unchecked → app estimates reduction from data.",
                            )
                            cm_red = gr.Slider(
                                0, 100, value=30, step=1,
                                label="Expected claim reduction %",
                                interactive=True,
                                visible=False,
                            )
                        cm_peak_md = gr.Markdown(
                            "*Enable CM, set a date, then Process & Forecast.*"
                        )

                    def _toggle_cm(on):
                        return gr.update(visible=bool(on))

                    def _toggle_red(know):
                        return gr.update(visible=bool(know))

                    cm_yes.change(fn=_toggle_cm, inputs=[cm_yes], outputs=[cm_panel])
                    cm_know_red.change(
                        fn=_toggle_red, inputs=[cm_know_red], outputs=[cm_red],
                    )

                with gr.Row():
                    train_btn = gr.Button(
                        "Train & Forecast Selected Part", variant="primary",
                    )
                train_status = gr.Markdown(
                    "*Tip: use **Process & Forecast** after uploading files, "
                    "or Train after editing the production sheet.*",
                    elem_id="train-status-md",
                )

                with gr.Accordion("④ Forecast results", open=True):
                    locked_md = gr.Markdown("—", elem_id="locked-md")
                    econ_table = gr.Dataframe(
                        label="History: CPV & Claim Ratio",
                        interactive=False, wrap=True,
                        elem_id="econ-table",
                    )
                    part_plot = gr.Plot(
                        label="Part Forecast (Historical + CM-adjusted)",
                    )
                    with gr.Row():
                        fc_plot = gr.Plot(label="12-Month Forecast (baseline vs CM)")
                        avp_plot = gr.Plot(label="Actual vs Predicted (Walk-Forward)")
                    fc_table = gr.Dataframe(
                        label="Forecast Table", interactive=False, wrap=True,
                        elem_id="fc-table",
                    )
                    with gr.Row():
                        rank_table = gr.Dataframe(
                            label="Model Ranking (RMSE / MAE / MAPE / R²)",
                            interactive=False, wrap=True,
                            elem_id="rank-table",
                        )
                        hp_table = gr.Dataframe(
                            label="Hyperparameters Used",
                            interactive=False, wrap=True,
                            elem_id="hp-table",
                        )

                def _merge_production(tmpl: pd.DataFrame, prod_file) -> tuple[pd.DataFrame, str]:
                    """Overlay uploaded production CSV onto the claims month template."""
                    if prod_file is None:
                        return tmpl, ""
                    try:
                        path = prod_file.name if hasattr(prod_file, "name") else str(prod_file)
                        uploaded = load_production_sheet_csv(path)
                        if tmpl is None or tmpl.empty:
                            return uploaded, f"✅ Production sheet loaded ({len(uploaded)} months)."
                        work = tmpl.copy()
                        up = uploaded.copy()
                        work["Month"] = work["Month"].astype(str)
                        up["Month"] = up["Month"].astype(str)
                        # Exact month match first
                        merged = work.merge(
                            up, on="Month", how="left", suffixes=("", "_up")
                        )
                        for col in ("Production", "Production_Cost"):
                            up_col = f"{col}_up"
                            if up_col in merged.columns:
                                merged[col] = merged[up_col].combine_first(merged[col])
                                merged = merged.drop(columns=[up_col])
                        # If upload has months not in template, append them
                        extra = up[~up["Month"].isin(work["Month"])]
                        if len(extra):
                            merged = pd.concat([merged, extra], ignore_index=True)
                        note = (
                            f"✅ Merged production into **{len(merged)}** months "
                            f"from uploaded CSV."
                        )
                        return merged[["Month", "Production", "Production_Cost"]], note
                    except Exception as exc:
                        return tmpl, f"⚠️ Production CSV not applied: {exc}"

                def _load_files(files, prod_file=None):
                    # New upload clears prior forecasts so other tabs stay blank
                    state["results"] = {}
                    state["annotated_df"] = None
                    state["univariate"] = {}
                    if not files:
                        return (
                            "⚠️ No claims file provided.",
                            gr.update(choices=[], value=None),
                            pd.DataFrame(columns=["Month", "Production", "Production_Cost"]),
                            "*Upload claims first…*",
                            gr.update(choices=[], value=None),
                            "",
                            "",
                        )
                    try:
                        paths = [f.name for f in files]
                        raw_df = load_and_prepare(paths)
                        ok, msgs = validate_claims_dataframe(raw_df)
                        if not ok:
                            return (
                                "❌ " + "; ".join(msgs),
                                gr.update(choices=[], value=None),
                                pd.DataFrame(columns=["Month", "Production", "Production_Cost"]),
                                "*Fix data and reload.*",
                                gr.update(choices=[], value=None),
                                "",
                                "",
                            )
                        parts = sorted(raw_df["Part Name"].dropna().unique().tolist())
                        if parts_filter:
                            parts = [p for p in parts if p in parts_filter] or parts
                        state["raw"] = raw_df
                        state["paths"] = paths
                        first = parts[0] if parts else None
                        tmpl = month_input_template(raw_df, first) if first else pd.DataFrame(
                            columns=["Month", "Production", "Production_Cost"]
                        )
                        tmpl, prod_note = _merge_production(tmpl, prod_file)
                        fcok = list_fcok_months(raw_df, first) if first else []
                        peak, npeak = peak_fcok_month(raw_df, first) if first else (None, 0)
                        peak_msg = (
                            f"Highest-claim FCO/K month for **{first}**: `{peak}` "
                            f"({npeak:,} claims). Used as manufacturing reference when CM is on."
                            if peak is not None else
                            "No FCO/K months found yet."
                        )
                        status = (
                            f"✅ Loaded **{len(raw_df):,}** claim rows · "
                            f"**{len(parts)}** parts · "
                            + " · ".join(msgs)
                            + f" · **{len(tmpl)}** months in production sheet."
                        )
                        if prod_note:
                            status += f" {prod_note}"
                        return (
                            status,
                            gr.update(choices=parts, value=first),
                            tmpl,
                            f"**{len(tmpl)} claim months** — confirm Production, then Process & Forecast.",
                            gr.update(choices=fcok, value=str(peak) if peak is not None else (fcok[0] if fcok else None)),
                            peak_msg,
                            prod_note or "",
                        )
                    except Exception:
                        return (
                            f"❌ {traceback.format_exc()}",
                            gr.update(choices=[], value=None),
                            pd.DataFrame(columns=["Month", "Production", "Production_Cost"]),
                            "*Error — see status.*",
                            gr.update(choices=[], value=None),
                            "",
                            "",
                        )

                def _on_part_change(part):
                    if state["raw"] is None or not part:
                        return (
                            pd.DataFrame(columns=["Month", "Production", "Production_Cost"]),
                            "*Select a part…*",
                            gr.update(choices=[], value=None),
                            "",
                        )
                    tmpl = month_input_template(state["raw"], part)
                    fcok = list_fcok_months(state["raw"], part)
                    peak, npeak = peak_fcok_month(state["raw"], part)
                    peak_msg = (
                        f"Highest-claim FCO/K month for **{part}**: `{peak}` "
                        f"({npeak:,} claims). CM will target this batch."
                        if peak is not None else "No FCO/K months found."
                    )
                    return (
                        tmpl,
                        f"**{len(tmpl)} claim months** — fill Production (+ cost for CPV).",
                        gr.update(
                            choices=fcok,
                            value=str(peak) if peak is not None else (fcok[0] if fcok else None),
                        ),
                        peak_msg,
                    )

                def _empty_train_pack(msg: str):
                    e = _empty_fig(msg)
                    empty_df = pd.DataFrame()
                    return (
                        msg, empty_df, e, e, e, empty_df, empty_df, empty_df, "—",
                        gr.update(), gr.update(), gr.update(),
                    )

                def _econ_hist_df(res: dict) -> pd.DataFrame:
                    months = [str(p) for p in res["monthly"]["period"]]
                    return pd.DataFrame({
                        "Month": months,
                        "Claims": res["claim_vals"],
                        "Production": res["production"],
                        "Production_Cost": res.get("production_cost", np.nan),
                        "CPV": res.get("cpv", np.nan),
                        "Claim_Ratio": res.get("claim_ratio", np.nan),
                        "Claim_Ratio_per_1k": res.get("claim_ratio_per_1k", np.nan),
                    }).round(4)

                def _train(part, retune, models, prod_df, cm_on, cm_date_val, cm_month,
                           cm_know, cm_pct,
                           progress=gr.Progress(track_tqdm=True)):
                    """Train Tab 1 using user production / cost / CM inputs."""
                    if state["raw"] is None:
                        return _empty_train_pack("⚠️ Load data first.")
                    if not part:
                        return _empty_train_pack("⚠️ Select a part name.")
                    if not models:
                        return _empty_train_pack(
                            "⚠️ Select at least one model "
                            "(CNN-LSTM, N-BEATS, Transformer, and/or SARIMA)."
                        )
                    try:
                        production, costs, _months = parse_month_inputs(
                            prod_df, require_production=False,
                        )
                    except ValueError as exc:
                        return _empty_train_pack(f"⚠️ {exc}")

                    cm_when = (str(cm_date_val or "").strip()
                               or (str(cm_month).strip() if cm_month else None))
                    # CM on only when user says yes AND a date (or FCO/K ref) exists
                    cm_active = bool(cm_on) and bool(cm_when)
                    if cm_on and not cm_when:
                        return _empty_train_pack(
                            "⚠️ Countermeasure is checked — enter a CM date "
                            "(or keep the FCO/K reference month)."
                        )

                    try:
                        from forecasting.pipeline.runner import train_uploaded_part

                        mode = (
                            f"single model ({models[0]})"
                            if len(models) == 1
                            else f"{len(models)} models → compare + best"
                        )
                        progress(0.02, desc=f"Starting {part} · {mode}…")
                        progress(0.05, desc="Walk-forward CV + fitting (see terminal)…")

                        res = train_uploaded_part(
                            state["raw"], part,
                            retune=bool(retune),
                            use_locked_params=not bool(retune),
                            selected_models=list(models),
                            production=production,
                            production_cost=costs,
                            cm_enabled=cm_active,
                            cm_month=cm_when or None,
                            cm_reduction_pct=float(cm_pct) if cm_know else None,
                            cm_reduction_known=bool(cm_know),
                        )

                        if res is None:
                            return _empty_train_pack(
                                f"⚠️ Could not train **{part}** "
                                "(too few months or no claims)."
                            )

                        progress(0.88, desc="Building charts & tables…")
                        slim = _slim_result(res)
                        state["results"] = {part: slim}
                        state["annotated_df"] = None
                        # NOTE: Univariate analysis is NOT run automatically here.
                        # It runs only when the user visits Tab 3 (Univariate Analysis)
                        # and clicks Run / Refresh, or selects a part on that tab.

                        fc_df = make_forecast_df(slim)
                        rank_df = slim.get("ranking_df")
                        if rank_df is None:
                            rank_df = pd.DataFrame()
                        else:
                            rank_df = rank_df.copy()
                        hp_df = make_best_params_df(slim)
                        econ_df = _econ_hist_df(slim)

                        sel = slim.get("selected_models", models)
                        cm_note = ""
                        if slim.get("has_cm") and slim.get("cm_sim"):
                            cms = slim["cm_sim"]
                            src = cms.get("reduction_source", "user")
                            cm_note = (
                                f" · CM `{cms.get('cm_month')}` "
                                f"reduction **{cms.get('reduction_pct', 0):.0f}%** "
                                f"({src}) → −{cms.get('improvement_pct', 0):.1f}% claims"
                            )
                            if cms.get("reduction_note"):
                                cm_note += f" — {cms['reduction_note']}"
                        prod_note = (
                            "" if slim.get("production_user_provided", True)
                            else " · claims-only (no production uploaded)"
                        )
                        econ = slim.get("economics") or {}
                        status = (
                            f"✅ Forecast ready for **{part}**. "
                            f"Selected: `{', '.join(sel)}`. "
                            f"Model: **{slim.get('best_model', '—')}**. "
                            f"Avg CPV: **{(econ.get('avg_cpv') or float('nan')):,.2f}** · "
                            f"Avg claim ratio: **{(econ.get('avg_claim_ratio') or float('nan')):.4f}**"
                            f"{cm_note}{prod_note}."
                        )
                        locked = (
                            f"**Forecast model:** {slim.get('best_model')} · "
                            f"**Selected:** {', '.join(sel)} · "
                            f"Params: `{BEST_PARAMS_PATH}`"
                        )

                        raw_df = state["raw"]
                        trained = sorted(state["results"].keys())
                        diag_upd = gr.update(choices=trained, value=part)
                        cm_upd = gr.update(choices=trained, value=part)
                        fcok = []
                        if raw_df is not None and "FCOK_MONTH" in raw_df.columns:
                            sub = raw_df[raw_df["Part Name"] == part]
                            fcok = sorted(
                                {str(p) for p in sub["FCOK_MONTH"].dropna().unique()}
                            )[-48:]
                        fcok_upd = gr.update(
                            choices=fcok,
                            value=fcok[-3:] if len(fcok) >= 3 else fcok,
                        )

                        progress(1.0, desc="Done")
                        return (
                            status,
                            econ_df,
                            _safe_fig(make_part_figure, slim, fallback="Part forecast"),
                            _safe_fig(make_12m_forecast_figure, slim, fallback="12m forecast"),
                            _safe_fig(make_actual_vs_predicted_figure, slim, fallback="Actual vs Pred"),
                            fc_df,
                            rank_df,
                            hp_df,
                            locked,
                            diag_upd,
                            cm_upd,
                            fcok_upd,
                        )
                    except Exception:
                        tb = traceback.format_exc()
                        logger.error("Training UI error:\n%s", tb)
                        return _empty_train_pack(f"❌ Training failed:\n```\n{tb}\n```")

                def _process_and_forecast(
                    claims_files, prod_file, part, retune, models,
                    prod_df, cm_on, cm_date_val, cm_month, cm_know, cm_pct,
                    progress=gr.Progress(track_tqdm=True),
                ):
                    """Load claims + optional production, then train automatically."""
                    load_out = _load_files(claims_files, prod_file)
                    status, part_upd, table, hint, cm_dd_upd, peak_md, prod_note = load_out
                    if state["raw"] is None:
                        empty = _empty_train_pack(status)
                        return (
                            status, part_upd, table, hint, cm_dd_upd, peak_md, prod_note,
                            *empty,
                        )
                    part_val = part
                    if isinstance(part_upd, dict):
                        part_val = part_upd.get("value") or part
                    if not part_val and isinstance(part_upd, dict):
                        choices = part_upd.get("choices") or []
                        part_val = choices[0] if choices else None
                    train_out = _train(
                        part_val, retune, models, table,
                        cm_on, cm_date_val, cm_month, cm_know, cm_pct,
                        progress=progress,
                    )
                    return (
                        status, part_upd, table, hint, cm_dd_upd, peak_md, prod_note,
                        *train_out,
                    )

                load_btn.click(
                    fn=_load_files,
                    inputs=[upload_box, up_prod_file],
                    outputs=[
                        load_status, part_dd, prod_table, prod_hint,
                        cm_month_dd, cm_peak_md, prod_csv_status,
                    ],
                )
                part_dd.change(
                    fn=_on_part_change,
                    inputs=[part_dd],
                    outputs=[prod_table, prod_hint, cm_month_dd, cm_peak_md],
                )

                def _download_prod_csv(part, table):
                    """Download current table, or rebuild template for selected part."""
                    try:
                        if table is not None and len(table) > 0:
                            df = pd.DataFrame(table)
                        elif state["raw"] is not None and part:
                            df = month_input_template(state["raw"], part)
                        else:
                            return None, "⚠️ Select a part (or fill the table) first."
                        safe = "".join(
                            c if c.isalnum() or c in "-_" else "_"
                            for c in str(part or "part")
                        )
                        fd, path = tempfile.mkstemp(
                            suffix=".csv",
                            prefix=f"production_cost_{safe}_",
                        )
                        os.close(fd)
                        production_sheet_to_csv(df, path)
                        return path, (
                            f"✅ CSV ready ({len(df)} months). "
                            "Fill **Production** (+ **Production_Cost**), save, then upload."
                        )
                    except Exception:
                        return None, f"❌ {traceback.format_exc()}"

                def _upload_prod_csv(file):
                    if file is None:
                        return (
                            gr.update(),
                            "⚠️ Choose a filled CSV file first.",
                        )
                    try:
                        path = file.name if hasattr(file, "name") else str(file)
                        df = load_production_sheet_csv(path)
                        # Soft-validate so user sees issues early
                        try:
                            parse_month_inputs(df, require_production=False)
                            note = f"✅ Loaded **{len(df)}** months from file."
                        except ValueError as exc:
                            note = (
                                f"✅ Loaded **{len(df)}** rows into the table. "
                                f"Note: {exc}"
                            )
                        return df, note
                    except Exception as exc:
                        return gr.update(), f"❌ Could not read CSV: {exc}"

                dl_prod_btn.click(
                    fn=_download_prod_csv,
                    inputs=[part_dd, prod_table],
                    outputs=[dl_prod_file, prod_csv_status],
                )
                up_prod_btn.click(
                    fn=_upload_prod_csv,
                    inputs=[up_prod_file],
                    outputs=[prod_table, prod_csv_status],
                )

                def _download_full_template():
                    try:
                        path = build_upload_templates()
                        return path, f"✅ Template ready: `{path}`"
                    except Exception:
                        return None, f"❌ {traceback.format_exc()}"

                dl_tmpl_btn.click(
                    fn=_download_full_template,
                    outputs=[dl_tmpl_file, tmpl_status],
                )

            # ── 2. Diagnostics ───────────────────────────────────────────
            with gr.Tab("2. Diagnostics"):
                with gr.Accordion("Diagnostics charts", open=True):
                    gr.Markdown(
                        "Pick a trained part and **Refresh** — charts update for that part."
                    )
                    with gr.Row():
                        diag_part_dd = gr.Dropdown(
                            choices=[], label="Trained part", interactive=True,
                        )
                        refresh_diag = gr.Button("Refresh Diagnostics", variant="primary")
                    with gr.Row():
                        d_prod = gr.Plot(label="Production")
                        d_age = gr.Plot(label="Vehicle Age")
                    with gr.Row():
                        d_odo = gr.Plot(label="Odometer")
                        d_heat = gr.Plot(label="FCO/K × Process Heatmap")
                    with gr.Row():
                        d_cv = gr.Plot(label="Rolling CV Performance")
                        d_cmp = gr.Plot(label="Model Comparison")

                def _refresh_diag(part):
                    trained = sorted(state["results"].keys())
                    if not part and trained:
                        part = trained[0]
                    upd = gr.update(
                        choices=trained,
                        value=part if part in trained else (trained[0] if trained else None),
                    )
                    if not part or part not in state["results"]:
                        e = _empty_fig("Train a part first")
                        return upd, e, e, e, e, e, e
                    r = state["results"][part]
                    raw_df = state["raw"]
                    return (
                        upd,
                        _safe_fig(make_production_figure, r),
                        _safe_fig(make_vehicle_age_figure, raw_df, part),
                        _safe_fig(make_odometer_figure, raw_df, part),
                        _safe_fig(make_fcok_heatmap, raw_df, part),
                        _safe_fig(make_cv_fold_figure, r),
                        _safe_fig(make_model_comparison_figure, r),
                    )

                refresh_diag.click(
                    fn=_refresh_diag,
                    inputs=[diag_part_dd],
                    outputs=[diag_part_dd, d_prod, d_age, d_odo, d_heat, d_cv, d_cmp],
                )
                diag_part_dd.change(
                    fn=_refresh_diag,
                    inputs=[diag_part_dd],
                    outputs=[diag_part_dd, d_prod, d_age, d_odo, d_heat, d_cv, d_cmp],
                )

            # ── 3. Countermeasure / Reduction ─────────────────────────────
            with gr.Tab("3. Univariate Analysis"):
                gr.Markdown(
                    "### Claims-only univariate analysis\n"
                    "Independent of production. Use **Tab 1 claims** *or* upload a "
                    "**monthly claims CSV** (Part Name + Month + Claims). "
                    "Runs after Tab 1 forecast, or click **Run / Refresh** here."
                )
                with gr.Accordion("Upload monthly claims template", open=True):
                    gr.Markdown(
                        "Columns: **Part Name**, **Month** (`YYYY-MM`), **Claims**. "
                        "One row per part-month. CSV upload here does not change Tab 1."
                    )
                    with gr.Row():
                        uni_dl_tmpl_btn = gr.Button(
                            "Download CSV template", variant="secondary",
                        )
                        uni_dl_tmpl_file = gr.File(label="Template CSV")
                    with gr.Row():
                        uni_csv_file = gr.File(
                            label="Drop monthly claims CSV / Excel",
                            file_types=[".csv", ".xlsx", ".xls"],
                            file_count="single",
                            height=120,
                        )
                        uni_csv_load_btn = gr.Button(
                            "Load monthly CSV", variant="secondary",
                        )
                    uni_csv_status = gr.Markdown(
                        "*Optional: download the template, fill monthly claims, then load.*"
                    )
                with gr.Row():
                    uni_part_dd = gr.Dropdown(choices=[], label="Part Name", interactive=True)
                    uni_run_btn = gr.Button("Run / Refresh Univariate Analysis", variant="primary")
                uni_model_ck = gr.CheckboxGroup(
                    choices=["Holt-Winters", "SARIMA", "CNN-LSTM", "Transformer", "N-BEATS"],
                    value=[],
                    label="Models for univariate analysis",
                    info="Select at least one model. ONE model → forecast with that model. MULTIPLE → compare, rank, and use the best (lowest RMSE).",
                )
                uni_status = gr.Markdown(
                    "*Load Tab 1 claims **or** a monthly CSV above, then run analysis.*"
                )

                with gr.Accordion("1. Data overview", open=True):
                    uni_overview = gr.Dataframe(
                        interactive=False, wrap=True, label="Overview",
                        elem_id="uni-overview-table",
                    )
                with gr.Accordion("2. Claim trend analysis", open=True):
                    with gr.Row():
                        uni_trend_plot = gr.Plot(label="Monthly claims + rolling averages")
                        uni_seas_plot = gr.Plot(label="Seasonality")
                    uni_trend_md = gr.Markdown("")
                with gr.Accordion("3. Univariate model results", open=True):
                    uni_rank = gr.Dataframe(
                        interactive=False, wrap=True,
                        label="Holt-Winters · SARIMA · CNN-LSTM · Transformer · N-BEATS",
                        elem_id="uni-rank-table",
                    )
                    uni_hw = gr.Markdown("")
                with gr.Accordion("4. Forecast output", open=True):
                    uni_fc_plot = gr.Plot(label="Univariate forecast + CI")
                    uni_fc_table = gr.Dataframe(
                        interactive=False, wrap=True, label="Forecast table",
                        elem_id="uni-fc-table",
                    )
                with gr.Accordion("5. Automated insights", open=True):
                    uni_insights = gr.Markdown("")

                def _empty_uni(msg: str):
                    e = _empty_fig(msg)
                    return (
                        msg,
                        pd.DataFrame(),
                        e, e, "",
                        pd.DataFrame(),
                        "",
                        e,
                        pd.DataFrame(),
                        "",
                        gr.update(),
                    )

                def _render_uni(res: dict):
                    t = res.get("trend") or {}
                    trend_txt = (
                        f"**Trend:** {t.get('direction')} · "
                        f"Growth share {t.get('growth_pct', 0):.1f}% · "
                        f"Decline share {t.get('decline_pct', 0):.1f}% · "
                        f"6-month change {t.get('chg_6m', 0):+.1f}%"
                    )
                    hw = res.get("holt_params") or {}
                    sar = res.get("sarima_params") or {}
                    _rmse_v = hw.get("holdout_rmse")
                    _rmse_str = f"{float(_rmse_v):.3f}" if _rmse_v is not None and np.isfinite(float(_rmse_v)) else "—"
                    hw_txt = (
                        f"**Holt-Winters best params** — α={hw.get('alpha')} · "
                        f"β={hw.get('beta')} · γ={hw.get('gamma')} "
                        f"(holdout RMSE {_rmse_str}).  \n"
                        f"**SARIMA** order={sar.get('order')} seasonal={sar.get('seasonal_order')}.  \n"
                        f"**Best model (lowest RMSE):** **{res.get('best_model')}**"
                    )
                    ins = res.get("insights") or []
                    ins_md = "\n".join(f"- {line}" for line in ins)
                    trained = sorted((state.get("univariate") or {}).keys())
                    part = res.get("part")
                    return (
                        f"✅ Univariate analysis ready for **{part}** "
                        f"(claims only · models: {', '.join(res.get('selected_models') or [])} · "
                        f"horizon {res.get('horizon')} mo).",
                        uni_overview_df(res),
                        _safe_fig(make_uni_trend_figure, res, fallback="Trend"),
                        _safe_fig(make_uni_season_figure, res, fallback="Seasonality"),
                        trend_txt,
                        res.get("ranking", pd.DataFrame()),
                        hw_txt,
                        _safe_fig(make_uni_forecast_figure, res, fallback="Forecast"),
                        uni_forecast_df(res),
                        ins_md,
                        gr.update(choices=trained, value=part),
                    )

                def _uni_part_choices():
                    parts = []
                    sheet = state.get("uni_monthly_sheet")
                    if sheet is not None and not sheet.empty and "Part Name" in sheet.columns:
                        parts.extend(
                            sorted(sheet["Part Name"].dropna().astype(str).unique().tolist())
                        )
                    raw_df = state.get("raw")
                    if raw_df is not None and "Part Name" in raw_df.columns:
                        for p in sorted(raw_df["Part Name"].dropna().unique().tolist()):
                            if str(p) not in parts:
                                parts.append(str(p))
                    return parts

                def _run_uni_impl(part, models, *, force: bool, progress):
                    parts = _uni_part_choices()
                    sheet = state.get("uni_monthly_sheet")
                    raw_df = state.get("raw")
                    models = list(models) if models else []
                    if not parts:
                        return _empty_uni(
                            "⚠️ Load a monthly claims CSV on this tab, "
                            "or load a claims file on Tab 1 first."
                        )
                    if not models:
                        return _empty_uni(
                            "⚠️ Select at least one model from the "
                            "**Models for univariate analysis** checkboxes above."
                        )
                    if not part:
                        part = parts[0]
                    part = str(part)
                    cached = (state.get("univariate") or {}).get(part)
                    same_models = (
                        cached is not None
                        and list(cached.get("selected_models") or []) == models
                    )
                    if cached is not None and not force and same_models:
                        out = list(_render_uni(cached))
                        out[-1] = gr.update(choices=parts or [part], value=part)
                        return tuple(out)
                    try:
                        progress(0.1, desc=f"Univariate analysis · {part}")
                        from forecasting.pipeline.univariate import (
                            monthly_from_claims_sheet,
                            run_univariate_analysis,
                            run_univariate_from_monthly,
                        )
                        uni_res = None
                        use_sheet = (
                            sheet is not None
                            and not sheet.empty
                            and str(part) in sheet["Part Name"].astype(str).values
                        )
                        if use_sheet:
                            monthly = monthly_from_claims_sheet(sheet, part)
                            uni_res = run_univariate_from_monthly(
                                monthly, part, selected_models=models,
                            )
                        elif raw_df is not None:
                            uni_res = run_univariate_analysis(
                                raw_df, part, selected_models=models,
                            )
                        if uni_res is None:
                            return _empty_uni(
                                f"⚠️ Not enough monthly claims for **{part}** "
                                "(need ~10+ months)."
                            )
                        state.setdefault("univariate", {})[part] = uni_res
                        progress(1.0, desc="Done")
                        out = list(_render_uni(uni_res))
                        out[-1] = gr.update(choices=parts or [part], value=part)
                        return tuple(out)
                    except Exception:
                        tb = traceback.format_exc()
                        logger.error("Univariate UI error:\n%s", tb)
                        return _empty_uni(f"❌ Univariate failed:\n```\n{tb}\n```")

                def _download_uni_csv_template():
                    try:
                        from forecasting.pipeline.univariate import monthly_claims_template_csv
                        path = monthly_claims_template_csv()
                        return path, f"✅ Template ready: `{path}`"
                    except Exception:
                        return None, f"❌ {traceback.format_exc()}"

                def _load_uni_monthly_csv(file):
                    if file is None:
                        return (
                            "*Choose a filled monthly claims CSV first.*",
                            gr.update(),
                        )
                    try:
                        from forecasting.pipeline.univariate import load_monthly_claims_csv
                        path = file.name if hasattr(file, "name") else str(file)
                        sheet = load_monthly_claims_csv(path)
                        state["uni_monthly_sheet"] = sheet
                        state["univariate"] = {}
                        parts = _uni_part_choices()
                        first = parts[0] if parts else None
                        n_parts = sheet["Part Name"].nunique()
                        n_rows = len(sheet)
                        return (
                            f"✅ Loaded **{n_rows}** monthly rows · **{n_parts}** part(s). "
                            "Select a part and click **Run / Refresh**.",
                            gr.update(choices=parts, value=first),
                        )
                    except Exception as exc:
                        return f"❌ Could not read monthly CSV: {exc}", gr.update()

                def _run_uni_refresh(part, models, progress=gr.Progress(track_tqdm=True)):
                    return _run_uni_impl(part, models, force=True, progress=progress)

                def _run_uni_show(part, models, progress=gr.Progress(track_tqdm=True)):
                    return _run_uni_impl(part, models, force=False, progress=progress)

                uni_dl_tmpl_btn.click(
                    fn=_download_uni_csv_template,
                    outputs=[uni_dl_tmpl_file, uni_csv_status],
                )
                uni_csv_load_btn.click(
                    fn=_load_uni_monthly_csv,
                    inputs=[uni_csv_file],
                    outputs=[uni_csv_status, uni_part_dd],
                )
                uni_run_btn.click(
                    fn=_run_uni_refresh,
                    inputs=[uni_part_dd, uni_model_ck],
                    outputs=[
                        uni_status, uni_overview, uni_trend_plot, uni_seas_plot,
                        uni_trend_md, uni_rank, uni_hw,
                        uni_fc_plot, uni_fc_table, uni_insights, uni_part_dd,
                    ],
                )
                uni_part_dd.change(
                    fn=_run_uni_show,
                    inputs=[uni_part_dd, uni_model_ck],
                    outputs=[
                        uni_status, uni_overview, uni_trend_plot, uni_seas_plot,
                        uni_trend_md, uni_rank, uni_hw,
                        uni_fc_plot, uni_fc_table, uni_insights, uni_part_dd,
                    ],
                )

            with gr.Tab("4. Reduction"):
                with gr.Accordion("Live reduction simulator", open=True):
                    gr.Markdown(
                        f"Drag **Estimate reduction %** — charts update live "
                        f"(~55% sensitivity floor). Warranty: **{WARRANTY_MONTHS} mo**."
                    )
                    with gr.Row():
                        cm_part_dd = gr.Dropdown(choices=[], label="Trained part")
                        cm_fcok = gr.Dropdown(
                            choices=[], multiselect=True, label="FCO/K Month(s)",
                        )
                    cm_pct = gr.Slider(
                        0, 100, value=30, step=1,
                        label="Estimate reduction %",
                        info="Charts refresh as you drag.",
                    )
                    cm_btn = gr.Button("Refresh simulation", variant="primary")
                    cm_kpi = gr.Markdown()
                    with gr.Row():
                        cm_impact = gr.Plot(label="Impact")
                        cm_base = gr.Plot(label="Baseline vs Adjusted")
                    cm_table = gr.Dataframe(interactive=False)
                    cm_file = gr.File(label="Download CM CSV")

                def _prep_cm_part(part):
                    trained = sorted(state["results"].keys())
                    raw_df = state["raw"]
                    fcok = []
                    if raw_df is not None and part:
                        sub = raw_df[raw_df["Part Name"] == part]
                        if "FCOK_MONTH" in sub.columns:
                            fcok = sorted(
                                {str(p) for p in sub["FCOK_MONTH"].dropna().unique()}
                            )[-48:]
                    return (
                        gr.update(
                            choices=trained,
                            value=part if part in trained else (trained[0] if trained else None),
                        ),
                        gr.update(choices=fcok, value=fcok[-3:] if len(fcok) >= 3 else fcok),
                    )

                def _run_cm(part, selected, pct):
                    if not part or part not in state["results"]:
                        e = _empty_fig("Train a part first")
                        return "⚠️ Train a part first.", e, e, pd.DataFrame(), None
                    if not selected:
                        e = _empty_fig("Select FCO/K months")
                        return "⚠️ Select at least one FCO/K month.", e, e, pd.DataFrame(), None
                    r = state["results"][part]
                    # Always simulate from pre-CM baseline so slider sensitivity is clear
                    baseline = r.get("best_forecast_raw", r.get("best_forecast", r["ensemble_raw"]))
                    sim = simulate_fcok_countermeasure(
                        baseline_forecast=baseline,
                        raw=state["raw"],
                        part=part,
                        selected_fcok_months=selected,
                        reduction_pct=pct,
                        future_periods=r["future_periods"],
                    )
                    rows = []
                    for i, fp in enumerate(r["future_periods"]):
                        rows.append({
                            "Month": p2s(fp),
                            "Original Forecast": round(float(sim["original"][i]), 2),
                            "Adjusted Forecast": round(float(sim["adjusted"][i]), 2),
                            "Monthly Reduction": round(float(sim["monthly_reduction"][i]), 2),
                            "Cumulative Reduction": round(float(sim["cumulative_reduction"][i]), 2),
                            "Share (effective)": round(float(sim.get("fcok_share_effective", sim["fcok_share"])[i]), 3),
                            "Improvement %": round(sim["improvement_pct"], 2),
                        })
                    df = pd.DataFrame(rows)
                    tmp = tempfile.NamedTemporaryFile(
                        mode="w", suffix=".csv", prefix="cm_",
                        delete=False, encoding="utf-8",
                    )
                    df.to_csv(tmp.name, index=False)
                    tmp.close()
                    delta = float(np.sum(sim["monthly_reduction"]))
                    kpi = (
                        f"**−{sim['improvement_pct']:.1f}%** total claims · "
                        f"**−{delta:,.0f}** claims removed · "
                        f"FCO/K `{', '.join(sim['selected_fcok'])}` · "
                        f"Reduction **{pct:.0f}%**"
                    )
                    return (
                        kpi,
                        _safe_fig(make_countermeasure_impact_figure, sim),
                        _safe_fig(make_baseline_vs_adjusted_figure, sim, r["future_periods"]),
                        df,
                        tmp.name,
                    )

                cm_part_dd.change(
                    fn=_prep_cm_part,
                    inputs=[cm_part_dd],
                    outputs=[cm_part_dd, cm_fcok],
                )
                cm_btn.click(
                    fn=_run_cm,
                    inputs=[cm_part_dd, cm_fcok, cm_pct],
                    outputs=[cm_kpi, cm_impact, cm_base, cm_table, cm_file],
                )
                # Live sensitivity: update plots when reduction % / FCO/K changes
                cm_pct.change(
                    fn=_run_cm,
                    inputs=[cm_part_dd, cm_fcok, cm_pct],
                    outputs=[cm_kpi, cm_impact, cm_base, cm_table, cm_file],
                )
                cm_fcok.change(
                    fn=_run_cm,
                    inputs=[cm_part_dd, cm_fcok, cm_pct],
                    outputs=[cm_kpi, cm_impact, cm_base, cm_table, cm_file],
                )

                # ── CM Engine Analysis panel (new production-adjusted model) ──
                with gr.Accordion(
                    "🔧 CM Engine Analysis (Production-Adjusted Baseline)",
                    open=True,
                ):
                    gr.Markdown(
                        "**Production-adjusted countermeasure model.** "
                        "Identifies peak FCOK months, computes adjusted production "
                        f"baseline (`avg_prod − avg_claims`), and projects claims "
                        f"across the **{WARRANTY_MONTHS}-month warranty window**. "
                        "Factor = **1.0** when reduction is known (full CM effectiveness, no softening)."
                    )
                    with gr.Row():
                        cm_eng_part_dd = gr.Dropdown(
                            choices=[], label="Trained part", interactive=True,
                            elem_id="cm-eng-part-dd",
                        )
                        cm_eng_refresh_btn = gr.Button(
                            "🔄 Refresh CM Engine Analysis", variant="primary",
                        )
                    cm_eng_savings = gr.HTML(
                        value="<i>Train a part with CM enabled to see the analysis.</i>",
                        label="Summary",
                    )
                    with gr.Row():
                        cm_eng_fig = gr.Plot(label="CM Engine — 4-Panel Analysis")
                        cm_eng_prod_fig = gr.Plot(label="Production Trajectory")
                    cm_eng_table = gr.Dataframe(
                        label="Month-by-month comparison table",
                        interactive=False, wrap=True,
                        elem_id="cm-eng-table",
                    )
                    cm_eng_file = gr.File(label="Download comparison CSV")

                def _run_cm_engine(part):
                    """Refresh the CM Engine Analysis panel from the stored result."""
                    trained = sorted(state["results"].keys())
                    part_upd = gr.update(
                        choices=trained,
                        value=part if part in trained else (trained[0] if trained else None),
                    )
                    if not part or part not in state["results"]:
                        e = _empty_fig("Train a part with CM enabled first.")
                        return (
                            part_upd,
                            "<i>Train a part with CM enabled to see the analysis.</i>",
                            e, e, pd.DataFrame(), None,
                        )
                    r = state["results"].get(part, {})
                    cm_analysis = r.get("cm_analysis")
                    if not cm_analysis:
                        e = _empty_fig("No CM analysis stored — enable CM and retrain.")
                        return (
                            part_upd,
                            "<i>Enable CM (in the upload panel) and retrain to see analysis.</i>",
                            e, e, pd.DataFrame(), None,
                        )
                    fig4 = _safe_fig(make_cm_analysis_figure, cm_analysis)
                    fig_prod = _safe_fig(make_cm_production_figure, cm_analysis)
                    savings_html = make_cm_savings_html(cm_analysis)
                    comp = cm_analysis.get("comparison", {})
                    comp_df = comp.get("comparison_df", pd.DataFrame())
                    # Export comparison CSV
                    tmp = None
                    if not comp_df.empty:
                        import tempfile as _tf
                        _f = _tf.NamedTemporaryFile(
                            mode="w", suffix=".csv",
                            prefix=f"cm_engine_{part}_",
                            delete=False, encoding="utf-8",
                        )
                        comp_df.to_csv(_f.name, index=False)
                        _f.close()
                        tmp = _f.name
                    return part_upd, savings_html, fig4, fig_prod, comp_df, tmp

                cm_eng_refresh_btn.click(
                    fn=_run_cm_engine,
                    inputs=[cm_eng_part_dd],
                    outputs=[
                        cm_eng_part_dd, cm_eng_savings,
                        cm_eng_fig, cm_eng_prod_fig,
                        cm_eng_table, cm_eng_file,
                    ],
                )
                cm_eng_part_dd.change(
                    fn=_run_cm_engine,
                    inputs=[cm_eng_part_dd],
                    outputs=[
                        cm_eng_part_dd, cm_eng_savings,
                        cm_eng_fig, cm_eng_prod_fig,
                        cm_eng_table, cm_eng_file,
                    ],
                )


            with gr.Tab("5. Annotated Walk-Forward"):
                gr.Markdown("""
### Reference-style last-6-month evaluation
Same data feeding as the annotated CNN-LSTM notebook:
**Part_Failure · Production · Warranty_Days · Countermeasure · FCOK_Jan_Aug**,
`time_step=4`, scaler fit only on history before each test month
(no leakage — model retrained per fold).
Supports **Keras CNN-LSTM** (if TensorFlow installed), NumPy CNN-LSTM / N-BEATS /
Transformer, and SARIMA.
                """)
                ann_choices = (
                    ["Keras-CNN-LSTM", "CNN-LSTM", "N-BEATS", "Transformer", "SARIMA"]
                    if HAS_TF else
                    ["CNN-LSTM", "N-BEATS", "Transformer", "SARIMA"]
                )
                with gr.Row():
                    ann_models = gr.CheckboxGroup(
                        choices=ann_choices,
                        value=["CNN-LSTM"] if not HAS_TF else ["Keras-CNN-LSTM"],
                        label="Annotated models",
                    )
                    ann_epochs = gr.Slider(
                        20, 100, value=40, step=5,
                        label="Keras epochs (Keras-CNN-LSTM only)",
                    )
                    ann_cm = gr.Textbox(
                        value="",
                        label="Countermeasure start (YYYY-MM-DD, optional)",
                        placeholder="e.g. 2025-08-01",
                    )
                ann_btn = gr.Button(
                    "Run Annotated Last-6-Month Forecast", variant="primary",
                )
                ann_status = gr.Markdown(
                    "*Load data & select a part on tab 1, then run here.*"
                )
                ann_plot = gr.Plot(label="Actuals vs Forecasts")
                ann_table = gr.Dataframe(
                    label="Results (Error / Error% / Accuracy%)",
                    interactive=False, wrap=True,
                )
                ann_file = gr.File(label="Download annotated CSV")

                def _run_annotated(part, models, epochs, cm_start,
                                   progress=gr.Progress(track_tqdm=True)):
                    if state["raw"] is None:
                        return (
                            "⚠️ Load data on tab 1 first.",
                            _empty_fig("Load data first"),
                            pd.DataFrame(),
                            None,
                        )
                    if not part:
                        return (
                            "⚠️ Select a part on tab 1.",
                            _empty_fig("Select a part"),
                            pd.DataFrame(),
                            None,
                        )
                    if not models:
                        return (
                            "⚠️ Select at least one annotated model.",
                            _empty_fig("Select models"),
                            pd.DataFrame(),
                            None,
                        )
                    try:
                        progress(0.05, desc=f"Annotated walk-forward · {part}…")
                        cm = (cm_start or "").strip() or None
                        df = forecast_last_n_months_annotated(
                            state["raw"],
                            part,
                            models=list(models),
                            time_step=4,
                            epochs=int(epochs),
                            n_test_months=N_CV_FOLDS,
                            countermeasure_start=cm,
                        )
                        if df is None or df.empty:
                            return (
                                f"⚠️ Not enough monthly history for **{part}** "
                                f"(need lookback + {N_CV_FOLDS} test months).",
                                _empty_fig("Insufficient history"),
                                pd.DataFrame(),
                                None,
                            )
                        path = df.attrs.get("save_path")
                        state["annotated_df"] = df
                        progress(1.0, desc="Done")
                        return (
                            f"✅ Annotated walk-forward complete for **{part}**. "
                            f"Models: `{', '.join(models)}`. "
                            + (f"Saved → `{path}`" if path else ""),
                            _safe_fig(
                                make_annotated_results_figure, df,
                                fallback="Annotated chart",
                            ),
                            df,
                            path,
                        )
                    except Exception:
                        tb = traceback.format_exc()
                        logger.error("Annotated UI error:\n%s", tb)
                        return (
                            f"❌ Annotated run failed:\n```\n{tb}\n```",
                            _empty_fig("Error"),
                            pd.DataFrame(),
                            None,
                        )

                ann_btn.click(
                    fn=_run_annotated,
                    inputs=[part_dd, ann_models, ann_epochs, ann_cm],
                    outputs=[ann_status, ann_plot, ann_table, ann_file],
                )

            # ── 5. Summary / Export ───────────────────────────────────────
            with gr.Tab("6. Summary & Export"):
                gr.Markdown(
                    "### Results overview\n"
                    "Filled automatically after training (or click Refresh)."
                )
                sum_btn = gr.Button("Refresh Summary", variant="secondary")
                kpi_html = gr.HTML(
                    "<p style='color:#000000;font-weight:600'>Train at least one part.</p>"
                )
                sum_table = gr.Dataframe(interactive=False)
                rank_global = gr.Dataframe(interactive=False)
                with gr.Row():
                    dl_fc = gr.Button("Download Forecasts CSV")
                    dl_fc_file = gr.File()
                    dl_rank = gr.Button("Download Ranking CSV")
                    dl_rank_file = gr.File()

                gr.HTML("""
<div class="aw-ppt-box">
  <div style="font-size:16px;font-weight:800;color:#000000;margin-bottom:4px;">
    PowerPoint export
  </div>
  <div style="font-size:13px;color:#000000;margin-bottom:2px;">
    Build a briefing deck with executive summary, best-model forecast,
    last-6-month accuracy (best vs second-best), CPV / CR, and insights.
    Univariate slides are added when Tab 3 has been run. Chart images need <code>kaleido</code>.
  </div>
</div>
                """)
                with gr.Row():
                    dl_ppt = gr.Button("Download PowerPoint (.pptx)", variant="primary")
                    dl_ppt_file = gr.File(label="PPTX file")
                ppt_status = gr.Markdown("")

                def _summary():
                    results_list = list(state["results"].values())
                    if not results_list:
                        return (
                            "<p style='color:#000000;font-weight:600'>Train at least one part.</p>",
                            pd.DataFrame(), pd.DataFrame(),
                        )
                    return (
                        make_kpi_html(results_list),
                        make_summary_df(results_list),
                        make_global_ranking_df(results_list),
                    )

                def _dl_fc():
                    frames = []
                    for r in state["results"].values():
                        df = make_forecast_df(r).copy()
                        df.insert(0, "Part", r["part"])
                        frames.append(df)
                    if not frames:
                        return None
                    tmp = tempfile.NamedTemporaryFile(
                        mode="w", suffix=".csv", prefix="forecasts_",
                        delete=False, encoding="utf-8",
                    )
                    pd.concat(frames, ignore_index=True).to_csv(tmp.name, index=False)
                    tmp.close()
                    return tmp.name

                def _dl_rank():
                    results_list = list(state["results"].values())
                    if not results_list:
                        return None
                    tmp = tempfile.NamedTemporaryFile(
                        mode="w", suffix=".csv", prefix="ranking_",
                        delete=False, encoding="utf-8",
                    )
                    make_global_ranking_df(results_list).to_csv(tmp.name, index=False)
                    tmp.close()
                    return tmp.name

                def _dl_ppt():
                    results_list = list(state["results"].values())
                    ann = state.get("annotated_df")
                    if not results_list and not (state.get("univariate") or {}) and (
                        ann is None or getattr(ann, "empty", True)
                    ):
                        return None, (
                            "⚠️ Train a part on Tab 1 and/or run univariate analysis "
                            "on Tab 3 before exporting PPT."
                        )
                    try:
                        from forecasting.dashboard.ppt_export import build_forecast_pptx
                        path = build_forecast_pptx(
                            results_list,
                            annotated_df=ann if isinstance(ann, pd.DataFrame) else None,
                            raw=state.get("raw"),
                            univariate=state.get("univariate") or {},
                        )
                        n_uni = len(state.get("univariate") or {})
                        extra = f" · univariate {n_uni} part(s)" if n_uni else ""
                        return path, (
                            f"✅ PowerPoint ready (multivariate{extra}): `{path}`"
                        )
                    except Exception:
                        tb = traceback.format_exc()
                        logger.error("PPT export failed:\n%s", tb)
                        return None, f"❌ PPT export failed:\n```\n{tb}\n```"

                sum_btn.click(fn=_summary, outputs=[kpi_html, sum_table, rank_global])
                dl_fc.click(fn=_dl_fc, outputs=[dl_fc_file])
                dl_rank.click(fn=_dl_rank, outputs=[dl_rank_file])
                dl_ppt.click(fn=_dl_ppt, outputs=[dl_ppt_file, ppt_status])

            # Process & Forecast: load claims+production then train
            auto_btn.click(
                fn=_process_and_forecast,
                inputs=[
                    upload_box, up_prod_file, part_dd, retune_ck, model_ck,
                    prod_table, cm_yes, cm_date, cm_month_dd, cm_know_red, cm_red,
                ],
                outputs=[
                    load_status, part_dd, prod_table, prod_hint,
                    cm_month_dd, cm_peak_md, prod_csv_status,
                    train_status, econ_table, part_plot, fc_plot, avp_plot,
                    fc_table, rank_table, hp_table, locked_md,
                    diag_part_dd, cm_part_dd, cm_fcok,
                ],
                show_progress="minimal",
            )
            train_btn.click(
                fn=_train,
                inputs=[
                    part_dd, retune_ck, model_ck, prod_table,
                    cm_yes, cm_date, cm_month_dd, cm_know_red, cm_red,
                ],
                outputs=[
                    train_status, econ_table, part_plot, fc_plot, avp_plot,
                    fc_table, rank_table, hp_table, locked_md,
                    diag_part_dd, cm_part_dd, cm_fcok,
                ],
                show_progress="minimal",
            )

        gr.Markdown(
            f"*Upload claims + production, optionally set a countermeasure date. "
            f"Warranty {WARRANTY_MONTHS} mo · No synthetic production.*"
        )

        # Ensure queue so long training jobs return plots reliably
        demo.queue(default_concurrency_limit=1)

    return demo
