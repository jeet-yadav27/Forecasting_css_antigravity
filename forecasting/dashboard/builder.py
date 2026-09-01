"""
forecasting/dashboard/builder.py
---------------------------------
Plotly figure builders and Gradio app assembler.

Includes:
  â€¢ Historical / production / age / odometer charts
  â€¢ Manufacturing-month Ã— claim-month heatmap
  â€¢ Actual vs predicted & rolling CV performance
  â€¢ Model comparison / ranking dashboard
  â€¢ 12-month forecast views
  â€¢ Interactive FCO/K manufacturing-month countermeasure simulation
  â€¢ CSV / Excel upload, validation, and export
"""

from __future__ import annotations

import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gradio as gr

from forecasting.config import (
    PRODUCTION_PER_MONTH,
    FORECAST_HORIZON,
    CM_DECAY_HALF_LIFE,
    N_CV_FOLDS,
    HP_TUNE,
    PALETTE,
    MODEL_COLORS,
    WARRANTY_MONTHS,
    MODEL_NAMES,
)
from forecasting.data.loader import (
    build_fcok_process_matrix,
    simulate_fcok_countermeasure,
)


# ---------------------------------------------------------------------------
# Colour constants
# ---------------------------------------------------------------------------

# High-contrast light theme — all UI / chart text black
_BG0   = "#E8EEF7"
_BG1   = "#FFFFFF"
_BG2   = "#D6E2F5"
_A1    = "#1D4ED8"
_A2    = "#047857"
_A3    = "#B45309"
_A4    = "#B91C1C"
_TXT   = "#000000"
_MUTED = "#000000"

_LAYOUT = dict(paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", font_color=_TXT)





def p2s(p) -> str:
    return str(p)


def _style(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    fig.update_layout(
        **_LAYOUT,
        title_text=f"<b>{title}</b>",
        title_font=dict(size=17, color=_TXT, family="DM Sans, Inter, sans-serif"),
        height=height,
        margin=dict(l=48, r=36, t=64, b=48),
        font=dict(color=_TXT, size=12, family="DM Sans, Inter, sans-serif"),
        legend=dict(
            bgcolor="rgba(255,255,255,0.96)",
            bordercolor="rgba(74,14,14,0.25)",
            borderwidth=1,
            font=dict(size=11, color=_TXT),
        ),
        coloraxis_colorbar=dict(
            tickfont=dict(color=_TXT),
            title_font=dict(color=_TXT),
        ),
    )
    fig.update_xaxes(
        showgrid=False, tickangle=-30,
        tickfont=dict(color=_TXT, size=11),
        title_font=dict(color=_TXT, size=12),
        linecolor="rgba(74,14,14,0.35)",
        zerolinecolor="rgba(74,14,14,0.2)",
    )
    fig.update_yaxes(
        showgrid=True, gridcolor="rgba(74,14,14,0.10)",
        tickfont=dict(color=_TXT, size=11),
        title_font=dict(color=_TXT, size=12),
        linecolor="rgba(74,14,14,0.35)",
        zerolinecolor="rgba(74,14,14,0.2)",
    )
    fig.update_annotations(font=dict(color=_TXT, size=12))
    return fig


def make_annotated_results_figure(results_df: pd.DataFrame) -> go.Figure:
    """Actuals vs annotated walk-forward forecasts (last N months)."""
    if results_df is None or results_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No annotated results", showarrow=False)
        return _style(fig, "Annotated Walk-Forward")

    df = results_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    x = [p2s(d) for d in df["Date"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=df["Actuals"], mode="lines+markers", name="Actuals",
        line=dict(color=_TXT, width=2.5), marker=dict(size=8),
    ))
    for col in df.columns:
        if not str(col).endswith(" Forecast"):
            continue
        fig.add_trace(go.Scatter(
            x=x, y=df[col], mode="lines+markers", name=col.replace(" Forecast", ""),
            line=dict(width=2), marker=dict(size=7, symbol="diamond"),
        ))
    return _style(fig, "Annotated Last-N-Month Walk-Forward")


# ---------------------------------------------------------------------------
# Per-part Plotly figure (3 x 2 subplot)
# ---------------------------------------------------------------------------


def make_part_figure(r: dict) -> go.Figure:
    part    = r["part"]
    monthly = r["monthly"]
    hist_x  = [p2s(p) for p in monthly["period"]]
    fut_x   = [p2s(p) for p in r["future_periods"]]
    primary = r.get("best_forecast", r["ensemble_raw"])
    baseline = r.get("best_forecast_raw", primary)
    has_cm = bool(r.get("has_cm"))

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f"Claims: Historical + Forecast"
            + (" (CM-adjusted)" if has_cm else ""),
            "Model-by-Model Forecasts",
            "Claim Rate (per 1,000 vehicles / month)",
            "Countermeasure Multiplier",
        ),
        vertical_spacing=0.14,
        horizontal_spacing=0.08,
    )

    fig.add_trace(go.Scatter(
        x=hist_x, y=r["claim_vals"], mode="lines+markers", name="Historical",
        line=dict(color=_A1, width=2), marker=dict(size=3),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=fut_x + fut_x[::-1],
        y=list(r["ci_high"]) + list(r["ci_low"][::-1]),
        fill="toself", fillcolor="rgba(255,187,53,0.12)",
        line=dict(color="rgba(0,0,0,0)"), name="CI",
    ), row=1, col=1)
    if has_cm:
        fig.add_trace(go.Scatter(
            x=fut_x, y=baseline, mode="lines+markers",
            name="Baseline (pre-CM)",
            line=dict(color="#94A3B8", width=2, dash="dash"),
            marker=dict(size=5),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=fut_x, y=primary, mode="lines+markers",
            name="CM-adjusted forecast",
            line=dict(color=_A2, width=3),
            marker=dict(size=8, symbol="diamond"),
        ), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(
            x=fut_x, y=primary, mode="lines+markers",
            name=f"Best: {r.get('best_model', 'Model')}",
            line=dict(color=_A3, width=3, dash="dot"),
            marker=dict(size=7, symbol="diamond"),
        ), row=1, col=1)

    for mn, fc in r["forecasts_raw"].items():
        fig.add_trace(go.Scatter(
            x=fut_x, y=fc, mode="lines+markers", name=mn,
            line=dict(color=MODEL_COLORS.get(mn, "#999"), width=1.5),
            marker=dict(size=4),
        ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=hist_x, y=r["hist_rate"], mode="lines", name="Historical Rate",
        line=dict(color=_A1, width=2),
        fill="tozeroy", fillcolor="rgba(108,99,255,0.18)",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=fut_x, y=r["forecast_rate"], mode="lines", name="Forecast Rate",
        line=dict(color=_A4, width=2, dash="dot"),
        fill="tozeroy", fillcolor="rgba(255,101,132,0.12)",
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=fut_x, y=r["cm_mults"], mode="lines+markers", name="CM Factor",
        line=dict(color=_A2, width=2), marker=dict(size=6),
        fill="tozeroy", fillcolor="rgba(67,217,173,0.12)",
    ), row=2, col=2)
    fig.add_hline(y=1.0, line_dash="dash", line_color=_A4,
                  annotation_text="No CM", row=2, col=2)

    fig.update_layout(
        height=720, paper_bgcolor="#FFFFFF", plot_bgcolor=_BG1,
        font=dict(color=_TXT, size=11, family="DM Sans, Inter, sans-serif"),
        title_text=f"<b>Warranty Forecast — {part}</b>",
        title_font=dict(size=20, color=_TXT),
        showlegend=True,
        legend=dict(
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="rgba(0,0,0,0.15)",
            font=dict(size=10, color=_TXT),
        ),
        margin=dict(l=40, r=40, t=80, b=40),
    )
    fig.update_xaxes(showgrid=False, tickangle=-30, tickfont=dict(color=_TXT))
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)", tickfont=dict(color=_TXT))
    fig.update_annotations(font=dict(color=_TXT, size=11))
    return fig


# ---------------------------------------------------------------------------
# New analysis figures
# ---------------------------------------------------------------------------

def make_historical_claims_figure(r: dict) -> go.Figure:
    monthly = r["monthly"]
    fig = go.Figure(go.Scatter(
        x=[p2s(p) for p in monthly["period"]], y=r["claim_vals"],
        mode="lines+markers", name="Claims",
        line=dict(color=_A1, width=2),
        fill="tozeroy", fillcolor="rgba(108,99,255,0.15)",
    ))
    return _style(fig, f"Historical Claims Trend â€” {r['part']}")


def make_production_figure(r: dict) -> go.Figure:
    monthly = r["monthly"]
    prod = r.get("production", monthly.get("production", []))
    fig = go.Figure(go.Scatter(
        x=[p2s(p) for p in monthly["period"]], y=prod,
        mode="lines+markers", name="Production",
        line=dict(color=_A2, width=2),
        fill="tozeroy", fillcolor="rgba(67,217,173,0.12)",
    ))
    return _style(fig, f"Production Trend â€” {r['part']}")


def make_fcok_heatmap(raw, part: str) -> go.Figure:
    mat = build_fcok_process_matrix(raw, part) if raw is not None else pd.DataFrame()
    if mat.empty:
        fig = go.Figure()
        fig.add_annotation(text="No FCO/K Ã— Process data", showarrow=False)
        return _style(fig, f"Manufacturing vs Claim Month â€” {part}")

    # Limit to most active FCOK months for readability
    top = mat.sum(axis=1).nlargest(min(24, len(mat))).index
    mat = mat.loc[top]
    fig = go.Figure(go.Heatmap(
        z=mat.values,
        x=[str(c) for c in mat.columns],
        y=[str(i) for i in mat.index],
        colorscale="Viridis",
        colorbar=dict(title="Claims"),
    ))
    return _style(fig, f"Manufacturing Month vs Claim Month â€” {part}", height=480)


def make_vehicle_age_figure(raw, part: str) -> go.Figure:
    if raw is None or "VEHICLE_AGE_MONTHS" not in getattr(raw, "columns", []):
        fig = go.Figure()
        return _style(fig, f"Vehicle Age Distribution â€” {part}")
    ages = raw.loc[raw["Part Name"] == part, "VEHICLE_AGE_MONTHS"].dropna()
    fig = go.Figure(go.Histogram(
        x=ages, nbinsx=36, marker_color=_A3, opacity=0.85, name="Vehicle Age",
    ))
    fig.add_vline(x=WARRANTY_MONTHS, line_dash="dash", line_color=_A4,
                  annotation_text="3-yr warranty")
    return _style(fig, f"Vehicle Age Distribution â€” {part}")


def make_odometer_figure(raw, part: str) -> go.Figure:
    if raw is None or "ODOMETER" not in getattr(raw, "columns", []):
        fig = go.Figure()
        return _style(fig, f"Odometer Distribution â€” {part}")
    od = raw.loc[raw["Part Name"] == part, "ODOMETER"].dropna()
    # Cap extreme outliers for display
    cap = od.quantile(0.99) if len(od) else 0
    od = od[od <= cap]
    fig = go.Figure(go.Histogram(
        x=od, nbinsx=40, marker_color=_A1, opacity=0.85, name="Odometer",
    ))
    return _style(fig, f"Odometer Distribution â€” {part}")


def make_actual_vs_predicted_figure(r: dict) -> go.Figure:
    actual = r.get("oos_actual", np.array([]))
    preds = r.get("oos_preds", {})
    best = r.get("best_model")
    fig = go.Figure()
    if len(actual) == 0:
        fig.add_annotation(text="No walk-forward predictions", showarrow=False)
        return _style(fig, f"Actual vs Predicted â€” {r['part']}")

    x = list(range(1, len(actual) + 1))
    fig.add_trace(go.Scatter(
        x=x, y=actual, mode="lines+markers", name="Actual",
        line=dict(color=_TXT, width=3), marker=dict(size=8),
    ))
    for mn, yp in preds.items():
        if len(yp) != len(actual):
            continue
        width = 3 if mn == best else 1.5
        dash = "solid" if mn == best else "dot"
        fig.add_trace(go.Scatter(
            x=x, y=yp, mode="lines+markers", name=mn,
            line=dict(color=MODEL_COLORS.get(mn, "#999"), width=width, dash=dash),
            marker=dict(size=5),
        ))
    return _style(fig, f"Actual vs Predicted (Walk-Forward) — {r['part']}")


def make_cv_fold_figure(r: dict) -> go.Figure:
    cv_folds = r.get("cv_mae_folds", {})
    fig = go.Figure()
    for mn, fold_maes in cv_folds.items():
        if not fold_maes:
            continue
        fig.add_trace(go.Scatter(
            x=list(range(1, len(fold_maes) + 1)), y=fold_maes,
            mode="lines+markers", name=mn,
            line=dict(color=MODEL_COLORS.get(mn, "#999"), width=2),
            marker=dict(size=6),
        ))
    return _style(fig, "Rolling Forecast Performance (CV MAE per Fold)")


def make_model_comparison_figure(r: dict) -> go.Figure:
    ranking = r.get("ranking_df")
    if ranking is None or ranking.empty:
        fig = go.Figure()
        return _style(fig, f"Model Comparison â€” {r['part']}")

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Error Metrics", "RÂ²"),
                        column_widths=[0.65, 0.35])
    for metric, color in [("RMSE", _A4), ("MAE", _A3), ("MAPE", _A1)]:
        fig.add_trace(go.Bar(
            name=metric, x=ranking["Model"], y=ranking[metric],
            marker_color=color, opacity=0.85,
        ), row=1, col=1)
    fig.add_trace(go.Bar(
        name="RÂ²", x=ranking["Model"], y=ranking["R2"],
        marker_color=_A2, showlegend=False,
    ), row=1, col=2)
    fig.update_layout(
        barmode="group", height=420, paper_bgcolor="#F8F4F2", plot_bgcolor=_BG1,
        font=dict(color=_TXT, size=11, family="DM Sans, Inter, sans-serif"),
        title_text=f"<b>Model Comparison Dashboard — {r['part']}</b>",
        title_font=dict(size=15, color=_TXT),
        legend=dict(bgcolor="rgba(255,255,255,0.95)", font=dict(size=10, color=_TXT)),
        margin=dict(l=40, r=40, t=70, b=60),
    )
    fig.update_xaxes(tickfont=dict(color=_TXT), title_font=dict(color=_TXT))
    fig.update_yaxes(tickfont=dict(color=_TXT), title_font=dict(color=_TXT),
                     gridcolor="rgba(74,14,14,0.10)")
    fig.update_annotations(font=dict(color=_TXT))
    return fig


def make_12m_forecast_figure(r: dict) -> go.Figure:
    fut_x = [p2s(p) for p in r["future_periods"]]
    primary = r.get("best_forecast", r["ensemble_raw"])
    baseline = r.get("best_forecast_raw", primary)
    has_cm = bool(r.get("has_cm"))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fut_x + fut_x[::-1],
        y=list(r["ci_high"]) + list(r["ci_low"][::-1]),
        fill="toself", fillcolor="rgba(255,187,53,0.12)",
        line=dict(color="rgba(0,0,0,0)"), name="CI",
    ))
    if has_cm:
        fig.add_trace(go.Bar(
            x=fut_x, y=baseline, name="Baseline (pre-CM)",
            marker_color="#94A3B8", opacity=0.55,
        ))
        fig.add_trace(go.Bar(
            x=fut_x, y=primary, name="CM-adjusted forecast",
            marker_color=_A2, opacity=0.9,
        ))
        cms = r.get("cm_sim") or {}
        if cms.get("monthly_reduction") is not None:
            fig.add_trace(go.Scatter(
                x=fut_x, y=cms["monthly_reduction"],
                mode="lines+markers", name="Monthly CM reduction",
                line=dict(color=_A4, width=2),
                marker=dict(size=6),
            ))
        title = f"12-Month Forecast (CM-adjusted) — {r['part']}"
    else:
        fig.add_trace(go.Bar(
            x=fut_x, y=primary, name=f"Best ({r.get('best_model')})",
            marker_color=_A3, opacity=0.85,
        ))
        title = f"12-Month Forecast — {r['part']}"
    fig.update_layout(barmode="group")
    return _style(fig, title)


def make_countermeasure_impact_figure(sim: dict) -> go.Figure:
    x = list(range(1, len(sim["original"]) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=sim["original"], mode="lines+markers", name="Original Forecast",
        line=dict(color=_A4, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=sim["adjusted"], mode="lines+markers", name="Adjusted Forecast",
        line=dict(color=_A2, width=2),
    ))
    fig.add_trace(go.Bar(
        x=x, y=sim["monthly_reduction"], name="Monthly Reduction",
        marker_color=_A3, opacity=0.7,
    ))
    fig.add_trace(go.Scatter(
        x=x, y=sim["cumulative_reduction"], mode="lines+markers",
        name="Cumulative Reduction", line=dict(color=_A1, width=2, dash="dot"),
        yaxis="y2",
    ))
    fig.update_layout(
        **_LAYOUT,
        title_text="<b>Countermeasure Impact Analysis</b>",
        title_font=dict(size=15, color=_TXT),
        height=460,
        yaxis=dict(
            title="Claims", showgrid=True, gridcolor="rgba(74,14,14,0.10)",
            tickfont=dict(color=_TXT), title_font=dict(color=_TXT),
        ),
        yaxis2=dict(
            title="Cumulative Reduction", overlaying="y", side="right",
            showgrid=False, tickfont=dict(color=_TXT), title_font=dict(color=_TXT),
        ),
        margin=dict(l=40, r=60, t=60, b=40),
        font=dict(color=_TXT, size=11, family="DM Sans, Inter, sans-serif"),
        legend=dict(bgcolor="rgba(255,255,255,0.95)", font=dict(size=10, color=_TXT)),
    )
    return fig


def make_baseline_vs_adjusted_figure(sim: dict, future_periods) -> go.Figure:
    x = [p2s(p) for p in future_periods]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Original", x=x, y=sim["original"], marker_color=_A4, opacity=0.75))
    fig.add_trace(go.Bar(name="Adjusted", x=x, y=sim["adjusted"], marker_color=_A2, opacity=0.85))
    return _style(fig, "Baseline vs Adjusted Forecast")


# ---------------------------------------------------------------------------
# NEW: Countermeasure Engine Analysis Figures
# ---------------------------------------------------------------------------

def make_cm_analysis_figure(cm_analysis: dict) -> go.Figure:
    """
    4-panel CM analysis chart:
      Top-left  : With vs Without CM forecast (lines)
      Top-right : Monthly claim reduction (bar)
      Bottom-left: Cumulative claim reduction (area)
      Bottom-right: CM-adjusted future production trajectory
    """
    if not cm_analysis or not cm_analysis.get("cm_active"):
        # CM inactive — show a simple info placeholder
        fig = go.Figure()
        fig.add_annotation(
            text="No countermeasure active — enable CM to view analysis.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color=_MUTED),
        )
        return _style(fig, "CM Analysis (Inactive)")

    comp = cm_analysis.get("comparison", {})
    baseline = np.asarray(comp.get("baseline", []), dtype=float)
    cm_fc = np.asarray(comp.get("cm_forecast", []), dtype=float)
    monthly_red = np.asarray(comp.get("monthly_reduction", []), dtype=float)
    cum_red = np.asarray(comp.get("cumulative_reduction", []), dtype=float)
    cm_prod = np.asarray(cm_analysis.get("cm_production", []), dtype=float)
    comp_df = comp.get("comparison_df", pd.DataFrame())

    months = (
        comp_df["Month"].tolist()
        if "Month" in comp_df.columns
        else [str(i + 1) for i in range(len(baseline))]
    )
    H = len(months)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "With vs Without CM — Claims Forecast",
            "Monthly Claim Reduction",
            "Cumulative Reduction",
            "CM-Adjusted Future Production",
        ),
        vertical_spacing=0.14,
        horizontal_spacing=0.10,
    )

    # ── Panel 1: With vs Without ─────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=months, y=baseline[:H],
        name="Without CM", mode="lines+markers",
        line=dict(color=_A4, width=2.5),
        marker=dict(size=6),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=months, y=cm_fc[:H],
        name="With CM", mode="lines+markers",
        line=dict(color=_A2, width=2.5, dash="dash"),
        marker=dict(size=6, symbol="diamond"),
        fill="tonexty",
        fillcolor="rgba(4,120,87,0.10)",
    ), row=1, col=1)

    # ── Panel 2: Monthly reduction bar ───────────────────────────────────
    pct_red = comp.get("pct_reduction", np.zeros(H))
    if not isinstance(pct_red, np.ndarray):
        pct_red = np.asarray(pct_red, dtype=float)
    fig.add_trace(go.Bar(
        x=months, y=monthly_red[:H],
        name="Monthly Reduction",
        marker_color=_A3,
        opacity=0.80,
        text=[f"{p:.1f}%" for p in pct_red[:H]],
        textposition="outside",
        textfont=dict(size=9, color=_TXT),
    ), row=1, col=2)

    # ── Panel 3: Cumulative reduction area ───────────────────────────────
    fig.add_trace(go.Scatter(
        x=months, y=cum_red[:H],
        name="Cumulative Reduction",
        mode="lines",
        line=dict(color=_A1, width=2.5),
        fill="tozeroy",
        fillcolor="rgba(29,78,216,0.12)",
    ), row=2, col=1)

    # ── Panel 4: Production trajectory ───────────────────────────────────
    if len(cm_prod) > 0:
        avg_pp = float(cm_analysis.get("avg_peak_prod", 0))
        fig.add_trace(go.Scatter(
            x=months[:len(cm_prod)], y=cm_prod[:H],
            name="CM Production", mode="lines+markers",
            line=dict(color="#7C3AED", width=2),
            marker=dict(size=5),
        ), row=2, col=2)
        if avg_pp > 0:
            fig.add_hline(
                y=avg_pp, line_dash="dot",
                line_color=_A4, line_width=1.5,
                annotation_text=f"Avg Peak Prod: {avg_pp:,.0f}",
                annotation_font_color=_A4,
                annotation_font_size=9,
                row=2, col=2,
            )

    fig.update_layout(
        **_LAYOUT,
        title_text="<b>Countermeasure Engine Analysis</b>",
        title_font=dict(size=15, color=_TXT),
        height=600,
        margin=dict(l=50, r=30, t=80, b=50),
        font=dict(color=_TXT, size=10, family="DM Sans, Inter, sans-serif"),
        legend=dict(
            bgcolor="rgba(255,255,255,0.95)",
            font=dict(size=9, color=_TXT),
            orientation="h", y=-0.07,
        ),
        showlegend=True,
    )
    for ax in fig.layout:
        if ax.startswith("xaxis") or ax.startswith("yaxis"):
            fig.layout[ax].update(
                tickfont=dict(color=_TXT, size=9),
                title_font=dict(color=_TXT, size=10),
                gridcolor="rgba(0,0,0,0.06)",
                showgrid=True,
            )
    return fig


def make_cm_production_figure(cm_analysis: dict) -> go.Figure:
    """
    Line chart showing how future production evolves after the CM:
    avg_peak_prod (flat baseline), adj_prod start, and the declining CM
    production sequence over the warranty window.
    """
    if not cm_analysis or not cm_analysis.get("cm_active"):
        fig = go.Figure()
        fig.add_annotation(
            text="CM not active.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=13, color=_MUTED),
        )
        return _style(fig, "CM Production Trajectory (Inactive)")

    cm_prod = np.asarray(cm_analysis.get("cm_production", []), dtype=float)
    avg_pp = float(cm_analysis.get("avg_peak_prod", 0))
    adj_start = float(cm_analysis.get("adj_prod", 0))
    comp_df = cm_analysis.get("comparison", {}).get("comparison_df", pd.DataFrame())
    months = (
        comp_df["Month"].tolist()
        if "Month" in comp_df.columns and len(comp_df) == len(cm_prod)
        else [f"M+{i + 1}" for i in range(len(cm_prod))]
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=months, y=cm_prod,
        name="CM-Adjusted Production",
        mode="lines+markers",
        line=dict(color="#7C3AED", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(124,58,237,0.08)",
    ))
    if avg_pp > 0:
        fig.add_hline(
            y=avg_pp, line_dash="dot", line_color=_A4, line_width=1.5,
            annotation_text=f"Avg Peak Production: {avg_pp:,.0f}",
            annotation_font_color=_A4, annotation_font_size=10,
        )
    if adj_start > 0:
        fig.add_hline(
            y=adj_start, line_dash="dashdot", line_color=_A2, line_width=1.5,
            annotation_text=f"Adjusted Baseline: {adj_start:,.0f}",
            annotation_font_color=_A2, annotation_font_size=10,
        )
    return _style(fig, "CM-Adjusted Future Production Trajectory", height=380)


def make_cm_savings_html(cm_analysis: dict) -> str:
    """
    Returns an HTML savings summary card for the CM analysis panel.
    Shows: total claim reduction, % reduction, estimated cost savings.
    """
    if not cm_analysis or not cm_analysis.get("cm_active"):
        return (
            "<div style='padding:16px;background:#f8f9fa;border-radius:8px;"
            "border-left:4px solid #9CA3AF;font-family:Inter,sans-serif;'>"
            "<b style='color:#6B7280'>No countermeasure active.</b>"
            " Enable CM to see savings analysis.</div>"
        )

    comp = cm_analysis.get("comparison", {})
    total_base = comp.get("total_baseline_claims", 0)
    total_cm = comp.get("total_cm_claims", 0)
    total_red = comp.get("total_reduction", 0)
    total_pct = comp.get("total_pct_reduction", 0)
    cost_sav = comp.get("total_cost_savings")
    avg_pp = cm_analysis.get("avg_peak_prod", 0)
    avg_pc = cm_analysis.get("avg_peak_claims", 0)
    adj_prod = cm_analysis.get("adj_prod", 0)
    peak_df = cm_analysis.get("peak_fcok_df", pd.DataFrame())
    factor = cm_analysis.get("factor", 1.0)
    msg = cm_analysis.get("message", "")

    peak_rows = ""
    if not peak_df.empty:
        for _, row in peak_df.iterrows():
            peak_rows += (
                f"<tr><td style='padding:3px 8px'>{row['rank']}</td>"
                f"<td style='padding:3px 8px'>{row['fcok_month']}</td>"
                f"<td style='padding:3px 8px'>{int(row['claim_count']):,}</td></tr>"
            )

    cost_html = ""
    if cost_sav is not None and np.isfinite(cost_sav):
        cost_html = (
            f"<div style='margin-top:10px;padding:8px 12px;"
            f"background:#ECFDF5;border-radius:6px;border-left:3px solid #047857;'>"
            f"<b style='color:#047857'>Estimated Warranty Cost Savings:</b> "
            f"<span style='font-size:1.15em;font-weight:700;color:#047857'>"
            f"{cost_sav:,.2f}</span></div>"
        )

    peak_table = ""
    if peak_rows:
        peak_table = (
            "<table style='border-collapse:collapse;font-size:0.9em;margin-top:6px;'>"
            "<thead><tr style='background:#E0E7FF'>"
            "<th style='padding:4px 8px'>Rank</th>"
            "<th style='padding:4px 8px'>FCOK Month</th>"
            "<th style='padding:4px 8px'>Claims</th></tr></thead>"
            f"<tbody>{peak_rows}</tbody></table>"
        )

    return f"""
<div style='padding:18px;background:#FFFFFF;border-radius:10px;
            border:1px solid #E2E8F0;font-family:Inter,sans-serif;
            box-shadow:0 2px 8px rgba(0,0,0,0.06);'>
  <h3 style='margin:0 0 12px 0;color:#1D4ED8;font-size:1.05em;'>
    🔧 Countermeasure Engine — Analysis Summary
  </h3>
  <div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px;'>
    <div style='padding:10px;background:#EFF6FF;border-radius:6px;text-align:center;'>
      <div style='font-size:0.78em;color:#6B7280'>Baseline Claims (12m)</div>
      <div style='font-size:1.3em;font-weight:700;color:#1D4ED8'>{total_base:,.0f}</div>
    </div>
    <div style='padding:10px;background:#ECFDF5;border-radius:6px;text-align:center;'>
      <div style='font-size:0.78em;color:#6B7280'>CM Claims (12m)</div>
      <div style='font-size:1.3em;font-weight:700;color:#047857'>{total_cm:,.0f}</div>
    </div>
    <div style='padding:10px;background:#FFF7ED;border-radius:6px;text-align:center;'>
      <div style='font-size:0.78em;color:#6B7280'>Total Reduction</div>
      <div style='font-size:1.3em;font-weight:700;color:#B45309'>{total_red:,.0f}</div>
    </div>
    <div style='padding:10px;background:#FEF2F2;border-radius:6px;text-align:center;'>
      <div style='font-size:0.78em;color:#6B7280'>% Reduction</div>
      <div style='font-size:1.3em;font-weight:700;color:#B91C1C'>{total_pct:.1f}%</div>
    </div>
  </div>
  <div style='margin-bottom:10px;font-size:0.88em;'>
    <b>Production Baseline:</b> Avg Peak Prod = {avg_pp:,.0f} &nbsp;|&nbsp;
    Avg Peak Claims = {avg_pc:,.1f} &nbsp;|&nbsp;
    <b>Adj. Prod = {adj_prod:,.0f}</b> &nbsp;|&nbsp;
    Factor = <b>{factor:.2f}</b>
  </div>
  {peak_table}
  {cost_html}
  <div style='margin-top:10px;font-size:0.82em;color:#6B7280;font-style:italic;'>
    {msg}
  </div>
</div>
"""



def make_overview_figure(results: list[dict]) -> go.Figure:
    fig = go.Figure()
    for i, r in enumerate(results):
        color = PALETTE[i % len(PALETTE)]
        monthly = r["monthly"]
        hist_x = [p2s(p) for p in monthly["period"]]
        fut_x = [p2s(p) for p in r["future_periods"]]
        primary = r.get("best_forecast", r["ensemble_raw"])
        fig.add_trace(go.Scatter(
            x=hist_x, y=r["claim_vals"], mode="lines",
            name=f"{r['part']} (hist)", line=dict(color=color, width=1.5),
            legendgroup=r["part"],
        ))
        fig.add_trace(go.Scatter(
            x=fut_x, y=primary, mode="lines",
            name=f"{r['part']} (fcst)", line=dict(color=color, width=2.5, dash="dot"),
            legendgroup=r["part"],
        ))
    return _style(fig, "All Parts â€” Historical & Forecast Overlay", height=480)


def make_mae_figure(results: list[dict]) -> go.Figure:
    fig = go.Figure()
    part_names = [r["part"] for r in results]
    for mn in MODEL_NAMES:
        maes = [r["val_mae"].get(mn, 0) for r in results]
        fig.add_trace(go.Bar(
            name=mn, x=part_names, y=maes,
            marker_color=MODEL_COLORS.get(mn, "#888"),
        ))
    fig.update_layout(
        **_LAYOUT, barmode="group", height=420,
        title_text="<b>Validation MAE — Model × Part (lower = better)</b>",
        title_font=dict(size=16, color=_TXT),
        xaxis_title="Part", yaxis_title="Validation MAE (scaled)",
        xaxis=dict(tickangle=-30, tickfont=dict(color=_TXT), title_font=dict(color=_TXT)),
        yaxis=dict(tickfont=dict(color=_TXT), title_font=dict(color=_TXT)),
        legend=dict(bgcolor="rgba(255,255,255,0.95)", font=dict(size=10, color=_TXT)),
        margin=dict(l=40, r=40, t=70, b=80),
        font=dict(color=_TXT, size=11, family="DM Sans, Inter, sans-serif"),
    )
    return fig


def make_summary_figures(results: list[dict]) -> tuple:
    donut_fig = go.Figure(go.Pie(
        labels=[r["part"] for r in results],
        values=[r["claim_vals"].sum() for r in results],
        hole=0.55, textinfo="label+percent",
        marker=dict(colors=PALETTE * 5), textfont_size=10,
    ))
    donut_fig.update_layout(
        title_text="Total Historical Claims by Part", title_font_size=16,
        height=420, margin=dict(l=20, r=20, t=60, b=20), **_LAYOUT,
    )

    bar_fig = go.Figure(go.Bar(
        x=[r["part"] for r in results],
        y=[r.get("best_forecast", r["ensemble_raw"]).sum() for r in results],
        marker_color=PALETTE[:len(results)],
        text=[f"{r.get('best_forecast', r['ensemble_raw']).sum():.0f}" for r in results],
        textposition="outside",
    ))
    bar_fig.update_layout(
        title_text="Forecasted Claims â€” Next 12 Months (Best Model)", title_font_size=16,
        xaxis_title="Part", yaxis_title="Count",
        height=420, margin=dict(l=20, r=20, t=60, b=80),
        xaxis=dict(tickangle=-30), **_LAYOUT,
    )

    model_names = list(MODEL_COLORS.keys())
    # Placeholder empty figure kept for API compatibility (ensemble weights removed)
    heat_fig = go.Figure()
    heat_fig.update_layout(
        title_text="Model ranking is shown in the ranking table",
        title_font_size=16,
        height=380,
        margin=dict(l=40, r=20, t=60, b=40),
        **_LAYOUT,
        annotations=[dict(
            text="Ensemble weight heatmap removed",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color=_TXT),
        )],
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return donut_fig, bar_fig, heat_fig


# ---------------------------------------------------------------------------
# DataFrame builders
# ---------------------------------------------------------------------------

def make_best_params_df(r: dict) -> pd.DataFrame:
    rows = []
    bp = r.get("best_params", {})
    cm = r.get("val_mae", {})
    for model, params in bp.items():
        param_str = ", ".join(f"{k}={v}" for k, v in params.items()) if params else "default"
        rows.append({
            "Model": model,
            "Best HP Config": param_str,
            "Mean CV MAE": round(cm.get(model, float("nan")), 6),
        })
    return pd.DataFrame(rows)


def make_forecast_df(r: dict) -> pd.DataFrame:
    primary = r.get("best_forecast", r["ensemble_raw"])
    raw_fc = r.get("best_forecast_raw", primary)
    fe = r.get("forecast_economics") or {}
    fut_ratio = fe.get("forecast_claim_ratio_per_1k")
    fut_cost = fe.get("forecast_claim_cost")
    fut_prod = fe.get("future_production")
    rows = []
    for i, fp in enumerate(r["future_periods"]):
        mid = int(round(float(primary[i])))
        lo = max(0, int(round(float(r["ci_low"][i]))))
        hi = int(round(float(r["ci_high"][i])))
        row = {
            "Month": p2s(fp),
            "Best Model Forecast": mid,
            "Pre-CM Forecast": int(round(float(raw_fc[i]))) if raw_fc is not None else mid,
            "CI Low": lo,
            "CI High": hi,
            "Rate / 1k veh": round(float(r["forecast_rate"][i]), 2),
            "CM Factor": f"{float(r['cm_mults'][i]) * 100:.0f}%" if r.get("has_cm") else "—",
            "Selected Model": r.get("best_model", "—"),
        }
        if fut_prod is not None and i < len(fut_prod):
            row["Future Production"] = round(float(fut_prod[i]), 1)
        if fut_ratio is not None and i < len(fut_ratio):
            v = fut_ratio[i]
            row["Claim Ratio / 1k"] = round(float(v), 3) if np.isfinite(v) else None
        if fut_cost is not None and i < len(fut_cost):
            v = fut_cost[i]
            row["Est. Claim Cost"] = round(float(v), 2) if np.isfinite(v) else None
        rows.append(row)
    return pd.DataFrame(rows)


def make_dq_df(results: list[dict], raw) -> pd.DataFrame:
    if raw is None or not len(raw):
        return pd.DataFrame()
    rows = []
    for part in [r["part"] for r in results]:
        sub = raw[raw["Part Name"] == part]
        n_rec = len(sub)
        miss_od = int(sub["ODOMETER"].isna().sum()) if "ODOMETER" in sub.columns else 0
        miss_pct = f"{miss_od / max(n_rec, 1) * 100:.1f}%"
        if "PROCESSING_DATE" in sub.columns and sub["PROCESSING_DATE"].notna().any():
            d_min = sub["PROCESSING_DATE"].min()
            d_max = sub["PROCESSING_DATE"].max()
            d_range = f"{d_min.strftime('%Y-%m')} -> {d_max.strftime('%Y-%m')}"
            n_months = (d_max.year - d_min.year) * 12 + (d_max.month - d_min.month) + 1
        else:
            d_range, n_months = "N/A", 0
        rows.append({
            "Part": part,
            "Records": n_rec,
            "Date Range (Processing)": d_range,
            "Months Covered": n_months,
            "Missing Odometer": f"{miss_od:,} ({miss_pct})",
            "Avg Claims / Month": round(n_rec / max(n_months, 1), 1),
        })
    return pd.DataFrame(rows)


def make_summary_df(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        hist_total = r["claim_vals"].sum()
        primary = r.get("best_forecast", r["ensemble_raw"])
        fc_total = primary.sum()
        pct_chg = (fc_total - hist_total) / (hist_total + 1e-9) * 100
        rows.append({
            "Part": r["part"],
            "Hist. Claims": int(hist_total),
            "Forecast Claims": int(round(fc_total)),
            "Delta vs History": f"{pct_chg:+.1f}%",
            "Peak Forecast Mo.": p2s(r["future_periods"][int(primary.argmax())]),
            "Peak Claims": int(round(primary.max())),
            "Best Model": r.get("best_model", max(r["weights"], key=r["weights"].get)),
            "CM Active": "Yes" if r["has_cm"] else "No",
        })
    return pd.DataFrame(rows)


def make_global_ranking_df(results: list[dict]) -> pd.DataFrame:
    """Average metrics across parts for a global model leaderboard."""
    rows = []
    for mn in MODEL_NAMES:
        rmses, maes, mapes, r2s = [], [], [], []
        for r in results:
            m = r.get("metrics_by_model", {}).get(mn, {})
            if np.isfinite(m.get("RMSE", np.nan)):
                rmses.append(m["RMSE"])
            if np.isfinite(m.get("MAE", np.nan)):
                maes.append(m["MAE"])
            if np.isfinite(m.get("MAPE", np.nan)):
                mapes.append(m["MAPE"])
            if np.isfinite(m.get("R2", np.nan)):
                r2s.append(m["R2"])
        rows.append({
            "Model": mn,
            "Avg RMSE": round(float(np.mean(rmses)), 4) if rmses else np.nan,
            "Avg MAE": round(float(np.mean(maes)), 4) if maes else np.nan,
            "Avg MAPE": round(float(np.mean(mapes)), 4) if mapes else np.nan,
            "Avg RÂ²": round(float(np.mean(r2s)), 4) if r2s else np.nan,
            "Times Selected Best": sum(1 for r in results if r.get("best_model") == mn),
        })
    df = pd.DataFrame(rows).sort_values("Avg RMSE", ascending=True).reset_index(drop=True)
    df.insert(0, "Rank", np.arange(1, len(df) + 1))
    return df


def make_kpi_html(results: list[dict]) -> str:
    total_hist = sum(r["claim_vals"].sum() for r in results)
    total_fc = sum(r.get("best_forecast", r["ensemble_raw"]).sum() for r in results)
    pct_chg = (total_fc - total_hist) / (total_hist + 1e-9) * 100
    n_cm = sum(1 for r in results if r["has_cm"])
    peak_part = max(results, key=lambda r: r.get("best_forecast", r["ensemble_raw"]).sum())["part"]
    # Most frequently selected best model
    from collections import Counter
    best_counts = Counter(r.get("best_model", "Ensemble") for r in results)
    top_model = best_counts.most_common(1)[0][0] if best_counts else "â€”"
    avg_rate = sum(r["forecast_rate"].mean() for r in results) / max(len(results), 1)
    trend_color = _A4 if pct_chg > 0 else _A2
    trend_icon = "&#128200;" if pct_chg > 0 else "&#128201;"

    cards = [
        ("📦", f"{int(total_hist):,}", "Historical Claims", _A1),
        ("🔮", f"{int(total_fc):,}", "Forecasted (12 mo)", _A2),
        (trend_icon, f"{pct_chg:+.1f}%", "Trend vs History", trend_color),
        ("🏆", top_model, "Top Auto-Selected Model", _A3),
        ("🧩", str(len(results)), "Parts Analysed", _A1),
        ("🏭", f"{PRODUCTION_PER_MONTH:,}", "Prod. Start / Month", _A2),
        ("📈", peak_part, "Highest Forecast Part", _A4),
        ("📉", f"{avg_rate:.2f}/1k", "Avg Forecast Rate", "#0284C7"),
    ]
    items_html = ""
    for icon, value, label, color in cards:
        items_html += f"""
<div style="background:{_BG1};border:2px solid rgba(15,23,42,0.12);
            border-left:5px solid {color};border-radius:14px;padding:18px 14px;
            text-align:center;box-shadow:0 8px 20px rgba(15,23,42,0.08);
            font-family:'Inter',sans-serif;">
  <div style="font-size:22px;margin-bottom:6px;">{icon}</div>
  <div style="font-size:24px;font-weight:800;color:#000000;letter-spacing:-0.02em;">{value}</div>
  <div style="font-size:12px;color:#000000;margin-top:6px;font-weight:600;
              text-transform:uppercase;letter-spacing:.4px;">{label}</div>
</div>"""
    return f"""
<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));
            gap:14px;margin-bottom:12px;">{items_html}</div>"""


# ---------------------------------------------------------------------------
# Gradio custom CSS (preserved)
# ---------------------------------------------------------------------------

_CUSTOM_CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&display=swap');

/* ═══════════════════════════════════════════════════════════
   CUSTOM DESIGN TOKENS
   ═══════════════════════════════════════════════════════════ */
:root {{
  /* App tokens */
  --aw-bg: {_BG0};
  --aw-card: {_BG1};
  --aw-ink: {_TXT};
  --aw-muted: #475569;
  --aw-blue: #3B82F6;
  --aw-blue-light: #BFDBFE;
  --aw-green: #10B981;
  --aw-green-light: #A7F3D0;
  --aw-purple: #8B5CF6;
  --aw-purple-light: #DDD6FE;
  --aw-rose: #F43F5E;
  --aw-rose-light: #FFE4E6;
  --aw-amber: #F59E0B;
  --aw-amber-light: #FEF3C7;
  --aw-soft: #EFF6FF;
  --aw-soft-2: #DBEAFE;
  --aw-accent: #3B82F6;
  --aw-border: rgba(59, 130, 246, 0.18);
  --aw-shadow: rgba(15, 23, 42, 0.06);

  /* ── Gradio internal variable overrides ──
     Gradio's Soft theme uses these variables to paint component
     backgrounds. Setting them to light values prevents dark surfaces
     from appearing without needing per-element selectors. */
  --neutral-50:  #F8FAFC;
  --neutral-100: #F1F5F9;
  --neutral-200: #E2E8F0;
  --neutral-300: #CBD5E1;
  --neutral-400: #94A3B8;
  --neutral-500: #64748B;
  --neutral-600: #475569;
  --neutral-700: #334155;
  --neutral-800: #F0F7FF;   /* ← normally very dark; keep light */
  --neutral-900: #EFF6FF;   /* ← normally almost-black; keep light */
  --neutral-950: #E8EEF7;   /* ← normally pitch-black; keep light */
  --color-background-primary: #FFFFFF;
  --body-background-fill: #EFF6FF;
  --block-background-fill: #FFFFFF;
  --border-color-primary: rgba(59, 130, 246, 0.20);
  --input-background-fill: #FFFFFF;
  --input-background-fill-focus: #EFF6FF;
  --table-even-background-fill: #FFFFFF;
  --table-odd-background-fill: #F0F9FF;
  --table-row-focus: #DBEAFE;
  --checkbox-background-color: #FFFFFF;
  --checkbox-background-color-focus: #EFF6FF;
  --checkbox-background-color-selected: #BFDBFE;
  --color-grey-100: #F1F5F9;
  --color-grey-200: #E2E8F0;
  --shadow-spread: 0px;
  --color-accent: #3B82F6;
  --color-accent-soft: #DBEAFE;
  /* Text colours */
  --body-text-color: #0F172A;
  --body-text-color-subdued: #475569;
  --block-label-text-color: #1E40AF;
  --block-title-text-color: #0F172A;
  --input-placeholder-color: #94A3B8;
}}

/* ═══════════════════════════════════════════════════════════
   SELECTIVE WHITE TEXT — only on dark / black backgrounds
   ═══════════════════════════════════════════════════════════
   These rules fire ONLY when an element carries an inline dark
   background style that we cannot remove (e.g. Gradio internal
   cell rendering).  All other elements keep their dark text.
   ═══════════════════════════════════════════════════════════ */

/* Force light colour-scheme everywhere so no browser auto-darkening */
*, *::before, *::after {{
    color-scheme: light !important;
}}

/* Catch inline black / near-black background styles */
[style*="background-color: rgb(0, 0, 0)"],
[style*="background-color:rgb(0,0,0)"],
[style*="background-color: #000"],
[style*="background-color:#000"],
[style*="background: rgb(0, 0, 0)"],
[style*="background: #000"],
[style*="background:#000"],
[style*="background-color: rgba(0, 0, 0"],
[style*="background-color: #0f172a"],
[style*="background-color: #1e293b"],
[style*="background-color: #111827"],
[style*="background-color: #18181b"],
[style*="background-color: #1f2937"],
[style*="background-color: #27272a"],
[style*="background-color: #0a0a0a"] {{
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}}
[style*="background-color: rgb(0, 0, 0)"] *,
[style*="background-color:#000"] *,
[style*="background-color: #000"] *,
[style*="background: rgb(0, 0, 0)"] *,
[style*="background:#000"] *,
[style*="background: #000"] * {{
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}}

/* Gradio dataframe interactive cell inputs — white text on dark cell bg */
.cell-wrap input,
.cell-wrap textarea,
[data-testid="dataframe"] td .cell-wrap,
[data-testid="dataframe"] td > div {{
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    caret-color: #FFFFFF !important;
}}

/* ── Base ── */
body, .gradio-container, footer {{
    background: linear-gradient(145deg, #E0E7FF 0%, #EFF6FF 35%, #F0FDF4 65%, #FFF7ED 100%) !important;
    font-family: 'DM Sans', Inter, system-ui, sans-serif !important;
    color: {_TXT} !important;
}}
.gradio-container {{
    max-width: 1440px !important;
    margin: 0 auto !important;
    padding: 8px 20px 32px !important;
    position: relative !important;
    z-index: 1 !important;
}}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 8px; height: 8px; }}
::-webkit-scrollbar-track {{ background: #F1F5F9; border-radius: 8px; }}
::-webkit-scrollbar-thumb {{ background: linear-gradient(180deg, #93C5FD, #C4B5FD); border-radius: 8px; }}
::-webkit-scrollbar-thumb:hover {{ background: linear-gradient(180deg, #60A5FA, #A78BFA); }}

/* ── Hero banner ── */
.aw-hero {{
    background: linear-gradient(135deg, #FFFFFF 0%, #EFF6FF 35%, #F0FDF4 65%, #FFF7ED 100%) !important;
    border-radius: 22px;
    padding: 28px 30px 24px;
    margin: 8px 0 18px;
    color: {_TXT} !important;
    box-shadow: 0 12px 36px rgba(59, 130, 246, 0.10), 0 2px 8px rgba(139, 92, 246, 0.07);
    border: 1px solid rgba(59, 130, 246, 0.15);
    position: relative;
    z-index: 1;
    overflow: hidden;
}}
.aw-hero::before {{
    content: "";
    position: absolute;
    left: -60px; bottom: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(16, 185, 129, 0.10), transparent 70%);
    pointer-events: none;
}}
.aw-hero::after {{
    content: "";
    position: absolute;
    right: -40px; top: -40px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(59, 130, 246, 0.12), transparent 70%);
    pointer-events: none;
}}
.aw-hero h1 {{
    margin: 0 0 8px !important;
    font-size: 30px !important;
    font-weight: 800 !important;
    color: #0F172A !important;
    letter-spacing: -0.03em !important;
    -webkit-text-fill-color: #0F172A !important;
    background: none !important;
}}
.aw-hero p, .aw-hero strong {{
    color: #1E293B !important;
    font-size: 14px !important;
    line-height: 1.55 !important;
}}

/* ── Chips ── */
.aw-chip {{
    display: inline-block;
    background: linear-gradient(135deg, #EFF6FF, #F0FDF4) !important;
    border: 1px solid rgba(59, 130, 246, 0.30);
    color: #1E40AF !important;
    border-radius: 999px;
    padding: 5px 12px;
    margin: 10px 6px 0 0;
    font-size: 12px !important;
    font-weight: 700 !important;
    transition: transform .15s ease, box-shadow .15s ease, background .15s ease;
}}
.aw-chip:hover {{
    transform: translateY(-1px);
    background: linear-gradient(135deg, #DBEAFE, #D1FAE5) !important;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.18);
}}

/* ── Steps ── */
.aw-steps {{
    display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px;
}}
.aw-step {{
    background: rgba(255, 255, 255, 0.80);
    backdrop-filter: blur(4px);
    border: 1px solid rgba(59, 130, 246, 0.18);
    border-radius: 10px;
    padding: 6px 10px;
    font-size: 12px;
    font-weight: 600;
    color: #0F172A !important;
}}
.aw-step b {{ color: #3B82F6 !important; }}

/* ── Status blocks ── */
.aw-status {{
    border-radius: 12px;
    padding: 12px 14px;
    margin: 8px 0 4px;
    border: 1px solid rgba(59, 130, 246, 0.20);
    background: #EFF6FF;
    font-weight: 600;
    color: #0F172A !important;
}}
.aw-status.ok  {{ background: #F0FDF4; border-color: #6EE7B7; }}
.aw-status.warn {{ background: #FFFBEB; border-color: #FCD34D; }}
.aw-status.err  {{ background: #FFF1F2; border-color: #FDA4AF; }}

/* ── Typography ── */
.gradio-container h1 {{
    font-size: 24px !important; font-weight: 800 !important;
    color: #000000 !important;
}}
.gradio-container h2, .gradio-container h3, .gradio-container h4 {{
    color: #0F172A !important; font-weight: 700 !important;
}}
.gradio-container p, .gradio-container li, .markdown-body,
.gradio-container span, .gradio-container label, .gradio-container td,
.gradio-container th, .gradio-container button {{
    color: #000000 !important;
}}
.markdown, .prose, .prose * {{
    color: #000000 !important;
}}

/* ── Tabs ── */
.tabs, .tab-container, #aw-main-tabs {{
    position: relative !important;
    z-index: 20 !important;
    overflow: visible !important;
    pointer-events: auto !important;
}}
.tabitem, .tab-content {{
    background: transparent !important;
    border: none !important;
    pointer-events: auto !important;
}}
.tab-nav, [role="tablist"] {{
    background: rgba(255, 255, 255, 0.92) !important;
    border: 1px solid rgba(59, 130, 246, 0.18) !important;
    border-radius: 16px !important;
    padding: 10px 12px !important;
    gap: 8px !important;
    flex-wrap: wrap !important;
    box-shadow: 0 8px 24px var(--aw-shadow) !important;
    margin-bottom: 14px !important;
    position: sticky !important;
    top: 0 !important;
    z-index: 1000 !important;
    pointer-events: auto !important;
    backdrop-filter: blur(8px);
}}
.tab-nav button, [role="tab"] {{
    color: #0F172A !important;
    background: linear-gradient(135deg, #F8FAFF, #F1F5F9) !important;
    border: 1px solid rgba(59, 130, 246, 0.15) !important;
    border-radius: 11px !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    padding: 9px 15px !important;
    cursor: pointer !important;
    pointer-events: auto !important;
    position: relative !important;
    z-index: 1001 !important;
    opacity: 1 !important;
    transition: background .15s ease, border-color .15s ease, transform .12s ease, box-shadow .15s ease;
}}
.tab-nav button:hover, [role="tab"]:hover {{
    background: linear-gradient(135deg, #DBEAFE, #D1FAE5) !important;
    border-color: rgba(59, 130, 246, 0.35) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.12) !important;
}}
.tab-nav button.selected, [role="tab"][aria-selected="true"] {{
    background: linear-gradient(135deg, #BFDBFE, #C4B5FD) !important;
    color: #0F172A !important;
    border-color: rgba(59, 130, 246, 0.45) !important;
    box-shadow: 0 4px 14px rgba(59, 130, 246, 0.18) !important;
}}

/* ── Blocks / panels ── */
.block, .block.padded, .panel {{
    background: rgba(255, 255, 255, 0.95) !important;
    border: 1px solid rgba(59, 130, 246, 0.12) !important;
    border-radius: 16px !important;
    box-shadow: 0 6px 20px var(--aw-shadow) !important;
}}
label, .label-wrap span {{
    color: #1E40AF !important;
    font-size: 12px !important;
    text-transform: uppercase !important;
    letter-spacing: .4px !important;
    font-weight: 700 !important;
}}

/* ── Accordion ── */
.accordion, .label-wrap {{
    border-radius: 14px !important;
}}
.accordion > .label-wrap {{
    background: linear-gradient(90deg, #EFF6FF, #F0FDF4) !important;
    border: 1px solid rgba(59, 130, 246, 0.18) !important;
    border-radius: 12px !important;
    padding: 10px 12px !important;
    margin-bottom: 6px !important;
}}
.accordion > .label-wrap span {{
    font-size: 14px !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
    color: #0F172A !important;
}}

/* ── Tables (keep dark header, light body) ── */
.table-wrap thead tr th,
table thead tr th,
.dataframe thead tr th,
.svelte-1gfkn6u thead tr th,
th {{
    background: linear-gradient(135deg, #1D4ED8, #3B82F6) !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    padding: 10px 12px !important;
    border-bottom: 2px solid #1D4ED8 !important;
}}
.table-wrap tbody tr td,
table tbody tr td,
.dataframe tbody tr td,
.svelte-1gfkn6u tbody tr td,
td {{
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    background: #FFFFFF !important;
    padding: 8px 12px !important;
    border-bottom: 1px solid rgba(15, 23, 42, 0.07) !important;
}}
.table-wrap tbody tr:nth-child(odd) td,
table tbody tr:nth-child(odd) td,
.dataframe tbody tr:nth-child(odd) td,
.svelte-1gfkn6u tbody tr:nth-child(odd) td,
tr:nth-child(odd) td {{
    background: #F0F9FF !important;
}}
.table-wrap tbody tr:nth-child(even) td,
table tbody tr:nth-child(even) td,
.dataframe tbody tr:nth-child(even) td,
.svelte-1gfkn6u tbody tr:nth-child(even) td,
tr:nth-child(even) td {{
    background: #FFFFFF !important;
}}
.table-wrap tbody tr:hover td,
table tbody tr:hover td,
.dataframe tbody tr:hover td,
tr:hover td {{
    background: linear-gradient(90deg, #DBEAFE, #D1FAE5) !important;
    color: #1E40AF !important;
    -webkit-text-fill-color: #1E40AF !important;
}}
/* Force Gradio inner table cells to always have visible text */
[data-testid="dataframe"] td,
[data-testid="dataframe"] th,
.table-wrap td *, .table-wrap th * {{
    color: inherit !important;
    -webkit-text-fill-color: inherit !important;
}}
.svelte-15lo0d8 td, .svelte-15lo0d8 th {{
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
    background: #F8FAFF !important;
}}

/* ── Interactive / editable Dataframe cells ── */
/* The table container and scroll wrapper */
[data-testid="dataframe"],
.table-wrap,
.gr-dataframe,
.overflow-auto {{
    background: #FFFFFF !important;
    border-radius: 12px !important;
    border: 1px solid rgba(59, 130, 246, 0.15) !important;
}}
/* Input elements rendered inside editable cells */
.cell-wrap input,
.cell-wrap textarea,
[data-testid="dataframe"] input,
[data-testid="dataframe"] textarea,
.table-wrap input,
.table-wrap textarea,
td input, td textarea,
.gr-dataframe input,
.gr-dataframe textarea {{
    background: #FFFFFF !important;
    background-color: #FFFFFF !important;
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
    border: 1px solid rgba(59, 130, 246, 0.20) !important;
    border-radius: 6px !important;
    padding: 2px 6px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    outline: none !important;
    box-shadow: none !important;
}}
.cell-wrap input:focus,
.cell-wrap textarea:focus,
[data-testid="dataframe"] input:focus,
.table-wrap input:focus {{
    border-color: rgba(59, 130, 246, 0.50) !important;
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.12) !important;
    background: #EFF6FF !important;
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
}}
/* The cell wrapper div itself */
.cell-wrap,
.cell-wrap * {{
    background: transparent !important;
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
}}
/* Gradio table toolbar / top bar (sometimes dark) */
.table-wrap > div:first-child,
[data-testid="dataframe"] > div:first-child {{
    background: #F8FAFF !important;
    border-bottom: 1px solid rgba(59, 130, 246, 0.12) !important;
}}
/* Selected cell highlight */
.selected, td.selected, .cell-wrap.selected {{
    background: #DBEAFE !important;
    outline: 2px solid rgba(59, 130, 246, 0.45) !important;
}}
/* Any remaining dark-bg svelte wrappers inside dataframe */
[data-testid="dataframe"] .svelte-1ipelgc,
[data-testid="dataframe"] .svelte-15lo0d8,
[data-testid="dataframe"] [class*="svelte-"] {{
    background: #FFFFFF !important;
    color: #0F172A !important;
}}

/* ── Buttons ── */
button.primary, .gr-button.primary {{
    background: linear-gradient(135deg, #93C5FD 0%, #86EFAC 50%, #C4B5FD 100%) !important;
    background-size: 200% 200% !important;
    color: #0F172A !important;
    border: 1px solid rgba(59, 130, 246, 0.40) !important;
    border-radius: 12px !important;
    font-weight: 800 !important;
    font-size: 14px !important;
    box-shadow: 0 6px 18px rgba(59, 130, 246, 0.18) !important;
    min-height: 44px !important;
    transition: transform .12s ease, box-shadow .12s ease, background-position .3s ease !important;
}}
button.primary:hover, .gr-button.primary:hover {{
    transform: translateY(-2px);
    box-shadow: 0 10px 24px rgba(59, 130, 246, 0.25) !important;
    background-position: right center !important;
}}
button.secondary, .gr-button.secondary {{
    background: linear-gradient(135deg, #FFFFFF, #F0F9FF) !important;
    color: #0F172A !important;
    border: 1px solid rgba(59, 130, 246, 0.25) !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    transition: background .12s ease, border-color .12s ease, transform .12s ease, box-shadow .12s ease;
}}
button.secondary:hover, .gr-button.secondary:hover {{
    background: linear-gradient(135deg, #EFF6FF, #ECFDF5) !important;
    border-color: rgba(59, 130, 246, 0.45) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.12) !important;
}}

/* ── File upload dropzones ── */
.file-preview, [data-testid="file"] {{
    border-radius: 14px !important;
    background: #FAFBFF !important;
}}
.wrap.svelte-1ipelgc, .upload-container {{
    border: 2px dashed rgba(59, 130, 246, 0.35) !important;
    border-radius: 14px !important;
    background: linear-gradient(135deg, #F8FAFF, #F0FDF4) !important;
    transition: border-color .15s ease, background .15s ease;
}}
.wrap.svelte-1ipelgc:hover, .upload-container:hover {{
    border-color: rgba(59, 130, 246, 0.55) !important;
    background: linear-gradient(135deg, #EFF6FF, #ECFDF5) !important;
}}

/* ── Progress bars ── */
.progress-bar, [role="progressbar"], .progress, progress {{
    background: linear-gradient(90deg, #BAE6FD, #A7F3D0, #DDD6FE) !important;
    border-radius: 999px !important;
    border: none !important;
    height: 8px !important;
}}
.progress-bar-wrap, .progress-track {{
    background: #F1F5F9 !important;
    border-radius: 999px !important;
    border: 1px solid rgba(59, 130, 246, 0.15) !important;
    overflow: hidden !important;
}}
/* Gradio-specific progress fill */
.progress-bar > div, [role="progressbar"] > div, .progress > div {{
    background: linear-gradient(90deg, #60A5FA, #34D399, #A78BFA) !important;
    border-radius: 999px !important;
    transition: width .3s ease !important;
}}
/* Eta / status text next to progress */
.progress-text, .meta-text, .eta-text {{
    color: #1E40AF !important;
    font-weight: 700 !important;
    -webkit-text-fill-color: #1E40AF !important;
    background: transparent !important;
}}
/* Waveform / loading spinner container */
.loading, .waveform-wrapper, .gr-box .generating {{
    background: linear-gradient(135deg, #EFF6FF, #F0FDF4) !important;
    border-radius: 12px !important;
}}
.generating {{
    pointer-events: none !important;
}}

/* ── Input fields ── */
input[type="text"], input[type="number"], textarea, select,
.gr-text-input, .gr-number-input {{
    background: #FAFBFF !important;
    border: 1.5px solid rgba(59, 130, 246, 0.22) !important;
    border-radius: 10px !important;
    color: #0F172A !important;
    transition: border-color .15s ease, box-shadow .15s ease;
}}
input[type="text"]:focus, input[type="number"]:focus, textarea:focus {{
    border-color: rgba(59, 130, 246, 0.55) !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.12) !important;
    outline: none !important;
    background: #FFFFFF !important;
}}

/* ── Slider ── */
.gr-slider input[type="range"] {{
    accent-color: #60A5FA;
}}

/* ── Checkboxes & dropdowns ── */
.gr-check-radio input[type="checkbox"]:checked,
.gr-check-radio input[type="radio"]:checked {{
    accent-color: #3B82F6;
}}

/* ── PPT export box ── */
.aw-ppt-box {{
    background: linear-gradient(135deg, #FFFFFF, #EFF6FF, #F0FDF4);
    border: 1px solid rgba(59, 130, 246, 0.25);
    border-radius: 16px;
    padding: 16px 18px;
    margin-top: 10px;
    color: #000000 !important;
    box-shadow: 0 8px 20px rgba(59, 130, 246, 0.08);
}}
.aw-ppt-box * {{
    color: #000000 !important;
}}

/* ── Plotly chart overrides ── */
.js-plotly-plot .gtitle,
.js-plotly-plot .xtick text,
.js-plotly-plot .ytick text,
.js-plotly-plot .legendtext,
.js-plotly-plot .annotation-text,
.js-plotly-plot text {{
    fill: #000000 !important;
    color: #000000 !important;
}}
.plotly-graph-div, .js-plotly-plot, .plot-container {{
    position: relative !important;
    z-index: 1 !important;
    color: #000000 !important;
    border-radius: 14px !important;
    overflow: hidden !important;
    border: 1px solid rgba(59, 130, 246, 0.10) !important;
    background: #FFFFFF !important;
}}

footer {{ display: none !important; }}

/* ── Production & Cost table: force white text on any background ── */
#prod-cost-table td,
#prod-cost-table th,
#prod-cost-table td *,
#prod-cost-table th *,
#prod-cost-table input,
#prod-cost-table textarea,
#prod-cost-table span,
#prod-cost-table [class*="svelte-"] {{
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}}
/* Header text stays white on blue, body text white on dark */
#prod-cost-table thead th {{
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}}
#prod-cost-table input,
#prod-cost-table textarea {{
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    caret-color: #FFFFFF !important;
}}

/* ── ALL output tables + locked-md: white text on dark background ──
   Targets every read-only result table and the forecast model info line.
   The interactive prod-cost-table is handled separately above.        */
#locked-md,
#locked-md *,
#train-status-md p,
#train-status-md span,
#train-status-md strong,
#train-status-md em,
#train-status-md code {{
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}}

/* Shared white-text rule for all output dataframe tables */
#econ-table td,
#econ-table th,
#econ-table td *,
#econ-table th *,
#econ-table span,
#econ-table div,
#econ-table [class*="svelte-"],
#fc-table td,
#fc-table th,
#fc-table td *,
#fc-table th *,
#fc-table span,
#fc-table div,
#fc-table [class*="svelte-"],
#rank-table td,
#rank-table th,
#rank-table td *,
#rank-table th *,
#rank-table span,
#rank-table div,
#rank-table [class*="svelte-"],
#hp-table td,
#hp-table th,
#hp-table td *,
#hp-table th *,
#hp-table span,
#hp-table div,
#hp-table [class*="svelte-"] {{
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}}

/* Table containers */
#econ-table,
#fc-table,
#rank-table,
#hp-table {{
    background: transparent !important;
}}

/* Hover row — keep white text on blue highlight */
#econ-table tr:hover td,
#fc-table tr:hover td,
#rank-table tr:hover td,
#hp-table tr:hover td {{
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}}

@media (max-width: 900px) {{
  .gradio-container {{ padding: 6px 10px 24px !important; }}
  .aw-hero h1 {{ font-size: 22px !important; }}
}}

/* ── Univariate tab tables: white text on dark background ── */
#uni-overview-table td,
#uni-overview-table th,
#uni-overview-table td *,
#uni-overview-table th *,
#uni-overview-table span,
#uni-overview-table div,
#uni-overview-table [class*="svelte-"],
#uni-rank-table td,
#uni-rank-table th,
#uni-rank-table td *,
#uni-rank-table th *,
#uni-rank-table span,
#uni-rank-table div,
#uni-rank-table [class*="svelte-"],
#uni-fc-table td,
#uni-fc-table th,
#uni-fc-table td *,
#uni-fc-table th *,
#uni-fc-table span,
#uni-fc-table div,
#uni-fc-table [class*="svelte-"] {{
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}}
#uni-overview-table tr:hover td,
#uni-rank-table tr:hover td,
#uni-fc-table tr:hover td {{
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}}
"""


# ---------------------------------------------------------------------------
# Gradio app builder (upload-first interactive UI)
# ---------------------------------------------------------------------------

def build_gradio_app(results=None, raw=None, preload_files=None, parts_filter=None):
    """Delegate to interactive upload-first Gradio UI."""
    from forecasting.dashboard.app_ui import build_interactive_app
    return build_interactive_app(
        results=results,
        raw=raw,
        preload_files=preload_files,
        parts_filter=parts_filter,
    )
