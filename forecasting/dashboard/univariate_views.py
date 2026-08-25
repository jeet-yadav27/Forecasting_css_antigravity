"""Plotly + table helpers for the Univariate Analysis tab."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

_TXT = "#000000"
_A1 = "#1D4ED8"
_A2 = "#047857"
_A4 = "#B91C1C"
_A3 = "#B45309"


def _style(fig: go.Figure, title: str, height: int = 380) -> go.Figure:
    fig.update_layout(
        title_text=f"<b>{title}</b>",
        title_font=dict(size=16, color=_TXT),
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color=_TXT, size=12, family="DM Sans, Inter, sans-serif"),
        legend=dict(bgcolor="rgba(255,255,255,0.95)", font=dict(color=_TXT)),
        margin=dict(l=48, r=28, t=56, b=48),
    )
    fig.update_xaxes(tickfont=dict(color=_TXT), title_font=dict(color=_TXT))
    fig.update_yaxes(tickfont=dict(color=_TXT), title_font=dict(color=_TXT),
                     gridcolor="rgba(0,0,0,0.08)")
    return fig


def uni_overview_df(res: dict) -> pd.DataFrame:
    o = res.get("overview") or {}
    return pd.DataFrame([{
        "Part Name": o.get("Part Name"),
        "Total Records": o.get("Total Records"),
        "Start Month": o.get("Start Month"),
        "End Month": o.get("End Month"),
        "Missing Months": o.get("Missing Months"),
        "Data Quality Status": o.get("Data Quality Status"),
    }])


def uni_stats_df(res: dict) -> pd.DataFrame:
    s = res.get("stats") or {}
    t = res.get("trend") or {}
    rows = [{k: v for k, v in s.items()}]
    df = pd.DataFrame(rows).T.reset_index()
    df.columns = ["Metric", "Value"]
    extra = pd.DataFrame([
        {"Metric": "Trend Direction", "Value": t.get("direction")},
        {"Metric": "Growth share %", "Value": round(float(t.get("growth_pct", 0)), 2)},
        {"Metric": "Decline share %", "Value": round(float(t.get("decline_pct", 0)), 2)},
        {"Metric": "6-month change %", "Value": round(float(t.get("chg_6m", 0)), 2)},
    ])
    out = pd.concat([df, extra], ignore_index=True)
    out["Value"] = out["Value"].apply(
        lambda v: round(float(v), 4) if isinstance(v, (int, float, np.floating)) and np.isfinite(v) else v
    )
    return out


def uni_forecast_df(res: dict) -> pd.DataFrame:
    fut = res.get("future_periods") or []
    primary = np.asarray(res.get("best_forecast"), dtype=float)
    lo = np.asarray(res.get("ci_low"), dtype=float)
    hi = np.asarray(res.get("ci_high"), dtype=float)
    rows = []
    for i, m in enumerate(fut):
        rows.append({
            "Month": m,
            "Forecast Claims": round(float(primary[i]), 2) if i < len(primary) else None,
            "CI Low": round(float(lo[i]), 2) if i < len(lo) else None,
            "CI High": round(float(hi[i]), 2) if i < len(hi) else None,
            "Best Model": res.get("best_model"),
        })
    return pd.DataFrame(rows)


def make_uni_trend_figure(res: dict) -> go.Figure:
    x = res["periods"]
    y = res["claims"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines+markers", name="Monthly claims",
        line=dict(color=_A1, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=res["rolling_mean_3"], mode="lines", name="Rolling mean 3",
        line=dict(color=_A3, width=2, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=res["rolling_mean_12"], mode="lines", name="Rolling mean 12",
        line=dict(color=_A2, width=2),
    ))
    return _style(fig, f"Monthly claims trend — {res['part']}")


def make_uni_season_figure(res: dict) -> go.Figure:
    df = res.get("seasonality")
    fig = go.Figure(go.Bar(
        x=df["Month"], y=df["Avg Claims"], marker_color=_A1, name="Avg claims",
    ))
    fig.update_xaxes(tickmode="array", tickvals=list(range(1, 13)),
                     ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    return _style(fig, "Seasonality (average claims by calendar month)")


def make_uni_forecast_figure(res: dict) -> go.Figure:
    hist_x = res["periods"]
    fut_x = res["future_periods"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_x, y=res["claims"], mode="lines+markers", name="History",
        line=dict(color=_A1, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=fut_x + fut_x[::-1],
        y=list(res["ci_high"]) + list(res["ci_low"][::-1]),
        fill="toself", fillcolor="rgba(4,120,87,0.12)",
        line=dict(color="rgba(0,0,0,0)"), name="95% CI",
    ))
    fig.add_trace(go.Scatter(
        x=fut_x, y=res["best_forecast"], mode="lines+markers",
        name=f"Best: {res.get('best_model')}",
        line=dict(color=_A2, width=3), marker=dict(size=8, symbol="diamond"),
    ))
    for name, fc in (res.get("forecasts") or {}).items():
        if name == res.get("best_model"):
            continue
        fig.add_trace(go.Scatter(
            x=fut_x, y=fc, mode="lines", name=name,
            line=dict(width=1.4, dash="dot"), opacity=0.7,
        ))
    return _style(fig, f"Univariate forecast — {res['part']}", height=420)
