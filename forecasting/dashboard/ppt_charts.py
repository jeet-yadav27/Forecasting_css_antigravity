"""
Charts and tables for the PowerPoint briefing.

Shared by multivariate and univariate result dicts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from forecasting.config import FORECAST_HORIZON, MODEL_COLORS

_TXT = "#000000"
_A1 = "#1D4ED8"
_A2 = "#047857"
_A3 = "#B45309"
_A4 = "#B91C1C"


def _style(fig: go.Figure, title: str, height: int = 520) -> go.Figure:
    fig.update_layout(
        title_text=f"<b>{title}</b>",
        title_font=dict(size=18, color=_TXT),
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color=_TXT, size=13, family="DM Sans, Inter, sans-serif"),
        legend=dict(
            bgcolor="rgba(255,255,255,0.95)",
            font=dict(color=_TXT, size=12),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0,
        ),
        margin=dict(l=56, r=56, t=80, b=56),
    )
    fig.update_xaxes(tickfont=dict(color=_TXT), title_font=dict(color=_TXT), title="Month")
    fig.update_yaxes(tickfont=dict(color=_TXT), title_font=dict(color=_TXT),
                     gridcolor="rgba(0,0,0,0.08)")
    return fig


def is_univariate(r: dict) -> bool:
    return "claim_vals" not in r and "claims" in r


def hist_claims(r: dict) -> np.ndarray:
    if is_univariate(r):
        return np.asarray(r.get("claims"), dtype=float).ravel()
    return np.asarray(r.get("claim_vals"), dtype=float).ravel()


def hist_periods(r: dict) -> list[str]:
    if is_univariate(r):
        return [str(p) for p in (r.get("periods") or [])]
    monthly = r.get("monthly")
    if monthly is None:
        return []
    return [str(p) for p in monthly["period"]]


def future_periods(r: dict) -> list[str]:
    return [str(p) for p in (r.get("future_periods") or [])]


def best_forecast(r: dict) -> np.ndarray:
    if is_univariate(r):
        return np.asarray(r.get("best_forecast"), dtype=float).ravel()
    return np.asarray(r.get("best_forecast", r.get("ensemble_raw")), dtype=float).ravel()


def ranking_df(r: dict) -> pd.DataFrame:
    df = r.get("ranking") if is_univariate(r) else r.get("ranking_df")
    if df is None:
        return pd.DataFrame()
    return df.copy()


def ranked_models(r: dict) -> list[str]:
    df = ranking_df(r)
    if df is None or df.empty or "Model" not in df.columns:
        best = r.get("best_model")
        return [best] if best else []
    return [str(m) for m in df["Model"].tolist()]


def second_best_model(r: dict) -> str | None:
    models = ranked_models(r)
    best = str(r.get("best_model") or "")
    rest = [m for m in models if m != best]
    return rest[0] if rest else None


def oos_bundle(r: dict, n: int = 6) -> dict:
    actual = np.asarray(r.get("oos_actual", []), dtype=float).ravel()
    preds = r.get("oos_preds") or {}
    months = r.get("oos_periods")
    if not months:
        hp = hist_periods(r)
        months = hp[-len(actual):] if len(hp) >= len(actual) else [str(i + 1) for i in range(len(actual))]
    n = min(n, len(actual)) if len(actual) else 0
    out_preds = {}
    for k, v in preds.items():
        arr = np.asarray(v, dtype=float).ravel()
        if len(arr) >= n and n:
            out_preds[str(k)] = arr[-n:]
    return {
        "months": list(months)[-n:] if n else [],
        "actual": actual[-n:] if n else actual,
        "preds": out_preds,
        "n": n,
    }


def holdout_table(r: dict, model: str | None = None, n: int = 6) -> pd.DataFrame:
    model = model or r.get("best_model")
    b = oos_bundle(r, n=n)
    pred = b["preds"].get(str(model), np.array([]))
    actual = b["actual"]
    months = b["months"]
    mlen = min(len(actual), len(pred), len(months))
    if mlen == 0:
        return pd.DataFrame(columns=["Month", "Actual", "Predicted", "Difference"])
    rows = []
    for i in range(mlen):
        a, p = float(actual[i]), float(pred[i])
        rows.append({
            "Month": months[i],
            "Actual": round(a, 2),
            "Predicted": round(p, 2),
            "Difference": round(a - p, 2),
        })
    return pd.DataFrame(rows)


def _sum_accuracy(actual: np.ndarray, pred: np.ndarray) -> dict:
    a = float(np.nansum(actual))
    p = float(np.nansum(pred))
    if abs(a) < 1e-9:
        err = float("nan")
        acc = float("nan")
    else:
        err = abs(a - p) / a * 100.0
        acc = 100.0 - err
    return {
        "Actual 6 Month Sum": round(a, 2),
        "Predicted 6 Month Sum": round(p, 2),
        "Difference": round(a - p, 2),
        "Error %": round(err, 2) if np.isfinite(err) else float("nan"),
        "Accuracy %": round(acc, 2) if np.isfinite(acc) else float("nan"),
    }


def accuracy_comparison_df(r: dict, n: int = 6) -> pd.DataFrame:
    b = oos_bundle(r, n=n)
    best = str(r.get("best_model") or "")
    second = second_best_model(r)
    rows = []
    for name, tag in ((best, "Best"), (second, "Second Best")):
        if not name:
            continue
        pred = b["preds"].get(name)
        if pred is None or len(pred) == 0:
            continue
        met = _sum_accuracy(b["actual"], pred)
        rows.append({"Role": tag, "Model Name": name, **met})
    df = pd.DataFrame(rows)
    if df.empty or "Accuracy %" not in df.columns:
        return df
    acc = pd.to_numeric(df["Accuracy %"], errors="coerce")
    if acc.notna().any():
        winner = acc.idxmax()
        df["Better"] = [ "★" if i == winner else "" for i in df.index]
    else:
        df["Better"] = ""
    return df


def trend_direction(r: dict) -> str:
    if is_univariate(r):
        return str((r.get("trend") or {}).get("direction") or "—")
    y = hist_claims(r)
    if len(y) < 6:
        return "—"
    a, b = float(np.mean(y[:6])), float(np.mean(y[-6:]))
    chg = (b - a) / (abs(a) + 1e-9) * 100
    if chg > 8:
        return "Increasing"
    if chg < -8:
        return "Decreasing"
    return "Stable"


def data_range(r: dict) -> str:
    if is_univariate(r):
        o = r.get("overview") or {}
        return f"{o.get('Start Month', '—')} → {o.get('End Month', '—')}"
    ps = hist_periods(r)
    if not ps:
        return "—"
    return f"{ps[0]} → {ps[-1]}"


def executive_rows(r: dict, mode: str) -> list[tuple[str, str]]:
    fc = best_forecast(r)
    hist = hist_claims(r)
    outlook = "—"
    if len(fc) and len(hist):
        chg = (float(np.mean(fc)) - float(np.mean(hist[-6:]))) / (abs(float(np.mean(hist[-6:]))) + 1e-9) * 100
        outlook = f"{'Up' if chg > 3 else 'Down' if chg < -3 else 'Flat'} ({chg:+.1f}% vs last 6m avg)"
    return [
        ("Part Name", str(r.get("part", "—"))),
        ("Analysis", mode),
        ("Forecast Horizon", f"{r.get('horizon', FORECAST_HORIZON)} months"),
        ("Best Selected Model", str(r.get("best_model", "—"))),
        ("Forecast Trend Direction", trend_direction(r)),
        ("Data Range Used", data_range(r)),
        ("Outlook", outlook),
    ]


def production_series(r: dict) -> np.ndarray | None:
    prod = r.get("production")
    if prod is None and not is_univariate(r) and r.get("monthly") is not None:
        try:
            prod = r["monthly"]["production"]
        except Exception:
            prod = None
    if prod is None:
        return None
    arr = np.asarray(prod, dtype=float).ravel()
    if len(arr) == 0 or not np.any(np.isfinite(arr) & (arr > 0)):
        return None
    return arr


def cpv_cr_frames(r: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Historical + forecast CPV (claims/production) and CR (claim rate)."""
    claims = hist_claims(r)
    months = hist_periods(r)
    n = min(len(claims), len(months))
    claims, months = claims[:n], months[:n]
    prod = production_series(r)
    if prod is None:
        return pd.DataFrame(), pd.DataFrame()
    prod = prod[:n]
    cpv = claims / np.where(prod > 0, prod, np.nan)
    cr = r.get("claim_ratio")
    if cr is None:
        cr = cpv.copy()
    else:
        cr = np.asarray(cr, dtype=float).ravel()[:n]
    hist = pd.DataFrame({
        "Month": months,
        "Claims": claims,
        "Production": prod,
        "CPV": cpv,
        "CR": cr,
    })

    fut_m = future_periods(r)
    fc = best_forecast(r)
    fe = r.get("forecast_economics") or {}
    fut_prod = fe.get("future_production")
    if fut_prod is None:
        last = float(np.nanmean(prod[-3:])) if n else float("nan")
        fut_prod = np.full(len(fc), last)
    fut_prod = np.asarray(fut_prod, dtype=float).ravel()
    H = min(len(fut_m), len(fc), len(fut_prod))
    fut_cpv = fc[:H] / np.where(fut_prod[:H] > 0, fut_prod[:H], np.nan)
    fut_cr = fe.get("forecast_claim_ratio")
    if fut_cr is None:
        fut_cr = fut_cpv
    else:
        fut_cr = np.asarray(fut_cr, dtype=float).ravel()[:H]
    fut = pd.DataFrame({
        "Month": fut_m[:H],
        "Claims": fc[:H],
        "Production": fut_prod[:H],
        "CPV": fut_cpv,
        "CR": fut_cr[:H],
    })
    return hist, fut


def make_historical_trend_figure(r: dict) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=hist_periods(r), y=hist_claims(r),
        mode="lines+markers", name="Historical Actual Claims",
        line=dict(color=_A1, width=3), marker=dict(size=6),
    ))
    return _style(fig, f"Historical Claims Trend — {r.get('part')}")


def make_best_model_forecast_figure(r: dict) -> go.Figure:
    """History + last-6 predicted overlay + future forecast + CI."""
    best = str(r.get("best_model", "Model"))
    hist_x = hist_periods(r)
    hist_y = hist_claims(r)
    fut_x = future_periods(r)
    fc = best_forecast(r)
    lo = np.asarray(r.get("ci_low", []), dtype=float).ravel()
    hi = np.asarray(r.get("ci_high", []), dtype=float).ravel()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_x, y=hist_y, mode="lines+markers", name="Historical Actual Claims",
        line=dict(color=_A1, width=2.5), marker=dict(size=5),
    ))
    b = oos_bundle(r, n=6)
    pred = b["preds"].get(best)
    if pred is not None and len(pred) and b["months"]:
        fig.add_trace(go.Scatter(
            x=b["months"], y=pred, mode="lines+markers",
            name="Predicted Claims (last 6 months)",
            line=dict(color=_A3, width=2.5, dash="dash"),
            marker=dict(size=8, symbol="diamond"),
        ))
        fig.add_trace(go.Scatter(
            x=b["months"], y=b["actual"], mode="markers",
            name="Actual vs Predicted (test)",
            marker=dict(size=9, color=_A4, symbol="x"),
        ))
    if len(fut_x) and len(lo) and len(hi):
        fig.add_trace(go.Scatter(
            x=list(fut_x) + list(fut_x[::-1]),
            y=list(hi[:len(fut_x)]) + list(lo[:len(fut_x)][::-1]),
            fill="toself", fillcolor="rgba(4,120,87,0.14)",
            line=dict(color="rgba(0,0,0,0)"), name="Confidence Interval",
        ))
    if len(fut_x) and len(fc):
        fig.add_trace(go.Scatter(
            x=fut_x, y=fc[:len(fut_x)], mode="lines+markers",
            name="Future Forecast",
            line=dict(color=_A2, width=3), marker=dict(size=8, symbol="diamond"),
        ))
    return _style(fig, f"Best Model Forecast - {best}", height=540)


def make_avp_model_figure(r: dict, model: str) -> go.Figure:
    tbl = holdout_table(r, model=model, n=6)
    fig = go.Figure()
    if tbl.empty:
        fig.add_annotation(text="No last-6-month test predictions", showarrow=False)
        return _style(fig, f"Actual vs Predicted — {model}")
    fig.add_trace(go.Scatter(
        x=tbl["Month"], y=tbl["Actual"], mode="lines+markers", name="Actual",
        line=dict(color=_TXT, width=3), marker=dict(size=8),
    ))
    fig.add_trace(go.Scatter(
        x=tbl["Month"], y=tbl["Predicted"], mode="lines+markers", name="Predicted",
        line=dict(color=MODEL_COLORS.get(model, _A2), width=3, dash="dash"),
        marker=dict(size=8, symbol="diamond"),
    ))
    return _style(fig, f"Actual vs Predicted (Last 6 Months) — {model}")


def make_accuracy_comparison_figure(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df is None or df.empty:
        fig.add_annotation(text="Need two models for accuracy comparison", showarrow=False)
        return _style(fig, "Accuracy Comparison (Last 6 Months Sum)")
    colors = [_A2 if str(v).strip() == "★" else _A1 for v in df.get("Better", [""] * len(df))]
    fig.add_trace(go.Bar(
        x=df["Model Name"], y=df["Accuracy %"], name="Accuracy %",
        marker_color=colors, text=[f"{v:.1f}%" if np.isfinite(v) else "—" for v in df["Accuracy %"]],
        textposition="outside",
    ))
    fig.update_yaxes(title="Accuracy %", range=[0, 110])
    return _style(fig, "Accuracy Comparison — Best vs Second Best (6-month sum)")


def make_model_comparison_rmse_figure(r: dict) -> go.Figure:
    df = ranking_df(r)
    fig = go.Figure()
    if df is None or df.empty:
        fig.add_annotation(text="No ranking", showarrow=False)
        return _style(fig, "Model Comparison")
    fig.add_trace(go.Bar(
        x=df["Model"], y=df["RMSE"], name="RMSE", marker_color=_A1,
    ))
    if "MAE" in df.columns:
        fig.add_trace(go.Bar(
            x=df["Model"], y=df["MAE"], name="MAE", marker_color=_A3,
        ))
    fig.update_layout(barmode="group")
    return _style(fig, f"Model Comparison Chart — {r.get('part')}")


def make_forecast_horizon_figure(r: dict) -> go.Figure:
    fut_x = future_periods(r)
    fc = best_forecast(r)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=fut_x, y=fc[:len(fut_x)], name="Forecast",
        marker_color=_A2, opacity=0.9,
    ))
    lo = np.asarray(r.get("ci_low", []), dtype=float).ravel()
    hi = np.asarray(r.get("ci_high", []), dtype=float).ravel()
    if len(lo) and len(hi):
        fig.add_trace(go.Scatter(
            x=fut_x, y=hi[:len(fut_x)], mode="lines", name="CI High",
            line=dict(color=_A3, width=1, dash="dot"),
        ))
        fig.add_trace(go.Scatter(
            x=fut_x, y=lo[:len(fut_x)], mode="lines", name="CI Low",
            line=dict(color=_A3, width=1, dash="dot"),
        ))
    return _style(fig, f"Forecast Horizon Projection — {r.get('part')}")


def make_cpv_trend_figure(r: dict) -> go.Figure:
    hist, fut = cpv_cr_frames(r)
    fig = go.Figure()
    if hist.empty:
        fig.add_annotation(text="Production required for CPV (Claims / Production)", showarrow=False)
        return _style(fig, "CPV Trend")
    fig.add_trace(go.Scatter(
        x=hist["Month"], y=hist["CPV"], mode="lines+markers", name="Actual CPV",
        line=dict(color=_A1, width=2.5),
    ))
    if not fut.empty:
        fig.add_trace(go.Scatter(
            x=fut["Month"], y=fut["CPV"], mode="lines+markers", name="Forecast CPV",
            line=dict(color=_A2, width=3, dash="dash"),
        ))
    return _style(fig, f"CPV Trend (Claims / Production) — {r.get('part')}")


def make_cr_trend_figure(r: dict) -> go.Figure:
    hist, fut = cpv_cr_frames(r)
    fig = go.Figure()
    if hist.empty:
        fig.add_annotation(text="Production required for CR (claims rate)", showarrow=False)
        return _style(fig, "CR Ratio Trend")
    fig.add_trace(go.Scatter(
        x=hist["Month"], y=hist["CR"], mode="lines+markers", name="Actual CR",
        line=dict(color=_A3, width=2.5),
    ))
    if not fut.empty:
        fig.add_trace(go.Scatter(
            x=fut["Month"], y=fut["CR"], mode="lines+markers", name="Forecast CR",
            line=dict(color=_A4, width=3, dash="dash"),
        ))
    return _style(fig, f"CR Ratio Trend — {r.get('part')}")


def make_cpv_cr_dual_figure(r: dict) -> go.Figure:
    hist, fut = cpv_cr_frames(r)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if hist.empty:
        fig.add_annotation(text="Upload production to compute CPV and CR", showarrow=False)
        return _style(fig, "CPV & CR Dual Axis")
    fig.add_trace(go.Scatter(
        x=hist["Month"], y=hist["Claims"], mode="lines+markers",
        name="Actual Claims", line=dict(color=_A1, width=2),
    ), secondary_y=False)
    if not fut.empty:
        fig.add_trace(go.Scatter(
            x=fut["Month"], y=fut["Claims"], mode="lines+markers",
            name="Forecast Claims", line=dict(color=_A1, width=2, dash="dot"),
        ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=hist["Month"], y=hist["CPV"], mode="lines",
        name="Actual CPV", line=dict(color=_A2, width=2),
    ), secondary_y=True)
    fig.add_trace(go.Scatter(
        x=hist["Month"], y=hist["CR"], mode="lines",
        name="Actual CR", line=dict(color=_A3, width=2),
    ), secondary_y=True)
    if not fut.empty:
        fig.add_trace(go.Scatter(
            x=fut["Month"], y=fut["CPV"], mode="lines",
            name="Forecast CPV", line=dict(color=_A2, width=2, dash="dash"),
        ), secondary_y=True)
        fig.add_trace(go.Scatter(
            x=fut["Month"], y=fut["CR"], mode="lines",
            name="Forecast CR", line=dict(color=_A3, width=2, dash="dash"),
        ), secondary_y=True)
    fig.update_yaxes(title_text="Claims", secondary_y=False)
    fig.update_yaxes(title_text="CPV / CR Ratio", secondary_y=True)
    return _style(fig, f"CPV & CR Dual Axis — {r.get('part')}", height=540)


def generate_business_insights(r: dict, acc_df: pd.DataFrame, mode: str) -> list[str]:
    best = str(r.get("best_model", "—"))
    lines = []
    lines.append(
        f"{best} achieved the lowest RMSE and was selected as the final {mode.lower()} model."
    )
    if acc_df is not None and not acc_df.empty:
        row = acc_df.iloc[0]
        acc = row.get("Accuracy %")
        if acc is not None and np.isfinite(acc):
            lines.append(
                f"The forecast accuracy for the best model was {acc:.1f}% "
                "based on the last 6-month validation period."
            )
        if len(acc_df) > 1:
            r2 = acc_df.iloc[1]
            a2 = r2.get("Accuracy %")
            if a2 is not None and np.isfinite(a2):
                lines.append(
                    f"Second-best model {r2.get('Model Name')} scored {a2:.1f}% "
                    "accuracy on the same 6-month sum method."
                )
            better = acc_df.loc[acc_df["Better"] == "★"] if "Better" in acc_df.columns else acc_df.iloc[[0]]
            if not better.empty:
                lines.append(
                    f"{better.iloc[0]['Model Name']} is the better performer on 6-month sum accuracy."
                )
    td = trend_direction(r)
    lines.append(f"Trend direction: {td}.")
    if is_univariate(r):
        for s in (r.get("insights") or [])[:4]:
            lines.append(str(s).replace("**", ""))
    hist, fut = cpv_cr_frames(r)
    if not hist.empty and not fut.empty:
        h_cpv, f_cpv = float(np.nanmean(hist["CPV"][-6:])), float(np.nanmean(fut["CPV"]))
        if np.isfinite(h_cpv) and np.isfinite(f_cpv):
            if f_cpv < h_cpv * 0.98:
                lines.append("CPV ratio is expected to decrease over the next forecast horizon.")
            elif f_cpv > h_cpv * 1.02:
                lines.append("CPV ratio is expected to increase over the next forecast horizon.")
            else:
                lines.append("CPV ratio remains broadly stable over the forecast horizon.")
        h_cr, f_cr = float(np.nanmean(hist["CR"][-6:])), float(np.nanmean(fut["CR"]))
        if np.isfinite(h_cr) and np.isfinite(f_cr):
            if abs(f_cr - h_cr) / (abs(h_cr) + 1e-9) < 0.05:
                lines.append("CR ratio remains stable with mild seasonal variations.")
            elif f_cr > h_cr:
                lines.append("CR ratio is projected to rise over the forecast horizon.")
            else:
                lines.append("CR ratio is projected to ease over the forecast horizon.")
    y = hist_claims(r)
    if len(y) >= 12:
        mu, sd = float(np.mean(y)), float(np.std(y))
        if sd > 0:
            spikes = int(np.sum(y > mu + 2 * sd))
            if spikes:
                lines.append(f"Significant spikes: {spikes} month(s) exceeded mean + 2σ.")
            else:
                lines.append("No extreme claim spikes (mean + 2σ) in the fitted history.")
    fc = best_forecast(r)
    if len(fc) and len(y):
        chg = (float(np.mean(fc)) - float(np.mean(y[-6:]))) / (abs(float(np.mean(y[-6:]))) + 1e-9) * 100
        lines.append(
            f"Forecast outlook: mean claims over the next {r.get('horizon', FORECAST_HORIZON)} "
            f"months are {chg:+.1f}% versus the last 6 historical months."
        )
    return lines


def attach_production_from_multivariate(uni: dict, mv: dict | None) -> dict:
    """Copy production / economics onto a univariate result when the same part was trained."""
    if not uni or not mv:
        return uni
    out = dict(uni)
    if production_series(out) is not None:
        return out
    if production_series(mv) is not None:
        out["production"] = mv.get("production")
        out["claim_ratio"] = mv.get("claim_ratio")
        out["forecast_economics"] = mv.get("forecast_economics")
        out["cpv"] = mv.get("cpv")
    return out
