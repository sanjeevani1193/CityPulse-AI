"""Plotly figures from validated saved summaries; no inference or network I/O."""
import plotly.graph_objects as go

from .artifacts import MODELS, POINT_COLUMNS
from .forecasts import time_label

COLORS = {"XGBoost": "#167E87", "MLP": "#375FB2", "TFT": "#7655A3", "LSTM": "#A85732",
          "Lag-24": "#516A7E", "Lag-168": "#7890A0"}
UNITS = {"MAE": "MAE · starts / station-hour", "RMSE": "RMSE · starts / station-hour",
         "WAPE_percent": "WAPE · % of accepted recorded starts"}


def style(figure, title, height=420):
    figure.update_layout(template="plotly_white", title=title, height=height,
                         font={"family": "Arial, sans-serif", "color": "#21384B", "size": 12},
                         paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                         title_font_size=16,
                         margin={"l": 12, "r": 40, "t": 65, "b": 100},
                         hoverlabel={"bgcolor": "#102D40", "font_color": "#FFFFFF"},
                         legend={"orientation": "h", "y": -0.3, "font_size": 11}, hovermode="closest")
    figure.update_xaxes(automargin=True, gridcolor="#E2EBF0", zerolinecolor="#BCCDD9")
    figure.update_yaxes(automargin=True, gridcolor="#E2EBF0", zerolinecolor="#BCCDD9")
    return figure


def ranking_chart(rows, metric, split):
    labels = [MODELS[r["model_id"]] for r in rows]
    values = [r[metric] for r in rows]
    figure = go.Figure(go.Bar(x=values, y=labels, orientation="h",
                             marker_color=[COLORS[x] for x in labels],
                             text=[f"{v:.3f}" for v in values], textposition="outside",
                             cliponaxis=False,
                             hovertemplate="%{y}<br>" + UNITS[metric] + ": %{x:.4f}<extra></extra>"))
    figure.update_yaxes(autorange="reversed")
    figure.update_xaxes(title=UNITS[metric], range=[0, max(values)*1.22])
    return style(figure, split.title() + " · 220,800 paired targets")


def diagnostic_chart(records, dimension, models):
    figure = go.Figure()
    labels = {MODELS[m] for m in models}
    for model in MODELS.values():
        if model not in labels:
            continue
        rows = [r for r in records if r["model"] == model]
        figure.add_trace(go.Scatter(x=[r[dimension] for r in rows], y=[r["MAE"] for r in rows],
                                   name=model, mode="lines+markers", line_color=COLORS[model],
                                   hovertemplate="%{x}<br>MAE: %{y:.4f}<extra>%{fullData.name}</extra>"))
    figure.update_yaxes(title="MAE · starts / station-hour", rangemode="tozero")
    figure.update_xaxes(title="NYC-local target hour" if dimension == "hour" else "NYC-local target month")
    return style(figure, "Saved full-test error by " + ("local hour" if dimension == "hour" else "month"))


DASHES = {"xgboost": "solid", "pytorch-mlp": "dot", "pytorch-tft": "dash",
          "tensorflow-lstm": "longdash", "lag-24": "dashdot", "lag-168": "longdashdot"}


def _time_axis(figure, sequence, zone):
    # Plot UTC instants; explicit tick text prevents browser-timezone conversion.
    indices = [0, 6, 12, 18, 23]
    ticks = []
    for i in indices:
        label = time_label(sequence[i]["target_hour_utc"], zone)
        ticks.append(label[11:16] + "<br>" + ("UTC" if zone == "UTC" else label[16:22]))
    figure.update_xaxes(type="date", tickmode="array",
                        tickvals=[sequence[i]["target_hour_utc"].isoformat() for i in indices],
                        ticktext=ticks, title="Target hour · " + ("NYC offset shown" if zone == "New York" else "UTC"))
    return figure


def forecast_chart(sequence, models, zone="New York", show_band=True):
    figure = go.Figure()
    x = [r["target_hour_utc"].isoformat() for r in sequence]
    if show_band and "pytorch-tft" in models:
        figure.add_trace(go.Scatter(x=x, y=[r["tft_q10_clipped"] for r in sequence], mode="lines",
                                   line_width=0, showlegend=False, hoverinfo="skip", name="TFT q10"))
        figure.add_trace(go.Scatter(x=x, y=[r["tft_q90_clipped"] for r in sequence], mode="lines",
                                   line_width=0, fill="tonexty", fillcolor="rgba(118,85,163,0.16)",
                                   name="TFT q10-q90", hoverinfo="skip"))
    times = [[time_label(r["target_hour_utc"]), time_label(r["target_hour_utc"], "UTC"), r["horizon_step"]]
             for r in sequence]
    hover = "NYC: %{customdata[0]}<br>UTC: %{customdata[1]}<br>Horizon: %{customdata[2]}"
    for model in models:
        custom = [t + [r["actual"], r[POINT_COLUMNS[model]] - r["actual"]] for t, r in zip(times, sequence)]
        quantile_hover = ""
        if model == "pytorch-tft":
            custom = [values + [r["tft_q10_clipped"], r["tft_q90_clipped"]] for values, r in zip(custom, sequence)]
            quantile_hover = "<br>q10: %{customdata[5]:.3f}<br>q90: %{customdata[6]:.3f}"
        figure.add_trace(go.Scatter(x=x, y=[r[POINT_COLUMNS[model]] for r in sequence],
            customdata=custom, mode="lines+markers", name=MODELS[model],
            line={"color": COLORS[MODELS[model]], "dash": DASHES[model], "width": 2}, marker_size=4,
            hovertemplate=hover + "<br>Prediction: %{y:.3f}<br>Actual: %{customdata[3]:.0f}<br>Error: %{customdata[4]:+.3f}" + quantile_hover + "<extra>%{fullData.name}</extra>"))
    figure.add_trace(go.Scatter(x=x, y=[r["actual"] for r in sequence], customdata=times,
        mode="lines+markers", name="Actual recorded starts", line={"color": "#102D40", "width": 3},
        marker_size=5, hovertemplate=hover + "<br>Actual: %{y:.0f}<extra>%{fullData.name}</extra>"))
    figure.update_yaxes(title="Accepted recorded starts / hour", rangemode="tozero")
    return _time_axis(style(figure, "Saved forecasts and observed starts", height=480), sequence, zone)


def horizon_error_chart(details, models):
    figure = go.Figure()
    for model in models:
        rows = [r for r in details if r["model_id"] == model]
        custom = [[time_label(r["target_hour_utc"]), time_label(r["target_hour_utc"], "UTC"), r["horizon_step"],
                   r["actual"], r["prediction"]] for r in rows]
        figure.add_trace(go.Scatter(x=[r["horizon_step"] for r in rows], y=[r["error"] for r in rows],
            customdata=custom, name=MODELS[model], mode="lines+markers",
            line={"color": COLORS[MODELS[model]], "dash": DASHES[model]},
            hovertemplate="Horizon %{x}<br>NYC: %{customdata[0]}<br>UTC: %{customdata[1]}<br>Actual: %{customdata[3]:.0f}<br>Prediction: %{customdata[4]:.3f}<br>Error: %{y:+.3f}<extra>%{fullData.name}</extra>"))
    figure.add_hline(y=0, line_color="#21384B", line_dash="dot")
    figure.update_xaxes(title="Forecast horizon (elapsed hours, 1-24)", tickmode="array",
                        tickvals=[1, 4, 8, 12, 16, 20, 24], range=[.5, 24.5])
    figure.update_yaxes(title="Prediction - actual (starts / hour)")
    return style(figure, "Signed errors by forecast horizon")
