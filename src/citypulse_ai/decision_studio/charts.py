"""Plotly figures from validated saved summaries; no inference or network I/O."""
import plotly.graph_objects as go

from .artifacts import MODELS

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
