"""Charts of frozen synthetic evidence; controls never recompute study outcomes."""
import plotly.graph_objects as go

from .charts import style
from .experiment_data import ESTIMATORS


def effect_chart(experiment, selected="cuped", compare=True):
    keys = list(ESTIMATORS) if compare else [selected]
    figure = go.Figure()
    for key in keys:
        row = experiment[key]
        figure.add_trace(go.Scatter(
            x=[row["effect"]], y=[ESTIMATORS[key]], mode="markers", name=ESTIMATORS[key],
            marker={"size": 13, "color": "#167E87" if key == "cuped" else "#516A7E"},
            error_x={"type": "data", "symmetric": False,
                     "array": [row["ci_high"]-row["effect"]],
                     "arrayminus": [row["effect"]-row["ci_low"]]},
            customdata=[[row["ci_low"], row["ci_high"], row["p_value"]]],
            hovertemplate="Effect: %{x:.4f}<br>95% CI: [%{customdata[0]:.4f}, %{customdata[1]:.4f}]"
                          "<br>p: %{customdata[2]:.3g}<extra>%{fullData.name}</extra>"))
    figure.add_vline(x=0, line_dash="dot", line_color="#516A7E")
    figure.add_vline(x=experiment["design"]["practical_threshold"], line_dash="dash", line_color="#A85732")
    figure.update_xaxes(title="Treatment - control (starts / station / local-calendar-day)")
    figure.update_layout(showlegend=False)
    return style(figure, "Saved effects and 95% Welch intervals", height=310)


def station_chart(rows, adjusted=False):
    column = "adjusted_post" if adjusted else "observed_post"
    figure = go.Figure()
    for arm, color in (("Control", "#516A7E"), ("Treatment", "#167E87")):
        records = [r for r in rows if r["arm"] == arm]
        if records:
            figure.add_trace(go.Box(y=[r[column] for r in records], name=arm, boxpoints="all",
                jitter=0, pointpos=0, marker_color=color,
                customdata=[r["station_id"] for r in records],
                hovertemplate="%{customdata}<br>Station mean: %{y:.4f}<extra>%{fullData.name}</extra>"))
    figure.update_yaxes(title="Adjusted station means" if adjusted else "Observed starts / station / day")
    return style(figure, "Synthetic station outcomes" + (" - CUPED scale" if adjusted else " - observed scale"))


def planning_chart(experiment, metric="power"):
    rows = [experiment["planning_raw"], experiment["planning_cuped"]]
    figure = go.Figure(go.Bar(x=["Unadjusted", "CUPED"], y=[r[metric] for r in rows],
        marker_color=["#516A7E", "#167E87"], hovertemplate="%{x}: %{y:.4f}<extra></extra>"))
    figure.add_hline(y=.8 if metric == "power" else experiment["design"]["practical_threshold"],
                     line_dash="dash", line_color="#A85732")
    figure.update_yaxes(title="Planning power (probability)" if metric == "power" else "MDE (starts / station / day)",
                        range=[0, 1.05] if metric == "power" else None)
    return style(figure, "Saved pre-treatment planning - 50 stations per arm")
