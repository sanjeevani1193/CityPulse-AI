"""Historical Forecast Explorer workspace backed only by the verified demo."""
import pandas as pd
import streamlit as st

from .artifacts import ArtifactError, MODELS
from .charts import forecast_chart, horizon_error_chart
from .forecasts import NYC, available_origins, display_rows, select_sequence, station_ids, time_label, window_errors


def forecast_explorer(bundle):
    st.caption("FORECASTING / SAVED HISTORICAL WINDOWS")
    st.title("Forecast Explorer")
    st.write("Explore an authentic 24-hour saved forecast against accepted recorded bike starts. No live model runs or real-time data.")
    rows = bundle["forecasts"]
    controls = st.columns([1, 2])
    with controls[0]:
        station = st.selectbox("Station ID", station_ids(rows), key="forecast_station")
    with controls[1]:
        origins = [o.isoformat() for o in available_origins(rows, station)]
        origin = st.selectbox("Historical forecast origin", origins,
                              format_func=time_label, key="forecast_origin")
    st.caption("Origin means the forecast's issuance at NYC-local midnight. Horizon 1 starts at that origin; horizon 24 starts 23 elapsed hours later.")
    try:
        sequence = select_sequence(rows, station, origin)
    except ArtifactError:
        st.error("The selected saved forecast is missing or invalid. No values have been filled or regenerated.")
        return
    selected = st.multiselect("Forecasting methods", list(MODELS),
                              default=["xgboost", "pytorch-tft"], format_func=MODELS.get,
                              key="forecast_models")
    settings = st.columns([1, 1])
    with settings[0]:
        zone = st.radio("Chart timestamp labels", ("New York", "UTC"), horizontal=True, key="forecast_zone")
    with settings[1]:
        band = st.checkbox("Show TFT q10-q90 band", value=True, disabled="pytorch-tft" not in selected,
                           key="forecast_band")
    stats = st.columns(3)
    stats[0].metric("Elapsed forecast horizon", "24 hours")
    stats[1].metric("Trusted actual targets", "24 / 24")
    stats[2].metric("Recorded starts in this window", f"{sum(r['actual'] for r in sequence):,.0f}")
    st.caption("Origin: " + time_label(sequence[0]["origin_utc"]) + " | " + time_label(sequence[0]["origin_utc"], "UTC"))
    if len({r["target_hour_utc"].astimezone(NYC).utcoffset() for r in sequence}) > 1:
        st.info("This window crosses a daylight-saving transition. UTC identities stay unique; repeated local times have different offsets. The horizon still contains exactly 24 elapsed hours.")
    if not selected:
        st.warning("Select one or more forecasting methods to compare. Only actual recorded starts are shown.")
    st.plotly_chart(forecast_chart(sequence, selected, zone, band and "pytorch-tft" in selected),
                    theme=None, width="stretch", key="forecast_sequence_chart")
    st.caption("Hover for the exact UTC target, NYC time with offset, horizon, actual count and prediction. Counts represent accepted recorded starts, not all physical rental demand.")
    if "pytorch-tft" in selected and band:
        st.caption("TFT shading spans saved q10-q90; its line is q50 (the predicted median). This is a nominal 80% prediction band if calibrated, not a confidence interval for a mean or a guarantee for these 24 observations. Saved quantiles use the fixed nonnegative clipping rule.")
    st.subheader("Selected-window diagnostics")
    st.caption("Only this station and these 24 trusted target hours. These descriptive errors do not replace the official 220,800-target full-test results in Model Comparison.")
    if selected:
        summaries, details = window_errors(sequence, selected)
        table = [{"Model": MODELS[r["model_id"]], "Window MAE": r["MAE"], "Mean signed error": r["mean_error"],
                  "Overprediction hours": r["overprediction_hours"], "Underprediction hours": r["underprediction_hours"],
                  "Exact hours": r["exact_hours"], "Largest absolute error": r["largest_absolute_error"],
                  "Largest-error horizon": r["largest_error_step"]} for r in summaries]
        st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch",
                     column_config={k: st.column_config.NumberColumn(format="%.3f") for k in
                                    ("Window MAE", "Mean signed error", "Largest absolute error")})
        st.plotly_chart(horizon_error_chart(details, selected), theme=None,
                        width="stretch", key="forecast_error_chart")
        st.caption("Signed error = prediction - actual. Positive values are overpredictions; negative values are underpredictions. MAE averages their absolute sizes, in starts per target hour.")
        with st.expander("Largest miss for each selected method"):
            for row in summaries:
                st.write(f"{MODELS[row['model_id']]}: horizon {row['largest_error_step']}, "
                         f"predicted {row['largest_error_prediction']:.3f} versus actual {row['largest_error_actual']:.0f}; "
                         f"absolute error {row['largest_absolute_error']:.3f} starts. Earliest horizon is shown if tied.")
            st.caption("A large miss describes a forecast discrepancy; it does not identify its cause or establish a model's overall ranking.")
    with st.expander("Exact timestamps and saved values"):
        st.dataframe(pd.DataFrame(display_rows(sequence, selected)), hide_index=True, width="stretch")
    st.caption(f"Demo scope: {len(station_ids(rows))} station IDs and {bundle['manifest']['subset']['windows']} complete windows selected independently of test errors. Histories assume recorded counts were available at bucket end; actual publisher latency was not replayed.")
