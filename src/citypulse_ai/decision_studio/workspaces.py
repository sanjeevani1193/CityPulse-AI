"""Overview and Model Comparison; Forecast Explorer lives in explorer.py."""
import pandas as pd
import streamlit as st

from .artifacts import MODELS
from .charts import diagnostic_chart, ranking_chart
from .metrics import improvement_percent, ranked_rows
from .ui import DOCS, REPOSITORY, provenance

TRADEOFFS = {
    "xgboost": ("Engineered tabular forecasting", "300 trees, depth 6; causal lag, calendar and summary features. Lowest aggregate held-out MAE and RMSE in this study; this does not establish universal superiority."),
    "pytorch-mlp": ("Compact pooled neural baseline", "6,785 parameters with station encoding and training-only numerical scaling. Very close to XGBoost in aggregate MAE; no significance claim is established."),
    "pytorch-tft": ("Sequence model with uncertainty", "Station embeddings, variable selection, gated residual blocks and temporal attention. Six completed epochs; epoch-006.pt. q10/q50/q90 quantify forecast uncertainty, unlike the point-only methods."),
    "tensorflow-lstm": ("Pooled recurrent forecasting", "8,953 parameters, 336×16 historical input, station identity and a future-calendar head. Validation-loss-selected epoch-018.keras from 20 epochs. Lower test RMSE than TFT, despite higher MAE."),
    "lag-24": ("Daily seasonal reference", "Repeat the corresponding observation from 24 elapsed hours earlier. No fitted model; a transparent comparison hurdle, not a zero-information baseline."),
    "lag-168": ("Weekly seasonal reference", "Repeat the corresponding observation from 168 elapsed hours earlier. UTC elapsed lags preserve causal timing but do not always align with NYC wall-clock time across DST."),
}


def overview(bundle):
    data = bundle["benchmark"]
    platform = data["platform"]
    st.markdown("""<section class="cp-hero"><div class="cp-kicker">URBAN MOBILITY · VERIFIED RESEARCH</div>
    <h1>From City Data to Better Questions.</h1>
    <p>Explore how CityPulse connects large-scale data engineering, station-level forecasting,
    guarded analytics and statistical experimentation-through saved, reproducible evidence.</p></section>""", unsafe_allow_html=True)
    cols = st.columns(4)
    cols[0].metric("Accepted recorded starts", f"{platform['final_window_ride_starts'] / 1e6:.2f}M")
    cols[1].metric("Finalized NYC-local months", "24")
    cols[2].metric("Training-selected stations", str(len(data["station_ids"])))
    cols[3].metric("Forecasting methods", str(len(MODELS)))
    st.caption("January 2024-December 2025 · 25 source archives including the January 2026 boundary. "
               "Full-history volume is distinct from the 100-station forecasting cohort.")
    st.subheader("Three questions. Three kinds of evidence.")
    cards = [("01 / FORECAST", "What is likely to happen?", "Compare six methods forecasting 24 elapsed hours of accepted recorded station starts from 336 historical hours.", "six_model_followup_benchmark.md"),
             ("02 / UNDERSTAND", "What happened in the data?", "SQLPilot connects a LangGraph workflow to MCP tools and restricted read-only DuckDB. Its first live Groq benchmark achieved 6/9 correct outcomes-not 100% accuracy.", "sqlpilot_release.md"),
             ("03 / EXPERIMENT", "How can an intervention be evaluated?", "ExperimentLab demonstrates randomized assignment, Welch confidence intervals, CUPED and power analysis using fully synthetic station outcomes.", "experimentlab.md")]
    for column, (label, title, body, doc) in zip(st.columns(3), cards):
        with column, st.container(border=True):
            st.caption(label)
            st.markdown("#### " + title)
            st.write(body)
            st.markdown(f"[Technical evidence]({DOCS}{doc})")
    st.subheader("Platform architecture")
    st.markdown("**Historical trip files → validated DuckDB / Parquet ETL → station-hour features → six-method evaluation**")
    st.markdown("**Separate components:** Forecasting & MLflow · SQLPilot (LangGraph → MCP → DuckDB) · synthetic ExperimentLab")
    st.markdown("**Historical deployment path:** GitHub Actions → keyless WIF → Cloud Build → Artifact Registry → authenticated XGBoost Cloud Run API; infrastructure managed with Terraform.")
    st.info("This Decision Studio reads local, precomputed artifacts only. It does not connect to the historical Cloud Run deployment, GCS, MLflow or Groq. It is not a real-time forecasting service.")
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("The forecasting study")
        st.dataframe(pd.DataFrame([
            {"Split": "Train", "NYC-local period": "Jan 2024-Jun 2025", "Purpose": "Station selection, scaling and model fitting"},
            {"Split": "Validation", "NYC-local period": "Jul-Sep 2025", "Purpose": "Configuration/checkpoint selection"},
            {"Split": "Test", "NYC-local period": "Oct-Dec 2025", "Purpose": "220,800 identical scored targets"}]), hide_index=True, width="stretch")
        st.caption("Counts use accepted_recorded_starts_v1. Certified source-scoped zeros are not proof of zero physical demand; station operating availability remains independently unknown.")
    with right:
        st.subheader("Start with the evidence")
        st.write("Use **Model Comparison** in the sidebar to rank methods, compare temporal splits and inspect saved error breakdowns.")
        st.markdown(f"[Repository & reproducibility]({REPOSITORY})  \n[Forecast methodology]({DOCS}final_tft_dataset.md)  \n[Cloud engineering evidence]({DOCS}cloudrun_cd.md)  \n[Synthetic experiment report]({DOCS}experimentlab_report.html)")
        st.caption("Forecast Explorer shows saved historical windows; SQLPilot executes curated offline SQL; ExperimentLab explores saved synthetic evidence.")
    provenance(bundle)


def comparison(bundle):
    benchmark = bundle["benchmark"]["benchmark"]
    st.caption("FORECASTING / FROZEN BENCHMARK")
    st.title("Model Comparison")
    st.write("Six methods. The same stations, forecast origins and eligible targets. Saved results in original recorded-start units.")
    controls = st.columns([1, 1, 2])
    with controls[0]:
        split = st.radio("Evaluation split", ("test", "validation"), format_func=str.title, key="split")
    with controls[1]:
        metric = st.selectbox("Error metric", ("MAE", "RMSE", "WAPE_percent"),
                              format_func=lambda x: "WAPE (%)" if x == "WAPE_percent" else x, key="metric")
    with controls[2]:
        selected = st.multiselect("Methods to compare", list(MODELS), default=list(MODELS),
                                  format_func=MODELS.get, key="models")
    if not selected:
        st.warning("Select at least one method to compare.")
        return
    ranked = ranked_rows(benchmark, split, metric, selected)
    best = ranked[0]
    cols = st.columns(3)
    cols[0].metric("Lowest selected " + metric.replace("_percent", ""), MODELS[best["model_id"]])
    cols[1].metric("Scored target positions", f"{best['targets']:,}")
    cols[2].metric("Station-origin windows", "9,200")
    st.plotly_chart(ranking_chart(ranked, metric, split), width="stretch", theme=None, key="ranking")
    frame = pd.DataFrame([{ "Rank (selected)": r["rank"], "Model": MODELS[r["model_id"]],
                            "MAE": r["MAE"], "RMSE": r["RMSE"], "WAPE (%)": r["WAPE_percent"]} for r in ranked])
    st.dataframe(frame, hide_index=True, width="stretch",
                 column_config={k: st.column_config.NumberColumn(format="%.4f") for k in ("MAE", "RMSE", "WAPE (%)")})
    st.caption("Full-cohort saved metrics only. These values never come from the 648-row demonstration subset. "
               "All forecasts use the fixed max(0, prediction) rule, without rounding; TFT point forecasts use q50.")
    with st.expander("What do the metrics measure?", expanded=False):
        xgb_mae = benchmark["metrics"]["xgboost"]["MAE"]
        st.markdown("**MAE** = Σ |prediction - actual| / N. Average absolute error, in starts per station-hour. "
                    f"XGBoost's test MAE of {xgb_mae:.4f} means about {xgb_mae:.2f} recorded starts of average absolute discrepancy-not percentage error.")
        st.markdown("**RMSE** = √(Σ (prediction - actual)² / N). Same units as MAE, but larger misses receive more weight.")
        st.markdown("**WAPE (%)** = 100 × Σ |prediction - actual| / Σ actual. Total absolute error relative to total recorded starts; not classification accuracy. It is undefined when total actual is zero.")
        st.write("Lower absolute test errors do not automatically imply better relative errors: test activity was lower and WAPE increased. Station/hour patterns are descriptive, not causal findings.")
    st.subheader("Compare with a seasonal baseline")
    baseline = st.radio("Baseline reference", ("lag-24", "lag-168"), format_func=MODELS.get, horizontal=True)
    source = benchmark["metrics" if split == "test" else "validation_metrics"]
    improvements = [{"Model": MODELS[r["model_id"]], "Error reduction (%)": improvement_percent(r[metric], source[baseline][metric])} for r in ranked]
    st.dataframe(improvements, hide_index=True, width="stretch",
                 column_config={"Error reduction (%)": st.column_config.NumberColumn(format="%.2f")})
    st.caption("100 × (baseline error - model error) / baseline error. Negative means worse than that baseline. No statistical significance is implied.")
    st.subheader("What each approach contributes")
    for model in selected:
        title, body = TRADEOFFS[model]
        with st.expander(MODELS[model] + " · " + title):
            st.write(body)
    st.subheader("Saved full-test diagnostics")
    st.caption("Always October-December 2025, regardless of the split selector above. These post-hoc tables do not refit or select models.")
    dimension = st.radio("Diagnostic view", ("month", "hour"), format_func=lambda x: "By month" if x == "month" else "By NYC-local hour", horizontal=True)
    st.plotly_chart(diagnostic_chart(bundle["diagnostics"][dimension], dimension, selected), width="stretch", theme=None, key="diagnostics")
    st.caption("Both fall-back 01:00 hours contribute to the local-hour summary; their underlying UTC identities remain distinct. Each origin scores 24 elapsed hours, not a padded 23/25-hour calendar day.")
    st.subheader("Uncertainty and evaluation boundaries")
    uncertainty = benchmark["TFT_uncertainty"]
    st.write(f"TFT q10-q90 test interval coverage: **{100*uncertainty['interval_coverage']:.3f}%** "
             f"(nominal 80%). Mean test pinball loss: **{uncertainty['mean_pinball']:.6f}**. Aggregate coverage does not prove calibration at every station or hour.")
    st.warning("This is a retrospective rolling-origin benchmark. MLP and LSTM were added after initial test scores were disclosed: the six-model extension is not fully blind. Actual publication latency was not replayed. Results do not establish physical demand, operating availability or live operational performance.")
    provenance(bundle)
