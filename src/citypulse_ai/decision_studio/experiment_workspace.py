"""Interactive presentation of the existing, fully synthetic ExperimentLab study."""
from pathlib import Path

import streamlit as st

from .artifacts import ArtifactError, sha256
from .experiment_charts import effect_chart, planning_chart, station_chart
from .experiment_data import ESTIMATORS, conclusions, load_experiment, p_value_label, station_rows
from .ui import DOCS


def experiment_workspace():
    st.title("ExperimentLab & Methodology")
    st.caption("SYNTHETIC RESEARCH / SAVED EVIDENCE / NO LIVE SIMULATION")
    st.warning("Fully synthetic study. No real Citi Bike intervention occurred; these are not observed historical treatment effects.")
    try:
        bundle = load_experiment()
    except ArtifactError:
        st.error("Saved synthetic experiment is missing or invalid. No statistical results can be displayed.")
        return
    experiment, manifest = bundle["experiment"], bundle["manifest"]
    rows = station_rows(experiment)
    st.markdown("**Question:** In a simulated redistribution intervention, does assigning a station to treatment increase recorded starts? "
                "The estimand is the station-average intention-to-treat effect over a fixed 28-day post period.")
    columns = st.columns(4)
    for column, label, value in zip(columns, ("Synthetic stations", "Treatment / control", "Post-period days", "Practical threshold"),
                                     ("100", "50 / 50", "28", "+2 starts/day")):
        column.metric(label, value)
    with st.expander("Recorded experimental design"):
        st.markdown("The simulator uses heterogeneous, stable latent station rates and Poisson counts. "
                    "A hypothetical intervention adds +3 starts/station/day to assigned treatment units. "
                    "This known simulation input is not an observed real-world benefit.")
        st.markdown(f'Simulation seed **{experiment["design"]["seed"]}** is separate from assignment seed '
                    f'**{experiment["design"]["assignment_seed"]}**. A seeded permutation assigns exactly half '
                    "of the 100 SIM station IDs to treatment, the rest to control. Assignment does not depend on post outcomes.")
        st.json(experiment["periods"])
        st.caption("Period endpoints are elapsed simulation-day indices [start, end), not historical dates. "
                   "The two calibration periods precede the baseline and fixed post period. Alpha = 0.05; no live randomization occurs here.")
    effect_tab, evidence_tab, planning_tab, method_tab = st.tabs(
        ["Effects & interpretation", "Station evidence", "Power & MDE", "Methodology"])
    with effect_tab:
        selected = st.selectbox("Estimator", list(ESTIMATORS), format_func=ESTIMATORS.get)
        compare = st.checkbox("Compare both saved estimates", value=True)
        row = experiment[selected]
        columns = st.columns(3)
        columns[0].metric("Effect (starts / station / day)", f'{row["effect"]:+.4f}')
        columns[1].metric("95% Welch confidence interval", f'[{row["ci_low"]:.4f}, {row["ci_high"]:.4f}]')
        columns[2].metric("Two-sided p-value", p_value_label(row["p_value"]))
        st.caption(f'Standard error: {row["standard_error"]:.4f} · Welch degrees of freedom: {row["df"]:.2f}. '
                   "Dotted line: no effect. Dashed line: +2 practical threshold.")
        st.plotly_chart(effect_chart(experiment, selected, compare), width="stretch")
        primary = conclusions(experiment["cuped"], experiment["design"]["practical_threshold"], experiment["design"]["alpha"])
        if primary["statistically_positive"] and not primary["practical_threshold_supported"]:
            st.info("Primary CUPED conclusion: statistically positive, but practical threshold uncertain. "
                    "The 95% interval [1.7190, 3.7871] excludes zero; its lower bound does not exceed +2.")
        st.caption("CUPED remains the pre-specified primary estimator. Switching the display does not select a new study conclusion.")
        st.dataframe([{"estimator": ESTIMATORS[key], "treatment mean": experiment[key]["treatment_mean"],
                       "control mean": experiment[key]["control_mean"], "SE": experiment[key]["standard_error"]}
                      for key in ESTIMATORS], hide_index=True, width="stretch")
        with st.expander("How to read the inference"):
            st.markdown("**Welch:** effect = mean(treatment) - mean(control); SE = sqrt(s²T/nT + s²C/nC). "
                        "A t distribution with Satterthwaite degrees of freedom gives the two-sided p-value and interval, "
                        "allowing unequal arm variances. Stations, not station-days, are the independent units.")
            st.markdown("**95% confidence:** over repeated studies following this procedure, approximately 95% of intervals "
                        "would cover the fixed true effect under the assumptions. This is not a 95% probability assigned "
                        "to this particular interval, an individual-outcome interval, or a TFT prediction-quantile band.")
            st.markdown("**Assumptions:** randomized assignment, independent station units, no interference, fixed complete outcomes "
                        "and a suitable sampling approximation. Real redistribution could violate independence or cause spillovers.")
    with evidence_tab:
        arm = st.selectbox("Station arm", ["Both", "Treatment", "Control"])
        adjusted = st.checkbox("Show CUPED-adjusted station means", value=False)
        filtered = [r for r in rows if arm == "Both" or r["arm"] == arm]
        st.plotly_chart(station_chart(filtered, adjusted), width="stretch")
        st.dataframe(filtered, hide_index=True, width="stretch")
        st.caption("Each point is one synthetic station's 28-day mean. SIM IDs are unrelated to the 100 real forecasting stations. "
                   "CUPED-adjusted values are statistical transformations, not additional observed counts.")
        with st.expander("CUPED and provenance"):
            st.code("Y_adjusted = Y - theta * (X - mean(X))")
            st.markdown(f'Theta **{experiment["calibration"]["theta"]:.6f}** was fitted on two earlier, disjoint pre-treatment periods. '
                        "X is the later pre-assignment baseline; Y is the post-period station mean. "
                        "The saved coefficient is reused, never fitted to post outcomes here.")
            st.markdown(f'Effect-estimator variance reduction: **{100*experiment["variance_reduction_fraction"]:.3f}%**. '
                        "The unusually strong reduction depends on this stable synthetic data-generating process; it is not a real-world guarantee.")
            st.caption("Earlier calibration unit records and an exact original implementation hash were not saved in the canonical JSON. "
                       "The coefficient's saved scope is checked; a calibration refit cannot be independently reproduced from this bundle alone.")
    with planning_tab:
        metric = st.selectbox("Planning measure", ["power", "mde"], format_func=lambda x: "Power for +2" if x == "power" else "Minimum detectable effect")
        st.plotly_chart(planning_chart(experiment, metric), width="stretch")
        st.dataframe([{"estimator": label, "power at +2": experiment[key]["power"], "MDE at 80% power": experiment[key]["mde"],
                       "required stations per arm": experiment[key]["required_stations_per_arm"]}
                      for key, label in (("planning_raw", "Unadjusted"), ("planning_cuped", "CUPED"))], hide_index=True, width="stretch")
        st.markdown("Planning uses pre-treatment calibration SDs, 50 stations per arm, two-sided alpha 0.05 and target power 80%. "
                    "Power is the chance of detecting a specified effect under the planning model. MDE is the effect detectable at that target power. "
                    "More independent stations generally increase power and lower MDE. These are approximate noncentral-t calculations, not observed power.")
        with st.expander("Saved simulation checks - no rerun"):
            st.json(experiment["simulation_checks"])
            st.caption("Only 200 saved repetitions per scenario. 200/200 detections is not proof of perfect power; Monte Carlo uncertainty remains.")
    with method_tab:
        sections = {
            "Historical data and temporal evaluation": "Validated Citi Bike sources feed DuckDB and Parquet station-hour reconciliation. "
                "24 NYC-local target months (Jan 2024-Dec 2025) use 25 archives including the January boundary. "
                "89,995,902 accepted recorded starts underpin the modelling window. Real station selection and scalers use Jan 2024-Jun 2025 only; "
                "validation is Jul-Sep 2025 and test Oct-Dec 2025. Windows use 336 past and 24 future elapsed UTC hours, with NYC calendar/DST features. "
                "Certified accepted-record zeros are distinct from unknown data and physical demand; station operating availability remains unknown without independent evidence.",
            "Six forecasting approaches": "XGBoost, PyTorch MLP, TensorFlow LSTM, native PyTorch TFT, lag-24 and lag-168 share aligned targets. "
                "Final validation/test comparisons use 220,800 targets each. MLP and LSTM were follow-up additions after initial test results were disclosed, "
                "so the six-method comparison is not a fully blind experiment. Forecast Explorer presents saved retrospective windows, not live inference.",
            "SQLPilot and ExperimentLab": "SQLPilot's separate application uses Streamlit → LangGraph → MCP client/server → restricted DuckDB. "
                "This Studio runs curated approved SQL only, with no LLM calls. The original nine-case hosted-LLM baseline achieved 6/9 correct outcomes. "
                "ExperimentLab is a separate synthetic randomized study, not causal evidence from the observational forecasting history.",
            "Historical engineering and deployment evidence": "The six-model FastAPI/Docker application and XGBoost-only authenticated Cloud Run deployment "
                "have historical verification evidence. Docker, Cloud Build, private Artifact Registry, Terraform, GitHub Actions CI/CD and keyless WIF "
                "support that engineering workflow. This offline workspace does not query GCP or establish that any service is currently active.",
            "Why verified offline artifacts": "Small checksum-verified bundles make the evidence inspectable without cloud credentials, model runtimes, "
                "historical dataset downloads, paid APIs or live simulations. Checksums detect changes relative to this repository's manifest; they are not external signatures. "
                "Study controls select saved evidence and display transformations, not new experimental results."
        }
        for title, content in sections.items():
            with st.expander(title):
                st.markdown(content)
        st.markdown(f'[ExperimentLab design]({DOCS}experimentlab.md) · [Forecasting protocol]({DOCS}tft_station_forecasting_design.md) · '
                    f'[Historical Cloud Run verification]({DOCS}cloudrun_xgboost.md)')
    with st.expander("Verified source and report"):
        st.caption("Byte-identical saved experiment.json; checksum-verified against the original release receipt and tracked HTML report. No simulation was rerun.")
        st.json(manifest)
        report = Path(__file__).resolve().parents[3] / "docs/experimentlab_report.html"
        expected = manifest["sources"]["docs/experimentlab_report.html"]
        if report.is_file() and not report.is_symlink() and sha256(report) == expected["sha256"]:
            st.download_button("Download saved synthetic HTML report", report.read_bytes(), "experimentlab_report.html", "text/html")
        else:
            st.warning("The original HTML report is unavailable or differs from its recorded checksum; download disabled.")
