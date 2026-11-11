"""Shared presentation and public documentation links; no secrets or backend calls."""
import streamlit as st

REPOSITORY = "https://github.com/sanjeevani1193/CityPulse-AI"
DOCS = REPOSITORY + "/blob/main/docs/"
WORKSPACES = ("Overview", "Forecast Explorer", "Model Comparison", "SQLPilot", "ExperimentLab & Methodology")

# Static, scoped CSS. Native theme tokens own widget/text colours; selectors use
# semantic roles/test IDs and our own classes, never Emotion's hashed classes.
STYLES = """<style>
[data-testid="stAppViewContainer"] {color-scheme: light;}
[data-testid="stHeader"] {
  background: #F3F7FA; border-bottom: 1px solid #E0E9EF;
}
[data-testid="stHeader"] button {color: #21384B;}
[data-testid="stMainBlockContainer"] {
  max-width: 1280px; padding: 3.5rem 2.25rem 3rem;
}
[data-testid="stMain"] h1,
[data-testid="stMain"] h2,
[data-testid="stMain"] h3 {letter-spacing: -.025em;}
[data-testid="stMain"] h1 {font-size: clamp(1.9rem, 3.4vw, 2.8rem);}
[data-testid="stMain"] h2 {margin-top: 1rem;}
[data-testid="stMain"] [data-testid="stCaptionContainer"] p {color: #506676;}
[data-testid="stMain"] input::placeholder {color: #506676; opacity: 1;}
[data-testid="stSidebar"] {border-right: 1px solid #365469;}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {padding-top: .75rem;}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {color: #FFFFFF;}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {color: #CBDCE7;}
[data-testid="stSidebar"] a {color: #9BE6E4; text-decoration: underline; text-underline-offset: 3px;}
[data-testid="stSidebar"] a:hover {color: #FFFFFF;}
[data-testid="stSidebar"] hr {border-color: #365469;}
[data-testid="stSidebar"] [role="radiogroup"] {gap: .35rem;}
[data-testid="stSidebar"] [role="radiogroup"] label {
  width: 100%; box-sizing: border-box; padding: .65rem .6rem;
  border: 1px solid transparent; border-left: 3px solid transparent;
  border-radius: .6rem; transition: background-color .12s ease;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {background: #22465A;}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
  background: #167E87; border-left-color: #9BE6E4;
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:focus-visible) {
  outline: 2px solid #9BE6E4; outline-offset: 2px;
}
[data-testid="stSidebarCollapseButton"] button {color: #F3F7FA;}
[data-testid="stSidebarCollapsedControl"] button {color: #102D40; background: #F3F7FA;}
[data-testid="stMain"] a {text-underline-offset: 3px;}
[data-testid="stMain"] a:hover {color: #102D40;}
a:focus-visible, button:focus-visible {outline: 2px solid #167E87; outline-offset: 3px;}
[data-testid="stSidebar"] a:focus-visible,
[data-testid="stSidebar"] button:focus-visible {outline-color: #9BE6E4;}
.cp-hero {
  padding: 1.8rem 2rem; border-radius: 16px; background: #102D40; color: #FFFFFF;
  border-left: 5px solid #66CFC7; margin-bottom: 1.6rem;
}
.cp-hero h1 {color: #FFFFFF; margin: .45rem 0 .7rem; line-height: 1.15;}
.cp-hero p {color: #DCE8F0; max-width: 800px; font-size: 1.05rem; line-height: 1.65;}
.cp-kicker {font-size: .72rem; font-weight: 700; letter-spacing: .14em; color: #9BE6E4;}
[data-testid="stMetric"] {
  background: #FFFFFF; border: 1px solid #D2DFE7; border-top: 3px solid #167E87;
  border-radius: 12px; padding: 1rem 1.1rem; min-height: 8.2rem;
}
[data-testid="stMetricLabel"] {color: #40596C; min-height: 2.4rem;}
[data-testid="stMetricValue"] {color: #102D40; font-size: clamp(1.5rem, 2.5vw, 2rem);}
[data-testid="stPlotlyChart"] {
  background: #FFFFFF; border: 1px solid #D2DFE7; border-radius: 12px; overflow: hidden;
}
[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 12px;}
@media(max-width: 700px) {
  [data-testid="stMainBlockContainer"] {padding: 3.75rem 1rem 2rem;}
  .cp-hero {padding: 1.25rem; margin-bottom: 1rem;}
  .cp-hero p {font-size: 1rem;}
  [data-testid="stMetric"] {min-height: 7rem;}
}
@media(prefers-reduced-motion: reduce) {
  [data-testid="stSidebar"] [role="radiogroup"] label {transition: none;}
}
</style>"""


def configure():
    st.set_page_config(page_title="CityPulse AI · Decision Studio", page_icon="🚲", layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown(STYLES, unsafe_allow_html=True)


def sidebar():
    with st.sidebar:
        st.markdown("### CITYPULSE / AI")
        st.caption("DECISION STUDIO · OFFLINE RESEARCH EDITION")
        choice = st.radio("Workspace", WORKSPACES, key="workspace")
        st.divider()
        st.markdown("**Saved evidence. No live backend.**")
        st.caption("Historical forecasts · accepted recorded starts · synthetic experimentation")
        st.markdown(f"[Repository]({REPOSITORY}) · [Methodology]({DOCS}six_model_followup_benchmark.md)")
        st.caption("Explore Overview, Forecast Explorer and Model Comparison. Other workspaces are previews only.")
    return choice


def provenance(bundle):
    with st.expander("Data provenance & reproducibility"):
        manifest = bundle["manifest"]
        st.write("Rankings use the frozen full-test cohort, not the demonstration subset.")
        st.code("Prediction source SHA-256: " + manifest["sources"][
            "outputs/forecasting-v1/saved-test-predictions.parquet"]["sha256"], language=None)
        st.caption(f"Prepared subset: {manifest['subset']['windows']} complete windows / "
                   f"{manifest['subset']['rows']} target positions. Not used for the leaderboard.")
        st.caption(manifest["subset"]["selection_rule"])
        st.markdown(f"[Frozen benchmark]({DOCS}six_model_followup_benchmark.md) · "
                    f"[Label policy]({DOCS}accepted_recorded_starts_labels.md) · "
                    f"[Error diagnostics]({DOCS}forecasting_v1_diagnostics.md)")


def upcoming(workspace):
    st.title(workspace)
    st.info("Upcoming workspace - not implemented in this milestone.")
    descriptions = {
        "Forecast Explorer": "Planned: exact station IDs, historical midnight origins, 24 elapsed-hour saved forecasts, actuals and TFT quantiles. No live forecasting is offered.",
        "SQLPilot": "Planned: curated offline SQL/results over an approved small sample. The existing LangGraph → MCP → secure DuckDB system remains separate; this page makes no LLM or database calls.",
        "ExperimentLab & Methodology": "Planned: the saved synthetic randomized experiment, Welch inference, CUPED, power/MDE and methodological limitations. No real intervention is claimed.",
    }
    st.write(descriptions[workspace])
