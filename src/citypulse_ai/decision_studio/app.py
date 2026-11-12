"""Five implemented, independently verifiable offline workspaces."""
import streamlit as st

from .artifacts import ArtifactError, load_bundle
from .ui import configure, sidebar, upcoming
from .workspaces import comparison, overview
from .explorer import forecast_explorer


def main():
    configure()
    workspace = sidebar()
    if workspace == "ExperimentLab & Methodology":
        from .experiment_workspace import experiment_workspace
        experiment_workspace()
        return
    if workspace == "SQLPilot":
        # Independent bundle: a forecasting artifact fault must not disable SQL.
        from .sql_workspace import sqlpilot_workspace
        sqlpilot_workspace()
        return
    try:
        bundle = load_bundle()
    except ArtifactError:
        st.error("Verified offline demo artifacts are missing or invalid. No results can be displayed.")
        st.caption("Regenerate the bundle from the approved local sources using scripts/build_decision_studio_bundle.py.")
        st.stop()
    if workspace == "Overview":
        overview(bundle)
    elif workspace == "Model Comparison":
        comparison(bundle)
    elif workspace == "Forecast Explorer":
        forecast_explorer(bundle)
    else:
        upcoming(workspace)


if __name__ == "__main__":
    main()
