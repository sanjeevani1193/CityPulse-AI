"""Five-workspace navigation; only two workspaces implemented in this milestone."""
import streamlit as st

from .artifacts import ArtifactError, load_bundle
from .ui import configure, sidebar, upcoming
from .workspaces import comparison, overview


def main():
    configure()
    workspace = sidebar()
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
    else:
        upcoming(workspace)


if __name__ == "__main__":
    main()
