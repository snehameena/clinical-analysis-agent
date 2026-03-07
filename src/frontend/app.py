"""
Streamlit main application for the Healthcare Content Intelligence Pipeline.
"""

import sys
from pathlib import Path

# Add project root to sys.path so `src.*` imports work when
# Streamlit is launched with `streamlit run src/frontend/app.py`
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st
from src.frontend.api_client import PipelineAPIClient
from src.frontend.components.sidebar import render_sidebar
from src.frontend.components.progress import render_progress
from src.frontend.components.report_view import render_report
from src.env import load_env

load_env()

st.set_page_config(
    page_title="Healthcare Intelligence Pipeline",
    page_icon="🏥",
    layout="wide",
)

st.title("Healthcare Content Intelligence Pipeline")
st.caption("Multi-agent clinical evidence synthesis powered by Claude and LangGraph")


def main():
    client = PipelineAPIClient()

    # Initialize session state
    if "run_id" not in st.session_state:
        st.session_state.run_id = None
    if "report_data" not in st.session_state:
        st.session_state.report_data = None

    # Sidebar: configuration and run button
    run_params = render_sidebar()

    if run_params:
        # Launch new pipeline run
        st.session_state.report_data = None
        try:
            result = client.start_run(**run_params)
            st.session_state.run_id = result["run_id"]
        except Exception as e:
            st.error(f"Failed to start pipeline: {e}")
            return

    # Main content area
    if st.session_state.run_id and not st.session_state.report_data:
        # Show progress
        col1, col2 = st.columns([2, 1])
        with col1:
            final_status = render_progress(client, st.session_state.run_id)

        if final_status and final_status.get("status") == "completed":
            try:
                report_data = client.get_report(st.session_state.run_id)
                st.session_state.report_data = report_data
            except Exception as e:
                st.error(f"Failed to retrieve report: {e}")

    if st.session_state.report_data:
        render_report(st.session_state.report_data)
    elif not st.session_state.run_id:
        # Landing state
        st.info("Configure your healthcare topic in the sidebar and click **Run Pipeline** to begin.")

        st.markdown("""
        ### How it works
        1. **Coordinator** — Plans research queries and defines scope
        2. **Research** — Searches medical literature via Tavily
        3. **Analysis** — Extracts clinical claims and identifies evidence gaps
        4. **Writing** — Composes a structured clinical report
        5. **Quality** — Reviews and may request revisions (up to N iterations)
        """)


if __name__ == "__main__":
    main()
