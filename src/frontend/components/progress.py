"""
Progress component: polls /status endpoint and shows agent progress.
"""

import time
import streamlit as st
from src.frontend.api_client import PipelineAPIClient

AGENT_ORDER = ["coordinator", "research", "analysis", "writing", "quality"]


def render_progress(client: PipelineAPIClient, run_id: str) -> dict:
    """
    Poll pipeline status and display progress bars until completion.

    Args:
        client: API client instance
        run_id: Pipeline run ID

    Returns:
        Final status dict from the API.
    """
    progress_container = st.container()
    status_text = st.empty()

    with progress_container:
        st.subheader("Pipeline Progress")
        agent_bars = {}
        for agent_name in AGENT_ORDER:
            col1, col2 = st.columns([1, 4])
            with col1:
                st.text(agent_name.capitalize())
            with col2:
                agent_bars[agent_name] = st.progress(0)

    final_status = None

    while True:
        try:
            status = client.get_status(run_id)
        except Exception as e:
            status_text.error(f"Error polling status: {e}")
            time.sleep(2)
            continue

        current_agent = status.get("current_agent", "")
        pipeline_status = status.get("status", "unknown")
        iteration = status.get("quality_iteration", 0)

        # Update progress bars
        for i, agent_name in enumerate(AGENT_ORDER):
            if agent_name == current_agent:
                agent_bars[agent_name].progress(50)
            elif AGENT_ORDER.index(agent_name) < AGENT_ORDER.index(current_agent) if current_agent in AGENT_ORDER else False:
                agent_bars[agent_name].progress(100)

        quality_text = f" (iteration {iteration})" if iteration > 0 else ""
        status_text.info(
            f"Status: **{pipeline_status}** | Current agent: **{current_agent}**{quality_text}"
        )

        if pipeline_status in ("completed", "failed"):
            # Mark all done
            for agent_name in AGENT_ORDER:
                if pipeline_status == "completed":
                    agent_bars[agent_name].progress(100)
            final_status = status
            break

        time.sleep(2)

    if pipeline_status == "failed":
        st.error(f"Pipeline failed: {status.get('error_message', 'Unknown error')}")
    else:
        st.success("Pipeline completed successfully!")

    return final_status
