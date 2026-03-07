"""
Sidebar component: topic input, configuration, and run button.
"""

import streamlit as st
from typing import Dict, Any, Optional


def render_sidebar() -> Optional[Dict[str, Any]]:
    """
    Render the sidebar with pipeline configuration.

    Returns:
        Dict with run parameters if Run button clicked, else None.
    """
    st.sidebar.header("Pipeline Configuration")

    topic = st.sidebar.text_area(
        "Healthcare Topic",
        value="",
        height=100,
        help="e.g., Effectiveness of telemedicine interventions for Type 2 diabetes management in adults",
    )

    scope_instructions = st.sidebar.text_area(
        "Scope Instructions",
        value="",
        height=80,
        help="e.g., Focus on adults 18+, exclude pediatric cases",
    )

    target_audience = st.sidebar.selectbox(
        "Target Audience",
        options=[
            "clinical practitioners",
            "researchers",
            "policy makers",
            "general public",
        ],
    )

    report_format = st.sidebar.selectbox(
        "Report Format",
        options=["clinical_brief", "full_report"],
    )

    max_quality_iterations = st.sidebar.slider(
        "Max Quality Iterations",
        min_value=1,
        max_value=5,
        value=3,
    )

    st.sidebar.markdown("---")

    run_clicked = st.sidebar.button("Run Pipeline")

    if run_clicked and topic.strip():
        return {
            "topic": topic.strip(),
            "scope_instructions": scope_instructions.strip(),
            "target_audience": target_audience,
            "report_format": report_format,
            "max_quality_iterations": max_quality_iterations,
        }

    if run_clicked and not topic.strip():
        st.sidebar.warning("Please enter a healthcare topic.")

    return None
