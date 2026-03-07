"""
Report view component: tabbed display of report, sections, sources, quality.
"""

import streamlit as st
from typing import Dict, List


def render_report(report_data: dict):
    """
    Render the final report in a tabbed layout.

    Args:
        report_data: Report response dict from the API.
    """
    st.header("Clinical Intelligence Report")

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Quality Score", f"{report_data.get('quality_score', 0):.0f}/100")
    with col2:
        st.metric("Verdict", report_data.get("quality_verdict", "N/A").upper())
    with col3:
        st.metric("Word Count", report_data.get("word_count", 0))

    st.markdown("---")

    # Tabs
    tab_report, tab_sections, tab_sources, tab_quality = st.tabs([
        "Full Report", "Sections", "Sources", "Quality Review"
    ])

    with tab_report:
        _render_full_report(report_data.get("report_markdown", ""))

    with tab_sections:
        _render_sections(report_data.get("report_sections", {}))

    with tab_sources:
        _render_sources(report_data.get("citations", []))

    with tab_quality:
        _render_quality(report_data)


def _render_full_report(markdown: str):
    """Render the full markdown report."""
    if markdown:
        st.markdown(markdown)
    else:
        st.warning("No report content available.")


def _render_sections(sections: Dict[str, str]):
    """Render individual report sections with expanders."""
    if not sections:
        st.info("No sections available.")
        return

    for section_name, content in sections.items():
        with st.expander(section_name, expanded=False):
            st.markdown(content)


def _render_sources(citations: List[str]):
    """Render the list of citations."""
    if not citations:
        st.info("No citations available.")
        return

    st.subheader(f"References ({len(citations)})")
    for i, citation in enumerate(citations, 1):
        st.markdown(f"{i}. {citation}")


def _render_quality(report_data: dict):
    """Render quality review details."""
    score = report_data.get("quality_score", 0)
    verdict = report_data.get("quality_verdict", "N/A")

    st.subheader("Quality Assessment")

    # Score bar
    st.markdown(f"**Score:** {score:.0f}/100")
    st.progress(min(score / 100, 1.0))

    # Verdict badge
    verdict_emoji = {"pass": "✅", "revise": "🔄", "reject": "❌"}
    emoji = verdict_emoji.get(verdict, "❓")
    st.markdown(f"**Verdict:** {emoji} {verdict.upper()}")
