"""
LangGraph node wrapper functions for each agent.
Each function instantiates the agent, runs execute(), and handles errors.
"""

import time
import logging
from src.state.schema import PipelineState
from src.agents.coordinator import CoordinatorAgent
from src.agents.research import ResearchAgent
from src.agents.analysis import AnalysisAgent
from src.agents.writing import WritingAgent
from src.agents.quality import QualityAgent
from src.debug.agent_io import log_agent_state

logger = logging.getLogger(__name__)


async def coordinator_node(state: PipelineState) -> PipelineState:
    """LangGraph node wrapper for CoordinatorAgent."""
    logger.info("Starting coordinator node")
    state["current_agent"] = "coordinator"
    state["pipeline_status"] = "running"
    start = time.time()
    try:
        agent = CoordinatorAgent()
        agent._current_run_id = state.get("run_id")
        log_agent_state(
            when="before",
            agent_name="coordinator",
            state=state,
            extra={
                "topic": state.get("topic"),
                "scope_instructions": state.get("scope_instructions"),
                "target_audience": state.get("target_audience"),
            },
        )
        state = await agent.execute(state)
        log_agent_state(
            when="after",
            agent_name="coordinator",
            state=state,
            extra={
                "research_queries": state.get("research_queries"),
                "scope_boundaries": state.get("scope_boundaries"),
                "priority_subtopics": state.get("priority_subtopics"),
            },
        )
    except Exception as e:
        logger.error(f"Coordinator node failed: {e}")
        state["error_message"] = f"Coordinator failed: {e}"
        state["pipeline_status"] = "failed"
        log_agent_state(when="error", agent_name="coordinator", state=state, extra={"error": str(e)})
        raise
    logger.info(f"Coordinator node completed in {time.time() - start:.1f}s")
    return state


async def research_node(state: PipelineState) -> PipelineState:
    """LangGraph node wrapper for ResearchAgent."""
    logger.info("Starting research node")
    state["current_agent"] = "research"
    start = time.time()
    try:
        agent = ResearchAgent()
        agent._current_run_id = state.get("run_id")
        log_agent_state(
            when="before",
            agent_name="research",
            state=state,
            extra={"research_queries": state.get("research_queries")},
        )
        state = await agent.execute(state)
        sources = state.get("deduplicated_sources", []) or []
        top_titles = [s.get("title") for s in sources[:5] if isinstance(s, dict)]
        log_agent_state(
            when="after",
            agent_name="research",
            state=state,
            extra={
                "raw_sources_count": len(state.get("raw_sources", []) or []),
                "deduplicated_sources_count": len(sources),
                "top_source_titles": top_titles,
                "research_summary": state.get("research_summary"),
            },
        )
    except Exception as e:
        logger.error(f"Research node failed: {e}")
        state["error_message"] = f"Research failed: {e}"
        state["pipeline_status"] = "failed"
        log_agent_state(when="error", agent_name="research", state=state, extra={"error": str(e)})
        raise
    logger.info(f"Research node completed in {time.time() - start:.1f}s")
    return state


async def analysis_node(state: PipelineState) -> PipelineState:
    """LangGraph node wrapper for AnalysisAgent."""
    logger.info("Starting analysis node")
    state["current_agent"] = "analysis"
    start = time.time()
    try:
        agent = AnalysisAgent()
        agent._current_run_id = state.get("run_id")
        log_agent_state(
            when="before",
            agent_name="analysis",
            state=state,
            extra={"deduplicated_sources_count": len(state.get("deduplicated_sources", []) or [])},
        )
        state = await agent.execute(state)
        log_agent_state(
            when="after",
            agent_name="analysis",
            state=state,
            extra={
                "clinical_claims_count": len(state.get("clinical_claims", []) or []),
                "evidence_gaps": state.get("evidence_gaps"),
                "contradictions": state.get("contradictions"),
                "analysis_narrative": state.get("analysis_narrative"),
            },
        )
    except Exception as e:
        logger.error(f"Analysis node failed: {e}")
        state["error_message"] = f"Analysis failed: {e}"
        state["pipeline_status"] = "failed"
        log_agent_state(when="error", agent_name="analysis", state=state, extra={"error": str(e)})
        raise
    logger.info(f"Analysis node completed in {time.time() - start:.1f}s")
    return state


async def writing_node(state: PipelineState) -> PipelineState:
    """LangGraph node wrapper for WritingAgent."""
    logger.info("Starting writing node")
    state["current_agent"] = "writing"
    start = time.time()
    try:
        agent = WritingAgent()
        agent._current_run_id = state.get("run_id")
        log_agent_state(
            when="before",
            agent_name="writing",
            state=state,
            extra={
                "quality_iteration": state.get("quality_iteration"),
                "revision_instructions": state.get("revision_instructions"),
                "analysis_narrative": state.get("analysis_narrative"),
            },
        )
        state = await agent.execute(state)
        log_agent_state(
            when="after",
            agent_name="writing",
            state=state,
            extra={
                "report_word_count": state.get("report_word_count"),
                "citations_count": len(state.get("citations", []) or []),
                "report_markdown": state.get("report_markdown"),
            },
        )
    except Exception as e:
        logger.error(f"Writing node failed: {e}")
        state["error_message"] = f"Writing failed: {e}"
        state["pipeline_status"] = "failed"
        log_agent_state(when="error", agent_name="writing", state=state, extra={"error": str(e)})
        raise
    logger.info(f"Writing node completed in {time.time() - start:.1f}s")
    return state


async def quality_node(state: PipelineState) -> PipelineState:
    """LangGraph node wrapper for QualityAgent."""
    logger.info("Starting quality node")
    state["current_agent"] = "quality"
    start = time.time()
    try:
        agent = QualityAgent()
        agent._current_run_id = state.get("run_id")
        log_agent_state(
            when="before",
            agent_name="quality",
            state=state,
            extra={
                "report_word_count": state.get("report_word_count"),
                "report_markdown": state.get("report_markdown"),
            },
        )
        state = await agent.execute(state)
        log_agent_state(
            when="after",
            agent_name="quality",
            state=state,
            extra={
                "quality_verdict": state.get("quality_verdict"),
                "quality_score": state.get("quality_score"),
                "quality_issues": state.get("quality_issues"),
                "revision_instructions": state.get("revision_instructions"),
            },
        )
    except Exception as e:
        logger.error(f"Quality node failed: {e}")
        state["error_message"] = f"Quality failed: {e}"
        state["pipeline_status"] = "failed"
        log_agent_state(when="error", agent_name="quality", state=state, extra={"error": str(e)})
        raise
    logger.info(f"Quality node completed in {time.time() - start:.1f}s")
    return state
