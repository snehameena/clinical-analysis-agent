"""
Pipeline endpoints: run, status, stream, report.
"""

import asyncio
import uuid
import json
import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException, BackgroundTasks
from sse_starlette.sse import EventSourceResponse
from src.api.models import PipelineRequest, PipelineResponse, StatusResponse, ReportResponse
from src.api.dependencies import get_graph, get_run_store
from src.state.schema import PipelineState

router = APIRouter(prefix="/api/v1/pipeline", tags=["pipeline"])
logger = logging.getLogger(__name__)


def _build_initial_state(request: PipelineRequest, run_id: str) -> PipelineState:
    """Build the initial pipeline state from a request."""
    return PipelineState(
        run_id=run_id,
        topic=request.topic,
        scope_instructions=request.scope_instructions,
        target_audience=request.target_audience,
        report_format=request.report_format,
        requested_at=datetime.now(),
        research_queries=[],
        scope_boundaries={"in_scope": [], "out_of_scope": []},
        priority_subtopics=[],
        raw_sources=[],
        deduplicated_sources=[],
        research_summary="",
        clinical_claims=[],
        evidence_gaps=[],
        contradictions=[],
        statistical_findings=[],
        analysis_narrative="",
        report_sections={},
        report_markdown="",
        citations=[],
        report_word_count=0,
        quality_issues=[],
        quality_verdict="pending",
        quality_score=0.0,
        revision_instructions="",
        quality_iteration=0,
        max_quality_iterations=request.max_quality_iterations,
        should_revise=False,
        current_agent="",
        agent_history=[],
        pipeline_status="pending",
        error_message=None,
    )


async def _run_pipeline(run_id: str):
    """Background task that executes the pipeline graph."""
    store = get_run_store()
    state = store[run_id]
    state["pipeline_status"] = "running"

    try:
        graph = get_graph()
        result = await graph.ainvoke(state)
        result["pipeline_status"] = "completed"
        store[run_id] = result
    except Exception as e:
        logger.error(f"Pipeline run {run_id} failed: {e}")
        state["pipeline_status"] = "failed"
        state["error_message"] = str(e)
        store[run_id] = state


@router.post("/run", response_model=PipelineResponse)
async def run_pipeline(request: PipelineRequest, background_tasks: BackgroundTasks):
    """Launch a new pipeline run as a background task."""
    run_id = str(uuid.uuid4())
    state = _build_initial_state(request, run_id)

    store = get_run_store()
    store[run_id] = state

    background_tasks.add_task(_run_pipeline, run_id)

    return PipelineResponse(
        run_id=run_id,
        status="pending",
        message="Pipeline run started",
    )


@router.get("/{run_id}/status", response_model=StatusResponse)
async def get_status(run_id: str):
    """Poll the current status of a pipeline run."""
    store = get_run_store()
    if run_id not in store:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    state = store[run_id]
    return StatusResponse(
        run_id=run_id,
        status=state.get("pipeline_status", "unknown"),
        current_agent=state.get("current_agent"),
        quality_iteration=state.get("quality_iteration", 0),
        error_message=state.get("error_message"),
    )


@router.get("/{run_id}/stream")
async def stream_status(run_id: str):
    """SSE stream that pushes status updates until pipeline completes."""
    store = get_run_store()
    if run_id not in store:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    async def event_generator():
        while True:
            state = store.get(run_id, {})
            status = state.get("pipeline_status", "unknown")

            data = json.dumps({
                "status": status,
                "current_agent": state.get("current_agent", ""),
                "quality_iteration": state.get("quality_iteration", 0),
                "error_message": state.get("error_message"),
            })

            yield {"event": "status", "data": data}

            if status in ("completed", "failed"):
                break

            await asyncio.sleep(2)

    return EventSourceResponse(event_generator())


@router.get("/{run_id}/report", response_model=ReportResponse)
async def get_report(run_id: str):
    """Retrieve the final report for a completed pipeline run."""
    store = get_run_store()
    if run_id not in store:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    state = store[run_id]
    status = state.get("pipeline_status", "unknown")

    if status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Pipeline is not completed yet (status: {status})",
        )

    return ReportResponse(
        run_id=run_id,
        report_markdown=state.get("report_markdown", ""),
        report_sections=state.get("report_sections", {}),
        citations=state.get("citations", []),
        quality_score=state.get("quality_score", 0.0),
        quality_verdict=state.get("quality_verdict", ""),
        word_count=state.get("report_word_count", 0),
    )
