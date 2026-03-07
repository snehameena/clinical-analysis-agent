"""
Pydantic schemas for the FastAPI endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict


class PipelineRequest(BaseModel):
    """Request body for launching a pipeline run."""
    topic: str = Field(..., description="Healthcare topic to investigate")
    scope_instructions: str = Field("", description="Scope constraints for the research")
    target_audience: str = Field("clinical practitioners", description="Target audience for the report")
    report_format: str = Field("clinical_brief", description="Report format (clinical_brief, full_report)")
    max_quality_iterations: int = Field(3, ge=1, le=5, description="Maximum quality revision loops")


class PipelineResponse(BaseModel):
    """Response after launching a pipeline run."""
    run_id: str
    status: str
    message: str


class StatusResponse(BaseModel):
    """Response for pipeline status polling."""
    run_id: str
    status: str
    current_agent: Optional[str] = None
    quality_iteration: int = 0
    error_message: Optional[str] = None


class ReportResponse(BaseModel):
    """Response containing the final report."""
    run_id: str
    report_markdown: str
    report_sections: Dict[str, str] = {}
    citations: List[str] = []
    quality_score: float = 0.0
    quality_verdict: str = ""
    word_count: int = 0
