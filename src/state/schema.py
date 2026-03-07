"""
PipelineState schema for healthcare intelligence pipeline.
Defines the complete state structure passed through all agents.
"""

from typing import TypedDict, Optional, List, Dict, Annotated
from datetime import datetime
import operator
from enum import Enum


class QualityVerdict(str, Enum):
    """Quality assessment verdicts"""
    PASS = "pass"
    REVISE = "revise"
    REJECT = "reject"


class PipelineStatus(str, Enum):
    """Pipeline execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class EvidenceLevel(str, Enum):
    """Clinical evidence hierarchy"""
    SYSTEMATIC_REVIEW = "Systematic Review"
    RCT = "Randomized Controlled Trial"
    COHORT_STUDY = "Cohort Study"
    CASE_REPORT = "Case Report"
    OPINION = "Expert Opinion"


class IssueSeverity(str, Enum):
    """Quality issue severity levels"""
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"


class Source(TypedDict, total=False):
    """Individual research source"""
    url: str
    title: str
    snippet: str
    relevance_score: float
    evidence_level: str
    publication_date: Optional[str]
    authors: Optional[List[str]]


class ClinicalClaim(TypedDict, total=False):
    """Extracted clinical finding"""
    claim: str
    evidence_level: str
    source_urls: List[str]
    effect_size: Optional[str]
    confidence_interval: Optional[str]
    p_value: Optional[str]


class QualityIssue(TypedDict, total=False):
    """Quality review issue"""
    severity: str
    section: str
    description: str
    recommendation: str


class AgentHistoryEntry(TypedDict, total=False):
    """Agent execution history"""
    agent_name: str
    start_time: datetime
    end_time: datetime
    token_count: Optional[int]
    status: str


class PipelineState(TypedDict, total=False):
    """
    Complete pipeline state - TypedDict passed through all agents.
    Annotated fields use operator.add for append-based accumulation.
    """

    # === REQUEST METADATA ===
    run_id: str
    topic: str
    scope_instructions: str
    target_audience: str
    report_format: str
    requested_at: datetime

    # === COORDINATOR OUTPUT ===
    research_queries: List[str]
    scope_boundaries: Dict[str, List[str]]  # {"in_scope": [...], "out_of_scope": [...]}
    priority_subtopics: List[str]

    # === RESEARCH OUTPUT ===
    raw_sources: List[Source]
    deduplicated_sources: List[Source]
    research_summary: str

    # === ANALYSIS OUTPUT ===
    clinical_claims: List[ClinicalClaim]
    evidence_gaps: List[str]
    contradictions: List[str]
    statistical_findings: List[Dict]
    analysis_narrative: str

    # === WRITING OUTPUT ===
    report_sections: Dict[str, str]  # section_name -> markdown
    report_markdown: str
    citations: List[str]
    report_word_count: int

    # === QUALITY OUTPUT ===
    quality_issues: List[QualityIssue]
    quality_verdict: str  # "pass" | "revise" | "reject"
    quality_score: float  # 0-100
    revision_instructions: str
    quality_iteration: int

    # === CONTROL FLOW ===
    max_quality_iterations: int
    should_revise: bool

    # === PIPELINE METADATA ===
    current_agent: str
    agent_history: Annotated[List[AgentHistoryEntry], operator.add]
    pipeline_status: str
    error_message: Optional[str]
