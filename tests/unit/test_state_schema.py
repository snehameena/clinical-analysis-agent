"""
Unit tests for PipelineState schema and related types.
"""

import pytest
from src.state.schema import (
    PipelineState,
    Source,
    ClinicalClaim,
    QualityIssue,
    AgentHistoryEntry,
    QualityVerdict,
    PipelineStatus,
    EvidenceLevel,
    IssueSeverity,
)


class TestEnums:
    """Test enum definitions and values."""

    def test_quality_verdict_values(self):
        assert QualityVerdict.PASS.value == "pass"
        assert QualityVerdict.REVISE.value == "revise"
        assert QualityVerdict.REJECT.value == "reject"

    def test_pipeline_status_values(self):
        assert PipelineStatus.PENDING.value == "pending"
        assert PipelineStatus.RUNNING.value == "running"
        assert PipelineStatus.COMPLETED.value == "completed"
        assert PipelineStatus.FAILED.value == "failed"

    def test_evidence_level_values(self):
        assert EvidenceLevel.SYSTEMATIC_REVIEW.value == "Systematic Review"
        assert EvidenceLevel.RCT.value == "Randomized Controlled Trial"
        assert EvidenceLevel.COHORT_STUDY.value == "Cohort Study"
        assert EvidenceLevel.CASE_REPORT.value == "Case Report"
        assert EvidenceLevel.OPINION.value == "Expert Opinion"

    def test_issue_severity_values(self):
        assert IssueSeverity.CRITICAL.value == "critical"
        assert IssueSeverity.MAJOR.value == "major"
        assert IssueSeverity.MINOR.value == "minor"

    def test_quality_verdict_is_string_enum(self):
        assert isinstance(QualityVerdict.PASS, str)
        assert QualityVerdict.PASS == "pass"


class TestTypedDicts:
    """Test TypedDict instantiation."""

    def test_source_creation(self):
        source = Source(
            url="https://example.com",
            title="Test Source",
            snippet="A test snippet",
            relevance_score=0.85,
            evidence_level="Systematic Review",
        )
        assert source["url"] == "https://example.com"
        assert source["relevance_score"] == 0.85

    def test_source_optional_fields(self):
        source = Source(url="https://example.com", title="Test")
        assert source["url"] == "https://example.com"
        assert "publication_date" not in source

    def test_clinical_claim_creation(self):
        claim = ClinicalClaim(
            claim="Treatment X improves outcomes",
            evidence_level="RCT",
            source_urls=["https://example.com"],
            effect_size="0.5",
            p_value="<0.05",
        )
        assert claim["claim"] == "Treatment X improves outcomes"
        assert claim["p_value"] == "<0.05"

    def test_quality_issue_creation(self):
        issue = QualityIssue(
            severity="major",
            section="Methods",
            description="Missing sample size",
            recommendation="Add sample size info",
        )
        assert issue["severity"] == "major"


class TestPipelineState:
    """Test PipelineState TypedDict."""

    def test_pipeline_state_creation(self, sample_pipeline_state):
        state = sample_pipeline_state
        assert state["run_id"] == "test-run-001"
        assert state["topic"] == "Effectiveness of telemedicine for Type 2 diabetes management"
        assert state["quality_iteration"] == 0
        assert state["max_quality_iterations"] == 3

    def test_pipeline_state_has_all_sections(self, sample_pipeline_state):
        state = sample_pipeline_state
        # Request metadata
        assert "run_id" in state
        assert "topic" in state
        # Coordinator output
        assert "research_queries" in state
        assert "scope_boundaries" in state
        # Research output
        assert "raw_sources" in state
        assert "deduplicated_sources" in state
        # Analysis output
        assert "clinical_claims" in state
        assert "evidence_gaps" in state
        # Writing output
        assert "report_sections" in state
        assert "report_markdown" in state
        # Quality output
        assert "quality_issues" in state
        assert "quality_verdict" in state
        # Control flow
        assert "should_revise" in state
        # Metadata
        assert "agent_history" in state
        assert "pipeline_status" in state
