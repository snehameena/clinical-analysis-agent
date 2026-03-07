"""
Shared pytest fixtures and mocks for testing.
Provides mock LLMs, Tavily client, and sample data.
"""

import pytest
import json
from datetime import datetime
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch
from src.state.schema import PipelineState, Source, ClinicalClaim, QualityIssue


@pytest.fixture
def sample_pipeline_state() -> PipelineState:
    """Create a sample pipeline state for testing"""
    return PipelineState(
        run_id="test-run-001",
        topic="Effectiveness of telemedicine for Type 2 diabetes management",
        scope_instructions="Focus on adults 18+, exclude pediatric cases",
        target_audience="clinical practitioners",
        report_format="clinical_brief",
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
        max_quality_iterations=3,
        should_revise=False,
        current_agent="coordinator",
        agent_history=[],
        pipeline_status="pending",
        error_message=None,
    )


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing"""
    client = MagicMock()
    return client


@pytest.fixture
def mock_tavily_client():
    """Mock Tavily client for search testing"""
    client = MagicMock()

    # Mock search response
    mock_response = {
        "results": [
            {
                "url": "https://pubmed.ncbi.nlm.nih.gov/example1",
                "title": "Telemedicine Effectiveness in Diabetes",
                "content": "A study showing effectiveness of remote monitoring",
                "score": 0.85,
            },
            {
                "url": "https://pubmed.ncbi.nlm.nih.gov/example2",
                "title": "Remote Patient Monitoring Outcomes",
                "content": "Long-term outcomes of RPM in T2D management",
                "score": 0.78,
            },
        ]
    }
    client.search = MagicMock(return_value=mock_response)
    return client


@pytest.fixture
def sample_sources() -> list[Source]:
    """Create sample research sources for testing"""
    return [
        Source(
            url="https://pubmed.ncbi.nlm.nih.gov/12345678",
            title="Systematic Review of Telemedicine in Diabetes Management",
            snippet="A comprehensive systematic review of 45 RCTs...",
            relevance_score=0.92,
            evidence_level="Systematic Review",
            publication_date="2023-01",
            authors=["Smith J", "Doe A"],
        ),
        Source(
            url="https://thelancet.com/example",
            title="RCT: Remote Monitoring vs Usual Care",
            snippet="This RCT compared remote monitoring to standard care...",
            relevance_score=0.88,
            evidence_level="Randomized Controlled Trial",
            publication_date="2022-06",
            authors=["Johnson M"],
        ),
        Source(
            url="https://pubmed.ncbi.nlm.nih.gov/87654321",
            title="Barriers to Telehealth Adoption in Diabetes",
            snippet="Qualitative study examining implementation barriers...",
            relevance_score=0.65,
            evidence_level="Cohort Study",
            publication_date="2023-03",
            authors=["Williams K", "Brown L"],
        ),
    ]


@pytest.fixture
def sample_clinical_claims() -> list[ClinicalClaim]:
    """Create sample clinical claims for testing"""
    return [
        ClinicalClaim(
            claim="Remote glucose monitoring improves HbA1c control",
            evidence_level="Systematic Review",
            source_urls=["https://pubmed.ncbi.nlm.nih.gov/12345678"],
            effect_size="Mean HbA1c reduction 0.5%",
            confidence_interval="(0.3%, 0.7%)",
            p_value="<0.001",
        ),
        ClinicalClaim(
            claim="Telehealth reduces clinic visit burden",
            evidence_level="RCT",
            source_urls=["https://thelancet.com/example"],
            effect_size="35% reduction in visits",
            confidence_interval="(20%, 50%)",
            p_value="0.002",
        ),
    ]


@pytest.fixture
def sample_quality_issues() -> list[QualityIssue]:
    """Create sample quality issues for testing"""
    return [
        QualityIssue(
            severity="minor",
            section="Evidence Synthesis",
            description="Missing publication year for one citation",
            recommendation="Add publication year: (2023) or note as 'In press'",
        ),
        QualityIssue(
            severity="major",
            section="Key Findings",
            description="Effect size for secondary outcome not quantified",
            recommendation="Add 95% CI or effect size magnitude from source Table 3",
        ),
    ]


@pytest.fixture
def mock_coordinator_response() -> dict:
    """Mock response from coordinator agent"""
    return {
        "research_queries": [
            "effectiveness telemedicine Type 2 diabetes management adults",
            "remote glucose monitoring HbA1c control outcomes",
            "telehealth adherence barriers diabetes",
        ],
        "scope_boundaries": {
            "in_scope": ["Adults 18+", "Type 2 diabetes", "Telemedicine/telehealth", "Clinical outcomes"],
            "out_of_scope": ["Pediatric diabetes", "Type 1 diabetes", "Administrative costs only"],
        },
        "priority_subtopics": ["Effectiveness", "Safety", "Barriers"],
    }


@pytest.fixture
def mock_research_response() -> dict:
    """Mock response from research agent"""
    return {
        "raw_sources": [
            {
                "url": "https://pubmed.ncbi.nlm.nih.gov/12345678",
                "title": "Systematic Review of Telemedicine",
                "snippet": "45 RCTs included",
                "evidence_level": "Systematic Review",
                "relevance_score": 0.92,
            }
        ],
        "deduplicated_sources": [
            {
                "url": "https://pubmed.ncbi.nlm.nih.gov/12345678",
                "title": "Systematic Review of Telemedicine",
                "snippet": "45 RCTs included",
                "evidence_level": "Systematic Review",
                "relevance_score": 0.92,
                "publication_date": "2023-01",
            }
        ],
        "research_summary": "Found 12 high-quality sources including 1 systematic review and 5 RCTs...",
    }


@pytest.fixture
def mock_analysis_response() -> dict:
    """Mock response from analysis agent"""
    return {
        "clinical_claims": [
            {
                "claim": "Remote monitoring improves HbA1c",
                "evidence_level": "Systematic Review",
                "source_urls": ["https://pubmed.ncbi.nlm.nih.gov/12345678"],
                "effect_size": "0.5% reduction",
                "p_value": "<0.001",
            }
        ],
        "evidence_gaps": [
            "Long-term adherence data beyond 12 months",
            "Cost-effectiveness in different healthcare systems",
        ],
        "contradictions": [],
        "statistical_findings": [
            {
                "metric": "HbA1c reduction",
                "estimate": "0.5%",
                "ci_lower": "0.3%",
                "ci_upper": "0.7%",
            }
        ],
        "analysis_narrative": "Evidence strongly supports telemedicine effectiveness...",
    }


@pytest.fixture
def mock_writing_response() -> dict:
    """Mock response from writing agent"""
    return {
        "report_sections": {
            "Executive Summary": "# Executive Summary\nThis report...",
            "Clinical Background": "# Clinical Background\nType 2 diabetes...",
            "Evidence Synthesis": "# Evidence Synthesis\nWe identified 12 high-quality sources...",
        },
        "report_markdown": "# Clinical Report\n## Executive Summary\n...",
        "citations": [
            "Smith J, et al. Telemedicine in Diabetes Management. *Lancet*. 2023;123:456-467.",
        ],
        "report_word_count": 2500,
    }


@pytest.fixture
def mock_quality_response() -> dict:
    """Mock response from quality agent"""
    return {
        "quality_issues": [
            {
                "severity": "minor",
                "section": "References",
                "description": "One reference missing journal name",
                "recommendation": "Add journal name for citation ID 5",
            }
        ],
        "quality_verdict": "pass",
        "quality_score": 88.0,
        "revision_instructions": "",
    }


@pytest.fixture
def mock_anthropic_response():
    """Mock LLM response from Anthropic API"""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=json.dumps({
        "research_queries": ["query1", "query2", "query3"],
        "scope_boundaries": {"in_scope": [], "out_of_scope": []},
        "priority_subtopics": ["topic1", "topic2"],
    }))]
    return mock_response


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing"""
    env_vars = {
        "ANTHROPIC_API_KEY": "sk-ant-test-key",
        "TAVILY_API_KEY": "tvly-test-key",
        "COORDINATOR_MODEL": "claude-3-5-sonnet-20241022",
        "RESEARCH_MODEL": "claude-3-5-sonnet-20241022",
        "ANALYSIS_MODEL": "claude-3-5-sonnet-20241022",
        "WRITING_MODEL": "claude-3-5-sonnet-20241022",
        "QUALITY_MODEL": "claude-3-5-sonnet-20241022",
        "MAX_QUALITY_ITERATIONS": "3",
        "API_HOST": "0.0.0.0",
        "API_PORT": "8000",
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    return env_vars
