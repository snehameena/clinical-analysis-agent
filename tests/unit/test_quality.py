"""
Unit tests for QualityAgent.
"""

import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock
from src.agents.quality import QualityAgent


@pytest.fixture
def quality_agent(mock_env_vars):
    """Create QualityAgent with mocked Anthropic client."""
    return QualityAgent()


class TestQualityAgent:
    """Test quality agent functionality."""

    @pytest.mark.asyncio
    async def test_execute_pass_verdict(self, quality_agent, sample_pipeline_state, mock_quality_response):
        """Test quality agent with pass verdict."""
        sample_pipeline_state["report_markdown"] = "# Full Clinical Report\n" * 100
        sample_pipeline_state["clinical_claims"] = [{"claim": "Test claim"}]
        sample_pipeline_state["evidence_gaps"] = ["Gap 1"]

        quality_agent._call_llm = AsyncMock(return_value=json.dumps(mock_quality_response))

        result = await quality_agent.execute(sample_pipeline_state)

        assert result["quality_verdict"] == "pass"
        assert result["quality_score"] == 88.0
        assert result["should_revise"] is False

    @pytest.mark.asyncio
    async def test_execute_revise_verdict(self, quality_agent, sample_pipeline_state):
        """Test quality agent with revise verdict."""
        sample_pipeline_state["report_markdown"] = "# Report content here"
        sample_pipeline_state["clinical_claims"] = []
        sample_pipeline_state["evidence_gaps"] = []

        revise_response = {
            "quality_score": 65.0,
            "quality_verdict": "revise",
            "quality_issues": [
                {"severity": "major", "section": "Evidence", "description": "Missing citations", "recommendation": "Add citations"}
            ],
            "revision_instructions": "Add proper citations to the Evidence section",
        }
        quality_agent._call_llm = AsyncMock(return_value=json.dumps(revise_response))

        result = await quality_agent.execute(sample_pipeline_state)

        assert result["quality_verdict"] == "revise"
        assert result["should_revise"] is True
        assert result["revision_instructions"] != ""

    @pytest.mark.asyncio
    async def test_execute_reject_verdict(self, quality_agent, sample_pipeline_state):
        """Test quality agent with reject verdict."""
        sample_pipeline_state["report_markdown"] = "# Report"
        sample_pipeline_state["clinical_claims"] = []
        sample_pipeline_state["evidence_gaps"] = []

        reject_response = {
            "quality_score": 30.0,
            "quality_verdict": "reject",
            "quality_issues": [
                {"severity": "critical", "section": "All", "description": "No evidence", "recommendation": "Rewrite"}
            ],
            "revision_instructions": "",
        }
        quality_agent._call_llm = AsyncMock(return_value=json.dumps(reject_response))

        result = await quality_agent.execute(sample_pipeline_state)

        assert result["quality_verdict"] == "reject"
        assert result["should_revise"] is False

    @pytest.mark.asyncio
    async def test_invalid_verdict_defaults_to_reject(self, quality_agent, sample_pipeline_state):
        """Test that invalid verdict string defaults to reject."""
        sample_pipeline_state["report_markdown"] = "# Report"
        sample_pipeline_state["clinical_claims"] = []
        sample_pipeline_state["evidence_gaps"] = []

        bad_response = {
            "quality_score": 50.0,
            "quality_verdict": "unknown_verdict",
            "quality_issues": [],
            "revision_instructions": "",
        }
        quality_agent._call_llm = AsyncMock(return_value=json.dumps(bad_response))

        result = await quality_agent.execute(sample_pipeline_state)

        assert result["quality_verdict"] == "reject"

    @pytest.mark.asyncio
    async def test_execute_raises_on_empty_report(self, quality_agent, sample_pipeline_state):
        """Test that quality agent raises error when no report provided."""
        sample_pipeline_state["report_markdown"] = ""

        with pytest.raises(ValueError, match="No report provided"):
            await quality_agent.execute(sample_pipeline_state)

    @pytest.mark.asyncio
    async def test_execute_records_history(self, quality_agent, sample_pipeline_state, mock_quality_response):
        """Test that quality agent records execution."""
        sample_pipeline_state["report_markdown"] = "# Report content"
        sample_pipeline_state["clinical_claims"] = []
        sample_pipeline_state["evidence_gaps"] = []

        quality_agent._call_llm = AsyncMock(return_value=json.dumps(mock_quality_response))

        result = await quality_agent.execute(sample_pipeline_state)

        assert result["current_agent"] == "quality"

    @pytest.mark.asyncio
    async def test_quality_issues_extracted(self, quality_agent, sample_pipeline_state, mock_quality_response):
        """Test that quality issues are properly extracted."""
        sample_pipeline_state["report_markdown"] = "# Report content"
        sample_pipeline_state["clinical_claims"] = []
        sample_pipeline_state["evidence_gaps"] = []

        quality_agent._call_llm = AsyncMock(return_value=json.dumps(mock_quality_response))

        result = await quality_agent.execute(sample_pipeline_state)

        assert len(result["quality_issues"]) == 1
        assert result["quality_issues"][0]["severity"] == "minor"
