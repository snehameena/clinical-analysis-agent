"""
Unit tests for WritingAgent.
"""

import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock
from src.agents.writing import WritingAgent


@pytest.fixture
def writing_agent(mock_env_vars):
    """Create WritingAgent with mocked Anthropic client."""
    return WritingAgent()


class TestWritingAgent:
    """Test writing agent functionality."""

    @pytest.mark.asyncio
    async def test_initial_report_generation(self, writing_agent, sample_pipeline_state, mock_writing_response):
        """Test initial report generation mode."""
        sample_pipeline_state["analysis_narrative"] = "Test analysis narrative"
        sample_pipeline_state["clinical_claims"] = [{"claim": "Test claim"}]
        sample_pipeline_state["evidence_gaps"] = ["Gap 1"]
        sample_pipeline_state["deduplicated_sources"] = [
            {"url": "https://example.com", "title": "Study", "evidence_level": "RCT"}
        ]
        sample_pipeline_state["quality_iteration"] = 0

        writing_agent._call_llm = AsyncMock(return_value=json.dumps(mock_writing_response))

        result = await writing_agent.execute(sample_pipeline_state)

        assert "report_sections" in result
        assert "Executive Summary" in result["report_sections"]
        assert result["report_markdown"] != ""
        assert len(result["citations"]) >= 1

    @pytest.mark.asyncio
    async def test_report_word_count(self, writing_agent, sample_pipeline_state, mock_writing_response):
        """Test that word count is computed from report markdown."""
        sample_pipeline_state["quality_iteration"] = 0
        sample_pipeline_state["analysis_narrative"] = "Analysis"
        sample_pipeline_state["clinical_claims"] = []
        sample_pipeline_state["evidence_gaps"] = []
        sample_pipeline_state["deduplicated_sources"] = []

        writing_agent._call_llm = AsyncMock(return_value=json.dumps(mock_writing_response))

        result = await writing_agent.execute(sample_pipeline_state)

        assert result["report_word_count"] > 0

    @pytest.mark.asyncio
    async def test_revision_mode(self, writing_agent, sample_pipeline_state, mock_writing_response):
        """Test revision mode when quality_iteration > 0 and revision_instructions set."""
        sample_pipeline_state["quality_iteration"] = 1
        sample_pipeline_state["revision_instructions"] = "Fix the evidence section"
        sample_pipeline_state["report_markdown"] = "# Existing Report\nSome content."
        sample_pipeline_state["report_sections"] = {"Summary": "Content"}

        writing_agent._call_llm = AsyncMock(return_value=json.dumps(mock_writing_response))

        result = await writing_agent.execute(sample_pipeline_state)

        # Revision should increment iteration
        assert result["quality_iteration"] == 2

    @pytest.mark.asyncio
    async def test_initial_mode_when_no_revision_instructions(self, writing_agent, sample_pipeline_state, mock_writing_response):
        """Test that agent uses initial mode when quality_iteration > 0 but no revision instructions."""
        sample_pipeline_state["quality_iteration"] = 1
        sample_pipeline_state["revision_instructions"] = ""
        sample_pipeline_state["analysis_narrative"] = "Analysis"
        sample_pipeline_state["clinical_claims"] = []
        sample_pipeline_state["evidence_gaps"] = []
        sample_pipeline_state["deduplicated_sources"] = []

        writing_agent._call_llm = AsyncMock(return_value=json.dumps(mock_writing_response))

        result = await writing_agent.execute(sample_pipeline_state)

        # Should NOT increment iteration (initial mode)
        assert result.get("quality_iteration") == 1

    @pytest.mark.asyncio
    async def test_execute_records_history(self, writing_agent, sample_pipeline_state, mock_writing_response):
        """Test that writing agent records execution."""
        sample_pipeline_state["quality_iteration"] = 0
        sample_pipeline_state["analysis_narrative"] = "Analysis"
        sample_pipeline_state["clinical_claims"] = []
        sample_pipeline_state["evidence_gaps"] = []
        sample_pipeline_state["deduplicated_sources"] = []

        writing_agent._call_llm = AsyncMock(return_value=json.dumps(mock_writing_response))

        result = await writing_agent.execute(sample_pipeline_state)

        assert result["current_agent"] == "writing"

    @pytest.mark.asyncio
    async def test_revision_raises_on_invalid_json(self, writing_agent, sample_pipeline_state):
        """Test that revision mode raises error on invalid JSON."""
        sample_pipeline_state["quality_iteration"] = 1
        sample_pipeline_state["revision_instructions"] = "Fix things"
        sample_pipeline_state["report_markdown"] = "# Report"

        writing_agent._call_llm = AsyncMock(return_value="bad json")

        with pytest.raises(ValueError, match="Writing agent revision failed"):
            await writing_agent.execute(sample_pipeline_state)
