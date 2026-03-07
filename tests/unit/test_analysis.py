"""
Unit tests for AnalysisAgent.
"""

import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock
from src.agents.analysis import AnalysisAgent


@pytest.fixture
def analysis_agent(mock_env_vars):
    """Create AnalysisAgent with mocked Anthropic client."""
    return AnalysisAgent()


class TestAnalysisAgent:
    """Test analysis agent functionality."""

    @pytest.mark.asyncio
    async def test_execute_extracts_claims(self, analysis_agent, sample_pipeline_state, mock_analysis_response):
        """Test that analysis agent extracts clinical claims."""
        sample_pipeline_state["deduplicated_sources"] = [
            {"url": "https://example.com/1", "title": "Study", "snippet": "Content", "relevance_score": 0.9, "evidence_level": "RCT"}
        ]
        analysis_agent._call_llm = AsyncMock(return_value=json.dumps(mock_analysis_response))

        result = await analysis_agent.execute(sample_pipeline_state)

        assert len(result["clinical_claims"]) >= 1
        assert result["clinical_claims"][0]["claim"] == "Remote monitoring improves HbA1c"

    @pytest.mark.asyncio
    async def test_execute_extracts_gaps(self, analysis_agent, sample_pipeline_state, mock_analysis_response):
        """Test that analysis agent identifies evidence gaps."""
        sample_pipeline_state["deduplicated_sources"] = [
            {"url": "https://example.com/1", "title": "Study", "snippet": "Content", "relevance_score": 0.9, "evidence_level": "RCT"}
        ]
        analysis_agent._call_llm = AsyncMock(return_value=json.dumps(mock_analysis_response))

        result = await analysis_agent.execute(sample_pipeline_state)

        assert len(result["evidence_gaps"]) == 2
        assert "Long-term adherence" in result["evidence_gaps"][0]

    @pytest.mark.asyncio
    async def test_execute_extracts_statistical_findings(self, analysis_agent, sample_pipeline_state, mock_analysis_response):
        """Test that analysis agent extracts statistical findings."""
        sample_pipeline_state["deduplicated_sources"] = [
            {"url": "https://example.com/1", "title": "Study", "snippet": "Content", "relevance_score": 0.9, "evidence_level": "RCT"}
        ]
        analysis_agent._call_llm = AsyncMock(return_value=json.dumps(mock_analysis_response))

        result = await analysis_agent.execute(sample_pipeline_state)

        assert len(result["statistical_findings"]) >= 1
        assert result["analysis_narrative"] != ""

    @pytest.mark.asyncio
    async def test_execute_raises_on_no_sources(self, analysis_agent, sample_pipeline_state):
        """Test that analysis agent raises error when no sources available."""
        sample_pipeline_state["deduplicated_sources"] = []

        with pytest.raises(ValueError, match="No deduplicated sources"):
            await analysis_agent.execute(sample_pipeline_state)

    @pytest.mark.asyncio
    async def test_execute_raises_on_invalid_json(self, analysis_agent, sample_pipeline_state):
        """Test that analysis agent raises error on unparseable LLM response."""
        sample_pipeline_state["deduplicated_sources"] = [
            {"url": "https://example.com/1", "title": "Study", "snippet": "Content", "relevance_score": 0.9, "evidence_level": "RCT"}
        ]
        analysis_agent._call_llm = AsyncMock(return_value="not valid json")

        with pytest.raises(ValueError, match="Analysis agent failed"):
            await analysis_agent.execute(sample_pipeline_state)

    @pytest.mark.asyncio
    async def test_execute_records_history(self, analysis_agent, sample_pipeline_state, mock_analysis_response):
        """Test that analysis agent records execution."""
        sample_pipeline_state["deduplicated_sources"] = [
            {"url": "https://example.com/1", "title": "Study", "snippet": "Content", "relevance_score": 0.9, "evidence_level": "RCT"}
        ]
        analysis_agent._call_llm = AsyncMock(return_value=json.dumps(mock_analysis_response))

        result = await analysis_agent.execute(sample_pipeline_state)

        assert result["current_agent"] == "analysis"

    def test_format_sources_for_analysis(self, analysis_agent):
        """Test source formatting for LLM analysis prompt."""
        sources = [
            {"title": "Study A", "snippet": "Content A", "evidence_level": "RCT", "relevance_score": 0.9},
            {"title": "Study B", "snippet": "Content B", "evidence_level": "Cohort Study", "relevance_score": 0.7},
        ]
        text = analysis_agent._format_sources_for_analysis(sources)

        assert "Study A" in text
        assert "[RCT]" in text
        assert "0.90" in text
