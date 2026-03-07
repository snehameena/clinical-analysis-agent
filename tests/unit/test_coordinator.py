"""
Unit tests for CoordinatorAgent.
"""

import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock
from src.agents.coordinator import CoordinatorAgent


@pytest.fixture
def coordinator(mock_env_vars):
    """Create CoordinatorAgent with mocked Anthropic client."""
    return CoordinatorAgent()


class TestCoordinatorAgent:
    """Test coordinator agent functionality."""

    @pytest.mark.asyncio
    async def test_execute_generates_queries(self, coordinator, sample_pipeline_state, mock_coordinator_response):
        """Test that coordinator generates research queries from topic."""
        coordinator._call_llm = AsyncMock(return_value=json.dumps(mock_coordinator_response))

        result = await coordinator.execute(sample_pipeline_state)

        assert len(result["research_queries"]) == 3
        assert "effectiveness telemedicine Type 2 diabetes management adults" in result["research_queries"]

    @pytest.mark.asyncio
    async def test_execute_sets_scope_boundaries(self, coordinator, sample_pipeline_state, mock_coordinator_response):
        """Test that coordinator sets scope boundaries."""
        coordinator._call_llm = AsyncMock(return_value=json.dumps(mock_coordinator_response))

        result = await coordinator.execute(sample_pipeline_state)

        assert "in_scope" in result["scope_boundaries"]
        assert "out_of_scope" in result["scope_boundaries"]
        assert "Adults 18+" in result["scope_boundaries"]["in_scope"]

    @pytest.mark.asyncio
    async def test_execute_sets_priority_subtopics(self, coordinator, sample_pipeline_state, mock_coordinator_response):
        """Test that coordinator sets priority subtopics."""
        coordinator._call_llm = AsyncMock(return_value=json.dumps(mock_coordinator_response))

        result = await coordinator.execute(sample_pipeline_state)

        assert len(result["priority_subtopics"]) == 3
        assert "Effectiveness" in result["priority_subtopics"]

    @pytest.mark.asyncio
    async def test_execute_records_history(self, coordinator, sample_pipeline_state, mock_coordinator_response):
        """Test that coordinator records execution in agent history."""
        coordinator._call_llm = AsyncMock(return_value=json.dumps(mock_coordinator_response))

        result = await coordinator.execute(sample_pipeline_state)

        assert result["current_agent"] == "coordinator"
        assert len(result["agent_history"]) == 1
        assert result["agent_history"][0]["agent_name"] == "coordinator"

    @pytest.mark.asyncio
    async def test_execute_raises_on_empty_topic(self, coordinator, sample_pipeline_state):
        """Test that coordinator raises error when topic is missing."""
        sample_pipeline_state["topic"] = ""

        with pytest.raises(ValueError, match="Topic not provided"):
            await coordinator.execute(sample_pipeline_state)

    @pytest.mark.asyncio
    async def test_execute_raises_on_invalid_json(self, coordinator, sample_pipeline_state):
        """Test that coordinator raises error on invalid JSON response."""
        coordinator._call_llm = AsyncMock(return_value="not json at all")

        with pytest.raises(ValueError, match="Coordinator failed to generate valid JSON"):
            await coordinator.execute(sample_pipeline_state)

    @pytest.mark.asyncio
    async def test_execute_handles_markdown_wrapped_json(self, coordinator, sample_pipeline_state, mock_coordinator_response):
        """Test that coordinator handles JSON wrapped in markdown code blocks."""
        wrapped = f"```json\n{json.dumps(mock_coordinator_response)}\n```"
        coordinator._call_llm = AsyncMock(return_value=wrapped)

        result = await coordinator.execute(sample_pipeline_state)

        assert len(result["research_queries"]) == 3
