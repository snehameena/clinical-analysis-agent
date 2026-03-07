"""
Unit tests for ResearchAgent.
"""

import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock
from src.agents.research import ResearchAgent
from src.tools.tavily_search import deduplicate_sources


@pytest.fixture
def research_agent(mock_env_vars):
    """Create ResearchAgent with mocked clients."""
    with patch("src.agents.research.HealthcareTavilySearch") as mock_tavily_cls:
        mock_tavily = MagicMock()
        mock_tavily_cls.return_value = mock_tavily
        agent = ResearchAgent()
        agent.tavily = mock_tavily
    return agent


class TestResearchAgent:
    """Test research agent functionality."""

    @pytest.mark.asyncio
    async def test_execute_searches_all_queries(self, research_agent, sample_pipeline_state):
        """Test that research agent searches for all provided queries."""
        sample_pipeline_state["research_queries"] = ["query1", "query2", "query3"]

        mock_search_result = {
            "results": [
                {
                    "url": "https://pubmed.ncbi.nlm.nih.gov/123",
                    "title": "Test Study",
                    "content": "Study content",
                    "score": 0.85,
                }
            ]
        }

        research_agent.tavily.search = AsyncMock(return_value=mock_search_result)
        research_agent.tavily.parse_results = MagicMock(return_value=[
            {
                "url": "https://pubmed.ncbi.nlm.nih.gov/123",
                "title": "Test Study",
                "snippet": "Study content",
                "relevance_score": 0.85,
                "evidence_level": "Systematic Review",
            }
        ])

        result = await research_agent.execute(sample_pipeline_state)

        assert research_agent.tavily.search.call_count == 3
        assert len(result["raw_sources"]) > 0

    @pytest.mark.asyncio
    async def test_execute_deduplicates_sources(self, research_agent, sample_pipeline_state):
        """Test that research agent deduplicates sources across queries."""
        sample_pipeline_state["research_queries"] = ["query1", "query2"]

        # Same URL returned by both queries
        source = {
            "url": "https://pubmed.ncbi.nlm.nih.gov/123",
            "title": "Test Study",
            "snippet": "Study content",
            "relevance_score": 0.85,
            "evidence_level": "Systematic Review",
        }

        research_agent.tavily.search = AsyncMock(return_value={"results": [{"url": "x", "score": 0.85}]})
        research_agent.tavily.parse_results = MagicMock(return_value=[source])

        result = await research_agent.execute(sample_pipeline_state)

        # 2 raw sources (one per query), but deduplicated to 1
        assert len(result["raw_sources"]) == 2
        assert len(result["deduplicated_sources"]) == 1

    @pytest.mark.asyncio
    async def test_execute_generates_summary(self, research_agent, sample_pipeline_state):
        """Test that research agent generates a text summary."""
        sample_pipeline_state["research_queries"] = ["query1"]

        research_agent.tavily.search = AsyncMock(return_value={"results": []})
        research_agent.tavily.parse_results = MagicMock(return_value=[
            {
                "url": "https://example.com/1",
                "title": "Study A",
                "snippet": "Content",
                "relevance_score": 0.9,
                "evidence_level": "Systematic Review",
            }
        ])

        result = await research_agent.execute(sample_pipeline_state)

        assert "research_summary" in result
        assert "1 relevant source" in result["research_summary"] or "Found" in result["research_summary"]

    @pytest.mark.asyncio
    async def test_execute_raises_on_no_queries(self, research_agent, sample_pipeline_state):
        """Test that research agent raises error when no queries provided."""
        sample_pipeline_state["research_queries"] = []

        with pytest.raises(ValueError, match="No research queries"):
            await research_agent.execute(sample_pipeline_state)

    @pytest.mark.asyncio
    async def test_execute_records_history(self, research_agent, sample_pipeline_state):
        """Test that research agent records execution history."""
        sample_pipeline_state["research_queries"] = ["query1"]

        research_agent.tavily.search = AsyncMock(return_value={"results": []})
        research_agent.tavily.parse_results = MagicMock(return_value=[])

        result = await research_agent.execute(sample_pipeline_state)

        assert result["current_agent"] == "research"


class TestDeduplicateSources:
    """Test the deduplicate_sources utility."""

    def test_deduplicates_by_url(self, sample_sources):
        # Add a duplicate with lower relevance
        dup = dict(sample_sources[0])
        dup["relevance_score"] = 0.5
        sources = sample_sources + [dup]

        result = deduplicate_sources(sources)

        urls = [s["url"] for s in result]
        assert len(urls) == len(set(urls))

    def test_keeps_highest_relevance(self):
        sources = [
            {"url": "https://example.com/1", "relevance_score": 0.7},
            {"url": "https://example.com/1", "relevance_score": 0.9},
        ]
        result = deduplicate_sources(sources)
        assert len(result) == 1
        assert result[0]["relevance_score"] == 0.9

    def test_filters_by_min_relevance(self):
        sources = [
            {"url": "https://example.com/1", "relevance_score": 0.3},
            {"url": "https://example.com/2", "relevance_score": 0.8},
        ]
        result = deduplicate_sources(sources, min_relevance=0.6)
        assert len(result) == 1
        assert result[0]["url"] == "https://example.com/2"

    def test_sorts_by_relevance_descending(self):
        sources = [
            {"url": "https://example.com/1", "relevance_score": 0.7},
            {"url": "https://example.com/2", "relevance_score": 0.9},
            {"url": "https://example.com/3", "relevance_score": 0.8},
        ]
        result = deduplicate_sources(sources)
        scores = [s["relevance_score"] for s in result]
        assert scores == sorted(scores, reverse=True)

    def test_empty_input(self):
        assert deduplicate_sources([]) == []
