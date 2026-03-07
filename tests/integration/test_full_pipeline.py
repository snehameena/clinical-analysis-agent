"""
Integration tests for the full LangGraph pipeline.
All LLM and Tavily calls are mocked — no API keys needed.
"""

import pytest
import json
from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock
from src.graph.builder import build_pipeline_graph
from src.state.schema import PipelineState


def _make_llm_router(coordinator_resp, analysis_resp, writing_resp, quality_resp):
    """Create an AsyncMock side_effect that routes based on message content."""

    async def route(user_message):
        msg_lower = user_message.lower()
        if "research queries" in msg_lower and "generate" in msg_lower:
            return json.dumps(coordinator_resp)
        elif "analyze these sources" in msg_lower or ("clinical claims" in msg_lower and "evidence gaps" not in msg_lower):
            return json.dumps(analysis_resp)
        elif "evaluate this report" in msg_lower or "quality" in msg_lower:
            return json.dumps(quality_resp)
        elif "structured clinical report" in msg_lower or "revision instructions" in msg_lower or "revise the report" in msg_lower:
            return json.dumps(writing_resp)
        # Fallback
        return json.dumps(coordinator_resp)

    return route


@pytest.fixture
def initial_state():
    """Create initial pipeline state for integration test."""
    return PipelineState(
        run_id="integration-test-001",
        topic="Effectiveness of telemedicine for Type 2 diabetes management",
        scope_instructions="Focus on adults 18+",
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
        current_agent="",
        agent_history=[],
        pipeline_status="pending",
        error_message=None,
    )


class TestFullPipeline:
    """Integration tests for the complete pipeline graph."""

    @pytest.mark.asyncio
    async def test_happy_path_pass(self, initial_state, mock_env_vars):
        """Test full pipeline ending with quality pass verdict."""
        coordinator_resp = {
            "research_queries": ["telemedicine diabetes outcomes", "remote monitoring HbA1c"],
            "scope_boundaries": {"in_scope": ["Adults 18+", "T2D"], "out_of_scope": ["Pediatric"]},
            "priority_subtopics": ["Effectiveness", "Safety"],
        }
        analysis_resp = {
            "clinical_claims": [{"claim": "Telemedicine improves HbA1c", "evidence_level": "RCT", "source_urls": ["https://example.com"]}],
            "evidence_gaps": ["Long-term data needed"],
            "contradictions": [],
            "statistical_findings": [{"metric": "HbA1c", "estimate": "0.5%"}],
            "analysis_narrative": "Evidence supports telemedicine effectiveness.",
        }
        writing_resp = {
            "report_sections": {"Executive Summary": "# Summary\nTelemedicine works."},
            "report_markdown": "# Clinical Report\nTelemedicine is effective for T2D management based on RCT evidence.",
            "citations": ["Smith et al. 2023"],
        }
        quality_resp = {
            "quality_score": 85.0,
            "quality_verdict": "pass",
            "quality_issues": [],
            "revision_instructions": "",
        }

        router = _make_llm_router(coordinator_resp, analysis_resp, writing_resp, quality_resp)

        mock_tavily_search = AsyncMock(return_value={
            "results": [
                {"url": "https://pubmed.ncbi.nlm.nih.gov/123", "title": "Study A", "content": "RCT content", "score": 0.9}
            ]
        })
        mock_tavily_parse = MagicMock(return_value=[
            {"url": "https://pubmed.ncbi.nlm.nih.gov/123", "title": "Study A", "snippet": "RCT content",
             "relevance_score": 0.9, "evidence_level": "Randomized Controlled Trial"}
        ])

        with patch.object(
                 __import__("src.agents.base", fromlist=["BaseAgent"]).BaseAgent,
                 "_call_llm",
                 side_effect=router,
             ), \
             patch("src.agents.research.HealthcareTavilySearch") as mock_tavily_cls:

            mock_tavily_instance = MagicMock()
            mock_tavily_instance.search = mock_tavily_search
            mock_tavily_instance.parse_results = mock_tavily_parse
            mock_tavily_cls.return_value = mock_tavily_instance

            graph = build_pipeline_graph()
            result = await graph.ainvoke(initial_state)

        assert result["quality_verdict"] == "pass"
        assert result["quality_score"] == 85.0
        assert result["report_markdown"] != ""
        assert len(result["research_queries"]) == 2

    @pytest.mark.asyncio
    async def test_revision_loop(self, initial_state, mock_env_vars):
        """Test pipeline with one revision loop before passing."""
        coordinator_resp = {
            "research_queries": ["query1"],
            "scope_boundaries": {"in_scope": ["T2D"], "out_of_scope": []},
            "priority_subtopics": ["Effectiveness"],
        }
        analysis_resp = {
            "clinical_claims": [{"claim": "Claim 1", "evidence_level": "RCT", "source_urls": []}],
            "evidence_gaps": ["Gap"],
            "contradictions": [],
            "statistical_findings": [],
            "analysis_narrative": "Analysis.",
        }
        writing_resp = {
            "report_sections": {"Summary": "# Report"},
            "report_markdown": "# Report with evidence content here",
            "citations": ["Ref 1"],
        }

        # Track quality calls to alternate between revise and pass
        quality_call_count = {"n": 0}

        async def quality_aware_router(user_message):
            msg_lower = user_message.lower()
            if "research queries" in msg_lower and "generate" in msg_lower:
                return json.dumps(coordinator_resp)
            elif "analyze these sources" in msg_lower:
                return json.dumps(analysis_resp)
            elif "evaluate this report" in msg_lower or "quality" in msg_lower:
                quality_call_count["n"] += 1
                if quality_call_count["n"] == 1:
                    return json.dumps({
                        "quality_score": 60.0,
                        "quality_verdict": "revise",
                        "quality_issues": [{"severity": "major", "section": "Evidence", "description": "Needs work", "recommendation": "Fix"}],
                        "revision_instructions": "Improve evidence section",
                    })
                else:
                    return json.dumps({
                        "quality_score": 82.0,
                        "quality_verdict": "pass",
                        "quality_issues": [],
                        "revision_instructions": "",
                    })
            else:
                # Writing (initial or revision)
                return json.dumps(writing_resp)

        mock_tavily_search = AsyncMock(return_value={"results": [{"url": "https://example.com/1", "title": "S", "content": "C", "score": 0.8}]})
        mock_tavily_parse = MagicMock(return_value=[
            {"url": "https://example.com/1", "title": "S", "snippet": "C", "relevance_score": 0.8, "evidence_level": "RCT"}
        ])

        with patch.object(
                 __import__("src.agents.base", fromlist=["BaseAgent"]).BaseAgent,
                 "_call_llm",
                 side_effect=quality_aware_router,
             ), \
             patch("src.agents.research.HealthcareTavilySearch") as mock_tavily_cls:

            mock_tavily_instance = MagicMock()
            mock_tavily_instance.search = mock_tavily_search
            mock_tavily_instance.parse_results = mock_tavily_parse
            mock_tavily_cls.return_value = mock_tavily_instance

            graph = build_pipeline_graph()
            result = await graph.ainvoke(initial_state)

        assert result["quality_verdict"] == "pass"
        # Should have gone through revision loop
        assert quality_call_count["n"] == 2
        assert result.get("quality_iteration", 0) == 1
