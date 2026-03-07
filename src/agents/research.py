"""
Research agent - Evidence gathering from medical literature.
Searches for and aggregates clinical evidence using Tavily.
"""

import yaml
from typing import Optional
from pathlib import Path
from src.agents.base import BaseAgent
from src.state.schema import PipelineState, Source
from src.tools.tavily_search import HealthcareTavilySearch, deduplicate_sources


class ResearchAgent(BaseAgent):
    """Agent responsible for searching and aggregating research evidence"""

    def __init__(self, model: Optional[str] = None):
        """Initialize research agent"""
        super().__init__(name="research", model=model)
        self.tavily = HealthcareTavilySearch()

    def _get_system_prompt(self) -> str:
        """Load system prompt from config"""
        try:
            repo_root = Path(__file__).resolve().parents[2]
            path = repo_root / "config" / "prompts.yaml"
            with open(path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                return config.get("research", {}).get("system", "")
        except FileNotFoundError:
            return "You are a Clinical Research Specialist. Search for and summarize medical evidence."

    async def execute(self, state: PipelineState) -> PipelineState:
        """
        Execute research queries and aggregate findings.

        Args:
            state: Pipeline state with research_queries

        Returns:
            Updated state with raw_sources, deduplicated_sources, research_summary
        """
        research_queries = state.get("research_queries", [])

        if not research_queries:
            raise ValueError("No research queries provided")

        # Collect all sources from all queries
        all_sources: list[Source] = []

        for query in research_queries:
            # Search using Tavily
            raw_results = await self.tavily.search(query)

            # Parse results into Source objects
            sources = self.tavily.parse_results(raw_results)
            all_sources.extend(sources)

        # Deduplicate sources
        deduplicated = deduplicate_sources(all_sources)

        # Generate research summary
        summary = self._generate_summary(deduplicated)

        # Update state
        state["raw_sources"] = all_sources
        state["deduplicated_sources"] = deduplicated
        state["research_summary"] = summary

        # Record execution
        state = self._record_execution(state)

        return state

    def _generate_summary(self, sources: list[Source]) -> str:
        """
        Generate text summary of research findings.

        Args:
            sources: List of deduplicated sources

        Returns:
            Summary string
        """
        if not sources:
            return "No relevant sources found."

        # Count by evidence level
        evidence_counts = {}
        for source in sources:
            level = source.get("evidence_level", "Unknown")
            evidence_counts[level] = evidence_counts.get(level, 0) + 1

        # Build summary
        summary_parts = [f"Found {len(sources)} relevant sources after deduplication:"]

        for level, count in sorted(evidence_counts.items()):
            summary_parts.append(f"- {count} {level}(s)")

        # Add top sources
        summary_parts.append("\nTop sources by relevance:")
        for i, source in enumerate(sources[:3], 1):
            title = source.get("title", "Unknown")
            relevance = source.get("relevance_score", 0)
            summary_parts.append(f"{i}. {title} (relevance: {relevance:.2f})")

        return "\n".join(summary_parts)
