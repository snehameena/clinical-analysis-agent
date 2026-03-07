"""
Writing agent - Clinical report composition and revision.
Generates structured markdown reports with sections and citations.
"""

import yaml
import json
from typing import Optional
from pathlib import Path
from src.agents.base import BaseAgent
from src.state.schema import PipelineState
from src.tools.text_utils import count_words
from src.state.models import WritingOutput


class WritingAgent(BaseAgent):
    """Agent responsible for composing clinical reports"""

    def __init__(self, model: Optional[str] = None):
        """Initialize writing agent"""
        super().__init__(name="writing", model=model)

    def _get_system_prompt(self) -> str:
        """Load system prompt from config"""
        try:
            repo_root = Path(__file__).resolve().parents[2]
            path = repo_root / "config" / "prompts.yaml"
            with open(path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                return config.get("writing", {}).get("system", "")
        except FileNotFoundError:
            return "You are a Clinical Report Writer. Compose structured, evidence-based reports."

    async def execute(self, state: PipelineState) -> PipelineState:
        """
        Generate or revise clinical report.

        Args:
            state: Pipeline state with clinical analysis

        Returns:
            Updated state with report_sections, report_markdown, citations
        """
        topic = state.get("topic", "")
        analysis_narrative = state.get("analysis_narrative", "")
        clinical_claims = state.get("clinical_claims", [])
        evidence_gaps = state.get("evidence_gaps", [])
        deduplicated_sources = state.get("deduplicated_sources", [])
        quality_iteration = state.get("quality_iteration", 0)
        revision_instructions = state.get("revision_instructions", "")

        # Determine if this is revision mode
        is_revision = bool(revision_instructions)

        if is_revision:
            return await self._revise_report(state, revision_instructions)
        else:
            return await self._generate_initial_report(
                state, topic, analysis_narrative, clinical_claims, evidence_gaps, deduplicated_sources
            )

    async def _generate_initial_report(
        self,
        state: PipelineState,
        topic: str,
        analysis_narrative: str,
        clinical_claims: list,
        evidence_gaps: list,
        sources: list,
    ) -> PipelineState:
        """Generate initial report"""

        # Format sources for report
        sources_text = self._format_sources_for_report(sources)

        user_message = f"""
Topic: {topic}

Clinical Analysis:
{analysis_narrative}

Key Clinical Claims:
{json.dumps(clinical_claims[:5], indent=2)}

Evidence Gaps:
{json.dumps(evidence_gaps[:3], indent=2)}

Available Sources:
{sources_text}

Please generate a structured clinical report with these sections:
1. Executive Summary (2-3 paragraphs)
2. Clinical Background
3. Evidence Synthesis (organized by key findings)
4. Key Findings and Recommendations
5. Evidence Gaps and Future Directions
6. References

        Return as JSON with:
- report_sections: dict mapping section names to markdown
- report_markdown: complete markdown report
- citations: list of formatted citations
"""

        try:
            result = await self._call_llm_json(
                user_message,
                required_keys=["report_sections", "report_markdown", "citations"],
            )
            validated = WritingOutput.model_validate(result).model_dump()
        except Exception as e:
            raise ValueError(f"Writing agent failed to generate report: {e}")

        # Update state
        state["report_sections"] = validated["report_sections"]
        state["report_markdown"] = validated["report_markdown"]
        state["citations"] = validated["citations"]
        state["report_word_count"] = count_words(validated["report_markdown"])

        # Record execution
        state = self._record_execution(state)

        return state

    async def _revise_report(self, state: PipelineState, revision_instructions: str) -> PipelineState:
        """Revise existing report based on quality feedback"""

        current_report = state.get("report_markdown", "")
        current_sections = state.get("report_sections", {})

        user_message = f"""
Current report:
{current_report}

Revision instructions:
{revision_instructions}

Please revise the report addressing the instructions above.
Only update the flagged sections - preserve unchanged sections exactly.

        Return as JSON with:
- report_sections: updated dict of sections
- report_markdown: complete revised markdown report
- citations: updated citations list
"""

        try:
            result = await self._call_llm_json(
                user_message,
                required_keys=["report_sections", "report_markdown", "citations"],
            )
            validated = WritingOutput.model_validate(result).model_dump()
        except Exception as e:
            raise ValueError(f"Writing agent revision failed: {e}")

        # Update state
        state["report_sections"] = validated.get("report_sections") or current_sections
        state["report_markdown"] = validated.get("report_markdown") or current_report
        state["citations"] = validated.get("citations") or state.get("citations", [])
        state["report_word_count"] = count_words(state.get("report_markdown", ""))

        # Increment iteration counter
        state["quality_iteration"] = state.get("quality_iteration", 0) + 1

        # Record execution
        state = self._record_execution(state)

        return state

    def _format_sources_for_report(self, sources: list) -> str:
        """Format sources for inclusion in report context"""
        formatted = []
        for i, source in enumerate(sources[:10], 1):  # Limit to top 10
            title = source.get("title", "Unknown")
            url = source.get("url", "")
            evidence_level = source.get("evidence_level", "Unknown")
            formatted.append(f"{i}. {title} ({evidence_level})\n   URL: {url}")

        return "\n".join(formatted)
