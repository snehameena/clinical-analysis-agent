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
1. Executive Summary
2. Clinical Background
3. Evidence Synthesis
4. Safety and Harms
5. Evidence Grading
6. Key Findings and Recommendations
7. Evidence Gaps and Future Directions
8. References

Keep the total report length ~1500-2500 words. Be concise and avoid repetition.
References/citations: <= 25.

        Return as JSON with:
- report_sections: dict mapping section names to markdown (section body only; do not repeat the section title heading)
- citations: list of formatted citations
- completeness_checklist: optional (if you choose to include it)
"""

        try:
            result = await self._call_llm_json(
                user_message,
                required_keys=["report_sections", "citations"],
            )
            validated = WritingOutput.model_validate(result).model_dump()
        except Exception as e:
            raise ValueError(f"Writing agent failed to generate report: {e}")

        sections = validated.get("report_sections") or {}
        citations = validated.get("citations") or []
        if not sections:
            raise ValueError(
                "Writing agent produced empty report_sections. This is usually caused by output truncation "
                f"(max_tokens={self.max_tokens}) or invalid JSON repair fallback. Increase writing.max_tokens "
                "or tighten prompt constraints and rerun."
            )

        # Normalize common section naming drift.
        if not (sections.get("References") or "").strip():
            alt = (sections.get("Reference") or "").strip()
            if alt:
                sections["References"] = alt

        # If the model forgot to populate References, build it deterministically from citations/sources.
        if not citations:
            for s in (sources or [])[:25]:
                title = str(s.get("title", "Unknown")).strip()
                url = str(s.get("url", "")).strip()
                level = str(s.get("evidence_level", "")).strip()
                if not url:
                    continue
                if level:
                    citations.append(f"{title} ({level}) - {url}")
                else:
                    citations.append(f"{title} - {url}")

        if not (sections.get("References") or "").strip() and citations:
            sections["References"] = "\n".join([f"- {c}" for c in citations])

        section_order = [
            "Executive Summary",
            "Clinical Background",
            "Evidence Synthesis",
            "Safety and Harms",
            "Evidence Grading",
            "Key Findings and Recommendations",
            "Evidence Gaps and Future Directions",
            "References",
        ]
        missing = [name for name in section_order if not (sections.get(name) or "").strip()]
        if missing:
            raise ValueError(f"Writing agent report_sections missing/empty required sections: {missing}")

        parts = []
        for name in section_order:
            body = (sections.get(name) or "").strip()
            parts.append(f"## {name}\n\n{body}".rstrip())
        report_md = "\n\n".join(parts).strip()
        if not report_md:
            raise ValueError("Writing agent produced empty report markdown after assembling sections.")

        # Update state
        state["report_sections"] = sections
        state["report_markdown"] = report_md
        state["citations"] = citations
        state["report_word_count"] = count_words(report_md)

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
- citations: updated citations list
"""

        try:
            result = await self._call_llm_json(
                user_message,
                required_keys=["report_sections", "citations"],
            )
            validated = WritingOutput.model_validate(result).model_dump()
        except Exception as e:
            raise ValueError(f"Writing agent revision failed: {e}")

        sections = validated.get("report_sections") or {}
        citations = validated.get("citations") or []
        if not sections:
            raise ValueError(
                "Writing agent revision produced an empty report. This is usually caused by output truncation "
                f"(max_tokens={self.max_tokens}). Increase writing.max_tokens or tighten prompt constraints and rerun."
            )

        # Merge to support LLMs that only return updated sections.
        merged_sections = dict(current_sections or {})
        merged_sections.update(sections)
        sections = merged_sections

        section_order = [
            "Executive Summary",
            "Clinical Background",
            "Evidence Synthesis",
            "Safety and Harms",
            "Evidence Grading",
            "Key Findings and Recommendations",
            "Evidence Gaps and Future Directions",
            "References",
        ]
        parts = []
        for name in section_order:
            body = (sections.get(name) or "").strip()
            parts.append(f"## {name}\n\n{body}".rstrip())
        report_md = "\n\n".join(parts).strip() or current_report

        # Update state
        state["report_sections"] = sections or current_sections
        state["report_markdown"] = report_md
        state["citations"] = citations or state.get("citations", [])
        state["report_word_count"] = count_words(report_md or "")

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
