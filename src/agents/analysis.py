"""
Analysis agent - Clinical evidence appraisal and synthesis.
Extracts claims, identifies gaps, detects contradictions.
"""

import yaml
import json
from typing import Optional
from pathlib import Path
from src.agents.base import BaseAgent
from src.state.schema import PipelineState
from src.state.models import AnalysisOutput


class AnalysisAgent(BaseAgent):
    """Agent responsible for clinical evidence analysis and appraisal"""

    def __init__(self, model: Optional[str] = None):
        """Initialize analysis agent"""
        super().__init__(name="analysis", model=model)

    def _get_system_prompt(self) -> str:
        """Load system prompt from config"""
        try:
            repo_root = Path(__file__).resolve().parents[2]
            path = repo_root / "config" / "prompts.yaml"
            with open(path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                return config.get("analysis", {}).get("system", "")
        except FileNotFoundError:
            return "You are a Clinical Evidence Analyst. Extract and appraise clinical findings."

    async def execute(self, state: PipelineState) -> PipelineState:
        """
        Analyze deduplicated sources and extract clinical evidence.

        Args:
            state: Pipeline state with deduplicated_sources

        Returns:
            Updated state with clinical_claims, evidence_gaps, contradictions, analysis_narrative
        """
        sources = state.get("deduplicated_sources", [])
        topic = state.get("topic", "")

        if not sources:
            raise ValueError("No deduplicated sources provided")

        # Format sources for analysis
        sources_text = self._format_sources_for_analysis(sources)

        # Build analysis prompt
        user_message = f"""
Topic: {topic}

Sources found in research:
{sources_text}

Please analyze these sources and extract:
1. Key clinical claims (specific findings about outcomes, safety, efficacy)
2. Evidence gaps (unanswered questions)
3. Contradictions (conflicting evidence)
4. Statistical findings (effect sizes, CIs, p-values)
5. Overall analysis narrative (2-3 paragraphs)

        Return as JSON with fields:
- clinical_claims: list of objects with claim, evidence_level, source_urls, effect_size, confidence_interval, p_value
- evidence_gaps: list of gaps
- contradictions: list of contradictions
- statistical_findings: list of statistical findings
- analysis_narrative: narrative summary
"""

        try:
            result = await self._call_llm_json(
                user_message,
                required_keys=[
                    "clinical_claims",
                    "evidence_gaps",
                    "contradictions",
                    "statistical_findings",
                    "analysis_narrative",
                ],
            )
            print("result::", result)
            validated = AnalysisOutput.model_validate(result).model_dump()
            print("validate::", validated)
            print('vcc::', validated["clinical_claims"])
        except Exception as e:
            raise ValueError(f"Analysis agent failed to parse response: {e}")

        # Update state
        # Pydantic normalizes list/dict/string variants into stable shapes.
        state["clinical_claims"] = validated["clinical_claims"]
        state["evidence_gaps"] = validated["evidence_gaps"]
        state["contradictions"] = validated["contradictions"]
        state["statistical_findings"] = validated["statistical_findings"]
        state["analysis_narrative"] = validated["analysis_narrative"]

        # Record execution
        state = self._record_execution(state)

        return state

    def _format_sources_for_analysis(self, sources: list) -> str:
        """
        Format sources into readable text for LLM analysis.

        Args:
            sources: List of Source objects

        Returns:
            Formatted sources text
        """
        formatted = []
        for i, source in enumerate(sources, 1):
            title = source.get("title", "Unknown")
            snippet = source.get("snippet", "")
            level = source.get("evidence_level", "Unknown")
            score = source.get("relevance_score", 0)

            formatted.append(
                f"{i}. [{level}] {title}\n"
                f"   Relevance: {score:.2f}\n"
                f"   Summary: {snippet[:300]}..."
            )

        return "\n".join(formatted)
