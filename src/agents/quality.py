"""
Quality agent - Report validation and QA.
Validates clinical accuracy, completeness, and clarity.
"""

import yaml
import json
from typing import Optional
from pathlib import Path
from src.agents.base import BaseAgent
from src.state.schema import PipelineState, QualityVerdict
from src.state.models import QualityOutput


class QualityAgent(BaseAgent):
    """Agent responsible for quality assurance and validation"""

    def __init__(self, model: Optional[str] = None):
        """Initialize quality agent"""
        super().__init__(name="quality", model=model)

    def _get_system_prompt(self) -> str:
        """Load system prompt from config"""
        try:
            repo_root = Path(__file__).resolve().parents[2]
            path = repo_root / "config" / "prompts.yaml"
            with open(path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                return config.get("quality", {}).get("system", "")
        except FileNotFoundError:
            return "You are a Quality Assurance Reviewer. Validate clinical reports."

    async def execute(self, state: PipelineState) -> PipelineState:
        """
        Evaluate report quality and provide verdict.

        Args:
            state: Pipeline state with report_markdown

        Returns:
            Updated state with quality_verdict, quality_score, quality_issues, revision_instructions
        """
        report = state.get("report_markdown", "") or ""
        clinical_claims = state.get("clinical_claims", [])
        evidence_gaps = state.get("evidence_gaps", [])
        topic = state.get("topic", "")

        if not report.strip():
            # If writing produced sections but not a full markdown blob, reconstruct a best-effort report.
            sections = state.get("report_sections") or {}
            if isinstance(sections, dict) and sections:
                parts = []
                for name, body in sections.items():
                    title = str(name).strip() or "Section"
                    text = "" if body is None else str(body).strip()
                    if not text:
                        continue
                    # Use a stable heading even if the body already contains headings.
                    parts.append(f"## {title}\n\n{text}")
                report = "\n\n".join(parts).strip()

        if not report.strip():
            raise ValueError("No report provided for quality review")

        sections = state.get("report_sections") or {}
        def _sec(name: str, limit: int = 1400) -> str:
            if isinstance(sections, dict):
                val = sections.get(name)
                if isinstance(val, str):
                    return val[:limit]
            return ""

        # Build QA prompt (bounded context: key sections only)
        user_message = f"""
Topic: {topic}

Key sections to review:

Executive Summary:
{_sec("Executive Summary")}

Evidence Synthesis:
{_sec("Evidence Synthesis")}

Safety and Harms:
{_sec("Safety and Harms")}

Evidence Grading:
{_sec("Evidence Grading")}

Clinical claims made:
{json.dumps(clinical_claims[:5], indent=2)}

Evidence gaps identified:
{json.dumps(evidence_gaps[:3], indent=2)}

Please evaluate this report on:
1. Clinical Accuracy - Are findings supported by cited evidence?
2. Completeness - Are all major subtopics addressed?
3. Safety - Are warnings/contraindications clearly stated?
4. Evidence Grading - Is evidence level explicitly stated?
5. Clarity - Is language clear to the target audience?

        Return as JSON with:
- quality_score: 0-100
- quality_verdict: "pass" (>75), "revise" (50-75), "reject" (<50)
- quality_issues: list (<= 10) with severity (critical/major/minor), section, description, recommendation
- revision_instructions: concise bullets (<= 600 words) describing specific sections to update (if revise verdict)
"""

        try:
            result = await self._call_llm_json(
                user_message,
                required_keys=["quality_score", "quality_verdict", "quality_issues", "revision_instructions"],
            )
            validated = QualityOutput.model_validate(result).model_dump()
        except Exception as e:
            raise ValueError(f"Quality agent failed to parse response: {e}")

        state["quality_score"] = float(validated.get("quality_score", 0.0))
        state["quality_verdict"] = validated.get("quality_verdict", "reject")
        state["quality_issues"] = validated.get("quality_issues", [])
        state["revision_instructions"] = validated.get("revision_instructions", "")
        state["should_revise"] = state["quality_verdict"] == "revise"

        # Record execution
        state = self._record_execution(state)

        return state
