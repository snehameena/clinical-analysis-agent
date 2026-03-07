"""
Coordinator agent - Planning and query generation.
Breaks down healthcare topics into focused research questions.
"""

import yaml
from typing import Optional
from pathlib import Path
from src.agents.base import BaseAgent
from src.state.schema import PipelineState
from src.state.models import CoordinatorOutput


class CoordinatorAgent(BaseAgent):
    """Agent responsible for planning and generating research queries"""

    def __init__(self, model: Optional[str] = None):
        """Initialize coordinator agent"""
        super().__init__(name="coordinator", model=model)

    def _get_system_prompt(self) -> str:
        """Load system prompt from config"""
        try:
            repo_root = Path(__file__).resolve().parents[2]
            path = repo_root / "config" / "prompts.yaml"
            with open(path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                return config.get("coordinator", {}).get("system", "")
        except FileNotFoundError:
            return "You are a Clinical Evidence Coordinator. Generate research queries for healthcare topics."

    async def execute(self, state: PipelineState) -> PipelineState:
        """
        Analyze topic and generate research plan.

        Args:
            state: Pipeline state with topic, scope_instructions, etc.

        Returns:
            Updated state with research_queries, scope_boundaries, priority_subtopics
        """
        topic = state.get("topic", "")
        scope_instructions = state.get("scope_instructions", "")
        target_audience = state.get("target_audience", "clinical practitioners")

        if not topic:
            raise ValueError("Topic not provided in state")

        # Build user message
        user_message = f"""
Topic: {topic}

Scope Instructions: {scope_instructions}

Target Audience: {target_audience}

Please generate:
1. 3-5 specific research queries optimized for medical literature search
2. Scope boundaries (what IS and IS NOT in scope)
3. 2-3 priority subtopics for deeper investigation

        Return as JSON with fields:
- research_queries: list of search queries
- scope_boundaries: object with "in_scope" and "out_of_scope" lists
- priority_subtopics: list of subtopics
"""

        try:
            result = await self._call_llm_json(
                user_message,
                required_keys=["research_queries", "scope_boundaries", "priority_subtopics"],
            )
            validated = CoordinatorOutput.model_validate(result).model_dump()
        except Exception as e:
            raise ValueError(f"Coordinator failed to generate valid JSON: {e}")

        # Update state
        state["research_queries"] = validated["research_queries"]
        state["scope_boundaries"] = validated["scope_boundaries"]
        state["priority_subtopics"] = validated["priority_subtopics"]

        # Record execution
        state = self._record_execution(state)

        return state
