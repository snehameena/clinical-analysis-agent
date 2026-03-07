"""
Chainlit UI for local usage (no FastAPI).

Modes:
1) Pipeline: each user message runs the full LangGraph pipeline and returns the final report.
2) Agent chat: chat directly with a selected agent using its configured system prompt.
"""

from __future__ import annotations

import sys
import uuid
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

# Add project root to sys.path so `src.*` imports work when
# launched via `chainlit run src/chainlit_app.py`
_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import chainlit as cl
from chainlit.input_widget import Select

from src.env import load_env
from src.graph.builder import build_pipeline_graph
from src.state.schema import PipelineState
from src.agents.coordinator import CoordinatorAgent
from src.agents.research import ResearchAgent
from src.agents.analysis import AnalysisAgent
from src.agents.writing import WritingAgent
from src.agents.quality import QualityAgent


AGENTS = {
    "coordinator": CoordinatorAgent,
    "research": ResearchAgent,
    "analysis": AnalysisAgent,
    "writing": WritingAgent,
    "quality": QualityAgent,
}

# Ensure API keys are available for provider/Tavily usage.
load_env()


@lru_cache(maxsize=1)
def _get_graph():
    return build_pipeline_graph()


def _build_initial_state(topic: str) -> PipelineState:
    run_id = str(uuid.uuid4())
    return PipelineState(
        run_id=run_id,
        topic=topic,
        scope_instructions="",
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


def _trim_history(history: List[Dict[str, str]], max_turns: int = 20) -> List[Dict[str, str]]:
    # max_turns counts total messages (user+assistant). Keep the tail.
    if len(history) <= max_turns:
        return history
    return history[-max_turns:]


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("mode", "pipeline")
    cl.user_session.set("agent", "coordinator")
    cl.user_session.set("history", [])
    cl.user_session.set("session_id", str(uuid.uuid4()))

    settings = cl.ChatSettings(
        [
            Select(
                id="mode",
                label="Mode",
                values=["pipeline", "agent_chat"],
                initial_index=0,
            ),
            Select(
                id="agent",
                label="Agent (for agent_chat)",
                values=list(AGENTS.keys()),
                initial_index=0,
            ),
        ]
    )
    await settings.send()

    await cl.Message(
        content=(
            "Chainlit UI ready.\n"
            "- `pipeline`: each message runs the full pipeline and returns a report.\n"
            "- `agent_chat`: chat directly with a selected agent prompt.\n"
            "\nCommands:\n"
            "- `/mode pipeline` or `/mode agent_chat`\n"
            "- `/agent coordinator|research|analysis|writing|quality`\n"
        ),
        actions=[
            cl.Action(name="set_mode", label="Pipeline mode", value="pipeline"),
            cl.Action(name="set_mode", label="Agent chat", value="agent_chat"),
            cl.Action(name="set_agent", label="Agent: coordinator", value="coordinator"),
            cl.Action(name="set_agent", label="Agent: research", value="research"),
            cl.Action(name="set_agent", label="Agent: analysis", value="analysis"),
            cl.Action(name="set_agent", label="Agent: writing", value="writing"),
            cl.Action(name="set_agent", label="Agent: quality", value="quality"),
            cl.Action(name="reset", label="Reset chat", value="reset"),
        ],
    ).send()


@cl.on_settings_update
async def on_settings_update(settings: Dict[str, Any]):
    mode = (settings.get("mode") or "pipeline").strip()
    agent = (settings.get("agent") or "coordinator").strip()
    if mode not in ("pipeline", "agent_chat"):
        mode = "pipeline"
    if agent not in AGENTS:
        agent = "coordinator"
    cl.user_session.set("mode", mode)
    cl.user_session.set("agent", agent)


@cl.action_callback("set_mode")
async def on_set_mode(action: cl.Action):
    mode = (action.value or "").strip()
    if mode not in ("pipeline", "agent_chat"):
        await cl.Message(content=f"Unknown mode: {mode!r}").send()
        return
    cl.user_session.set("mode", mode)
    await cl.Message(content=f"Mode set to `{mode}`.").send()


@cl.action_callback("set_agent")
async def on_set_agent(action: cl.Action):
    agent = (action.value or "").strip()
    if agent not in AGENTS:
        await cl.Message(content=f"Unknown agent: {agent!r}").send()
        return
    cl.user_session.set("agent", agent)
    cl.user_session.set("history", [])
    await cl.Message(content=f"Agent set to `{agent}` (history cleared).").send()


@cl.action_callback("reset")
async def on_reset(action: cl.Action):
    cl.user_session.set("history", [])
    await cl.Message(content="Session history cleared.").send()


@cl.on_message
async def on_message(message: cl.Message):
    text = (message.content or "").strip()

    if text.lower().startswith("/mode"):
        parts = text.split(maxsplit=1)
        if len(parts) != 2:
            await cl.Message(content="Usage: `/mode pipeline` or `/mode agent_chat`").send()
            return
        mode = parts[1].strip()
        if mode not in ("pipeline", "agent_chat"):
            await cl.Message(content=f"Unknown mode: {mode!r}").send()
            return
        cl.user_session.set("mode", mode)
        await cl.Message(content=f"Mode set to `{mode}`.").send()
        return

    if text.lower().startswith("/agent"):
        parts = text.split(maxsplit=1)
        if len(parts) != 2:
            await cl.Message(content="Usage: `/agent coordinator|research|analysis|writing|quality`").send()
            return
        agent_name = parts[1].strip()
        if agent_name not in AGENTS:
            await cl.Message(content=f"Unknown agent: {agent_name!r}").send()
            return
        cl.user_session.set("agent", agent_name)
        cl.user_session.set("history", [])
        await cl.Message(content=f"Agent set to `{agent_name}` (history cleared).").send()
        return

    mode = cl.user_session.get("mode") or "pipeline"

    if mode == "pipeline":
        topic = text
        if not topic:
            await cl.Message(content="Please provide a non-empty topic.").send()
            return

        status = cl.Message(content="Running pipeline...")
        await status.send()

        state = _build_initial_state(topic)
        try:
            graph = _get_graph()
            result = await graph.ainvoke(state)
        except Exception as e:
            status.content = f"Pipeline failed: {e}"
            await status.update()
            return

        verdict = result.get("quality_verdict", "")
        score = result.get("quality_score", 0.0)
        wc = result.get("report_word_count", 0)
        status.content = f"Pipeline completed. verdict={verdict} score={score:.0f} word_count={wc}"
        await status.update()

        report_md = result.get("report_markdown", "") or "(No report_markdown returned.)"
        await cl.Message(content=report_md).send()
        return

    # agent_chat mode
    agent_name = cl.user_session.get("agent") or "coordinator"
    agent_cls = AGENTS.get(agent_name, CoordinatorAgent)
    agent = agent_cls()
    agent._current_run_id = cl.user_session.get("session_id")

    history: List[Dict[str, str]] = cl.user_session.get("history") or []
    history.append({"role": "user", "content": message.content})
    history = _trim_history(history)

    try:
        response_text = await agent._call_llm_messages(history)
    except Exception as e:
        await cl.Message(content=f"Agent call failed: {e}").send()
        return

    history.append({"role": "assistant", "content": response_text})
    history = _trim_history(history)
    cl.user_session.set("history", history)

    await cl.Message(content=response_text).send()
