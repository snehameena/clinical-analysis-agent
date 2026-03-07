"""
LangGraph conditional edge routing functions.
"""

from src.state.schema import PipelineState


def quality_routing(state: PipelineState) -> str:
    """
    Route from quality node based on verdict and iteration count.

    Returns:
        "writing" — if verdict is "revise" and iterations remain
        "end"     — if verdict is "pass", "reject", or max iterations reached
    """
    verdict = state.get("quality_verdict", "reject")
    iteration = state.get("quality_iteration", 0)
    max_iter = state.get("max_quality_iterations", 3)

    if verdict == "pass":
        state["pipeline_status"] = "completed"
        return "end"

    if verdict == "revise" and iteration < max_iter:
        return "writing"

    # reject or max iterations reached
    state["pipeline_status"] = "completed"
    return "end"
