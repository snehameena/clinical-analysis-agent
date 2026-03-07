"""
LangGraph StateGraph builder for the healthcare intelligence pipeline.
"""

from langgraph.graph import StateGraph, START, END
from src.state.schema import PipelineState
from src.graph.nodes import (
    coordinator_node,
    research_node,
    analysis_node,
    writing_node,
    quality_node,
)
from src.graph.edges import quality_routing


def build_pipeline_graph() -> StateGraph:
    """
    Build and compile the healthcare intelligence pipeline graph.

    Flow:
        START → coordinator → research → analysis → writing → quality
        quality → writing  (if verdict == "revise" and iteration < max)
        quality → END      (if verdict == "pass" / "reject" / max iterations)

    Returns:
        Compiled LangGraph StateGraph
    """
    graph = StateGraph(PipelineState)

    # Add nodes
    graph.add_node("coordinator", coordinator_node)
    graph.add_node("research", research_node)
    graph.add_node("analysis", analysis_node)
    graph.add_node("writing", writing_node)
    graph.add_node("quality", quality_node)

    # Linear edges
    graph.add_edge(START, "coordinator")
    graph.add_edge("coordinator", "research")
    graph.add_edge("research", "analysis")
    graph.add_edge("analysis", "writing")
    graph.add_edge("writing", "quality")

    # Conditional edge from quality → writing (revise) or END
    graph.add_conditional_edges(
        "quality",
        quality_routing,
        {"writing": "writing", "end": END},
    )

    return graph.compile()
