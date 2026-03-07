"""
Shared dependencies: compiled graph singleton and run store.
"""

from typing import Dict
from src.state.schema import PipelineState

# In-memory store for pipeline run states keyed by run_id
run_store: Dict[str, PipelineState] = {}

_compiled_graph = None


def get_graph():
    """Get or create the compiled LangGraph pipeline (singleton)."""
    global _compiled_graph
    if _compiled_graph is None:
        from src.graph.builder import build_pipeline_graph
        _compiled_graph = build_pipeline_graph()
    return _compiled_graph


def get_run_store() -> Dict[str, PipelineState]:
    """Get the run store dict."""
    return run_store
