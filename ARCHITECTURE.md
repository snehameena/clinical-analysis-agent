# Architecture: Healthcare Content Intelligence Pipeline

## System Overview

This system implements a **multi-agent pipeline** where five specialized AI agents collaborate sequentially to produce evidence-based clinical reports. The pipeline is orchestrated by **LangGraph** as a directed state graph with a conditional revision loop.

## Data Flow

```
Input (topic, scope, audience)
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│                    PipelineState (TypedDict)              │
│  Shared mutable state passed through every agent node    │
│  ~30 fields covering: request → coordinator → research   │
│  → analysis → writing → quality → control flow           │
└──────────────────────────────────────────────────────────┘
    │
    ▼
[Coordinator] → research_queries, scope_boundaries, priority_subtopics
    │
    ▼
[Research]    → raw_sources, deduplicated_sources, research_summary
    │            (via Tavily healthcare-domain search)
    ▼
[Analysis]    → clinical_claims, evidence_gaps, contradictions,
    │            statistical_findings, analysis_narrative
    ▼
[Writing]     → report_sections, report_markdown, citations, word_count
    │
    ▼
[Quality]     → quality_score, quality_verdict, quality_issues,
    │            revision_instructions
    │
    ├── verdict == "pass"  ──→ END (pipeline_status = "completed")
    ├── verdict == "reject" ──→ END
    └── verdict == "revise" && iteration < max ──→ [Writing] (loop)
```

## Agent Contracts

### CoordinatorAgent
- **Input**: `topic`, `scope_instructions`, `target_audience`
- **Output**: `research_queries` (3-5 strings), `scope_boundaries` (in/out of scope lists), `priority_subtopics`
- **LLM call**: Single call generating a JSON research plan

### ResearchAgent
- **Input**: `research_queries`
- **Output**: `raw_sources`, `deduplicated_sources`, `research_summary`
- **External tool**: Tavily search (healthcare domain filtering, min relevance 0.6)
- **Logic**: Iterates queries → search → parse → deduplicate by URL → summarize

### AnalysisAgent
- **Input**: `deduplicated_sources`, `topic`
- **Output**: `clinical_claims`, `evidence_gaps`, `contradictions`, `statistical_findings`, `analysis_narrative`
- **LLM call**: Single call analyzing formatted sources

### WritingAgent
- **Input (initial)**: `analysis_narrative`, `clinical_claims`, `evidence_gaps`, `deduplicated_sources`
- **Input (revision)**: `report_markdown`, `revision_instructions`
- **Output**: `report_sections`, `report_markdown`, `citations`, `report_word_count`
- **Modes**: Initial generation vs. targeted revision (based on `quality_iteration`)

### QualityAgent
- **Input**: `report_markdown`, `clinical_claims`, `evidence_gaps`
- **Output**: `quality_score` (0-100), `quality_verdict` (pass/revise/reject), `quality_issues`, `revision_instructions`
- **Logic**: Evaluates accuracy, completeness, safety, evidence grading, clarity

## LangGraph Wiring

```python
graph = StateGraph(PipelineState)
# Linear: START → coordinator → research → analysis → writing → quality
# Conditional: quality → writing (revise) | END (pass/reject/max iterations)
```

- **Nodes** (`src/graph/nodes.py`): Thin wrappers that instantiate agents, set `current_agent`, time execution, catch errors
- **Edges** (`src/graph/edges.py`): `quality_routing()` checks verdict + iteration count
- **Builder** (`src/graph/builder.py`): Assembles and compiles the StateGraph

## API Layer

FastAPI with 4 pipeline endpoints + health check:

- `POST /run` — Creates initial state, spawns background task, returns `run_id`
- `GET /status` — Reads current state from in-memory `run_store`
- `GET /stream` — SSE endpoint polling every 2 seconds
- `GET /report` — Returns final report (only when `pipeline_status == "completed"`)

## Frontend

Streamlit single-page app with three zones:

1. **Sidebar** — Input form (topic, scope, audience, format, max iterations)
2. **Progress** — Real-time agent progress bars via status polling
3. **Report** — Tabbed view (Full Report / Sections / Sources / Quality Review)

## Async Model

- All agents use `async def execute()` for LangGraph compatibility
- The sync `Anthropic` client is wrapped with `asyncio.to_thread()` to avoid blocking the event loop
- Similarly, the sync `TavilyClient.search()` is wrapped with `asyncio.to_thread()`
- FastAPI runs the pipeline in a background task via `BackgroundTasks`

## Testing Strategy

- **Unit tests**: Each agent tested in isolation with `_call_llm` mocked via `AsyncMock`
- **Integration tests**: Full pipeline graph with all LLM/Tavily calls mocked
- **API tests**: FastAPI `TestClient` with mocked graph compilation
- **No API keys needed**: All tests run without external service access
