# Healthcare Content Intelligence Pipeline

A multi-agent clinical evidence synthesis system built with **Claude (Anthropic)**, **LangGraph**, **Tavily**, **FastAPI**, and **Streamlit**.

Given a healthcare topic, the pipeline automatically searches medical literature, extracts clinical claims, composes a structured report, and validates quality through an iterative review loop.

## Architecture

```
[User] → Streamlit UI → FastAPI Backend → LangGraph Pipeline
                                              │
                     ┌────────────────────────┘
                     ▼
              ┌─────────────┐
              │ Coordinator  │  Plan queries & scope
              └──────┬───────┘
                     ▼
              ┌─────────────┐
              │  Research    │  Tavily search + dedup
              └──────┬───────┘
                     ▼
              ┌─────────────┐
              │  Analysis    │  Extract claims & gaps
              └──────┬───────┘
                     ▼
              ┌─────────────┐
              │  Writing     │  Compose markdown report
              └──────┬───────┘
                     ▼
              ┌─────────────┐
              │  Quality     │──── pass ──→ [END]
              └──────┬───────┘
                     │ revise (≤3 iterations)
                     └──→ [Writing]
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env.local
# Edit .env.local with your API keys (recommended):
#   ANTHROPIC_API_KEY=sk-ant-...
#   TAVILY_API_KEY=tvly-...
#   OPENAI_API_KEY=sk-... (only needed if you select provider: openai in config/agents.yaml)
```

Notes:
- `.env.local` is ignored by git and is the preferred place for secrets.
- `.env` is optional fallback; `.env.local` takes precedence.

### 2a. Choose models/providers (config-only)

Per-agent provider/model settings are defined in `config/agents.yaml`.

Example (ResearchAgent on OpenAI, others on Claude):

```yaml
research:
  provider: openai
  model: gpt-4o-mini
  temperature: 0.2
  max_tokens: 2000
  timeout: 60
```

### 3. Run tests (no API keys needed)

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests (all mocked)
pytest tests/integration/ -v

# All tests
pytest -v
```

### 4. Start the backend

```bash
uvicorn src.api.main:app --reload --port 8000
```

### 5. Start the frontend

```bash
streamlit run src/frontend/app.py --server.port 8501
```

### 5a. Alternative UI: Chainlit (no API)

Chainlit runs the pipeline locally (no FastAPI) and also supports chatting directly with a selected agent prompt.

```bash
chainlit run src/chainlit_app.py
```

### Debug logs (agent I/O)

By default, the pipeline writes a JSONL debug log to `logs/agent_io.jsonl` containing:
- Per-agent state input/output summaries
- LLM prompt/response payloads (truncated)

You can disable this by setting:
`AGENT_IO_LOG_ENABLED=0`

### 6. Try a sample topic

Open http://localhost:8501 and enter:

> Effectiveness of telemedicine interventions for Type 2 diabetes management in adults

## Project Structure

```
├── config/                  # YAML configuration
│   ├── agents.yaml          # Per-agent model/temperature settings
│   ├── pipeline.yaml        # LangGraph pipeline config
│   └── prompts.yaml         # System prompts for each agent
├── src/
│   ├── state/schema.py      # PipelineState TypedDict + enums
│   ├── agents/              # 5 agent classes (coordinator, research, analysis, writing, quality)
│   │   └── base.py          # Abstract BaseAgent with async LLM calls
│   ├── tools/               # Tavily search + text utilities
│   ├── graph/               # LangGraph nodes, edges, builder
│   ├── api/                 # FastAPI backend (models, routes, dependencies)
│   └── frontend/            # Streamlit UI (sidebar, progress, report view)
├── tests/
│   ├── conftest.py          # Shared fixtures and mocks
│   ├── unit/                # Agent-level unit tests
│   └── integration/         # Pipeline + API integration tests
├── requirements.txt
└── .env.example
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/health` | Health check |
| POST | `/api/v1/pipeline/run` | Launch a pipeline run |
| GET | `/api/v1/pipeline/{run_id}/status` | Poll run status |
| GET | `/api/v1/pipeline/{run_id}/stream` | SSE status stream |
| GET | `/api/v1/pipeline/{run_id}/report` | Get final report |

## Key Design Decisions

- **Sync Anthropic client + `asyncio.to_thread`**: Simpler than `AsyncAnthropic`, avoids dual-client maintenance
- **TypedDict state**: LangGraph-compatible state schema with `Annotated` accumulator for `agent_history`
- **Quality revision loop**: Conditional LangGraph edge routes back to writing agent up to N iterations
- **In-memory run store**: Suitable for single-server demo; swap for Redis/DB in production

## Tech Stack

- **LLM**: Claude 3.5 Sonnet (Anthropic)
- **Search**: Tavily (healthcare domain filtering)
- **Orchestration**: LangGraph (StateGraph)
- **Backend**: FastAPI + SSE
- **Frontend**: Streamlit
- **Testing**: pytest + pytest-asyncio
