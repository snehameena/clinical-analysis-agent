# Plans And Change Log

This file captures:
- Proposed plans we agreed on (to implement later)
- A running summary of major changes already implemented

## Proposed Plan (Saved)

### SQLite Run Index + Per-Run Log Files + Per-Agent Timing (Multi-Call Safe)

Goal: Add a local SQLite DB to index each run (`run_id`, topic/request metadata, status, per-run log path) and record per-agent execution timing in a way that supports multiple invocations of the same agent (revision loops). Keep detailed I/O logs in a per-run JSONL file keyed by `run_id`.

Design:
1. SQLite DB (local run registry)
   - Default path: `logs/runs.sqlite` (configurable via `RUN_DB_PATH`)
   - Tables:
     - `runs` (1 row per run_id)
       - `run_id TEXT PRIMARY KEY`
       - `created_at TEXT`
       - `source TEXT` (fastapi|chainlit|notebook)
       - `topic TEXT`
       - `scope_instructions TEXT NULL`
       - `target_audience TEXT NULL`
       - `report_format TEXT NULL`
       - `status TEXT` (pending|running|completed|failed)
       - `error_message TEXT NULL`
       - `log_path TEXT` (e.g. `logs/runs/<run_id>.jsonl`)
       - Per-agent aggregates (totals + counts):
         - `coordinator_total_ms INTEGER`, `coordinator_calls INTEGER`
         - `research_total_ms INTEGER`, `research_calls INTEGER`
         - `analysis_total_ms INTEGER`, `analysis_calls INTEGER`
         - `writing_total_ms INTEGER`, `writing_calls INTEGER`
         - `quality_total_ms INTEGER`, `quality_calls INTEGER`
         - `total_ms INTEGER`
     - `agent_steps` (1 row per agent invocation; supports revisions/retries)
       - `id INTEGER PRIMARY KEY AUTOINCREMENT`
       - `run_id TEXT` (FK -> runs.run_id)
       - `step_index INTEGER` (monotonic per run)
       - `agent_name TEXT`
       - `quality_iteration INTEGER NULL`
       - `started_at TEXT`, `ended_at TEXT`, `duration_ms INTEGER`
       - `status TEXT` (ok|error)
       - `error_message TEXT NULL`
2. Per-run log files (JSONL)
   - Default sink: `logs/runs/<run_id>.jsonl`
   - Optional global sink: `logs/agent_io.jsonl` (toggle by env var; default off)
   - Each JSONL event includes `run_id`
3. Hook points
   - Run creation:
     - FastAPI `POST /api/v1/pipeline/run`
     - Chainlit pipeline start
     - Notebook helper (optional)
   - Run execution:
     - Set run status to running/completed/failed
   - Agent steps (LangGraph node wrappers):
     - Start step row before agent executes
     - End step row after success/error
     - Update per-agent aggregates in `runs`
4. Concurrency + reliability
   - SQLite WAL mode
   - One-connection-per-operation
   - Atomic `step_index` assignment (transaction)

Smoke acceptance checks:
- One run inserts into `runs`
- Each agent invocation inserts into `agent_steps` (5+ rows if revision loops)
- Aggregate totals/calls in `runs` reflect revisions (e.g. writing_calls > 1)
- Per-run JSONL exists and contains agent input/output events

## Summary Of Changes Implemented So Far

### LLM Provider Abstraction + Config-Driven Model Selection
- Added `src/llm/*`:
  - `src/llm/agent_config.py` loads per-agent LLM config from `config/agents.yaml` (config-only).
  - `src/llm/providers.py` implements `AnthropicProvider`, `OpenAIProvider` (Gemini stub).
- Refactored `src/agents/base.py` to route LLM calls through providers and to log request/response metadata.
- `config/agents.yaml` now includes `provider` and has been switched to `openai` + `gpt-5.1` for all agents.
- Increased analysis output token limit:
  - `analysis.max_tokens` set to `10000`.

### Robust JSON Handling For Agent Outputs
- Added Pydantic normalization models in `src/state/models.py`:
  - Normalizes common LLM shape drift (e.g., list vs string, dict vs list).
  - Notably, `QualityOutput.revision_instructions` now supports dict/list and is coerced into a string.
- Added `BaseAgent._call_llm_json()`:
  - Parse JSON; if invalid, repair once; if still invalid, request a minimal skeleton object.
- Improved JSON extraction in `BaseAgent._parse_json_response()`:
  - Added YAML fallback
  - Added "first balanced JSON object" extraction for mixed outputs

### Pipeline Orchestration + Revision Loop Fix
- Fixed the quality revise loop trigger in `src/agents/writing.py`:
  - Revision mode now activates when `revision_instructions` is present (not gated by `quality_iteration > 0`).

### Chainlit UI (No API)
- Added `src/chainlit_app.py`:
  - Pipeline mode (runs full pipeline locally)
  - Agent chat mode with `/mode` and `/agent` commands plus action buttons
- Fixed Chainlit settings widget usage for Chainlit 2.3.0 by importing `Select` from `chainlit.input_widget`.

### Environment Loading
- Added `src/env.py` with `load_env()`:
  - Loads `.env.local` then `.env` (process env always wins).
- Wired env loading into:
  - `src/api/main.py`
  - `src/frontend/app.py`
  - `src/chainlit_app.py`

### Debug Logging (JSONL)
- Added `src/debug/agent_io.py` writing `logs/agent_io.jsonl`:
  - `agent_state` events: before/after/error with state summaries
  - `llm_io` events: prompts/responses (truncated) + meta (usage/finish_reason where available)
- Updated LangGraph node wrappers in `src/graph/nodes.py` to emit agent_state logs and capture run_id.

### Notebook: Step-By-Step Agent Execution
- Added a step-by-step notebook and updated it after moving it under `src/debug/`:
  - `src/debug/agent_flow_debug.ipynb`
  - Repo root discovery + imports fixed to `src.*`.

### Fix: Writing Output Truncation Causing Empty Reports (2026-03-07)
- Increased `writing.max_tokens` to avoid JSON truncation while generating full reports.
- Improved `BaseAgent._call_llm_json()` to detect likely truncation (e.g., `usage.output_tokens` pegged at `max_tokens`) and raise an actionable error instead of silently falling back to an empty skeleton JSON.
- Added a fail-fast guard in `WritingAgent` to raise if it produces an empty report/sections, preventing `QualityAgent` from running with missing `report_markdown`.
- Updated `QualityAgent` to reconstruct a best-effort report from `report_sections` when `report_markdown` is missing.

### Fix: Analysis Timeout Surfaces As Blank Error (2026-03-07)
- Increased `analysis.timeout` to account for long generations when `analysis.max_tokens` is high.
- Updated `OpenAIProvider.generate()` to re-raise `asyncio.TimeoutError` with a clear message (timeout/model and what to change).
- Updated `AnalysisAgent` to surface the underlying exception type/message (e.g. timeout) instead of reporting a JSON parse failure.

### Fix: AnalysisOutput Clinical Claims Type Drift (2026-03-07)
- Updated `AnalysisOutput` normalization so `clinical_claims` can be either `list[object]` or `list[str]` (strings are coerced into `{"claim": "..."}`) before validating into `ClinicalClaimModel`.

### Runtime Bounding (<= 180s Per Agent Call) (2026-03-07)
- Updated prompts to include explicit max item/length limits so agents stay concise.
- Reduced `analysis` / `writing` `max_tokens` and set `writing`/`quality` timeouts to `<= 180s`.
- Switched WritingAgent to request **sections-only JSON** and assemble `report_markdown` locally, reducing output size and truncation risk.
- Bounded AnalysisAgent input context (top sources + shorter snippets) to reduce latency.

## Proposed Plan (Saved)

### SQLite Performance Monitoring + Metrics Collection (Runs + Steps + LLM Calls)

Goal: Add local performance monitoring and metrics collection using SQLite, without storing raw prompts/responses in the DB. Keep raw I/O in JSONL logs. Support revise loops (multiple invocations of writing/quality).

DB:
- Path: `logs/runs.sqlite` (env override: `RUN_DB_PATH`)
- Enable: `RUN_DB_ENABLED=1` (default on)
- Reliability: WAL mode + busy_timeout; one-connection-per-op

Tables:
1) `runs` (1 row per run_id)
- run metadata: run_id, created_at, ended_at, source (fastapi|chainlit|notebook), topic, scope_instructions, target_audience, report_format, status, error_message, log_path
- per-agent aggregates: {agent}_calls + {agent}_total_ms, total_ms
- token aggregates: llm_calls, llm_input_tokens, llm_output_tokens, llm_total_tokens, llm_truncated_calls, llm_error_calls
- volume: tavily_calls, tavily_results, raw_sources_count, deduplicated_sources_count, clinical_claims_count, citations_count, report_word_count
- final: quality_iteration_final, quality_verdict_final, quality_score_final

2) `agent_steps` (1 row per agent invocation)
- run_id, step_index (monotonic), agent_name, quality_iteration
- started_at, ended_at, duration_ms, status (ok|error), error_message
- end-of-step snapshots: raw_sources_count, deduplicated_sources_count, clinical_claims_count, citations_count, report_word_count

3) `llm_calls` (1 row per provider call, including JSON repair calls)
- run_id, step_id (nullable), agent_name, provider, model, api
- call_kind (primary|repair1|repair2)
- started_at, ended_at, duration_ms, status (ok|error), error_type, error_message
- usage: input_tokens, output_tokens, total_tokens
- finish_reason, truncated

Hook points:
- Run lifecycle:
  - FastAPI `/api/v1/pipeline/run`: insert run row (pending)
  - pipeline start: running
  - pipeline end: completed/failed + finalize from state
  - Chainlit pipeline mode: same lifecycle
- Agent steps:
  - LangGraph node wrappers: step_start/step_end + per-agent aggregate updates
  - Pass step_id into agent instance so LLM calls can link to steps
- LLM calls:
  - BaseAgent._call_llm logs each provider call to llm_calls and updates run token aggregates
  - _call_llm_json marks call_kind for repair calls
- Tavily:
  - ResearchAgent increments tavily_calls and tavily_results per query

JSONL logs:
- Keep `logs/agent_io.jsonl`
- Add per-run JSONL files `logs/runs/<run_id>.jsonl`
- Env: `AGENT_IO_LOG_MODE=global|per_run|both` (default both)

Smoke checks:
- One run => 1 `runs` row, >=5 `agent_steps` rows, multiple `llm_calls` rows
- Revise loop => writing/quality have multiple step rows and call aggregates > 1
- Error/timeout => runs.status=failed, last step status=error, llm_calls.status=error
