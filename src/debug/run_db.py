"""
SQLite-backed run index and metrics collection.

This DB is intended for local/dev debugging and performance monitoring.
It stores metrics only (durations, token usage, counters). Prompts/responses
remain in JSONL logs.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


_lock = threading.Lock()


def _enabled() -> bool:
    return os.getenv("RUN_DB_ENABLED", "1").strip().lower() not in ("0", "false", "no")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _db_path() -> Path:
    raw = os.getenv("RUN_DB_PATH", "").strip()
    if raw:
        return Path(raw)
    return _repo_root() / "logs" / "runs.sqlite"


def _artifacts_enabled() -> bool:
    return os.getenv("RUN_ARTIFACTS_ENABLED", "1").strip().lower() not in ("0", "false", "no")


def _artifacts_root() -> Path:
    raw = os.getenv("RUN_ARTIFACTS_DIR", "").strip()
    if raw:
        return Path(raw)
    return _repo_root() / "logs" / "artifacts"


def _artifact_dir(run_id: str) -> Path:
    # One directory per run.
    return _artifacts_root() / str(run_id)


def _safe_write_text(path: Path, text: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text or "")
    except Exception:
        return


def _safe_write_json(path: Path, obj: Any) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=True, indent=2)
    except Exception:
        return


def _write_run_artifacts_from_state(state: Dict[str, Any]) -> None:
    """
    Persist per-run artifacts for monitoring/debugging.

    This is intentionally best-effort and never raises, since it's called from pipeline finalization.
    """
    if not _enabled() or not _artifacts_enabled():
        return
    run_id = (state or {}).get("run_id")
    if not run_id:
        return

    base = _artifact_dir(str(run_id))

    report_md = state.get("report_markdown") or ""
    if isinstance(report_md, str) and report_md.strip():
        _safe_write_text(base / "report.md", report_md)

    sources = state.get("deduplicated_sources") or []
    # Keep sources as JSON (list[dict]) for the monitor UI.
    if isinstance(sources, list) and sources:
        _safe_write_json(base / "sources.json", sources)

    quality_obj = {
        "quality_score": state.get("quality_score"),
        "quality_verdict": state.get("quality_verdict"),
        "quality_iteration": state.get("quality_iteration"),
        "quality_issues": state.get("quality_issues") or [],
        "revision_instructions": state.get("revision_instructions") or "",
    }
    # Write even if empty so the UI can distinguish "not present" vs "present but empty".
    _safe_write_json(base / "quality.json", quality_obj)

    # A small summary that can be useful for offline inspection.
    summary = {
        "run_id": run_id,
        "topic": state.get("topic"),
        "status": state.get("pipeline_status"),
        "error_message": state.get("error_message"),
        "counts": {
            "raw_sources": len(state.get("raw_sources", []) or []),
            "deduplicated_sources": len(state.get("deduplicated_sources", []) or []),
            "clinical_claims": len(state.get("clinical_claims", []) or []),
            "citations": len(state.get("citations", []) or []),
        },
        "report_word_count": state.get("report_word_count"),
    }
    _safe_write_json(base / "summary.json", summary)


def _now_iso() -> str:
    return datetime.now().isoformat()


def _truncate(s: Optional[str], n: int = 2000) -> Optional[str]:
    if s is None:
        return None
    s2 = str(s)
    if len(s2) <= n:
        return s2
    return s2[: n - 20] + f"...(truncated,len={len(s2)})"


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path), timeout=5.0)
    con.row_factory = sqlite3.Row
    # Best-effort pragmas for local concurrency.
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA busy_timeout=5000;")
    return con


_initialized = False


def init_db() -> None:
    global _initialized
    if not _enabled():
        return
    with _lock:
        if _initialized:
            return
        try:
            con = _connect()
            with con:
                con.execute(
                    """
                    CREATE TABLE IF NOT EXISTS runs (
                        run_id TEXT PRIMARY KEY,
                        created_at TEXT,
                        ended_at TEXT,
                        source TEXT,
                        topic TEXT,
                        scope_instructions TEXT,
                        target_audience TEXT,
                        report_format TEXT,
                        status TEXT,
                        error_message TEXT,
                        log_path TEXT,

                        total_ms INTEGER DEFAULT 0,

                        coordinator_calls INTEGER DEFAULT 0,
                        coordinator_total_ms INTEGER DEFAULT 0,
                        research_calls INTEGER DEFAULT 0,
                        research_total_ms INTEGER DEFAULT 0,
                        analysis_calls INTEGER DEFAULT 0,
                        analysis_total_ms INTEGER DEFAULT 0,
                        writing_calls INTEGER DEFAULT 0,
                        writing_total_ms INTEGER DEFAULT 0,
                        quality_calls INTEGER DEFAULT 0,
                        quality_total_ms INTEGER DEFAULT 0,

                        llm_calls INTEGER DEFAULT 0,
                        llm_input_tokens INTEGER DEFAULT 0,
                        llm_output_tokens INTEGER DEFAULT 0,
                        llm_total_tokens INTEGER DEFAULT 0,
                        llm_truncated_calls INTEGER DEFAULT 0,
                        llm_error_calls INTEGER DEFAULT 0,

                        tavily_calls INTEGER DEFAULT 0,
                        tavily_results INTEGER DEFAULT 0,

                        raw_sources_count INTEGER DEFAULT 0,
                        deduplicated_sources_count INTEGER DEFAULT 0,
                        clinical_claims_count INTEGER DEFAULT 0,
                        citations_count INTEGER DEFAULT 0,
                        report_word_count INTEGER DEFAULT 0,

                        quality_iteration_final INTEGER DEFAULT 0,
                        quality_verdict_final TEXT,
                        quality_score_final REAL
                    );
                    """
                )
                con.execute(
                    """
                    CREATE TABLE IF NOT EXISTS agent_steps (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        step_index INTEGER NOT NULL,
                        agent_name TEXT NOT NULL,
                        quality_iteration INTEGER,
                        started_at TEXT,
                        ended_at TEXT,
                        duration_ms INTEGER,
                        status TEXT,
                        error_message TEXT,
                        raw_sources_count INTEGER,
                        deduplicated_sources_count INTEGER,
                        clinical_claims_count INTEGER,
                        citations_count INTEGER,
                        report_word_count INTEGER,
                        FOREIGN KEY(run_id) REFERENCES runs(run_id)
                    );
                    """
                )
                con.execute(
                    """
                    CREATE TABLE IF NOT EXISTS llm_calls (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        step_id INTEGER,
                        agent_name TEXT NOT NULL,
                        provider TEXT,
                        model TEXT,
                        api TEXT,
                        call_kind TEXT,
                        started_at TEXT,
                        ended_at TEXT,
                        duration_ms INTEGER,
                        status TEXT,
                        error_type TEXT,
                        error_message TEXT,
                        input_tokens INTEGER,
                        output_tokens INTEGER,
                        total_tokens INTEGER,
                        finish_reason TEXT,
                        truncated INTEGER,
                        FOREIGN KEY(run_id) REFERENCES runs(run_id),
                        FOREIGN KEY(step_id) REFERENCES agent_steps(id)
                    );
                    """
                )
                con.execute("CREATE INDEX IF NOT EXISTS idx_agent_steps_run ON agent_steps(run_id, step_index);")
                con.execute("CREATE INDEX IF NOT EXISTS idx_llm_calls_run ON llm_calls(run_id);")
                con.execute("CREATE INDEX IF NOT EXISTS idx_llm_calls_step ON llm_calls(step_id);")
            con.close()
            _initialized = True
        except Exception:
            # Metrics should never break execution.
            _initialized = True


def ensure_run(
    *,
    run_id: str,
    source: str = "unknown",
    topic: Optional[str] = None,
    scope_instructions: Optional[str] = None,
    target_audience: Optional[str] = None,
    report_format: Optional[str] = None,
    status: str = "pending",
    log_path: Optional[str] = None,
) -> None:
    if not _enabled() or not run_id:
        return
    init_db()
    try:
        con = _connect()
        with con:
            con.execute(
                """
                INSERT OR IGNORE INTO runs (
                    run_id, created_at, source, topic, scope_instructions, target_audience, report_format, status, log_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    run_id,
                    _now_iso(),
                    _truncate(source, 100),
                    _truncate(topic, 2000),
                    _truncate(scope_instructions, 2000),
                    _truncate(target_audience, 500),
                    _truncate(report_format, 200),
                    _truncate(status, 50),
                    _truncate(log_path, 2000),
                ),
            )
        con.close()
    except Exception:
        return


def set_run_status(run_id: str, status: str, *, error_message: Optional[str] = None) -> None:
    if not _enabled() or not run_id:
        return
    init_db()
    try:
        con = _connect()
        with con:
            con.execute(
                """
                UPDATE runs
                SET status = ?, error_message = COALESCE(?, error_message)
                WHERE run_id = ?;
                """,
                (_truncate(status, 50), _truncate(error_message, 2000), run_id),
            )
        con.close()
    except Exception:
        return


def finalize_run_from_state(state: Dict[str, Any], *, status: str) -> None:
    run_id = (state or {}).get("run_id")
    if not _enabled() or not run_id:
        return
    init_db()
    try:
        con = _connect()
        with con:
            con.execute(
                """
                UPDATE runs
                SET
                    ended_at = ?,
                    status = ?,
                    error_message = ?,
                    raw_sources_count = ?,
                    deduplicated_sources_count = ?,
                    clinical_claims_count = ?,
                    citations_count = ?,
                    report_word_count = ?,
                    quality_iteration_final = ?,
                    quality_verdict_final = ?,
                    quality_score_final = ?
                WHERE run_id = ?;
                """,
                (
                    _now_iso(),
                    _truncate(status, 50),
                    _truncate(state.get("error_message"), 2000),
                    len(state.get("raw_sources", []) or []),
                    len(state.get("deduplicated_sources", []) or []),
                    len(state.get("clinical_claims", []) or []),
                    len(state.get("citations", []) or []),
                    int(state.get("report_word_count", 0) or 0),
                    int(state.get("quality_iteration", 0) or 0),
                    _truncate(state.get("quality_verdict"), 50),
                    float(state.get("quality_score", 0.0) or 0.0),
                    run_id,
                ),
            )
        con.close()
        # Best-effort artifact persistence for Monitor drill-down.
        _write_run_artifacts_from_state(state or {})
    except Exception:
        return


def step_start(*, run_id: str, agent_name: str, quality_iteration: Optional[int]) -> Tuple[Optional[int], Optional[int]]:
    if not _enabled() or not run_id:
        return (None, None)
    init_db()
    try:
        con = _connect()
        with con:
            cur = con.execute(
                "SELECT COALESCE(MAX(step_index), 0) + 1 AS next_idx FROM agent_steps WHERE run_id = ?;",
                (run_id,),
            )
            row = cur.fetchone()
            step_index = int(row["next_idx"]) if row else 1
            cur2 = con.execute(
                """
                INSERT INTO agent_steps (
                    run_id, step_index, agent_name, quality_iteration, started_at, status
                ) VALUES (?, ?, ?, ?, ?, ?);
                """,
                (
                    run_id,
                    step_index,
                    _truncate(agent_name, 50),
                    int(quality_iteration) if quality_iteration is not None else None,
                    _now_iso(),
                    "running",
                ),
            )
            step_id = int(cur2.lastrowid)
        con.close()
        return (step_id, step_index)
    except Exception:
        return (None, None)


def step_end(
    *,
    step_id: Optional[int],
    run_id: str,
    agent_name: str,
    duration_ms: int,
    status: str,
    error_message: Optional[str],
    snapshot: Dict[str, int],
) -> None:
    if not _enabled() or not run_id or not step_id:
        return
    init_db()
    try:
        con = _connect()
        with con:
            con.execute(
                """
                UPDATE agent_steps
                SET
                    ended_at = ?,
                    duration_ms = ?,
                    status = ?,
                    error_message = ?,
                    raw_sources_count = ?,
                    deduplicated_sources_count = ?,
                    clinical_claims_count = ?,
                    citations_count = ?,
                    report_word_count = ?
                WHERE id = ?;
                """,
                (
                    _now_iso(),
                    int(duration_ms),
                    _truncate(status, 20),
                    _truncate(error_message, 2000),
                    int(snapshot.get("raw_sources_count", 0)),
                    int(snapshot.get("deduplicated_sources_count", 0)),
                    int(snapshot.get("clinical_claims_count", 0)),
                    int(snapshot.get("citations_count", 0)),
                    int(snapshot.get("report_word_count", 0)),
                    int(step_id),
                ),
            )

            # Update run aggregates.
            agent = (agent_name or "").strip().lower()
            if agent in ("coordinator", "research", "analysis", "writing", "quality"):
                con.execute(
                    f"""
                    UPDATE runs
                    SET
                        total_ms = total_ms + ?,
                        {agent}_calls = {agent}_calls + 1,
                        {agent}_total_ms = {agent}_total_ms + ?
                    WHERE run_id = ?;
                    """,
                    (int(duration_ms), int(duration_ms), run_id),
                )
            else:
                con.execute(
                    "UPDATE runs SET total_ms = total_ms + ? WHERE run_id = ?;",
                    (int(duration_ms), run_id),
                )
        con.close()
    except Exception:
        return


def llm_call_insert(
    *,
    run_id: str,
    step_id: Optional[int],
    agent_name: str,
    provider: str,
    model: str,
    api: Optional[str],
    call_kind: str,
    duration_ms: int,
    status: str,
    error_type: Optional[str],
    error_message: Optional[str],
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    total_tokens: Optional[int],
    finish_reason: Optional[str],
    truncated: Optional[bool],
) -> None:
    if not _enabled() or not run_id:
        return
    init_db()
    try:
        con = _connect()
        with con:
            con.execute(
                """
                INSERT INTO llm_calls (
                    run_id, step_id, agent_name, provider, model, api, call_kind,
                    started_at, ended_at, duration_ms,
                    status, error_type, error_message,
                    input_tokens, output_tokens, total_tokens,
                    finish_reason, truncated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    run_id,
                    int(step_id) if step_id is not None else None,
                    _truncate(agent_name, 50),
                    _truncate(provider, 50),
                    _truncate(model, 200),
                    _truncate(api, 200),
                    _truncate(call_kind, 20),
                    None,
                    _now_iso(),
                    int(duration_ms),
                    _truncate(status, 20),
                    _truncate(error_type, 200),
                    _truncate(error_message, 2000),
                    int(input_tokens) if isinstance(input_tokens, int) else None,
                    int(output_tokens) if isinstance(output_tokens, int) else None,
                    int(total_tokens) if isinstance(total_tokens, int) else None,
                    _truncate(finish_reason, 50),
                    1 if truncated else 0,
                ),
            )

            # Update run aggregates (best-effort).
            in_t = int(input_tokens) if isinstance(input_tokens, int) else 0
            out_t = int(output_tokens) if isinstance(output_tokens, int) else 0
            tot_t = int(total_tokens) if isinstance(total_tokens, int) else (in_t + out_t)
            trunc_inc = 1 if truncated else 0
            err_inc = 1 if (status or "").lower() == "error" else 0
            con.execute(
                """
                UPDATE runs
                SET
                    llm_calls = llm_calls + 1,
                    llm_input_tokens = llm_input_tokens + ?,
                    llm_output_tokens = llm_output_tokens + ?,
                    llm_total_tokens = llm_total_tokens + ?,
                    llm_truncated_calls = llm_truncated_calls + ?,
                    llm_error_calls = llm_error_calls + ?
                WHERE run_id = ?;
                """,
                (in_t, out_t, tot_t, trunc_inc, err_inc, run_id),
            )
        con.close()
    except Exception:
        return


def tavily_record(*, run_id: str, results_count: int) -> None:
    if not _enabled() or not run_id:
        return
    init_db()
    try:
        con = _connect()
        with con:
            con.execute(
                """
                UPDATE runs
                SET tavily_calls = tavily_calls + 1,
                    tavily_results = tavily_results + ?
                WHERE run_id = ?;
                """,
                (int(results_count or 0), run_id),
            )
        con.close()
    except Exception:
        return
