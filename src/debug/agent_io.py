"""
Debug logging for agent inputs/outputs.

Writes JSONL events to logs/agent_io.jsonl for later debugging.
This is intentionally lightweight and truncates large payloads.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


_lock = threading.Lock()


def _enabled() -> bool:
    return os.getenv("AGENT_IO_LOG_ENABLED", "1").strip().lower() not in ("0", "false", "no")

def _mode() -> str:
    # global: logs/agent_io.jsonl only
    # per_run: logs/runs/<run_id>.jsonl only
    # both: write to both sinks
    return os.getenv("AGENT_IO_LOG_MODE", "both").strip().lower() or "both"


def _log_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    default_path = repo_root / "logs" / "agent_io.jsonl"
    raw = os.getenv("AGENT_IO_LOG_PATH", "").strip()
    if not raw:
        return default_path
    return Path(raw)

def _run_log_path(run_id: str) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    runs_dir = repo_root / "logs" / "runs"
    return runs_dir / f"{run_id}.jsonl"


def _truncate_str(s: str, max_len: int = 4000) -> str:
    if s is None:
        return ""
    if len(s) <= max_len:
        return s
    return s[: max_len - 20] + f"...(truncated,len={len(s)})"


def _sanitize(obj: Any, *, max_str: int = 4000, max_list: int = 50, _depth: int = 0) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (bool, int, float)):
        return obj
    if isinstance(obj, str):
        return _truncate_str(obj, max_len=max_str)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        if _depth > 6:
            return {"_truncated": True, "type": "dict"}
        return {str(k): _sanitize(v, max_str=max_str, max_list=max_list, _depth=_depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        if _depth > 6:
            return [{"_truncated": True, "type": "list"}]
        items = list(obj)
        if len(items) > max_list:
            head = items[:max_list]
            return _sanitize(head, max_str=max_str, max_list=max_list, _depth=_depth + 1) + [
                {"_truncated": True, "original_len": len(items)}
            ]
        return [_sanitize(v, max_str=max_str, max_list=max_list, _depth=_depth + 1) for v in items]
    return _truncate_str(repr(obj), max_len=max_str)


def summarize_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produce a compact summary of PipelineState for logs.
    """
    def _count(key: str) -> int:
        v = state.get(key)
        return len(v) if isinstance(v, list) else 0

    report = state.get("report_markdown", "") or ""
    types = {
        "research_queries": type(state.get("research_queries")).__name__,
        "raw_sources": type(state.get("raw_sources")).__name__,
        "deduplicated_sources": type(state.get("deduplicated_sources")).__name__,
        "clinical_claims": type(state.get("clinical_claims")).__name__,
        "evidence_gaps": type(state.get("evidence_gaps")).__name__,
        "quality_issues": type(state.get("quality_issues")).__name__,
        "report_sections": type(state.get("report_sections")).__name__,
    }
    return _sanitize(
        {
            "run_id": state.get("run_id"),
            "topic": state.get("topic"),
            "current_agent": state.get("current_agent"),
            "pipeline_status": state.get("pipeline_status"),
            "quality_iteration": state.get("quality_iteration"),
            "quality_verdict": state.get("quality_verdict"),
            "quality_score": state.get("quality_score"),
            "error_message": state.get("error_message"),
            "types": types,
            "counts": {
                "research_queries": _count("research_queries"),
                "raw_sources": _count("raw_sources"),
                "deduplicated_sources": _count("deduplicated_sources"),
                "clinical_claims": _count("clinical_claims"),
                "evidence_gaps": _count("evidence_gaps"),
                "quality_issues": _count("quality_issues"),
            },
            "report_word_count": state.get("report_word_count"),
            "report_preview": report[:300],
        }
    )


def log_event(event: Dict[str, Any]) -> None:
    if not _enabled():
        return

    payload = dict(event)
    payload.setdefault("ts", datetime.now().isoformat())

    mode = _mode()
    paths = []
    if mode in ("global", "both"):
        paths.append(_log_path())
    if mode in ("per_run", "both"):
        rid = payload.get("run_id")
        if isinstance(rid, str) and rid.strip():
            paths.append(_run_log_path(rid.strip()))

    if not paths:
        return

    line = json.dumps(_sanitize(payload), ensure_ascii=True)
    with _lock:
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


def log_agent_state(
    *,
    when: str,
    agent_name: str,
    state: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    log_event(
        {
            "type": "agent_state",
            "when": when,
            "agent": agent_name,
            "run_id": state.get("run_id"),
            "state": summarize_state(state),
            "extra": extra or {},
        }
    )


def log_llm_io(
    *,
    agent_name: str,
    run_id: Optional[str],
    provider: str,
    model: str,
    system_prompt: str,
    messages: Any,
    meta: Optional[Dict[str, Any]] = None,
    response: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    log_event(
        {
            "type": "llm_io",
            "agent": agent_name,
            "run_id": run_id,
            "provider": provider,
            "model": model,
            "system_prompt": system_prompt,
            "messages": messages,
            "meta": meta or {},
            "response": response,
            "error": error,
        }
    )
