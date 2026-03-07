"""
Agent LLM configuration loader (config-only).

Source of truth: config/agents.yaml
Env vars are used only for API keys and runtime URLs, not provider/model selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Optional

import yaml
from pathlib import Path


@dataclass(frozen=True)
class AgentLLMConfig:
    provider: str
    model: str
    temperature: float
    max_tokens: int
    timeout: Optional[float] = None


@lru_cache(maxsize=1)
def _load_agents_yaml() -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "config" / "agents.yaml"
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("config/agents.yaml must be a YAML mapping at the top level")
    return data


def get_agent_llm_config(agent_name: str) -> AgentLLMConfig:
    """
    Return the LLM config for a given agent.

    Raises ValueError if required fields are missing.
    """
    data = _load_agents_yaml()
    raw = data.get(agent_name)
    if not isinstance(raw, dict):
        raise ValueError(f"Missing agent config for '{agent_name}' in config/agents.yaml")

    provider = str(raw.get("provider", "")).strip().lower()
    model = str(raw.get("model", "")).strip()

    if not provider:
        raise ValueError(f"Agent '{agent_name}' is missing required field 'provider'")
    if not model:
        raise ValueError(f"Agent '{agent_name}' is missing required field 'model'")

    temperature = raw.get("temperature", 0.3)
    max_tokens = raw.get("max_tokens", 2000)
    timeout = raw.get("timeout")

    try:
        temperature_f = float(temperature)
    except Exception as e:
        raise ValueError(f"Agent '{agent_name}' has invalid temperature: {temperature!r}") from e

    try:
        max_tokens_i = int(max_tokens)
    except Exception as e:
        raise ValueError(f"Agent '{agent_name}' has invalid max_tokens: {max_tokens!r}") from e

    timeout_f: Optional[float]
    if timeout is None or timeout == "":
        timeout_f = None
    else:
        try:
            timeout_f = float(timeout)
        except Exception as e:
            raise ValueError(f"Agent '{agent_name}' has invalid timeout: {timeout!r}") from e

    return AgentLLMConfig(
        provider=provider,
        model=model,
        temperature=temperature_f,
        max_tokens=max_tokens_i,
        timeout=timeout_f,
    )
