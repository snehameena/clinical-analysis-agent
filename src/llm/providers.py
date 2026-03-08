"""
LLM provider implementations.

Provider selection is configured per agent in config/agents.yaml.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class LLMResult:
    text: str
    meta: Dict[str, Any]


class LLMProvider(Protocol):
    async def generate(
        self,
        *,
        system_prompt: str,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: Optional[float] = None,
    ) -> LLMResult:
        ...


def _require_env(key: str) -> str:
    val = os.getenv(key, "").strip()
    if not val:
        raise ValueError(f"{key} not set in environment")
    return val


@dataclass
class AnthropicProvider:
    """
    Anthropic Claude provider (sync client wrapped in asyncio.to_thread).

    Note: do not validate API key on init so unit tests can construct agents without keys.
    """

    def _get_client(self):
        from anthropic import Anthropic
        api_key = _require_env("ANTHROPIC_API_KEY")
        return Anthropic(api_key=api_key)

    async def generate(
        self,
        *,
        system_prompt: str,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: Optional[float] = None,
    ) -> LLMResult:
        client = self._get_client()

        async def _do_call():
            resp = await asyncio.to_thread(
                client.messages.create,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=messages,
            )
            if not getattr(resp, "content", None):
                raise ValueError("Empty response from Anthropic")
            text = resp.content[0].text

            usage = getattr(resp, "usage", None)
            meta: Dict[str, Any] = {"api": "anthropic.messages.create"}
            if usage is not None:
                meta["usage"] = {
                    "input_tokens": getattr(usage, "input_tokens", None),
                    "output_tokens": getattr(usage, "output_tokens", None),
                }
            return LLMResult(text=text, meta=meta)

        if timeout:
            try:
                return await asyncio.wait_for(_do_call(), timeout=timeout)
            except asyncio.TimeoutError as e:
                # asyncio.TimeoutError often stringifies to empty; include actionable context.
                raise asyncio.TimeoutError(
                    f"OpenAI request timed out after {timeout}s (model={model}). "
                    "Increase this agent timeout in config/agents.yaml or reduce max_tokens/context."
                ) from e
        return await _do_call()


@dataclass
class OpenAIProvider:
    """
    OpenAI provider (sync client wrapped in asyncio.to_thread).
    """

    def _get_client(self):
        from openai import OpenAI
        api_key = _require_env("OPENAI_API_KEY")
        return OpenAI(api_key=api_key)

    async def generate(
        self,
        *,
        system_prompt: str,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: Optional[float] = None,
    ) -> LLMResult:
        client = self._get_client()

        # OpenAI expects system prompt as a "system" role message.
        oa_messages: List[Dict[str, str]] = []
        if system_prompt:
            oa_messages.append({"role": "system", "content": system_prompt})
        oa_messages.extend(messages)

        def _extract_chat_text(resp: Any) -> str:
            try:
                msg = resp.choices[0].message
            except Exception as e:
                raise ValueError("Unexpected response shape from OpenAI") from e

            content = getattr(msg, "content", None)
            if isinstance(content, str) and content.strip():
                return content

            # Some responses may populate a refusal field instead of content.
            refusal = getattr(msg, "refusal", None)
            if isinstance(refusal, str) and refusal.strip():
                return refusal

            # Defensive: content as structured parts.
            if isinstance(content, list):
                parts: List[str] = []
                for part in content:
                    if isinstance(part, str):
                        parts.append(part)
                        continue
                    if isinstance(part, dict):
                        txt = part.get("text") or part.get("content") or ""
                        if isinstance(txt, str):
                            parts.append(txt)
                joined = "".join(parts).strip()
                if joined:
                    return joined

            return ""

        async def _do_call():
            # Newer OpenAI models (e.g. GPT-5.*) require max_completion_tokens.
            # Prefer that parameter, but retry if the server rejects it.
            try:
                resp = await asyncio.to_thread(
                    client.chat.completions.create,
                    model=model,
                    messages=oa_messages,
                    temperature=temperature,
                    max_completion_tokens=max_tokens,
                )
                api_used = "openai.chat.completions.create"
            except TypeError:
                # Older client versions may not accept max_completion_tokens.
                resp = await asyncio.to_thread(
                    client.chat.completions.create,
                    model=model,
                    messages=oa_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                api_used = "openai.chat.completions.create"
            except Exception as e:
                msg = str(e)
                # Server-side error (400) for unsupported parameter name.
                if "max_completion_tokens" in msg and "not supported" in msg:
                    resp = await asyncio.to_thread(
                        client.chat.completions.create,
                        model=model,
                        messages=oa_messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    api_used = "openai.chat.completions.create"
                elif "max_tokens" in msg and "not supported" in msg and "max_completion_tokens" in msg:
                    resp = await asyncio.to_thread(
                        client.chat.completions.create,
                        model=model,
                        messages=oa_messages,
                        temperature=temperature,
                        max_completion_tokens=max_tokens,
                    )
                    api_used = "openai.chat.completions.create"
                else:
                    raise

            text = _extract_chat_text(resp)
            if text:
                finish_reason = None
                try:
                    finish_reason = resp.choices[0].finish_reason
                except Exception:
                    pass
                usage = getattr(resp, "usage", None)
                meta: Dict[str, Any] = {"api": api_used, "finish_reason": finish_reason, "truncated": finish_reason == "length"}
                if usage is not None:
                    meta["usage"] = {
                        "prompt_tokens": getattr(usage, "prompt_tokens", None),
                        "completion_tokens": getattr(usage, "completion_tokens", None),
                        "total_tokens": getattr(usage, "total_tokens", None),
                    }
                return LLMResult(text=text, meta=meta)

            # If chat.completions returns no content (some models), fall back to Responses API.
            resp2 = await asyncio.to_thread(
                client.responses.create,
                model=model,
                instructions=system_prompt or None,
                input=messages,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            out_text = getattr(resp2, "output_text", None)
            if isinstance(out_text, str) and out_text.strip():
                usage = getattr(resp2, "usage", None)
                meta = {"api": "openai.responses.create"}
                if usage is not None:
                    meta["usage"] = {
                        "input_tokens": getattr(usage, "input_tokens", None),
                        "output_tokens": getattr(usage, "output_tokens", None),
                        "total_tokens": getattr(usage, "total_tokens", None),
                    }
                return LLMResult(text=out_text, meta=meta)

            raise ValueError("Empty response from OpenAI (no chat content and no responses output_text)")

        if timeout:
            return await asyncio.wait_for(_do_call(), timeout=timeout)
        return await _do_call()


@dataclass
class GeminiProvider:
    async def generate(
        self,
        *,
        system_prompt: str,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: Optional[float] = None,
    ) -> LLMResult:
        raise NotImplementedError(
            "Gemini provider is not implemented yet (project decision: skip Gemini for now)."
        )


@lru_cache(maxsize=None)
def get_llm_provider(provider: str) -> LLMProvider:
    key = (provider or "").strip().lower()
    if key in ("anthropic", "claude"):
        return AnthropicProvider()
    if key in ("openai", "gpt"):
        return OpenAIProvider()
    if key in ("gemini",):
        return GeminiProvider()
    raise ValueError(f"Unknown LLM provider: {provider!r}")
