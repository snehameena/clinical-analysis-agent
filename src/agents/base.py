"""
Base agent class defining the interface for all agents.
"""

from abc import ABC, abstractmethod
import json
import time
from typing import Optional
from datetime import datetime
from src.state.schema import PipelineState, AgentHistoryEntry
from src.llm.agent_config import get_agent_llm_config
from src.llm.providers import get_llm_provider
from src.debug.agent_io import log_llm_io
from src.debug.run_db import llm_call_insert, ensure_run


class BaseAgent(ABC):
    """Abstract base class for all agents in the pipeline"""

    def __init__(
        self,
        name: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        provider: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        """
        Initialize the agent.

        Args:
            name: Agent name (for logging)
            model: Model name override (defaults to config/agents.yaml)
            temperature: Temperature override (defaults to config/agents.yaml)
            max_tokens: Max tokens override (defaults to config/agents.yaml)
            provider: Provider override (defaults to config/agents.yaml)
            timeout: Timeout override in seconds (defaults to config/agents.yaml)
        """
        self.name = name

        cfg = get_agent_llm_config(name)
        self.provider_name = provider or cfg.provider
        self.model = model or cfg.model
        self.temperature = float(temperature) if temperature is not None else cfg.temperature
        self.max_tokens = int(max_tokens) if max_tokens is not None else cfg.max_tokens
        self.timeout = float(timeout) if timeout is not None else cfg.timeout

        self.llm = get_llm_provider(self.provider_name)
        self.system_prompt = self._get_system_prompt()

        # Captured from the last provider response for better error messages (e.g. truncation).
        self._last_llm_meta: dict = {}

    def _get_system_prompt(self) -> str:
        """Get system prompt from config or override"""
        # This will be overridden in subclasses
        return ""

    @abstractmethod
    async def execute(self, state: PipelineState) -> PipelineState:
        """
        Execute the agent on the given state.

        Args:
            state: Current pipeline state

        Returns:
            Updated pipeline state
        """
        pass

    async def _call_llm(self, user_message: str) -> str:
        """
        Call the configured LLM provider with a single user message.

        Args:
            user_message: User message to send to LLM

        Returns:
            LLM response text

        Raises:
            ValueError: If LLM returns an empty response
        """
        messages = [{"role": "user", "content": user_message}]
        run_id = None
        # Best-effort: if caller already put a run_id in the agent instance, log it.
        # (Not required for correctness; pipeline state logging also includes run_id.)
        run_id = getattr(self, "_current_run_id", None)
        step_id = getattr(self, "_current_step_id", None)
        call_kind = getattr(self, "_current_call_kind", "primary")

        if isinstance(run_id, str) and run_id.strip():
            ensure_run(run_id=run_id.strip(), source="unknown")

        log_llm_io(
            agent_name=self.name,
            run_id=run_id,
            provider=self.provider_name,
            model=self.model,
            system_prompt=self.system_prompt,
            messages=messages,
            meta={"temperature": self.temperature, "max_tokens": self.max_tokens, "timeout": self.timeout},
        )

        t0 = time.time()
        try:
            result = await self.llm.generate(
                system_prompt=self.system_prompt,
                messages=messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
            )
            if isinstance(result, str):
                resp_text = result
                resp_meta = {}
            else:
                resp_text = getattr(result, "text", "")
                resp_meta = getattr(result, "meta", {}) or {}
        except Exception as e:
            self._last_llm_meta = {}
            llm_call_insert(
                run_id=run_id,
                step_id=step_id,
                agent_name=self.name,
                provider=self.provider_name,
                model=self.model,
                api=None,
                call_kind=call_kind,
                duration_ms=int((time.time() - t0) * 1000),
                status="error",
                error_type=type(e).__name__,
                error_message=str(e),
                input_tokens=None,
                output_tokens=None,
                total_tokens=None,
                finish_reason=None,
                truncated=None,
            )
            log_llm_io(
                agent_name=self.name,
                run_id=run_id,
                provider=self.provider_name,
                model=self.model,
                system_prompt=self.system_prompt,
                messages=messages,
                meta={"temperature": self.temperature, "max_tokens": self.max_tokens, "timeout": self.timeout},
                error=str(e),
            )
            raise

        # Store meta for downstream JSON parsing and error messages.
        # Include request-side limits so we can reason about truncation.
        self._last_llm_meta = {**resp_meta, **{"max_tokens": self.max_tokens}}

        # Normalize token usage across provider shapes.
        api_used = resp_meta.get("api") if isinstance(resp_meta, dict) else None
        finish_reason = resp_meta.get("finish_reason") if isinstance(resp_meta, dict) else None
        truncated = resp_meta.get("truncated") if isinstance(resp_meta, dict) else None
        usage = resp_meta.get("usage") if isinstance(resp_meta, dict) else None
        in_t = out_t = tot_t = None
        if isinstance(usage, dict):
            # OpenAI chat.completions
            if isinstance(usage.get("prompt_tokens"), int) or isinstance(usage.get("completion_tokens"), int):
                in_t = usage.get("prompt_tokens")
                out_t = usage.get("completion_tokens")
                tot_t = usage.get("total_tokens")
            # OpenAI responses / Anthropic
            elif isinstance(usage.get("input_tokens"), int) or isinstance(usage.get("output_tokens"), int):
                in_t = usage.get("input_tokens")
                out_t = usage.get("output_tokens")
                tot_t = usage.get("total_tokens")
        if tot_t is None and isinstance(in_t, int) and isinstance(out_t, int):
            tot_t = in_t + out_t

        llm_call_insert(
            run_id=run_id,
            step_id=step_id,
            agent_name=self.name,
            provider=self.provider_name,
            model=self.model,
            api=api_used,
            call_kind=call_kind,
            duration_ms=int((time.time() - t0) * 1000),
            status="ok",
            error_type=None,
            error_message=None,
            input_tokens=in_t,
            output_tokens=out_t,
            total_tokens=tot_t,
            finish_reason=finish_reason,
            truncated=bool(truncated) if truncated is not None else None,
        )

        log_llm_io(
            agent_name=self.name,
            run_id=run_id,
            provider=self.provider_name,
            model=self.model,
            system_prompt=self.system_prompt,
            messages=messages,
            meta={**{"temperature": self.temperature, "max_tokens": self.max_tokens, "timeout": self.timeout}, **resp_meta},
            response=resp_text,
        )
        return resp_text

    async def _call_llm_messages(self, messages: list[dict]) -> str:
        """
        Call the configured LLM provider with a full message history.

        messages: [{"role": "user"|"assistant", "content": "..."}]
        """
        run_id = getattr(self, "_current_run_id", None)
        step_id = getattr(self, "_current_step_id", None)
        call_kind = getattr(self, "_current_call_kind", "primary")

        if isinstance(run_id, str) and run_id.strip():
            ensure_run(run_id=run_id.strip(), source="unknown")
        log_llm_io(
            agent_name=self.name,
            run_id=run_id,
            provider=self.provider_name,
            model=self.model,
            system_prompt=self.system_prompt,
            messages=messages,
            meta={"temperature": self.temperature, "max_tokens": self.max_tokens, "timeout": self.timeout},
        )
        try:
            t0 = time.time()
            result = await self.llm.generate(
                system_prompt=self.system_prompt,
                messages=messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
            )
            if isinstance(result, str):
                resp_text = result
                resp_meta = {}
            else:
                resp_text = getattr(result, "text", "")
                resp_meta = getattr(result, "meta", {}) or {}
        except Exception as e:
            llm_call_insert(
                run_id=run_id,
                step_id=step_id,
                agent_name=self.name,
                provider=self.provider_name,
                model=self.model,
                api=None,
                call_kind=call_kind,
                duration_ms=int((time.time() - t0) * 1000) if "t0" in locals() else 0,
                status="error",
                error_type=type(e).__name__,
                error_message=str(e),
                input_tokens=None,
                output_tokens=None,
                total_tokens=None,
                finish_reason=None,
                truncated=None,
            )
            log_llm_io(
                agent_name=self.name,
                run_id=run_id,
                provider=self.provider_name,
                model=self.model,
                system_prompt=self.system_prompt,
                messages=messages,
                meta={"temperature": self.temperature, "max_tokens": self.max_tokens, "timeout": self.timeout},
                error=str(e),
            )
            raise

        # Store meta for downstream error messages.
        self._last_llm_meta = {**(resp_meta or {}), **{"max_tokens": self.max_tokens}}

        api_used = resp_meta.get("api") if isinstance(resp_meta, dict) else None
        finish_reason = resp_meta.get("finish_reason") if isinstance(resp_meta, dict) else None
        truncated = resp_meta.get("truncated") if isinstance(resp_meta, dict) else None
        usage = resp_meta.get("usage") if isinstance(resp_meta, dict) else None
        in_t = out_t = tot_t = None
        if isinstance(usage, dict):
            if isinstance(usage.get("prompt_tokens"), int) or isinstance(usage.get("completion_tokens"), int):
                in_t = usage.get("prompt_tokens")
                out_t = usage.get("completion_tokens")
                tot_t = usage.get("total_tokens")
            elif isinstance(usage.get("input_tokens"), int) or isinstance(usage.get("output_tokens"), int):
                in_t = usage.get("input_tokens")
                out_t = usage.get("output_tokens")
                tot_t = usage.get("total_tokens")
        if tot_t is None and isinstance(in_t, int) and isinstance(out_t, int):
            tot_t = in_t + out_t

        llm_call_insert(
            run_id=run_id,
            step_id=step_id,
            agent_name=self.name,
            provider=self.provider_name,
            model=self.model,
            api=api_used,
            call_kind=call_kind,
            duration_ms=int((time.time() - t0) * 1000) if "t0" in locals() else 0,
            status="ok",
            error_type=None,
            error_message=None,
            input_tokens=in_t,
            output_tokens=out_t,
            total_tokens=tot_t,
            finish_reason=finish_reason,
            truncated=bool(truncated) if truncated is not None else None,
        )

        log_llm_io(
            agent_name=self.name,
            run_id=run_id,
            provider=self.provider_name,
            model=self.model,
            system_prompt=self.system_prompt,
            messages=messages,
            meta={**{"temperature": self.temperature, "max_tokens": self.max_tokens, "timeout": self.timeout}, **resp_meta},
            response=resp_text,
        )
        return resp_text

    async def _call_llm_json(
        self,
        user_message: str,
        *,
        required_keys: Optional[list[str]] = None,
        schema_hint: str = "",
    ) -> dict:
        """
        Call the LLM and return a parsed JSON object (dict). If parsing fails,
        attempt a single "repair" call to convert the raw output into valid JSON.
        """

        def _likely_truncated(meta: dict) -> bool:
            """
            Best-effort detection of response truncation (token cap reached).
            Works for OpenAI Chat Completions and Responses usage shapes.
            """
            try:
                if (meta or {}).get("truncated") is True:
                    return True
                if str((meta or {}).get("finish_reason") or "").strip().lower() in ("length", "max_tokens"):
                    return True
                usage = (meta or {}).get("usage") or {}
                # chat.completions usage
                completion_tokens = usage.get("completion_tokens")
                # responses usage
                output_tokens = usage.get("output_tokens")
                limit = int((meta or {}).get("max_tokens") or self.max_tokens or 0)
                if limit > 0 and isinstance(completion_tokens, int) and completion_tokens >= limit:
                    return True
                if limit > 0 and isinstance(output_tokens, int) and output_tokens >= limit:
                    return True
            except Exception:
                return False
            return False

        def _validate_obj(obj: object) -> dict:
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object, got {type(obj).__name__}")
            if required_keys:
                missing = [k for k in required_keys if k not in obj]
                if missing:
                    raise ValueError(f"Missing required keys: {missing}")
            return obj

        self._current_call_kind = "primary"
        raw = await self._call_llm(user_message)
        try:
            parsed = self._parse_json_response(raw)
            return _validate_obj(parsed)
        except Exception as e:
            meta0 = getattr(self, "_last_llm_meta", {}) or {}
            if _likely_truncated(meta0):
                raise ValueError(
                    f"LLM output appears truncated (agent={self.name}, model={self.model}, max_tokens={self.max_tokens}). "
                    f"Increase this agent's max_tokens in config/agents.yaml. Parse error: {e}"
                )

            hint = f"\nSchema hint:\n{schema_hint}\n" if schema_hint else ""
            keys = f"{required_keys}" if required_keys else "N/A"
            repair_prompt = (
                "Your previous output was not valid JSON.\n"
                "Convert the following text into a single valid JSON object.\n"
                "Return ONLY JSON. No markdown. No commentary.\n"
                "If the output would be long, you MUST keep it short by:\n"
                "- limiting list lengths (e.g. <= 5 items)\n"
                "- using empty lists/empty strings where needed\n"
                "The JSON MUST be syntactically valid and complete (all braces closed).\n"
                f"Required keys: {keys}\n"
                f"{hint}\n"
                "Text to convert:\n"
                "-----\n"
                f"{raw}\n"
                "-----\n"
            )
            self._current_call_kind = "repair1"
            repaired = await self._call_llm(repair_prompt)
            try:
                parsed2 = self._parse_json_response(repaired)
                return _validate_obj(parsed2)
            except Exception:
                meta1 = getattr(self, "_last_llm_meta", {}) or {}
                if _likely_truncated(meta1):
                    raise ValueError(
                        f"LLM repair output appears truncated (agent={self.name}, model={self.model}, max_tokens={self.max_tokens}). "
                        f"Increase this agent's max_tokens in config/agents.yaml."
                    )

                # Last-resort: ask for a minimal skeleton object with required keys only.
                if required_keys:
                    skeleton_lines = []
                    for k in required_keys:
                        # Heuristic defaults: try to match common agent output shapes.
                        if any(tok in k for tok in ("claims", "gaps", "contradictions", "findings", "issues", "queries", "citations", "subtopics", "sources")):
                            skeleton_lines.append(f'  "{k}": []')
                        elif any(tok in k for tok in ("sections", "boundaries", "scores")) or k.endswith("_by_evidence_level"):
                            skeleton_lines.append(f'  "{k}": {{}}')
                        elif "score" in k:
                            skeleton_lines.append(f'  "{k}": 0')
                        else:
                            skeleton_lines.append(f'  "{k}": ""')
                    skeleton = "{\n" + ",\n".join(skeleton_lines) + "\n}"
                else:
                    skeleton = "{}"

                repair2_prompt = (
                    "Return ONLY a minimal valid JSON object with the required keys.\n"
                    "Do not include any extra keys, markdown, or commentary.\n"
                    f"JSON to return exactly:\n{skeleton}\n"
                )
                self._current_call_kind = "repair2"
                repaired2 = await self._call_llm(repair2_prompt)
                parsed3 = self._parse_json_response(repaired2)
                return _validate_obj(parsed3)

    def _parse_json_response(self, response: str) -> dict:
        """
        Parse JSON from LLM response.

        Tries to extract JSON from response, handling cases where
        LLM wraps JSON in markdown code blocks.

        Args:
            response: LLM response text

        Returns:
            Parsed JSON dict

        Raises:
            ValueError: If JSON cannot be parsed
        """

        def _extract_first_balanced_json_object(text: str) -> Optional[str]:
            """
            Extract the first balanced JSON object substring from text.
            Returns None if no balanced object is found.
            """
            start = text.find("{")
            if start < 0:
                return None
            depth = 0
            in_str = False
            escape = False
            for i in range(start, len(text)):
                ch = text[i]
                if in_str:
                    if escape:
                        escape = False
                        continue
                    if ch == "\\":
                        escape = True
                        continue
                    if ch == "\"":
                        in_str = False
                    continue
                else:
                    if ch == "\"":
                        in_str = True
                        continue
                    if ch == "{":
                        depth += 1
                        continue
                    if ch == "}":
                        depth -= 1
                        if depth == 0:
                            return text[start : i + 1]
            return None
        # Try direct JSON parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        except TypeError:
            pass

        # Try YAML (handles some "almost JSON" variants better)
        try:
            import yaml
            loaded = yaml.safe_load(response)
            if isinstance(loaded, (dict, list)):
                return loaded
        except Exception:
            pass

        # Try to extract from markdown code blocks
        import re
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response, re.DOTALL)
        if json_match:
            block = json_match.group(1)
            try:
                return json.loads(block)
            except Exception:
                try:
                    import yaml
                    loaded = yaml.safe_load(block)
                    if isinstance(loaded, (dict, list)):
                        return loaded
                except Exception:
                    pass

        # Try to extract the first balanced {...} object from the response.
        extracted = _extract_first_balanced_json_object(response)
        if extracted is not None:
            try:
                return json.loads(extracted)
            except Exception:
                try:
                    import yaml
                    loaded = yaml.safe_load(extracted)
                    if isinstance(loaded, (dict, list)):
                        return loaded
                except Exception:
                    pass

        raise ValueError(f"Could not parse JSON from response: {response[:200]}")

    def _record_execution(
        self, state: PipelineState, token_count: Optional[int] = None
    ) -> PipelineState:
        """
        Record agent execution in pipeline history.

        Args:
            state: Pipeline state
            token_count: Optional token count for this execution

        Returns:
            Updated state with execution history
        """
        entry = AgentHistoryEntry(
            agent_name=self.name,
            start_time=datetime.now(),
            end_time=datetime.now(),
            token_count=token_count,
            status="completed",
        )

        if "agent_history" not in state:
            state["agent_history"] = []

        state["agent_history"].append(entry)
        state["current_agent"] = self.name

        return state
