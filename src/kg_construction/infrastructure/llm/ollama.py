"""Ollama LLM provider with structured JSON retry/repair."""

from __future__ import annotations

import json
import re
import urllib.request
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from kg_construction.infrastructure.llm.prompts import build_repair_prompt

JsonDict = dict[str, Any]
Transport = Callable[[JsonDict], str | JsonDict]

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", flags=re.IGNORECASE | re.MULTILINE)


class OllamaOutputError(RuntimeError):
    """Raised when Ollama does not return valid structured output."""


class OllamaConfig(BaseModel):
    """Ollama model and output configuration."""

    model_config = ConfigDict(extra="forbid")

    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "qwen3:1.7b-q4_K_M"
    temperature: float = 0.0
    structured_output: bool = True
    evidence_required: bool = True
    retry_on_invalid_output: bool = True
    max_retries: int = Field(default=3, ge=0)
    prompt_version: str = "ollama_graph_extraction_v1"


class OllamaProvider:
    """Local Ollama provider for graph extraction and schema proposal prompts."""

    name = "ollama_provider"

    def __init__(self, config: OllamaConfig | None = None, *, transport: Transport | None = None) -> None:
        self.config = config or OllamaConfig()
        self.transport = transport

    def complete(self, prompt: str) -> str:
        """Call Ollama chat and return the assistant text."""

        body: JsonDict = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": self.config.temperature},
        }
        if self.config.structured_output:
            body["format"] = "json"

        if self.transport is not None:
            result = self.transport(body)
            return self._content_from_response(result)

        request = urllib.request.Request(
            f"{self.config.base_url.rstrip('/')}/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return self._content_from_response(payload)

    def complete_json(
        self,
        prompt: str,
        *,
        validator: Callable[[Any], bool] | None = None,
    ) -> JsonDict:
        """Return validated JSON, retrying with a repair prompt when needed."""

        attempts = self.config.max_retries + 1 if self.config.retry_on_invalid_output else 1
        current_prompt = prompt
        invalid_output = ""
        last_error = "unknown error"
        for _attempt in range(attempts):
            invalid_output = self.complete(current_prompt)
            try:
                payload = json.loads(self._extract_json(invalid_output))
            except json.JSONDecodeError as exc:
                last_error = str(exc)
            else:
                if isinstance(payload, dict) and (validator is None or validator(payload)):
                    return payload
                last_error = "JSON payload failed schema validation."
            current_prompt = build_repair_prompt(
                original_prompt=prompt,
                invalid_output=invalid_output,
                error=last_error,
            )
        raise OllamaOutputError(
            f"Ollama did not return valid JSON after {attempts} attempt(s): {last_error}. "
            f"Last output: {invalid_output[:500]}"
        )

    @staticmethod
    def _extract_json(text: str) -> str:
        stripped = _JSON_FENCE_RE.sub("", text.strip()).strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            return stripped
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            return stripped[start : end + 1]
        return stripped

    @staticmethod
    def _content_from_response(result: str | JsonDict) -> str:
        if isinstance(result, str):
            return result
        message = result.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"]
        response = result.get("response")
        if isinstance(response, str):
            return response
        return json.dumps(result)


__all__ = ["OllamaConfig", "OllamaOutputError", "OllamaProvider"]
