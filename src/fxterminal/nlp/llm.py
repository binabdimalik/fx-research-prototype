"""LLM client: Anthropic implementation behind a Protocol.

Contract for every implementation: `complete_json(system, prompt, schema)`
returns a *validated* pydantic instance or raises `LLMError` after exhausting
retries. Retry covers: transport errors, malformed JSON, schema validation,
and evidence-verification failures reported back to the model.
"""

from __future__ import annotations

from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ValidationError

from fxterminal.core.config import Settings
from fxterminal.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """LLM call failed terminally (after retries)."""


@runtime_checkable
class LLMClient(Protocol):
    """Swappable LLM backend. `model`/`prompt_version` are recorded on every
    stored output for reproducibility."""

    model: str

    async def complete_json(
        self,
        system: str,
        prompt: str,
        schema: type[T],
        *,
        validation_feedback: str | None = None,
    ) -> T: ...


class AnthropicLLM:
    """Anthropic Messages API client. Uses prompt-instructed strict JSON +
    Pydantic validation — deliberately avoids SDK-specific structured-output
    helpers so the contract stays simple across SDK versions."""

    def __init__(self, settings: Settings) -> None:
        key = settings.anthropic_api_key
        if key is None or not key.get_secret_value():
            raise LLMError("FXT_ANTHROPIC_API_KEY is not configured")
        import anthropic

        self._client = anthropic.AsyncAnthropic(
            api_key=key.get_secret_value(),
            timeout=settings.llm_timeout_seconds,
        )
        self.model = settings.llm_model
        self._max_retries = settings.llm_max_retries

    async def complete_json(
        self,
        system: str,
        prompt: str,
        schema: type[T],
        *,
        validation_feedback: str | None = None,
    ) -> T:
        import anthropic

        messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
        last_err: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            if validation_feedback:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous output failed validation. Fix it and reply "
                            f"with corrected JSON only. Errors: {validation_feedback}"
                        ),
                    }
                )
            try:
                resp = await self._client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=system,
                    messages=messages,
                )
            except anthropic.APIError as e:
                last_err = LLMError(f"anthropic API error: {e}")
                logger.warning("llm_retry", attempt=attempt, error=str(e)[:200])
                continue

            text = "".join(b.text for b in resp.content if b.type == "text")
            messages.append({"role": "assistant", "content": text})

            parsed, err = _parse_and_validate(text, schema)
            if parsed is not None:
                return parsed
            last_err = err
            validation_feedback = str(err)
            logger.warning("llm_invalid_output", attempt=attempt, error=str(err)[:300])

        raise LLMError(f"failed after {self._max_retries} attempts: {last_err}")


class OpenAICompatLLM:
    """OpenAI-compatible /chat/completions client (Groq, OpenRouter, DeepSeek,
    Together, Ollama at http://localhost:11434/v1, etc.).

    Plain httpx — no SDK dependency. Same contract as AnthropicLLM:
    schema-validated JSON or LLMError after retries. Uses
    `response_format={"type": "json_object"}` when the endpoint supports it;
    providers that reject it still work because the prompt demands JSON and
    `_parse_and_validate` extracts it.
    """

    def __init__(self, settings: Settings) -> None:
        if not settings.llm_base_url:
            raise LLMError("FXT_LLM_BASE_URL is required for llm_provider=openai_compat")
        import httpx

        key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else ""
        self._client = httpx.AsyncClient(
            base_url=settings.llm_base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {key}"} if key else {},
            timeout=settings.llm_timeout_seconds,
        )
        self.model = settings.llm_model
        self._max_retries = settings.llm_max_retries

    async def complete_json(
        self,
        system: str,
        prompt: str,
        schema: type[T],
        *,
        validation_feedback: str | None = None,
    ) -> T:
        import httpx

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        last_err: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            if validation_feedback:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous output failed validation. Fix it and reply "
                            f"with corrected JSON only. Errors: {validation_feedback}"
                        ),
                    }
                )
            try:
                resp = await self._client.post(
                    "/chat/completions",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": 0,
                        "response_format": {"type": "json_object"},
                    },
                )
                if resp.status_code == 400 and "response_format" in resp.text:
                    # endpoint doesn't support json_object mode — retry without it
                    resp = await self._client.post(
                        "/chat/completions",
                        json={
                            "model": self.model,
                            "messages": messages,
                            "temperature": 0,
                        },
                    )
                resp.raise_for_status()
                data = resp.json()
            except (httpx.HTTPError, ValueError, KeyError) as e:
                last_err = LLMError(f"openai-compat API error: {e}")
                logger.warning("llm_retry", attempt=attempt, error=str(e)[:200])
                continue

            try:
                text = data["choices"][0]["message"]["content"] or ""
            except (KeyError, IndexError, TypeError) as e:
                last_err = LLMError(f"openai-compat: unexpected response shape: {e}")
                continue
            messages.append({"role": "assistant", "content": text})

            parsed, err = _parse_and_validate(text, schema)
            if parsed is not None:
                return parsed
            last_err = err
            validation_feedback = str(err)
            logger.warning("llm_invalid_output", attempt=attempt, error=str(err)[:300])

        raise LLMError(f"failed after {self._max_retries} attempts: {last_err}")


def build_llm(settings: Settings) -> LLMClient:
    """Construct the configured LLM backend. Raises LLMError when the chosen
    provider is missing its key/endpoint — callers degrade cleanly."""
    if settings.llm_provider == "openai_compat":
        return OpenAICompatLLM(settings)
    return AnthropicLLM(settings)


def _parse_and_validate[T: BaseModel](
    text: str, schema: type[T]
) -> tuple[T | None, Exception | None]:
    """Extract the first JSON object in the text and validate it."""
    import json

    candidate = text.strip()
    # tolerate ```json fences
    if "```" in candidate:
        import re

        m = re.search(r"```(?:json)?\s*(.*?)```", candidate, re.DOTALL)
        if m:
            candidate = m.group(1).strip()
    try:
        obj: Any = json.loads(candidate)
    except json.JSONDecodeError:
        # last resort: first {...} block
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            return None, ValueError("no JSON object in response")
        try:
            obj = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as e:
            return None, e
    try:
        return schema.model_validate(obj), None
    except ValidationError as e:
        return None, e
