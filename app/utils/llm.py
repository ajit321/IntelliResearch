"""
app/utils/llm.py
OpenRouter LLM wrapper via LiteLLM.
Handles model routing, token logging, structured outputs, and error recovery.
"""

import json
from typing import Any

from litellm import acompletion
from pydantic import BaseModel

from app.config import get_settings
from app.utils.logging import get_logger
from app.utils.decorators import retry

logger = get_logger(__name__)
settings = get_settings()

# ── OpenRouter headers required by the API ────────────────────────────────────
_OPENROUTER_HEADERS: dict[str, str] = {
    "HTTP-Referer": "http://localhost:8501",
    "X-Title": "IntelliResearch",
}


@retry(max_attempts=3, delay=1.0, backoff=2.0)
async def call_llm(
    prompt: str,
    system: str = "You are a helpful AI research assistant.",
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    response_format: type[BaseModel] | None = None,
) -> str:
    """
    Call an LLM via OpenRouter using LiteLLM.

    Automatically logs token usage, latency, and model info.
    Supports structured output via Pydantic model schema injection.

    Args:
        prompt: The user message / query to send.
        system: The system prompt defining agent behaviour.
        model: Override the default model from settings.
        temperature: Override the default temperature from settings.
        max_tokens: Override the default max_tokens from settings.
        response_format: Optional Pydantic model for structured JSON output.

    Returns:
        The LLM response content as a string.

    Raises:
        RuntimeError: If the LLM call fails after all retry attempts.

    Example:
        result = await call_llm(
            prompt="Summarise quantum computing trends",
            system="You are a research analyst.",
        )
    """
    _model       = model       or settings.llm_model
    _temperature = temperature or settings.llm_temperature
    _max_tokens  = max_tokens  or settings.llm_max_tokens

    # Inject JSON schema instruction for structured outputs
    _system = system
    if response_format is not None:
        schema = response_format.model_json_schema()
        _system = (
            f"{system}\n\nYou MUST respond with a valid JSON object matching "
            f"this schema:\n{json.dumps(schema, indent=2)}\n"
            "Do not include any text outside the JSON object."
        )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": _system},
        {"role": "user",   "content": prompt},
    ]

    logger.info(
        "LLM call started",
        model=_model,
        temperature=_temperature,
        max_tokens=_max_tokens,
        prompt_chars=len(prompt),
    )

    response = await acompletion(
        model=_model,
        api_key=settings.openrouter_api_key,
        api_base="https://openrouter.ai/api/v1",
        messages=messages,
        temperature=_temperature,
        max_tokens=_max_tokens,
        timeout=90,
        extra_headers=_OPENROUTER_HEADERS,
    )

    # Log token usage for cost tracking
    usage = getattr(response, "usage", None)
    if usage:
        logger.info(
            "LLM call completed",
            model=_model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )

    content: str = response.choices[0].message.content or ""
    return content


async def call_llm_structured(
    prompt: str,
    system: str,
    output_model: type[BaseModel],
    model: str | None = None,
) -> BaseModel | None:
    """
    Call an LLM and parse the response into a Pydantic model.

    Uses structured output prompting to enforce JSON schema compliance.
    Falls back to None if parsing fails after retries.

    Args:
        prompt: The user message.
        system: The system prompt.
        output_model: Pydantic model class to parse the response into.
        model: Optional model override.

    Returns:
        Parsed Pydantic model instance, or None on failure.

    Example:
        class AnalysisOutput(BaseModel):
            summary: str
            contradictions: list[str]
            confidence: float

        result = await call_llm_structured(
            prompt="Analyse these sources...",
            system="You are a critical analyst.",
            output_model=AnalysisOutput,
        )
    """
    raw = await call_llm(
        prompt=prompt,
        system=system,
        model=model,
        response_format=output_model,
    )

    try:
        # Strip markdown code fences if present
        clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return output_model.model_validate_json(clean)
    except Exception as exc:
        logger.error(
            "Structured output parsing failed",
            model_name=output_model.__name__,
            raw_response=raw[:200],
            error=str(exc),
        )
        return None


def get_model_for_task(task_type: str) -> str:
    """
    Select the optimal model based on the agent task type.

    Smart model routing: fast cheap models for retrieval/simple tasks,
    powerful models for analysis and reasoning.

    Args:
        task_type: One of 'retrieval', 'analysis', 'reasoning', 'judge'.

    Returns:
        LiteLLM model string for OpenRouter.
    """
    model_map: dict[str, str] = {
        "retrieval": "openrouter/openai/gpt-4o-mini",       # Fast + cheap
        "analysis":  "openrouter/openai/gpt-4o",            # Accurate
        "reasoning": "openrouter/anthropic/claude-3.5-sonnet", # Best reasoning
        "judge":     "openrouter/openai/gpt-4o",            # Reliable scoring
        "default":   settings.llm_model,
    }
    selected = model_map.get(task_type, model_map["default"])
    logger.debug("Model selected for task", task_type=task_type, model=selected)
    return selected
