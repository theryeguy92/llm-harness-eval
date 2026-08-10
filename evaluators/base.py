"""Base classes and data models for LLM response evaluators."""
import asyncio
import json
import re
from abc import ABC, abstractmethod

import httpx
from pydantic import BaseModel, Field

_ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"


class EvalResult(BaseModel):
    """Result of a single evaluator run."""

    score: float = Field(..., ge=0.0, le=1.0, description="Quality score between 0.0 and 1.0")
    explanation: str = Field(..., description="Natural-language explanation of the score")
    parse_failed: bool = Field(
        False,
        description="True when the judge response could not be parsed and the score is a 0.5 fallback.",
    )


async def call_judge(
    api_key: str,
    model: str,
    system: str,
    user_content: str,
    max_tokens: int = 256,
    retries: int = 5,
) -> str:
    """Call the Anthropic Messages API as an LLM judge and return the raw text reply.

    Retries with exponential backoff on 429/529 and on network/timeout errors,
    matching the retry behavior of the runners.

    Args:
        api_key: Anthropic API key.
        model: Judge model ID.
        system: Judge system prompt (rubric).
        user_content: User message with the material to judge.
        max_tokens: Max tokens for the judge reply.
        retries: Number of attempts before giving up.

    Returns:
        The judge's raw text content.

    Raises:
        httpx.HTTPStatusError: On a non-retryable error, or after all retries fail.
    """
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user_content}],
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        for attempt in range(retries):
            try:
                r = await client.post(_ANTHROPIC_MESSAGES_URL, headers=headers, json=body)
            except (httpx.TimeoutException, httpx.TransportError):
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
                continue
            if r.status_code in (429, 529):
                await asyncio.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            break
        else:
            r.raise_for_status()
    return r.json()["content"][0]["text"]


def parse_judge_response(text: str) -> "EvalResult":
    """Extract score and explanation from a judge model's raw text output.

    Handles plain JSON, markdown-fenced JSON, and JSON preceded by preamble text.
    Returns a neutral fallback EvalResult if no valid JSON can be extracted.
    """
    # Strip markdown fences, then try parsing the whole cleaned string
    clean = re.sub(r"```(?:json)?\s*", "", text).strip()
    try:
        parsed = json.loads(clean)
        return EvalResult(score=float(parsed["score"]), explanation=parsed["explanation"])
    except (json.JSONDecodeError, KeyError, ValueError):
        pass

    # Fall back to finding the first {...} block anywhere in the original text
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group())
            return EvalResult(score=float(parsed["score"]), explanation=parsed["explanation"])
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    return EvalResult(
        score=0.5,
        explanation="Judge response could not be parsed.",
        parse_failed=True,
    )


class BaseEvaluator(ABC):
    """Abstract base for all response quality evaluators.

    Subclasses implement a single quality dimension (coherence, faithfulness, etc.)
    and return a normalized score in [0, 1] alongside a human-readable explanation.
    All scoring is async to support concurrent evaluation pipelines.

    Subclasses must set NAME and PROMPT_VERSION so historical reports stay
    comparable when judge prompts change.
    """

    NAME: str = ""
    PROMPT_VERSION: str = ""

    @abstractmethod
    async def score(
        self,
        prompt: str,
        response: str,
        context: str | None = None,
    ) -> EvalResult:
        """Score a model response on this evaluator's quality dimension.

        Args:
            prompt: The original user prompt sent to the model.
            response: The model-generated response to evaluate.
            context: Optional reference text, e.g. retrieved documents for RAG evals.

        Returns:
            EvalResult with a score in [0, 1] and a natural-language explanation.
        """
