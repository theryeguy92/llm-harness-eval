"""Coherence evaluator: is the response logically consistent and well-structured?"""
import os

import httpx

from .base import BaseEvaluator, EvalResult, parse_judge_response


_SYSTEM = """\
You are an expert evaluator assessing the logical coherence of AI-generated text.
Coherence measures whether the response is internally consistent, well-structured,
and makes sense on its own — independent of factual accuracy.

Score the response from 0.0 to 1.0:
  1.0 — Perfectly coherent; ideas flow logically with no contradictions
  0.7 — Mostly coherent with minor structural or logical issues
  0.5 — Partially coherent; noticeable gaps or inconsistencies
  0.3 — Low coherence; frequent jumps or contradictions
  0.0 — Completely incoherent

Return ONLY a JSON object, no extra text:
{"score": <float 0.0-1.0>, "explanation": "<one or two sentences>"}"""


class CoherenceEvaluator(BaseEvaluator):
    """Scores whether a response is logically coherent and internally consistent.

    Uses Claude as an LLM judge via the Anthropic Messages API.
    """

    NAME = "coherence"
    PROMPT_VERSION = "v1"

    def __init__(self, judge_model: str = "claude-haiku-4-5-20251001") -> None:
        """
        Args:
            judge_model: Anthropic model ID to use as the judge.
        """
        self._model = judge_model
        self._api_key = os.environ["ANTHROPIC_API_KEY"]

    async def score(
        self,
        prompt: str,
        response: str,
        context: str | None = None,
    ) -> EvalResult:
        """Score coherence of a model response.

        Args:
            prompt: The original user prompt.
            response: The model response to evaluate.
            context: Not used for coherence; ignored if provided.

        Returns:
            EvalResult with coherence score and explanation.
        """
        user_content = f"Prompt:\n{prompt}\n\nResponse:\n{response}"
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self._model,
                    "max_tokens": 256,
                    "system": _SYSTEM,
                    "messages": [{"role": "user", "content": user_content}],
                },
            )
            r.raise_for_status()

        return parse_judge_response(r.json()["content"][0]["text"])
