"""Relevance evaluator: does the response directly address the prompt?"""
import json
import os

import httpx

from .base import BaseEvaluator, EvalResult


_SYSTEM = """\
You are an expert evaluator assessing the relevance of AI-generated responses.
Relevance measures how directly and completely the response addresses the user's prompt.
Ignore stylistic issues; focus only on whether the content answers what was asked.

Score the response from 0.0 to 1.0:
  1.0 — Fully addresses the prompt with no off-topic content
  0.7 — Mostly on-topic; minor tangents or partial answers
  0.5 — Partially relevant; key aspects of the prompt are missed
  0.3 — Mostly off-topic or only superficially related
  0.0 — Completely irrelevant to the prompt

Return ONLY a JSON object, no extra text:
{"score": <float 0.0-1.0>, "explanation": "<one or two sentences>"}"""


class RelevanceEvaluator(BaseEvaluator):
    """Scores how directly and completely a response addresses the prompt.

    Uses Claude as an LLM judge via the Anthropic Messages API.
    """

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
        """Score how relevant the response is to the original prompt.

        Args:
            prompt: The original user prompt.
            response: The model response to evaluate.
            context: Not used for relevance; ignored if provided.

        Returns:
            EvalResult with relevance score and explanation.
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

        raw = r.json()["content"][0]["text"].strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(raw)
        return EvalResult(score=float(parsed["score"]), explanation=parsed["explanation"])
