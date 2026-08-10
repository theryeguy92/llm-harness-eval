"""Pairwise evaluators: compare two responses head-to-head for a given quality dimension."""
import json
import re
from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from .base import call_judge
from env import require_key


class PairwiseResult(BaseModel):
    """Outcome of a pairwise comparison between two model responses."""

    winner: Literal["a", "b", "tie"]
    confidence: float = Field(..., ge=0.0, le=1.0, description="Judge's certainty in the verdict")
    reasoning: str = Field(..., description="One or two sentences explaining the verdict")


def _parse_pairwise_response(text: str) -> PairwiseResult:
    """Extract a PairwiseResult from a judge model's raw text output.

    Handles plain JSON, markdown-fenced JSON, and JSON preceded by preamble.
    Returns a tie with confidence 0.0 if no valid JSON can be extracted.
    """
    clean = re.sub(r"```(?:json)?\s*", "", text).strip()
    for candidate in [clean]:
        try:
            parsed = json.loads(candidate)
            return PairwiseResult(
                winner=parsed["winner"],
                confidence=float(parsed["confidence"]),
                reasoning=parsed["reasoning"],
            )
        except (json.JSONDecodeError, KeyError, ValueError, ValidationError):
            pass

    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group())
            return PairwiseResult(
                winner=parsed["winner"],
                confidence=float(parsed["confidence"]),
                reasoning=parsed["reasoning"],
            )
        except (json.JSONDecodeError, KeyError, ValueError, ValidationError):
            pass

    return PairwiseResult(winner="tie", confidence=0.0, reasoning="Judge response could not be parsed.")


class BasePairwiseEvaluator(ABC):
    """Abstract base for pairwise LLM-as-judge evaluators.

    Pairwise evaluation asks the judge to pick a winner between two responses
    rather than assigning each an independent score. This catches preference
    differences that pointwise scores miss — a judge may rate both responses
    0.8, yet reliably prefer one over the other in direct comparison.

    **Positional bias warning.** LLM judges disproportionately favor whichever
    response appears first (position A). To get reliable results, callers should
    run ``compare(prompt, a, b)`` and ``compare(prompt, b, a)`` and check
    whether the winner is consistent. If the verdict flips on swap, treat the
    result as a tie. A helper for this is::

        async def compare_debiased(ev, prompt, a, b):
            r1 = await ev.compare(prompt, a, b)
            r2 = await ev.compare(prompt, b, a)  # a and b swapped
            # r2.winner is from b's perspective, so invert it
            inv = {"a": "b", "b": "a", "tie": "tie"}
            if r1.winner == inv[r2.winner]:
                return r1  # consistent verdict
            return PairwiseResult(winner="tie", confidence=0.0,
                                  reasoning="Verdict reversed on position swap — treating as tie.")

    Subclasses must set NAME and PROMPT_VERSION and implement ``compare()``.
    """

    NAME: str = ""
    PROMPT_VERSION: str = ""

    @abstractmethod
    async def compare(
        self,
        prompt: str,
        response_a: str,
        response_b: str,
    ) -> PairwiseResult:
        """Compare two responses for this evaluator's quality dimension.

        Args:
            prompt: The original user prompt both responses were generated from.
            response_a: The first candidate response (label "A" in the judge prompt).
            response_b: The second candidate response (label "B" in the judge prompt).

        Returns:
            PairwiseResult indicating the winner, confidence, and reasoning.
        """


_EXPLAINABILITY_SYSTEM = """\
You are an expert evaluator assessing which of two AI-generated responses
better explains a concept. Evaluate on three axes:
  Clarity       — Is the explanation easy to follow step by step?
  Accuracy      — Does it correctly represent the concept?
  Accessibility — Would a motivated non-expert understand it?

You will be shown a prompt and two responses labelled A and B.
Pick the response that explains the concept more clearly and correctly.
Use "tie" only when the responses are genuinely equivalent on all axes.

Return ONLY a JSON object, no extra text:
{"winner": "a" | "b" | "tie", "confidence": <float 0.0-1.0>, "reasoning": "<one or two sentences>"}

confidence reflects your certainty: 1.0 = clear winner, 0.5 = marginal preference."""


class ExplainabilityPairwise(BasePairwiseEvaluator):
    """Compares two responses on how well they explain the concept in the prompt.

    Uses Claude as an LLM judge via the Anthropic Messages API.
    Useful for evaluating teaching quality, documentation clarity, or
    tutorial-style responses where explanation depth matters more than brevity.

    See ``BasePairwiseEvaluator`` for positional bias considerations.
    """

    NAME = "explainability"
    PROMPT_VERSION = "v1"

    def __init__(self, judge_model: str = "claude-haiku-4-5-20251001") -> None:
        """
        Args:
            judge_model: Anthropic model ID to use as the judge.
        """
        self._model = judge_model
        self._api_key = require_key("ANTHROPIC_API_KEY")

    async def compare(
        self,
        prompt: str,
        response_a: str,
        response_b: str,
    ) -> PairwiseResult:
        """Judge which response better explains the concept in the prompt.

        Rubric:
            winner="a"   — Response A explains the concept more clearly and accurately.
            winner="b"   — Response B explains the concept more clearly and accurately.
            winner="tie" — Both responses are equivalent in clarity and accuracy.
            confidence   — 1.0 means the judge is certain; 0.5 means marginal preference.

        Args:
            prompt: The original user prompt (e.g. "Explain how attention works").
            response_a: First candidate response.
            response_b: Second candidate response.

        Returns:
            PairwiseResult with winner, confidence, and one-sentence reasoning.
        """
        user_content = (
            f"Prompt: {prompt}\n\n"
            f"Response A:\n{response_a}\n\n"
            f"Response B:\n{response_b}"
        )
        text = await call_judge(self._api_key, self._model, _EXPLAINABILITY_SYSTEM, user_content)
        return _parse_pairwise_response(text)
