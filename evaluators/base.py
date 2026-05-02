"""Base classes and data models for LLM response evaluators."""
import json
import re
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class EvalResult(BaseModel):
    """Result of a single evaluator run."""

    score: float = Field(..., ge=0.0, le=1.0, description="Quality score between 0.0 and 1.0")
    explanation: str = Field(..., description="Natural-language explanation of the score")


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

    return EvalResult(score=0.5, explanation="Judge response could not be parsed.")


class BaseEvaluator(ABC):
    """Abstract base for all response quality evaluators.

    Subclasses implement a single quality dimension (coherence, faithfulness, etc.)
    and return a normalized score in [0, 1] alongside a human-readable explanation.
    All scoring is async to support concurrent evaluation pipelines.
    """

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
