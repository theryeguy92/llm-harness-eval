"""Exact-match evaluator: case-insensitive string equality against expected output."""
from .base import BaseEvaluator, EvalResult


class ExactMatchEvaluator(BaseEvaluator):
    """Scores 1.0 when the response exactly matches the expected output, 0.0 otherwise.

    Comparison is case-insensitive and strips leading/trailing whitespace.
    No API calls are made — scoring runs locally.
    Pass the expected (reference) answer as the ``context`` argument.
    Best suited for closed-form tasks (classification labels, factual lookups,
    structured extraction) where exactly one correct answer exists.
    Use an LLM-as-judge evaluator instead for open-ended or multi-valid-answer tasks.
    """

    NAME = "exact_match"
    PROMPT_VERSION = "v1"

    async def score(
        self,
        prompt: str,
        response: str,
        context: str | None = None,
    ) -> EvalResult:
        """Check whether *response* exactly matches the reference in *context*.

        Args:
            prompt: The original user prompt (unused; included for interface compatibility).
            response: The model-generated response to evaluate.
            context: Expected (reference) output to compare against. Returns 0.0 if None.

        Returns:
            EvalResult with score 1.0 (match) or 0.0 (no match).
        """
        if context is None:
            return EvalResult(
                score=0.0,
                explanation="No expected output provided; pass the reference answer as context.",
            )
        match = response.strip().lower() == context.strip().lower()
        return EvalResult(
            score=1.0 if match else 0.0,
            explanation="Exact match." if match else "Response does not match expected output.",
        )
