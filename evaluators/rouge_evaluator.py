"""ROUGE-L evaluator: lexical overlap between response and expected output."""
from rouge_score import rouge_scorer

from .base import BaseEvaluator, EvalResult

_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


class RougeEvaluator(BaseEvaluator):
    """Scores a response using ROUGE-L F1 against an expected output.

    No API calls are made — scoring is purely lexical and runs locally.
    Pass the expected (reference) answer as the ``context`` argument.
    Best suited for tasks with deterministic or near-deterministic answers
    (e.g. extraction, summarization with a reference summary, QA with gold answers).
    Use an LLM-as-judge evaluator instead when acceptable answers are open-ended.
    """

    NAME = "rouge_l"
    PROMPT_VERSION = "v1"

    async def score(
        self,
        prompt: str,
        response: str,
        context: str | None = None,
    ) -> EvalResult:
        """Compute ROUGE-L F1 between *response* and the reference in *context*.

        Args:
            prompt: The original user prompt (unused; included for interface compatibility).
            response: The model-generated response to evaluate.
            context: Expected (reference) output to score against. Returns 0.0 if None.

        Returns:
            EvalResult with ROUGE-L F1 score in [0, 1] and a short explanation.
        """
        if context is None:
            return EvalResult(
                score=0.0,
                explanation="No expected output provided; pass the reference answer as context.",
            )
        scores = _SCORER.score(target=context, prediction=response)
        f1 = round(scores["rougeL"].fmeasure, 4)
        return EvalResult(
            score=f1,
            explanation=f"ROUGE-L F1 = {f1:.4f} (precision={scores['rougeL'].precision:.4f}, recall={scores['rougeL'].recall:.4f}).",
        )
