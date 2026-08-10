"""Faithfulness evaluator: does the response stay grounded in the provided context?"""

from .base import BaseEvaluator, EvalResult, call_judge, parse_judge_response
from env import require_key


_SYSTEM = """\
You are an expert evaluator assessing the faithfulness of AI-generated responses.
Faithfulness measures whether all claims in the response are grounded in the provided
context. Penalize hallucinations — statements that contradict or go beyond the context.

Score the response from 0.0 to 1.0:
  1.0 — Every claim is directly supported by the context
  0.7 — Most claims are supported; minor unsupported extrapolation
  0.5 — Roughly half the claims are grounded; notable hallucinations present
  0.3 — Few claims are grounded; significant hallucination
  0.0 — Response contradicts or entirely ignores the context

Return ONLY a JSON object, no extra text:
{"score": <float 0.0-1.0>, "explanation": "<one or two sentences>"}"""


class FaithfulnessEvaluator(BaseEvaluator):
    """Scores whether a response is grounded in the reference context.

    Primarily useful for RAG pipelines where the response should be derived from
    retrieved documents. Uses Claude as an LLM judge.
    """

    NAME = "faithfulness"
    PROMPT_VERSION = "v1"

    def __init__(self, judge_model: str = "claude-haiku-4-5-20251001") -> None:
        """
        Args:
            judge_model: Anthropic model ID to use as the judge.
        """
        self._model = judge_model
        self._api_key = require_key("ANTHROPIC_API_KEY")

    async def score(
        self,
        prompt: str,
        response: str,
        context: str | None = None,
    ) -> EvalResult:
        """Score faithfulness of a response relative to the provided context.

        Rubric:
            1.0 — Every claim is directly supported by the context.
            0.7 — Most claims are supported; minor unsupported extrapolation.
            0.5 — Roughly half the claims are grounded; notable hallucinations present.
            0.3 — Few claims are grounded; significant hallucination.
            0.0 — Response contradicts or entirely ignores the context.

        Args:
            prompt: The original user prompt.
            response: The model response to evaluate.
            context: Reference text the response should be grounded in.
                     Returns a neutral 0.5 score with a note if omitted.

        Returns:
            EvalResult with faithfulness score and explanation.
        """
        if context is None:
            return EvalResult(
                score=0.5,
                explanation="No reference context provided; faithfulness cannot be assessed.",
            )

        user_content = f"Context:\n{context}\n\nPrompt:\n{prompt}\n\nResponse:\n{response}"
        text = await call_judge(self._api_key, self._model, _SYSTEM, user_content)
        return parse_judge_response(text)
