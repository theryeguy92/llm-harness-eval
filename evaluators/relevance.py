"""Relevance evaluator: does the response directly address the prompt?"""

from .base import BaseEvaluator, EvalResult, call_judge, parse_judge_response
from env import require_key


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

    NAME = "relevance"
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
        """Score how relevant the response is to the original prompt.

        Rubric:
            1.0 — Fully addresses the prompt with no off-topic content.
            0.7 — Mostly on-topic; minor tangents or partial answers.
            0.5 — Partially relevant; key aspects of the prompt are missed.
            0.3 — Mostly off-topic or only superficially related.
            0.0 — Completely irrelevant to the prompt.

        Args:
            prompt: The original user prompt.
            response: The model response to evaluate.
            context: Not used for relevance; ignored if provided.

        Returns:
            EvalResult with relevance score and explanation.
        """
        user_content = f"Prompt:\n{prompt}\n\nResponse:\n{response}"
        text = await call_judge(self._api_key, self._model, _SYSTEM, user_content)
        return parse_judge_response(text)
