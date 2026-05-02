"""Tests for RougeEvaluator and ExactMatchEvaluator."""
import pytest

from evaluators.exact_match import ExactMatchEvaluator
from evaluators.rouge_evaluator import RougeEvaluator


# ---------------------------------------------------------------------------
# RougeEvaluator
# ---------------------------------------------------------------------------


async def test_rouge_perfect_match():
    ev = RougeEvaluator()
    result = await ev.score("q", "The sky is blue.", context="The sky is blue.")
    assert result.score == pytest.approx(1.0)


async def test_rouge_zero_overlap():
    ev = RougeEvaluator()
    result = await ev.score("q", "Dogs bark loudly.", context="The sky is blue.")
    assert result.score == pytest.approx(0.0, abs=0.1)


async def test_rouge_partial_overlap():
    ev = RougeEvaluator()
    result = await ev.score("q", "The sky is very blue today.", context="The sky is blue.")
    assert 0.0 < result.score < 1.0


async def test_rouge_no_context_returns_zero():
    ev = RougeEvaluator()
    result = await ev.score("q", "some response", context=None)
    assert result.score == pytest.approx(0.0)
    assert "expected output" in result.explanation.lower()


async def test_rouge_score_in_bounds():
    ev = RougeEvaluator()
    result = await ev.score("q", "Hello world foo bar", context="Hello world")
    assert 0.0 <= result.score <= 1.0


async def test_rouge_explanation_contains_f1():
    ev = RougeEvaluator()
    result = await ev.score("q", "Paris is in France.", context="Paris is the capital of France.")
    assert "rougeL F1" in result.explanation or "ROUGE-L F1" in result.explanation


async def test_rouge_case_insensitive():
    ev = RougeEvaluator()
    upper = await ev.score("q", "THE SKY IS BLUE", context="the sky is blue")
    lower = await ev.score("q", "the sky is blue", context="the sky is blue")
    # Stemmer should produce same or very similar scores
    assert abs(upper.score - lower.score) < 0.05


async def test_rouge_name_and_version():
    ev = RougeEvaluator()
    assert ev.NAME == "rouge_l"
    assert ev.PROMPT_VERSION == "v1"


async def test_rouge_prompt_arg_ignored():
    """Different prompt values with same response/context must yield same score."""
    ev = RougeEvaluator()
    r1 = await ev.score("prompt A", "The answer is 42.", context="The answer is 42.")
    r2 = await ev.score("prompt B", "The answer is 42.", context="The answer is 42.")
    assert r1.score == pytest.approx(r2.score)


# ---------------------------------------------------------------------------
# ExactMatchEvaluator
# ---------------------------------------------------------------------------


async def test_exact_match_identical():
    ev = ExactMatchEvaluator()
    result = await ev.score("q", "Paris", context="Paris")
    assert result.score == pytest.approx(1.0)
    assert "match" in result.explanation.lower()


async def test_exact_match_case_insensitive():
    ev = ExactMatchEvaluator()
    result = await ev.score("q", "paris", context="Paris")
    assert result.score == pytest.approx(1.0)


async def test_exact_match_strips_whitespace():
    ev = ExactMatchEvaluator()
    result = await ev.score("q", "  Paris  ", context="Paris")
    assert result.score == pytest.approx(1.0)


async def test_exact_match_mismatch():
    ev = ExactMatchEvaluator()
    result = await ev.score("q", "London", context="Paris")
    assert result.score == pytest.approx(0.0)
    assert "does not match" in result.explanation.lower()


async def test_exact_match_no_context_returns_zero():
    ev = ExactMatchEvaluator()
    result = await ev.score("q", "some response", context=None)
    assert result.score == pytest.approx(0.0)
    assert "expected output" in result.explanation.lower()


async def test_exact_match_partial_content_is_mismatch():
    """Substring is not a match — only full equality counts."""
    ev = ExactMatchEvaluator()
    result = await ev.score("q", "Paris is the capital of France.", context="Paris")
    assert result.score == pytest.approx(0.0)


async def test_exact_match_name_and_version():
    ev = ExactMatchEvaluator()
    assert ev.NAME == "exact_match"
    assert ev.PROMPT_VERSION == "v1"


async def test_exact_match_makes_no_api_calls(respx_mock):
    """Reference-based evaluators must not make any HTTP calls."""
    ev = ExactMatchEvaluator()
    await ev.score("q", "Paris", context="Paris")
    assert len(respx_mock.calls) == 0


async def test_rouge_makes_no_api_calls(respx_mock):
    """Reference-based evaluators must not make any HTTP calls."""
    ev = RougeEvaluator()
    await ev.score("q", "The sky is blue.", context="The sky is blue.")
    assert len(respx_mock.calls) == 0
