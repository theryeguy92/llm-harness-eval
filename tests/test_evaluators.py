"""Tests for CoherenceEvaluator, RelevanceEvaluator, and FaithfulnessEvaluator."""
import httpx
import pytest
import respx

from evaluators.coherence import CoherenceEvaluator
from evaluators.faithfulness import FaithfulnessEvaluator
from evaluators.relevance import RelevanceEvaluator

from .conftest import claude_response_body

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
JUDGE_JSON = '{"score": 0.9, "explanation": "Well structured and clear."}'
JUDGE_JSON_FENCED = '```json\n{"score": 0.8, "explanation": "Mostly relevant."}\n```'
JUDGE_JSON_PREAMBLE = 'Here is my evaluation:\n{"score": 0.7, "explanation": "Partially faithful."}'
JUDGE_MALFORMED = "I cannot provide a numeric score for this response."


# ---------------------------------------------------------------------------
# CoherenceEvaluator
# ---------------------------------------------------------------------------


async def test_coherence_happy_path(anthropic_env, respx_mock):
    respx_mock.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(200, json=claude_response_body(JUDGE_JSON))
    )
    evaluator = CoherenceEvaluator()
    result = await evaluator.score("What is X?", "X is Y because of Z.")

    assert result.score == pytest.approx(0.9)
    assert result.explanation == "Well structured and clear."


async def test_coherence_fenced_json(anthropic_env, respx_mock):
    """Judge wraps output in ```json fences — must still parse correctly."""
    respx_mock.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(200, json=claude_response_body(JUDGE_JSON_FENCED))
    )
    evaluator = CoherenceEvaluator()
    result = await evaluator.score("What is X?", "X is Y.")

    assert result.score == pytest.approx(0.8)
    assert "relevant" in result.explanation.lower()


async def test_coherence_json_with_preamble(anthropic_env, respx_mock):
    """Judge adds explanatory text before the JSON object — must still parse."""
    respx_mock.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(200, json=claude_response_body(JUDGE_JSON_PREAMBLE))
    )
    evaluator = CoherenceEvaluator()
    result = await evaluator.score("What is X?", "X is Y.")

    assert result.score == pytest.approx(0.7)


async def test_coherence_fallback_on_malformed(anthropic_env, respx_mock):
    """Malformed judge response must return a neutral fallback, not raise."""
    respx_mock.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(200, json=claude_response_body(JUDGE_MALFORMED))
    )
    evaluator = CoherenceEvaluator()
    result = await evaluator.score("What is X?", "X is Y.")

    assert 0.0 <= result.score <= 1.0
    assert len(result.explanation) > 0


async def test_coherence_sends_correct_headers(anthropic_env, respx_mock):
    route = respx_mock.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(200, json=claude_response_body(JUDGE_JSON))
    )
    evaluator = CoherenceEvaluator(judge_model="claude-haiku-4-5-20251001")
    await evaluator.score("p", "r")

    request = route.calls.last.request
    assert request.headers["x-api-key"] == "test-anthropic-key"
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert request.headers["content-type"] == "application/json"


async def test_coherence_sends_correct_body(anthropic_env, respx_mock):
    route = respx_mock.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(200, json=claude_response_body(JUDGE_JSON))
    )
    evaluator = CoherenceEvaluator(judge_model="claude-haiku-4-5-20251001")
    await evaluator.score("my prompt", "my response")

    body = route.calls.last.request.read()
    import json
    payload = json.loads(body)
    assert payload["model"] == "claude-haiku-4-5-20251001"
    assert payload["messages"][0]["role"] == "user"
    assert "my prompt" in payload["messages"][0]["content"]
    assert "my response" in payload["messages"][0]["content"]


# ---------------------------------------------------------------------------
# RelevanceEvaluator
# ---------------------------------------------------------------------------


async def test_relevance_happy_path(anthropic_env, respx_mock):
    respx_mock.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(200, json=claude_response_body(JUDGE_JSON))
    )
    evaluator = RelevanceEvaluator()
    result = await evaluator.score("What is X?", "X is Y.")

    assert result.score == pytest.approx(0.9)


async def test_relevance_fenced_json(anthropic_env, respx_mock):
    respx_mock.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(200, json=claude_response_body(JUDGE_JSON_FENCED))
    )
    evaluator = RelevanceEvaluator()
    result = await evaluator.score("What is X?", "X is Y.")

    assert result.score == pytest.approx(0.8)


async def test_relevance_fallback_on_malformed(anthropic_env, respx_mock):
    respx_mock.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(200, json=claude_response_body(JUDGE_MALFORMED))
    )
    evaluator = RelevanceEvaluator()
    result = await evaluator.score("What is X?", "X is Y.")

    assert 0.0 <= result.score <= 1.0


async def test_relevance_ignores_context(anthropic_env, respx_mock):
    """Relevance evaluator accepts context arg but doesn't include it in the API call."""
    route = respx_mock.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(200, json=claude_response_body(JUDGE_JSON))
    )
    evaluator = RelevanceEvaluator()
    await evaluator.score("prompt", "response", context="some reference doc")

    import json
    payload = json.loads(route.calls.last.request.read())
    assert "some reference doc" not in payload["messages"][0]["content"]


# ---------------------------------------------------------------------------
# FaithfulnessEvaluator
# ---------------------------------------------------------------------------


async def test_faithfulness_no_context_returns_neutral(anthropic_env):
    """No HTTP call should be made when context is None."""
    evaluator = FaithfulnessEvaluator()
    result = await evaluator.score("What is X?", "X is Y.", context=None)

    assert result.score == pytest.approx(0.5)
    assert "context" in result.explanation.lower()


async def test_faithfulness_happy_path(anthropic_env, respx_mock):
    respx_mock.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(200, json=claude_response_body(JUDGE_JSON))
    )
    evaluator = FaithfulnessEvaluator()
    result = await evaluator.score("What is X?", "X is Y.", context="X is indeed Y.")

    assert result.score == pytest.approx(0.9)


async def test_faithfulness_fenced_json(anthropic_env, respx_mock):
    respx_mock.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(200, json=claude_response_body(JUDGE_JSON_FENCED))
    )
    evaluator = FaithfulnessEvaluator()
    result = await evaluator.score("p", "r", context="ctx")

    assert result.score == pytest.approx(0.8)


async def test_faithfulness_fallback_on_malformed(anthropic_env, respx_mock):
    respx_mock.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(200, json=claude_response_body(JUDGE_MALFORMED))
    )
    evaluator = FaithfulnessEvaluator()
    result = await evaluator.score("p", "r", context="ctx")

    assert 0.0 <= result.score <= 1.0


async def test_faithfulness_includes_context_in_request(anthropic_env, respx_mock):
    """The reference context must appear in the payload sent to the judge."""
    route = respx_mock.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(200, json=claude_response_body(JUDGE_JSON))
    )
    evaluator = FaithfulnessEvaluator()
    await evaluator.score("What is X?", "X is Y.", context="The document says X equals Y.")

    import json
    payload = json.loads(route.calls.last.request.read())
    assert "The document says X equals Y." in payload["messages"][0]["content"]
