"""Tests for BasePairwiseEvaluator and ExplainabilityPairwise."""
import json

import httpx
import pytest
import respx

from evaluators.pairwise import ExplainabilityPairwise, PairwiseResult, _parse_pairwise_response

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"


def _judge_body(text: str) -> dict:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-4-5-20251001",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 20, "output_tokens": 30},
    }


# ---------------------------------------------------------------------------
# PairwiseResult model
# ---------------------------------------------------------------------------


def test_pairwise_result_valid_winners():
    for winner in ("a", "b", "tie"):
        r = PairwiseResult(winner=winner, confidence=0.8, reasoning="ok")
        assert r.winner == winner


def test_pairwise_result_rejects_invalid_winner():
    with pytest.raises(Exception):
        PairwiseResult(winner="c", confidence=0.8, reasoning="bad")


def test_pairwise_result_confidence_bounds():
    with pytest.raises(Exception):
        PairwiseResult(winner="a", confidence=1.5, reasoning="out of range")
    with pytest.raises(Exception):
        PairwiseResult(winner="a", confidence=-0.1, reasoning="out of range")


# ---------------------------------------------------------------------------
# _parse_pairwise_response
# ---------------------------------------------------------------------------


def test_parse_plain_json_a_wins():
    text = '{"winner": "a", "confidence": 0.9, "reasoning": "A is clearer."}'
    result = _parse_pairwise_response(text)
    assert result.winner == "a"
    assert result.confidence == pytest.approx(0.9)
    assert result.reasoning == "A is clearer."


def test_parse_plain_json_b_wins():
    text = '{"winner": "b", "confidence": 0.7, "reasoning": "B covers more ground."}'
    result = _parse_pairwise_response(text)
    assert result.winner == "b"


def test_parse_tie():
    text = '{"winner": "tie", "confidence": 0.5, "reasoning": "Both are equivalent."}'
    result = _parse_pairwise_response(text)
    assert result.winner == "tie"


def test_parse_fenced_json():
    text = '```json\n{"winner": "a", "confidence": 0.85, "reasoning": "Clearer."}\n```'
    result = _parse_pairwise_response(text)
    assert result.winner == "a"
    assert result.confidence == pytest.approx(0.85)


def test_parse_json_with_preamble():
    text = 'After careful review:\n{"winner": "b", "confidence": 0.6, "reasoning": "More accurate."}'
    result = _parse_pairwise_response(text)
    assert result.winner == "b"


def test_parse_malformed_returns_tie_fallback():
    result = _parse_pairwise_response("I cannot determine a winner.")
    assert result.winner == "tie"
    assert result.confidence == 0.0
    assert "could not be parsed" in result.reasoning


# ---------------------------------------------------------------------------
# ExplainabilityPairwise class constants
# ---------------------------------------------------------------------------


def test_explainability_has_name_and_version():
    assert ExplainabilityPairwise.NAME == "explainability"
    assert ExplainabilityPairwise.PROMPT_VERSION == "v1"


def test_instance_inherits_class_constants(anthropic_env):
    ev = ExplainabilityPairwise()
    assert ev.NAME == "explainability"
    assert ev.PROMPT_VERSION == "v1"


# ---------------------------------------------------------------------------
# ExplainabilityPairwise.compare() — happy paths
# ---------------------------------------------------------------------------


async def test_compare_returns_a_wins(anthropic_env, respx_mock):
    verdict = '{"winner": "a", "confidence": 0.9, "reasoning": "A explains the concept step by step."}'
    respx_mock.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(200, json=_judge_body(verdict))
    )
    ev = ExplainabilityPairwise()
    result = await ev.compare("Explain recursion.", "Response A text.", "Response B text.")

    assert result.winner == "a"
    assert result.confidence == pytest.approx(0.9)
    assert "step by step" in result.reasoning


async def test_compare_returns_b_wins(anthropic_env, respx_mock):
    verdict = '{"winner": "b", "confidence": 0.75, "reasoning": "B uses a clearer analogy."}'
    respx_mock.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(200, json=_judge_body(verdict))
    )
    ev = ExplainabilityPairwise()
    result = await ev.compare("Explain sorting.", "Response A.", "Response B.")

    assert result.winner == "b"
    assert result.confidence == pytest.approx(0.75)


async def test_compare_returns_tie(anthropic_env, respx_mock):
    verdict = '{"winner": "tie", "confidence": 0.5, "reasoning": "Both are equally clear."}'
    respx_mock.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(200, json=_judge_body(verdict))
    )
    ev = ExplainabilityPairwise()
    result = await ev.compare("Explain gravity.", "Response A.", "Response B.")

    assert result.winner == "tie"


# ---------------------------------------------------------------------------
# HTTP request shape
# ---------------------------------------------------------------------------


async def test_compare_sends_correct_headers(anthropic_env, respx_mock):
    verdict = '{"winner": "a", "confidence": 0.8, "reasoning": "Clear."}'
    route = respx_mock.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(200, json=_judge_body(verdict))
    )
    ev = ExplainabilityPairwise()
    await ev.compare("Explain X.", "A text.", "B text.")

    headers = route.calls.last.request.headers
    assert headers["x-api-key"] == "test-anthropic-key"
    assert headers["anthropic-version"] == "2023-06-01"
    assert headers["content-type"] == "application/json"


async def test_compare_body_contains_both_responses(anthropic_env, respx_mock):
    verdict = '{"winner": "a", "confidence": 0.8, "reasoning": "Clear."}'
    route = respx_mock.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(200, json=_judge_body(verdict))
    )
    ev = ExplainabilityPairwise(judge_model="claude-sonnet-4-6")
    await ev.compare("What is X?", "alpha response", "beta response")

    payload = json.loads(route.calls.last.request.read())
    assert payload["model"] == "claude-sonnet-4-6"
    user_msg = payload["messages"][0]["content"]
    assert "alpha response" in user_msg
    assert "beta response" in user_msg
    assert "Response A" in user_msg
    assert "Response B" in user_msg


async def test_compare_raises_on_http_error(anthropic_env, respx_mock):
    respx_mock.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(500, json={"error": "internal"})
    )
    ev = ExplainabilityPairwise()
    with pytest.raises(httpx.HTTPStatusError):
        await ev.compare("Explain Y.", "A.", "B.")
