"""Tests for ClaudeRunner, OpenAIRunner, and GeminiRunner."""
import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest
import respx

from runners.claude_runner import ClaudeRunner
from runners.gemini_runner import GeminiRunner
from runners.openai_runner import OpenAIRunner

from .conftest import claude_response_body, gemini_response_body, openai_response_body

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent"


# ---------------------------------------------------------------------------
# ClaudeRunner
# ---------------------------------------------------------------------------


async def test_claude_run_returns_result(anthropic_env, respx_mock):
    respx_mock.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(200, json=claude_response_body("Hello!"))
    )
    runner = ClaudeRunner(model="claude-haiku-4-5-20251001")
    result = await runner.run("Say hello.")

    assert result.response == "Hello!"
    assert result.model == "claude-haiku-4-5-20251001"
    assert result.input_tokens == 10
    assert result.output_tokens == 20
    assert result.latency_ms > 0
    # 10 input @ $0.80/M + 20 output @ $4.00/M = $0.000088
    assert result.cost_usd == pytest.approx(0.000088)


async def test_claude_sends_correct_headers(anthropic_env, respx_mock):
    route = respx_mock.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(200, json=claude_response_body("Hi"))
    )
    runner = ClaudeRunner(model="claude-haiku-4-5-20251001")
    await runner.run("Hello")

    headers = route.calls.last.request.headers
    assert headers["x-api-key"] == "test-anthropic-key"
    assert headers["anthropic-version"] == "2023-06-01"
    assert headers["content-type"] == "application/json"


async def test_claude_sends_correct_body(anthropic_env, respx_mock):
    route = respx_mock.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(200, json=claude_response_body("Hi"))
    )
    runner = ClaudeRunner(model="claude-haiku-4-5-20251001", max_tokens=256)
    await runner.run("Test prompt")

    payload = json.loads(route.calls.last.request.read())
    assert payload["model"] == "claude-haiku-4-5-20251001"
    assert payload["max_tokens"] == 256
    assert payload["messages"] == [{"role": "user", "content": "Test prompt"}]


async def test_claude_includes_system_prompt(anthropic_env, respx_mock):
    route = respx_mock.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(200, json=claude_response_body("Hi"))
    )
    runner = ClaudeRunner(system="You are helpful.")
    await runner.run("Hello")

    payload = json.loads(route.calls.last.request.read())
    assert payload["system"] == "You are helpful."


async def test_claude_omits_system_when_none(anthropic_env, respx_mock):
    route = respx_mock.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(200, json=claude_response_body("Hi"))
    )
    runner = ClaudeRunner()
    await runner.run("Hello")

    payload = json.loads(route.calls.last.request.read())
    assert "system" not in payload


async def test_claude_retries_on_429(anthropic_env, respx_mock, monkeypatch):
    """ClaudeRunner must retry on 429 and succeed on the second attempt."""
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    respx_mock.post(ANTHROPIC_URL).mock(
        side_effect=[
            httpx.Response(429, json={"error": {"type": "rate_limit_error"}}),
            httpx.Response(200, json=claude_response_body("Retried!")),
        ]
    )
    runner = ClaudeRunner(model="claude-haiku-4-5-20251001")
    result = await runner.run("Hello")

    assert result.response == "Retried!"
    assert respx_mock.calls.call_count == 2


async def test_claude_retries_on_529(anthropic_env, respx_mock, monkeypatch):
    """ClaudeRunner must also retry on Anthropic's 529 overloaded status."""
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    respx_mock.post(ANTHROPIC_URL).mock(
        side_effect=[
            httpx.Response(529, json={"error": {"type": "overloaded_error"}}),
            httpx.Response(200, json=claude_response_body("Recovered!")),
        ]
    )
    runner = ClaudeRunner()
    result = await runner.run("Hello")

    assert result.response == "Recovered!"


# ---------------------------------------------------------------------------
# OpenAIRunner
# ---------------------------------------------------------------------------


async def test_openai_run_returns_result(openai_env, respx_mock):
    respx_mock.post(OPENAI_URL).mock(
        return_value=httpx.Response(200, json=openai_response_body("Hello!"))
    )
    runner = OpenAIRunner(model="gpt-4o")
    result = await runner.run("Say hello.")

    assert result.response == "Hello!"
    assert result.model == "gpt-4o"
    assert result.input_tokens == 10
    assert result.output_tokens == 15
    # 10 input @ $2.50/M + 15 output @ $10.00/M = $0.000175
    assert result.cost_usd == pytest.approx(0.000175)


async def test_openai_sends_bearer_auth(openai_env, respx_mock):
    route = respx_mock.post(OPENAI_URL).mock(
        return_value=httpx.Response(200, json=openai_response_body("Hi"))
    )
    runner = OpenAIRunner()
    await runner.run("Hello")

    assert route.calls.last.request.headers["authorization"] == "Bearer test-openai-key"


async def test_openai_sends_correct_body(openai_env, respx_mock):
    route = respx_mock.post(OPENAI_URL).mock(
        return_value=httpx.Response(200, json=openai_response_body("Hi"))
    )
    runner = OpenAIRunner(model="gpt-4o", max_tokens=512)
    await runner.run("Test prompt")

    payload = json.loads(route.calls.last.request.read())
    assert payload["model"] == "gpt-4o"
    assert payload["max_tokens"] == 512
    assert payload["messages"][-1] == {"role": "user", "content": "Test prompt"}


async def test_openai_includes_system_message(openai_env, respx_mock):
    route = respx_mock.post(OPENAI_URL).mock(
        return_value=httpx.Response(200, json=openai_response_body("Hi"))
    )
    runner = OpenAIRunner(system="Be concise.")
    await runner.run("Hello")

    payload = json.loads(route.calls.last.request.read())
    assert payload["messages"][0] == {"role": "system", "content": "Be concise."}
    assert payload["messages"][1]["role"] == "user"


async def test_openai_omits_system_when_none(openai_env, respx_mock):
    route = respx_mock.post(OPENAI_URL).mock(
        return_value=httpx.Response(200, json=openai_response_body("Hi"))
    )
    runner = OpenAIRunner()
    await runner.run("Hello")

    payload = json.loads(route.calls.last.request.read())
    assert all(m["role"] != "system" for m in payload["messages"])


async def test_openai_retries_on_429(openai_env, respx_mock, monkeypatch):
    """OpenAIRunner must retry on 429 and succeed on the second attempt."""
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    respx_mock.post(OPENAI_URL).mock(
        side_effect=[
            httpx.Response(429, json={"error": {"type": "rate_limit_error"}}),
            httpx.Response(200, json=openai_response_body("Retried!")),
        ]
    )
    runner = OpenAIRunner(model="gpt-4o")
    result = await runner.run("Hello")

    assert result.response == "Retried!"
    assert respx_mock.calls.call_count == 2


# ---------------------------------------------------------------------------
# GeminiRunner
# ---------------------------------------------------------------------------


async def test_gemini_run_returns_result(google_env, respx_mock):
    respx_mock.post(GEMINI_URL).mock(
        return_value=httpx.Response(200, json=gemini_response_body("Hello!"))
    )
    runner = GeminiRunner(model="gemini-flash-latest")
    result = await runner.run("Say hello.")

    assert result.response == "Hello!"
    assert result.model == "gemini-flash-latest"
    assert result.input_tokens == 10
    assert result.output_tokens == 15
    # 10 input @ $0.075/M + 15 output @ $0.30/M = $0.00000525
    assert result.cost_usd == pytest.approx(0.00000525)


async def test_gemini_sends_api_key_header(google_env, respx_mock):
    route = respx_mock.post(GEMINI_URL).mock(
        return_value=httpx.Response(200, json=gemini_response_body("Hi"))
    )
    runner = GeminiRunner()
    await runner.run("Hello")

    assert route.calls.last.request.headers["x-goog-api-key"] == "test-google-key"


async def test_gemini_sends_correct_body(google_env, respx_mock):
    route = respx_mock.post(GEMINI_URL).mock(
        return_value=httpx.Response(200, json=gemini_response_body("Hi"))
    )
    runner = GeminiRunner(model="gemini-flash-latest", max_tokens=128)
    await runner.run("Test prompt")

    payload = json.loads(route.calls.last.request.read())
    assert payload["contents"][0]["parts"][0]["text"] == "Test prompt"
    assert payload["generationConfig"]["maxOutputTokens"] == 128


async def test_gemini_includes_system_instruction(google_env, respx_mock):
    route = respx_mock.post(GEMINI_URL).mock(
        return_value=httpx.Response(200, json=gemini_response_body("Hi"))
    )
    runner = GeminiRunner(system="Be brief.")
    await runner.run("Hello")

    payload = json.loads(route.calls.last.request.read())
    assert payload["systemInstruction"]["parts"][0]["text"] == "Be brief."


async def test_gemini_retries_on_429(google_env, respx_mock, monkeypatch):
    """GeminiRunner must retry on 429 and succeed on the second attempt."""
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    respx_mock.post(GEMINI_URL).mock(
        side_effect=[
            httpx.Response(429, json={"error": {"code": 429, "message": "Rate limited"}}),
            httpx.Response(200, json=gemini_response_body("Retried!")),
        ]
    )
    runner = GeminiRunner()
    result = await runner.run("Hello")

    assert result.response == "Retried!"
    assert respx_mock.calls.call_count == 2


async def test_gemini_raises_after_max_retries(google_env, respx_mock, monkeypatch):
    """GeminiRunner must raise after exhausting all retry attempts."""
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    respx_mock.post(GEMINI_URL).mock(
        return_value=httpx.Response(429, json={"error": {"code": 429}})
    )
    runner = GeminiRunner()
    with pytest.raises(httpx.HTTPStatusError):
        await runner.run("Hello")
