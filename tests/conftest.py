"""Shared fixtures for the test suite."""
import pytest


@pytest.fixture
def anthropic_env(monkeypatch):
    """Set a fake Anthropic API key so evaluators and ClaudeRunner can be instantiated."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")


@pytest.fixture
def google_env(monkeypatch):
    """Set a fake Google API key so GeminiRunner can be instantiated."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")


@pytest.fixture
def openai_env(monkeypatch):
    """Set a fake OpenAI API key and clear any local base URL override."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)


def claude_response_body(text: str, model: str = "claude-haiku-4-5-20251001") -> dict:
    """Build a minimal Anthropic Messages API response body."""
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }


def openai_response_body(text: str, model: str = "gpt-4o") -> dict:
    """Build a minimal OpenAI Chat Completions API response body."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
    }


def gemini_response_body(text: str) -> dict:
    """Build a minimal Gemini generateContent API response body."""
    return {
        "candidates": [{"content": {"role": "model", "parts": [{"text": text}]}}],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 15, "totalTokenCount": 25},
    }
