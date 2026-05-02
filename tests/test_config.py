"""Tests for config validation, runner factory, and evaluator factory."""
import pytest
from pydantic import ValidationError

from run_eval import EvalConfig, RunnerConfig, _build_evaluators, _build_runner


# ---------------------------------------------------------------------------
# EvalConfig validation
# ---------------------------------------------------------------------------


def test_eval_config_valid():
    cfg = EvalConfig.model_validate(
        {
            "name": "test_run",
            "prompts": [{"id": "q1", "text": "What is X?"}],
            "runners": [{"type": "claude", "model": "claude-haiku-4-5-20251001"}],
            "evaluators": ["coherence"],
        }
    )
    assert cfg.name == "test_run"
    assert cfg.output_dir == "reports/"


def test_eval_config_prompt_context_optional():
    cfg = EvalConfig.model_validate(
        {
            "name": "test",
            "prompts": [{"id": "q1", "text": "Hello", "context": None}],
            "runners": [{"type": "claude", "model": "x"}],
            "evaluators": [],
        }
    )
    assert cfg.prompts[0].context is None


def test_eval_config_missing_name_raises():
    with pytest.raises(ValidationError):
        EvalConfig.model_validate(
            {
                "prompts": [{"id": "q1", "text": "Hello"}],
                "runners": [],
                "evaluators": [],
            }
        )


def test_eval_config_missing_prompts_and_dataset_raises():
    """Both prompts and dataset absent must raise a ValidationError."""
    with pytest.raises(ValidationError):
        EvalConfig.model_validate({"name": "x", "runners": [], "evaluators": []})


def test_runner_config_defaults():
    cfg = RunnerConfig.model_validate({"type": "claude", "model": "claude-sonnet-4-6"})
    assert cfg.max_tokens == 1024
    assert cfg.system is None


# ---------------------------------------------------------------------------
# _build_runner factory
# ---------------------------------------------------------------------------


def test_build_runner_claude(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
    runner = _build_runner(RunnerConfig(type="claude", model="claude-haiku-4-5-20251001"))
    from runners.claude_runner import ClaudeRunner
    assert isinstance(runner, ClaudeRunner)


def test_build_runner_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    runner = _build_runner(RunnerConfig(type="openai", model="gpt-4o"))
    from runners.openai_runner import OpenAIRunner
    assert isinstance(runner, OpenAIRunner)


def test_build_runner_gemini(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "key")
    runner = _build_runner(RunnerConfig(type="gemini", model="gemini-flash-latest"))
    from runners.gemini_runner import GeminiRunner
    assert isinstance(runner, GeminiRunner)


def test_build_runner_unknown_raises():
    with pytest.raises(ValueError, match="Unknown runner type"):
        _build_runner(RunnerConfig(type="unknown_provider", model="x"))


# ---------------------------------------------------------------------------
# _build_evaluators factory
# ---------------------------------------------------------------------------


def test_build_evaluators_all_known(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
    evaluators = _build_evaluators(["coherence", "relevance", "faithfulness"])
    assert set(evaluators.keys()) == {"coherence", "relevance", "faithfulness"}


def test_build_evaluators_unknown_raises(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
    with pytest.raises(ValueError, match="Unknown evaluator"):
        _build_evaluators(["coherence", "nonexistent"])


def test_build_evaluators_empty_list(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
    assert _build_evaluators([]) == {}
