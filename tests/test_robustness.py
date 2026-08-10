"""Tests for failure tolerance, judge retry, and repeat-averaged win rates."""
import httpx
import pytest

import run_eval
from evaluators.base import BaseEvaluator, EvalResult, call_judge
from run_eval import EvalConfig, ResultRow, _run_eval, compute_summary
from runners.base import BaseRunner, RunResult
from tests.conftest import claude_response_body

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"


def _row(model: str, prompt_id: str, **scores: float) -> ResultRow:
    return ResultRow(
        prompt_id=prompt_id,
        prompt="p",
        context=None,
        runner_type="claude",
        model=model,
        response="r",
        latency_ms=1.0,
        input_tokens=1,
        output_tokens=1,
        cost_usd=0.0,
        scores={k: EvalResult(score=v, explanation="") for k, v in scores.items()},
    )


# ---------------------------------------------------------------------------
# call_judge retry
# ---------------------------------------------------------------------------


async def test_call_judge_retries_on_429_then_succeeds(respx_mock):
    route = respx_mock.post(ANTHROPIC_URL).mock(
        side_effect=[
            httpx.Response(429),
            httpx.Response(200, json=claude_response_body('{"score": 0.9, "explanation": "ok"}')),
        ]
    )
    text = await call_judge("key", "model", "system", "user")
    assert '"score": 0.9' in text
    assert route.call_count == 2


# ---------------------------------------------------------------------------
# _run_eval failure tolerance
# ---------------------------------------------------------------------------


class _OkRunner(BaseRunner):
    async def run(self, prompt: str) -> RunResult:
        return RunResult(
            model="ok-model", latency_ms=1.0, input_tokens=1,
            output_tokens=1, response="yes", cost_usd=0.0,
        )


class _BoomRunner(BaseRunner):
    async def run(self, prompt: str) -> RunResult:
        raise RuntimeError("boom")


def _config(runner_models: list[str], evaluators: list[str]) -> EvalConfig:
    return EvalConfig.model_validate({
        "name": "t",
        "prompts": [{"id": "p1", "text": "q", "expected_output": "yes"}],
        "runners": [{"type": "claude", "model": m} for m in runner_models],
        "evaluators": evaluators,
    })


async def test_run_eval_records_failures_instead_of_aborting(monkeypatch):
    def fake_build(cfg):
        return _BoomRunner() if cfg.model == "boom" else _OkRunner()

    monkeypatch.setattr(run_eval, "_build_runner", fake_build)
    report = await _run_eval(_config(["ok", "boom"], ["exact_match"]))

    assert len(report.results) == 1
    assert report.results[0].model == "ok-model"
    assert len(report.failures) == 1
    assert "boom" in report.failures[0]


class _UnparseableEvaluator(BaseEvaluator):
    NAME = "stub"
    PROMPT_VERSION = "v1"

    async def score(self, prompt, response, context=None):
        return EvalResult(score=0.5, explanation="x", parse_failed=True)


async def test_run_eval_counts_judge_parse_failures(monkeypatch):
    monkeypatch.setattr(run_eval, "_build_runner", lambda cfg: _OkRunner())
    monkeypatch.setitem(run_eval._EVALUATOR_MAP, "stub", _UnparseableEvaluator)
    report = await _run_eval(_config(["ok"], ["stub"]))
    assert report.judge_parse_failures == 1


# ---------------------------------------------------------------------------
# Win-rate averages repeated runs before comparing
# ---------------------------------------------------------------------------


def test_win_rate_averages_repeats_before_comparing():
    rows = [
        _row("a", "p1", coherence=0.9),
        _row("a", "p1", coherence=0.1),  # mean 0.5
        _row("b", "p1", coherence=0.6),
        _row("b", "p1", coherence=0.6),  # mean 0.6
    ]
    summary = compute_summary(rows, ["coherence"])
    assert summary["b"].metrics["coherence"].win_rate == 1.0
    assert summary["a"].metrics["coherence"].win_rate == 0.0
