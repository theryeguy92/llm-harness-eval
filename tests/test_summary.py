"""Tests for compute_summary() aggregation logic."""
import pytest

from evaluators.base import EvalResult
from run_eval import ResultRow, compute_summary


def _row(
    model: str,
    runner_type: str,
    prompt_id: str,
    latency_ms: float,
    cost_usd: float,
    **scores: float,
) -> ResultRow:
    return ResultRow(
        prompt_id=prompt_id,
        prompt=f"prompt {prompt_id}",
        context=None,
        runner_type=runner_type,
        model=model,
        response="response",
        latency_ms=latency_ms,
        input_tokens=10,
        output_tokens=10,
        cost_usd=cost_usd,
        scores={k: EvalResult(score=v, explanation="") for k, v in scores.items()},
    )


# ---------------------------------------------------------------------------
# Empty / single-model edge cases
# ---------------------------------------------------------------------------


def test_empty_results_returns_empty_dict():
    assert compute_summary([], ["coherence"]) == {}


def test_single_model_single_prompt():
    rows = [_row("gpt-4o", "openai", "p1", 100.0, 0.01, coherence=0.8)]
    summary = compute_summary(rows, ["coherence"])

    assert "gpt-4o" in summary
    s = summary["gpt-4o"]
    assert s.runner_type == "openai"
    assert s.avg_latency_ms == pytest.approx(100.0)
    assert s.total_cost_usd == pytest.approx(0.01)
    assert s.metrics["coherence"].mean_score == pytest.approx(0.8)
    # sole model — wins every prompt by default
    assert s.metrics["coherence"].win_rate == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Mean score
# ---------------------------------------------------------------------------


def test_mean_score_two_prompts():
    rows = [
        _row("model-a", "claude", "p1", 100, 0.01, coherence=0.6),
        _row("model-a", "claude", "p2", 100, 0.01, coherence=0.8),
    ]
    summary = compute_summary(rows, ["coherence"])
    assert summary["model-a"].metrics["coherence"].mean_score == pytest.approx(0.7)


def test_mean_scores_are_independent_per_metric():
    rows = [
        _row("model-a", "claude", "p1", 100, 0.01, coherence=1.0, relevance=0.0),
        _row("model-b", "gemini", "p1", 200, 0.02, coherence=0.0, relevance=1.0),
    ]
    summary = compute_summary(rows, ["coherence", "relevance"])
    assert summary["model-a"].metrics["coherence"].mean_score == pytest.approx(1.0)
    assert summary["model-a"].metrics["relevance"].mean_score == pytest.approx(0.0)
    assert summary["model-b"].metrics["coherence"].mean_score == pytest.approx(0.0)
    assert summary["model-b"].metrics["relevance"].mean_score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Win rate
# ---------------------------------------------------------------------------


def test_win_rate_clear_winner_each_prompt():
    # model-a wins p1, model-b wins p2 → each has win_rate = 0.5
    rows = [
        _row("model-a", "claude", "p1", 100, 0.01, coherence=0.9),
        _row("model-b", "gemini", "p1", 200, 0.02, coherence=0.5),
        _row("model-a", "claude", "p2", 100, 0.01, coherence=0.4),
        _row("model-b", "gemini", "p2", 200, 0.02, coherence=0.8),
    ]
    summary = compute_summary(rows, ["coherence"])
    assert summary["model-a"].metrics["coherence"].win_rate == pytest.approx(0.5)
    assert summary["model-b"].metrics["coherence"].win_rate == pytest.approx(0.5)


def test_win_rate_one_model_dominates():
    rows = [
        _row("model-a", "claude", "p1", 100, 0.01, coherence=0.9),
        _row("model-b", "gemini", "p1", 200, 0.02, coherence=0.5),
        _row("model-a", "claude", "p2", 100, 0.01, coherence=0.8),
        _row("model-b", "gemini", "p2", 200, 0.02, coherence=0.3),
    ]
    summary = compute_summary(rows, ["coherence"])
    assert summary["model-a"].metrics["coherence"].win_rate == pytest.approx(1.0)
    assert summary["model-b"].metrics["coherence"].win_rate == pytest.approx(0.0)


def test_tie_gives_no_win_to_either_model():
    rows = [
        _row("model-a", "claude", "p1", 100, 0.01, coherence=0.7),
        _row("model-b", "gemini", "p1", 200, 0.02, coherence=0.7),
    ]
    summary = compute_summary(rows, ["coherence"])
    assert summary["model-a"].metrics["coherence"].win_rate == pytest.approx(0.0)
    assert summary["model-b"].metrics["coherence"].win_rate == pytest.approx(0.0)


def test_win_rates_computed_independently_per_metric():
    # model-a wins coherence on p1, model-b wins relevance on p1
    rows = [
        _row("model-a", "claude", "p1", 100, 0.01, coherence=0.9, relevance=0.3),
        _row("model-b", "gemini", "p1", 200, 0.02, coherence=0.4, relevance=0.8),
    ]
    summary = compute_summary(rows, ["coherence", "relevance"])
    assert summary["model-a"].metrics["coherence"].win_rate == pytest.approx(1.0)
    assert summary["model-a"].metrics["relevance"].win_rate == pytest.approx(0.0)
    assert summary["model-b"].metrics["coherence"].win_rate == pytest.approx(0.0)
    assert summary["model-b"].metrics["relevance"].win_rate == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Latency and cost
# ---------------------------------------------------------------------------


def test_avg_latency_across_prompts():
    rows = [
        _row("model-a", "claude", "p1", 100.0, 0.01, coherence=0.5),
        _row("model-a", "claude", "p2", 300.0, 0.01, coherence=0.5),
    ]
    summary = compute_summary(rows, ["coherence"])
    assert summary["model-a"].avg_latency_ms == pytest.approx(200.0)


def test_total_cost_summed_across_prompts():
    rows = [
        _row("model-a", "claude", "p1", 100, 0.005, coherence=0.5),
        _row("model-a", "claude", "p2", 100, 0.003, coherence=0.5),
    ]
    summary = compute_summary(rows, ["coherence"])
    assert summary["model-a"].total_cost_usd == pytest.approx(0.008)


def test_latency_and_cost_tracked_per_model():
    rows = [
        _row("model-a", "claude", "p1", 100.0, 0.01, coherence=0.5),
        _row("model-b", "gemini", "p1", 500.0, 0.001, coherence=0.5),
    ]
    summary = compute_summary(rows, ["coherence"])
    assert summary["model-a"].avg_latency_ms == pytest.approx(100.0)
    assert summary["model-b"].avg_latency_ms == pytest.approx(500.0)
    assert summary["model-a"].total_cost_usd == pytest.approx(0.01)
    assert summary["model-b"].total_cost_usd == pytest.approx(0.001)


# ---------------------------------------------------------------------------
# Multi-metric, multi-model, multi-prompt integration
# ---------------------------------------------------------------------------


def test_full_two_model_two_prompt_two_metric():
    rows = [
        _row("model-a", "claude",  "p1", 100, 0.01, coherence=0.9, relevance=0.4),
        _row("model-b", "gemini",  "p1", 200, 0.02, coherence=0.6, relevance=0.8),
        _row("model-a", "claude",  "p2", 120, 0.01, coherence=0.7, relevance=0.9),
        _row("model-b", "gemini",  "p2", 220, 0.02, coherence=0.5, relevance=0.3),
    ]
    summary = compute_summary(rows, ["coherence", "relevance"])

    a = summary["model-a"]
    b = summary["model-b"]

    # mean scores
    assert a.metrics["coherence"].mean_score == pytest.approx(0.8)   # (0.9+0.7)/2
    assert b.metrics["coherence"].mean_score == pytest.approx(0.55)  # (0.6+0.5)/2
    assert a.metrics["relevance"].mean_score == pytest.approx(0.65)  # (0.4+0.9)/2
    assert b.metrics["relevance"].mean_score == pytest.approx(0.55)  # (0.8+0.3)/2

    # win rates: model-a wins coherence on both → 1.0; model-b wins relevance on p1 → 0.5
    assert a.metrics["coherence"].win_rate == pytest.approx(1.0)
    assert b.metrics["coherence"].win_rate == pytest.approx(0.0)
    assert a.metrics["relevance"].win_rate == pytest.approx(0.5)   # wins p2
    assert b.metrics["relevance"].win_rate == pytest.approx(0.5)   # wins p1

    # cost and latency
    assert a.total_cost_usd == pytest.approx(0.02)
    assert b.total_cost_usd == pytest.approx(0.04)
    assert a.avg_latency_ms == pytest.approx(110.0)
    assert b.avg_latency_ms == pytest.approx(210.0)
