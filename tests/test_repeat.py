"""Tests for _bootstrap_ci and aggregate_repeated_runs."""
import numpy as np
import pytest

from evaluators.base import EvalResult
from run_eval import AggregatedRow, ResultRow, ScoreStats, _bootstrap_ci, aggregate_repeated_runs


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
# _bootstrap_ci
# ---------------------------------------------------------------------------


def test_bootstrap_ci_single_value_returns_point():
    lo, hi = _bootstrap_ci([0.8])
    assert lo == pytest.approx(0.8)
    assert hi == pytest.approx(0.8)


def test_bootstrap_ci_identical_values():
    rng = np.random.default_rng(0)
    lo, hi = _bootstrap_ci([0.7] * 20, rng=rng)
    assert lo == pytest.approx(0.7)
    assert hi == pytest.approx(0.7)


def test_bootstrap_ci_lower_le_mean_le_upper():
    scores = [0.5, 0.6, 0.7, 0.8, 0.9]
    rng = np.random.default_rng(42)
    lo, hi = _bootstrap_ci(scores, rng=rng)
    mean = sum(scores) / len(scores)
    assert lo <= mean <= hi


def test_bootstrap_ci_within_data_range():
    scores = [0.2, 0.4, 0.6, 0.8]
    rng = np.random.default_rng(42)
    lo, hi = _bootstrap_ci(scores, rng=rng)
    assert 0.2 <= lo
    assert hi <= 0.8


def test_bootstrap_ci_wider_with_higher_variance():
    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(7)
    lo_tight, hi_tight = _bootstrap_ci([0.75, 0.75, 0.75, 0.75], rng=rng_a)
    lo_wide, hi_wide = _bootstrap_ci([0.0, 0.5, 1.0, 0.25], rng=rng_b)
    assert (hi_wide - lo_wide) > (hi_tight - lo_tight)


# ---------------------------------------------------------------------------
# aggregate_repeated_runs — empty / single run
# ---------------------------------------------------------------------------


def test_aggregate_empty_returns_empty():
    assert aggregate_repeated_runs([], ["coherence"]) == []


def test_aggregate_single_run_n1():
    rows = [_row("model-a", "claude", "p1", 100.0, 0.01, coherence=0.8)]
    rng = np.random.default_rng(0)
    agg = aggregate_repeated_runs(rows, ["coherence"], rng=rng)

    assert len(agg) == 1
    s = agg[0].scores["coherence"]
    assert agg[0].n_runs == 1
    assert s.n == 1
    assert s.mean == pytest.approx(0.8)
    assert s.std == pytest.approx(0.0)
    assert s.ci_lower == pytest.approx(0.8)
    assert s.ci_upper == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# aggregate_repeated_runs — mean and std
# ---------------------------------------------------------------------------


def test_aggregate_mean_two_runs():
    rows = [
        _row("model-a", "claude", "p1", 100.0, 0.01, coherence=0.6),
        _row("model-a", "claude", "p1", 200.0, 0.02, coherence=0.8),
    ]
    rng = np.random.default_rng(0)
    agg = aggregate_repeated_runs(rows, ["coherence"], rng=rng)
    s = agg[0].scores["coherence"]
    assert s.mean == pytest.approx(0.7)
    assert s.n == 2


def test_aggregate_std_uses_sample_ddof1():
    scores_list = [0.6, 0.8]
    rows = [
        _row("model-a", "claude", "p1", 100.0, 0.01, coherence=scores_list[0]),
        _row("model-a", "claude", "p1", 200.0, 0.02, coherence=scores_list[1]),
    ]
    rng = np.random.default_rng(0)
    agg = aggregate_repeated_runs(rows, ["coherence"], rng=rng)
    expected_std = float(np.std(scores_list, ddof=1))
    assert agg[0].scores["coherence"].std == pytest.approx(expected_std, abs=1e-4)


def test_aggregate_ci_contains_mean():
    rows = [
        _row("model-a", "claude", "p1", 100.0, 0.01, coherence=0.5),
        _row("model-a", "claude", "p1", 100.0, 0.01, coherence=0.7),
        _row("model-a", "claude", "p1", 100.0, 0.01, coherence=0.9),
    ]
    rng = np.random.default_rng(0)
    agg = aggregate_repeated_runs(rows, ["coherence"], rng=rng)
    s = agg[0].scores["coherence"]
    assert s.ci_lower <= s.mean <= s.ci_upper


# ---------------------------------------------------------------------------
# aggregate_repeated_runs — latency and cost
# ---------------------------------------------------------------------------


def test_aggregate_avg_latency():
    rows = [
        _row("model-a", "claude", "p1", 100.0, 0.01, coherence=0.5),
        _row("model-a", "claude", "p1", 300.0, 0.01, coherence=0.5),
    ]
    agg = aggregate_repeated_runs(rows, ["coherence"])
    assert agg[0].avg_latency_ms == pytest.approx(200.0)


def test_aggregate_total_and_avg_cost():
    rows = [
        _row("model-a", "claude", "p1", 100.0, 0.01, coherence=0.5),
        _row("model-a", "claude", "p1", 100.0, 0.03, coherence=0.5),
    ]
    agg = aggregate_repeated_runs(rows, ["coherence"])
    assert agg[0].total_cost_usd == pytest.approx(0.04)
    assert agg[0].avg_cost_usd == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# aggregate_repeated_runs — grouping
# ---------------------------------------------------------------------------


def test_aggregate_groups_by_prompt_and_model():
    rows = [
        _row("model-a", "claude", "p1", 100.0, 0.01, coherence=0.8),
        _row("model-a", "claude", "p1", 110.0, 0.01, coherence=0.6),  # repeat of p1/model-a
        _row("model-b", "gemini", "p1", 200.0, 0.02, coherence=0.7),  # different model
        _row("model-a", "claude", "p2", 120.0, 0.01, coherence=0.9),  # different prompt
    ]
    rng = np.random.default_rng(0)
    agg = aggregate_repeated_runs(rows, ["coherence"], rng=rng)

    assert len(agg) == 3
    groups = {(r.prompt_id, r.model): r for r in agg}

    assert groups[("p1", "model-a")].n_runs == 2
    assert groups[("p1", "model-a")].scores["coherence"].mean == pytest.approx(0.7)
    assert groups[("p1", "model-b")].n_runs == 1
    assert groups[("p2", "model-a")].n_runs == 1


def test_aggregate_multiple_metrics_independent():
    rows = [
        _row("model-a", "claude", "p1", 100.0, 0.01, coherence=0.6, relevance=0.9),
        _row("model-a", "claude", "p1", 100.0, 0.01, coherence=0.8, relevance=0.5),
    ]
    rng = np.random.default_rng(0)
    agg = aggregate_repeated_runs(rows, ["coherence", "relevance"], rng=rng)

    assert agg[0].scores["coherence"].mean == pytest.approx(0.7)
    assert agg[0].scores["relevance"].mean == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# ScoreStats model validation
# ---------------------------------------------------------------------------


def test_score_stats_round_trip():
    s = ScoreStats(mean=0.75, std=0.05, ci_lower=0.65, ci_upper=0.85, n=10)
    assert s.mean == 0.75
    assert s.n == 10


def test_aggregate_same_seed_reproducible_ci():
    rows = [
        _row("m1", "claude", "p1", 100.0, 0.001, coherence=s)
        for s in (0.6, 0.7, 0.8, 0.9)
    ]
    a = aggregate_repeated_runs(rows, ["coherence"], rng=np.random.default_rng(42))
    b = aggregate_repeated_runs(rows, ["coherence"], rng=np.random.default_rng(42))
    assert a[0].scores["coherence"].ci_lower == b[0].scores["coherence"].ci_lower
    assert a[0].scores["coherence"].ci_upper == b[0].scores["coherence"].ci_upper
