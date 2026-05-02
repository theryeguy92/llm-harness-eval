"""Tests for evaluator versioning and its presence in report output."""
import json

import pytest

from evaluators.coherence import CoherenceEvaluator
from evaluators.faithfulness import FaithfulnessEvaluator
from evaluators.relevance import RelevanceEvaluator
from run_eval import EvalConfig, EvalReport, EvaluatorInfo, PromptConfig, RunnerConfig


# ---------------------------------------------------------------------------
# Class-level constants
# ---------------------------------------------------------------------------


def test_coherence_evaluator_has_name_and_version():
    assert CoherenceEvaluator.NAME == "coherence"
    assert CoherenceEvaluator.PROMPT_VERSION == "v1"


def test_relevance_evaluator_has_name_and_version():
    assert RelevanceEvaluator.NAME == "relevance"
    assert RelevanceEvaluator.PROMPT_VERSION == "v1"


def test_faithfulness_evaluator_has_name_and_version():
    assert FaithfulnessEvaluator.NAME == "faithfulness"
    assert FaithfulnessEvaluator.PROMPT_VERSION == "v1"


def test_instance_inherits_class_constants(anthropic_env):
    ev = CoherenceEvaluator()
    assert ev.NAME == "coherence"
    assert ev.PROMPT_VERSION == "v1"


# ---------------------------------------------------------------------------
# EvalReport JSON serialization
# ---------------------------------------------------------------------------


def _minimal_config(evaluator_names: list[str]) -> EvalConfig:
    return EvalConfig(
        name="test-run",
        prompts=[PromptConfig(id="q1", text="hello")],
        runners=[],
        evaluators=evaluator_names,
    )


def test_evaluator_versions_in_report_json():
    report = EvalReport(
        name="test-run",
        timestamp="2026-01-01T00:00:00+00:00",
        config=_minimal_config(["coherence", "relevance", "faithfulness"]),
        results=[],
        evaluator_versions={
            "coherence": EvaluatorInfo(name="coherence", prompt_version="v1"),
            "relevance": EvaluatorInfo(name="relevance", prompt_version="v1"),
            "faithfulness": EvaluatorInfo(name="faithfulness", prompt_version="v1"),
        },
    )

    data = json.loads(report.model_dump_json())
    ev = data["evaluator_versions"]

    assert ev["coherence"]["name"] == "coherence"
    assert ev["coherence"]["prompt_version"] == "v1"
    assert ev["relevance"]["name"] == "relevance"
    assert ev["relevance"]["prompt_version"] == "v1"
    assert ev["faithfulness"]["name"] == "faithfulness"
    assert ev["faithfulness"]["prompt_version"] == "v1"


def test_evaluator_versions_key_present_even_when_empty():
    report = EvalReport(
        name="test-run",
        timestamp="2026-01-01T00:00:00+00:00",
        config=_minimal_config([]),
        results=[],
    )
    data = json.loads(report.model_dump_json())
    assert "evaluator_versions" in data
    assert data["evaluator_versions"] == {}


def test_prompt_version_survives_round_trip():
    original = EvaluatorInfo(name="coherence", prompt_version="v1")
    reloaded = EvaluatorInfo.model_validate_json(original.model_dump_json())
    assert reloaded.name == "coherence"
    assert reloaded.prompt_version == "v1"
