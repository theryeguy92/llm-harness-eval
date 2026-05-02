"""Tests for dataset loading (JSONL and CSV) and EvalConfig.dataset field."""
import json
import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from run_eval import EvalConfig, PromptConfig, load_dataset


# ---------------------------------------------------------------------------
# load_dataset — JSONL
# ---------------------------------------------------------------------------


def test_load_jsonl_required_fields(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_text('{"id": "q1", "input": "What is X?"}\n')
    prompts = load_dataset(str(p))
    assert len(prompts) == 1
    assert prompts[0].id == "q1"
    assert prompts[0].text == "What is X?"
    assert prompts[0].context is None
    assert prompts[0].expected_output is None


def test_load_jsonl_all_fields(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_text(
        '{"id": "q1", "input": "Q", "context": "doc text", "expected_output": "answer"}\n'
    )
    prompts = load_dataset(str(p))
    assert prompts[0].context == "doc text"
    assert prompts[0].expected_output == "answer"


def test_load_jsonl_multiple_rows(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_text(
        '{"id": "a", "input": "Q1"}\n'
        '{"id": "b", "input": "Q2"}\n'
        '{"id": "c", "input": "Q3"}\n'
    )
    prompts = load_dataset(str(p))
    assert len(prompts) == 3
    assert [pr.id for pr in prompts] == ["a", "b", "c"]


def test_load_jsonl_skips_blank_lines(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_text('\n{"id": "q1", "input": "Q"}\n\n')
    prompts = load_dataset(str(p))
    assert len(prompts) == 1


def test_load_jsonl_missing_required_column_raises(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_text('{"id": "q1"}\n')  # missing 'input'
    with pytest.raises(ValueError, match="missing required column"):
        load_dataset(str(p))


# ---------------------------------------------------------------------------
# load_dataset — CSV
# ---------------------------------------------------------------------------


def test_load_csv_required_fields(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("id,input\nq1,What is X?\n")
    prompts = load_dataset(str(p))
    assert len(prompts) == 1
    assert prompts[0].id == "q1"
    assert prompts[0].text == "What is X?"


def test_load_csv_all_columns(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("id,input,context,expected_output\nq1,Q,doc,answer\n")
    prompts = load_dataset(str(p))
    assert prompts[0].context == "doc"
    assert prompts[0].expected_output == "answer"


def test_load_csv_empty_optional_columns_become_none(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("id,input,context,expected_output\nq1,Q,,\n")
    prompts = load_dataset(str(p))
    assert prompts[0].context is None
    assert prompts[0].expected_output is None


def test_load_csv_multiple_rows(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("id,input\na,Q1\nb,Q2\n")
    prompts = load_dataset(str(p))
    assert len(prompts) == 2


# ---------------------------------------------------------------------------
# load_dataset — error cases
# ---------------------------------------------------------------------------


def test_load_dataset_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_dataset("/nonexistent/path/data.jsonl")


def test_load_dataset_unsupported_extension(tmp_path):
    p = tmp_path / "data.txt"
    p.write_text("hello")
    with pytest.raises(ValueError, match="Unsupported dataset format"):
        load_dataset(str(p))


def test_load_dataset_relative_path_resolved_from_config_dir(tmp_path):
    data = tmp_path / "prompts.jsonl"
    data.write_text('{"id": "q1", "input": "Q"}\n')
    prompts = load_dataset("prompts.jsonl", config_dir=tmp_path)
    assert len(prompts) == 1


# ---------------------------------------------------------------------------
# EvalConfig — dataset field wiring
# ---------------------------------------------------------------------------


def test_eval_config_accepts_dataset_field(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_text('{"id": "q1", "input": "Q"}\n')
    cfg = EvalConfig.model_validate(
        {
            "name": "test",
            "dataset": str(p),
            "runners": [{"type": "claude", "model": "claude-haiku-4-5-20251001"}],
            "evaluators": [],
        }
    )
    assert cfg.dataset == str(p)
    assert cfg.prompts == []


def test_eval_config_requires_prompts_or_dataset():
    with pytest.raises(ValidationError, match="prompts.*dataset|dataset.*prompts"):
        EvalConfig.model_validate(
            {"name": "x", "runners": [{"type": "claude", "model": "m"}], "evaluators": []}
        )


def test_eval_config_inline_prompts_still_work():
    cfg = EvalConfig.model_validate(
        {
            "name": "test",
            "prompts": [{"id": "q1", "text": "Hello"}],
            "runners": [{"type": "claude", "model": "m"}],
            "evaluators": [],
        }
    )
    assert len(cfg.prompts) == 1


def test_prompt_config_expected_output_field():
    pc = PromptConfig(id="q1", text="Q", expected_output="A")
    assert pc.expected_output == "A"
    assert pc.context is None


def test_sample_dataset_is_valid_jsonl():
    """The committed sample dataset must parse without errors."""
    sample = Path("examples/sample_dataset.jsonl")
    assert sample.exists(), "examples/sample_dataset.jsonl must be committed"
    prompts = load_dataset(str(sample))
    assert len(prompts) >= 1
    for p in prompts:
        assert p.id
        assert p.text
