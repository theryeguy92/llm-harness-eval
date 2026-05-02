"""CLI entrypoint for running LLM evaluations."""
import asyncio
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import typer
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator
from rich.console import Console
from rich.table import Table

from evaluators import (
    CoherenceEvaluator,
    ExactMatchEvaluator,
    FaithfulnessEvaluator,
    RelevanceEvaluator,
    RougeEvaluator,
)
from evaluators.base import BaseEvaluator, EvalResult
from runners import ClaudeRunner, GeminiRunner, OpenAIRunner
from runners.base import BaseRunner, RunResult

load_dotenv()

app = typer.Typer(help="LLM evaluation harness — run prompt A/B tests and score quality.")
console = Console()


# ---------------------------------------------------------------------------
# Config models
# ---------------------------------------------------------------------------


class PromptConfig(BaseModel):
    """A single prompt entry in the evaluation config."""

    id: str = Field(..., description="Unique identifier for this prompt")
    text: str = Field(..., description="The prompt text to send to the model")
    context: str | None = Field(None, description="Optional reference text for RAG evals")
    expected_output: str | None = Field(
        None,
        description="Ground-truth answer for reference-based evaluators (rouge_l, exact_match).",
    )


class RunnerConfig(BaseModel):
    """Configuration for a single model runner."""

    type: str = Field(..., description='Runner type: "claude", "openai", or "gemini"')
    model: str = Field(..., description="Model ID to pass to the provider API")
    max_tokens: int = Field(1024, description="Maximum tokens to generate")
    system: str | None = Field(None, description="Optional system prompt")


class EvalConfig(BaseModel):
    """Top-level evaluation configuration loaded from a YAML file."""

    name: str = Field(..., description="Human-readable name for this evaluation run")
    prompts: list[PromptConfig] = Field(
        default_factory=list,
        description="Inline prompts to evaluate. Ignored when 'dataset' is set.",
    )
    dataset: str | None = Field(
        None,
        description=(
            "Path to a JSONL or CSV dataset file with columns: "
            "id, input, context (optional), expected_output (optional). "
            "When present, replaces the inline 'prompts' list."
        ),
    )
    runners: list[RunnerConfig] = Field(..., description="Models to run prompts through")
    evaluators: list[str] = Field(..., description="Evaluator names to apply to each response")
    output_dir: str = Field("reports/", description="Directory for JSON and markdown output")

    @model_validator(mode="after")
    def _require_prompts_or_dataset(self) -> "EvalConfig":
        if not self.prompts and not self.dataset:
            raise ValueError("Provide either 'prompts' (inline list) or 'dataset' (path to JSONL/CSV).")
        return self


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------


def load_dataset(path: str, config_dir: Path | None = None) -> list[PromptConfig]:
    """Load prompts from a JSONL or CSV dataset file.

    The file must have the columns/keys ``id`` and ``input``.  The optional
    columns ``context`` and ``expected_output`` are passed through as-is.

    Args:
        path: Absolute path, or a path relative to ``config_dir``.
        config_dir: Directory of the YAML config file; used to resolve
            relative paths.  Defaults to the current working directory.

    Returns:
        List of PromptConfig objects ready for evaluation.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If the file extension is not ``.jsonl`` or ``.csv``,
            or if required columns are missing.
    """
    resolved = Path(path)
    if not resolved.is_absolute() and config_dir is not None:
        resolved = config_dir / resolved
    resolved = resolved.resolve()

    if not resolved.exists():
        raise FileNotFoundError(f"Dataset file not found: {resolved}")

    suffix = resolved.suffix.lower()
    if suffix == ".jsonl":
        rows = [json.loads(line) for line in resolved.read_text().splitlines() if line.strip()]
    elif suffix == ".csv":
        with resolved.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    else:
        raise ValueError(f"Unsupported dataset format: {suffix!r}. Use .jsonl or .csv.")

    prompts: list[PromptConfig] = []
    for i, row in enumerate(rows):
        if "id" not in row or "input" not in row:
            raise ValueError(f"Row {i} is missing required column(s) 'id' and/or 'input': {row}")
        prompts.append(
            PromptConfig(
                id=str(row["id"]),
                text=str(row["input"]),
                context=row.get("context") or None,
                expected_output=row.get("expected_output") or None,
            )
        )
    return prompts


# ---------------------------------------------------------------------------
# Report models
# ---------------------------------------------------------------------------


class ResultRow(BaseModel):
    """One (prompt × runner) result with all evaluator scores."""

    prompt_id: str
    prompt: str
    context: str | None
    runner_type: str
    model: str
    response: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    scores: dict[str, EvalResult]


class ScoreStats(BaseModel):
    """Aggregated statistics for one metric across N repeated runs."""

    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    n: int


class AggregatedRow(BaseModel):
    """(prompt × runner) result aggregated across N repeated runs."""

    prompt_id: str
    prompt: str
    context: str | None
    runner_type: str
    model: str
    n_runs: int
    avg_latency_ms: float
    avg_cost_usd: float
    total_cost_usd: float
    scores: dict[str, ScoreStats]


class EvaluatorInfo(BaseModel):
    """Snapshot of an evaluator's identity recorded at run time."""

    name: str
    prompt_version: str


class MetricSummary(BaseModel):
    """Aggregate statistics for one metric across all prompts for one model."""

    mean_score: float
    win_rate: float  # fraction of prompts where this model strictly outscored all others


class ModelSummary(BaseModel):
    """Per-model aggregate statistics for one evaluation run."""

    model: str
    runner_type: str
    avg_latency_ms: float
    total_cost_usd: float
    metrics: dict[str, MetricSummary]


class EvalReport(BaseModel):
    """Complete report for an evaluation run."""

    name: str
    timestamp: str
    config: EvalConfig
    results: list[ResultRow]
    summary: dict[str, ModelSummary] = Field(default_factory=dict)
    evaluator_versions: dict[str, EvaluatorInfo] = Field(default_factory=dict)
    repeat: int = Field(1, description="Number of times each prompt was run")
    aggregated_results: list[AggregatedRow] = Field(default_factory=list)
    statistical_note: str | None = None


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

_EVALUATOR_MAP: dict[str, type[BaseEvaluator]] = {
    "coherence": CoherenceEvaluator,
    "exact_match": ExactMatchEvaluator,
    "faithfulness": FaithfulnessEvaluator,
    "relevance": RelevanceEvaluator,
    "rouge_l": RougeEvaluator,
}


def _build_runner(cfg: RunnerConfig) -> BaseRunner:
    """Instantiate the correct runner subclass from a RunnerConfig."""
    match cfg.type:
        case "claude":
            return ClaudeRunner(model=cfg.model, max_tokens=cfg.max_tokens, system=cfg.system)
        case "openai":
            return OpenAIRunner(model=cfg.model, max_tokens=cfg.max_tokens, system=cfg.system)
        case "gemini":
            return GeminiRunner(model=cfg.model, max_tokens=cfg.max_tokens, system=cfg.system)
        case _:
            raise ValueError(f"Unknown runner type: {cfg.type!r}. Choose from: claude, openai, gemini")


def _build_evaluators(names: list[str]) -> dict[str, BaseEvaluator]:
    """Instantiate evaluators by name, raising on unknown names."""
    unknown = [n for n in names if n not in _EVALUATOR_MAP]
    if unknown:
        raise ValueError(f"Unknown evaluator(s): {unknown}. Choose from: {list(_EVALUATOR_MAP)}")
    return {name: _EVALUATOR_MAP[name]() for name in names}


# ---------------------------------------------------------------------------
# Async eval logic
# ---------------------------------------------------------------------------


async def _evaluate_one(
    prompt_cfg: PromptConfig,
    runner_cfg: RunnerConfig,
    runner: BaseRunner,
    evaluators: dict[str, BaseEvaluator],
) -> ResultRow:
    """Run a single (prompt × runner) pair and score with all evaluators concurrently."""
    if prompt_cfg.context:
        full_prompt = f"Document:\n{prompt_cfg.context}\n\nQuestion: {prompt_cfg.text}"
    else:
        full_prompt = prompt_cfg.text
    run: RunResult = await runner.run(full_prompt)

    # Reference-based evaluators (rouge_l, exact_match) read their reference from
    # expected_output when available; fall back to context so RAG-only configs still work.
    eval_context = prompt_cfg.expected_output or prompt_cfg.context
    eval_names = list(evaluators.keys())
    eval_results = await asyncio.gather(
        *[ev.score(prompt_cfg.text, run.response, eval_context) for ev in evaluators.values()]
    )
    scores: dict[str, EvalResult] = dict(zip(eval_names, eval_results))

    return ResultRow(
        prompt_id=prompt_cfg.id,
        prompt=prompt_cfg.text,
        context=prompt_cfg.context,
        runner_type=runner_cfg.type,
        model=run.model,
        response=run.response,
        latency_ms=run.latency_ms,
        input_tokens=run.input_tokens,
        output_tokens=run.output_tokens,
        cost_usd=run.cost_usd,
        scores=scores,
    )


async def _run_eval(config: EvalConfig, repeat: int = 1) -> EvalReport:
    """Execute the full prompt × runner × repeat evaluation matrix and return a report."""
    evaluators = _build_evaluators(config.evaluators)
    runners = [(cfg, _build_runner(cfg)) for cfg in config.runners]

    tasks = [
        _evaluate_one(prompt_cfg, runner_cfg, runner, evaluators)
        for prompt_cfg in config.prompts
        for runner_cfg, runner in runners
        for _ in range(repeat)
    ]
    n_combos = len(config.prompts) * len(runners)
    console.print(f"[bold]Running {len(tasks)} evaluation(s) ({repeat} × {n_combos} prompt/runner combinations)…[/bold]")
    results = await asyncio.gather(*tasks)

    result_list = list(results)
    evaluator_versions = {
        name: EvaluatorInfo(name=ev.NAME, prompt_version=ev.PROMPT_VERSION)
        for name, ev in evaluators.items()
    }
    note = (
        "Single run per prompt (--repeat 1). Results are illustrative; "
        "sample size is too small for statistical significance."
        if repeat == 1 else None
    )
    return EvalReport(
        name=config.name,
        timestamp=datetime.now(timezone.utc).isoformat(),
        config=config,
        results=result_list,
        summary=compute_summary(result_list, config.evaluators),
        evaluator_versions=evaluator_versions,
        repeat=repeat,
        aggregated_results=aggregate_repeated_runs(result_list, config.evaluators),
        statistical_note=note,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def compute_summary(
    results: list[ResultRow],
    eval_names: list[str],
) -> dict[str, ModelSummary]:
    """Compute per-model aggregate statistics from a flat list of ResultRows.

    Args:
        results: All (prompt × runner) result rows from an evaluation run.
        eval_names: Ordered list of evaluator metric names to aggregate.

    Returns:
        Mapping from model ID to its ModelSummary.
    """
    if not results:
        return {}

    by_model: dict[str, list[ResultRow]] = {}
    for row in results:
        by_model.setdefault(row.model, []).append(row)

    by_prompt: dict[str, list[ResultRow]] = {}
    for row in results:
        by_prompt.setdefault(row.prompt_id, []).append(row)

    n_prompts = len(by_prompt)

    wins: dict[str, dict[str, int]] = {m: {e: 0 for e in eval_names} for m in by_model}
    for prompt_rows in by_prompt.values():
        for metric in eval_names:
            scored = {
                row.model: row.scores[metric].score
                for row in prompt_rows
                if metric in row.scores
            }
            if not scored:
                continue
            max_score = max(scored.values())
            top = [m for m, s in scored.items() if s == max_score]
            if len(top) == 1:
                wins[top[0]][metric] += 1

    summary: dict[str, ModelSummary] = {}
    for model, rows in by_model.items():
        avg_latency = sum(r.latency_ms for r in rows) / len(rows)
        total_cost = sum(r.cost_usd for r in rows)
        metrics: dict[str, MetricSummary] = {}
        for metric in eval_names:
            metric_scores = [r.scores[metric].score for r in rows if metric in r.scores]
            mean = sum(metric_scores) / len(metric_scores) if metric_scores else 0.0
            win_rate = wins[model][metric] / n_prompts if n_prompts > 0 else 0.0
            metrics[metric] = MetricSummary(
                mean_score=round(mean, 4),
                win_rate=round(win_rate, 4),
            )
        summary[model] = ModelSummary(
            model=model,
            runner_type=rows[0].runner_type,
            avg_latency_ms=round(avg_latency, 1),
            total_cost_usd=round(total_cost, 8),
            metrics=metrics,
        )
    return summary


def _bootstrap_ci(
    scores: list[float],
    n_bootstrap: int = 1000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Return a 95% bootstrap confidence interval for the mean of scores.

    With N=1 the CI degenerates to (score, score) — callers should surface a
    note rather than presenting this as a meaningful interval.
    """
    arr = np.array(scores, dtype=float)
    if len(arr) == 1:
        return float(arr[0]), float(arr[0])
    if rng is None:
        rng = np.random.default_rng()
    indices = rng.integers(0, len(arr), size=(n_bootstrap, len(arr)))
    boot_means = arr[indices].mean(axis=1)
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def aggregate_repeated_runs(
    results: list[ResultRow],
    eval_names: list[str],
    n_bootstrap: int = 1000,
    rng: np.random.Generator | None = None,
) -> list[AggregatedRow]:
    """Group repeated runs of the same (prompt × runner) pair and compute stats.

    Args:
        results: Flat list of all ResultRows, including all repeated runs.
        eval_names: Ordered list of evaluator metric names.
        n_bootstrap: Number of bootstrap samples for the 95% CI.
        rng: Optional seeded numpy Generator for reproducible CIs in tests.

    Returns:
        One AggregatedRow per unique (prompt_id, model) pair, in insertion order.
    """
    if not results:
        return []

    groups: dict[tuple[str, str], list[ResultRow]] = {}
    for row in results:
        groups.setdefault((row.prompt_id, row.model), []).append(row)

    aggregated: list[AggregatedRow] = []
    for (prompt_id, model), rows in groups.items():
        n = len(rows)
        avg_latency = sum(r.latency_ms for r in rows) / n
        total_cost = sum(r.cost_usd for r in rows)

        scores: dict[str, ScoreStats] = {}
        for metric in eval_names:
            metric_scores = [r.scores[metric].score for r in rows if metric in r.scores]
            if not metric_scores:
                continue
            arr = np.array(metric_scores, dtype=float)
            mean = float(arr.mean())
            std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
            ci_lower, ci_upper = _bootstrap_ci(metric_scores, n_bootstrap=n_bootstrap, rng=rng)
            scores[metric] = ScoreStats(
                mean=round(mean, 4),
                std=round(std, 4),
                ci_lower=round(ci_lower, 4),
                ci_upper=round(ci_upper, 4),
                n=len(metric_scores),
            )

        aggregated.append(AggregatedRow(
            prompt_id=prompt_id,
            prompt=rows[0].prompt,
            context=rows[0].context,
            runner_type=rows[0].runner_type,
            model=model,
            n_runs=n,
            avg_latency_ms=round(avg_latency, 1),
            avg_cost_usd=round(total_cost / n, 8),
            total_cost_usd=round(total_cost, 8),
            scores=scores,
        ))

    return aggregated


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------


def _write_json(report: EvalReport, path: Path) -> None:
    """Serialize the full report to a JSON file."""
    path.write_text(report.model_dump_json(indent=2))


def _fmt_score_stats(s: ScoreStats) -> str:
    """Format a ScoreStats cell for markdown. Suppresses CI display when n=1."""
    if s.n == 1:
        return f"{s.mean:.2f} ± —"
    return f"{s.mean:.2f} ± {s.std:.2f} [{s.ci_lower:.2f}–{s.ci_upper:.2f}]"


def _write_markdown(report: EvalReport, path: Path) -> None:
    """Write a markdown report: optional stat note, summary table, aggregated results."""
    eval_names = report.config.evaluators
    blank = "—"
    lines = [f"# {report.name}", f"_Generated: {report.timestamp}_", ""]

    if report.statistical_note:
        lines += [f"> **Note:** {report.statistical_note}", ""]

    # Model-level summary table
    if report.summary:
        lines += ["## Summary", ""]
        sum_headers = (
            ["Model", "Runner", "Avg Latency (ms)", "Total Cost ($)"]
            + [f"{e.title()} (mean / win%)" for e in eval_names]
        )
        lines.append("| " + " | ".join(sum_headers) + " |")
        lines.append("| " + " | ".join("---" for _ in sum_headers) + " |")
        for ms in report.summary.values():
            metric_cells = []
            for metric in eval_names:
                m = ms.metrics.get(metric)
                metric_cells.append(f"{m.mean_score:.2f} / {m.win_rate * 100:.0f}%" if m else blank)
            lines.append(
                "| "
                + " | ".join(
                    [ms.model, ms.runner_type, f"{ms.avg_latency_ms:.0f}", f"${ms.total_cost_usd:.6f}"]
                    + metric_cells
                )
                + " |"
            )
        lines.append("")

    # Aggregated results table (mean ± std [CI] per metric)
    repeat = report.repeat
    lines += [f"## Results (n={repeat} run{'s' if repeat != 1 else ''} per prompt)", ""]
    agg_headers = (
        ["Prompt", "Runner", "Model", "Runs", "Avg Latency (ms)", "Avg Cost ($)"]
        + [f"{e.title()} mean ± std [95% CI]" for e in eval_names]
    )
    lines.append("| " + " | ".join(agg_headers) + " |")
    lines.append("| " + " | ".join("---" for _ in agg_headers) + " |")
    for agg in report.aggregated_results:
        prompt_short = agg.prompt[:50].replace("|", "\\|")
        if len(agg.prompt) > 50:
            prompt_short += "…"
        score_cells = [_fmt_score_stats(agg.scores[e]) if e in agg.scores else blank for e in eval_names]
        cells = [
            prompt_short, agg.runner_type, agg.model,
            str(agg.n_runs), f"{agg.avg_latency_ms:.0f}", f"${agg.avg_cost_usd:.6f}",
        ] + score_cells
        lines.append("| " + " | ".join(cells) + " |")

    # Cost totals per model
    by_model: dict[tuple[str, str], list[AggregatedRow]] = {}
    for agg in report.aggregated_results:
        by_model.setdefault((agg.runner_type, agg.model), []).append(agg)

    n_cols = len(agg_headers)
    lines.append("| " + " | ".join("---" for _ in agg_headers) + " |")
    for (runner_type, model), agg_rows in by_model.items():
        total = sum(r.total_cost_usd for r in agg_rows)
        avg_per_prompt = total / len(agg_rows)
        empty = [blank] * (n_cols - 6)
        lines.append("| " + " | ".join(["**Total**", runner_type, model, blank, blank, f"**${total:.6f}**"] + empty) + " |")
        lines.append("| " + " | ".join(["**Avg/prompt**", runner_type, model, blank, blank, f"**${avg_per_prompt:.6f}**"] + empty) + " |")

    path.write_text("\n".join(lines) + "\n")


def _print_summary(report: EvalReport) -> None:
    """Print a rich table of aggregated results to the terminal."""
    eval_names = report.config.evaluators
    blank = "—"

    if report.statistical_note:
        console.print(f"\n[yellow]Note:[/yellow] {report.statistical_note}\n")

    title = f"Eval: {report.name}  (n={report.repeat} run{'s' if report.repeat != 1 else ''} per prompt)"
    table = Table(title=title, show_lines=True)
    table.add_column("Prompt", max_width=28)
    table.add_column("Runner")
    table.add_column("Model")
    table.add_column("Runs", justify="right")
    table.add_column("Avg Latency", justify="right")
    table.add_column("Avg Cost ($)", justify="right")
    for e in eval_names:
        table.add_column(f"{e.title()} mean±std", justify="right")

    for agg in report.aggregated_results:
        prompt_label = agg.prompt[:28] + ("…" if len(agg.prompt) > 28 else "")
        score_cells = [
            _fmt_score_stats(agg.scores[e]) if e in agg.scores else blank
            for e in eval_names
        ]
        table.add_row(
            prompt_label, agg.runner_type, agg.model,
            str(agg.n_runs), f"{agg.avg_latency_ms:.0f} ms", f"${agg.avg_cost_usd:.6f}",
            *score_cells,
        )

    by_model: dict[tuple[str, str], list[AggregatedRow]] = {}
    for agg in report.aggregated_results:
        by_model.setdefault((agg.runner_type, agg.model), []).append(agg)

    for (runner_type, model), agg_rows in by_model.items():
        total = sum(r.total_cost_usd for r in agg_rows)
        avg = total / len(agg_rows)
        n_blanks = len(eval_names)
        table.add_row(
            "[bold]Total[/bold]", runner_type, model, blank, blank,
            f"[bold]${total:.6f}[/bold]", *([blank] * n_blanks),
        )
        table.add_row(
            "[bold]Avg/prompt[/bold]", runner_type, model, blank, blank,
            f"[bold]${avg:.6f}[/bold]", *([blank] * n_blanks),
        )

    console.print(table)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    config: Path = typer.Option(..., help="Path to a YAML evaluation config."),
    output_dir: Optional[Path] = typer.Option(None, help="Override the output directory from the config."),
    repeat: int = typer.Option(1, min=1, help="Number of times to run each prompt. Use ≥10 for meaningful confidence intervals."),
) -> None:
    """Run an LLM evaluation defined by a YAML config file.

    Prompts are run through all configured model runners concurrently, then scored
    by all configured evaluators. With --repeat N, each prompt runs N times and
    reports include mean ± std and 95% bootstrap confidence intervals per metric.
    Results are written to reports/ as JSON and markdown.
    """
    raw = yaml.safe_load(config.read_text())
    eval_cfg = EvalConfig.model_validate(raw)

    if eval_cfg.dataset:
        eval_cfg.prompts = load_dataset(eval_cfg.dataset, config_dir=config.parent)

    if output_dir is not None:
        eval_cfg.output_dir = str(output_dir)

    out = Path(eval_cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    report = asyncio.run(_run_eval(eval_cfg, repeat=repeat))

    slug = eval_cfg.name.replace(" ", "_").lower()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{slug}_{ts}"

    json_path = out / f"{stem}.json"
    md_path = out / f"{stem}.md"

    _write_json(report, json_path)
    _write_markdown(report, md_path)
    _print_summary(report)

    console.print("\n[green]Reports written:[/green]")
    console.print(f"  JSON     → {json_path}")
    console.print(f"  Markdown → {md_path}")


if __name__ == "__main__":
    app()
