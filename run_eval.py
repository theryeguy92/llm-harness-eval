"""CLI entrypoint for running LLM evaluations."""
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

from evaluators import CoherenceEvaluator, FaithfulnessEvaluator, RelevanceEvaluator
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


class RunnerConfig(BaseModel):
    """Configuration for a single model runner."""

    type: str = Field(..., description='Runner type: "claude", "openai", or "gemini"')
    model: str = Field(..., description="Model ID to pass to the provider API")
    max_tokens: int = Field(1024, description="Maximum tokens to generate")
    system: str | None = Field(None, description="Optional system prompt")


class EvalConfig(BaseModel):
    """Top-level evaluation configuration loaded from a YAML file."""

    name: str = Field(..., description="Human-readable name for this evaluation run")
    prompts: list[PromptConfig] = Field(..., description="Prompts to evaluate")
    runners: list[RunnerConfig] = Field(..., description="Models to run prompts through")
    evaluators: list[str] = Field(..., description="Evaluator names to apply to each response")
    output_dir: str = Field("reports/", description="Directory for JSON and markdown output")


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


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

_EVALUATOR_MAP: dict[str, type[BaseEvaluator]] = {
    "coherence": CoherenceEvaluator,
    "faithfulness": FaithfulnessEvaluator,
    "relevance": RelevanceEvaluator,
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

    eval_names = list(evaluators.keys())
    eval_results = await asyncio.gather(
        *[ev.score(prompt_cfg.text, run.response, prompt_cfg.context) for ev in evaluators.values()]
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


async def _run_eval(config: EvalConfig) -> EvalReport:
    """Execute the full prompt × runner evaluation matrix and return a report."""
    evaluators = _build_evaluators(config.evaluators)
    runners = [(cfg, _build_runner(cfg)) for cfg in config.runners]

    tasks = [
        _evaluate_one(prompt_cfg, runner_cfg, runner, evaluators)
        for prompt_cfg in config.prompts
        for runner_cfg, runner in runners
    ]
    console.print(f"[bold]Running {len(tasks)} evaluation(s)…[/bold]")
    results = await asyncio.gather(*tasks)

    result_list = list(results)
    return EvalReport(
        name=config.name,
        timestamp=datetime.now(timezone.utc).isoformat(),
        config=config,
        results=result_list,
        summary=compute_summary(result_list, config.evaluators),
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


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------


def _write_json(report: EvalReport, path: Path) -> None:
    """Serialize the full report to a JSON file."""
    path.write_text(report.model_dump_json(indent=2))


def _write_markdown(report: EvalReport, path: Path) -> None:
    """Write a markdown report with a summary table followed by per-row results."""
    eval_names = report.config.evaluators
    blank = "—"
    lines = [f"# {report.name}", f"_Generated: {report.timestamp}_", ""]

    # Summary table (at the top)
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

    # Per-row results table
    lines += ["## Results", ""]
    headers = ["Prompt", "Runner", "Model", "Latency (ms)", "Tokens", "Cost ($)"] + [e.title() for e in eval_names]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in report.results:
        score_cells = [f"{row.scores[e].score:.2f}" if e in row.scores else blank for e in eval_names]
        prompt_short = row.prompt[:50].replace("|", "\\|")
        if len(row.prompt) > 50:
            prompt_short += "…"
        cost_cell = f"${row.cost_usd:.6f}" if row.cost_usd else blank
        cells = [
            prompt_short, row.runner_type, row.model,
            f"{row.latency_ms:.0f}", str(row.input_tokens + row.output_tokens), cost_cell,
        ] + score_cells
        lines.append("| " + " | ".join(cells) + " |")

    # Cost totals grouped by model
    by_model: dict[tuple[str, str], list[ResultRow]] = {}
    for row in report.results:
        by_model.setdefault((row.runner_type, row.model), []).append(row)

    n_cols = len(headers)
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for (runner_type, model), rows in by_model.items():
        total = sum(r.cost_usd for r in rows)
        avg = total / len(rows)
        empty = [blank] * (n_cols - 6)
        lines.append("| " + " | ".join(["**Total**", runner_type, model, blank, blank, f"**${total:.6f}**"] + empty) + " |")
        lines.append("| " + " | ".join(["**Avg/prompt**", runner_type, model, blank, blank, f"**${avg:.6f}**"] + empty) + " |")

    path.write_text("\n".join(lines) + "\n")


def _print_summary(report: EvalReport) -> None:
    """Print a rich table summary to the terminal."""
    eval_names = report.config.evaluators
    table = Table(title=f"Eval: {report.name}", show_lines=True)
    table.add_column("Prompt", max_width=32)
    table.add_column("Runner")
    table.add_column("Model")
    table.add_column("Latency", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Cost ($)", justify="right")
    for e in eval_names:
        table.add_column(e.title(), justify="right")

    for row in report.results:
        prompt_label = row.prompt[:32] + ("…" if len(row.prompt) > 32 else "")
        cost_cell = f"${row.cost_usd:.6f}" if row.cost_usd else "—"
        score_cells = [f"{row.scores[e].score:.2f}" if e in row.scores else "—" for e in eval_names]
        table.add_row(
            prompt_label,
            row.runner_type,
            row.model,
            f"{row.latency_ms:.0f} ms",
            str(row.input_tokens + row.output_tokens),
            cost_cell,
            *score_cells,
        )

    # Summary rows
    by_model: dict[tuple[str, str], list[ResultRow]] = {}
    for row in report.results:
        key = (row.runner_type, row.model)
        by_model.setdefault(key, []).append(row)

    for (runner_type, model), rows in by_model.items():
        total = sum(r.cost_usd for r in rows)
        avg = total / len(rows)
        blank = "—"
        n_score_blanks = len(eval_names)
        table.add_row(
            "[bold]Total[/bold]", runner_type, model, blank, blank,
            f"[bold]${total:.6f}[/bold]", *([blank] * n_score_blanks),
        )
        table.add_row(
            "[bold]Avg/prompt[/bold]", runner_type, model, blank, blank,
            f"[bold]${avg:.6f}[/bold]", *([blank] * n_score_blanks),
        )

    console.print(table)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    config: Path = typer.Option(..., help="Path to a YAML evaluation config."),
    output_dir: Optional[Path] = typer.Option(None, help="Override the output directory from the config."),
) -> None:
    """Run an LLM evaluation defined by a YAML config file.

    Prompts are run through all configured model runners concurrently, then scored
    by all configured evaluators. Results are written to reports/ as JSON and markdown.
    """
    raw = yaml.safe_load(config.read_text())
    eval_cfg = EvalConfig.model_validate(raw)

    if output_dir is not None:
        eval_cfg.output_dir = str(output_dir)

    out = Path(eval_cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    report = asyncio.run(_run_eval(eval_cfg))

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
