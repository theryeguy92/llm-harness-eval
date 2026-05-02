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
    scores: dict[str, EvalResult]


class EvalReport(BaseModel):
    """Complete report for an evaluation run."""

    name: str
    timestamp: str
    config: EvalConfig
    results: list[ResultRow]


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

    return EvalReport(
        name=config.name,
        timestamp=datetime.now(timezone.utc).isoformat(),
        config=config,
        results=list(results),
    )


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------


def _write_json(report: EvalReport, path: Path) -> None:
    """Serialize the full report to a JSON file."""
    path.write_text(report.model_dump_json(indent=2))


def _write_markdown(report: EvalReport, path: Path) -> None:
    """Write a markdown table summarizing scores across all prompt × runner pairs."""
    eval_names = report.config.evaluators
    headers = ["Prompt", "Runner", "Model", "Latency (ms)", "Tokens"] + [e.title() for e in eval_names]
    header_row = "| " + " | ".join(headers) + " |"
    sep_row = "| " + " | ".join("---" for _ in headers) + " |"

    lines = [f"# {report.name}", f"_Generated: {report.timestamp}_", "", header_row, sep_row]
    for row in report.results:
        score_cells = [
            f"{row.scores[e].score:.2f}" if e in row.scores else "—" for e in eval_names
        ]
        prompt_short = row.prompt[:50].replace("|", "\\|")
        if len(row.prompt) > 50:
            prompt_short += "…"
        cells = [
            prompt_short,
            row.runner_type,
            row.model,
            f"{row.latency_ms:.0f}",
            str(row.input_tokens + row.output_tokens),
        ] + score_cells
        lines.append("| " + " | ".join(cells) + " |")

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
    for e in eval_names:
        table.add_column(e.title(), justify="right")

    for row in report.results:
        prompt_label = row.prompt[:32] + ("…" if len(row.prompt) > 32 else "")
        score_cells = [f"{row.scores[e].score:.2f}" if e in row.scores else "—" for e in eval_names]
        table.add_row(
            prompt_label,
            row.runner_type,
            row.model,
            f"{row.latency_ms:.0f} ms",
            str(row.input_tokens + row.output_tokens),
            *score_cells,
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
