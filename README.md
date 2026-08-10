# llm-eval-harness

[![CI](https://github.com/theryeguy92/llm-harness-eval/actions/workflows/test.yml/badge.svg)](https://github.com/theryeguy92/llm-harness-eval/actions/workflows/test.yml)

Most LLM failures aren't model failures — they're evaluation failures. Without a systematic way to measure coherence, faithfulness, and relevance, prompt changes and model upgrades are guesswork. This framework gives you a reproducible eval loop: define prompts in YAML or load them from a dataset file, run them through any combination of models in parallel, and get back structured scores. It's built for RAG quality measurement, A/B testing prompts or models, and latency benchmarking.

**What's included:**
- **5 evaluators:** `coherence`, `relevance`, `faithfulness` (LLM-as-judge via Claude) · `rouge_l`, `exact_match` (reference-based, no API calls)
- **3 runners:** Claude, OpenAI, Gemini — plus any OpenAI-compatible endpoint (Ollama, vLLM)
- **Dataset loader:** feed a `.jsonl` or `.csv` file instead of inline YAML prompts; columns `id`, `input`, `context`, `expected_output`
- **Bootstrap confidence intervals:** `--repeat N` runs each prompt N times and reports mean ± std with 95% CIs
- **Pairwise evaluation:** head-to-head comparison with documented position-swap debiasing
- **Cost tracking:** per-call USD estimates with a per-provider pricing table covering all supported models

## Quick Start

```bash
pip install -e ".[dev]"

cp .env.example .env
# Set ANTHROPIC_API_KEY (required — used by runners and evaluators)
# Set GOOGLE_API_KEY or OPENAI_API_KEY for those runners

# Run with inline YAML prompts
python run_eval.py --config examples/basic.yaml
# → prints a score table and writes reports/<name>_<timestamp>.{json,md}

# Repeat each prompt for confidence intervals
python run_eval.py --config examples/basic.yaml --repeat 3

# To load prompts from a dataset file instead, see below —
# examples/sample_dataset.jsonl shows the expected format:
#   {"id": "q1", "input": "...", "context": "...", "expected_output": "..."}
```

To use the dataset loader, replace the `prompts:` block in your config with:

```yaml
dataset: examples/sample_dataset.jsonl   # path relative to the config file
```

## Benchmark: Claude Haiku vs Sonnet (`--repeat 3`)

3 RAG prompts, each model scored on coherence, relevance, and faithfulness by a Claude Haiku judge. Each prompt was run 3 times; scores show mean ± std with 95% bootstrap confidence intervals. Full output in [`examples/sample_output/rag_comparison_haiku_vs_sonnet.md`](examples/sample_output/rag_comparison_haiku_vs_sonnet.md).

_Generated: 2026-05-02 · n=3 runs per prompt_

| Model | Avg Latency (ms) | Total Cost | Coherence (mean / win%) | Relevance (mean / win%) | Faithfulness (mean / win%) |
| --- | --- | --- | --- | --- | --- |
| claude-haiku-4-5-20251001 | 2,354 | $0.009051 | 0.87 / 0% | 0.44 / 0% | 0.99 / 0% |
| claude-sonnet-4-6 | 5,056 | $0.037287 | 0.88 / 67% | 0.57 / 67% | 0.99 / 33% |

**Per-prompt breakdown** (mean ± std [95% CI]):

| Prompt | Model | Latency (ms) | Cost | Coherence | Relevance | Faithfulness |
| --- | --- | --- | --- | --- | --- | --- |
| Solar panel efficiency | haiku-4-5 | 2,311 | $0.000853 | 0.85 ± 0.00 [0.85–0.85] | 0.00 ± 0.00 [0.00–0.00] | 1.00 ± 0.00 [1.00–1.00] |
| Solar panel efficiency | sonnet-4-6 | 4,683 | $0.003659 | 0.85 ± 0.00 [0.85–0.85] | 0.00 ± 0.00 [0.00–0.00] | 1.00 ± 0.00 [1.00–1.00] |
| Self-attention mechanism | haiku-4-5 | 2,455 | $0.001116 | 0.93 ± 0.02 [0.92–0.95] | 0.57 ± 0.12 [0.50–0.70] | 0.98 ± 0.03 [0.95–1.00] |
| Self-attention mechanism | sonnet-4-6 | 4,825 | $0.004319 | 0.94 ± 0.02 [0.92–0.95] | 0.85 ± 0.00 [0.85–0.85] | 0.98 ± 0.03 [0.95–1.00] |
| Paris Agreement targets | haiku-4-5 | 2,295 | $0.001048 | 0.82 ± 0.06 [0.75–0.85] | 0.77 ± 0.03 [0.75–0.80] | 1.00 ± 0.00 [1.00–1.00] |
| Paris Agreement targets | sonnet-4-6 | 5,661 | $0.004451 | 0.85 ± 0.00 [0.85–0.85] | 0.85 ± 0.00 [0.85–0.85] | 0.98 ± 0.03 [0.95–1.00] |

Haiku is 2.1× faster and 4.1× cheaper per run. Sonnet wins on relevance across most prompts (67% win rate) with noticeably higher scores on the transformer and Paris Agreement tasks. Faithfulness is near-identical for both, which makes sense — it has the tightest objective anchor. The zero relevance scores on the solar prompt appear on both models, indicating a judge calibration issue for that particular phrasing rather than a model failure.

## How Evaluation Works

Each evaluator sends the prompt, the model's response, and (for RAG evals) the reference context to a Claude judge model. The judge returns a float score in `[0.0, 1.0]` with a one-sentence explanation. All scores for a given (prompt × model) pair are gathered concurrently.

| Evaluator | What it measures | API call? | Needs reference? |
|-----------|-----------------|-----------|-----------------|
| `coherence` | Internal logical consistency — ideas flow without contradictions | Yes (judge) | No |
| `relevance` | Whether the response directly answers what was asked | Yes (judge) | No |
| `faithfulness` | Whether every claim is grounded in the retrieved context — catches hallucination | Yes (judge) | `context` |
| `rouge_l` | ROUGE-L F1 against an expected output — purely lexical, no judge | No | `expected_output` |
| `exact_match` | Case-insensitive exact string equality — useful for classification and extraction | No | `expected_output` |

The LLM-as-judge pattern scales to arbitrary criteria without labeled data, making it practical for teams that can't run human evals on every prompt change. The tradeoff: judge scores inherit the judge model's biases, so treat them as a signal rather than ground truth. Use `rouge_l` and `exact_match` when a correct reference answer exists and correctness is unambiguous.

## Evaluator Methodology & Known Limitations

### How LLM-as-judge works here

Each evaluator sends three things to Claude: the original user prompt, the model's response, and (for faithfulness) the reference context. A system prompt defines the scoring rubric for that dimension. The judge returns a JSON object — `{"score": float, "explanation": string}` — which is parsed and stored alongside the response.

Evaluations for each (prompt × model) pair run concurrently via `asyncio.gather`, so wall-clock time scales with the slowest judge call, not with the number of evaluations.

### Known biases

**Verbosity bias.** LLM judges systematically assign higher scores to longer responses, even when a concise answer is more accurate. The coherence and relevance evaluators do not penalize length, so models that generate more tokens will tend to score higher on those dimensions. Faithfulness is less affected because it grounds scoring in a reference document.

**Positional bias.** In pairwise-comparison setups, LLM judges favor whichever response appears first. This framework evaluates each response independently rather than side-by-side, which avoids the most direct form. A residual form can still appear if content placement in the judge prompt influences what the model attends to.

**Self-preference.** Claude judges tend to favor Claude-style outputs — responses that are structured, hedged, and formatted the way Claude writes. This bias is less pronounced when comparing two Claude models (as in the benchmark above), but matters most when comparing Claude against non-Claude models like Gemini or GPT-4.

### What this means when reading scores

- A gap smaller than ~0.1 between two models on any single metric is within the noise of judge variability. Don't draw conclusions from it.
- Faithfulness is the most reliable metric because it has an objective anchor (the reference document). Coherence and relevance are more subjective and more susceptible to the biases above.
- Scores are relative signals within a single run, not absolute quality measurements. A coherence score of 0.85 does not mean "85% coherent" in any well-defined sense.
- If you are using this framework to compare Claude against other models, Claude-as-judge will systematically favor Claude. Human spot-checks are the appropriate corrective for high-stakes decisions.

### What a `--swap-judge` flag would do

A `--swap-judge <model>` option would re-score all responses from a completed run using a different judge model — for example, re-evaluating with GPT-4o after an initial Claude-judged run. If scores converge, the results are more trustworthy. If they diverge by more than ~0.1 on average, the divergence is itself a finding: either the judge models have meaningfully different standards for the metric, or one is exhibiting strong self-preference.

This is not yet implemented. The JSON output stores all raw responses, so the data needed for a retrospective re-score is already there.

### Prompt versioning

Each evaluator class carries a `PROMPT_VERSION` constant (currently `"v1"` for all three). This value is recorded in every JSON report under `evaluator_versions`.

The reason: changing the judge system prompt — even a small wording change — can shift scores by 0.1–0.2. Without version tracking, you cannot tell whether a score change between two runs reflects a better model or a different evaluator. The convention is to bump `PROMPT_VERSION` to `"v2"` whenever the system prompt changes, and to re-run historical baselines before comparing across versions.

## Reference-Based vs LLM-as-Judge Evaluators

Use reference-based evaluators (`rouge_l`, `exact_match`) when a ground-truth answer exists and correctness is objective — extraction tasks, closed-form QA, structured outputs; use LLM-as-judge evaluators (`coherence`, `relevance`, `faithfulness`) when acceptable answers are open-ended or when measuring qualities like tone, reasoning quality, or explanatory depth that no single reference string can capture.

For reference-based evaluators, pass the expected answer as the `context` field in your prompt config (or `expected_output` column when loading from a dataset).

## Pairwise vs Pointwise Evaluation

This framework supports two evaluation modes.

**Pointwise** (the default): each response is scored independently on a `[0, 1]` scale for a given dimension. You get absolute-ish scores that aggregate across prompts and models — useful for dashboards, trend tracking, and identifying regressions. The evaluators in `evaluators/coherence.py`, `relevance.py`, and `faithfulness.py` are all pointwise.

**Pairwise**: two responses are shown to the judge simultaneously and it picks a winner (`"a"`, `"b"`, or `"tie"`). This catches preference differences that pointwise scores miss — a judge may rate both responses `0.80`, yet reliably prefer one when they appear side by side. `evaluators/pairwise.py` provides the base class and `ExplainabilityPairwise` as a concrete example.

```python
from evaluators.pairwise import ExplainabilityPairwise

ev = ExplainabilityPairwise()
result = await ev.compare(
    prompt="Explain how transformers use attention.",
    response_a=claude_output,
    response_b=gemini_output,
)
# PairwiseResult(winner='a', confidence=0.8, reasoning='...')
```

**MT-Bench and AlpacaEval** both use pairwise evaluation at scale — MT-Bench uses GPT-4 to judge 80 multi-turn questions, AlpacaEval uses a win-rate against a reference model (text-davinci-003 or GPT-4). The key finding from both benchmarks: pairwise judgments correlate more strongly with human preference rankings than pointwise scores do, but they are more expensive (two judge calls per comparison instead of one) and harder to aggregate across more than two models.

**Positional bias in pairwise evaluation.** LLM judges favor position A by a measurable margin — roughly 60-65% of the time in published studies even when responses are equivalent. The `BasePairwiseEvaluator` docstring shows the standard mitigation: run the comparison twice with A and B swapped and treat a verdict flip as a tie. This doubles judge cost but produces much more reliable results. If you skip the swap, bias the benchmark against the model that tends to appear in position B.

## Config Format

**Option A — inline prompts:**

```yaml
name: my_eval
output_dir: reports/
seed: 42          # bootstrap CI seed — keeps confidence intervals reproducible
concurrency: 10   # max simultaneous prompt×runner tasks

prompts:
  - id: q1
    text: "According to the document, what does X say about Y?"
    context: |
      <your retrieved passage here>
    expected_output: "The expected answer for ROUGE/exact-match scoring"

runners:
  - type: claude
    model: claude-sonnet-4-6
    max_tokens: 1024
  - type: gemini
    model: gemini-flash-latest
    max_tokens: 1024
  - type: openai
    model: gpt-4o
    max_tokens: 1024

evaluators:
  - coherence
  - relevance
  - faithfulness
  - rouge_l
  - exact_match
```

**Option B — dataset file** (replaces the `prompts:` block):

```yaml
name: my_eval
dataset: examples/sample_dataset.jsonl   # path relative to this config file

runners:
  - type: claude
    model: claude-sonnet-4-6
    max_tokens: 1024

evaluators:
  - rouge_l
  - exact_match
```

Dataset format (JSONL — one JSON object per line, or CSV with the same column names):

```jsonl
{"id": "q1", "input": "What is the capital of France?", "expected_output": "Paris"}
{"id": "q2", "input": "Summarise the document.", "context": "<passage>", "expected_output": "<reference summary>"}
```

Each run is fully reproducible from its YAML. Reports include all raw responses and per-evaluator explanations in the JSON output.

## Adding an Evaluator

Subclass `BaseEvaluator`, implement one async method, register it in `_EVALUATOR_MAP` in `run_eval.py`:

```python
from evaluators.base import BaseEvaluator, EvalResult

class ToxicityEvaluator(BaseEvaluator):
    """Scores whether the response contains harmful content."""

    async def score(self, prompt: str, response: str, context: str | None = None) -> EvalResult:
        # call your judge — Claude, a fine-tuned classifier, or a rules-based check
        return EvalResult(score=0.95, explanation="No harmful content detected.")
```

## Adding a Runner

Subclass `BaseRunner`, implement one async method, register it in `_build_runner()` in `run_eval.py`:

```python
from runners.base import BaseRunner, RunResult

class MistralRunner(BaseRunner):
    """Runs prompts through Mistral's API."""

    async def run(self, prompt: str) -> RunResult:
        # async HTTP call, parse response
        return RunResult(model="mistral-large", latency_ms=980, input_tokens=42, output_tokens=110, response="...")
```

The harness handles concurrency, report writing, and CLI wiring — the runner only needs to move bytes in and out.

## Project Structure

```
evaluators/            # Coherence, faithfulness, relevance (LLM-as-judge) · rouge_l, exact_match (reference-based)
runners/               # Claude, Gemini, OpenAI wrappers — uniform async interface, with retry + cost tracking
reports/               # JSON (full data) + markdown (table) output per run
examples/              # Worked YAML configs and sample dataset
  basic.yaml           #   inline-prompt config: Haiku vs Gemini Flash
  sample_dataset.jsonl #   dataset-loader example: id/input/context/expected_output
  sample_output/       #   real eval output committed for reviewer inspection
run_eval.py            # CLI: --config, --output-dir, --repeat N
```

## Environment Variables

| Variable | Required for |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Claude runner and all evaluators (LLM-as-judge) |
| `GOOGLE_API_KEY` | Gemini runner |
| `OPENAI_API_KEY` | OpenAI runner |

---

**GitHub topics:** `llm-evaluation` · `rag` · `benchmarking` · `llm-as-judge` · `python` · `anthropic` · `a-b-testing`
