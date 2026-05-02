# llm-eval-harness

[![CI](https://github.com/theryeguy92/llm-harness-eval/actions/workflows/test.yml/badge.svg)](https://github.com/theryeguy92/llm-harness-eval/actions/workflows/test.yml)

Most LLM failures aren't model failures — they're evaluation failures. Without a systematic way to measure coherence, faithfulness, and relevance, prompt changes and model upgrades are guesswork. This framework gives you a reproducible eval loop: define prompts and context in YAML, run them through any combination of models in parallel, and get back structured scores from an LLM-as-judge. It's built for RAG quality measurement, A/B testing prompts or models, and latency benchmarking — the three things you need before you can ship a retrieval system with confidence.

## Quick Start

```bash
pip install -e ".[dev]"

cp .env.example .env
# Set ANTHROPIC_API_KEY (required — used by runners and evaluators)
# Set GOOGLE_API_KEY or OPENAI_API_KEY for those runners

python run_eval.py --config examples/basic.yaml
# → prints a score table and writes reports/<name>_<timestamp>.{json,md}
```

## Benchmark: Claude Haiku vs Gemini Flash

3 RAG prompts, each model scored on coherence, relevance, and faithfulness by a Claude-as-judge. Run with `examples/basic.yaml`.

_Generated: 2026-05-02_

| Prompt | Model | Latency (ms) | Tokens | Coherence | Relevance | Faithfulness |
| --- | --- | --- | --- | --- | --- | --- |
| Solar panel efficiency improvements | claude-haiku-4-5-20251001 | 2474 | 361 | 0.85 | 0.00 | 0.95 |
| Solar panel efficiency improvements | gemini-flash-latest | 3142 | 239 | 0.85 | 0.00 | 1.00 |
| Self-attention in transformers | claude-haiku-4-5-20251001 | 1821 | 387 | 0.90 | 0.85 | 1.00 |
| Self-attention in transformers | gemini-flash-latest | 3530 | 173 | 0.30 | 0.50 | 1.00 |
| Paris Agreement targets | claude-haiku-4-5-20251001 | 1849 | 389 | 0.85 | 0.85 | 1.00 |
| Paris Agreement targets | gemini-flash-latest | 2794 | 229 | 0.60 | 0.30 | 0.85 |

Claude Haiku is faster and more consistent; Gemini Flash matches or exceeds faithfulness scores at lower token counts but shows higher variance in coherence.

## How Evaluation Works

Each evaluator sends the prompt, the model's response, and (for RAG evals) the reference context to a Claude judge model. The judge returns a float score in `[0.0, 1.0]` with a one-sentence explanation. All scores for a given (prompt × model) pair are gathered concurrently.

| Evaluator | What it measures | Needs context? |
|-----------|-----------------|----------------|
| `coherence` | Internal logical consistency — ideas flow without contradictions | No |
| `relevance` | Whether the response directly answers what was asked | No |
| `faithfulness` | Whether every claim is grounded in the retrieved context — catches hallucination | Yes |

The LLM-as-judge pattern scales to arbitrary criteria without labeled data, making it practical for teams that can't run human evals on every prompt change. The tradeoff: judge scores inherit the judge model's biases, so treat them as a signal rather than ground truth.

## Evaluator Methodology & Known Limitations

### How LLM-as-judge works here

Each evaluator sends three things to Claude: the original user prompt, the model's response, and (for faithfulness) the reference context. A system prompt defines the scoring rubric for that dimension. The judge returns a JSON object — `{"score": float, "explanation": string}` — which is parsed and stored alongside the response.

Evaluations for each (prompt × model) pair run concurrently via `asyncio.gather`, so wall-clock time scales with the slowest judge call, not with the number of evaluations.

### Known biases

**Verbosity bias.** LLM judges systematically assign higher scores to longer responses, even when a concise answer is more accurate. The coherence and relevance evaluators do not penalize length, so models that generate more tokens will tend to score higher on those dimensions. Faithfulness is less affected because it grounds scoring in a reference document.

**Positional bias.** In pairwise-comparison setups, LLM judges favor whichever response appears first. This framework evaluates each response independently rather than side-by-side, which avoids the most direct form. A residual form can still appear if content placement in the judge prompt influences what the model attends to.

**Self-preference.** Claude judges tend to favor Claude-style outputs — responses that are structured, hedged, and formatted the way Claude writes. The benchmark table above compares Claude Haiku against Gemini Flash using Claude as the judge, which likely inflates Claude's coherence and relevance scores. This is the strongest caveat on those numbers.

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

```yaml
name: my_eval
output_dir: reports/

prompts:
  - id: q1
    text: "According to the document, what does X say about Y?"
    context: |
      <your retrieved passage here>

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
evaluators/   # Coherence, faithfulness, relevance — each scores one dimension
runners/      # Claude, Gemini, OpenAI wrappers — uniform async interface
reports/      # JSON (full data) + markdown (table) output per run
examples/     # Worked YAML configs with real outputs
run_eval.py   # CLI: --config to run, --output-dir to override report path
```

## Environment Variables

| Variable | Required for |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Claude runner and all evaluators (LLM-as-judge) |
| `GOOGLE_API_KEY` | Gemini runner |
| `OPENAI_API_KEY` | OpenAI runner |

---

**GitHub topics:** `llm-evaluation` · `rag` · `benchmarking` · `llm-as-judge` · `python` · `anthropic` · `a-b-testing`
