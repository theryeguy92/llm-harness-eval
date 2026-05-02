# llm-eval-harness

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
