# rag_comparison_sample
_Generated: 2026-05-02T16:08:29.191110+00:00_

## Summary

| Model | Runner | Avg Latency (ms) | Total Cost ($) | Coherence (mean / win%) | Relevance (mean / win%) | Faithfulness (mean / win%) |
| --- | --- | --- | --- | --- | --- | --- |
| claude-haiku-4-5-20251001 | claude | 2354 | $0.009051 | 0.87 / 0% | 0.44 / 0% | 0.99 / 0% |
| claude-sonnet-4-6 | claude | 5056 | $0.037287 | 0.88 / 67% | 0.57 / 67% | 0.99 / 33% |

## Results (n=3 runs per prompt)

| Prompt | Runner | Model | Runs | Avg Latency (ms) | Avg Cost ($) | Coherence mean ± std [95% CI] | Relevance mean ± std [95% CI] | Faithfulness mean ± std [95% CI] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Based on the document, what efficiency improvement… | claude | claude-haiku-4-5-20251001 | 3 | 2311 | $0.000853 | 0.85 ± 0.00 [0.85–0.85] | 0.00 ± 0.00 [0.00–0.00] | 1.00 ± 0.00 [1.00–1.00] |
| Based on the document, what efficiency improvement… | claude | claude-sonnet-4-6 | 3 | 4683 | $0.003659 | 0.85 ± 0.00 [0.85–0.85] | 0.00 ± 0.00 [0.00–0.00] | 1.00 ± 0.00 [1.00–1.00] |
| According to the document, how does the self-atten… | claude | claude-haiku-4-5-20251001 | 3 | 2455 | $0.001116 | 0.93 ± 0.02 [0.92–0.95] | 0.57 ± 0.12 [0.50–0.70] | 0.98 ± 0.03 [0.95–1.00] |
| According to the document, how does the self-atten… | claude | claude-sonnet-4-6 | 3 | 4825 | $0.004319 | 0.94 ± 0.02 [0.92–0.95] | 0.85 ± 0.00 [0.85–0.85] | 0.98 ± 0.03 [0.95–1.00] |
| What specific emissions targets and financial comm… | claude | claude-haiku-4-5-20251001 | 3 | 2295 | $0.001048 | 0.82 ± 0.06 [0.75–0.85] | 0.77 ± 0.03 [0.75–0.80] | 1.00 ± 0.00 [1.00–1.00] |
| What specific emissions targets and financial comm… | claude | claude-sonnet-4-6 | 3 | 5661 | $0.004451 | 0.85 ± 0.00 [0.85–0.85] | 0.85 ± 0.00 [0.85–0.85] | 0.98 ± 0.03 [0.95–1.00] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Total** | claude | claude-haiku-4-5-20251001 | — | — | **$0.009051** | — | — | — |
| **Avg/prompt** | claude | claude-haiku-4-5-20251001 | — | — | **$0.003017** | — | — | — |
| **Total** | claude | claude-sonnet-4-6 | — | — | **$0.037287** | — | — | — |
| **Avg/prompt** | claude | claude-sonnet-4-6 | — | — | **$0.012429** | — | — | — |
