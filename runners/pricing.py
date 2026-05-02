"""Per-token pricing for supported LLM providers.

All prices are in USD per 1,000,000 tokens (i.e. per-million).
Update this file when provider pricing changes.

Sources (checked 2026-05-02):
  Anthropic : https://www.anthropic.com/pricing
  OpenAI    : https://openai.com/api/pricing
  Google    : https://ai.google.dev/gemini-api/docs/pricing
"""

from typing import NamedTuple


class ModelPricing(NamedTuple):
    """Input and output price in USD per 1 million tokens."""

    input: float
    output: float


# Keys are model ID prefixes.  get_cost_usd() picks the longest prefix that
# matches the model ID returned by the API, so versioned IDs like
# "claude-haiku-4-5-20251001" correctly resolve to "claude-haiku-4-5".
PRICES: dict[str, ModelPricing] = {
    # -------------------------------------------------------------------------
    # Anthropic Claude
    # -------------------------------------------------------------------------
    "claude-haiku-4-5":     ModelPricing(input=0.80,  output=4.00),
    "claude-sonnet-4-6":    ModelPricing(input=3.00,  output=15.00),
    "claude-opus-4-7":      ModelPricing(input=15.00, output=75.00),
    # Claude 3 / 3.5 legacy
    "claude-3-haiku":       ModelPricing(input=0.25,  output=1.25),
    "claude-3-5-haiku":     ModelPricing(input=0.80,  output=4.00),
    "claude-3-5-sonnet":    ModelPricing(input=3.00,  output=15.00),
    "claude-3-opus":        ModelPricing(input=15.00, output=75.00),
    # -------------------------------------------------------------------------
    # OpenAI
    # -------------------------------------------------------------------------
    "gpt-4o-mini":          ModelPricing(input=0.15,  output=0.60),
    "gpt-4o":               ModelPricing(input=2.50,  output=10.00),
    "gpt-4-turbo":          ModelPricing(input=10.00, output=30.00),
    "gpt-3.5-turbo":        ModelPricing(input=0.50,  output=1.50),
    # -------------------------------------------------------------------------
    # Google Gemini  (prices are for prompts ≤1M tokens; larger context costs more)
    # -------------------------------------------------------------------------
    "gemini-flash-latest":  ModelPricing(input=0.075, output=0.30),
    "gemini-2.0-flash":     ModelPricing(input=0.075, output=0.30),
    "gemini-1.5-flash":     ModelPricing(input=0.075, output=0.30),
    "gemini-pro-latest":    ModelPricing(input=1.25,  output=5.00),
    "gemini-1.5-pro":       ModelPricing(input=1.25,  output=5.00),
}


def get_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return the USD cost of a model call.

    Looks up the model in PRICES using exact match then longest-prefix fallback.
    Returns 0.0 if the model is not in the price table.

    Args:
        model: Model identifier as returned by the provider API.
        input_tokens: Number of prompt tokens consumed.
        output_tokens: Number of completion tokens generated.

    Returns:
        Cost in USD, rounded to eight decimal places.
    """
    tier = PRICES.get(model)
    if tier is None:
        matches = [(k, v) for k, v in PRICES.items() if model.startswith(k)]
        if matches:
            tier = max(matches, key=lambda kv: len(kv[0]))[1]
    if tier is None:
        return 0.0
    cost = (input_tokens * tier.input + output_tokens * tier.output) / 1_000_000
    return round(cost, 8)
