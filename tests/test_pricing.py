"""Tests for runners/pricing.py."""
import pytest

from runners.pricing import PRICES, ModelPricing, get_cost_usd


def test_prices_dict_has_expected_providers():
    keys = " ".join(PRICES.keys())
    assert "claude" in keys
    assert "gpt" in keys
    assert "gemini" in keys


def test_model_pricing_fields():
    tier = PRICES["gpt-4o"]
    assert isinstance(tier, ModelPricing)
    assert tier.input > 0
    assert tier.output > 0


# ---------------------------------------------------------------------------
# Exact-match lookups
# ---------------------------------------------------------------------------


def test_exact_match_gpt4o():
    cost = get_cost_usd("gpt-4o", input_tokens=1_000_000, output_tokens=0)
    assert cost == pytest.approx(2.50)


def test_exact_match_gpt4o_mini():
    cost = get_cost_usd("gpt-4o-mini", input_tokens=1_000_000, output_tokens=0)
    assert cost == pytest.approx(0.15)


def test_gpt4o_and_gpt4o_mini_resolve_independently():
    """gpt-4o must not accidentally match gpt-4o-mini's pricing and vice-versa."""
    cost_4o = get_cost_usd("gpt-4o", input_tokens=0, output_tokens=1_000_000)
    cost_mini = get_cost_usd("gpt-4o-mini", input_tokens=0, output_tokens=1_000_000)
    assert cost_4o == pytest.approx(10.00)
    assert cost_mini == pytest.approx(0.60)
    assert cost_4o != cost_mini


# ---------------------------------------------------------------------------
# Prefix-match lookups (versioned model IDs)
# ---------------------------------------------------------------------------


def test_versioned_claude_haiku_resolves():
    """claude-haiku-4-5-20251001 should resolve via the claude-haiku-4-5 prefix."""
    cost = get_cost_usd("claude-haiku-4-5-20251001", input_tokens=1_000_000, output_tokens=0)
    assert cost == pytest.approx(0.80)


def test_versioned_claude_sonnet_resolves():
    cost = get_cost_usd("claude-sonnet-4-6", input_tokens=1_000_000, output_tokens=0)
    assert cost == pytest.approx(3.00)


def test_gemini_flash_latest_resolves():
    cost = get_cost_usd("gemini-flash-latest", input_tokens=1_000_000, output_tokens=0)
    assert cost == pytest.approx(0.075)


# ---------------------------------------------------------------------------
# Math correctness
# ---------------------------------------------------------------------------


def test_cost_combines_input_and_output():
    # gpt-4o: $2.50/M input, $10.00/M output
    cost = get_cost_usd("gpt-4o", input_tokens=500_000, output_tokens=100_000)
    expected = (500_000 * 2.50 + 100_000 * 10.00) / 1_000_000
    assert cost == pytest.approx(expected)


def test_cost_rounds_to_eight_decimal_places():
    cost = get_cost_usd("gemini-flash-latest", input_tokens=1, output_tokens=1)
    assert cost == round(cost, 8)


def test_zero_tokens_returns_zero():
    assert get_cost_usd("gpt-4o", input_tokens=0, output_tokens=0) == 0.0


# ---------------------------------------------------------------------------
# Unknown model fallback
# ---------------------------------------------------------------------------


def test_unknown_model_returns_zero():
    assert get_cost_usd("some-unknown-model-v99", input_tokens=1000, output_tokens=500) == 0.0


def test_partial_match_does_not_fire_on_unrelated_prefix():
    """A model starting with 'gpt' but not matching any key should return 0.0."""
    assert get_cost_usd("gpt-future-model-x", input_tokens=1000, output_tokens=500) == 0.0
