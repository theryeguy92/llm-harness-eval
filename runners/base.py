"""Base classes and data models for LLM runners."""
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class RunResult(BaseModel):
    """Result of a single model inference call."""

    model: str = Field(..., description="Model identifier as returned by the API")
    latency_ms: float = Field(..., description="Wall-clock time for the API call in milliseconds")
    input_tokens: int = Field(..., description="Number of tokens in the input prompt")
    output_tokens: int = Field(..., description="Number of tokens in the generated response")
    response: str = Field(..., description="The model-generated text response")


class BaseRunner(ABC):
    """Abstract base for all model runners.

    Each runner wraps a single LLM provider (Anthropic, OpenAI, local, etc.)
    and exposes a uniform async interface so evaluation pipelines can treat
    all providers identically.
    """

    @abstractmethod
    async def run(self, prompt: str) -> RunResult:
        """Execute a prompt and return a structured result.

        Args:
            prompt: The user message to send to the model.

        Returns:
            RunResult containing the response, token counts, and latency.
        """
