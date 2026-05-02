"""Runner for Anthropic Claude models via the Messages API."""
import os
import time

import httpx

from .base import BaseRunner, RunResult


class ClaudeRunner(BaseRunner):
    """Executes prompts against an Anthropic Claude model.

    Uses the Anthropic Messages API directly over async httpx.
    Reads ANTHROPIC_API_KEY from the environment.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 1024,
        system: str | None = None,
    ) -> None:
        """
        Args:
            model: Anthropic model ID (e.g. "claude-sonnet-4-6").
            max_tokens: Maximum tokens to generate.
            system: Optional system prompt.
        """
        self._model = model
        self._max_tokens = max_tokens
        self._system = system
        self._api_key = os.environ["ANTHROPIC_API_KEY"]

    async def run(self, prompt: str) -> RunResult:
        """Send a prompt to Claude and return the structured result.

        Args:
            prompt: User message to send.

        Returns:
            RunResult with response text, token counts, and latency.
        """
        body: dict = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self._system:
            body["system"] = self._system

        start = time.perf_counter()
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=body,
            )
            r.raise_for_status()
        latency_ms = (time.perf_counter() - start) * 1000

        data = r.json()
        return RunResult(
            model=data["model"],
            latency_ms=round(latency_ms, 1),
            input_tokens=data["usage"]["input_tokens"],
            output_tokens=data["usage"]["output_tokens"],
            response=data["content"][0]["text"],
        )
