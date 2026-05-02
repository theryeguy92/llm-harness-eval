"""Runner for OpenAI-compatible chat models via the Chat Completions API."""
import asyncio
import os
import time

import httpx

from .base import BaseRunner, RunResult

_DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAIRunner(BaseRunner):
    """Executes prompts against any OpenAI-compatible chat endpoint.

    Uses the Chat Completions API directly over async httpx. Supports OpenAI,
    Ollama, vLLM, and any other OpenAI-compatible provider via OPENAI_BASE_URL.
    Reads OPENAI_API_KEY and optionally OPENAI_BASE_URL from the environment.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        max_tokens: int = 1024,
        system: str | None = None,
    ) -> None:
        """
        Args:
            model: Model ID to request (e.g. "gpt-4o", "gemma3:12b").
            max_tokens: Maximum tokens to generate.
            system: Optional system message content.
        """
        self._model = model
        self._max_tokens = max_tokens
        self._system = system
        self._api_key = os.environ["OPENAI_API_KEY"]
        base = os.environ.get("OPENAI_BASE_URL", _DEFAULT_BASE_URL).rstrip("/")
        self._url = f"{base}/chat/completions"

    async def run(self, prompt: str) -> RunResult:
        """Send a prompt to the chat completions endpoint and return a structured result.

        Args:
            prompt: User message to send.

        Returns:
            RunResult with response text, token counts, and latency.
        """
        messages = []
        if self._system:
            messages.append({"role": "system", "content": self._system})
        messages.append({"role": "user", "content": prompt})

        start = time.perf_counter()
        async with httpx.AsyncClient(timeout=120.0) as client:
            for attempt in range(5):
                r = await client.post(
                    self._url,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "content-type": "application/json",
                    },
                    json={
                        "model": self._model,
                        "max_tokens": self._max_tokens,
                        "messages": messages,
                    },
                )
                if r.status_code == 429:
                    await asyncio.sleep(2 ** attempt)
                    continue
                r.raise_for_status()
                break
            else:
                r.raise_for_status()
        latency_ms = (time.perf_counter() - start) * 1000

        data = r.json()
        usage = data["usage"]
        return RunResult(
            model=data["model"],
            latency_ms=round(latency_ms, 1),
            input_tokens=usage["prompt_tokens"],
            output_tokens=usage["completion_tokens"],
            response=data["choices"][0]["message"]["content"],
        )
