"""Runner for Google Gemini models via the Gemini REST API."""
import asyncio
import os
import time

import httpx

from .base import BaseRunner, RunResult

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiRunner(BaseRunner):
    """Executes prompts against a Google Gemini model.

    Uses the Gemini generateContent REST API directly over async httpx.
    Reads GOOGLE_API_KEY from the environment.
    """

    def __init__(
        self,
        model: str = "gemini-flash-latest",
        max_tokens: int = 1024,
        system: str | None = None,
    ) -> None:
        """
        Args:
            model: Gemini model ID (e.g. "gemini-flash-latest", "gemini-pro-latest").
            max_tokens: Maximum tokens to generate.
            system: Optional system instruction.
        """
        self._model = model
        self._max_tokens = max_tokens
        self._system = system
        self._api_key = os.environ["GOOGLE_API_KEY"]

    async def run(self, prompt: str) -> RunResult:
        """Send a prompt to Gemini and return the structured result.

        Args:
            prompt: User message to send.

        Returns:
            RunResult with response text, token counts, and latency.
        """
        body: dict = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": self._max_tokens},
        }
        if self._system:
            body["systemInstruction"] = {"parts": [{"text": self._system}]}

        url = f"{_BASE_URL}/{self._model}:generateContent"

        start = time.perf_counter()
        async with httpx.AsyncClient(timeout=60.0) as client:
            for attempt in range(5):
                r = await client.post(
                    url,
                    headers={
                        "content-type": "application/json",
                        "X-goog-api-key": self._api_key,
                    },
                    json=body,
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
        usage = data["usageMetadata"]
        return RunResult(
            model=self._model,
            latency_ms=round(latency_ms, 1),
            input_tokens=usage["promptTokenCount"],
            output_tokens=usage["candidatesTokenCount"],
            response=data["candidates"][0]["content"]["parts"][0]["text"],
        )
