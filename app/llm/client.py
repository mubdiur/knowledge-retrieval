"""Async OpenAI-compatible client for local LLM inference.

Connects to a vLLM/LM Studio/Ollama OpenAI-compatible endpoint
(e.g. http://100.73.222.42:1234).

Falls back to None responses if the endpoint is unavailable,
so callers should always have a rule-based fallback.
"""

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_URL = "http://100.73.222.42:1234/v1"
DEFAULT_MODEL = "qwen3.5-0.8b"


class OllamaClient:
    """Async client for OpenAI-compatible API.

    Falls back to None responses if the endpoint is unavailable,
    so callers should always have a rule-based fallback.
    """

    def __init__(self, base_url: str = DEFAULT_URL, model: str = DEFAULT_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0))
        return self._client

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 50,
        temperature: float = 0.3,
    ) -> str | None:
        """Generate text from a prompt. Returns None on failure."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
        }

        try:
            client = await self._get_client()
            response = await client.post(f"{self.base_url}/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            logger.debug("LLM response (%d chars): %s...", len(content), content[:80])
            return content or None
        except httpx.TimeoutException:
            logger.warning("LLM request timed out")
            return None
        except Exception as e:
            logger.warning("LLM request failed: %s", e)
            return None

    async def is_available(self) -> bool:
        """Check if the endpoint is running and the model is loaded."""
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/models", timeout=5.0)
            response.raise_for_status()
            data = response.json()
            models = [m.get("id", "") for m in data.get("data", [])]
            return self.model in models
        except Exception:
            return False

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
