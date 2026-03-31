from __future__ import annotations

from urllib.parse import urljoin

import httpx

from app.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL


class OpenRouterClient:
    def __init__(self, timeout: int = 90):
        self.api_key = OPENROUTER_API_KEY
        self.base_url = OPENROUTER_BASE_URL.rstrip("/")
        self.timeout = timeout

    async def complete(self, model: str, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
        if not self.api_key:
            return f"[DRY RUN] {model}: {user_prompt[:500]}"
        endpoint = urljoin(f"{self.base_url}/", "api/v1/chat/completions")
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            "temperature": temperature,
            "max_tokens": 900,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
