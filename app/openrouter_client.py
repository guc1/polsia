from __future__ import annotations

import os
from urllib.parse import urljoin

import httpx


class OpenRouterClient:
    def __init__(self, timeout: int = 90):
        self.api_key = os.getenv("OPENROUTER_API_KEY", "")
        default_base_url = "https://openrouter.ai"
        configured_base_url = os.getenv("OPENROUTER_BASE_URL", default_base_url).strip()
        self.base_url = configured_base_url.rstrip("/")
        self.timeout = timeout

    @staticmethod
    def _extract_error_message(response: httpx.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            text = response.text.strip()
            return text[:400] if text else "No response body"

        if isinstance(payload, dict):
            err = payload.get("error")
            if isinstance(err, dict):
                message = err.get("message") or err.get("metadata")
                if message:
                    return str(message)
            if isinstance(err, str):
                return err
        return str(payload)[:400]

    async def complete(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1200,
    ) -> str:
        if not self.api_key:
            return (
                "[DRY RUN] OPENROUTER_API_KEY missing. "
                f"Model={model}. Prompt digest: {user_prompt[:240]}"
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if self.base_url.rstrip("/").endswith("/api/v1"):
            endpoint_path = "chat/completions"
        else:
            endpoint_path = "api/v1/chat/completions"
        endpoint = urljoin(f"{self.base_url}/", endpoint_path)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            res = await client.post(endpoint, headers=headers, json=payload)
            if res.is_error:
                reason = self._extract_error_message(res)
                raise RuntimeError(
                    f"OpenRouter request failed ({res.status_code}) for model '{model}' at '{endpoint}'. "
                    f"Reason: {reason}"
                )
            data = res.json()
            return data["choices"][0]["message"]["content"].strip()
