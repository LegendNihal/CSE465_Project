"""Thin async client for the local vLLM OpenAI-compatible server."""
from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class Completion:
    text: str
    finish_reason: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class VLLMClient:
    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY",
                 concurrency: int = 24, timeout_s: float = 3600.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.sem = asyncio.Semaphore(concurrency)
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_s, connect=30.0),
            headers={"Authorization": f"Bearer {api_key}"},
            limits=httpx.Limits(max_connections=concurrency + 8),
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def chat(self, prompt: str, *, n: int = 1, temperature: float = 1.0,
                   top_p: float = 0.95, top_k: int = -1, max_tokens: int = 32768,
                   stop: list[str] | None = None,
                   max_retries: int = 4) -> list[Completion]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "n": n,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        # vLLM accepts top_k via extra params; -1 disables it.
        if top_k is not None:
            body["top_k"] = top_k
        if stop:
            body["stop"] = stop

        delay = 5.0
        last_err = None
        for attempt in range(max_retries):
            try:
                async with self.sem:
                    r = await self._client.post(f"{self.base_url}/chat/completions",
                                                json=body)
                if r.status_code >= 400:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:400]}")
                data = r.json()
                usage = data.get("usage") or {}
                return [
                    Completion(
                        text=c["message"]["content"] or "",
                        finish_reason=c.get("finish_reason", "unknown"),
                        prompt_tokens=usage.get("prompt_tokens", 0),
                        completion_tokens=usage.get("completion_tokens", 0),
                    )
                    for c in data["choices"]
                ]
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt == max_retries - 1:
                    break
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60.0)
        raise RuntimeError(f"chat failed after {max_retries} tries: {last_err}")


class Progress:
    """Minimal stderr progress line; no extra dependencies."""

    def __init__(self, total: int, label: str = ""):
        self.total = max(total, 1)
        self.done = 0
        self.label = label
        self.t0 = time.time()
        self._last = 0.0

    def tick(self, k: int = 1) -> None:
        self.done += k
        now = time.time()
        if now - self._last < 1.0 and self.done < self.total:
            return
        self._last = now
        el = now - self.t0
        rate = self.done / el if el > 0 else 0
        eta = (self.total - self.done) / rate if rate > 0 else 0
        sys.stderr.write(
            f"\r{self.label} {self.done}/{self.total} "
            f"({100 * self.done / self.total:5.1f}%) "
            f"elapsed {el / 60:6.1f}m eta {eta / 60:6.1f}m   "
        )
        sys.stderr.flush()

    def close(self) -> None:
        sys.stderr.write("\n")
        sys.stderr.flush()


async def wait_for_server(base_url: str, tries: int = 60) -> None:
    async with httpx.AsyncClient(timeout=10.0) as c:
        for i in range(tries):
            try:
                r = await c.get(f"{base_url.rstrip('/')}/models")
                if r.status_code == 200:
                    return
            except Exception:  # noqa: BLE001
                pass
            await asyncio.sleep(5)
    raise RuntimeError(f"vLLM server at {base_url} never came up")
