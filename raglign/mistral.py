"""Minimal Mistral API client for the optional generation-metrics path.

Three things this handles that a bare `urlopen` does not:

1. **Working TLS.** This machine's Python reports its default CA bundle as
   `C:\\Program Files\\Common Files\\ssl\\cert.pem`, which does not exist, so
   every HTTPS request fails with `[ASN1: NOT_ENOUGH_DATA]`. The context here is
   built explicitly from `certifi`'s bundle. Pinning the trust store rather than
   disabling verification: turning verification off would "fix" it while sending
   an API key to an unauthenticated peer.

2. **A response cache on disk.** Judging is idempotent for a fixed (model,
   prompt, temperature, seed), so a re-run should not re-spend. This matters
   more than usual here: the metrics get recomputed whenever the reporting
   changes, and without a cache each report costs real money and drifts, making
   two "identical" runs disagree.

3. **Retry with backoff on transient failures only.** 429 and 5xx are retried;
   a 401 or 400 is raised immediately, because retrying a bad key or a malformed
   request just burns time and hides the real error.

No SDK dependency: this is one endpoint and about a hundred lines, and the
retrieval side of the project runs with no credentials and no network at all.
Keeping the generation path thin keeps that contrast obvious.
"""

from __future__ import annotations

import hashlib
import json
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import apikeys

API_BASE = "https://api.mistral.ai/v1"
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "mistral"

# Small model on purpose. The judge task (does this text support that claim?) is
# not a frontier-model problem, and a cheaper model means the metric can be run
# with repeats -- which matters more for a noisy LLM judge than raw capability.
DEFAULT_MODEL = "mistral-small-latest"

RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


def _ssl_context() -> ssl.SSLContext:
    import certifi

    return ssl.create_default_context(cafile=certifi.where())


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    cached: int = 0

    def add(self, payload: dict, *, cached: bool) -> None:
        self.calls += 1
        if cached:
            self.cached += 1
            return
        u = payload.get("usage") or {}
        self.prompt_tokens += int(u.get("prompt_tokens", 0))
        self.completion_tokens += int(u.get("completion_tokens", 0))

    def as_dict(self) -> dict:
        return {
            "calls": self.calls,
            "cache_hits": self.cached,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


class MistralClient:
    """Chat completions with caching, retries and usage accounting."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        cache: bool = True,
        max_retries: int = 4,
        timeout: int = 90,
    ):
        self.model = model
        self.cache = cache
        self.max_retries = max_retries
        self.timeout = timeout
        self.usage = Usage()
        self._ctx = _ssl_context()
        self._key = apikeys.get("MISTRAL_API_KEY")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, body: dict) -> Path:
        # The key deliberately excludes the API key, and the cached file holds
        # only the response -- so .cache/ never contains a credential.
        digest = hashlib.sha256(
            json.dumps(body, sort_keys=True).encode("utf-8")
        ).hexdigest()[:32]
        return CACHE_DIR / f"{digest}.json"

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(
            f"{API_BASE}{path}",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout, context=self._ctx) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if exc.code not in RETRY_STATUS:
                    detail = exc.read().decode("utf-8", "replace")[:300]
                    raise RuntimeError(f"Mistral HTTP {exc.code}: {detail}") from exc
                last = exc
            except (urllib.error.URLError, TimeoutError) as exc:
                last = exc
            time.sleep(min(2**attempt, 8))
        raise RuntimeError(f"Mistral request failed after {self.max_retries} attempts: {last}")

    def chat(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = 0,
        json_mode: bool = False,
    ) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        body: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            body["random_seed"] = seed
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        path = self._cache_path(body)
        if self.cache and path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.usage.add(payload, cached=True)
            return payload["choices"][0]["message"]["content"]

        payload = self._post("/chat/completions", body)
        if self.cache:
            path.write_text(json.dumps(payload), encoding="utf-8")
        self.usage.add(payload, cached=False)
        return payload["choices"][0]["message"]["content"]

    def chat_json(self, prompt: str, *, system: str | None = None, **kw) -> dict:
        """Chat with JSON output, tolerant of a model that wraps it in prose.

        Returns {} on unparseable output rather than raising: one malformed judge
        response should degrade that single score, not abort a run of hundreds.
        The caller counts the failures.
        """
        raw = self.chat(prompt, system=system, json_mode=True, **kw)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            start, end = raw.find("{"), raw.rfind("}")
            if start != -1 and end > start:
                try:
                    return json.loads(raw[start : end + 1])
                except json.JSONDecodeError:
                    return {}
            return {}


def available() -> bool:
    return apikeys.available("MISTRAL_API_KEY")
