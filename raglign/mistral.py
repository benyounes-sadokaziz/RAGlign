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
import http.client
import json
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from . import apikeys

API_BASE = "https://api.mistral.ai/v1"
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "mistral"

# Model entitlement is per-model on Mistral, not per-account: a key can hold a
# valid subscription while specific models return 429 with
# `x-ratelimit-limit-req-minute: 0`. On the key used here mistral-small/medium are
# capped at zero while the ministral and nemo families are not -- so defaults
# point at models that are actually entitled, and `probe_models` exists to find
# out rather than guess.
DEFAULT_MODEL = "ministral-8b-latest"

# A DIFFERENT model judges than generates, on purpose. LLM judges show
# self-preference: they score their own outputs higher than equivalent text from
# another model. Using one family to answer and another to judge removes the most
# obvious way for the faithfulness number to be flattering rather than accurate.
DEFAULT_JUDGE_MODEL = "open-mistral-nemo"

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
        max_retries: int = 7,
        timeout: int = 90,
        requests_per_minute: int = 90,
    ):
        self.model = model
        self.cache = cache
        self.max_retries = max_retries
        self.timeout = timeout
        # Well under the entitled limits (188/min for nemo and ministral-8b) so a
        # long run leaves room for whatever else shares the key.
        self.min_interval = 60.0 / requests_per_minute if requests_per_minute else 0.0
        self._last_call = 0.0
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

    def _throttle(self) -> None:
        """Keep a minimum gap between calls.

        The server drops connections when a few hundred requests arrive back to
        back, which surfaces as RemoteDisconnected rather than 429 -- so pacing
        client-side is not politeness, it is what stops the run from failing
        halfway through. Derived from the documented per-minute limit with
        headroom, since the limit is shared with anything else using the key.
        """
        if self.min_interval <= 0:
            return
        wait = self.min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)

    def _post(self, path: str, body: dict) -> dict:
        payload = json.dumps(body).encode("utf-8")
        last: Exception | None = None
        for attempt in range(self.max_retries):
            # A fresh Request per attempt: urllib mutates the object during a
            # failed open (redirect state, host headers), and reusing it after an
            # error has produced confusing follow-on failures.
            req = urllib.request.Request(
                f"{API_BASE}{path}",
                data=payload,
                headers={
                    "Authorization": f"Bearer {self._key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                method="POST",
            )
            try:
                self._throttle()
                with urllib.request.urlopen(req, timeout=self.timeout, context=self._ctx) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if exc.code not in RETRY_STATUS:
                    detail = exc.read().decode("utf-8", "replace")[:300]
                    raise RuntimeError(f"Mistral HTTP {exc.code}: {detail}") from exc
                last = exc
            # OSError covers ConnectionResetError and RemoteDisconnected, which
            # are NOT URLError subclasses and so escaped an earlier, narrower
            # except clause -- killing a long run on a single dropped connection.
            except (urllib.error.URLError, OSError, http.client.HTTPException, TimeoutError) as exc:
                last = exc
            finally:
                self._last_call = time.monotonic()
            # Cap at 30s, not 8: a WinError 10060 outage lasted longer than
            # five short backoffs could ride out, and killed a paid run.
            time.sleep(min(2**attempt, 30))
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


def probe_models(candidates: Sequence[str], *, pause: float = 1.2) -> dict[str, int | None]:
    """Return {model: requests-per-minute limit}, or None where it is unusable.

    Worth having as a function rather than a one-off script: entitlement differs
    per key and per tier, so "which models can this key actually call" is a
    question any new environment has to re-answer. Guessing it wrong reads as an
    account problem when it is only a model choice.
    """
    out: dict[str, int | None] = {}
    for model in candidates:
        body = {
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4,
        }
        req = urllib.request.Request(
            f"{API_BASE}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {apikeys.get('MISTRAL_API_KEY')}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=60, context=_ssl_context()) as resp:
                limit = resp.headers.get("x-ratelimit-limit-req-minute")
                out[model] = int(limit) if limit and limit.isdigit() else 0
        except urllib.error.HTTPError as exc:
            limit = exc.headers.get("x-ratelimit-limit-req-minute")
            out[model] = int(limit) if limit and limit.isdigit() and limit != "0" else None
        except Exception:  # noqa: BLE001
            out[model] = None
        time.sleep(pause)
    return out
