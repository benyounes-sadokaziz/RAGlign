"""API key resolution, for the optional generation-metrics path.

Keys are read from the process environment, falling back to a `.env` file at the
repo root. `.env` is already gitignored (v1), and nothing in this module ever
prints, logs, or returns a key alongside other data -- a key that reaches a run
manifest or a results table is a key that reaches a git history.

No third-party dotenv dependency: the parsing needed here is fifteen lines, and
the retrieval side of this project deliberately runs with no network and no
credentials at all. Generation is the only part that needs a key, and it stays
opt-in.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class MissingKeyError(RuntimeError):
    """Raised with setup instructions rather than a bare KeyError."""


def _parse_env_file(path: Path = ENV_FILE) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip().strip("'\"")
    return out


def get(name: str, *, required: bool = True) -> str | None:
    """Return one key from the environment or .env.

    Environment wins over the file, so a shell export can override a committed
    placeholder without editing anything.
    """
    value = os.environ.get(name) or _parse_env_file().get(name)
    if value:
        return value
    if not required:
        return None
    raise MissingKeyError(
        f"{name} is not set.\n"
        f"  Either:  set {name}=... in your shell\n"
        f"  Or:      add a line `{name}=...` to {ENV_FILE.name} at the repo root\n"
        f"  ({ENV_FILE.name} is gitignored, so the key stays out of git.)"
    )


def available(name: str) -> bool:
    """True when a key is present, without raising. Used to skip optional paths."""
    return get(name, required=False) is not None


def redact(value: str) -> str:
    """Safe-to-display form, for the rare case a key must be acknowledged at all."""
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}{'*' * (len(value) - 8)}{value[-4:]}"
