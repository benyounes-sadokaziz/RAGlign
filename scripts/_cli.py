"""Shared CLI plumbing for the scripts.

Every entry point needs the same three things: the repo on sys.path, a console
that can print UTF-8 on Windows, and a --corpus argument. Centralised so adding
a corpus never means touching five argument parsers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from raglign import corpora  # noqa: E402


def parser(description: str, *, qa: bool = True) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--corpus",
        default=corpora.DEFAULT_CORPUS,
        choices=corpora.names(),
        help="which registered corpus to run against",
    )
    if qa:
        p.add_argument(
            "--qa",
            default=None,
            help="QA set filename inside data/qa/<corpus>/ (default: the corpus's long-span set)",
        )
    return p


def resolve_qa(corpus: str, qa: str | None) -> str:
    """Pick the QA set, defaulting to the long-span one.

    The long-span set is the default because the short-span sets are saturated
    (TC-14) -- they show the pipeline works but cannot separate configs, so
    defaulting to one would make every comparison look like a tie.
    """
    sets = corpora.get(corpus).qa_sets()
    if qa:
        if qa not in sets:
            raise SystemExit(f"no QA set {qa!r} for corpus {corpus!r}; have: {', '.join(sets) or 'none'}")
        return qa
    longspan = [s for s in sets if "longspan" in s]
    if longspan:
        return longspan[0]
    if sets:
        return sets[0]
    raise SystemExit(f"corpus {corpus!r} has no QA sets in data/qa/{corpus}/")
