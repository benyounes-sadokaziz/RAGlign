"""Corpus loading.

One rule: read bytes, decode UTF-8, hand the text back untouched. No stripping,
no whitespace collapsing, no newline normalisation beyond what is forced by the
decode. Every character offset in the ground truth is an index into exactly what
this module returns.

Newlines are the one real trap on Windows: reading in text mode would translate
CRLF to LF and shift every offset after the first line break. All reads here are
binary + explicit decode, so what is on disk is what is indexed. Git is
configured with core.autocrlf=false for the same reason.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from . import corpora
from .models import Document


def load_corpus(name: str = corpora.DEFAULT_CORPUS) -> list[Document]:
    """Load a registered corpus by name."""
    c = corpora.get(name)
    return load_documents(c.path, c.pattern)


def load_documents(root: Path | str, pattern: str = "**/*.md") -> list[Document]:
    """Load every matching file under `root` as a Document.

    Document ids are POSIX-style paths relative to `root`, so they are stable
    across operating systems and readable in results tables.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"corpus root not found: {root}")

    docs: list[Document] = []
    for path in sorted(root.glob(pattern)):
        if not path.is_file():
            continue
        text = path.read_bytes().decode("utf-8")
        docs.append(Document(id=path.relative_to(root).as_posix(), path=str(path), text=text))

    if not docs:
        raise FileNotFoundError(f"no files matched {pattern!r} under {root}")
    return docs


def corpus_fingerprint(docs: list[Document]) -> str:
    """Stable hash of the loaded corpus.

    Recorded in every run manifest. If this changes, previously computed spans
    may no longer point where they did, and old results are not comparable to
    new ones -- this is the tripwire that makes that detectable.
    """
    h = hashlib.sha256()
    for doc in sorted(docs, key=lambda d: d.id):
        h.update(doc.id.encode("utf-8"))
        h.update(doc.text.encode("utf-8"))
    return h.hexdigest()[:16]


def index_by_id(docs: list[Document]) -> dict[str, Document]:
    return {d.id: d for d in docs}
