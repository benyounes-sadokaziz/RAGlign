"""Ground-truth QA sets: on-disk schema, and quote -> span resolution.

The central idea (TC-2): a ground-truth item never stores character offsets.
It stores a quote copied verbatim out of the source document. Offsets are
*derived* by locating that quote at load time.

This buys three things that hand-typed offsets do not:

1. **Self-validation.** A quote that no longer appears verbatim fails loudly.
   The corpus is its own checksum, so a corpus edit cannot silently rot the
   ground truth into pointing at the wrong text.
2. **Authoring speed.** Copy-paste a sentence instead of counting characters.
3. **Safe generation.** When an LLM generates QA items (Phase 3), a
   hallucinated quote cannot be located and is rejected automatically. The
   generator is therefore unable to invent ground truth -- the rejection rate
   becomes a reportable quality signal rather than silent contamination.

Ambiguity is treated as an error, not resolved by guessing: a quote occurring
more than once must say which occurrence it means. Picking the first match
silently would place ground truth in the wrong section of the document.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from .models import Document, QAItem, Span

QA_ROOT = Path(__file__).resolve().parent.parent / "data" / "qa"

# Quotes shorter than this are too weak to be evidence: a 10-character string
# may appear anywhere and pins down no particular passage.
MIN_QUOTE_CHARS = 25


class QuoteRef(BaseModel):
    """A pointer to evidence, expressed as text rather than as a position."""

    doc: str = Field(description="document id, e.g. 'tutorial/query-params.md'")
    quote: str = Field(description="text copied verbatim from that document")
    occurrence: int = Field(
        default=0,
        description="which occurrence to use when the quote is not unique (0-based)",
    )

    @field_validator("quote")
    @classmethod
    def _long_enough(cls, v: str) -> str:
        if len(v.strip()) < MIN_QUOTE_CHARS:
            raise ValueError(f"quote shorter than {MIN_QUOTE_CHARS} chars: {v!r}")
        return v


class QARecord(BaseModel):
    """One question as authored on disk."""

    id: str
    question: str
    answer: str
    quotes: list[QuoteRef] = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)


@dataclass
class Rejection:
    record_id: str
    reason: str
    detail: str


@dataclass
class QASet:
    items: list[QAItem]
    rejections: list[Rejection]

    @property
    def acceptance_rate(self) -> float:
        total = len(self.items) + len(self.rejections)
        return len(self.items) / total if total else 0.0

    def __len__(self) -> int:
        return len(self.items)


def resolve_quote(doc: Document, ref: QuoteRef) -> Span:
    """Locate `ref.quote` in `doc` and return its span.

    Raises ValueError with an actionable reason; callers collect these as
    rejections rather than crashing the run.
    """
    starts: list[int] = []
    idx = doc.text.find(ref.quote)
    while idx != -1:
        starts.append(idx)
        idx = doc.text.find(ref.quote, idx + 1)

    if not starts:
        raise ValueError(f"quote not found verbatim in {doc.id}")
    if len(starts) > 1 and ref.occurrence == 0 and len(starts) > 1:
        # Only an error if the author did not disambiguate.
        if ref.occurrence >= len(starts):
            raise ValueError(f"occurrence {ref.occurrence} but only {len(starts)} matches")
    if ref.occurrence >= len(starts):
        raise ValueError(f"occurrence {ref.occurrence} out of range ({len(starts)} matches)")

    start = starts[ref.occurrence]
    return Span(doc_id=doc.id, start=start, end=start + len(ref.quote))


def load_qa_set(path: Path | str, docs: dict[str, Document], *, strict_unique: bool = True) -> QASet:
    """Load a QA JSON file and resolve every quote to a span.

    Args:
        strict_unique: reject non-unique quotes that did not declare an
            occurrence index. Keep this on for authored sets; the flag exists
            so a generated set can be triaged rather than rejected wholesale.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    records = [QARecord(**r) for r in raw]

    items: list[QAItem] = []
    rejections: list[Rejection] = []
    seen_ids: set[str] = set()

    for rec in records:
        if rec.id in seen_ids:
            rejections.append(Rejection(rec.id, "duplicate_id", "id already used"))
            continue
        seen_ids.add(rec.id)

        spans: list[Span] = []
        error: Rejection | None = None
        for ref in rec.quotes:
            doc = docs.get(ref.doc)
            if doc is None:
                error = Rejection(rec.id, "unknown_doc", ref.doc)
                break
            if strict_unique and ref.occurrence == 0 and doc.text.count(ref.quote) > 1:
                error = Rejection(
                    rec.id,
                    "ambiguous_quote",
                    f"{doc.text.count(ref.quote)} matches in {ref.doc}; set 'occurrence'",
                )
                break
            try:
                spans.append(resolve_quote(doc, ref))
            except ValueError as exc:
                error = Rejection(rec.id, "unresolved_quote", f"{ref.doc}: {exc}")
                break

        if error is not None:
            rejections.append(error)
            continue

        items.append(
            QAItem(
                id=rec.id,
                question=rec.question,
                answer=rec.answer,
                spans=tuple(spans),
                meta={"tags": rec.tags},
            )
        )

    return QASet(items=items, rejections=rejections)
