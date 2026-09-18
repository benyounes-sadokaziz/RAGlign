"""Scaffold a QA set: pick candidate evidence passages, leave the questions blank.

Hand-authoring ground truth has one failure mode that matters: retyping the
quote. A quote copied by hand can pick up a smart quote, a collapsed double
space, or a dropped line break, and then it will not resolve -- the TC-2 gate
catches it, but only after the work is done and the run has been refused.

So the machine copies the quote and the human writes the question. The passage
is sliced straight out of the corpus, so it is verbatim by construction; the
`question` and `answer` fields come out empty for a person to fill in.

Selection criteria, and why each one:

- 300-700 characters: comparable to chunk size, which is the regime where
  chunking actually matters (TC-14). Shorter evidence produces a saturated set.
- multi-sentence: a single sentence is usually answerable by keyword overlap,
  which makes retrieval trivially easy and the benchmark uninformative.
- unique within its document: a passage appearing twice cannot be resolved to
  one span without an explicit occurrence index.
- at most one per document, and documents already used are skipped: reusing a
  document concentrates the ground truth in a few files, so a retriever that
  happens to like those files looks better than it is.

Run: .venv/Scripts/python.exe scripts/author_qa.py --corpus prose --count 30
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from _cli import parser

from raglign import corpora
from raglign.loader import load_corpus
from raglign.qa import QA_ROOT

# Lines that are structure or boilerplate rather than prose evidence.
_SKIP_PREFIX = (
    "#", "|", "!!!", "///", "```", "~~~", "CHAPTER", "SECTION", "BOOK", "PART",
    "FOOTNOTE", "Appendices", "* * *", "---", "===",
)
_SENT_END = re.compile(r"[.!?][\"')\]]*(\s|$)")


_BLANK = re.compile(r"\n[ \t]*\n")


def _blocks(text: str) -> list[tuple[int, int]]:
    """Paragraph spans as (start, end) offsets, skipping fenced code.

    Offsets rather than split strings: a candidate is then `text[start:end]`,
    exact by construction. Rejoining `split("\\n\\n")` pieces would corrupt any
    passage separated by three or more newlines, and that corruption would only
    surface later as an unresolvable quote.
    """
    spans: list[tuple[int, int]] = []
    pos, in_fence = 0, False
    for m in list(_BLANK.finditer(text)) + [None]:
        end = m.start() if m else len(text)
        raw = text[pos:end]
        stripped = raw.strip()
        if stripped.startswith(("```", "~~~")):
            in_fence = not in_fence
        elif stripped and not in_fence and not stripped.startswith(_SKIP_PREFIX):
            lead = len(raw) - len(raw.lstrip())
            trail = len(raw) - len(raw.rstrip())
            spans.append((pos + lead, end - trail))
        pos = m.end() if m else len(text)
    return spans


def candidate_passages(text: str, lo: int, hi: int, max_blocks: int = 3) -> list[str]:
    """Passages of one to `max_blocks` adjacent paragraphs, in the size range.

    Allowing adjacent paragraphs matters on documentation corpora: individual
    paragraphs there are often well under 300 characters, and requiring a single
    block would restrict ground truth to the handful of long ones -- which are
    disproportionately bullet lists, biasing the question set toward one shape
    of evidence.
    """
    spans = _blocks(text)
    out: list[str] = []
    for i in range(len(spans)):
        for j in range(i, min(i + max_blocks, len(spans))):
            start, end = spans[i][0], spans[j][1]
            passage = text[start:end]
            if not (lo < len(passage) < hi):
                continue
            bullets = sum(1 for ln in passage.splitlines() if ln.strip().startswith(("*", "-")))
            if len(_SENT_END.findall(passage)) < 2 and bullets < 3:
                continue
            if not passage[:1].isupper() and bullets < 3:
                continue
            out.append(passage)
    return out


def used_documents(corpus: str) -> set[str]:
    """Documents already carrying ground truth, so new items go elsewhere."""
    used: set[str] = set()
    for path in (QA_ROOT / corpus).glob("*.json"):
        if path.name.startswith("_"):
            continue
        for rec in json.loads(path.read_text(encoding="utf-8")):
            for q in rec["quotes"]:
                used.add(q["doc"])
    return used


def used_quotes(corpus: str) -> set[str]:
    """Exact passages already used, so --allow-reuse cannot duplicate evidence."""
    out: set[str] = set()
    for path in (QA_ROOT / corpus).glob("*.json"):
        if path.name.startswith("_"):
            continue
        for rec in json.loads(path.read_text(encoding="utf-8")):
            for q in rec["quotes"]:
                out.add(q["quote"])
    return out


def main() -> int:
    p = parser(__doc__, qa=False)
    p.add_argument("--count", type=int, default=30)
    p.add_argument("--min-chars", type=int, default=300)
    p.add_argument("--max-chars", type=int, default=700)
    p.add_argument("--prefix", default=None, help="id prefix (default: corpus initial)")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--out", default="_scaffold.json")
    p.add_argument(
        "--allow-reuse",
        action="store_true",
        help="allow a second passage from a document that already has ground truth "
             "(still never the same passage twice)",
    )
    args = p.parse_args()

    docs = load_corpus(args.corpus)
    already = set() if args.allow_reuse else used_documents(args.corpus)
    seen_quotes = used_quotes(args.corpus)
    rng = random.Random(args.seed)
    rng.shuffle(docs)

    prefix = args.prefix or args.corpus[0].upper()
    picked: list[dict] = []
    for doc in docs:
        if len(picked) >= args.count:
            break
        if doc.id in already:
            continue
        cands = [
            c
            for c in candidate_passages(doc.text, args.min_chars, args.max_chars)
            if doc.text.count(c) == 1 and c not in seen_quotes
        ]
        if not cands:
            continue
        quote = rng.choice(cands)
        seen_quotes.add(quote)
        picked.append(
            {
                "id": f"{prefix}{len(picked) + 1:03d}",
                "question": "",
                "answer": "",
                "quotes": [{"doc": doc.id, "quote": quote}],
                "tags": ["long-span"],
            }
        )

    out_path = QA_ROOT / args.corpus / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(picked, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"corpus    : {args.corpus} ({len(docs)} docs, {len(already)} already used)")
    print(f"scaffolded: {len(picked)} passages -> {out_path.relative_to(out_path.parents[3])}\n")
    for item in picked:
        q = item["quotes"][0]
        body = " ".join(q["quote"].split())
        print(f"{item['id']}  {q['doc']}  [{len(q['quote'])}ch]")
        print(f"      {body[:190]}")
        if len(body) > 240:
            print(f"      ... {body[-60:]}")
        print()

    if len(picked) < args.count:
        print(f"note: only {len(picked)} of {args.count} requested -- ran out of unused documents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
