"""Verify every chunker preserves exact character offsets, on any corpus.

Every chunk of every strategy is re-sliced out of its source document and
compared byte-for-byte. Coverage is reported too: a strategy that silently drops
content would otherwise look fine here while making ground-truth spans
unreachable by construction.

Run: .venv/Scripts/python.exe scripts/check_chunkers.py --corpus prose
"""

from __future__ import annotations

import time

from _cli import parser

from raglign import corpora
from raglign.chunking import (
    FixedSizeChunker,
    HeadingChunker,
    RecursiveChunker,
    SemanticChunker,
    coverage_ratio,
)
from raglign.loader import corpus_fingerprint, load_corpus


def main() -> int:
    args = parser(__doc__, qa=False).parse_args()
    c = corpora.get(args.corpus)
    docs = load_corpus(args.corpus)

    print(f"corpus     : {c.name} -- {c.description}")
    print(f"structure  : {c.structure}")
    print(f"size       : {len(docs)} docs, {sum(d.length for d in docs):,} chars")
    print(f"fingerprint: {corpus_fingerprint(docs)}\n")

    chunkers = [
        FixedSizeChunker(size=800, overlap=100),
        FixedSizeChunker(size=300, overlap=50),
        RecursiveChunker(size=800, overlap=100),
        HeadingChunker(max_size=1200, min_size=200),
        SemanticChunker(percentile=90.0, max_size=1200, min_size=200),
    ]

    print(f"{'strategy':<24} {'chunks':>7} {'mean':>6} {'p10':>6} {'p90':>6} {'cover':>7} {'sec':>7}")
    print("-" * 69)

    failures = 0
    for chunker in chunkers:
        t0 = time.perf_counter()
        all_chunks, covers = [], []
        try:
            for doc in docs:
                chunks = chunker.split(doc)
                for ch in chunks:
                    ch.validate_against(doc)
                covers.append(coverage_ratio(doc, chunks))
                all_chunks.extend(chunks)
        except Exception as exc:  # noqa: BLE001
            print(f"{chunker.name:<24} FAILED: {type(exc).__name__}: {exc}")
            failures += 1
            continue

        lens = sorted(ch.length for ch in all_chunks)
        n = len(lens)
        print(
            f"{chunker.name:<24} {n:>7} {sum(lens) / n:>6.0f} {lens[n // 10]:>6} "
            f"{lens[(9 * n) // 10]:>6} {sum(covers) / len(covers):>6.1%} "
            f"{time.perf_counter() - t0:>7.2f}"
        )

    print()
    if failures:
        print(f"FAIL: {failures} strategy/strategies broke the offset invariant")
        return 1
    print("PASS: every chunk of every strategy re-slices to itself in the source")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
