"""Step 1 verification: do all three chunkers preserve exact character offsets?

Every chunk of every strategy is re-sliced out of its source document and
compared byte-for-byte against the chunk's own text. Coverage is reported too:
a strategy that quietly drops content would otherwise look fine here while
making some ground-truth spans unreachable by construction.

Run: .venv/Scripts/python.exe scripts/check_chunkers.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from raglign.chunking import FixedSizeChunker, HeadingChunker, RecursiveChunker, SemanticChunker, coverage_ratio
from raglign.loader import corpus_fingerprint, load_documents


def main() -> int:
    docs = load_documents()
    print(f"corpus: {len(docs)} docs, {sum(d.length for d in docs):,} chars")
    print(f"fingerprint: {corpus_fingerprint(docs)}\n")

    chunkers = [
        FixedSizeChunker(size=800, overlap=100),
        RecursiveChunker(size=800, overlap=100),
        HeadingChunker(max_size=1200, min_size=200),
        SemanticChunker(percentile=90.0, max_size=1200, min_size=200),
    ]

    print(f"{'strategy':<24} {'chunks':>7} {'mean':>6} {'p10':>6} {'p90':>6} {'cover':>7} {'sec':>6}")
    print("-" * 68)

    failures = 0
    for chunker in chunkers:
        t0 = time.perf_counter()
        all_chunks = []
        covers = []
        try:
            for doc in docs:
                chunks = chunker.split(doc)
                for c in chunks:
                    c.validate_against(doc)  # the invariant, per chunk
                covers.append(coverage_ratio(doc, chunks))
                all_chunks.extend(chunks)
        except Exception as exc:  # noqa: BLE001 - surface the whole failure
            print(f"{chunker.name:<24} FAILED: {type(exc).__name__}: {exc}")
            failures += 1
            continue

        elapsed = time.perf_counter() - t0
        lens = sorted(c.length for c in all_chunks)
        n = len(lens)
        print(
            f"{chunker.name:<24} {n:>7} {sum(lens) / n:>6.0f} "
            f"{lens[n // 10]:>6} {lens[(9 * n) // 10]:>6} "
            f"{sum(covers) / len(covers):>6.1%} {elapsed:>6.2f}"
        )

    print()
    if failures:
        print(f"FAIL: {failures} strategy/strategies broke the offset invariant")
        return 1
    print("PASS: every chunk of every strategy re-slices to itself in the source")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
