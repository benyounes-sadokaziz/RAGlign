"""Vendor the second corpus: unstructured public-domain prose.

Why a second corpus at all: every v1 conclusion came from FastAPI markdown, and
the heading-aware chunker won there partly because that corpus is dense with
headings. A strategy that exploits structure cannot be fairly judged on a corpus
made of structure. This one has none: flowing paragraphs, no markup, no code.

Four works from different domains (physics, biology, economics, memoir) rather
than one long book -- a corpus where every document discusses the same subject
makes retrieval artificially hard, because every chunk looks equally relevant to
every query.

Documents are chapters, not whole books: a 300 KB document would be one
retrieval unit far larger than any answer, which is not a realistic RAG setup.

Run once: .venv/Scripts/python.exe scripts/vendor_prose_corpus.py
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "corpus" / "prose"

# (gutenberg id, slug, title, max documents to keep)
# Caps are tuned so the total lands near corpus/fastapi's 540 KB. Matching size
# is deliberate: comparing a 540 KB corpus against a 1.5 MB one would confound
# "unstructured vs structured" with "small vs large", and corpus size affects
# retrieval difficulty on its own (more chunks = more chances to rank a
# distractor first).
WORKS = [
    (30155, "relativity", "Relativity: The Special and General Theory", 18),
    (2009, "origin", "On the Origin of Species", 18),
    (3300, "wealth", "An Inquiry into the Nature and Causes of the Wealth of Nations", 18),
    (20203, "franklin", "Autobiography of Benjamin Franklin", 16),
]

# Gutenberg wraps every text in a licence header and footer. Left in, that
# boilerplate would be chunked and indexed like content -- and being nearly
# identical across works, it would retrieve for everything.
START_RE = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I | re.S)
END_RE = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I | re.S)

# Chapter/section openers as they actually appear in these texts.
CHAPTER_RE = re.compile(
    r"^[ \t]*((?:CHAPTER|Chapter|SECTION|Section|BOOK|Book|PART|Part)\s+"
    r"(?:[IVXLCDM]+|\d+|[A-Z][a-z]+)\b.*|[IVXLCDM]{1,7}\.?)[ \t]*$",
    re.M,
)


def fetch(book_id: int) -> str:
    """Download one work, preferring urllib and falling back to curl.

    The fallback is not defensive padding: this machine's Python SSL stack fails
    on gutenberg.org ("ASN1: NOT_ENOUGH_DATA") while system curl succeeds on the
    same URL. Vendoring runs once, so a working path matters more than a pure
    stdlib one.
    """
    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "RAGlign-corpus-vendor/1.0"})
        with urllib.request.urlopen(req, timeout=90) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception:
        out = subprocess.run(
            ["curl", "-sSL", "--max-time", "120", url],
            capture_output=True,
            check=True,
        )
        return out.stdout.decode("utf-8", errors="replace")


def strip_boilerplate(text: str) -> str:
    m = START_RE.search(text)
    if m:
        text = text[m.end() :]
    m = END_RE.search(text)
    if m:
        text = text[: m.start()]
    return text.strip("\n")


def split_chapters(text: str, min_chars: int = 2000, max_chars: int = 8000) -> list[str]:
    """Split at chapter markers; drop fragments too short to be real chapters.

    Over-long chapters are split at paragraph boundaries rather than truncated:
    dropping text would silently make parts of the corpus unreachable, which is
    exactly the failure mode `coverage_ratio` exists to catch.
    """
    cuts = [m.start() for m in CHAPTER_RE.finditer(text)]
    cuts = [0] + cuts + [len(text)]
    raw = [text[a:b].strip("\n") for a, b in zip(cuts, cuts[1:])]

    out: list[str] = []
    for piece in raw:
        if len(piece) < min_chars:
            continue
        while len(piece) > max_chars:
            # Prefer a paragraph break, then a sentence end, then a line break.
            # A hard character cut is the last resort because it leaves documents
            # beginning mid-word ("ne, we brought them all away"), which is not
            # something a real corpus looks like and makes authored quotes near
            # the boundary read as corrupt.
            split_at = piece.rfind("\n\n", max_chars // 2, max_chars)
            if split_at <= 0:
                m = None
                for m in re.finditer(r"[.!?][\"')\]]*\s", piece[: max_chars]):
                    pass
                split_at = m.end() if m else -1
            if split_at <= 0:
                split_at = piece.rfind("\n", max_chars // 2, max_chars)
            if split_at <= 0:
                split_at = max_chars
            out.append(piece[:split_at].strip("\n"))
            piece = piece[split_at:].strip("\n")
        if len(piece) >= min_chars:
            out.append(piece)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__ or "")
    ap.add_argument(
        "--force",
        action="store_true",
        help="re-download and overwrite an existing corpus (invalidates ground truth)",
    )
    args = ap.parse_args()

    # Refusing by default is not politeness, it is correctness. Ground truth is
    # character offsets into these exact bytes; re-vendoring can shift them (a
    # different upstream revision, a changed split heuristic) and every authored
    # quote would then either fail to resolve or, worse, resolve somewhere else.
    # A one-shot script that silently re-runs is a footgun pointed at the data
    # the whole project depends on.
    existing = list(DEST.rglob("*.txt"))
    if existing and not args.force:
        print(f"corpus/prose already holds {len(existing)} documents -- refusing to overwrite.")
        print("Ground-truth spans are character offsets into these exact bytes; re-vendoring")
        print("can shift them and silently invalidate data/qa/prose/. Pass --force if you")
        print("intend that, then re-run scripts/check_qa.py --corpus prose to confirm the")
        print("quotes still resolve.")
        return 1

    DEST.mkdir(parents=True, exist_ok=True)
    manifest: list[str] = []
    total = 0

    for book_id, slug, title, cap in WORKS:
        print(f"  fetching {slug} ({title[:40]}) ...", end=" ", flush=True)
        try:
            text = fetch(book_id)
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED: {exc}")
            return 1

        chapters = split_chapters(strip_boilerplate(text))[:cap]
        if not chapters:
            print("FAILED: no chapters matched")
            return 1

        work_dir = DEST / slug
        work_dir.mkdir(parents=True, exist_ok=True)
        for i, chapter in enumerate(chapters, 1):
            # LF only, written as bytes: offsets must not depend on platform
            # newline translation (see TC-11).
            path = work_dir / f"{i:03d}.txt"
            path.write_bytes((chapter.replace("\r\n", "\n").rstrip() + "\n").encode("utf-8"))

        size = sum(len(c) for c in chapters)
        total += size
        print(f"{len(chapters)} chapters, {size:,} chars")
        manifest.append(f"| {title} | `{slug}/` | {book_id} | {len(chapters)} | {size:,} |")

    (ROOT / "corpus" / "PROVENANCE_PROSE.md").write_text(
        "# Prose Corpus Provenance\n\n"
        "Unstructured public-domain prose: flowing paragraphs, no markup, no code.\n"
        "Vendored as the structural opposite of `corpus/fastapi`, so that chunking\n"
        "strategies which exploit document structure can be tested on a corpus that\n"
        "has none.\n\n"
        "Four works from different domains rather than one long book: a corpus where\n"
        "every document covers the same subject makes retrieval artificially hard,\n"
        "since every chunk looks equally relevant to every query.\n\n"
        "| Work | Path | Gutenberg ID | Documents | Chars |\n|---|---|---|---|---|\n"
        + "\n".join(manifest)
        + f"\n\n**Total: {total:,} chars**\n\n"
        "Source: https://www.gutenberg.org (public domain in the US).\n"
        "Gutenberg licence header/footer stripped: left in, that boilerplate would be\n"
        "indexed as content and, being near-identical across works, would retrieve for\n"
        "every query.\n\n"
        "Documents are chapters, not whole books -- a 300 KB document would be a single\n"
        "retrieval unit far larger than any answer.\n\n"
        "Reproduce: `.venv/Scripts/python.exe scripts/vendor_prose_corpus.py`\n",
        encoding="utf-8",
    )
    print(f"\ntotal: {total:,} chars across {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
