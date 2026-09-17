"""Step 2 verification: resolve every authored quote to a character span.

This is the TC-2 quality gate. A quote that no longer appears verbatim, or that
appears in more than one place without disambiguation, is rejected here rather
than silently pointing at the wrong passage.

Run: .venv/Scripts/python.exe scripts/check_qa.py [qa_file]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from raglign.loader import corpus_fingerprint, index_by_id, load_documents
from raglign.qa import QA_ROOT, load_qa_set


def main(argv: list[str]) -> int:
    qa_path = Path(argv[1]) if len(argv) > 1 else QA_ROOT / "handwritten_v1.json"

    docs = load_documents()
    by_id = index_by_id(docs)
    print(f"corpus fingerprint: {corpus_fingerprint(docs)}")
    print(f"qa file: {qa_path.name}\n")

    qa = load_qa_set(qa_path, by_id)

    print(f"accepted : {len(qa.items)}")
    print(f"rejected : {len(qa.rejections)}")
    print(f"rate     : {qa.acceptance_rate:.1%}\n")

    if qa.rejections:
        print("REJECTIONS")
        for r in qa.rejections:
            print(f"  {r.record_id:<6} {r.reason:<18} {r.detail}")
        print()

    spans = [s for it in qa.items for s in it.spans]
    if spans:
        lens = sorted(s.length for s in spans)
        print(f"spans    : {len(spans)} across {len({s.doc_id for s in spans})} documents")
        print(f"span len : min {lens[0]}  median {lens[len(lens) // 2]}  max {lens[-1]}")
        multi = [it.id for it in qa.items if len(it.spans) > 1]
        print(f"multi-span items: {len(multi)} {multi}")

    print()
    if qa.rejections:
        print("FAIL: fix the quotes above (copy them verbatim from the source)")
        return 1
    print("PASS: every quote resolved to an unambiguous character span")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
