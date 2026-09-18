"""Resolve every authored quote to a character span, for one corpus.

The TC-2 quality gate. A quote that no longer appears verbatim, or appears in
more than one place without disambiguation, is rejected here rather than
silently pointing at the wrong passage.

Run: .venv/Scripts/python.exe scripts/check_qa.py --corpus prose
"""

from __future__ import annotations

from _cli import parser, resolve_qa

from raglign.loader import corpus_fingerprint, index_by_id, load_corpus
from raglign.qa import load_qa_set, qa_path


def main() -> int:
    args = parser(__doc__).parse_args()
    qa_file = resolve_qa(args.corpus, args.qa)

    docs = load_corpus(args.corpus)
    print(f"corpus     : {args.corpus} ({len(docs)} docs)")
    print(f"fingerprint: {corpus_fingerprint(docs)}")
    print(f"qa file    : {qa_file}\n")

    qa = load_qa_set(qa_path(args.corpus, qa_file), index_by_id(docs))

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
    raise SystemExit(main())
