# Prose Corpus Provenance

Unstructured public-domain prose: flowing paragraphs, no markup, no code.
Vendored as the structural opposite of `corpus/fastapi`, so that chunking
strategies which exploit document structure can be tested on a corpus that
has none.

Four works from different domains rather than one long book: a corpus where
every document covers the same subject makes retrieval artificially hard,
since every chunk looks equally relevant to every query.

| Work | Path | Gutenberg ID | Documents | Chars |
|---|---|---|---|---|
| Relativity: The Special and General Theory | `relativity/` | 30155 | 18 | 131,914 |
| On the Origin of Species | `origin/` | 2009 | 18 | 127,732 |
| An Inquiry into the Nature and Causes of the Wealth of Nations | `wealth/` | 3300 | 18 | 127,692 |
| Autobiography of Benjamin Franklin | `franklin/` | 20203 | 16 | 126,238 |

**Total: 513,576 chars**

Source: https://www.gutenberg.org (public domain in the US).
Gutenberg licence header/footer stripped: left in, that boilerplate would be
indexed as content and, being near-identical across works, would retrieve for
every query.

Documents are chapters, not whole books -- a 300 KB document would be a single
retrieval unit far larger than any answer.

Reproduce: `.venv/Scripts/python.exe scripts/vendor_prose_corpus.py`
