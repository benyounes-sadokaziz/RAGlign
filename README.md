# RAGlign

**Point it at your documents. It tells you which RAG configuration to use, and why.**

Building a RAG pipeline means choosing how to chunk your documents, how to retrieve, and
whether to rerank. Those choices interact, the right answer depends on your corpus, and
most teams pick by intuition. RAGlign measures instead — and it measures in a way that
standard RAG evaluation gets wrong.

---

## How it works

```
your documents  ──►  30 pipelines built and evaluated  ──►  ranked by your priorities
     +                (5 chunkers × 3 retrievers                       │
  ~40 questions         × reranker on/off)                             ▼
                                                        "use heading-based chunking
                                                         with vector search — here's
                                                         the evidence, and here's
                                                         what's limiting you"
```

**1. Give it a corpus.** Drop your documents in `corpus/<name>/`. Markdown, plain text,
whatever — it reads bytes and never reformats them.

**2. Give it questions.** Around 40 works. Each one carries a short quote copied out of
your own documents — the passage that answers it. You don't record character positions;
the tool finds the quote and derives them. If a quote doesn't appear verbatim, it's
rejected, so the ground truth can't silently rot.

`scripts/author_qa.py` does most of this for you: it pulls candidate passages straight out
of your corpus so the quotes are correct by construction, and you only write the questions.

**3. Run the grid.** It builds all 30 pipelines, runs every question through every one, and
scores them.

**4. Say what you care about.** Move the Quality / Latency / Context sliders. The ranking
updates instantly — the measurements never change, only how they're weighted.

```
config                        MRR   hit@5   ms/query   context
heading1200/dense+rerank    0.683   0.805       2844      3624
heading1200/dense           0.589   0.756        0.4      3428   ← if latency matters
fixed300/bm25               0.222   0.439        4.9      1486
```

**5. Find out what's actually limiting you.** For every question it missed, the tool names
the stage that lost it — the chunker, the retriever, or the reranker — because each needs
a different fix.

---

## Why the measurement is the interesting part

To score retrieval you need ground truth: which piece of text answers this question?
Almost everyone records that as a **chunk ID** — "the answer is in chunk 57."

But chunk 57 only exists inside one chunking strategy. Chop the documents differently and
it isn't there any more. So when you compare strategy B against ground truth written while
using strategy A, you're measuring *how similar B's cut points are to A's* — not how well
B retrieves.

RAGlign stores ground truth as a **character span in the original document** instead:

```
ground truth   tutorial/body.md [4210:4580]     ← a property of your documents
chunk from A   tutorial/body.md [3800:4600]     ← contains it     → hit
chunk from B   tutorial/body.md [4100:4400]     ← contains 51%    → depends on threshold
```

Now every strategy is asked the same question — *did you retrieve the text that answers
this?* — and none of them wrote the question.

### How big is the bias? Larger than the thing it's measuring

`scripts/validation_study.py` scores every strategy twice against the same questions and
**the same retrievals** — once by chunk-ID matching, once by span alignment. Any difference
comes from the ground truth alone. The ground truth is then re-authored against each
strategy in turn, because if the effect were an artifact of a convenient choice, rotating
the author would dissolve it.

**What authoring the ground truth is worth — the author's own hit@5 minus the others':**

| corpus | questions | chunk-ID ground truth | span alignment |
|---|---:|---|---|
| FastAPI docs | 41 | **+0.569 … +0.675** | +0.016 … +0.049 |
| Prose | 43 | **+0.589 … +0.744** | −0.116 … +0.101 |

Whoever writes the ground truth wins, by roughly **0.6 hit@5** — several times the real
quality spread between the strategies. **The recommended configuration flips in 3 of 4
rotations.** It replicates on a corpus with no markup at all, so it isn't a property of
one document style.

*"Isn't that just strict matching?"* Relax what counts as "the gold chunk" from IoU ≥ 0.9
down to 0.3 and the bias decays smoothly to zero — chunk-ID matching is unbiased only once
it's loosened into being span overlap, which is the argument, not a counterexample.

---

## What span-level ground truth makes possible

### Knowing which component to fix

A score tells you a question failed. It can't tell you *which stage* lost it — and the
stages need opposite fixes.

| cause | what to do |
|---|---|
| `chunker_destroyed` | no chunk covers the evidence → change chunking; no embedding model can help |
| `chunker_degraded` | evidence survived only partially → increase chunk size first |
| `retriever_missed` | indexed and intact but never retrieved → try hybrid, or a reranker |
| `reranker_demoted` | first stage had it, reranker pushed it out → drop or retune reranking |
| `ranked_low` | found, but not first → a reranker is the targeted fix |

On a configuration using 300-character chunks:

```
fixed300/bm25    chunker 8   retriever 0   ranked low 1   ok 2
```

**8 of 11 failures are the chunker. Zero are the retriever.** Standard evaluation reports
this as a low score and sends you shopping for embedding models. The chunks are smaller
than the answers, and no retriever can fix that.

Asking *"could any chunk in this index have answered this?"* is only possible when ground
truth is a position. A chunk ID can't express it.

### Seeing the ceiling before retrieval runs

Oracle reachability — the share of evidence some chunk could satisfy, ignoring the
retriever entirely:

| chunker | τ=0.5 | τ=0.7 |
|---|---:|---:|
| `fixed_800` | 1.000 | 0.927 |
| `heading_1200` | 1.000 | 0.829 |
| `fixed_300` | 0.976 | **0.463** |

`fixed_300` destroys over half its evidence *before retrieval happens*.

---

## The answer depends on your corpus

Same 30 configurations, two structurally opposite corpora:

| corpus | structure | best configuration |
|---|---|---|
| FastAPI docs | headings, code blocks, lists | `heading1200/dense` |
| Public-domain prose | flowing paragraphs, no markup | `recursive800/dense` |

Config choice does not transfer across document structure. That's the condition that makes
a per-corpus optimizer worth using rather than a blog post naming a default.

---

## Does better retrieval actually produce better answers?

Everything above measures retrieval, which only matters if it predicts what reaches the
user. `scripts/run_generation.py` closes the loop: a real LLM answers from each
configuration's retrieved context, and a **different** model judges the answers — LLM
judges score their own output more favourably.

```
retrieval MRR vs gold recall        r = +0.984
retrieval MRR vs span grounding     r = +0.968
retrieval MRR vs faithful coverage  r = +0.951
```

**Retrieval quality predicts answer quality at r ≈ +0.95.** It runs without an API key too,
via `--generator extractive`, which makes the top-ranked chunk the answer — a genuine floor,
since a reader cannot state what was never retrieved.

---

## Running it

**Requirements:** Python 3.11+, ~2 GB disk, no GPU. First run downloads a ~130 MB embedding
model; everything after is offline.

```bash
git clone https://github.com/benyounes-sadokaziz/RAGlign && cd RAGlign
python -m venv .venv

# Windows
.venv\Scripts\python.exe -m pip install -r requirements.txt
set P=.venv\Scripts\python.exe

# macOS / Linux
.venv/bin/python -m pip install -r requirements.txt
export P=.venv/bin/python
```

### Fastest result (~2 minutes)

```bash
$P scripts/validation_study.py
```

Self-contained — builds its own indexes and prints the bias table. Nothing else needs to
have run first.

### Full pipeline

```bash
$P scripts/check_chunkers.py   # ~1 min   do all chunkers preserve exact offsets?
$P scripts/check_qa.py         # instant  does every quote still resolve?
$P scripts/run_grid.py         # ~25 min  evaluate all 30 configs  ← writes runs/
$P scripts/recommend.py        # instant  ranked table + explanation
$P scripts/diagnose.py         # instant  which stage lost each question
$P scripts/compare_corpora.py  # instant  does the winner transfer?

$P -m pytest -q                            # 86 tests, all offline
.venv/Scripts/streamlit.exe run app.py     # dashboard on :8501
```

Most of the 25 minutes is the 15 reranker configurations at ~1.5 s/query. Embeddings are
cached, so re-runs are far quicker. Add `--corpus prose` to any command; `fastapi` is the
default.

### Using your own documents

1. Put them in `corpus/<name>/`
2. Add an entry in `raglign/corpora.py` — path, file glob, one-line description
3. Author questions at `data/qa/<name>/longspan_v1.json`, or scaffold them with
   `$P scripts/author_qa.py --corpus <name> --count 30`
4. `$P scripts/check_qa.py --corpus <name>` — confirms every quote resolves
5. `$P scripts/run_grid.py --corpus <name>`

No other code changes.

### The dashboard

`streamlit run app.py` — five views: **Recommendation**, **Full grid**, **Failure
diagnosis**, **Validation study**, **Cross-corpus**. It reads saved results and never
evaluates anything, so interaction is instant and every number matches what the CLI wrote
to disk.

---

## Repository layout

```
raglign/          library — chunking, retrieval, alignment, metrics, diagnosis, generation
scripts/          CLI entry points (each takes --corpus)
ui/               dashboard — theme, components, charts, data prep
tests/            86 tests, no network required
corpus/           vendored documents, pinned to upstream commits
data/qa/          ground-truth question sets, one folder per corpus
runs/             generated results (gitignored — rebuildable)
```

**One invariant holds the whole thing up**, asserted on every chunk of every strategy:

```python
assert doc.text[chunk.start:chunk.end] == chunk.text
```

Offset drift doesn't crash anything — it silently produces plausible, wrong metrics. So it
is checked rather than trusted. It caught a real bug: the corpus was checked out with CRLF
line endings, which made every multi-line quote fail to resolve.

---

## Design decisions worth knowing

- **Retrieval metrics are computed here, not by an LLM judge.** Using an LLM to validate a
  methodology whose selling point is determinism would be circular. Hit@k, MRR and nDCG are
  arithmetic over span overlap and rerun bit-identically.
- **Overlap is measured as coverage, not IoU.** Measured, not assumed: heading chunking
  covers 11/11 evidence spans perfectly yet scores *worst* on IoU, purely because its
  chunks are larger. IoU ranks strategies by chunk size. Chunk bloat is a context-budget
  cost, so it's reported separately.
- **No vector database.** ~2,000 chunks × 384 dims is a 3 MB numpy matrix; brute force is
  exact and sub-millisecond. Approximate search would be a confound in a study about
  retrieval correctness.
- **The overlap threshold τ is swept, never fixed.** A single reported threshold invites
  the assumption it was tuned.
- **Explanations are templated over measured deltas, not LLM-written**, so they cannot
  invent a cause.

---

## Honest positioning

**Span-based ground truth is not a new technique.** Several 2025–26 studies use it: a
chunking-methods study defining relevance by overlap with the extractive answer span; the
"Chunk Twice, Embed Once" chemistry-RAG framework annotating start/end character indices;
MHTS mapping gold evidence chunks during dataset construction.

What none of them do is ship it as a **reusable evaluation layer inside a general-purpose
tool you can point at your own corpus** — they are one-off academic pipelines. That
packaging gap is what this project fills. Smaller than "novel method", and accurate.

**Limits, stated plainly:**

- **41 and 43 questions.** Bootstrap intervals confirm no top-3 difference on either corpus
  is statistically separable. The large effects hold — the ~0.6 ground-truth bias, the
  `fixed_300` collapse — but the fine-grained ranking needs roughly 150 questions.
- **A finding didn't survive growing the sample.** With 11–14 questions, chunking appeared
  to matter far more on structured documents (spread 0.223 vs 0.045). At 41–43 those became
  0.106 and 0.104 — the gap was noise. Recorded here rather than quietly dropped, because
  the effect that replicated and the one that evaporated were reported with equal confidence
  when the sample was small.
- **Ground truth is extractive by construction**, so questions with no single source span
  are out of scope.
- **One embedding model throughout.** The vector store is deliberately not a variable.

**Not included:** automatic embedding/LLM selection, prompt optimization, query-adaptive
routing, a production monitoring loop, or $/query cost modelling.
