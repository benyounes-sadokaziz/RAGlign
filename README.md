<div align="center">

# RAGlign

### Point it at your documents. It tells you which RAG configuration to use — and proves it.

[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-86%20passing-22c55e)](tests/)
[![Runs offline](https://img.shields.io/badge/runs-100%25%20local%20CPU-8b5cf6)]()
[![No vector DB](https://img.shields.io/badge/vector%20DB-not%20required-64748b)]()
[![Corpora](https://img.shields.io/badge/validated%20on-2%20corpora%20%C2%B7%2084%20questions-2f6df6)]()

**Standard RAG evaluation has a bias larger than the differences it's trying to measure.
This project measures that bias, removes it, and then uses the corrected metrics to pick
your pipeline.**

</div>

---

## The 30-second version

You're building RAG. You must choose a chunking strategy, a retriever, and whether to
rerank. Those choices interact, the right answer depends on your documents, and most teams
pick by intuition.

RAGlign measures instead — **30 configurations, scored on evidence, ranked by the
priorities you set.**

```
your documents          ┌─────────────────────────────┐
      +          ──────►│  5 chunkers                 │──────►  ranked configurations
 ~40 questions          │  × 3 retrievers             │         + why each one placed
                        │  × reranker on / off        │         + what's limiting you
                        └─────────────────────────────┘
```

> [!IMPORTANT]
> **The measurement is the hard part, not the search.** Ground truth in RAG evaluation is
> normally a chunk ID — *"the answer is in chunk 57."* But chunk 57 only exists inside one
> chunking strategy. Score a different strategy against it and you are measuring **how
> similar its cut points are**, not how well it retrieves.
>
> Measured here: that bias is worth **~0.6 hit@5**, several times the real quality spread
> between the strategies being compared. It flips the recommended configuration in **3 of 4**
> tests.

---

## What you get

| | |
|---|---|
| **Ranked configurations** | 30 pipelines scored on your corpus, re-ranked live as you change what you value |
| **A reason, not just a score** | *"heading chunking beats fixed-size by 0.043 MRR holding retrieval fixed; reranking was rejected — it would add 2.8 s/query for +0.096"* |
| **Root-cause diagnosis** | per question, the stage that lost it: chunker, retriever, or reranker — each needs a different fix |
| **Confidence intervals** | on every metric, so you know which differences are real and which are noise |
| **A dashboard** | five views, instant, reading saved results — never recomputes |
| **Proof it matters** | retrieval quality predicts real answer quality at **r = +0.95** |

---

## How it works

**1 · Give it a corpus.** Drop documents in `corpus/<name>/`. Markdown, plain text —
it reads bytes and never reformats them.

**2 · Give it questions.** Around 40. Each carries a short quote copied from your own
documents — the passage that answers it. You never record character positions; the tool
locates the quote and derives them.

> [!TIP]
> A quote that doesn't appear **verbatim is rejected**, so ground truth cannot silently rot
> when documents change. `scripts/author_qa.py` slices candidate passages straight out of
> your corpus — the quotes are correct by construction and you only write the questions.

**3 · Run the grid.** All 30 pipelines built, every question run through every one.

**4 · Say what you care about.** Move the Quality / Latency / Context weights. The ranking
updates instantly — the *measurements* never change, only how they're weighted.

```
config                        MRR   hit@5   ms/query   context
heading1200/dense+rerank    0.683   0.805       2844      3624   ← best quality
heading1200/dense           0.589   0.756        0.4      3428   ← 7000× faster, −0.09 MRR
fixed300/bm25               0.222   0.439        4.9      1486
```

**5 · Find out what's actually limiting you.** For every missed question, the tool names
the stage that lost it.

---

## Why the numbers can be trusted

RAGlign stores ground truth as a **character span in the original document**, never as a
chunk ID:

```
ground truth   tutorial/body.md [4210:4580]     ← a property of your documents
chunk from A   tutorial/body.md [3800:4600]     ← contains it     → hit
chunk from B   tutorial/body.md [4100:4400]     ← contains 51%    → depends on threshold
```

Every strategy is then asked the same question — *did you retrieve the text that answers
this?* — and none of them wrote the question.

### Proving the bias is real

`scripts/validation_study.py` scores every strategy **twice** against the same questions
and the **same retrievals** — once by chunk-ID matching, once by span alignment. Any
difference comes from the ground truth alone.

The ground truth is then re-authored against each strategy in turn. If the effect were an
artifact of a convenient choice, rotating the author would dissolve it.

**What merely authoring the ground truth is worth:**

<div align="center">

| corpus | questions | chunk-ID ground truth | span alignment |
|---|:---:|:---:|:---:|
| **FastAPI docs** | 41 | 🔴 **+0.569 … +0.675** | 🟢 +0.016 … +0.049 |
| **Public-domain prose** | 43 | 🔴 **+0.589 … +0.744** | 🟢 −0.116 … +0.101 |

</div>

> [!WARNING]
> **Whoever writes the ground truth wins.** By ~0.6 hit@5 — several times the genuine
> quality spread. The recommended configuration flips in **3 of 4** rotations. It replicates
> on a corpus with **no markup at all**, so it is not a property of one document style.

*"Isn't that just strict matching?"* Relax what counts as the gold chunk from IoU ≥ 0.9
down to 0.3, and the bias decays smoothly to zero — chunk-ID matching becomes unbiased only
once it's loosened into being span overlap. Which is the argument, not a counterexample.

---

## What span-level ground truth unlocks

### 🔧 Knowing which component to fix

A score says a question failed. It cannot say **which stage** lost it — and the stages need
opposite fixes.

| cause | what to actually do |
|---|---|
| 🔴 `chunker_destroyed` | no chunk covers the evidence → change chunking. **No embedding model can help.** |
| 🟠 `chunker_degraded` | evidence survived only partially → increase chunk size before touching retrieval |
| 🔵 `retriever_missed` | indexed and intact but never retrieved → try hybrid, or a reranker |
| 🟣 `reranker_demoted` | first stage had it, the reranker pushed it out → drop or retune reranking |
| ⚪ `ranked_low` | found, but not first → a reranker is the targeted fix |

On a configuration using 300-character chunks:

```
fixed300/bm25     chunker 8    retriever 0    ranked low 1    ok 2
```

> [!NOTE]
> **8 of 11 failures are the chunker. Zero are the retriever.** Standard evaluation reports
> this as "low hit@5" and sends you shopping for embedding models. The chunks are smaller
> than the answers — no retriever can fix that.
>
> Asking *"could any chunk in this index have answered this?"* is only possible when ground
> truth is a position. A chunk ID cannot express it.

### 📉 Seeing the ceiling before retrieval even runs

**Oracle reachability** — the share of evidence some chunk could satisfy, ignoring the
retriever entirely:

| chunker | τ=0.5 | τ=0.7 |
|---|:---:|:---:|
| `fixed_800` | 1.000 | 0.927 |
| `heading_1200` | 1.000 | 0.829 |
| `fixed_300` | 0.976 | 🔴 **0.463** |

`fixed_300` destroys over half its evidence **before retrieval happens**.

---

## 🔀 The answer depends on your corpus

Same 30 configurations, two structurally opposite corpora:

| corpus | structure | winner |
|---|---|---|
| FastAPI docs | headings, code blocks, lists | `heading1200/dense` |
| Public-domain prose | flowing paragraphs, zero markup | `recursive800/dense` |

> [!IMPORTANT]
> **Configuration choice does not transfer across document structure.** That's precisely the
> condition that makes a per-corpus optimizer worth running instead of a blog post naming a
> default.

---

## ✅ Does better retrieval actually produce better answers?

Everything above measures retrieval — which only matters if it predicts what reaches the
user. `scripts/run_generation.py` closes the loop: a real LLM answers from each
configuration's retrieved context, and a **different model judges** the answers, because
LLM judges score their own output more favourably.

<div align="center">

| correlation with retrieval MRR | r |
|---|:---:|
| gold recall | **+0.984** |
| span grounding | **+0.968** |
| faithful coverage | **+0.951** |

</div>

> [!TIP]
> **Retrieval quality predicts answer quality at r ≈ +0.95** — the evidence that optimizing
> retrieval is worth doing at all. Runs without an API key via `--generator extractive`,
> which makes the top-ranked chunk the answer: a genuine floor, since a reader cannot state
> what was never retrieved.

---

## Quick start

> [!NOTE]
> **Requirements:** Python 3.11+, ~2 GB disk, **no GPU, no API key, no vector database.**
> First run downloads a ~130 MB embedding model; everything after that is offline.

```bash
git clone https://github.com/benyounes-sadokaziz/RAGlign && cd RAGlign
python -m venv .venv

# Windows
.venv\Scripts\python.exe -m pip install -r requirements.txt && set P=.venv\Scripts\python.exe

# macOS / Linux
.venv/bin/python -m pip install -r requirements.txt && export P=.venv/bin/python
```

### See the headline result in ~2 minutes

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
cached, so re-runs are far quicker. Add `--corpus prose` to any command.

### Using your own documents

```bash
# 1. put your files in corpus/<name>/
# 2. add an entry in raglign/corpora.py  (path, glob, one-line description)
# 3. scaffold questions from your own text:
$P scripts/author_qa.py --corpus <name> --count 30
# 4. confirm every quote resolves:
$P scripts/check_qa.py --corpus <name>
# 5. evaluate:
$P scripts/run_grid.py --corpus <name>
```

No other code changes.

---

## Under the hood

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

> [!CAUTION]
> Offset drift doesn't crash anything — it silently produces **plausible, wrong metrics**.
> So it is checked rather than trusted. It caught a real bug: the corpus was checked out
> with CRLF line endings, which made every multi-line quote fail to resolve.

<details>
<summary><b>Design decisions worth knowing</b> (click to expand)</summary>

<br>

- **Retrieval metrics are computed here, not by an LLM judge.** Using an LLM to validate a
  methodology whose selling point is determinism would be circular. Hit@k, MRR and nDCG are
  arithmetic over span overlap and rerun bit-identically.

- **Overlap is coverage, not IoU** — measured, not assumed. Heading chunking covers 11/11
  evidence spans perfectly yet scores *worst* on IoU, purely because its chunks are larger.
  IoU would rank strategies by chunk size. Chunk bloat is a context-budget cost, reported
  separately.

- **No vector database.** ~2,000 chunks × 384 dims is a 3 MB numpy matrix; brute force is
  exact and sub-millisecond. Approximate search would be a confound in a study about
  retrieval correctness.

- **The overlap threshold τ is swept, never fixed.** A single reported threshold invites the
  assumption it was tuned.

- **Explanations are templated over measured deltas, not LLM-written**, so they cannot
  invent a cause.

- **The judge model differs from the generator model**, because LLM judges show
  self-preference toward their own output.

</details>

---

## Honest positioning

> [!NOTE]
> **Span-based ground truth is not a new technique.** Several 2025–26 studies use it: a
> chunking-methods study defining relevance by overlap with the extractive answer span; the
> *Chunk Twice, Embed Once* chemistry-RAG framework annotating start/end character indices;
> MHTS mapping gold evidence chunks during dataset construction.
>
> What none of them do is ship it as a **reusable evaluation layer inside a general-purpose
> tool you can point at your own corpus** — they are one-off academic pipelines. That
> packaging gap is what this fills. A smaller claim than "novel method", and the accurate one.

**Limits, stated plainly:**

- **41 and 43 questions.** Bootstrap intervals confirm no top-3 difference on either corpus
  is statistically separable. The large effects hold — the ~0.6 ground-truth bias, the
  `fixed_300` collapse — but fine-grained ranking needs roughly 150 questions.

- **A finding didn't survive growing the sample.** At 11–14 questions, chunking appeared to
  matter far more on structured documents (spread 0.223 vs 0.045). At 41–43 those became
  0.106 and 0.104 — the gap was noise. Recorded here rather than quietly dropped: the effect
  that replicated and the one that evaporated were reported with *equal confidence* when the
  sample was small. That is exactly what the confidence intervals exist to prevent.

- **Ground truth is extractive by construction**, so questions with no single source span
  are out of scope.

- **One embedding model throughout.** The vector store is deliberately not a variable.

**Not included:** automatic embedding/LLM selection, prompt optimization, query-adaptive
routing, production monitoring, or $/query cost modelling.
