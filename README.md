# RAGlign

**Evaluation-driven RAG pipeline optimization with chunking-independent ground truth.**

RAGlign compares RAG configurations — chunking × retrieval × reranking — and recommends one.
The part worth your attention is *how it decides what "correct" means*.

---

## The problem

Retrieval metrics need to know which retrieved unit was right. The obvious encoding is
"chunk #57 holds the answer". That breaks the moment you compare chunking strategies,
because chunk #57 only exists inside one strategy's segmentation.

Score strategy B against ground truth authored on strategy A's boundaries and you are
measuring *how much B's cut points resemble A's* — not how well B retrieves.

**Measured on this corpus, that bias is worth ~0.6 hit@5 — roughly 7× the real quality
spread between the strategies being compared.**

## The fix

Ground truth is stored as a **character span in the source document**, never as a chunk ID.
A retrieved chunk is also a span. "Did we retrieve the evidence?" becomes a question about
interval overlap, which every strategy can be asked identically.

```
ground truth   tutorial/body.md [4210:4580]      ← a property of the corpus
chunk (A)      tutorial/body.md [3800:4600]      ← coverage 1.00  → hit
chunk (B)      tutorial/body.md [4100:4400]      ← coverage 0.51  → depends on τ
```

---

## The headline result

`scripts/validation_study.py` scores every strategy twice against the same questions and
**the same retrievals** — once by chunk-ID matching, once by span alignment. Any difference
is attributable to the ground truth alone.

The study rotates which strategy authored the ground truth, because if the effect were an
artefact of a convenient choice of author, rotating would dissolve it.

**Self-preference — the author's own hit@5 minus the mean of the others:**

| ground truth authored by | chunk-ID GT | span GT |
|---|---:|---:|
| `fixed_800_100` | **+0.606** | −0.030 |
| `recursive_800_100` | **+0.576** | −0.030 |
| `heading_1200_200` | **+0.697** | +0.091 |
| `semantic_90` | **+0.576** | −0.030 |

Whoever writes the ground truth wins. Under span alignment the spread collapses to the
genuine differences. **The recommended config flips in 3 of 4 rotations.**

### "Isn't that just strict matching?"

The fair objection: chunk-ID matching required IoU ≥ 0.9 against the gold chunk. Relax it:

| author | IoU≥0.9 | IoU≥0.7 | IoU≥0.5 | IoU≥0.3 |
|---|---:|---:|---:|---:|
| `fixed_800_100` | +0.606 | +0.333 | +0.182 | −0.030 |
| `recursive_800_100` | +0.576 | +0.364 | +0.091 | −0.030 |
| `heading_1200_200` | +0.697 | +0.576 | +0.333 | +0.061 |
| `semantic_90` | +0.576 | +0.394 | +0.242 | +0.061 |

The bias decays monotonically and converges **exactly** on the span-GT column. Chunk-ID
matching becomes unbiased only once it is loosened so far that it effectively *is* span
overlap — which is the argument, not a counterexample.

---

## A second thing the span layer buys: oracle reachability

The fraction of evidence spans that *any* chunk in the index could satisfy, ignoring the
retriever entirely — the ceiling imposed by segmentation alone.

| chunker | τ=0.1 | τ=0.3 | τ=0.5 | τ=0.7 |
|---|---:|---:|---:|---:|
| `fixed_800` | 1.000 | 1.000 | 1.000 | 0.927 |
| `recursive_800` | 1.000 | 1.000 | 1.000 | 0.902 |
| `heading_1200` | 1.000 | 1.000 | 1.000 | 0.829 |
| `semantic_1200` | 1.000 | 1.000 | 1.000 | 0.805 |
| `fixed_300` | 1.000 | 1.000 | 0.976 | **0.463** |

At the smaller question set every viable chunker read 1.000 across the board and τ did
nothing. With 41 questions the evidence spans vary more, and τ=0.7 now separates them —
the segmentation ceiling finally discriminates instead of saturating.

`fixed_300` destroys over half the evidence *before retrieval happens*. Hit@k alone reports
this as a retrieval failure and sends you tuning embeddings that cannot possibly help.
Separating the two failure modes requires spans; chunk IDs cannot express it.

---

## Results

30 configs = 5 chunker variants × 3 retrievers × reranker on/off. Long-span question set:

| config | hit@5 | MRR | nDCG | ms/query | context chars |
|---|---:|---:|---:|---:|---:|
| `heading1200/dense+rerank` | 0.805 | **0.683** | 0.746 | 2844 | 3624 |
| `heading1200/hybrid+rerank` | **0.829** | 0.676 | 0.746 | 3152 | 3532 |
| `heading1200/bm25+rerank` | 0.780 | 0.645 | 0.726 | 2909 | 3664 |
| `fixed800/bm25+rerank` | 0.805 | 0.631 | 0.692 | 1419 | 3904 |
| `heading1200/dense` | 0.756 | 0.589 | 0.639 | **0.4** | 3428 |
| `fixed300/bm25` | 0.439 | 0.222 | 0.483 | 4.9 | 1486 |

### What it tests

| strategy | cuts where | family |
|---|---|---|
| `fixed_800` / `fixed_300` | every N characters | position |
| `recursive_800` | paragraph → line → word breaks | separator |
| `heading_1200` | markdown headings, code-fence aware | document structure |
| `semantic_90` | where sentence embeddings diverge | meaning |

**Semantic chunking did not win here**, and the reason is specific and useful:
it **fragments enumerations**. Both of its failures were bullet lists cut
mid-list — each item genuinely *is* a topic shift (different rule, different
field), so the sentence-distance signal spikes between them, but the list as a
whole is the answer unit. One security checklist got split with only 0.578
coverage remaining.

On structured API docs, explicit structure beats inferred structure: the heading
chunker keeps that list intact because the list lives under one heading. It also
costs ~4× more to build, since it embeds every sentence in the corpus before it
can cut anything.

**Tested fix that didn't work.** The standard remedy is sentence *buffering* —
embed each position as a window of its neighbours so a run of bullets reads as
one topic. It made things monotonically worse (spans intact 9/11 → 8/11 → 4/11
at buffer ±0/±1/±2). The mechanism: chunk counts barely moved (685 → 685 → 674),
because a *percentile* threshold always cuts the most abrupt 10% of boundaries.
Smoothing the curve can't reduce the number of cuts — it only removes the signal
that decides where they land. Buffering and percentile thresholds are
antagonistic; buffering only pays off with an absolute threshold, which
reintroduces the non-transferability problem the percentile was chosen to avoid.
Kept as a documented, defaulted-off parameter (TC-17).

Reranking buys ~+0.11 to +0.23 MRR for ~+2.8 s/query. Whether that trade is worth taking is
the user's call, so the optimizer reports the **Pareto frontier** before applying any weights.

### Two regimes worth knowing about

Running the same grid against short-span questions (evidence ~123 chars, well inside a
~700-char chunk) gives **0.83–1.00 for nearly every config**. When evidence is much smaller
than a chunk, chunk boundaries barely matter, τ is inert, and reranking dominates.

The chunking-independence problem only bites when evidence is comparable to chunk size.
That boundary condition is a finding, not a caveat — it tells you when to bother.

---

## v2: does any of this hold on a different corpus?

Every v1 conclusion came from FastAPI markdown, and the heading-aware chunker won
there partly *because* that corpus is dense with headings. So v2 added a second corpus
that is structurally its opposite — and matched on everything else, so the comparison
isolates structure:

| | fastapi | prose |
|---|---|---|
| structure | headings, code fences, bullet lists | flowing paragraphs, **no markup at all** |
| size | 97 docs, 540 KB | 70 docs, 514 KB |
| evidence spans | 320–533 chars | 306–479 chars |
| source | FastAPI docs | public-domain physics, biology, economics, memoir |

### The core claim replicates

Self-preference under chunk-ID ground truth vs span alignment, rotating the author:

| corpus | n | chunk-ID GT | span GT |
|---|---:|---|---|
| fastapi | 41 | +0.569 … **+0.675** | +0.016 … +0.049 |
| prose | 43 | +0.589 … **+0.744** | −0.116 … +0.101 |

The ground-truth bias is just as large on a corpus with no structure at all, and the
match-strictness sweep decays to ~0 on both. **This is the result that matters most in
v2** — the methodology's central finding is not an artifact of one corpus.

### The winner *does* differ by corpus

| corpus | best config |
|---|---|
| fastapi (structured markdown) | `heading1200/dense` |
| prose (no markup) | `recursive800/dense` |

Config choice does not transfer across document structure — which is exactly the condition
that makes a per-corpus optimizer worth building rather than just publishing a default.

**A finding from the smaller question sets did not survive.** At n=11/14 the same config
won both corpora, and the spread between viable chunkers looked dramatically different
(0.223 on fastapi vs 0.045 on prose), which I reported as "chunking matters far more on
structured documents." At n=41/43 those spreads are **0.106 and 0.104** — essentially
equal. The apparent difference was small-sample noise, and the earlier conclusion is
retracted here rather than quietly dropped.

This is the clearest illustration of why the confidence intervals were added: the effect
that replicated (ground-truth bias, ~0.6) and the effect that evaporated (chunker
sensitivity by corpus, 0.18) were reported with equal confidence when n was small.

### The caveat the confidence intervals forced

v2 added bootstrap CIs, and they immediately disciplined the v1 claims:

```
fastapi  heading1200/dense          0.598  [0.478, 0.726]
         fixed800/hybrid            0.623  [0.499, 0.751]
         heading1200/hybrid+rerank  0.680  [0.558, 0.802]
prose    recursive800/dense         0.765  [0.657, 0.860]
```

**No top-3 difference on either corpus survives its own 95% interval.** Widening the
question sets from 11/14 to 41/43 halved the interval widths from ~0.50 to ~0.25 —
exactly the √n improvement statistics predicts — and best-vs-worst became separable where
it had not been. Neighbouring configs still are not. Only the large effects hold: the
`fixed300` collapse, and the ~0.6 ground-truth bias. This is why the results tables above
are framed as "which knobs matter" rather than "this exact config is best."

More questions — not more configs, not more chunkers — is what buys resolution.

---

## v2: automatic failure diagnosis

A metric says a question failed. It can't say *which stage lost it* — and the stages have
different fixes. `scripts/diagnose.py` classifies every question, per config:

| cause | fix |
|---|---|
| `chunker_destroyed` | no chunk covers the evidence → change chunking; no embedding model helps |
| `chunker_degraded` | evidence survived only partially → increase chunk size first |
| `retriever_missed` | indexed and intact, never retrieved → hybrid, or a reranker |
| `reranker_demoted` | first stage had it, reranker pushed it out → drop/retune reranking |
| `ranked_low` | found but not at rank 1 → a reranker is the targeted fix |

Across all 30 configs:

| | fastapi | prose |
|---|---:|---:|
| chunker_degraded | 17.0% | 12.9% |
| retriever_missed | 17.9% | 9.3% |
| ranked_low | 40.9% | 38.3% |
| ok | 23.3% | 39.5% |

The concrete payoff, on `fixed300`:

```
fixed300/dense    destroyed 0   degraded 8   unranked 0   low 1   ok 2
```

**8 of 11 failures are the chunker; zero are the retriever.** Standard evaluation reports
`fixed300` as "low hit@5" and sends you shopping for embedding models. This says the chunks
are smaller than the answers, which no retriever can fix.

This required a real correction during the build: the first version checked reachability at
τ=0.5, so a chunker whose best chunk covered 51% of every answer passed as blameless and the
retriever absorbed the blame. `fixed300` showed **zero** chunker failures while its oracle
reachability was visibly collapsing. Recording threshold-free `max_achievable` fixed it, and
the misattribution is now a regression test.

**This only works with span ground truth.** "Could any chunk have answered this?" is
unaskable when ground truth is a chunk ID from a segmentation that never produced that chunk.
Fair comparison was the advertised benefit; actionable diagnosis is the one that matters more
day to day.

---

## Does better retrieval actually produce better answers?

Everything above measures *retrieval*. That only matters if it predicts what reaches the
user, so `scripts/run_generation.py` closes the loop: a real LLM answers from each
configuration's retrieved context, and the answers are scored.

Five configurations spanning the MRR range, all 41 fastapi questions:

| config | retrieval MRR | refusal | faithful coverage |
|---|---:|---:|---:|
| `fixed800/dense+rerank` | 0.616 | 0.10 | **0.854** |
| `fixed800/bm25+rerank` | 0.631 | 0.07 | 0.830 |
| `fixed800/bm25` | 0.539 | 0.15 | 0.741 |
| `fixed300/hybrid` | 0.327 | 0.10 | 0.690 |
| `fixed300/bm25` | 0.222 | 0.20 | 0.644 |

```
retrieval MRR vs gold recall        r = +0.984
retrieval MRR vs span grounding     r = +0.968
retrieval MRR vs faithful coverage  r = +0.951
```

**Retrieval quality predicts answer quality at r ≈ +0.95.** That is the evidence that
optimizing retrieval is worth doing at all — and it would have been just as publishable
had it come out weak.

**Two tiers, deliberately separated.** The deterministic tier (refusal rate, token recall
against the authored answer, grounding against the evidence span) needs no API and reruns
identically. Only span-level ground truth makes it computable: a chunk ID gives you no
reference answer and no reference evidence text. The judged tier (faithfulness, relevance)
uses `ministral-8b` to answer and **`open-mistral-nemo` to judge** — a different model,
because LLM judges score their own output more favourably.

**A bias I had to correct mid-build.** Refusals are excluded from judging, since asking
whether "I don't know" is supported by a context measures nothing. But that makes the
judged sample self-selected: a config refusing 83% of questions gets judged only on its
easiest few, and scored *perfect* faithfulness. Correlation with retrieval MRR came out at
**−0.918** — apparently proving better retrieval produces less faithful answers. Reporting
`faithfulness × answer rate` instead fixes it (+0.951). The raw figure is still shown,
labelled conditional, with its sample size.

It also runs with **no API key** via `--generator extractive`, where the top-ranked chunk
becomes the answer — a genuine floor, since a reader cannot state what was never retrieved.

---

## The dashboard

`streamlit run app.py` — a read-only view over saved run manifests. It never evaluates
anything, so the controls are instant and every number on screen is byte-identical to what
the CLI wrote to disk.

Five views: **Recommendation** (winner, KPI tiles with intervals, generated explanation,
Pareto frontier, quality-vs-latency scatter), **Full grid** (every metric at every cutoff),
**Failure diagnosis** (per-question root cause), **Validation study**, **Cross-corpus**.

Corpus, question set, overlap threshold τ and the three priority weights are live controls;
the weights re-rank everything and change the ranking only, never the measurements.

---

## Honest positioning

**Span-based ground truth is not a new technique.** Several 2025–26 studies already use it:
a chunking-methods study defining relevance by overlap with the extractive answer span; the
"Chunk Twice, Embed Once" chemistry-RAG framework annotating start/end character indices;
MHTS mapping gold evidence chunks during dataset construction.

What none of them do is ship it as a **reusable evaluation layer inside a general-purpose
optimization tool you can point at your own corpus** — they are one-off academic pipelines.
That packaging gap is RAGlign's contribution. It is a smaller claim than "novel method", and
it is the accurate one.

**Limits, stated plainly:**
- n=41 (fastapi) and n=43 (prose) long-span questions. Bootstrap intervals confirm no top-3
  config difference is resolvable at this size. The validation-study effect (~0.6) is far
  outside that noise and replicates on both corpora; the grid's finer distinctions are not.
- Ground truth is extractive by construction (TC-2), so questions with no single source span
  are out of scope.
- One embedding model throughout; the vector store is not an axis (TC-4). Two corpora now,
  but both English and both fairly clean text.

---

## Running it

**Requirements:** Python 3.11+, ~2 GB disk, no GPU. The first run downloads a ~130 MB
embedding model; everything after that is offline.

```bash
git clone <repo> && cd raglign
python -m venv .venv

# Windows
.venv\Scripts\python.exe -m pip install -r requirements.txt
set P=.venv\Scripts\python.exe

# macOS / Linux
.venv/bin/python -m pip install -r requirements.txt
export P=.venv/bin/python
```

### Quickest path to a result (~2 minutes)

```bash
$P scripts/validation_study.py --corpus fastapi
```

Self-contained — it builds its own indexes and prints the headline bias table. Nothing
else has to have been run first.

### Full pipeline

Order matters: everything after `run_grid.py` reads the run manifests it writes.

```bash
$P scripts/check_chunkers.py      # ~1 min   offset invariant holds on every chunker?
$P scripts/check_qa.py            # instant  does every authored quote still resolve?
$P scripts/run_grid.py            # ~25 min  evaluate all 30 configs  ← writes runs/
$P scripts/recommend.py           # instant  ranked table + explanation
$P scripts/diagnose.py            # instant  which stage lost each question
$P scripts/compare_corpora.py     # instant  does the winner transfer between corpora?
```

Most of the 25 minutes is the 15 reranker configurations at ~1.5 s/query. Chunk
embeddings are cached in `.cache/`, so re-runs are far quicker. Add `--corpus prose` to
any of them; `fastapi` is the default.

### Optional

```bash
$P scripts/run_generation.py --generator extractive   # no API key needed
$P scripts/run_generation.py                          # needs MISTRAL_API_KEY in .env
$P scripts/author_qa.py --corpus fastapi --count 30   # scaffold new questions
$P -m pytest -q                                       # 86 tests, all offline
.venv/Scripts/streamlit.exe run app.py                # dashboard on :8501
```

### Adding your own corpus

1. Put the documents in `corpus/<name>/`
2. Add a registry entry in `raglign/corpora.py` (path, glob, one-line description)
3. Author a QA set at `data/qa/<name>/longspan_v1.json` — or scaffold one with
   `scripts/author_qa.py`, which slices candidate passages out of your corpus so the
   quotes are verbatim by construction and you only write the questions
4. `$P scripts/check_qa.py --corpus <name>` to confirm every quote resolves
5. `$P scripts/run_grid.py --corpus <name>`

No other code changes.

### Repository layout

```
raglign/          library — models, loader, chunking/, retrieval/, alignment,
                  metrics, diagnosis, generation, optimizer, corpora
scripts/          CLI entry points (every one takes --corpus)
ui/               dashboard: theme, components, charts, data prep
tests/            86 tests, no network required
corpus/           vendored documents, pinned to upstream commits
data/qa/<corpus>/ ground-truth question sets
runs/<corpus>/    generated results (gitignored — rebuildable)
```

Everything runs locally on CPU. No API keys, no vector database, no network at eval time —
the generation layer is the single opt-in exception.

## How it fits together

```
corpus/*  ──► Loader ──► Chunkers (5) ──► Embedder ──► Retrievers (3) ──► [Reranker]
                   │                                                             │
                   └──► QA set: question + verbatim quote                         │
                             │                                                    │
                             ▼  quote located by exact search                     │
                        SPAN GROUND TRUTH ──────► ALIGNMENT LAYER ◄───────────────┘
                                                        │
                                                        ▼
                                     Hit@k · Recall@k · MRR · nDCG · oracle reachability
                                                        │
                                                        ▼
                                          Pareto frontier ──► weighted rank ──► explanation
```

Key invariant, enforced on every chunk of every strategy:

```python
assert doc.text[chunk.start:chunk.end] == chunk.text
```

Offset desync doesn't crash anything — it silently produces plausible, wrong metrics. So it
is asserted rather than trusted. (This caught a real bug: the corpus was checked out with
CRLF line endings, which made every multi-line quote fail to resolve.)

## Design decisions

Recorded as decision records — decision, alternatives rejected, and what would reverse it.
Highlights:

- **Ragas is scoped out of retrieval metrics.** Its `context_precision`/`context_recall` are
  LLM-judged; using an LLM judge to validate a methodology whose selling point is determinism
  is circular. Span alignment gives the same quantities deterministically and free.
- **`coverage` overlap, not IoU.** Measured: `heading` covers 11/11 evidence spans perfectly
  but scores *worst* on IoU (0.2–0.4) purely because its chunks are larger. IoU ranks
  strategies by chunk size. Chunk bloat is a context-budget cost, so it's reported as
  `chars_retrieved` rather than smuggled into the hit decision.
- **No vector database.** ~2k chunks × 384 dims is a 3 MB numpy matrix; brute force is exact
  and sub-millisecond. ANN would add approximate recall — a confound in a study about
  retrieval correctness.
- **τ is swept, never fixed.** A single reported threshold invites the assumption it was tuned.
- **Explanations are templated over measured deltas, not LLM-written**, so they cannot
  hallucinate a cause.

## Not in v1

Automatic embedding/LLM selection · prompt optimization · query-adaptive routing ·
continuous production loop · automatic failure root-cause diagnosis · $/query cost modeling.

Generation-side metrics now exist (see above), kept in a tier clearly separated from the
deterministic retrieval metrics. The remaining gaps are a larger question set — ~150 would
be needed to resolve neighbouring configs — and an embedding-model axis, which is only
worth opening once differences are resolvable.

---

Corpus: 97 FastAPI documentation files, vendored with upstream commit SHA — see
`corpus/PROVENANCE.md`. Character offsets are only meaningful against byte-identical text.
