"""Generation-side evaluation: does better retrieval actually produce better answers?

Kept rigorously separate from the retrieval metrics, for the reason in TC-3: the
retrieval numbers are deterministic arithmetic over span overlap and rerun
bit-identically, while anything here involves either a generator or an LLM judge
and therefore carries variance. Mixing them into one score would contaminate the
project's one genuinely precise measurement with its least precise one.

The metrics split into two tiers, and the split is the point:

DETERMINISTIC (no API, free, exact)
  answered        did the model produce a non-refusal at all
  gold_f1         token F1 between the generated answer and the authored answer
  span_grounding  token recall of the *ground-truth evidence span* in the answer

  These are computable because this project already has span-level ground truth.
  Most RAG evaluation cannot do this: with only a chunk ID there is no reference
  answer text and no evidence text to compare against.

JUDGED (needs the API, costs money, noisy)
  faithfulness    is every claim in the answer supported by the retrieved context
  relevance       does the answer actually address the question

  Reported with repeats and spread, never as a bare point estimate. A judge that
  is rerun and disagrees with itself is telling you something about the metric,
  and hiding that behind an average would be the same mistake as reporting a
  config ranking without confidence intervals (TC-20).

The headline question this module exists to answer is the correlation: across
configs, does higher retrieval MRR predict higher faithfulness? If it does, the
retrieval optimization this project performs is worth doing. If it does not,
that is a finding worth reporting rather than hiding -- and it would mean the
retrieval metrics, however precise, are optimizing the wrong thing.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from statistics import mean, pstdev
from typing import Callable, Protocol, Sequence

from .models import Chunk, QAItem

# A chat function: (prompt, system=..., temperature=...) -> text. Declared as a
# protocol rather than taking a MistralClient so tests can inject a deterministic
# stub and the module never needs the network to be exercised.
ChatFn = Callable[..., str]

# (question, ranked chunks) -> answer text. The seam that lets an LLM and an
# offline baseline be scored through an identical path.
AnswerFn = Callable[[str, Sequence[Chunk]], str]


class JsonChatFn(Protocol):
    def __call__(self, prompt: str, *, system: str | None = ..., **kw) -> dict: ...


_TOKEN = re.compile(r"[a-z0-9]+")

# Openers a model uses when it declines or cannot find the answer. Detected
# because a refusal is not a wrong answer -- it is the system correctly reporting
# that retrieval failed, and lumping the two together would credit a confidently
# wrong answer above an honest "not in the context".
_REFUSALS = (
    "i don't know",
    "i do not know",
    "not in the",
    "no information",
    "cannot answer",
    "can't answer",
    "not provided",
    "does not contain",
    "doesn't contain",
    "unable to answer",
    "insufficient",
)

ANSWER_SYSTEM = (
    "Answer the question using ONLY the provided context. "
    "If the context does not contain the answer, reply exactly: I don't know. "
    "Be concise: one or two sentences."
)

FAITHFULNESS_SYSTEM = (
    "You judge whether an answer is supported by a context passage. "
    "Respond only with JSON."
)

RELEVANCE_SYSTEM = (
    "You judge whether an answer addresses the question that was asked. "
    "Respond only with JSON."
)


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def token_f1(pred: str, gold: str) -> float:
    """Bag-of-tokens F1 between a generated and a reference answer.

    Deliberately crude. It cannot recognise a correct paraphrase that shares no
    vocabulary, so it is reported as a signal rather than a verdict -- but it is
    free, deterministic, and it moves in the right direction across configs,
    which is what a comparison needs. The judged metrics below cover semantics.
    """
    p, g = Counter(tokens(pred)), Counter(tokens(gold))
    if not p or not g:
        return 0.0
    overlap = sum((p & g).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(p.values())
    recall = overlap / sum(g.values())
    return 2 * precision * recall / (precision + recall)


def gold_recall(pred: str, gold: str) -> float:
    """Fraction of the reference answer's tokens present in the generated answer.

    Companion to token_f1, and necessary because F1 carries a precision term and
    therefore penalises length. That makes F1 invalid for the extractive baseline,
    whose "answer" is an entire chunk: measured that way, an 800-character chunker
    scores worse than a 300-character one purely for being longer, and retrieval
    MRR appeared to *anti*-correlate with answer quality (r = -0.73) on the first
    run. Recall is length-insensitive and asks the question actually intended --
    is the reference answer's content present?

    Read F1 for a real generator, which produces a short answer and for which
    precision is meaningful; read recall when answer length is an artefact of the
    method rather than a property of the answer.
    """
    p_, g = Counter(tokens(pred)), Counter(tokens(gold))
    if not g:
        return 0.0
    return sum((p_ & g).values()) / sum(g.values())


def span_recall(answer: str, evidence: str) -> float:
    """Fraction of the evidence span's content words that appear in the answer.

    Recall, not F1: the answer should be shorter than the evidence, so penalising
    it for omitting most of the passage would be measuring brevity rather than
    grounding.
    """
    a, e = Counter(tokens(answer)), Counter(tokens(evidence))
    if not e:
        return 0.0
    return sum((a & e).values()) / sum(e.values())


def is_refusal(answer: str) -> bool:
    low = answer.strip().lower()
    if not low:
        return True
    return any(r in low for r in _REFUSALS)


def extractive_answer(chunks: Sequence[Chunk], max_chars: int = 900) -> str:
    """The no-LLM baseline: hand back the top-ranked chunk as the answer.

    Not a serious answering strategy, but a serious *baseline*, and it needs no
    API key. It measures how much of the answer the retrieval already put in
    first place, which sets the floor any generator has to beat: a reader model
    cannot state what was never retrieved.

    Its real use is diagnostic. If a generator scores no better than this, the
    generator is not adding anything and the pipeline's problem is upstream.
    """
    if not chunks:
        return ""
    return chunks[0].text.strip()[:max_chars]


def build_prompt(question: str, chunks: Sequence[Chunk], max_chars: int = 6000) -> str:
    """Assemble the context block exactly as a real pipeline would.

    Chunks are included in retrieval rank order and truncated at a character
    budget, because that is what a production system does with a finite context
    window -- and it means a config that wins on hit@10 but buries the evidence
    at rank 9 can still lose here. That is the retrieval-quality signal reaching
    the generator, which is the whole thing being measured.
    """
    parts, used = [], 0
    for i, c in enumerate(chunks, 1):
        block = f"[{i}] {c.text.strip()}"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    context = "\n\n".join(parts) if parts else "(no context retrieved)"
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"


@dataclass
class GenerationScore:
    question_id: str
    answer: str
    refused: bool
    gold_f1: float
    gold_recall: float
    span_grounding: float
    # None when no judge was run (offline / no API quota), so a missing judgement
    # is never silently reported as a zero score.
    faithfulness: float | None = None
    faithfulness_spread: float | None = None
    relevance: float | None = None
    relevance_spread: float | None = None
    judge_failures: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class GenerationSummary:
    config_id: str
    n: int
    refusal_rate: float
    gold_f1: float
    gold_recall: float
    span_grounding: float
    faithfulness: float | None
    relevance: float | None
    judge_failures: int
    per_question: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def _score_0_1(payload: dict, key: str = "score") -> float | None:
    raw = payload.get(key)
    if isinstance(raw, bool):
        return 1.0 if raw else 0.0
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, value))


def judge_faithfulness(chat_json: JsonChatFn, answer: str, context: str, repeats: int = 1) -> tuple[float | None, float | None, int]:
    """Is every claim in the answer supported by the context? Mean over repeats."""
    prompt = (
        "Context:\n" + context + "\n\nAnswer:\n" + answer + "\n\n"
        'Return JSON {"score": <0..1>, "unsupported": "<the first unsupported claim, or empty>"}. '
        "score 1 means every claim in the answer is supported by the context; "
        "0 means none of it is."
    )
    scores, failures = [], 0
    for seed in range(repeats):
        got = _score_0_1(chat_json(prompt, system=FAITHFULNESS_SYSTEM, seed=seed))
        if got is None:
            failures += 1
        else:
            scores.append(got)
    if not scores:
        return None, None, failures
    return mean(scores), (pstdev(scores) if len(scores) > 1 else 0.0), failures


def judge_relevance(chat_json: JsonChatFn, answer: str, question: str, repeats: int = 1) -> tuple[float | None, float | None, int]:
    prompt = (
        "Question:\n" + question + "\n\nAnswer:\n" + answer + "\n\n"
        'Return JSON {"score": <0..1>}. score 1 means the answer directly and '
        "completely addresses the question; 0 means it does not address it at all."
    )
    scores, failures = [], 0
    for seed in range(repeats):
        got = _score_0_1(chat_json(prompt, system=RELEVANCE_SYSTEM, seed=seed))
        if got is None:
            failures += 1
        else:
            scores.append(got)
    if not scores:
        return None, None, failures
    return mean(scores), (pstdev(scores) if len(scores) > 1 else 0.0), failures


def llm_answerer(chat: ChatFn, *, max_context_chars: int = 6000) -> AnswerFn:
    """Adapt a chat function into an answerer that sees ranked context."""

    def answer(question: str, chunks: Sequence[Chunk]) -> str:
        return chat(build_prompt(question, chunks, max_context_chars), system=ANSWER_SYSTEM)

    return answer


def extractive_answerer(max_chars: int = 900) -> AnswerFn:
    """The offline baseline as an answerer."""

    def answer(question: str, chunks: Sequence[Chunk]) -> str:
        return extractive_answer(chunks, max_chars)

    return answer


def evaluate_generation(
    config_id: str,
    questions: Sequence[QAItem],
    retrievals: dict[str, list[Chunk]],
    evidence: dict[str, str],
    *,
    answerer: AnswerFn,
    chat_json: JsonChatFn | None = None,
    repeats: int = 1,
    max_context_chars: int = 6000,
) -> GenerationSummary:
    """Generate an answer per question from that config's retrieval, then score it.

    Args:
        retrievals: question id -> the chunks that config retrieved, in rank order.
        evidence: question id -> the ground-truth span text, for span_grounding.
        answerer: (question, ranked chunks) -> answer text. Taking the question
            and chunks rather than a finished prompt is what lets an offline
            baseline and an LLM be scored through exactly the same path -- an
            earlier version passed only the assembled prompt, which forced the
            baseline to parse the question back out of it.
        chat_json: omit to skip the judged tier. The deterministic metrics still
            produce a complete comparison.
    """
    scores: list[GenerationScore] = []
    for q in questions:
        chunks = retrievals.get(q.id, [])
        answer = (answerer(q.question, chunks) or "").strip()
        refused = is_refusal(answer)

        s = GenerationScore(
            question_id=q.id,
            answer=answer,
            refused=refused,
            gold_f1=round(token_f1(answer, q.answer), 4),
            gold_recall=round(gold_recall(answer, q.answer), 4),
            span_grounding=round(span_recall(answer, evidence.get(q.id, "")), 4),
        )

        # A refusal is not judged: asking "is this supported by the context"
        # about the text "I don't know" measures nothing, and scoring it would
        # drag faithfulness toward whatever the judge happens to say about a
        # non-answer.
        if chat_json is not None and not refused:
            context = build_prompt(q.question, chunks, max_context_chars).split("Question:")[0]
            f, fs, ff = judge_faithfulness(chat_json, answer, context, repeats)
            r, rs, rf = judge_relevance(chat_json, answer, q.question, repeats)
            s.faithfulness, s.faithfulness_spread = f, fs
            s.relevance, s.relevance_spread = r, rs
            s.judge_failures = ff + rf
        scores.append(s)

    judged_f = [s.faithfulness for s in scores if s.faithfulness is not None]
    judged_r = [s.relevance for s in scores if s.relevance is not None]
    return GenerationSummary(
        config_id=config_id,
        n=len(scores),
        refusal_rate=round(mean(1.0 if s.refused else 0.0 for s in scores), 4) if scores else 0.0,
        gold_f1=round(mean(s.gold_f1 for s in scores), 4) if scores else 0.0,
        gold_recall=round(mean(s.gold_recall for s in scores), 4) if scores else 0.0,
        span_grounding=round(mean(s.span_grounding for s in scores), 4) if scores else 0.0,
        faithfulness=round(mean(judged_f), 4) if judged_f else None,
        relevance=round(mean(judged_r), 4) if judged_r else None,
        judge_failures=sum(s.judge_failures for s in scores),
        per_question=[s.as_dict() for s in scores],
    )


def correlation(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Pearson r, or None when either series is constant.

    Used for the headline question: does retrieval MRR predict faithfulness
    across configs? Returning None rather than 0.0 on a constant series matters,
    because "every config scored the same" and "no relationship" are different
    conclusions and only one of them is about the retrieval.
    """
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    mx, my = mean(xs), mean(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    den = (sum(a * a for a in dx) ** 0.5) * (sum(b * b for b in dy) ** 0.5)
    if den == 0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / den
