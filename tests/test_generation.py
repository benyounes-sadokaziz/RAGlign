"""Tests for the generation-side metrics.

Every test here runs offline. The LLM is injected as a plain callable, so the
whole scoring path -- prompt assembly, refusal detection, deterministic metrics,
judge aggregation, correlation -- is exercised with no API key, no network, and
no cost. That is deliberate: a metric suite that can only be tested by spending
money does not get tested.
"""

from __future__ import annotations

import pytest

from raglign.generation import (
    GenerationScore,
    build_prompt,
    correlation,
    evaluate_generation,
    extractive_answerer,
    is_refusal,
    llm_answerer,
    judge_faithfulness,
    span_recall,
    token_f1,
)
from raglign.models import Chunk, QAItem, Span


def chunk(text: str, doc: str = "d", start: int = 0) -> Chunk:
    return Chunk(doc, start, start + len(text), text)


def item(qid: str, question: str, answer: str) -> QAItem:
    return QAItem(id=qid, question=question, answer=answer, spans=(Span("d", 0, 10),))


class TestDeterministicMetrics:
    def test_identical_text_is_f1_one(self):
        assert token_f1("the cat sat", "the cat sat") == pytest.approx(1.0)

    def test_disjoint_text_is_f1_zero(self):
        assert token_f1("alpha beta", "gamma delta") == 0.0

    def test_f1_is_case_and_punctuation_insensitive(self):
        assert token_f1("The Cat, sat!", "the cat sat") == pytest.approx(1.0)

    def test_empty_prediction_scores_zero_not_crash(self):
        assert token_f1("", "something") == 0.0

    def test_partial_overlap_is_between(self):
        score = token_f1("the cat sat on the mat", "the cat")
        assert 0.0 < score < 1.0

    def test_span_recall_ignores_extra_answer_words(self):
        """Recall, not F1: a concise answer must not be punished for brevity."""
        assert span_recall("alpha beta plus lots of other words", "alpha beta") == pytest.approx(1.0)

    def test_span_recall_counts_missing_evidence(self):
        assert span_recall("alpha", "alpha beta gamma delta") == pytest.approx(0.25)

    def test_empty_evidence_is_zero(self):
        assert span_recall("anything", "") == 0.0


class TestRefusalDetection:
    @pytest.mark.parametrize(
        "text",
        ["I don't know.", "I do not know", "The context does not contain that",
         "Unable to answer from the provided context", "", "   "],
    )
    def test_refusals_detected(self, text):
        assert is_refusal(text)

    @pytest.mark.parametrize(
        "text",
        ["Use WSGIMiddleware to wrap the app.", "The answer is 0.95.1.",
         "Set Cache-Control: no-cache."],
    )
    def test_real_answers_not_flagged(self, text):
        assert not is_refusal(text)


class TestPromptAssembly:
    def test_chunks_appear_in_rank_order(self):
        prompt = build_prompt("q?", [chunk("first"), chunk("second")])
        assert prompt.index("first") < prompt.index("second")

    def test_context_is_truncated_at_the_budget(self):
        """A config that buries evidence at rank 9 must be able to lose here.

        If everything retrieved were always included, retrieval *ranking* could
        not affect answer quality and the correlation would be meaningless.
        """
        chunks = [chunk("x" * 100) for _ in range(50)]
        prompt = build_prompt("q?", chunks, max_chars=500)
        assert prompt.count("x" * 100) < 50

    def test_no_chunks_still_produces_a_valid_prompt(self):
        prompt = build_prompt("q?", [])
        assert "no context retrieved" in prompt
        assert "q?" in prompt


class TestJudging:
    def test_scores_are_clamped_to_unit_range(self):
        out_of_range = iter([{"score": 5}, {"score": -3}])
        score, _, failures = judge_faithfulness(
            lambda *a, **k: next(out_of_range), "a", "c", repeats=2
        )
        assert score == pytest.approx(0.5)  # clamped to 1.0 and 0.0
        assert failures == 0

    def test_booleans_are_accepted(self):
        score, _, _ = judge_faithfulness(lambda *a, **k: {"score": True}, "a", "c")
        assert score == 1.0

    def test_unparseable_judgements_are_counted_not_scored(self):
        """A broken judge response must degrade one score, never become a zero."""
        score, spread, failures = judge_faithfulness(lambda *a, **k: {}, "a", "c", repeats=3)
        assert score is None and spread is None
        assert failures == 3

    def test_spread_is_reported_across_repeats(self):
        vals = iter([{"score": 0.0}, {"score": 1.0}])
        score, spread, _ = judge_faithfulness(
            lambda *a, **k: next(vals), "a", "c", repeats=2
        )
        assert score == pytest.approx(0.5)
        assert spread is not None and spread > 0


class TestEvaluateGeneration:
    def _items(self):
        return [
            item("q1", "What wraps a WSGI app?", "WSGIMiddleware wraps it."),
            item("q2", "What is unknown?", "Nothing."),
        ]

    def test_deterministic_tier_runs_without_a_judge(self):
        s = evaluate_generation(
            "cfg",
            self._items(),
            {"q1": [chunk("WSGIMiddleware wraps it.")], "q2": [chunk("irrelevant")]},
            {"q1": "WSGIMiddleware wraps it.", "q2": "Nothing."},
            answerer=llm_answerer(lambda p, system=None: "WSGIMiddleware wraps it."),
        )
        assert s.n == 2
        assert s.gold_f1 > 0
        # No judge supplied -> None, never 0.0, so a missing judgement cannot be
        # read as a bad score.
        assert s.faithfulness is None and s.relevance is None

    def test_refusals_are_counted_and_not_judged(self):
        judged = []

        def chat_json(prompt, system=None, **kw):
            judged.append(prompt)
            return {"score": 1.0}

        s = evaluate_generation(
            "cfg",
            self._items(),
            {"q1": [chunk("ctx")], "q2": [chunk("ctx")]},
            {"q1": "a", "q2": "b"},
            answerer=llm_answerer(lambda p, system=None: "I don't know."),
            chat_json=chat_json,
        )
        assert s.refusal_rate == 1.0
        assert judged == []  # nothing judged, because nothing was answered

    def test_judged_tier_populates_scores(self):
        s = evaluate_generation(
            "cfg",
            self._items(),
            {"q1": [chunk("ctx")], "q2": [chunk("ctx")]},
            {"q1": "a", "q2": "b"},
            answerer=llm_answerer(lambda p, system=None: "A real answer."),
            chat_json=lambda p, system=None, **kw: {"score": 0.75},
            repeats=2,
        )
        assert s.faithfulness == pytest.approx(0.75)
        assert s.relevance == pytest.approx(0.75)
        assert s.judge_failures == 0

    def test_missing_retrieval_does_not_crash(self):
        s = evaluate_generation(
            "cfg",
            self._items(),
            {},  # nothing retrieved for either question
            {},
            answerer=llm_answerer(lambda p, system=None: "I don't know."),
        )
        assert s.n == 2 and s.refusal_rate == 1.0


class TestCorrelation:
    def test_perfect_positive(self):
        assert correlation([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)

    def test_perfect_negative(self):
        assert correlation([1, 2, 3], [6, 4, 2]) == pytest.approx(-1.0)

    def test_constant_series_is_undefined_not_zero(self):
        """'All configs scored the same' is not 'no relationship'."""
        assert correlation([1, 1, 1], [1, 2, 3]) is None

    def test_too_few_points_is_undefined(self):
        assert correlation([1, 2], [1, 2]) is None

    def test_mismatched_lengths_is_undefined(self):
        assert correlation([1, 2, 3], [1, 2]) is None


class TestExtractiveBaseline:
    """The offline baseline: the top chunk is the answer.

    Exists so the generation path is runnable and comparable with no API key,
    and so a generator that adds nothing over raw retrieval is visible as such.
    """

    def test_returns_the_top_ranked_chunk(self):
        answerer = extractive_answerer()
        assert answerer("q?", [chunk("first chunk"), chunk("second")]) == "first chunk"

    def test_empty_retrieval_reads_as_a_refusal(self):
        answerer = extractive_answerer()
        assert is_refusal(answerer("q?", []))

    def test_perfect_retrieval_scores_perfectly_on_grounding(self):
        gold = "WSGIMiddleware wraps it."
        s = evaluate_generation(
            "cfg",
            [item("q1", "What wraps a WSGI app?", gold)],
            {"q1": [chunk(gold)]},
            {"q1": gold},
            answerer=extractive_answerer(),
        )
        assert s.span_grounding == pytest.approx(1.0)
        assert s.refusal_rate == 0.0


class TestJudgeShapeTolerance:
    """Models disagree about JSON shape even when told the exact format.

    open-mistral-nemo wraps its answer as {"response": {"score": 0.9}} where
    mistral returns {"score": 0.9}. A top-level-only lookup turned every nemo
    judgement into a parse failure, quietly emptying the judged tier.
    """

    def test_top_level_score(self):
        score, _, fails = judge_faithfulness(lambda *a, **k: {"score": 0.4}, "a", "c")
        assert score == pytest.approx(0.4) and fails == 0

    def test_nested_score_is_found(self):
        score, _, fails = judge_faithfulness(
            lambda *a, **k: {"response": {"score": 0.4}}, "a", "c"
        )
        assert score == pytest.approx(0.4) and fails == 0

    def test_deeply_nested_score_is_found(self):
        score, _, _ = judge_faithfulness(
            lambda *a, **k: {"result": {"judgement": {"score": 1.0}}}, "a", "c"
        )
        assert score == pytest.approx(1.0)

    def test_percentage_scale_is_normalised(self):
        """A model asked for 0..1 sometimes answers 85."""
        score, _, _ = judge_faithfulness(lambda *a, **k: {"score": 85}, "a", "c")
        assert score == pytest.approx(0.85)

    def test_numeric_string_is_accepted(self):
        score, _, _ = judge_faithfulness(lambda *a, **k: {"score": "0.6"}, "a", "c")
        assert score == pytest.approx(0.6)

    def test_genuinely_missing_score_still_fails(self):
        score, _, fails = judge_faithfulness(lambda *a, **k: {"verdict": "good"}, "a", "c")
        assert score is None and fails == 1

    def test_small_overshoot_is_clamped_not_rescaled(self):
        """A judge answering 5 on a 0..1 scale meant high, not 5%."""
        score, _, _ = judge_faithfulness(lambda *a, **k: {"score": 5}, "a", "c")
        assert score == pytest.approx(1.0)
