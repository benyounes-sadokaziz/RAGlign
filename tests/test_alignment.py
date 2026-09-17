"""Tests for the alignment layer and the offset invariant.

These cover the arithmetic the project's central claim rests on. If overlap or
the metric reductions are wrong, every number in the README is wrong, and no
amount of correct pipeline code would reveal it -- the failure mode is
plausible-looking output, not a crash.
"""

from __future__ import annotations

import pytest

from raglign.alignment import OverlapMode, align, overlap_score
from raglign.chunking import FixedSizeChunker, HeadingChunker, RecursiveChunker
from raglign.metrics import compute_metrics
from raglign.models import Chunk, Document, OffsetError, Span


def span(start: int, end: int, doc: str = "d") -> Span:
    return Span(doc, start, end)


def chunk(start: int, end: int, doc: str = "d") -> Chunk:
    return Chunk(doc, start, end, "x" * (end - start))


class TestOverlap:
    def test_full_containment_is_full_coverage(self):
        assert overlap_score(span(0, 1000), span(400, 500)) == 1.0

    def test_disjoint_is_zero(self):
        assert overlap_score(span(0, 100), span(200, 300)) == 0.0

    def test_touching_ranges_do_not_overlap(self):
        # Spans are half-open: [0,100) and [100,200) share no character.
        assert overlap_score(span(0, 100), span(100, 200)) == 0.0

    def test_partial_coverage_is_the_covered_fraction(self):
        # truth [400,500); chunk covers [450,500) -> half the evidence
        assert overlap_score(span(450, 600), span(400, 500)) == pytest.approx(0.5)

    def test_different_documents_never_overlap(self):
        assert overlap_score(span(0, 100, "a"), span(0, 100, "b")) == 0.0

    def test_coverage_ignores_chunk_size_but_iou_does_not(self):
        """The measured reason coverage is the default (README / TC-8).

        Two chunks both fully contain the evidence and are therefore equally
        useful to a generator, but IoU rates the larger one far worse -- so IoU
        would rank chunking strategies by chunk size.
        """
        truth = span(500, 620)
        small, large = span(450, 700), span(0, 1400)
        assert overlap_score(small, truth) == overlap_score(large, truth) == 1.0
        assert overlap_score(small, truth, OverlapMode.IOU) > 4 * overlap_score(
            large, truth, OverlapMode.IOU
        )


class TestAlignedResult:
    def test_first_hit_rank_is_one_based(self):
        r = align("q", [chunk(0, 100), chunk(400, 500)], [span(400, 500)])
        assert r.first_hit_rank(0.5) == 2

    def test_no_hit_returns_none(self):
        r = align("q", [chunk(0, 100)], [span(400, 500)])
        assert r.first_hit_rank(0.5) is None

    def test_k_cutoff_excludes_later_hits(self):
        """Regression: metrics once reduced over the full list, ignoring k."""
        r = align("q", [chunk(0, 10), chunk(400, 500)], [span(400, 500)])
        assert r.first_hit_rank(0.5, k=1) is None
        assert r.first_hit_rank(0.5, k=2) == 2

    def test_multi_span_recall_counts_distinct_evidence(self):
        """Retrieving the same evidence twice is not finding both pieces."""
        truths = [span(0, 100), span(900, 1000)]
        twice = align("q", [chunk(0, 100), chunk(0, 100)], truths)
        both = align("q", [chunk(0, 100), chunk(900, 1000)], truths)
        assert twice.span_recall(0.5) == 0.5
        assert both.span_recall(0.5) == 1.0

    def test_span_recall_respects_k(self):
        truths = [span(0, 100), span(900, 1000)]
        r = align("q", [chunk(0, 100), chunk(900, 1000)], truths)
        assert r.span_recall(0.5, k=1) == 0.5
        assert r.span_recall(0.5, k=2) == 1.0

    def test_soft_score_is_threshold_free(self):
        r = align("q", [chunk(450, 600)], [span(400, 500)])
        assert r.soft_score() == pytest.approx(0.5)


class TestMetrics:
    def test_mrr_rewards_earlier_hits(self):
        early = [align("q", [chunk(400, 500), chunk(0, 10)], [span(400, 500)])]
        late = [align("q", [chunk(0, 10), chunk(400, 500)], [span(400, 500)])]
        assert compute_metrics(early, k=5, threshold=0.5).mrr == 1.0
        assert compute_metrics(late, k=5, threshold=0.5).mrr == pytest.approx(0.5)

    def test_unanswered_counts_questions_with_no_hit(self):
        results = [
            align("a", [chunk(400, 500)], [span(400, 500)]),
            align("b", [chunk(0, 10)], [span(400, 500)]),
        ]
        m = compute_metrics(results, k=5, threshold=0.5)
        assert m.unanswered == 1
        assert m.hit_at_k == pytest.approx(0.5)

    def test_chars_retrieved_respects_k(self):
        r = [align("q", [chunk(0, 100), chunk(100, 400)], [span(0, 100)])]
        assert compute_metrics(r, k=1, threshold=0.5).mean_chars_retrieved == 100
        assert compute_metrics(r, k=2, threshold=0.5).mean_chars_retrieved == 400


class TestOffsetInvariant:
    def test_mismatched_text_is_rejected(self):
        doc = Document("d", "p", "hello world")
        with pytest.raises(OffsetError):
            Chunk("d", 0, 5, "WRONG").validate_against(doc)

    def test_matching_text_passes(self):
        doc = Document("d", "p", "hello world")
        Chunk("d", 0, 5, "hello").validate_against(doc)  # no raise

    @pytest.mark.parametrize(
        "chunker",
        [
            FixedSizeChunker(size=200, overlap=40),
            RecursiveChunker(size=200, overlap=40),
            HeadingChunker(max_size=300, min_size=60),
        ],
        ids=["fixed", "recursive", "heading"],
    )
    def test_every_chunker_preserves_offsets(self, chunker):
        text = (
            "# Title\n\nIntro paragraph that is reasonably long so it gets split.\n\n"
            "## Section one\n\nSome body text here.\n\n```python\n# not a heading\n"
            "x = 1\n```\n\n## Section two\n\nMore text.\n" * 4
        )
        doc = Document("d.md", "d.md", text)
        chunks = chunker.split(doc)
        assert chunks
        for c in chunks:
            c.validate_against(doc)

    def test_heading_detection_skips_fenced_code(self):
        """A '# comment' inside a code block must not start a section."""
        text = "# Real\n\nbody text goes here\n\n```python\n# fake heading\ny = 2\n```\n\nmore body\n"
        doc = Document("d.md", "d.md", text)
        chunks = HeadingChunker(max_size=5000, min_size=1).split(doc)
        assert len(chunks) == 1  # one real heading -> one section


class TestSpanValidation:
    def test_negative_and_inverted_spans_rejected(self):
        with pytest.raises(ValueError):
            Span("d", -1, 5)
        with pytest.raises(ValueError):
            Span("d", 10, 5)
