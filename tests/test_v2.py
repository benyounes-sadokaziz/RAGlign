"""Tests for the v2 additions: corpus registry, bootstrap CIs, failure diagnosis."""

from __future__ import annotations

import pytest

from raglign import corpora
from raglign.alignment import align
from raglign.diagnosis import Cause, diagnose_config, diagnose_question, recommended_action
from raglign.metrics import bootstrap_ci, indistinguishable
from raglign.models import Chunk, Span


def span(a: int, b: int, doc: str = "d") -> Span:
    return Span(doc, a, b)


def chunk(a: int, b: int, doc: str = "d") -> Chunk:
    return Chunk(doc, a, b, "x" * (b - a))


class TestCorpusRegistry:
    def test_registered_corpora_exist_on_disk(self):
        for name in corpora.names():
            c = corpora.get(name)
            assert c.path.exists(), f"{name} missing at {c.path}"
            assert list(c.path.glob(c.pattern)), f"{name} matched no files"

    def test_every_corpus_has_a_qa_set(self):
        """A corpus with no ground truth cannot be evaluated, so it is a broken entry."""
        for name in corpora.names():
            assert corpora.get(name).qa_sets(), f"{name} has no QA sets"

    def test_unknown_corpus_is_an_error(self):
        with pytest.raises(KeyError):
            corpora.get("nope")


class TestBootstrap:
    def _results(self, n_hit: int, n_miss: int):
        hits = [align(f"h{i}", [chunk(0, 100)], [span(0, 100)]) for i in range(n_hit)]
        misses = [align(f"m{i}", [chunk(500, 600)], [span(0, 100)]) for i in range(n_miss)]
        return hits + misses

    def test_interval_brackets_the_point_estimate(self):
        point, lo, hi = bootstrap_ci(self._results(8, 4), k=5, threshold=0.5, metric="mrr")
        assert lo <= point <= hi

    def test_unanimous_results_give_a_degenerate_interval(self):
        """All questions identical -> every resample is identical -> zero width."""
        point, lo, hi = bootstrap_ci(self._results(10, 0), k=5, threshold=0.5, metric="mrr")
        assert point == lo == hi == 1.0

    def test_more_questions_narrow_the_interval(self):
        """The whole reason CIs were added: n is what buys resolution."""
        _, lo_small, hi_small = bootstrap_ci(self._results(6, 6), k=5, threshold=0.5, metric="hit_at_k")
        _, lo_big, hi_big = bootstrap_ci(self._results(60, 60), k=5, threshold=0.5, metric="hit_at_k")
        assert (hi_big - lo_big) < (hi_small - lo_small)

    def test_is_deterministic_for_a_fixed_seed(self):
        a = bootstrap_ci(self._results(7, 5), k=5, threshold=0.5, seed=1)
        b = bootstrap_ci(self._results(7, 5), k=5, threshold=0.5, seed=1)
        assert a == b

    def test_overlap_detection(self):
        assert indistinguishable((0.5, 0.3, 0.7), (0.6, 0.4, 0.8))
        assert not indistinguishable((0.5, 0.4, 0.45), (0.9, 0.8, 0.95))


class TestDiagnosis:
    def test_unreachable_evidence_blames_the_chunker(self):
        f = diagnose_question({"qid": "q", "rank": None, "reachable": 0.0, "best_coverage": 0.4})
        assert f.cause is Cause.CHUNKER_DESTROYED
        assert "chunking" in f.fix

    def test_chunker_blamed_even_when_a_rank_exists(self):
        """A good rank on unreachable evidence is a partial-overlap coincidence.

        Blaming the retriever here would send the user to swap embedding models
        for a problem no embedding model can fix.
        """
        f = diagnose_question({"qid": "q", "rank": 1, "reachable": 0.0, "best_coverage": 0.3})
        assert f.cause is Cause.CHUNKER_DESTROYED

    def test_reachable_but_unretrieved_blames_the_retriever(self):
        f = diagnose_question(
            {"qid": "q", "rank": None, "reachable": 1.0, "max_achievable": 1.0, "best_coverage": 0.0}
        )
        assert f.cause is Cause.RETRIEVER_MISSED

    def test_partial_survival_blames_the_chunker_not_the_retriever(self):
        """Regression for a real misattribution.

        A chunker whose best chunk covers 60% of every answer passes the tau=0.5
        reachability check and looks blameless, so every failure landed on the
        retriever. fixed300 showed zero chunker failures while its oracle
        reachability was collapsing. max_achievable catches it.
        """
        f = diagnose_question(
            {"qid": "q", "rank": None, "reachable": 1.0, "max_achievable": 0.6, "best_coverage": 0.6}
        )
        assert f.cause is Cause.CHUNKER_DEGRADED
        assert "chunk size" in f.fix

    def test_degraded_does_not_fire_when_evidence_was_retrieved(self):
        """Partial coverage that still ranked first is not a failure to report."""
        f = diagnose_question(
            {"qid": "q", "rank": 1, "reachable": 1.0, "max_achievable": 0.6, "best_coverage": 0.6}
        )
        assert f.cause is Cause.OK

    def test_missing_max_achievable_defaults_to_intact(self):
        """Older manifests lack the field; they must not all read as degraded."""
        f = diagnose_question({"qid": "q", "rank": None, "reachable": 1.0, "best_coverage": 0.0})
        assert f.cause is Cause.RETRIEVER_MISSED

    def test_reranker_blamed_only_when_the_baseline_had_it(self):
        stats = {"qid": "q", "rank": None, "reachable": 1.0, "max_achievable": 1.0, "best_coverage": 0.0}
        assert diagnose_question(stats, {"rank": 2}).cause is Cause.RERANKER_DEMOTED
        assert diagnose_question(stats, {"rank": None}).cause is Cause.RETRIEVER_MISSED

    def test_rank_one_is_ok_and_worse_ranks_are_flagged(self):
        base = {"qid": "q", "reachable": 1.0, "best_coverage": 1.0}
        assert diagnose_question({**base, "rank": 1}).cause is Cause.OK
        assert diagnose_question({**base, "rank": 4}).cause is Cause.RANKED_LOW

    def test_recommended_action_prioritises_the_chunker(self):
        """Chunking caps every retriever downstream, so it outranks a more
        frequent but less constraining cause."""
        per_q = {
            "a": {"rank": None, "reachable": 0.0, "max_achievable": 0.1, "best_coverage": 0.1},
            "b": {"rank": 3, "reachable": 1.0, "max_achievable": 1.0, "best_coverage": 1.0},
            "c": {"rank": 3, "reachable": 1.0, "max_achievable": 1.0, "best_coverage": 1.0},
        }
        action = recommended_action(diagnose_config(per_q))
        assert "chunking" in action
        assert action.startswith("1/3")

    def test_clean_config_recommends_nothing(self):
        per_q = {"a": {"rank": 1, "reachable": 1.0, "max_achievable": 1.0, "best_coverage": 1.0}}
        assert recommended_action(diagnose_config(per_q)) == ""


class TestScriptsStayWired:
    """Every script must at least import and parse --help.

    run_phase1.py silently broke during the v2 loader refactor and stayed broken
    in the repo, because nothing exercised it. A shipped script that crashes on
    launch is worse than no script: it reads as rot. This is the cheapest check
    that catches a signature change breaking an entry point.
    """

    def test_every_script_runs_help(self):
        import subprocess
        import sys
        from pathlib import Path

        scripts = sorted(
            p for p in (Path(__file__).resolve().parent.parent / "scripts").glob("*.py")
            if not p.name.startswith("_")
        )
        assert scripts, "no scripts found"
        for path in scripts:
            proc = subprocess.run(
                [sys.executable, str(path), "--help"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            assert proc.returncode == 0, f"{path.name} --help failed:\n{proc.stderr[-800:]}"
