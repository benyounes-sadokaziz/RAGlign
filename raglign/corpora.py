"""The corpus registry.

v1 hardcoded one corpus, which quietly made every conclusion a statement about
FastAPI documentation rather than about chunking. A corpus is now a named,
first-class object that flows through loading, ground truth, runs and results.

Adding a third corpus is a dict entry plus a QA file -- deliberately, because
"works on any corpus" is the project's claim and a claim that costs a code
change to exercise is not really true.

Every run manifest records both the corpus name and its content fingerprint, so
results from different corpora can never be silently averaged together: they
answer different questions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS_ROOT = ROOT / "corpus"
QA_ROOT = ROOT / "data" / "qa"


@dataclass(frozen=True)
class Corpus:
    name: str
    path: Path
    pattern: str
    description: str
    # What makes this corpus structurally distinct. Recorded because the whole
    # point of having more than one is that the winning config differs, and the
    # reader needs to know what differs about the input.
    structure: str

    @property
    def qa_dir(self) -> Path:
        return QA_ROOT / self.name

    def qa_sets(self) -> list[str]:
        if not self.qa_dir.exists():
            return []
        return sorted(p.name for p in self.qa_dir.glob("*.json"))


CORPORA: dict[str, Corpus] = {
    "fastapi": Corpus(
        name="fastapi",
        path=CORPUS_ROOT / "fastapi",
        pattern="**/*.md",
        description="FastAPI documentation (tutorial, advanced, how-to)",
        structure="dense markdown: deep heading hierarchy, fenced code blocks, bullet lists",
    ),
    "prose": Corpus(
        name="prose",
        path=CORPUS_ROOT / "prose",
        pattern="**/*.txt",
        description="Public-domain prose: physics, biology, economics, memoir",
        structure="unstructured: flowing paragraphs, no markup, no headings, no code",
    ),
}

DEFAULT_CORPUS = "fastapi"


def get(name: str) -> Corpus:
    if name not in CORPORA:
        raise KeyError(f"unknown corpus {name!r}; known: {', '.join(sorted(CORPORA))}")
    return CORPORA[name]


def names() -> list[str]:
    return sorted(CORPORA)
