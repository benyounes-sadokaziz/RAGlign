"""Local CPU embeddings via fastembed (ONNX), with an on-disk cache.

Caching matters more than it looks: the experiment grid re-embeds the same
chunk text across configs that differ only in retriever or reranker. Keyed by
(model, text) content hash, so a cache hit is correct by construction and
independent of chunk ordering or config identity.

The model cache is pinned into the project's .cache/ rather than the system temp
directory, so a reboot does not silently re-download 130 MB mid-experiment.

`embedding_text` is where the heading chunker's context prefix is applied. The
prefix deliberately lives in chunk metadata rather than chunk text (see
chunking/heading.py): splicing it into `text` would break the
doc.text[start:end] == text invariant that the whole alignment layer rests on.
Applying it here keeps the retrieval benefit without touching the offsets.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from .models import Chunk

CACHE_ROOT = Path(__file__).resolve().parent.parent / ".cache"
MODEL_CACHE = CACHE_ROOT / "models"
VEC_CACHE = CACHE_ROOT / "vectors"

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


def embedding_text(chunk: Chunk) -> str:
    """Text actually handed to the embedder for a chunk.

    Prepends the heading path when the chunker supplied one, so a chunk reading
    "You can also pass a list." embeds as
    "Query Parameters > Optional parameters\n\nYou can also pass a list."
    """
    prefix = chunk.meta.get("context_prefix")
    return f"{prefix}\n\n{chunk.text}" if prefix else chunk.text


class Embedder:
    """Thin wrapper over fastembed exposing encode(list[str]) -> np.ndarray.

    Kept deliberately narrow: swapping in sentence-transformers, OpenAI, or a
    GPU backend is a change to this class alone (TC-5).
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, *, cache: bool = True):
        self.model_name = model_name
        self.cache = cache
        MODEL_CACHE.mkdir(parents=True, exist_ok=True)
        VEC_CACHE.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        self._model = None  # loaded on first use; ~25 s cold, once per process

    def _ensure(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(self.model_name, cache_dir=str(MODEL_CACHE))
        return self._model

    def _cache_path(self, texts: Sequence[str]) -> Path:
        h = hashlib.sha256(self.model_name.encode())
        for t in texts:
            h.update(b"\x00")
            h.update(t.encode("utf-8"))
        return VEC_CACHE / f"{h.hexdigest()[:24]}.npy"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return L2-normalised float32 embeddings, shape [len(texts), dim].

        Normalising here means cosine similarity is a plain dot product
        downstream -- one less place for a similarity bug to hide.
        """
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)

        path = self._cache_path(texts)
        if self.cache and path.exists():
            return np.load(path)

        model = self._ensure()
        vecs = np.asarray(list(model.embed(list(texts))), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.maximum(norms, 1e-12)

        if self.cache:
            np.save(path, vecs)
        return vecs

    def encode_chunks(self, chunks: Sequence[Chunk]) -> np.ndarray:
        return self.encode([embedding_text(c) for c in chunks])
