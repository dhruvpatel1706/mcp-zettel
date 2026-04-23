"""Optional semantic search over the note corpus.

Uses fastembed (ONNX, CPU-only, no torch) so we don't drag in a heavy ML
dependency just for this. Embeddings for all notes are held in memory keyed
by note id; the cache is invalidated whenever a note is written or deleted.

At low-thousands scale (which is where a personal zettelkasten lives) a full
in-memory numpy matrix + dot-product search is faster and simpler than any
vector DB. If you blow past ~50k notes this will start to feel it — swap in
LanceDB at that point.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mcp_zettel.models import Note, SearchHit

if TYPE_CHECKING:  # avoid importing fastembed at module load (it's optional-ish)
    from fastembed import TextEmbedding


@dataclass
class _CachedVec:
    id: str
    title: str
    tags: tuple[str, ...]
    body_snippet: str
    vec: tuple[float, ...]


_model_lock = threading.Lock()
_model: "TextEmbedding | None" = None
_model_name_loaded: str | None = None


def _get_model(model_name: str) -> "TextEmbedding":
    global _model, _model_name_loaded
    with _model_lock:
        if _model is None or _model_name_loaded != model_name:
            from fastembed import TextEmbedding

            _model = TextEmbedding(model_name=model_name)
            _model_name_loaded = model_name
        return _model


def _embed_one(text: str, *, model_name: str) -> tuple[float, ...]:
    model = _get_model(model_name)
    vec = next(iter(model.embed([text])))
    return tuple(float(x) for x in vec.tolist())


def _embed_many(texts: list[str], *, model_name: str) -> list[tuple[float, ...]]:
    if not texts:
        return []
    model = _get_model(model_name)
    return [tuple(float(x) for x in v.tolist()) for v in model.embed(texts)]


def _cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    # vectors come out L2-ish from BGE; do the defensive thing anyway
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a)) or 1.0
    db = math.sqrt(sum(y * y for y in b)) or 1.0
    return num / (da * db)


class SemanticIndex:
    """In-memory embedding index. Rebuilt on demand."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self._items: dict[str, _CachedVec] = {}
        self._built = False

    def clear(self) -> None:
        self._items.clear()
        self._built = False

    def build(self, notes: list[Note]) -> None:
        """Replace the in-memory index with embeddings for `notes`."""
        # Combining title + first chunk of body gives the embedding a chance to
        # anchor on both. Don't embed the full body — BGE truncates anyway and
        # title/lede is where the topic lives.
        to_embed: list[str] = []
        for n in notes:
            tag_str = " ".join(n.tags)
            head = n.body[:800]
            to_embed.append(f"{n.title}\n{tag_str}\n{head}".strip())
        vecs = _embed_many(to_embed, model_name=self.model_name)
        self._items = {}
        for n, v in zip(notes, vecs):
            self._items[n.id] = _CachedVec(
                id=n.id,
                title=n.title,
                tags=tuple(n.tags),
                body_snippet=n.body[:200].replace("\n", " "),
                vec=v,
            )
        self._built = True

    @property
    def ready(self) -> bool:
        return self._built

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        tags: list[str] | None = None,
    ) -> list[SearchHit]:
        if not self._built or not self._items:
            return []
        qvec = _embed_one(query, model_name=self.model_name)
        tagset = set(tags or [])
        scored: list[tuple[float, _CachedVec]] = []
        for item in self._items.values():
            if tagset and not tagset.issubset(set(item.tags)):
                continue
            scored.append((_cosine(qvec, item.vec), item))
        scored.sort(key=lambda x: -x[0])
        hits: list[SearchHit] = []
        for score, item in scored[:limit]:
            hits.append(
                SearchHit(
                    id=item.id,
                    title=item.title,
                    snippet=item.body_snippet,
                    score=float(score),
                    tags=list(item.tags),
                )
            )
        return hits
