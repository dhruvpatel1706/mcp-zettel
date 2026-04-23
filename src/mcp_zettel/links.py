"""Suggest existing notes to link when writing a new one.

The workflow this is trying to make frictionless: you're about to
`create_note` a thought, but you probably have related notes from last
week you've forgotten about. Before writing, the client (or the agent)
calls `suggest_links(text)` and gets back the closest existing notes by
both keyword and semantic match.

Ranking fuses the two retrievers with Reciprocal Rank Fusion (same trick
we use in `personal-rag` hybrid retrieval). Keyword catches proper nouns
and exact code identifiers; semantic catches paraphrases and synonyms.
Neither alone does both.
"""

from __future__ import annotations

from typing import Protocol

from mcp_zettel.models import Note, SearchHit
from mcp_zettel.search import search as keyword_search

# Classic RRF constant. The only reason it's 60 and not tuned is that every
# retrieval paper picks 60 and it works fine.
_RRF_CONST = 60


class _SemanticLike(Protocol):
    """Subset of SemanticIndex we actually need. Handy for injecting stubs."""

    ready: bool

    def build(self, notes: list[Note]) -> None: ...

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        tags: list[str] | None = None,
    ) -> list[SearchHit]: ...


def _rrf_fuse(
    rankings: list[list[str]],
    *,
    k: int = _RRF_CONST,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion across multiple ranked id lists."""
    score: dict[str, float] = {}
    for rank_list in rankings:
        for rank, doc_id in enumerate(rank_list, start=1):
            score[doc_id] = score.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(score.items(), key=lambda x: -x[1])


def suggest_links(
    notes: list[Note],
    semantic: _SemanticLike | None,
    text: str,
    *,
    exclude_ids: set[str] | None = None,
    limit: int = 5,
    overfetch: int = 4,
) -> list[SearchHit]:
    """Rank notes most likely to warrant a link from `text`.

    `semantic` can be None — the function gracefully falls back to keyword-only
    if the caller doesn't want the embedding cost. When provided, results are
    hybrid-fused with RRF.

    `exclude_ids` is the escape hatch for "don't suggest the note I just wrote
    as a link to itself" — the server passes the newly-created note's id.
    """
    exclude = set(exclude_ids or ())
    want = overfetch * limit

    # Build an id -> Note map so we can reassemble SearchHits after RRF.
    by_id = {n.id: n for n in notes}

    keyword_hits = keyword_search(notes, text, limit=want)
    keyword_ids = [h.id for h in keyword_hits if h.id not in exclude]

    semantic_ids: list[str] = []
    semantic_hits: list[SearchHit] = []
    if semantic is not None:
        if not semantic.ready:
            semantic.build(notes)
        semantic_hits = semantic.search(text, limit=want)
        semantic_ids = [h.id for h in semantic_hits if h.id not in exclude]

    # If neither retriever returned anything, bail cleanly.
    if not keyword_ids and not semantic_ids:
        return []

    # Fuse — keyword-only degrades naturally since it's the only ranking.
    rankings = [ids for ids in (keyword_ids, semantic_ids) if ids]
    fused = _rrf_fuse(rankings)

    # Rehydrate SearchHits from whichever retriever scored each id first,
    # preferring semantic's richer snippet when both had it.
    kw_by_id = {h.id: h for h in keyword_hits}
    sem_by_id = {h.id: h for h in semantic_hits}
    out: list[SearchHit] = []
    for doc_id, score in fused:
        if doc_id in exclude or doc_id not in by_id:
            continue
        source_hit = sem_by_id.get(doc_id) or kw_by_id.get(doc_id)
        if source_hit is None:
            note = by_id[doc_id]
            source_hit = SearchHit(
                id=note.id,
                title=note.title,
                snippet=note.body[:200].replace("\n", " "),
                score=0.0,
                tags=list(note.tags),
            )
        # Overwrite score with the fused rank score so callers can see a
        # consistent number (bigger = more confident).
        out.append(
            SearchHit(
                id=source_hit.id,
                title=source_hit.title,
                snippet=source_hit.snippet,
                score=float(score),
                tags=list(source_hit.tags),
            )
        )
        if len(out) >= limit:
            break
    return out
