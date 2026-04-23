"""Lightweight scoring for note search.

Uses a simple TF-like scorer over title + body + tags. Good enough for a
personal zettelkasten (up to low thousands of notes). For larger corpora,
swap in BM25 or a vector store.
"""

from __future__ import annotations

import re

from mcp_zettel.models import Note, SearchHit

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _score(note: Note, query_tokens: list[str]) -> float:
    if not query_tokens:
        return 0.0
    title_tokens = _tokenize(note.title)
    body_tokens = _tokenize(note.body)
    tag_tokens = _tokenize(" ".join(note.tags))
    score = 0.0
    for qt in query_tokens:
        title_hits = title_tokens.count(qt)
        body_hits = body_tokens.count(qt)
        tag_hits = tag_tokens.count(qt)
        # Title hits and tag hits weighted more heavily than body hits.
        score += 3.0 * title_hits + 2.0 * tag_hits + 1.0 * body_hits
    return score


def _snippet(note: Note, query_tokens: list[str], width: int = 200) -> str:
    """Return a short excerpt around the first query match in the body."""
    if not query_tokens or not note.body:
        return note.body[:width].replace("\n", " ")
    lowered = note.body.lower()
    positions = [lowered.find(qt) for qt in query_tokens]
    positions = [p for p in positions if p >= 0]
    if not positions:
        return note.body[:width].replace("\n", " ")
    start = max(0, min(positions) - width // 3)
    end = min(len(note.body), start + width)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(note.body) else ""
    return (prefix + note.body[start:end] + suffix).replace("\n", " ")


def search(
    notes: list[Note],
    query: str,
    *,
    tags: list[str] | None = None,
    limit: int = 10,
) -> list[SearchHit]:
    """Score and rank `notes`. Tag filter is intersection (all required)."""
    tokens = _tokenize(query)
    if not tokens:
        return []
    hits: list[SearchHit] = []
    for note in notes:
        if tags and not set(tags).issubset(set(note.tags)):
            continue
        score = _score(note, tokens)
        if score == 0:
            continue
        hits.append(
            SearchHit(
                id=note.id,
                title=note.title,
                snippet=_snippet(note, tokens),
                score=score,
                tags=list(note.tags),
            )
        )
    hits.sort(key=lambda h: (-h.score, h.title.lower()))
    return hits[:limit]
