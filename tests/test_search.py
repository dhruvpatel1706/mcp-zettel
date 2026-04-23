"""Tests for keyword search and ranking."""

from __future__ import annotations

from mcp_zettel.models import Note
from mcp_zettel.search import search


def _n(note_id: str, title: str, body: str = "", tags: list[str] | None = None) -> Note:
    return Note(id=note_id, title=title, body=body, tags=tags or [])


def test_no_query_returns_empty():
    notes = [_n("a1b2c3", "Whatever", "content")]
    assert search(notes, "") == []


def test_title_match_wins_over_body():
    notes = [
        _n("aaaaaa", "About retrieval", "unrelated body text"),
        _n("bbbbbb", "Something else", "body mentions retrieval once"),
    ]
    hits = search(notes, "retrieval")
    assert hits[0].id == "aaaaaa"
    assert hits[0].score > hits[1].score


def test_tag_filter_is_intersection():
    notes = [
        _n("aaaaaa", "A", "ml content", tags=["ml"]),
        _n("bbbbbb", "B", "ml content", tags=["ml", "rag"]),
        _n("cccccc", "C", "ml content", tags=["web"]),
    ]
    hits = search(notes, "ml", tags=["rag"])
    assert [h.id for h in hits] == ["bbbbbb"]


def test_snippet_contains_query_context():
    body = "Lorem ipsum dolor sit. " * 5 + "The target word is retrieval. " + "more content " * 10
    notes = [_n("aaaaaa", "Unrelated title", body)]
    hits = search(notes, "retrieval")
    assert "retrieval" in hits[0].snippet.lower()


def test_limit_respected():
    notes = [_n(f"{i:06x}", f"note {i}", "rag rag rag") for i in range(20)]
    hits = search(notes, "rag", limit=5)
    assert len(hits) == 5


def test_zero_score_filtered_when_query_present():
    notes = [
        _n("aaaaaa", "matches", "rag here"),
        _n("bbbbbb", "no match", "completely unrelated"),
    ]
    hits = search(notes, "rag")
    assert len(hits) == 1
    assert hits[0].id == "aaaaaa"
