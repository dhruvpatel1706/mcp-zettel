"""Tests for the mermaid graph builder."""

from __future__ import annotations

from mcp_zettel.graph import _safe_label, build_mermaid
from mcp_zettel.models import Note


def _n(nid: str, title: str, body: str = "", tags: list[str] | None = None) -> Note:
    return Note(id=nid, title=title, body=body, tags=tags or [])


def test_empty_returns_placeholder():
    out = build_mermaid([])
    assert out.startswith("graph LR")
    assert "no notes yet" in out


def test_single_note_with_no_links_still_declares_node():
    out = build_mermaid([_n("aaaaaa", "Intro to RAG")])
    assert "aaaaaa[" in out
    assert "Intro to RAG" in out


def test_edges_from_wiki_links():
    notes = [
        _n("aaaaaa", "A", body="see [[bbbbbb]] for context"),
        _n("bbbbbb", "B", body="body of B"),
    ]
    out = build_mermaid(notes)
    assert "aaaaaa --> bbbbbb" in out


def test_labeled_edge_keeps_label():
    notes = [
        _n("aaaaaa", "A", body="this [[bbbbbb|contradicts]] B"),
        _n("bbbbbb", "B"),
    ]
    out = build_mermaid(notes)
    assert 'aaaaaa -- "contradicts" --> bbbbbb' in out


def test_dangling_links_silently_dropped():
    # [[ffffff]] points to a non-existent note — we shouldn't declare a ghost node
    notes = [_n("aaaaaa", "A", body="see [[ffffff]] which was deleted")]
    out = build_mermaid(notes)
    assert "ffffff" not in out


def test_focus_tag_includes_neighbors():
    notes = [
        _n("aaaaaa", "rag note", body="see [[bbbbbb]]", tags=["rag"]),
        _n("bbbbbb", "general", body="unrelated"),
        _n("cccccc", "unrelated-2", body="nothing"),
    ]
    out = build_mermaid(notes, focus_tag="rag")
    # aaaaaa is seed (has tag), bbbbbb is its neighbor
    assert "aaaaaa[" in out
    assert "bbbbbb[" in out
    # cccccc isn't connected to anything rag-tagged
    assert "cccccc[" not in out


def test_focus_tag_no_matches():
    out = build_mermaid([_n("aaaaaa", "x", tags=["foo"])], focus_tag="bar")
    assert "no notes match tag" in out


def test_label_escapes_double_quotes():
    assert '"' not in _safe_label('some "quoted" thing')


def test_label_truncates_long_titles():
    out = _safe_label("a" * 80, max_len=20)
    assert len(out) == 20
    assert out.endswith("…")
