"""Tests for the link-suggestion hybrid ranker."""

from __future__ import annotations

from datetime import datetime, timezone

from mcp_zettel.links import suggest_links
from mcp_zettel.models import Note, SearchHit


def _note(id_: str, title: str, body: str = "", tags: list[str] | None = None) -> Note:
    now = datetime(2026, 4, 1, tzinfo=timezone.utc)
    return Note(
        id=id_,
        title=title,
        body=body,
        tags=tags or [],
        created_at=now,
        updated_at=now,
    )


class _StaticSemantic:
    """Semantic index stub whose `search` just replays a fixed ranking."""

    def __init__(self, ranked_ids: list[str], notes: list[Note]):
        self.ranked_ids = ranked_ids
        self._notes_by_id = {n.id: n for n in notes}
        self.ready = False
        self.build_calls = 0

    def build(self, notes):
        self.build_calls += 1
        self._notes_by_id = {n.id: n for n in notes}
        self.ready = True

    def search(self, query, *, limit=10, tags=None):
        # Return hits in the order we were configured with, one per id we know.
        out: list[SearchHit] = []
        for i, note_id in enumerate(self.ranked_ids[:limit]):
            n = self._notes_by_id.get(note_id)
            if n is None:
                continue
            out.append(
                SearchHit(
                    id=n.id,
                    title=n.title,
                    snippet=n.body[:120],
                    score=float(limit - i),
                    tags=list(n.tags),
                )
            )
        return out


def test_keyword_only_when_semantic_is_none():
    notes = [
        _note("a", "Attention is all you need", body="transformer paper summary"),
        _note("b", "Transformer tokenizer notes", body="subword tokenization"),
        _note("c", "Cat videos", body="unrelated stuff"),
    ]
    hits = suggest_links(notes, None, "transformer attention")
    # Only the two relevant ones should come back — note c shares nothing.
    assert {h.id for h in hits} == {"a", "b"}


def test_hybrid_rrf_promotes_notes_found_by_both_retrievers(monkeypatch):
    notes = [
        _note("kw1", "KW one hit on 'transformer'", body="transformer"),
        _note("kw2", "KW unique", body="transformer attention"),
        _note("sem1", "Semantic unique", body="attention layers"),
        _note("both", "Agreed by both", body="transformer layers"),
    ]
    # Keyword order: both, kw2, kw1 (all contain "transformer")
    # Semantic order: both, sem1
    sem = _StaticSemantic(ranked_ids=["both", "sem1"], notes=notes)

    hits = suggest_links(notes, sem, "transformer layers", limit=3)
    # `both` should be #1 — it's in both lists. The other two slots go to the
    # next-ranked from each list.
    assert hits[0].id == "both"
    rest = {h.id for h in hits[1:]}
    # Either {kw2, sem1} in some order, but `both` must be first.
    assert "both" not in rest


def test_exclude_ids_dropped_from_both_retrievers():
    notes = [
        _note("a", "Attention", body="attention"),
        _note("b", "Transformer", body="transformer architecture"),
        _note("c", "BERT", body="bert attention"),
    ]
    sem = _StaticSemantic(ranked_ids=["a", "b", "c"], notes=notes)
    hits = suggest_links(notes, sem, "attention", exclude_ids={"a"}, limit=5)
    assert {h.id for h in hits} == {"b", "c"}


def test_limit_truncates_output():
    notes = [_note(f"n{i}", f"note {i}", body="common body") for i in range(10)]
    sem = _StaticSemantic(ranked_ids=[n.id for n in notes], notes=notes)
    hits = suggest_links(notes, sem, "common", limit=3)
    assert len(hits) == 3


def test_empty_corpus_returns_empty():
    hits = suggest_links([], None, "anything")
    assert hits == []


def test_no_matches_returns_empty_list():
    # Every note's body contains 'elephant' — query asks for something else
    # entirely with no semantic index fallback.
    notes = [_note("a", "Elephant notes", body="elephant trunks")]
    hits = suggest_links(notes, None, "quantum chromodynamics")
    assert hits == []


def test_semantic_builds_lazily_only_once(monkeypatch):
    notes = [_note("a", "Attention", body="attention paper")]
    sem = _StaticSemantic(ranked_ids=["a"], notes=notes)
    # First call should trigger build
    assert sem.build_calls == 0
    suggest_links(notes, sem, "attention")
    assert sem.build_calls == 1
    # Second call should NOT rebuild
    suggest_links(notes, sem, "attention")
    assert sem.build_calls == 1


def test_score_is_overwritten_with_fused_rank():
    # suggest_links should not leak the raw keyword TF-like score; callers
    # rely on the fused RRF number for comparison.
    notes = [_note("a", "Transformer", body="transformer attention paper")]
    hits = suggest_links(notes, None, "transformer")
    # Raw keyword score would be >=3 (title hit); RRF single-list is 1/(60+1).
    assert hits[0].score < 1.0
