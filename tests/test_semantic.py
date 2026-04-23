"""Tests for the semantic index — network-free: we don't actually call fastembed."""

from __future__ import annotations

import sys
import types

from mcp_zettel.models import Note


def _make_stub_fastembed(monkeypatch):
    """Install a minimal stub module in place of fastembed.

    fastembed pulls in onnxruntime + downloads a model on first use; neither
    is fair to a unit test runner. We mock it with a deterministic hash-bucket
    embedder so the numeric behavior (cosine, ranking) is still meaningful.
    """

    class _StubEmbedding:
        def __init__(self, model_name: str):
            self.model_name = model_name

        def embed(self, texts):
            import hashlib

            for t in texts:
                h = hashlib.sha256(t.encode("utf-8")).digest()

                # 32 bytes -> 32 floats in [-1, 1]
                class _V:
                    def __init__(self, data):
                        self._data = data
                        self.shape = (len(data),)

                    def tolist(self):
                        return list(self._data)

                yield _V([((b - 128) / 128.0) for b in h])

    mod = types.ModuleType("fastembed")
    mod.TextEmbedding = _StubEmbedding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", mod)

    # Also reset the lazy-loaded module singleton inside mcp_zettel.semantic
    from mcp_zettel import semantic as sem_mod

    monkeypatch.setattr(sem_mod, "_model", None, raising=False)
    monkeypatch.setattr(sem_mod, "_model_name_loaded", None, raising=False)


def test_index_starts_empty(monkeypatch):
    _make_stub_fastembed(monkeypatch)
    from mcp_zettel.semantic import SemanticIndex

    idx = SemanticIndex()
    assert not idx.ready
    assert idx.search("anything") == []


def test_build_then_search_returns_ranked_hits(monkeypatch):
    _make_stub_fastembed(monkeypatch)
    from mcp_zettel.semantic import SemanticIndex

    notes = [
        Note(id="a1b2c3", title="RAG chunk sizing", body="800 chars, 15% overlap", tags=["rag"]),
        Note(id="d4e5f6", title="Kubernetes pod eviction", body="OOMKiller basics", tags=["k8s"]),
        Note(
            id="777aaa",
            title="Retrieval augmented generation",
            body="dense vs sparse",
            tags=["rag"],
        ),
    ]
    idx = SemanticIndex()
    idx.build(notes)
    assert idx.ready

    hits = idx.search("anything")
    # All notes come back; scores are sorted desc and every hit has the shape we
    # expect. We don't assert a specific ranking — the stub's cosine space is
    # pseudo-random — only that ranking exists and is deterministic.
    assert len(hits) == 3
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert {h.id for h in hits} == {"a1b2c3", "d4e5f6", "777aaa"}

    # Snippet comes from body and preserves first-200-chars behavior.
    by_id = {h.id: h for h in hits}
    assert by_id["a1b2c3"].snippet.startswith("800 chars")

    # Searching for the exact indexed text of a given note should rank that
    # note first (cosine of identical vectors = 1.0, beats all others).
    target = notes[0]
    query_text = f"{target.title}\n{' '.join(target.tags)}\n{target.body[:800]}".strip()
    exact_hits = idx.search(query_text, limit=1)
    assert exact_hits[0].id == target.id


def test_tag_filter_respected(monkeypatch):
    _make_stub_fastembed(monkeypatch)
    from mcp_zettel.semantic import SemanticIndex

    notes = [
        Note(id="aaaaaa", title="alpha", body="x", tags=["rag"]),
        Note(id="bbbbbb", title="beta", body="y", tags=["k8s"]),
    ]
    idx = SemanticIndex()
    idx.build(notes)
    hits = idx.search("anything at all", tags=["k8s"])
    assert [h.id for h in hits] == ["bbbbbb"]


def test_clear_invalidates_cache(monkeypatch):
    _make_stub_fastembed(monkeypatch)
    from mcp_zettel.semantic import SemanticIndex

    idx = SemanticIndex()
    idx.build([Note(id="aaaaaa", title="x", body="y")])
    assert idx.ready
    idx.clear()
    assert not idx.ready
    assert idx.search("x") == []


def test_server_wires_semantic_tool(tmp_path, monkeypatch):
    """Integration-ish: the server exposes search_notes_semantic and it works
    through the same build/search path we tested above."""
    _make_stub_fastembed(monkeypatch)

    from mcp_zettel.semantic import SemanticIndex
    from mcp_zettel.server import build_server

    server = build_server(tmp_path, semantic=SemanticIndex())

    import asyncio

    async def _run():
        tools = await server.list_tools()
        names = {t.name for t in tools}
        assert "search_notes_semantic" in names

    asyncio.run(_run())
