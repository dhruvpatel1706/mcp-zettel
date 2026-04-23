"""Tests for the file-based note store."""

from __future__ import annotations

import pytest

from mcp_zettel.storage import NoteNotFoundError, Store, _parse_frontmatter


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path)


def test_create_and_read_roundtrip(store):
    n = store.create(title="Intro to RAG", body="RAG = retrieval + generation.", tags=["rag", "ml"])
    assert len(n.id) == 6
    back = store.get(n.id)
    assert back.title == n.title
    assert back.body == "RAG = retrieval + generation."
    assert back.tags == ["ml", "rag"]  # sorted on write


def test_update_preserves_id_and_touches_updated_at(store):
    n = store.create(title="Original", body="v1")
    updated = store.update(n.id, title="Rewritten", body="v2")
    assert updated.id == n.id
    assert updated.title == "Rewritten"
    assert updated.body == "v2"
    assert updated.updated_at >= n.updated_at


def test_delete_removes_note(store):
    n = store.create(title="Throwaway", body="x")
    store.delete(n.id)
    with pytest.raises(NoteNotFoundError):
        store.get(n.id)


def test_get_unknown_raises(store):
    with pytest.raises(NoteNotFoundError):
        store.get("does_not_exist")


def test_list_all_filters_by_tag(store):
    a = store.create(title="Alpha", body="x", tags=["ml", "rag"])
    b = store.create(title="Beta", body="y", tags=["ml", "agents"])
    store.create(title="Gamma", body="z", tags=["web"])
    ids = {n.id for n in store.list_all(tags=["ml"])}
    assert ids == {a.id, b.id}
    ids = {n.id for n in store.list_all(tags=["ml", "rag"])}
    assert ids == {a.id}


def test_bidirectional_links(store):
    a = store.create(title="A", body="Point to nothing yet.")
    b = store.create(title="B", body=f"See [[{a.id}]] for background.")
    # b references a
    assert store.linked_ids(b.id) == [a.id]
    # a has a backlink from b
    backs = store.backlinks(a.id)
    assert len(backs) == 1
    assert backs[0].id == b.id


def test_add_link_appends_wiki_link(store):
    a = store.create(title="A", body="first line")
    b = store.create(title="B", body="target")
    updated = store.add_link(a.id, b.id, label="related")
    assert f"[[{b.id}]]" in updated.body
    assert "related" in updated.body


def test_add_link_to_missing_raises(store):
    a = store.create(title="A", body="x")
    with pytest.raises(NoteNotFoundError):
        store.add_link(a.id, "ffffff")


def test_frontmatter_parse_no_frontmatter():
    meta, body = _parse_frontmatter("just body text, no yaml")
    assert meta == {}
    assert body == "just body text, no yaml"


def test_frontmatter_parse_happy_path():
    text = "---\ntitle: Hi\ntags:\n- a\n- b\n---\n\nbody here\n"
    meta, body = _parse_frontmatter(text)
    assert meta["title"] == "Hi"
    assert meta["tags"] == ["a", "b"]
    assert body.strip() == "body here"
