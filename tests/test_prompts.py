"""Tests for the prompt-template strings themselves (no MCP runtime needed)."""

from __future__ import annotations

from mcp_zettel.prompts import (
    daily_note,
    distill_conversation,
    find_linkable_notes,
    summarize_by_tag,
)


def test_distill_includes_transcript_and_note_cap():
    out = distill_conversation("user: hello\nclaude: hi", max_notes=3)
    assert "user: hello" in out
    assert "3" in out
    assert "atomic" in out.lower()


def test_find_linkable_uses_semantic_search():
    out = find_linkable_notes("contextual retrieval")
    assert "contextual retrieval" in out
    assert "search_notes_semantic" in out


def test_daily_note_defaults_to_today():
    from datetime import date

    out = daily_note()
    assert date.today().isoformat() in out


def test_daily_note_accepts_explicit_date():
    out = daily_note("2026-04-01")
    assert "2026-04-01" in out


def test_summarize_style_variants():
    bullets = summarize_by_tag("rag", style="bullets")
    essay = summarize_by_tag("rag", style="essay")
    outline = summarize_by_tag("rag", style="outline")
    assert "bulleted" in bullets.lower()
    assert "flowing paragraphs" in essay.lower()
    assert "outline" in outline.lower()


def test_summarize_unknown_style_falls_back():
    out = summarize_by_tag("rag", style="interpretive-dance")
    assert "bulleted digest" in out.lower()
