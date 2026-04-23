"""Pydantic models for notes and search results."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Note(BaseModel):
    """A single atomic note."""

    id: str = Field(description="Short hex ID, e.g. 'a3f2c9'. Used in [[wiki-links]].")
    title: str
    body: str = Field(description="Markdown body. [[note_id]] creates a link.")
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)


class SearchHit(BaseModel):
    """One result from a note search."""

    id: str
    title: str
    snippet: str = Field(description="~200 char excerpt showing the match.")
    score: float = Field(description="Relevance score (higher = more relevant).")
    tags: list[str] = Field(default_factory=list)


class Link(BaseModel):
    """A directed link between two notes."""

    from_id: str
    to_id: str
    label: str = Field(default="", description="Optional label, e.g. 'contradicts'.")
