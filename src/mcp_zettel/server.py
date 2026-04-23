"""FastMCP server exposing the zettel store as MCP tools + resources."""

from __future__ import annotations

import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from mcp_zettel.models import Note, SearchHit
from mcp_zettel.search import search as _search
from mcp_zettel.storage import NoteNotFoundError, Store

DEFAULT_ROOT = Path(os.environ.get("MCP_ZETTEL_ROOT", Path.home() / ".mcp-zettel"))


def build_server(root: Path | None = None) -> FastMCP:
    """Construct the MCP server bound to a given storage root."""
    store = Store(root or DEFAULT_ROOT)
    mcp = FastMCP(
        "zettel",
        instructions=(
            "Zettelkasten personal knowledge base. Notes are atomic markdown documents "
            "identified by 6-10 char hex IDs. Use [[note_id]] wiki-links in note bodies "
            "to cross-reference. Prefer many small focused notes over few long ones."
        ),
    )

    @mcp.tool()
    def create_note(title: str, body: str, tags: list[str] | None = None) -> Note:
        """Create a new atomic note. Returns the note (including its generated id).

        Use [[other_note_id]] in `body` to link to existing notes.
        """
        return store.create(title=title, body=body, tags=list(tags or []))

    @mcp.tool()
    def read_note(note_id: str) -> Note:
        """Fetch a single note by id (body + tags + timestamps)."""
        try:
            return store.get(note_id)
        except NoteNotFoundError as exc:
            raise ValueError(f"Note {note_id!r} not found.") from exc

    @mcp.tool()
    def update_note(
        note_id: str,
        title: str | None = None,
        body: str | None = None,
        tags: list[str] | None = None,
    ) -> Note:
        """Update fields of an existing note. Any omitted field is left unchanged."""
        try:
            return store.update(note_id, title=title, body=body, tags=tags)
        except NoteNotFoundError as exc:
            raise ValueError(f"Note {note_id!r} not found.") from exc

    @mcp.tool()
    def delete_note(note_id: str) -> str:
        """Delete a note permanently. Other notes' [[note_id]] links become dangling."""
        try:
            store.delete(note_id)
        except NoteNotFoundError as exc:
            raise ValueError(f"Note {note_id!r} not found.") from exc
        return f"Deleted note {note_id}."

    @mcp.tool()
    def list_notes(tags: list[str] | None = None) -> list[Note]:
        """List all notes, optionally filtered to those having ALL of the given tags."""
        return store.list_all(tags=list(tags or []) or None)

    @mcp.tool()
    def search_notes(
        query: str,
        tags: list[str] | None = None,
        limit: int = 10,
    ) -> list[SearchHit]:
        """Search notes by keyword. Hits are ranked — titles and tags weigh more than body."""
        notes = store.list_all()
        return _search(notes, query, tags=list(tags or []) or None, limit=limit)

    @mcp.tool()
    def link_notes(from_id: str, to_id: str, label: str = "") -> Note:
        """Append a [[to_id]] link at the end of `from_id`'s body. Optional `label`."""
        try:
            return store.add_link(from_id, to_id, label=label)
        except NoteNotFoundError as exc:
            raise ValueError(f"One of the notes doesn't exist: {exc}") from exc

    @mcp.tool()
    def get_backlinks(note_id: str) -> list[Note]:
        """Return every note whose body contains a [[note_id]] reference to this one."""
        return store.backlinks(note_id)

    @mcp.tool()
    def linked_notes(note_id: str) -> list[str]:
        """Return the ids of notes that `note_id`'s body references via [[...]]."""
        try:
            return store.linked_ids(note_id)
        except NoteNotFoundError as exc:
            raise ValueError(f"Note {note_id!r} not found.") from exc

    @mcp.resource("zettel://all")
    def all_notes_resource() -> str:
        """Index of every note, one per line."""
        lines = []
        for n in store.list_all():
            tagline = f" [{', '.join(n.tags)}]" if n.tags else ""
            lines.append(f"{n.id}: {n.title}{tagline}")
        return "\n".join(lines) if lines else "(no notes yet)"

    @mcp.resource("zettel://{note_id}")
    def single_note_resource(note_id: str) -> str:
        """Full markdown content of a note (frontmatter + body)."""
        try:
            n = store.get(note_id)
        except NoteNotFoundError:
            return f"(note {note_id!r} not found)"
        tags = ", ".join(n.tags) if n.tags else "(none)"
        return f"# {n.title}\n\n_tags: {tags}_\n_updated: {n.updated_at.isoformat()}_\n\n{n.body}"

    return mcp


def main() -> None:
    """Entry point for `mcp-zettel-server`. Runs over stdio (the standard MCP transport)."""
    server = build_server()
    server.run()


if __name__ == "__main__":
    main()
