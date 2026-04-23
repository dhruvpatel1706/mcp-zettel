"""File-based persistence: markdown files with YAML frontmatter."""

from __future__ import annotations

import re
import secrets
from datetime import datetime, timezone
from pathlib import Path

import yaml

from mcp_zettel.models import Note


class NoteNotFoundError(KeyError):
    """Raised when a note ID can't be resolved to a file."""


WIKI_LINK = re.compile(r"\[\[([a-f0-9]{6,10})(?:\|([^\]]+))?\]\]")


def _note_path(root: Path, note_id: str) -> Path:
    return root / "notes" / f"{note_id}.md"


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split YAML frontmatter from body. Returns (meta, body)."""
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end == -1:
        return {}, text
    meta_raw = text[4:end]
    body = text[end + 5 :]
    meta = yaml.safe_load(meta_raw) or {}
    if not isinstance(meta, dict):
        meta = {}
    return meta, body


def _render_frontmatter(meta: dict) -> str:
    """Render YAML frontmatter block. Keys are written in a stable order."""
    ordered: dict = {}
    for key in ("title", "tags", "created_at", "updated_at"):
        if key in meta:
            ordered[key] = meta[key]
    dumped = yaml.safe_dump(ordered, sort_keys=False, allow_unicode=True).strip()
    return f"---\n{dumped}\n---\n\n"


class Store:
    """Filesystem-backed note store. Notes live under `<root>/notes/<id>.md`."""

    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()
        (self.root / "notes").mkdir(parents=True, exist_ok=True)

    def new_id(self) -> str:
        """Generate a fresh short hex ID that isn't already in use."""
        while True:
            candidate = secrets.token_hex(3)  # 6 chars, 16M combos
            if not _note_path(self.root, candidate).exists():
                return candidate

    def create(self, title: str, body: str, tags: list[str] | None = None) -> Note:
        tags = sorted(set(tags or []))
        note = Note(id=self.new_id(), title=title, body=body, tags=tags)
        self._write(note)
        return note

    def get(self, note_id: str) -> Note:
        path = _note_path(self.root, note_id)
        if not path.exists():
            raise NoteNotFoundError(note_id)
        text = path.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(text)
        return Note(
            id=note_id,
            title=meta.get("title", "(untitled)"),
            body=body.strip("\n"),
            tags=list(meta.get("tags", []) or []),
            created_at=_to_datetime(meta.get("created_at")),
            updated_at=_to_datetime(meta.get("updated_at")),
        )

    def update(
        self,
        note_id: str,
        *,
        title: str | None = None,
        body: str | None = None,
        tags: list[str] | None = None,
    ) -> Note:
        note = self.get(note_id)
        if title is not None:
            note.title = title
        if body is not None:
            note.body = body
        if tags is not None:
            note.tags = sorted(set(tags))
        note.updated_at = datetime.now(timezone.utc)
        self._write(note)
        return note

    def delete(self, note_id: str) -> None:
        path = _note_path(self.root, note_id)
        if not path.exists():
            raise NoteNotFoundError(note_id)
        path.unlink()

    def list_all(self, *, tags: list[str] | None = None) -> list[Note]:
        notes: list[Note] = []
        for p in sorted((self.root / "notes").glob("*.md")):
            try:
                n = self.get(p.stem)
            except NoteNotFoundError:
                continue
            if tags and not set(tags).issubset(set(n.tags)):
                continue
            notes.append(n)
        return notes

    def linked_ids(self, note_id: str) -> list[str]:
        """Extract all [[id]] references in the body of this note."""
        note = self.get(note_id)
        return sorted({m.group(1) for m in WIKI_LINK.finditer(note.body)})

    def backlinks(self, note_id: str) -> list[Note]:
        """Find all notes whose body contains a [[note_id]] link."""
        results: list[Note] = []
        target = f"[[{note_id}"
        for n in self.list_all():
            if target in n.body:
                results.append(n)
        return results

    def add_link(self, from_id: str, to_id: str, label: str = "") -> Note:
        """Append a markdown link line to `from_id`'s body."""
        from_note = self.get(from_id)
        # Ensure target exists (raises NoteNotFoundError if missing)
        self.get(to_id)
        link_text = f"- [[{to_id}]]" if not label else f"- [[{to_id}]] — {label}"
        new_body = from_note.body.rstrip() + "\n\n" + link_text + "\n"
        return self.update(from_id, body=new_body)

    # ----- internal helpers -----

    def _write(self, note: Note) -> None:
        path = _note_path(self.root, note.id)
        meta = {
            "title": note.title,
            "tags": note.tags,
            "created_at": note.created_at.isoformat(),
            "updated_at": note.updated_at.isoformat(),
        }
        content = _render_frontmatter(meta) + note.body.strip() + "\n"
        path.write_text(content, encoding="utf-8")


def _to_datetime(val) -> datetime:  # type: ignore[no-untyped-def]
    if isinstance(val, datetime):
        if val.tzinfo is None:
            return val.replace(tzinfo=timezone.utc)
        return val
    if isinstance(val, str):
        try:
            dt = datetime.fromisoformat(val)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            pass
    return datetime.now(timezone.utc)
