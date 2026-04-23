"""Build a mermaid graph diagram from the note link structure.

Mermaid renders inline in most MCP clients (Claude Desktop, Obsidian plugins,
anything that speaks markdown+mermaid). For larger vaults the full graph gets
noisy — callers can filter to a tag subset via `zettel://graph/tag/{tag}`.
"""

from __future__ import annotations

import re

from mcp_zettel.models import Note
from mcp_zettel.storage import WIKI_LINK


def _safe_label(text: str, max_len: int = 40) -> str:
    """Make a string safe to drop inside a mermaid node label.

    Strip double-quotes (they end the label), trim to `max_len`, escape any
    stray backslashes. Mermaid's own escapes are `\"` -> quoting — simplest to
    just remove them.
    """
    text = text.replace('"', "'").replace("\\", "/")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_len:
        text = text[: max_len - 1].rstrip() + "…"
    return text


def _outbound_links(note: Note) -> list[tuple[str, str]]:
    """Return (target_id, label) for each [[target]] or [[target|label]] in body."""
    out = []
    for m in WIKI_LINK.finditer(note.body):
        target_id = m.group(1)
        label = (m.group(2) or "").strip()
        out.append((target_id, label))
    return out


def build_mermaid(notes: list[Note], *, focus_tag: str | None = None) -> str:
    """Produce a mermaid `graph LR` document describing the note graph.

    When `focus_tag` is set, only notes with that tag and their immediate
    neighbors (inbound or outbound) are included, so the view isn't
    dominated by unrelated regions of the graph.
    """
    if not notes:
        return 'graph LR\n  empty["(no notes yet)"]'

    by_id = {n.id: n for n in notes}
    if focus_tag:
        # 1. seed: notes with the tag
        seeds = {n.id for n in notes if focus_tag in n.tags}
        # 2. pull in any note directly linked from/to a seed
        neighbors = set()
        for n in notes:
            for target, _ in _outbound_links(n):
                if n.id in seeds and target in by_id:
                    neighbors.add(target)
                if target in seeds:
                    neighbors.add(n.id)
        visible = seeds | neighbors
        notes = [by_id[nid] for nid in visible if nid in by_id]

    if not notes:
        return f'graph LR\n  empty["(no notes match tag {focus_tag!r})"]'

    lines = ["graph LR"]
    # Declare each node explicitly so isolated notes still appear in the diagram.
    declared: set[str] = set()
    for n in notes:
        label = _safe_label(n.title)
        lines.append(f'  {n.id}["{label}"]')
        declared.add(n.id)

    for n in notes:
        for target_id, link_label in _outbound_links(n):
            if target_id not in declared:
                # link points to a note that was filtered out (or is missing)
                continue
            if link_label:
                safe = _safe_label(link_label, max_len=24)
                lines.append(f'  {n.id} -- "{safe}" --> {target_id}')
            else:
                lines.append(f"  {n.id} --> {target_id}")

    return "\n".join(lines)
