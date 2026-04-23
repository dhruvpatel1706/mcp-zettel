"""Management CLI (separate from the MCP server)."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from mcp_zettel import __version__
from mcp_zettel.search import search as _search
from mcp_zettel.server import DEFAULT_ROOT
from mcp_zettel.storage import NoteNotFoundError, Store

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Manage your zettelkasten from the command line (outside the MCP server).",
)
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"mcp-zettel {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    return


@app.command("list")
def list_cmd(
    root: Path = typer.Option(DEFAULT_ROOT, help="Storage root."),
    tag: list[str] = typer.Option(None, "--tag", "-t", help="Filter by tag (repeatable)."),
) -> None:
    """List notes in a table."""
    store = Store(root)
    notes = store.list_all(tags=list(tag) if tag else None)
    if not notes:
        console.print("[dim]No notes found.[/dim]")
        return
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("id", no_wrap=True)
    table.add_column("title")
    table.add_column("tags")
    table.add_column("updated", no_wrap=True)
    for n in notes:
        table.add_row(n.id, n.title, ", ".join(n.tags), n.updated_at.strftime("%Y-%m-%d"))
    console.print(table)


@app.command("create")
def create_cmd(
    title: str = typer.Argument(...),
    body: str = typer.Option("", "--body", "-b", help="Note body (markdown)."),
    tag: list[str] = typer.Option(None, "--tag", "-t", help="Tag (repeatable)."),
    root: Path = typer.Option(DEFAULT_ROOT, help="Storage root."),
) -> None:
    """Create a new note."""
    store = Store(root)
    note = store.create(title=title, body=body, tags=list(tag) if tag else [])
    console.print(f"[green]Created[/green] {note.id}: {note.title}")


@app.command("show")
def show_cmd(
    note_id: str,
    root: Path = typer.Option(DEFAULT_ROOT, help="Storage root."),
) -> None:
    """Print a single note to stdout."""
    store = Store(root)
    try:
        n = store.get(note_id)
    except NoteNotFoundError:
        console.print(f"[red]Note {note_id!r} not found[/red]")
        raise typer.Exit(1)
    console.print(f"[bold]{n.title}[/bold]  [dim]({n.id})[/dim]")
    if n.tags:
        console.print(f"[dim]tags: {', '.join(n.tags)}[/dim]")
    console.print(f"\n{n.body}\n")


@app.command("search")
def search_cmd(
    query: str,
    tag: list[str] = typer.Option(None, "--tag", "-t"),
    limit: int = typer.Option(10, "--limit", "-l"),
    root: Path = typer.Option(DEFAULT_ROOT, help="Storage root."),
) -> None:
    """Keyword search over all notes."""
    store = Store(root)
    notes = store.list_all()
    hits = _search(notes, query, tags=list(tag) if tag else None, limit=limit)
    if not hits:
        console.print("[dim]No matches.[/dim]")
        return
    for h in hits:
        console.print(f"\n[bold]{h.title}[/bold]  [dim]({h.id}) score={h.score:.1f}[/dim]")
        if h.tags:
            console.print(f"[dim]tags: {', '.join(h.tags)}[/dim]")
        console.print(f"{h.snippet}")


@app.command("backlinks")
def backlinks_cmd(
    note_id: str,
    root: Path = typer.Option(DEFAULT_ROOT, help="Storage root."),
) -> None:
    """Show all notes that link TO this one."""
    store = Store(root)
    notes = store.backlinks(note_id)
    if not notes:
        console.print("[dim]No backlinks.[/dim]")
        return
    for n in notes:
        console.print(f"{n.id}: {n.title}")


@app.command("serve")
def serve_cmd() -> None:
    """Start the MCP server on stdio (same as `mcp-zettel-server`)."""
    from mcp_zettel.server import main as server_main

    server_main()


if __name__ == "__main__":
    app()
