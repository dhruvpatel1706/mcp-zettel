# mcp-zettel

**Give Claude (or any MCP client) persistent memory in the form of a Zettelkasten.**

An [MCP](https://modelcontextprotocol.io) server that exposes a personal knowledge base of atomic, interconnected markdown notes to Claude Desktop, Claude Code, Cursor, or any MCP-compatible client. Plug it in and the assistant can create notes, cross-reference them with `[[wiki-links]]`, search by keyword or tag, and walk backlinks — letting it build and use durable context across sessions without you copy-pasting anything.

---

## Why

Every chat you start with an LLM begins with zero context about what you've already decided, written, or learned. The Zettelkasten method (small atomic notes + explicit links between them) is a strong fit for LLM-accessible memory: chunks are naturally small, links make relevance explicit, and the storage is plain markdown on your own disk.

This MCP server exposes that knowledge base to any LLM client via MCP tools, so the model can:
- **Create** a new note when you share a decision or insight worth keeping
- **Search** for notes on a topic before answering ("what did I decide about X?")
- **Link** notes bidirectionally to build up a graph ("this contradicts [[a3f2c9]]")
- **Walk backlinks** to find everything connected to a concept

You keep plain markdown files on disk. The model gets structured access to them.

## How it looks in a client

After connecting the server, the LLM can do things like this (your client will show actual tool calls):

```
> What did I conclude about RAG chunk sizes?
[searches notes with query "rag chunk size"]
[reads 2 matching notes]
Based on your notes a3f2c9 ("RAG chunk sizing") and b7e412 ("Sentence-boundary
splitting"), you concluded: 800 chars with ~15% overlap, sentence-aligned.
You flagged that pure character chunking ([[2f00a1]]) hurt recall on your
arxiv set and moved away from it.
```

## Exposed MCP tools

| Tool | Purpose |
| --- | --- |
| `create_note(title, body, tags)` | Create a new atomic note. Use `[[other_id]]` in body to link. |
| `read_note(note_id)` | Fetch a single note. |
| `update_note(note_id, title?, body?, tags?)` | Update any fields, leaving others unchanged. |
| `delete_note(note_id)` | Remove a note permanently. |
| `list_notes(tags?)` | List every note; optional tag filter is intersection. |
| `search_notes(query, tags?, limit?)` | Keyword search — titles and tags weigh more than body. |
| `search_notes_semantic(query, tags?, limit?)` | **v0.2.** Embedding-based search for conceptual queries. Uses fastembed on-device (no API call). |
| `link_notes(from_id, to_id, label?)` | Append a `[[to_id]]` wiki-link to `from_id`'s body. |
| `get_backlinks(note_id)` | Every note whose body references this one. |
| `linked_notes(note_id)` | The ids this note links to (outbound). |

Plus two MCP resources:
- `zettel://all` — one-line index of every note
- `zettel://{note_id}` — full rendered note

### Two search tools, not one

Keyword search is what you want when you know the term. It's cheap, the ranking is predictable, and exact matches always beat similar-sounding ones. Semantic search wins when the query wording doesn't match the note wording — asking for "rate limiting" when the note calls it "throttling", or "why my cache is cold" when the note is about "TTL tuning". The LLM can call whichever makes sense; the tool descriptions tell it which is which.

The embedding model is `BAAI/bge-small-en-v1.5` by default (384-dim, ~130 MB, CPU-only). Override with `MCP_ZETTEL_EMBEDDING_MODEL`. The index rebuilds lazily on the first semantic query after any write, so there's a short wait the first time — after that it stays in memory for the life of the server process.

## Install

```bash
git clone https://github.com/dhruvpatel1706/mcp-zettel.git
cd mcp-zettel
pip install -e .
```

Python 3.10+.

## Wire it up

### Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or the equivalent on your OS, and add:

```json
{
  "mcpServers": {
    "zettel": {
      "command": "mcp-zettel-server"
    }
  }
}
```

Restart Claude Desktop. The zettel tools are now available to the model.

### Claude Code

```bash
claude mcp add zettel -- mcp-zettel-server
```

### Cursor / Continue / any stdio MCP client

Point the client at `mcp-zettel-server` as a command; the server speaks MCP over stdio.

### Custom storage location

Set `MCP_ZETTEL_ROOT` to override the default `~/.mcp-zettel`:

```json
{
  "mcpServers": {
    "zettel": {
      "command": "mcp-zettel-server",
      "env": { "MCP_ZETTEL_ROOT": "/Users/you/vault" }
    }
  }
}
```

## Use the CLI directly (no MCP client needed)

The same store is accessible via a plain CLI — useful for power users who want to inspect, edit, or seed the knowledge base outside of an LLM session.

```bash
mcp-zettel create "RAG chunk sizing" \
  --body "Settled on 800 chars, 15% overlap, sentence-aligned. See [[b7e412]]." \
  --tag rag --tag decisions

mcp-zettel list --tag rag
mcp-zettel search "sentence boundary"
mcp-zettel show a3f2c9
mcp-zettel backlinks a3f2c9
```

## On-disk layout

```
~/.mcp-zettel/
└── notes/
    ├── a3f2c9.md          ← one markdown file per note
    ├── b7e412.md          ← YAML frontmatter: title, tags, created_at, updated_at
    └── ...                ← body is plain markdown; [[id]] is a wiki-link
```

Every note is a single file. That means: easy backups (git), easy grep, no lock-in. If you ever stop using this server, you still have a directory of markdown files.

## Design choices

- **Files, not a database.** One note per markdown file means you can edit in any editor, back up with git, and inspect without tooling. The store is thin glue on top.
- **Short hex IDs, not slugified titles.** `[[a3f2c9]]` is stable — rename the title and all inbound links still resolve. Also shorter than a filename-based slug.
- **Bidirectional links derived, not stored.** Backlinks are computed on read by scanning every note's body for `[[target_id]]`. No separate index to keep consistent. Trivial at the scale this is designed for (≤ low thousands of notes).
- **Title/tag weighted search.** Title hits count 3×, tag hits 2×, body hits 1×. Matches the intuition that if a note's title mentions "retrieval" it's more about retrieval than a note that says the word once in the middle of the body.
- **FastMCP, not low-level MCP.** The [MCP Python SDK's](https://github.com/modelcontextprotocol/python-sdk) decorator-based FastMCP surface means tools are just Python functions with Pydantic-typed args — no manual JSON Schema authoring.

## Development

```bash
pip install -e ".[dev]"
pytest
black --check src tests
isort --check-only --profile black src tests
flake8 src tests --max-line-length=100 --ignore=E501,W503,E203
```

CI runs on Python 3.10 / 3.11 / 3.12.

Inspect the server interactively with the MCP inspector:

```bash
npx @modelcontextprotocol/inspector mcp-zettel-server
```

## Roadmap

- [x] **v0.2 — embedding-backed semantic search alongside keyword search**
- [ ] v0.3 — `@mcp.prompt()` templates for common note operations ("distill this chat into atomic notes")
- [ ] v0.4 — graph-view resource (`zettel://graph`) returning a mermaid diagram of links
- [ ] v0.5 — remote Streamable HTTP transport for multi-device access

## License

MIT. See [LICENSE](LICENSE).
