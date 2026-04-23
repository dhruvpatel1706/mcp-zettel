"""Prompt templates registered as MCP `@prompt` endpoints.

MCP prompts are little instruction payloads clients can pick from a menu.
They don't execute tools themselves — they return a string that the LLM
then acts on using whatever tools are available in the session. The useful
thing about keeping them server-side is that the wording stays consistent
across Claude Desktop / Cursor / Claude Code / whatever client the user is on.
"""

from __future__ import annotations

from datetime import date


def distill_conversation(conversation: str, max_notes: int = 8) -> str:
    """Turn a messy chat transcript into suggested atomic notes.

    Args:
        conversation: Paste the chat (yours, the assistant's, both) you want to distill.
        max_notes: Soft upper bound on how many notes to suggest.
    """
    return (
        "You are given the transcript of a conversation. Extract up to "
        f"{max_notes} discrete, atomic insights worth preserving as zettelkasten notes. "
        "For each insight:\n"
        "  1. Pick a short, specific title (the kind that would fit in a search result).\n"
        "  2. Draft a 2-5 sentence body that captures the insight in your own words.\n"
        "  3. Suggest 2-4 lowercase tags.\n\n"
        "Avoid overlap between notes — each one should be the smallest unit of "
        "useful knowledge. Skip chit-chat and anything tied to the specific day.\n\n"
        "When I approve a draft, create it using `create_note`. If a new note "
        "logically links to a previous one in this batch, add a `[[...]]` "
        "link in the body.\n\n"
        "--- CONVERSATION ---\n"
        f"{conversation}\n"
        "--- END ---"
    )


def find_linkable_notes(concept: str, limit: int = 8) -> str:
    """Surface existing notes that might want a link to/from a new concept.

    Args:
        concept: A phrase or sentence naming the concept you're writing about.
        limit: How many candidates to consider.
    """
    return (
        f"The user is writing a note about: **{concept}**.\n\n"
        f"Your job is to find up to {limit} existing notes that might benefit "
        "from a bidirectional link. Use `search_notes_semantic` with the "
        "concept as the query (it handles paraphrases better than keyword "
        "search for this). For each candidate:\n"
        "  - Read the note with `read_note`.\n"
        "  - Decide if there's a genuine conceptual relationship (supports, "
        "contradicts, generalizes, is an instance of, etc.) — not just a "
        "shared word.\n"
        "  - If yes, propose the link and its direction. Wait for my go-ahead "
        "before calling `link_notes`."
    )


def daily_note(prompt_date: str | None = None) -> str:
    """Return a template for today's (or a chosen date's) daily journal note.

    Args:
        prompt_date: Optional ISO date (YYYY-MM-DD). Defaults to today in the server's timezone.
    """
    d = prompt_date or date.today().isoformat()
    return (
        f"Create a daily journal note for **{d}**. Use this template for the body:\n\n"
        "```\n"
        "## What I worked on\n"
        "- \n\n"
        "## What I learned\n"
        "- \n\n"
        "## What's blocking me / open questions\n"
        "- \n\n"
        "## Notes created today\n"
        "(list [[ids]] of any new atomic notes I dropped in)\n"
        "```\n\n"
        f"Title it `Daily — {d}`. Tag it with `daily` and `journal`. "
        "Ask me to dictate each bullet — don't invent content."
    )


def summarize_by_tag(tag: str, style: str = "bullets") -> str:
    """Summarize everything the user has written under a tag.

    Args:
        tag: The tag to pull.
        style: 'bullets' (short), 'essay' (flowing paragraphs), or 'outline' (nested).
    """
    style_direction = {
        "bullets": "a bulleted digest of 5-12 items, each a one-liner pointing to the source note id",
        "essay": "2-4 flowing paragraphs that cite source notes inline like [[a1b2c3]]",
        "outline": "a nested outline grouping related notes under sub-headings",
    }.get(style, "a bulleted digest")
    return (
        f"Pull every note tagged with `{tag}` using "
        f'`list_notes(tags=["{tag}"])`, read each, and produce {style_direction}. '
        "Don't summarize a note in a way that loses its key nuance — if two "
        "notes disagree, surface the disagreement. Flag any note that feels "
        "out of place under this tag so the user can re-tag it."
    )
