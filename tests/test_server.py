"""Smoke tests for the FastMCP server: build it, inspect registered tools/resources."""

from __future__ import annotations

from mcp_zettel.server import build_server


def test_server_builds(tmp_path):
    server = build_server(tmp_path)
    assert server.name == "zettel"


async def test_tools_registered(tmp_path):
    server = build_server(tmp_path)
    tools = await server.list_tools()
    names = {t.name for t in tools}
    expected = {
        "create_note",
        "read_note",
        "update_note",
        "delete_note",
        "list_notes",
        "search_notes",
        "link_notes",
        "get_backlinks",
        "linked_notes",
    }
    assert expected.issubset(names)


async def test_resources_registered(tmp_path):
    server = build_server(tmp_path)
    # Static resources (no templating)
    resources = await server.list_resources()
    uris = {str(r.uri) for r in resources}
    assert "zettel://all" in uris
    # Templated resources (e.g. zettel://{note_id})
    templates = await server.list_resource_templates()
    template_uris = {t.uriTemplate for t in templates}
    assert "zettel://{note_id}" in template_uris


async def test_prompts_registered(tmp_path):
    server = build_server(tmp_path)
    prompts = await server.list_prompts()
    names = {p.name for p in prompts}
    assert {
        "distill_conversation",
        "find_linkable_notes",
        "daily_note",
        "summarize_by_tag",
    }.issubset(names)
