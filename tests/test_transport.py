"""Transport selection logic in server.main()."""

from __future__ import annotations

import os

import pytest


def test_invalid_transport_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_ZETTEL_ROOT", str(tmp_path))
    from mcp_zettel import server

    with pytest.raises(SystemExit, match="Unknown transport"):
        server.main("gopher")


def test_bad_port_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_ZETTEL_ROOT", str(tmp_path))
    monkeypatch.setenv("MCP_ZETTEL_PORT", "not-a-number")
    from mcp_zettel import server

    called = {}

    def _fake_run(*args, **kwargs):
        called["kwargs"] = kwargs

    monkeypatch.setattr("mcp.server.fastmcp.FastMCP.run", lambda self, *a, **k: _fake_run(*a, **k))
    with pytest.raises(SystemExit, match="MCP_ZETTEL_PORT must be an integer"):
        server.main("streamable-http")


def test_stdio_is_default(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_ZETTEL_ROOT", str(tmp_path))
    monkeypatch.delenv("MCP_ZETTEL_TRANSPORT", raising=False)
    from mcp_zettel import server

    captured = {}

    def _fake_run(self, *args, **kwargs):
        captured["transport"] = kwargs.get("transport", "stdio")

    monkeypatch.setattr("mcp.server.fastmcp.FastMCP.run", _fake_run)
    server.main()
    assert captured["transport"] == "stdio"


def test_env_var_overrides_default(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_ZETTEL_ROOT", str(tmp_path))
    monkeypatch.setenv("MCP_ZETTEL_TRANSPORT", "streamable-http")
    monkeypatch.setenv("MCP_ZETTEL_HOST", "0.0.0.0")
    monkeypatch.setenv("MCP_ZETTEL_PORT", "9001")

    from mcp_zettel import server

    captured = {}

    def _fake_run(self, *args, **kwargs):
        captured["transport"] = kwargs.get("transport")
        captured["host"] = self.settings.host
        captured["port"] = self.settings.port

    monkeypatch.setattr("mcp.server.fastmcp.FastMCP.run", _fake_run)
    server.main()
    assert captured["transport"] == "streamable-http"
    assert captured["host"] == "0.0.0.0"
    assert captured["port"] == 9001


def test_explicit_arg_beats_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_ZETTEL_ROOT", str(tmp_path))
    monkeypatch.setenv("MCP_ZETTEL_TRANSPORT", "streamable-http")  # should be ignored

    from mcp_zettel import server

    captured = {}

    def _fake_run(self, *args, **kwargs):
        # stdio path calls run() with no args
        captured["transport"] = kwargs.get("transport", "stdio")

    monkeypatch.setattr("mcp.server.fastmcp.FastMCP.run", _fake_run)
    server.main("stdio")
    assert captured["transport"] == "stdio"
