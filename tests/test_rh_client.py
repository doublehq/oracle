"""Tests for Oracle's read-only Robinhood MCP client."""

import pytest

from src.mcp_server.rh_client import READ_ONLY_TOOLS


def test_read_only_allowlist_excludes_write_tools():
    assert "place_equity_order" not in READ_ONLY_TOOLS
    assert "review_equity_order" not in READ_ONLY_TOOLS
    assert "get_accounts" in READ_ONLY_TOOLS
    assert "get_equity_fundamentals" in READ_ONLY_TOOLS


@pytest.mark.asyncio
async def test_call_rejects_write_tools():
    from src.mcp_server.rh_client import call

    class FakeSession:
        pass

    with pytest.raises(PermissionError, match="read-only"):
        await call(FakeSession(), "place_equity_order", {})
