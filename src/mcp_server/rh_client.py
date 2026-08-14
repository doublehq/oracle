"""Read-only authenticated Robinhood Trading MCP client for Oracle bulk fetch.

Oracle registers as its own OAuth client (dynamic client registration + PKCE).
Tokens live in ``~/.oracle/robinhood_mcp_tokens.json``. Run ``oracle-rh-login``
once to authenticate; ``rh_session()`` refreshes tokens automatically.

Only the nine read tools required for portfolio snapshots are allowed. Any other
tool name raises ``PermissionError``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Sequence

from mcp import ClientSession, types
from mcp.client.auth import OAuthClientProvider
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client
from mcp.shared.auth import (
    AuthorizationCodeResult,
    OAuthClientInformationFull,
    OAuthClientMetadata,
    OAuthToken,
)

logger = logging.getLogger(__name__)

RH_MCP_URL = "https://agent.robinhood.com/mcp/trading"
DEFAULT_TOKEN_PATH = Path.home() / ".oracle" / "robinhood_mcp_tokens.json"

READ_ONLY_TOOLS = frozenset(
    {
        "get_accounts",
        "get_portfolio",
        "get_equity_positions",
        "get_equity_tax_lots",
        "get_equity_quotes",
        "get_equity_fundamentals",
        "get_pnl_trade_history",
        "get_equity_orders",
        "get_equity_tradability",
    }
)

NOT_AUTHENTICATED_MSG = (
    "Robinhood MCP is not authenticated for Oracle. Run: oracle-rh-login"
)


class RobinhoodMcpError(RuntimeError):
    """Raised when a Robinhood MCP tool call fails."""


class FileTokenStorage:
    """Persist OAuth client info and tokens to a local JSON file."""

    def __init__(self, path: Path | str = DEFAULT_TOKEN_PATH) -> None:
        self.path = Path(path)

    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self, payload: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            os.replace(temp_path, self.path)
        except BaseException:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise

    async def get_tokens(self) -> OAuthToken | None:
        raw = self._load().get("tokens")
        return OAuthToken.model_validate(raw) if raw else None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        data = self._load()
        data["tokens"] = tokens.model_dump(mode="json")
        self._save(data)

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        raw = self._load().get("client_info")
        return OAuthClientInformationFull.model_validate(raw) if raw else None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        data = self._load()
        data["client_info"] = client_info.model_dump(mode="json")
        self._save(data)


def default_client_metadata(redirect_uri: str) -> OAuthClientMetadata:
    return OAuthClientMetadata(
        client_name="Oracle Portfolio Optimizer",
        redirect_uris=[redirect_uri],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        token_endpoint_auth_method="none",
    )


def build_oauth_provider(
    *,
    storage: FileTokenStorage,
    redirect_uri: str | None = None,
    redirect_handler: Callable[[str], Any] | None = None,
    callback_handler: Callable[[], Any] | None = None,
) -> OAuthClientProvider:
    uri = redirect_uri or "http://127.0.0.1:8765/callback"
    return OAuthClientProvider(
        server_url=RH_MCP_URL,
        client_metadata=default_client_metadata(uri),
        storage=storage,
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
    )


def _unwrap_tool_result(result: types.CallToolResult) -> Any:
    # mcp SDK: older camelCase isError vs current snake_case is_error
    is_error = getattr(result, "is_error", None)
    if is_error is None:
        is_error = getattr(result, "isError", False)
    if is_error:
        parts = []
        for block in result.content or []:
            if isinstance(block, types.TextContent):
                parts.append(block.text)
        raise RobinhoodMcpError(" ".join(parts) or "Robinhood MCP tool returned an error")
    if not result.content:
        return {}
    for block in result.content:
        if isinstance(block, types.TextContent):
            try:
                return json.loads(block.text)
            except json.JSONDecodeError:
                return block.text
    return {}


async def call(session: ClientSession, tool: str, arguments: Dict[str, Any] | None = None) -> Any:
    if tool not in READ_ONLY_TOOLS:
        raise PermissionError(
            f"Tool {tool!r} is not allowed. Oracle's Robinhood client is read-only. "
            f"Allowed: {sorted(READ_ONLY_TOOLS)}"
        )
    result = await session.call_tool(tool, arguments or {})
    payload = _unwrap_tool_result(result)
    if isinstance(payload, dict) and "data" in payload:
        return payload
    return {"data": payload}


def _chunks(items: Sequence[str], size: int) -> List[List[str]]:
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


async def batched(
    session: ClientSession,
    tool: str,
    symbols: Sequence[str],
    *,
    batch_size: int,
    arg_key: str = "symbols",
    max_concurrency: int = 4,
    on_batch: Callable[[int, int], None] | None = None,
    cache_get: Callable[[str], Any | None] | None = None,
    cache_put: Callable[[str, Any], None] | None = None,
    deadline: float | None = None,
) -> tuple[List[Any], Dict[str, Any]]:
    """Call ``tool`` in batches with bounded concurrency, optional cache, and deadline."""
    unique = sorted({s.upper() for s in symbols if s})
    batches = _chunks(unique, batch_size)
    progress = {"done": 0, "total": len(batches), "cached": 0, "fetched": 0}
    results: List[Any] = []
    sem = asyncio.Semaphore(max_concurrency)

    async def _one(batch_idx: int, batch: List[str]) -> None:
        cache_key = f"{tool}:{','.join(batch)}"
        if cache_get is not None:
            cached = cache_get(cache_key)
            if cached is not None:
                results.append(cached)
                progress["cached"] += 1
                progress["done"] += 1
                if on_batch:
                    on_batch(progress["done"], progress["total"])
                return
        async with sem:
            if deadline is not None and time.monotonic() >= deadline:
                return
            payload = await call(session, tool, {arg_key: batch})
            if cache_put is not None:
                cache_put(cache_key, payload)
            results.append(payload)
            progress["fetched"] += 1
            progress["done"] += 1
            if on_batch:
                on_batch(progress["done"], progress["total"])

    tasks = [_one(i, batch) for i, batch in enumerate(batches)]
    await asyncio.gather(*tasks)
    # Preserve deterministic order by re-sorting via cache keys isn't needed for merge helpers
    return results, progress


@dataclass
class RhSession:
    session: ClientSession


@asynccontextmanager
async def rh_session(
    *,
    token_path: Path | str = DEFAULT_TOKEN_PATH,
    redirect_uri: str | None = None,
    redirect_handler: Callable[[str], Any] | None = None,
    callback_handler: Callable[[], Any] | None = None,
) -> AsyncIterator[RhSession]:
    """Authenticated Robinhood MCP session.

    For normal (non-interactive) use, omit redirect/callback handlers. If tokens
    are missing or expired beyond refresh, raises with instructions to run
    ``oracle-rh-login``.

    Set ``ORACLE_RH_BEARER`` to use a static bearer token (e.g. copied from an
    authenticated MCP client session).
    """
    bearer = os.environ.get("ORACLE_RH_BEARER", "").strip()
    if bearer:
        import httpx

        class _StaticBearerAuth(httpx.Auth):
            def __init__(self, token: str) -> None:
                self._token = token

            def auth_flow(self, request: httpx.Request):
                request.headers["Authorization"] = f"Bearer {self._token}"
                yield request

        http_client = httpx.AsyncClient(auth=_StaticBearerAuth(bearer), timeout=120.0)
    else:
        storage = FileTokenStorage(token_path)
        provider = build_oauth_provider(
            storage=storage,
            redirect_uri=redirect_uri,
            redirect_handler=redirect_handler,
            callback_handler=callback_handler,
        )
        http_client = create_mcp_http_client(auth=provider)
    try:
        async with streamable_http_client(RH_MCP_URL, http_client=http_client) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield RhSession(session=session)
    except Exception as exc:  # noqa: BLE001 - surface auth failures clearly
        msg = str(exc).lower()
        if any(token in msg for token in ("401", "oauth", "authentication", "authorize", "token")):
            raise RobinhoodMcpError(NOT_AUTHENTICATED_MSG) from exc
        raise


def tokens_present(path: Path | str = DEFAULT_TOKEN_PATH) -> bool:
    storage = FileTokenStorage(path)
    try:
        data = storage._load()
    except json.JSONDecodeError:
        return False
    return bool(data.get("tokens", {}).get("access_token"))
