"""One-time interactive OAuth login for Oracle's Robinhood MCP client."""

from __future__ import annotations

import argparse
import asyncio
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Event, Thread
from typing import Optional
from urllib.parse import parse_qs, urlparse

from mcp.shared.auth import AuthorizationCodeResult

from src.mcp_server.rh_client import (
    DEFAULT_TOKEN_PATH,
    NOT_AUTHENTICATED_MSG,
    RobinhoodMcpError,
    build_oauth_provider,
    call,
    rh_session,
    tokens_present,
)
from src.mcp_server.rh_client import FileTokenStorage


class _CallbackState:
    def __init__(self) -> None:
        self.result: Optional[AuthorizationCodeResult] = None
        self.error: Optional[str] = None
        self.event = Event()


def _make_callback_handler(state: _CallbackState):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/callback":
                self.send_response(404)
                self.end_headers()
                return
            params = parse_qs(parsed.query)
            if "error" in params:
                state.error = params["error"][0]
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Authorization failed. You can close this tab.")
                state.event.set()
                return
            code = params.get("code", [None])[0]
            returned_state = params.get("state", [None])[0]
            iss = params.get("iss", [None])[0]
            if not code:
                state.error = "missing authorization code"
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Missing code.")
                state.event.set()
                return
            state.result = AuthorizationCodeResult(code=code, state=returned_state, iss=iss)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(
                b"<html><body><h2>Oracle Robinhood login successful</h2>"
                b"<p>You can close this tab and return to the terminal.</p></body></html>"
            )
            state.event.set()

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

    return Handler


async def _login_loopback(port: int, token_path: str, open_browser: bool) -> None:
    redirect_uri = f"http://127.0.0.1:{port}/callback"
    state = _CallbackState()

    server = HTTPServer(("127.0.0.1", port), _make_callback_handler(state))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    async def redirect_handler(url: str) -> None:
        print(f"\nOpen this URL to authorize Oracle with Robinhood:\n\n  {url}\n")
        if open_browser:
            webbrowser.open(url)

    async def callback_handler() -> AuthorizationCodeResult:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, state.event.wait, 300)
        server.shutdown()
        if state.error:
            raise RobinhoodMcpError(f"OAuth callback error: {state.error}")
        if state.result is None:
            raise RobinhoodMcpError("Timed out waiting for OAuth callback (300s)")
        return state.result

    storage = FileTokenStorage(token_path)
    provider = build_oauth_provider(
        storage=storage,
        redirect_uri=redirect_uri,
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
    )

    async with rh_session(
        token_path=token_path,
        redirect_uri=redirect_uri,
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
    ) as rh:
        accounts = await call(rh.session, "get_accounts")
        n = len((accounts.get("data") or {}).get("accounts") or [])
        print(f"Login successful. Found {n} brokerage account(s).")


async def _login_manual_paste(port: int, token_path: str) -> None:
    """Fallback when loopback redirect URI is rejected by the authorization server."""
    redirect_uri = f"http://127.0.0.1:{port}/callback"
    storage = FileTokenStorage(token_path)

    async def redirect_handler(url: str) -> None:
        print("\nLoopback redirect may not be supported. Open this URL manually:\n")
        print(f"  {url}\n")
        print("After approving, paste the FULL redirect URL (or just the ?code=...&state=... query):\n")

    captured: dict[str, AuthorizationCodeResult | None] = {"result": None}

    async def callback_handler() -> AuthorizationCodeResult:
        if captured["result"] is not None:
            return captured["result"]
        raw = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
        text = raw.strip()
        if "code=" not in text:
            raise RobinhoodMcpError("Paste must include code=...")
        if not text.startswith("http"):
            text = redirect_uri + ("&" if "?" in text else "?") + text.lstrip("?")
        parsed = urlparse(text)
        params = parse_qs(parsed.query)
        code = params.get("code", [None])[0]
        state_val = params.get("state", [None])[0]
        iss = params.get("iss", [None])[0]
        if not code:
            raise RobinhoodMcpError("Could not parse authorization code from pasted URL")
        captured["result"] = AuthorizationCodeResult(code=code, state=state_val, iss=iss)
        return captured["result"]

    async with rh_session(
        token_path=token_path,
        redirect_uri=redirect_uri,
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
    ) as rh:
        await call(rh.session, "get_accounts")
        print("Login successful (manual paste).")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Authenticate Oracle with the Robinhood Trading MCP.")
    parser.add_argument("--token-path", default=str(DEFAULT_TOKEN_PATH))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--manual", action="store_true", help="Use manual URL paste instead of loopback callback.")
    parser.add_argument("--check", action="store_true", help="Exit 0 if tokens exist, 1 otherwise.")
    args = parser.parse_args(argv)

    if args.check:
        raise SystemExit(0 if tokens_present(args.token_path) else 1)

    if tokens_present(args.token_path) and not args.manual:
        print(f"Tokens already present at {args.token_path}. Re-authenticating will replace them.")

    try:
        if args.manual:
            asyncio.run(_login_manual_paste(args.port, args.token_path))
        else:
            asyncio.run(_login_loopback(args.port, args.token_path, open_browser=not args.no_browser))
    except RobinhoodMcpError as exc:
        print(f"Login failed: {exc}", file=sys.stderr)
        if "redirect" in str(exc).lower() or "registration" in str(exc).lower():
            print("\nTry again with: oracle-rh-login --manual", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
