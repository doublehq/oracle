"""Fetch a Robinhood account + SPY index snapshot for Oracle direct indexing.

Bulk-fetches account state, SPY targets, quotes, fundamentals, and wash-sale
history via Oracle's read-only Robinhood MCP client. Results are cached per
batch and written to a single artifact JSON file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.mcp_server.factors import build_factor_model_from_fundamentals
from src.mcp_server.index_targets import fetch_etf_holdings_targets
from src.mcp_server.rh_client import READ_ONLY_TOOLS, RobinhoodMcpError, batched, call, rh_session
from src.mcp_server.robinhood import flatten_robinhood_tax_lots
from src.mcp_server.wash_sale_history import build_wash_sale_history

logger = logging.getLogger(__name__)

DEFAULT_OUT_PATH = "artifacts/rh_snapshot.json"
DEFAULT_CACHE_DIR = "artifacts/.rh_cache"

QUOTE_BATCH_SIZE = 20
FUNDAMENTALS_BATCH_SIZE = 10


def _write_private_json(path: Path, payload: Any, *, indent: Optional[int] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=indent)
        os.replace(temp_path, path)
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


def _cache_path(cache_dir: Path, key: str) -> Path:
    digest = hashlib.sha256(key.encode()).hexdigest()[:32]
    return cache_dir / f"{digest}.json"


class BatchCache:
    def __init__(
        self,
        cache_dir: Path | str,
        *,
        max_age_seconds: float = 300.0,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.max_age_seconds = max_age_seconds
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> Any | None:
        path = _cache_path(self.cache_dir, key)
        if not path.exists():
            return None
        if time.time() - path.stat().st_mtime > self.max_age_seconds:
            path.unlink(missing_ok=True)
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def put(self, key: str, value: Any) -> None:
        path = _cache_path(self.cache_dir, key)
        _write_private_json(path, value)

    def clear(self) -> None:
        for path in self.cache_dir.glob("*.json"):
            path.unlink(missing_ok=True)


def _select_agentic_account(accounts_payload: Dict[str, Any]) -> Tuple[str, str]:
    accounts = (accounts_payload.get("data") or {}).get("accounts") or []
    agentic = [a for a in accounts if a.get("agentic_allowed")]
    if not agentic:
        raise RobinhoodMcpError("No agentic_allowed=true account found in get_accounts response.")
    acct = agentic[0]
    account_number = str(acct["account_number"])
    rhs_account_number = str(acct.get("rhs_account_number") or account_number)
    return account_number, rhs_account_number


def _parse_cash(portfolio_payload: Dict[str, Any]) -> float:
    data = portfolio_payload.get("data") or {}
    buying_power = (data.get("buying_power") or {}).get("buying_power")
    if buying_power is not None:
        return float(buying_power)
    cash = data.get("cash")
    if cash is not None:
        return float(cash)
    return 0.0


def _symbols_from_targets(targets: Sequence[Dict[str, Any]]) -> List[str]:
    symbols: List[str] = []
    for row in targets:
        sym = row.get("symbol") or (row.get("identifiers") or [None])[0]
        if sym and str(sym).upper() != "CASH":
            symbols.append(str(sym).upper())
    return sorted(set(symbols))


def _quotes_to_prices(quote_batches: Sequence[Any]) -> List[Dict[str, Any]]:
    prices: Dict[str, float] = {}
    for batch in quote_batches:
        data = (batch or {}).get("data") or batch or {}
        for row in data.get("results") or []:
            if not isinstance(row, dict):
                continue
            # Robinhood returns {quote: {symbol, last_trade_price, ...}, close: {...}}
            quote = row.get("quote") if isinstance(row.get("quote"), dict) else row
            symbol = str(quote.get("symbol") or row.get("symbol") or "").upper()
            if not symbol:
                continue
            price = quote.get("last_trade_price") or quote.get("price") or row.get("price")
            if price is None:
                bid, ask = quote.get("bid_price"), quote.get("ask_price")
                if bid is not None and ask is not None:
                    price = (float(bid) + float(ask)) / 2
            if price is None:
                close = row.get("close") if isinstance(row.get("close"), dict) else {}
                price = close.get("price")
            if price is not None:
                prices[symbol] = float(price)
    return [{"symbol": sym, "price": px} for sym, px in sorted(prices.items())]


async def _merge_paginated_pnl(
    session,
    *,
    rhs_account_number: str,
    cache: BatchCache,
    deadline: float | None,
) -> Tuple[List[Any], Dict[str, Any]]:
    pages: List[Any] = []
    cursor: Optional[str] = None
    meta = {"pages": 0, "has_unread_next": False}
    while True:
        if deadline is not None and time.monotonic() >= deadline:
            meta["has_unread_next"] = True
            break
        key = f"get_pnl_trade_history:{rhs_account_number}:{cursor or 'start'}"
        cached = cache.get(key)
        if cached is not None:
            payload = cached
        else:
            args: Dict[str, Any] = {
                "account_number": rhs_account_number,
                "span": "month",
            }
            if cursor:
                args["cursor"] = cursor
            payload = await call(session, "get_pnl_trade_history", args)
            cache.put(key, payload)
        pages.append(payload)
        meta["pages"] += 1
        data = (payload.get("data") or {})
        cursor = data.get("next_cursor")
        if not cursor:
            break
    return pages, meta


async def _merge_paginated_orders(
    session,
    *,
    account_number: str,
    cache: BatchCache,
    deadline: float | None,
    window_days: int = 31,
) -> Tuple[List[Any], Dict[str, Any]]:
    pages: List[Any] = []
    cursor: Optional[str] = None
    meta = {"pages": 0, "has_unread_next": False}
    created_at_gte = (date.today() - timedelta(days=window_days)).isoformat()
    while True:
        if deadline is not None and time.monotonic() >= deadline:
            meta["has_unread_next"] = True
            break
        key = f"get_equity_orders:{account_number}:{cursor or 'start'}"
        cached = cache.get(key)
        if cached is not None:
            payload = cached
        else:
            args: Dict[str, Any] = {
                "account_number": account_number,
                "state": "filled",
                "created_at_gte": created_at_gte,
            }
            if cursor:
                args["cursor"] = cursor
            payload = await call(session, "get_equity_orders", args)
            cache.put(key, payload)
        pages.append(payload)
        meta["pages"] += 1
        data = (payload.get("data") or {})
        next_url = data.get("next")
        if not next_url:
            break
        from src.mcp_server.wash_sale_history import extract_next_cursor

        cursor = extract_next_cursor(next_url)
        if not cursor:
            break
    return pages, meta


async def fetch_robinhood_snapshot_async(
    *,
    account_number: Optional[str] = None,
    out_path: str | Path = DEFAULT_OUT_PATH,
    top_n: Optional[int] = None,
    include_fundamentals: bool = True,
    include_wash_sale_history: bool = True,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    max_seconds: Optional[float] = None,
    progress_callback: Callable[[Dict[str, Any]], None] | None = None,
) -> Dict[str, Any]:
    """Fetch account + SPY snapshot. Resumable via per-batch cache."""
    out = Path(out_path)
    cache = BatchCache(cache_dir)
    warnings: List[str] = []
    started = time.monotonic()
    deadline = (started + max_seconds) if max_seconds is not None else None

    def _progress(stage: str, **extra: Any) -> None:
        payload = {"stage": stage, **extra}
        if progress_callback:
            progress_callback(payload)
        logger.info("snapshot %s", payload)

    def _timed_out() -> bool:
        return deadline is not None and time.monotonic() >= deadline

    partial_state: Dict[str, Any] = {}
    if out.exists():
        try:
            partial_state = json.loads(out.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            partial_state = {}

    async with rh_session() as rh:
        session = rh.session

        _progress("accounts")
        accounts_payload = await call(session, "get_accounts")
        if not account_number:
            account_number, rhs_account_number = _select_agentic_account(accounts_payload)
        else:
            rhs_account_number = account_number
            for acct in (accounts_payload.get("data") or {}).get("accounts") or []:
                if str(acct.get("account_number")) == str(account_number):
                    rhs_account_number = str(acct.get("rhs_account_number") or account_number)
                    break

        _progress("portfolio", account_number=account_number)
        portfolio_payload = await call(session, "get_portfolio", {"account_number": account_number})
        cash = _parse_cash(portfolio_payload)

        _progress("positions")
        positions_payload = await call(session, "get_equity_positions", {"account_number": account_number})
        held_symbols = [
            str(p.get("symbol", "")).upper()
            for p in (positions_payload.get("data") or {}).get("positions") or []
            if p.get("symbol")
        ]

        tax_lot_responses: List[Any] = []
        for symbol in held_symbols:
            if _timed_out():
                break
            key = f"get_equity_tax_lots:{account_number}:{symbol}"
            cached = cache.get(key)
            if cached is not None:
                tax_lot_responses.append(cached)
            else:
                lot_payload = await call(
                    session,
                    "get_equity_tax_lots",
                    {"account_number": account_number, "symbol": symbol},
                )
                cache.put(key, lot_payload)
                tax_lot_responses.append(lot_payload)
        tax_lots = flatten_robinhood_tax_lots(tax_lot_responses) if tax_lot_responses else []

        if _timed_out():
            return _partial_result(
                out, partial_state, account_number, rhs_account_number, cash, tax_lots,
                warnings, started, "positions/tax_lots",
            )

        _progress("spy_targets")
        spy = fetch_etf_holdings_targets(top_n=top_n)
        targets = spy.get("targets") or []
        target_symbols = _symbols_from_targets(targets)
        all_symbols = sorted(set(target_symbols) | set(held_symbols))

        quote_batches, quote_progress = await batched(
            session,
            "get_equity_quotes",
            all_symbols,
            batch_size=QUOTE_BATCH_SIZE,
            cache_get=cache.get,
            cache_put=cache.put,
            deadline=deadline,
        )
        prices = _quotes_to_prices(quote_batches)

        if _timed_out() or quote_progress["done"] < quote_progress["total"]:
            return _partial_result(
                out, partial_state, account_number, rhs_account_number, cash, tax_lots,
                warnings, started, "quotes",
                extra={"targets": targets, "prices": prices, "progress": quote_progress},
            )

        fundamentals_batches: List[Any] = []
        fund_progress = {"done": 0, "total": 0}
        if include_fundamentals:
            fundamentals_batches, fund_progress = await batched(
                session,
                "get_equity_fundamentals",
                target_symbols,
                batch_size=FUNDAMENTALS_BATCH_SIZE,
                cache_get=cache.get,
                cache_put=cache.put,
                deadline=deadline,
            )

        if include_fundamentals and (
            _timed_out() or fund_progress["done"] < fund_progress["total"]
        ):
            return _partial_result(
                out, partial_state, account_number, rhs_account_number, cash, tax_lots,
                warnings, started, "fundamentals",
                extra={
                    "targets": targets,
                    "prices": prices,
                    "fundamentals_batches": len(fundamentals_batches),
                    "progress": fund_progress,
                },
            )

        pnl_pages: List[Any] = []
        order_pages: List[Any] = []
        wash_meta: Dict[str, Any] = {}
        recently_closed_lots: List[Dict[str, Any]] = []
        if include_wash_sale_history:
            pnl_pages, pnl_meta = await _merge_paginated_pnl(
                session,
                rhs_account_number=rhs_account_number,
                cache=cache,
                deadline=deadline,
            )
            if _timed_out() or pnl_meta.get("has_unread_next"):
                return _partial_result(
                    out,
                    partial_state,
                    account_number,
                    rhs_account_number,
                    cash,
                    tax_lots,
                    warnings,
                    started,
                    "wash_sale_history",
                    extra={"targets": targets, "prices": prices},
                )
            order_pages, order_meta = await _merge_paginated_orders(
                session,
                account_number=account_number,
                cache=cache,
                deadline=deadline,
            )
            if _timed_out() or order_meta.get("has_unread_next"):
                return _partial_result(
                    out,
                    partial_state,
                    account_number,
                    rhs_account_number,
                    cash,
                    tax_lots,
                    warnings,
                    started,
                    "wash_sale_history",
                    extra={"targets": targets, "prices": prices},
                )
            wash = build_wash_sale_history(
                pnl_trade_history=pnl_pages,
                equity_orders=order_pages,
                tax_lots=tax_lot_responses,
            )
            recently_closed_lots = wash.get("recently_closed_lots") or []
            wash_meta = {
                "coverage": wash.get("coverage"),
                "warnings": wash.get("warnings") or [],
            }
            warnings.extend(wash.get("warnings") or [])

        factor_model: List[Dict[str, Any]] = []
        factor_diag: Dict[str, Any] = {}
        if include_fundamentals and fundamentals_batches:
            built = build_factor_model_from_fundamentals(fundamentals_batches, prices=prices)
            factor_model = built.get("factor_model") or []
            factor_diag = {
                "coverage": built.get("coverage"),
                "sectors": built.get("sectors"),
                "diagnostics": built.get("diagnostics"),
            }

        missing_prices = [s for s in target_symbols if s not in {p["symbol"] for p in prices}]
        if missing_prices:
            warnings.append(f"Missing prices for {len(missing_prices)} target symbols.")

        artifact = {
            "status": "complete",
            "account_number": account_number,
            "rhs_account_number": rhs_account_number,
            "cash": cash,
            "tax_lots": tax_lots,
            "prices": prices,
            "targets": targets,
            "factor_model": factor_model,
            "recently_closed_lots": recently_closed_lots,
            "summary": {
                "held_symbols": len(held_symbols),
                "target_symbols": len(target_symbols),
                "tax_lots": len(tax_lots),
                "prices": len(prices),
                "factor_model_rows": len(factor_model),
                "recently_closed_lots": len(recently_closed_lots),
                "spy_holdings_as_of": spy.get("holdings_as_of"),
                "factor_diagnostics": factor_diag,
                "wash_sale": wash_meta,
                "elapsed_seconds": round(time.monotonic() - started, 2),
            },
            "warnings": warnings,
        }
        _write_private_json(out, artifact, indent=2)
        cache.clear()

        return {
            "status": "complete",
            "out_path": str(out),
            "summary": artifact["summary"],
            "warnings": warnings,
        }


def _partial_result(
    out: Path,
    prior: Dict[str, Any],
    account_number: str,
    rhs_account_number: str,
    cash: float,
    tax_lots: List[Dict[str, Any]],
    warnings: List[str],
    started: float,
    stage: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    extra = extra or {}
    progress = extra.get("progress") or {}
    artifact = {
        **prior,
        "status": "partial",
        "account_number": account_number,
        "rhs_account_number": rhs_account_number,
        "cash": cash,
        "tax_lots": tax_lots,
        "targets": extra.get("targets") or prior.get("targets") or [],
        "prices": extra.get("prices") or prior.get("prices") or [],
        "factor_model": prior.get("factor_model") or [],
        "recently_closed_lots": prior.get("recently_closed_lots") or [],
        "warnings": list(set(warnings + (prior.get("warnings") or []))),
    }
    _write_private_json(out, artifact, indent=2)
    return {
        "status": "partial",
        "out_path": str(out),
        "stage": stage,
        "progress": progress,
        "summary": {
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "stage": stage,
        },
        "warnings": artifact["warnings"],
    }


def load_snapshot(path: str | Path) -> Dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("status") == "partial":
        raise ValueError(
            f"Snapshot {path} is partial; resume fetch_robinhood_snapshot before optimizing"
        )
    required = ("cash", "prices", "targets")
    for key in required:
        if key not in data:
            raise ValueError(f"Snapshot {path} missing required key: {key}")
    return data


def fetch_robinhood_snapshot(**kwargs: Any) -> Dict[str, Any]:
    """Sync wrapper for MCP tool and CLI."""
    import asyncio

    return asyncio.run(fetch_robinhood_snapshot_async(**kwargs))


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Fetch Robinhood account + SPY snapshot for Oracle.")
    parser.add_argument("--account-number", default=None)
    parser.add_argument("--out", default=DEFAULT_OUT_PATH)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--no-fundamentals", action="store_true")
    parser.add_argument("--no-wash-sale", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    result = fetch_robinhood_snapshot(
        account_number=args.account_number,
        out_path=args.out,
        top_n=args.top_n,
        include_fundamentals=not args.no_fundamentals,
        include_wash_sale_history=not args.no_wash_sale,
        cache_dir=args.cache_dir,
        max_seconds=None,
    )
    print(json.dumps(result, indent=2))
    if result.get("status") != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
