"""Build Oracle ``recently_closed_lots`` from Robinhood order and PnL payloads.

The orchestrating agent fetches raw JSON from Robinhood's ``get_pnl_trade_history``
(primary; uses ``rhs_account_number``) and ``get_equity_orders`` (fallback /
cross-check; uses ``account_number``), then passes those responses here unchanged.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urlparse

from src.mcp_server.robinhood import flatten_robinhood_tax_lots

_FILLED_STATES = frozenset({"filled", "partially_filled", "partially_filled_rest_cancelled"})
_REALIZED_GAIN_KEYS = (
    "realized_gain",
    "realized_pnl",
    "realized_gain_loss",
    "realized_gain_loss_amount",
    "pnl",
    "realized_profit_loss",
)
_EQUITY_ALIASES = frozenset({"equity", "stock", "stocks", "etf", "etp"})
_NON_EQUITY_ALIASES = frozenset(
    {"option", "options", "crypto", "cryptocurrency", "prediction", "prediction_market"}
)
_ASSUMED_LOSS = 0.01


def extract_next_cursor(next_url: Optional[str]) -> Optional[str]:
    """Pull the ``cursor`` query param out of a Robinhood orders ``next`` URL."""
    if not next_url:
        return None
    parsed = urlparse(str(next_url))
    values = parse_qs(parsed.query).get("cursor")
    return values[0] if values else None


def _norm_symbol(raw: Dict[str, Any]) -> str:
    symbol = raw.get("symbol") or raw.get("identifier") or raw.get("ticker")
    if not symbol:
        raise ValueError(f"Missing symbol/identifier in record: {raw}")
    return str(symbol).upper()


def _parse_date(value: Any) -> Optional[str]:
    if value is None or value == "":
        return None
    text = str(value)
    if "T" in text:
        return text[:10]
    return text[:10] if len(text) >= 10 else text


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _iter_order_pages(responses: Optional[Sequence[Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Normalize one or many ``get_equity_orders`` payloads into a flat order list."""
    orders: List[Dict[str, Any]] = []
    meta = {
        "pages": 0,
        "orders_seen": 0,
        "orders_kept": 0,
        "has_unread_next": False,
        "next_cursor": None,
    }
    if not responses:
        return orders, meta

    for response in responses:
        if isinstance(response, list):
            page_orders = response
            next_url = None
        else:
            data = (response or {}).get("data") or response or {}
            page_orders = data.get("orders") or []
            next_url = data.get("next")
        meta["pages"] += 1
        meta["orders_seen"] += len(page_orders)
        if next_url:
            meta["has_unread_next"] = True
            meta["next_cursor"] = extract_next_cursor(next_url)
        for order in page_orders:
            if not isinstance(order, dict):
                continue
            side = str(order.get("side") or "").lower()
            state = str(order.get("state") or "").lower()
            if side != "sell" or state not in _FILLED_STATES:
                continue
            orders.append(order)
            meta["orders_kept"] += 1
    return orders, meta


def flatten_equity_orders(
    responses: Optional[Sequence[Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[str]]:
    """Flatten ``get_equity_orders`` payloads into normalized sell rows."""
    warnings: List[str] = []
    raw_orders, meta = _iter_order_pages(responses)
    rows: List[Dict[str, Any]] = []

    for order in raw_orders:
        symbol = _norm_symbol(order)
        executions = order.get("executions") or []
        if executions:
            quantity = sum(_to_float(ex.get("quantity")) for ex in executions)
            proceeds = sum(
                _to_float(ex.get("price")) * _to_float(ex.get("quantity")) for ex in executions
            )
            timestamps = [ex.get("timestamp") for ex in executions if ex.get("timestamp")]
            date_sold = _parse_date(max(timestamps)) if timestamps else None
            fees = sum(_to_float(ex.get("fees")) for ex in executions)
        else:
            quantity = _to_float(order.get("cumulative_quantity") or order.get("quantity"))
            avg_price = order.get("average_price") or order.get("price")
            if avg_price is None and order.get("dollar_based_amount"):
                warnings.append(
                    f"{symbol}: skipped dollar_based sell with no executions "
                    f"(order {order.get('id', '?')})."
                )
                continue
            proceeds = _to_float(avg_price) * quantity
            date_sold = _parse_date(order.get("last_transaction_at") or order.get("created_at"))
            fees = _to_float(order.get("fees"))

        if quantity <= 0:
            continue
        if not date_sold:
            date_sold = _parse_date(order.get("last_transaction_at") or order.get("created_at"))
        if not date_sold:
            warnings.append(f"{symbol}: sell order missing sale date (order {order.get('id', '?')}).")
            continue

        rows.append(
            {
                "symbol": symbol,
                "identifier": symbol,
                "quantity": quantity,
                "proceeds": proceeds,
                "date_sold": date_sold,
                "realized_gain": None,
                "date_acquired": None,
                "source": "equity_orders",
                "order_id": order.get("id"),
                "fees": fees,
                "state": order.get("state"),
            }
        )
    return rows, meta, warnings


def _instrument_class(row: Dict[str, Any]) -> Optional[str]:
    for key in ("instrument_type", "asset_class", "asset_type", "security_type", "type"):
        value = row.get(key)
        if value is not None:
            return str(value).lower()
    return None


def _is_equity_row(row: Dict[str, Any]) -> bool:
    side = str(row.get("side") or "").lower()
    if side and side not in {"sell", "short"}:
        return False
    instrument = _instrument_class(row)
    if instrument in _NON_EQUITY_ALIASES:
        return False
    if instrument in _EQUITY_ALIASES:
        return True
    # Unknown instrument class: keep if it looks like an equity ticker row.
    return bool(row.get("symbol") or row.get("identifier"))


def _lookup_realized_gain(row: Dict[str, Any]) -> Tuple[Optional[float], Optional[str]]:
    for key in _REALIZED_GAIN_KEYS:
        if key not in row:
            continue
        value = row.get(key)
        if value is None:
            return None, key
        return _to_float(value), key
    return None, None


def _iter_pnl_pages(responses: Optional[Sequence[Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    trades: List[Dict[str, Any]] = []
    meta = {
        "pages": 0,
        "trades_seen": 0,
        "trades_kept": 0,
        "has_unread_next": False,
        "next_cursor": None,
    }
    if not responses:
        return trades, meta

    for response in responses:
        if isinstance(response, list):
            page_trades = response
            next_cursor = None
        else:
            data = (response or {}).get("data") or response or {}
            page_trades = data.get("trades") or []
            next_cursor = data.get("next_cursor")
        meta["pages"] += 1
        meta["trades_seen"] += len(page_trades)
        if next_cursor:
            meta["has_unread_next"] = True
            meta["next_cursor"] = str(next_cursor)
        for trade in page_trades:
            if isinstance(trade, dict) and _is_equity_row(trade):
                trades.append(trade)
                meta["trades_kept"] += 1
    return trades, meta


def flatten_pnl_trade_history(
    responses: Optional[Sequence[Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[str]]:
    """Flatten ``get_pnl_trade_history`` payloads into normalized sell rows."""
    warnings: List[str] = []
    raw_trades, meta = _iter_pnl_pages(responses)
    rows: List[Dict[str, Any]] = []

    for trade in raw_trades:
        try:
            symbol = _norm_symbol(trade)
        except ValueError:
            warnings.append(f"Skipped PnL row missing symbol: {trade}")
            continue

        quantity = _to_float(trade.get("quantity"))
        if quantity <= 0:
            continue

        price = trade.get("price") or trade.get("execution_price") or trade.get("average_price")
        proceeds = trade.get("proceeds")
        if proceeds is None:
            if price is None:
                warnings.append(f"{symbol}: PnL row missing price/proceeds: {trade}")
                continue
            proceeds = _to_float(price) * quantity
        else:
            proceeds = _to_float(proceeds)

        date_sold = _parse_date(
            trade.get("date_sold")
            or trade.get("closed_at")
            or trade.get("executed_at")
            or trade.get("timestamp")
            or trade.get("trade_date")
        )
        if not date_sold:
            warnings.append(f"{symbol}: PnL row missing sale date: {trade}")
            continue

        realized_gain, gain_key = _lookup_realized_gain(trade)
        if gain_key and realized_gain is None:
            warnings.append(
                f"{symbol}: PnL row has null {gain_key} on {date_sold}; "
                "treating as unknown basis."
            )

        date_acquired = _parse_date(
            trade.get("date_acquired")
            or trade.get("open_date")
            or trade.get("purchase_date")
            or trade.get("acquired_at")
        )
        cost_basis = trade.get("cost_basis") or trade.get("tax_cost_basis")
        if cost_basis is not None:
            cost_basis = _to_float(cost_basis)
        elif realized_gain is not None:
            cost_basis = proceeds - realized_gain

        rows.append(
            {
                "symbol": symbol,
                "identifier": symbol,
                "quantity": quantity,
                "proceeds": proceeds,
                "date_sold": date_sold,
                "realized_gain": realized_gain,
                "date_acquired": date_acquired,
                "cost_basis": cost_basis,
                "source": "pnl_trade_history",
                "pnl_gain_field": gain_key,
            }
        )
    return rows, meta, warnings


def _dedupe_key(row: Dict[str, Any]) -> Tuple[str, str, float]:
    return (
        str(row["symbol"]).upper(),
        str(row["date_sold"]),
        round(float(row["quantity"]), 6),
    )


def _merge_sale_rows(
    order_rows: Sequence[Dict[str, Any]],
    pnl_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged: Dict[Tuple[str, str, float], Dict[str, Any]] = {}
    for row in order_rows:
        merged[_dedupe_key(row)] = dict(row)
    for row in pnl_rows:
        key = _dedupe_key(row)
        existing = merged.get(key)
        if existing:
            combined = dict(existing)
            for field, value in row.items():
                if value is not None:
                    combined[field] = value
            combined["source"] = "pnl_trade_history"
            merged[key] = combined
        else:
            merged[key] = dict(row)
    return list(merged.values())


def _lookup_cost_basis_hint(
    hints: Optional[Dict[str, Any]],
    symbol: str,
    date_sold: str,
) -> Optional[float]:
    if not hints:
        return None
    keyed = hints.get(f"{symbol}:{date_sold}")
    if keyed is not None:
        return _to_float(keyed)
    plain = hints.get(symbol)
    if plain is not None:
        return _to_float(plain)
    return None


def _resolve_basis(
    row: Dict[str, Any],
    *,
    cost_basis_hints: Optional[Dict[str, Any]],
    unknown_basis_policy: str,
    needs_review: List[Dict[str, Any]],
    warnings: List[str],
) -> Optional[Dict[str, Any]]:
    symbol = row["symbol"]
    proceeds = float(row["proceeds"])
    date_sold = row["date_sold"]
    realized_gain = row.get("realized_gain")
    cost_basis = row.get("cost_basis")
    date_acquired = row.get("date_acquired")
    basis_source = None

    if realized_gain is not None and cost_basis is None:
        cost_basis = proceeds - float(realized_gain)
        basis_source = "pnl_trade_history"
    elif cost_basis is not None and realized_gain is None:
        realized_gain = proceeds - float(cost_basis)
        basis_source = row.get("source") or "pnl_trade_history"

    if cost_basis is None:
        hint = _lookup_cost_basis_hint(cost_basis_hints, symbol, date_sold)
        if hint is not None:
            cost_basis = hint
            realized_gain = proceeds - cost_basis
            basis_source = "cost_basis_hint"

    if cost_basis is None:
        policy = (unknown_basis_policy or "conservative").lower()
        if policy == "exclude":
            warnings.append(
                f"{symbol}: excluded sell on {date_sold} ({row['quantity']} shares) "
                "because cost basis is unknown."
            )
            return None
        cost_basis = proceeds + _ASSUMED_LOSS
        realized_gain = -_ASSUMED_LOSS
        basis_source = "assumed_loss"
        needs_review.append(
            {
                "symbol": symbol,
                "date_sold": date_sold,
                "quantity": row["quantity"],
                "proceeds": proceeds,
                "reason": "unknown_basis_assumed_loss",
                "policy": policy,
            }
        )

    if basis_source is None and row.get("source") == "pnl_trade_history":
        basis_source = "pnl_trade_history"

    if not date_acquired:
        date_acquired = date_sold

    closed = {
        "symbol": symbol,
        "identifier": symbol,
        "quantity": float(row["quantity"]),
        "cost_basis": float(cost_basis),
        "proceeds": proceeds,
        "date_acquired": date_acquired,
        "date_sold": date_sold,
        "realized_gain": float(realized_gain),
        "basis_source": basis_source,
    }
    if row.get("order_id"):
        closed["order_id"] = row["order_id"]
    if row.get("source"):
        closed["data_source"] = row["source"]
    return closed


def _parse_current_date(current_date: Optional[str]) -> date:
    if current_date:
        return datetime.strptime(str(current_date)[:10], "%Y-%m-%d").date()
    return date.today()


def _build_wash_sale_preview(
    closed_lots: Sequence[Dict[str, Any]],
    open_lots: Sequence[Dict[str, Any]],
    *,
    current: date,
    window_days: int,
) -> Dict[str, Any]:
    window_start = current - timedelta(days=window_days)
    buy_restricted: List[Dict[str, Any]] = []
    for lot in closed_lots:
        if float(lot["realized_gain"]) >= 0:
            continue
        sold = datetime.strptime(lot["date_sold"], "%Y-%m-%d").date()
        buy_restricted.append(
            {
                "symbol": lot["symbol"],
                "date_sold": lot["date_sold"],
                "realized_gain": lot["realized_gain"],
                "restriction_ends_after": (sold + timedelta(days=30)).isoformat(),
                "basis_source": lot.get("basis_source"),
            }
        )

    recent_open: List[Dict[str, Any]] = []
    not_selectable: List[Dict[str, Any]] = []
    for lot in open_lots:
        acquired = lot.get("date_acquired") or lot.get("date") or lot.get("open_date")
        acquired_date = datetime.strptime(str(acquired)[:10], "%Y-%m-%d").date()
        if acquired_date >= window_start:
            recent_open.append(
                {
                    "symbol": lot.get("symbol") or lot.get("identifier"),
                    "open_lot_id": lot.get("open_lot_id") or lot.get("tax_lot_id"),
                    "date_acquired": str(acquired)[:10],
                    "quantity": lot.get("quantity"),
                    "risk": "buy_buy_sell",
                }
            )
        if lot.get("is_selectable") is False:
            not_selectable.append(
                {
                    "symbol": lot.get("symbol") or lot.get("identifier"),
                    "open_lot_id": lot.get("open_lot_id") or lot.get("tax_lot_id"),
                    "date_acquired": str(acquired)[:10],
                    "quantity": lot.get("quantity"),
                    "reason": "is_selectable_false",
                }
            )

    return {
        "buy_restricted_symbols": buy_restricted,
        "recent_open_lots": recent_open,
        "not_selectable_lots": not_selectable,
    }


def build_wash_sale_history(
    *,
    equity_orders: Optional[Sequence[Any]] = None,
    pnl_trade_history: Optional[Sequence[Any]] = None,
    tax_lots: Optional[Sequence[Any]] = None,
    current_date: Optional[str] = None,
    window_days: int = 31,
    cost_basis_hints: Optional[Dict[str, Any]] = None,
    unknown_basis_policy: str = "conservative",
) -> Dict[str, Any]:
    """Merge Robinhood payloads into Oracle ``recently_closed_lots`` wash-sale input."""
    if unknown_basis_policy.lower() not in {"conservative", "exclude"}:
        raise ValueError("unknown_basis_policy must be 'conservative' or 'exclude'")

    current = _parse_current_date(current_date)
    window_start = current - timedelta(days=window_days)

    order_rows, order_meta, order_warnings = flatten_equity_orders(equity_orders)
    pnl_rows, pnl_meta, pnl_warnings = flatten_pnl_trade_history(pnl_trade_history)
    merged = _merge_sale_rows(order_rows, pnl_rows)

    warnings: List[str] = list(order_warnings) + list(pnl_warnings)
    needs_review: List[Dict[str, Any]] = []
    recently_closed_lots: List[Dict[str, Any]] = []

    for row in merged:
        sold = datetime.strptime(row["date_sold"], "%Y-%m-%d").date()
        if sold < window_start or sold > current:
            continue
        resolved = _resolve_basis(
            row,
            cost_basis_hints=cost_basis_hints,
            unknown_basis_policy=unknown_basis_policy,
            needs_review=needs_review,
            warnings=warnings,
        )
        if resolved:
            recently_closed_lots.append(resolved)

    open_lots: List[Dict[str, Any]] = []
    if tax_lots:
        try:
            open_lots = flatten_robinhood_tax_lots(list(tax_lots))
        except Exception as exc:  # pragma: no cover - defensive for malformed payloads
            warnings.append(f"Could not flatten tax_lots for preview: {exc}")

    preview = _build_wash_sale_preview(
        recently_closed_lots,
        open_lots,
        current=current,
        window_days=window_days,
    )

    coverage = {
        "current_date": current.isoformat(),
        "window_days": window_days,
        "window_start": window_start.isoformat(),
        "window_end": current.isoformat(),
        "equity_orders_pages": order_meta["pages"],
        "equity_orders_seen": order_meta["orders_seen"],
        "equity_orders_sells_kept": order_meta["orders_kept"],
        "equity_orders_unread_next": order_meta["has_unread_next"],
        "equity_orders_next_cursor": order_meta["next_cursor"],
        "pnl_trade_history_pages": pnl_meta["pages"],
        "pnl_trades_seen": pnl_meta["trades_seen"],
        "pnl_trades_kept": pnl_meta["trades_kept"],
        "pnl_trade_history_unread_next": pnl_meta["has_unread_next"],
        "pnl_trade_history_next_cursor": pnl_meta["next_cursor"],
        "merged_sells_in_window": len(recently_closed_lots),
        "loss_sells_in_window": sum(
            1 for lot in recently_closed_lots if float(lot["realized_gain"]) < 0
        ),
    }

    if order_meta["has_unread_next"]:
        warnings.append(
            "equity_orders response has unread pages; pass additional pages or paginate "
            f"with cursor={order_meta['next_cursor']!r}."
        )
    if pnl_meta["has_unread_next"]:
        warnings.append(
            "pnl_trade_history response has unread pages; paginate with "
            f"next_cursor={pnl_meta['next_cursor']!r}."
        )

    return {
        "recently_closed_lots": recently_closed_lots,
        "wash_sale_preview": preview,
        "needs_review": needs_review,
        "warnings": warnings,
        "coverage": coverage,
    }
