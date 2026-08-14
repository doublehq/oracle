"""Robinhood Trading MCP field mapping and order payload builders.

Maps ``get_equity_tax_lots`` / Oracle lot-level trades into the live
``review_equity_order`` and ``place_equity_order`` request bodies (string
quantities, optional ``tax_lots`` on sells).
"""

from __future__ import annotations

from decimal import Decimal, ROUND_DOWN
from typing import Any, Dict, List, Optional, Sequence, Tuple
from uuid import uuid4

from src.service.helpers.constants import CASH_CUSIP_ID

MAX_TAX_LOTS_PER_ORDER = 30
_QTY_TOL = Decimal("0.000001")


def _norm_symbol(raw: Dict[str, Any]) -> str:
    symbol = raw.get("symbol") or raw.get("identifier") or raw.get("ticker")
    if not symbol:
        raise ValueError(f"Missing symbol/identifier in record: {raw}")
    return str(symbol).upper()


def _parse_decimal(value: Any) -> Decimal:
    if value is None or value == "":
        return Decimal("0")
    return Decimal(str(value))


def _format_qty(quantity: Decimal) -> str:
    """Robinhood expects decimal strings (fractional market up to 6 dp)."""
    q = quantity.quantize(Decimal("0.000001"), rounding=ROUND_DOWN)
    text = format(q.normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _lot_open_id(raw: Dict[str, Any]) -> str:
    lot_id = raw.get("open_lot_id") or raw.get("tax_lot_id") or raw.get("lot_id")
    if not lot_id:
        raise ValueError(f"Tax lot missing open_lot_id/tax_lot_id: {raw}")
    return str(lot_id)


def normalize_robinhood_tax_lot(raw: Dict[str, Any], index: int = 0) -> Dict[str, Any]:
    """Normalize one Robinhood ``get_equity_tax_lots`` row for Oracle."""
    symbol = _norm_symbol(raw)
    quantity = raw.get("quantity")
    if quantity is None:
        quantity = raw.get("quantity_available")
    if quantity is None:
        raise ValueError(f"Tax lot missing quantity: {raw}")

    acquired = (
        raw.get("open_date")
        or raw.get("date_acquired")
        or raw.get("date")
        or raw.get("acquisition_date")
    )
    if not acquired:
        raise ValueError(f"Tax lot missing acquisition date (open_date): {raw}")

    qty_f = float(quantity)
    cost_basis = raw.get("tax_cost_basis") or raw.get("cost_basis") or raw.get("total_cost_basis")
    if cost_basis is None:
        per_share = raw.get("cost_per_share") or raw.get("average_cost")
        if per_share is None:
            raise ValueError(
                f"Tax lot missing cost basis (tax_cost_basis or cost_per_share): {raw}"
            )
        cost_basis = float(per_share) * qty_f

    open_lot_id = _lot_open_id(raw)
    return {
        "symbol": symbol,
        "identifier": symbol,
        "open_lot_id": open_lot_id,
        "tax_lot_id": open_lot_id,
        "quantity": qty_f,
        "cost_basis": float(cost_basis),
        "date_acquired": str(acquired)[:10],
        "date": str(acquired)[:10],
        "is_selectable": raw.get("is_selectable"),
        "term": raw.get("term"),
    }


def flatten_robinhood_tax_lots(
    responses: Sequence[Dict[str, Any]],
    *,
    symbol: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Flatten one or more ``get_equity_tax_lots`` tool responses into Oracle inputs.

    Each response may be the full MCP payload ``{data: {symbol, tax_lots: [...]}}``
    or a bare list of lot dicts. When ``symbol`` is omitted it is taken from each
    response's ``data.symbol`` or from each lot row.
    """
    out: List[Dict[str, Any]] = []
    idx = 0
    for response in responses:
        if isinstance(response, list):
            rows = response
            default_symbol = symbol
        else:
            data = response.get("data") or response
            rows = data.get("tax_lots") or data.get("lots") or []
            default_symbol = symbol or data.get("symbol")
        for row in rows:
            lot = dict(row)
            if default_symbol and not lot.get("symbol"):
                lot["symbol"] = default_symbol
            out.append(normalize_robinhood_tax_lot(lot, idx))
            idx += 1
    return out


def _is_sell(trade: Dict[str, Any]) -> bool:
    action = str(trade.get("trade_type") or trade.get("action") or "").upper()
    return action == "SELL"


def _is_buy(trade: Dict[str, Any]) -> bool:
    action = str(trade.get("trade_type") or trade.get("action") or "").upper()
    return action == "BUY"


def _trade_symbol(trade: Dict[str, Any]) -> str:
    return str(trade.get("symbol") or trade.get("identifier") or "").upper()


def _trade_open_lot_id(trade: Dict[str, Any]) -> str:
    return str(trade.get("open_lot_id") or "")


def _choose_order_type(quantity: Decimal, requested_type: str) -> Tuple[str, Optional[str]]:
    """Return (type, market_hours). Fractional qty requires market + regular_hours."""
    order_type = (requested_type or "market").lower()
    if quantity != quantity.to_integral_value():
        if order_type != "market":
            raise ValueError(
                f"Fractional quantity {_format_qty(quantity)} requires type=market "
                f"(got {order_type!r}). Specified-lot fractional sells also require "
                "market_hours=regular_hours."
            )
        return "market", "regular_hours"
    return order_type, None


def build_robinhood_equity_orders(
    account_number: str,
    trades: Sequence[Dict[str, Any]],
    *,
    order_type: str = "market",
    market_hours: str = "regular_hours",
    time_in_force: str = "gfd",
    use_specified_tax_lots: bool = True,
    include_ref_id: bool = False,
) -> Dict[str, Any]:
    """Build ``review_equity_order`` / ``place_equity_order`` payloads from Oracle trades.

    Uses lot-level Oracle ``trades`` (strategy results), not netted symbol rows.
    Sells with ``open_lot_id`` / ``tax_lot_id`` become specified-lot orders when
    ``use_specified_tax_lots`` is true. Buys never include ``tax_lots``.
    """
    if not account_number:
        raise ValueError("account_number is required (Agentic account with agentic_allowed=true)")
    if (order_type or "market").lower() != "market":
        raise ValueError(
            "Oracle order payload generation only supports market orders. "
            "Build limit and stop orders directly with their required prices."
        )

    review_orders: List[Dict[str, Any]] = []
    place_orders: List[Dict[str, Any]] = []
    warnings: List[str] = []

    sells_by_symbol: Dict[str, List[Dict[str, Any]]] = {}
    buys: List[Dict[str, Any]] = []

    for trade in trades:
        symbol = _trade_symbol(trade)
        if not symbol or symbol in {CASH_CUSIP_ID, "CASH"}:
            continue
        qty = _parse_decimal(trade.get("quantity"))
        if qty <= _QTY_TOL:
            continue
        if _is_sell(trade):
            sells_by_symbol.setdefault(symbol, []).append(dict(trade))
        elif _is_buy(trade):
            buys.append({"symbol": symbol, "quantity": qty})

    for symbol, symbol_sells in sorted(sells_by_symbol.items()):
        chunks = _chunk_specified_lot_sells(symbol, symbol_sells, use_specified_tax_lots, warnings)
        for chunk in chunks:
            qty = (
                sum(_parse_decimal(lot["quantity"]) for lot in chunk["tax_lots"])
                if chunk["tax_lots"]
                else sum(_parse_decimal(t["quantity"]) for t in chunk["trades"])
            )
            chosen_type, forced_hours = _choose_order_type(qty, order_type)
            hours = forced_hours or market_hours
            payload: Dict[str, Any] = {
                "account_number": account_number,
                "symbol": symbol,
                "side": "sell",
                "type": chosen_type,
                "quantity": _format_qty(qty),
                "time_in_force": time_in_force,
                "market_hours": hours,
            }
            if chunk["tax_lots"]:
                payload["tax_lots"] = chunk["tax_lots"]
            review_orders.append(dict(payload))
            place_payload = dict(payload)
            if include_ref_id:
                place_payload["ref_id"] = str(uuid4())
            place_orders.append(place_payload)

    for buy in buys:
        qty = buy["quantity"]
        chosen_type, forced_hours = _choose_order_type(qty, order_type)
        hours = forced_hours or market_hours
        payload = {
            "account_number": account_number,
            "symbol": buy["symbol"],
            "side": "buy",
            "type": chosen_type,
            "quantity": _format_qty(qty),
            "time_in_force": time_in_force,
            "market_hours": hours,
        }
        review_orders.append(dict(payload))
        place_payload = dict(payload)
        if include_ref_id:
            place_payload["ref_id"] = str(uuid4())
        place_orders.append(place_payload)

    return {
        "account_number": account_number,
        "use_specified_tax_lots": use_specified_tax_lots,
        "review_equity_orders": review_orders,
        "place_equity_orders": place_orders,
        "warnings": warnings,
        "workflow": (
            "Call review_equity_order for each payload in review_equity_orders, "
            "present results to the user, then place_equity_order using the matching "
            "place_equity_orders entry (add a fresh ref_id per logical order on first "
            "place; reuse the same ref_id on transport retries)."
        ),
    }


def _chunk_specified_lot_sells(
    symbol: str,
    trades: List[Dict[str, Any]],
    use_specified_tax_lots: bool,
    warnings: List[str],
) -> List[Dict[str, Any]]:
    """Group lot-level sells into RH orders (max 30 tax_lots per order)."""
    if not use_specified_tax_lots:
        total = sum(_parse_decimal(t["quantity"]) for t in trades)
        return [{"trades": trades, "tax_lots": None}]

    lot_rows: List[Dict[str, str]] = []
    missing_ids = 0
    for trade in trades:
        lot_id = _trade_open_lot_id(trade)
        qty = _parse_decimal(trade["quantity"])
        if not lot_id:
            missing_ids += 1
            continue
        lot_rows.append({"open_lot_id": lot_id, "quantity": _format_qty(qty)})

    if missing_ids:
        warnings.append(
            f"{symbol}: {missing_ids} sell row(s) missing open_lot_id — omitting tax_lots "
            "for the entire symbol order; Robinhood will use default FIFO."
        )
        return [{"trades": trades, "tax_lots": None}]

    if not lot_rows:
        total = sum(_parse_decimal(t["quantity"]) for t in trades)
        return [{"trades": trades, "tax_lots": None}]

    chunks: List[Dict[str, Any]] = []
    for i in range(0, len(lot_rows), MAX_TAX_LOTS_PER_ORDER):
        batch = lot_rows[i : i + MAX_TAX_LOTS_PER_ORDER]
        if i + MAX_TAX_LOTS_PER_ORDER < len(lot_rows):
            warnings.append(
                f"{symbol}: split specified-lot sell into multiple orders "
                f"(>{MAX_TAX_LOTS_PER_ORDER} lots)."
            )
        chunks.append({"trades": trades, "tax_lots": batch})
    return chunks
