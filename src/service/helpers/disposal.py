"""Account-level tax-lot disposal methods (FIFO / LIFO / HIFO).

Robinhood's ``place_equity_order`` supports optional specified-lot sells via
``tax_lots: [{open_lot_id, quantity}, ...]``. When omitted, Robinhood uses
FIFO. These helpers sort lots for FIFO/LIFO/HIFO simulation (fallback path),
compare Oracle recommendations to default FIFO disposal, and support the
``enforce_disposal_order`` optimizer constraint.
"""

from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.service.helpers.constants import CASH_CUSIP_ID

VALID_DISPOSAL_METHODS = ("FIFO", "LIFO", "HIFO")
_QTY_TOL = 1e-8


def normalize_disposal_method(method: Optional[str]) -> str:
    """Return a canonical disposal method (FIFO, LIFO, or HIFO)."""
    if method is None or str(method).strip() == "":
        return "FIFO"
    canonical = str(method).strip().upper()
    aliases = {
        "HIGHEST_COST": "HIFO",
        "HIGHESTCOST": "HIFO",
    }
    canonical = aliases.get(canonical, canonical)
    if canonical not in VALID_DISPOSAL_METHODS:
        raise ValueError(
            f"Invalid disposal_method {method!r}. Valid values: {', '.join(VALID_DISPOSAL_METHODS)}"
        )
    return canonical


def _as_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _cost_per_share(lot: Dict[str, Any]) -> float:
    quantity = float(lot.get("quantity") or 0)
    if quantity == 0:
        return 0.0
    if lot.get("cost_per_share") is not None:
        return float(lot["cost_per_share"])
    return float(lot["cost_basis"]) / quantity


def _lot_id(lot: Dict[str, Any]) -> str:
    return str(lot.get("open_lot_id") or lot.get("tax_lot_id") or lot.get("lot_id") or "")


def _identifier(lot: Dict[str, Any]) -> str:
    return str(lot.get("identifier") or lot.get("symbol") or "").upper()


def lots_in_disposal_order(
    lots: Sequence[Dict[str, Any]],
    disposal_method: str,
) -> List[Dict[str, Any]]:
    """Return a new list of lots ordered as the broker would dispose them."""
    method = normalize_disposal_method(disposal_method)

    def sort_key(lot: Dict[str, Any]) -> Tuple:
        acquired = _as_date(lot.get("date") or lot.get("date_acquired"))
        cps = _cost_per_share(lot)
        lot_id = _lot_id(lot)
        if method == "FIFO":
            return (acquired, cps, lot_id)
        if method == "LIFO":
            return (date.max - acquired, -cps, lot_id)
        # HIFO: highest cost per share first; older lot wins ties
        return (-cps, acquired, lot_id)

    return sorted((dict(lot) for lot in lots), key=sort_key)


def simulate_disposal(
    lots: Sequence[Dict[str, Any]],
    quantity_to_sell: float,
    disposal_method: str,
) -> Tuple[List[Dict[str, Any]], float]:
    """Consume ``quantity_to_sell`` shares from ``lots`` in disposal order.

    Returns (consumed_lots, leftover_quantity). leftover > 0 means the sell
    exceeded available shares.
    """
    remaining = float(quantity_to_sell)
    consumed: List[Dict[str, Any]] = []
    if remaining <= _QTY_TOL:
        return consumed, 0.0

    for lot in lots_in_disposal_order(lots, disposal_method):
        available = float(lot.get("quantity") or 0)
        if available <= _QTY_TOL or remaining <= _QTY_TOL:
            continue
        take = min(available, remaining)
        cps = _cost_per_share(lot)
        consumed.append(
            {
                "tax_lot_id": _lot_id(lot),
                "identifier": _identifier(lot),
                "quantity": take,
                "cost_basis": take * cps,
                "cost_per_share": cps,
                "date": str(lot.get("date") or lot.get("date_acquired"))[:10],
            }
        )
        remaining -= take

    leftover = remaining if remaining > _QTY_TOL else 0.0
    return consumed, leftover


def _rate_map(tax_rates: Optional[Any]) -> Dict[str, float]:
    defaults = {"short_term": 0.37, "long_term": 0.20, "qualified_dividend": 0.15}
    if not tax_rates:
        return defaults
    if isinstance(tax_rates, dict) and "short_term" in tax_rates:
        return {
            "short_term": float(tax_rates.get("short_term", defaults["short_term"])),
            "long_term": float(tax_rates.get("long_term", defaults["long_term"])),
            "qualified_dividend": float(
                tax_rates.get("qualified_dividend", defaults["qualified_dividend"])
            ),
        }
    rates = dict(defaults)
    for row in tax_rates:
        rates[str(row["gain_type"])] = float(row.get("total_rate", rates.get(row["gain_type"], 0)))
    return rates


def _price_map(prices: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for row in prices:
        symbol = str(row.get("identifier") or row.get("symbol") or "").upper()
        price = row.get("price")
        if price is None:
            bid, ask = row.get("bid_price"), row.get("ask_price")
            if bid is not None and ask is not None:
                price = (float(bid) + float(ask)) / 2
        if symbol and price is not None:
            out[symbol] = float(price)
    return out


def _gain_type(acquired: date, current: date) -> str:
    try:
        anniversary = acquired.replace(year=acquired.year + 1)
    except ValueError:
        # February 29 has no direct anniversary in a non-leap year.
        anniversary = acquired.replace(year=acquired.year + 1, day=28)
    return "long_term" if current > anniversary else "short_term"


def _tax_cost_of_lots(
    lots: Sequence[Dict[str, Any]],
    prices: Dict[str, float],
    current: date,
    rates: Dict[str, float],
) -> Dict[str, float]:
    short_term = 0.0
    long_term = 0.0
    tax = 0.0
    for lot in lots:
        symbol = _identifier(lot)
        qty = float(lot["quantity"])
        cps = _cost_per_share(lot)
        price = prices.get(symbol)
        if price is None:
            continue
        realized = qty * (price - cps)
        term = _gain_type(_as_date(lot.get("date") or lot.get("date_acquired")), current)
        if term == "long_term":
            long_term += realized
        else:
            short_term += realized
        tax += realized * rates.get(term, 0.0)
    return {
        "short_term_realized": short_term,
        "long_term_realized": long_term,
        "tax_cost": tax,
    }


def _is_sell(trade: Dict[str, Any]) -> bool:
    action = str(trade.get("trade_type") or trade.get("action") or "").upper()
    return action == "SELL"


def _is_buy(trade: Dict[str, Any]) -> bool:
    action = str(trade.get("trade_type") or trade.get("action") or "").upper()
    return action == "BUY"


def _qty_map(lots: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for lot in lots:
        lot_id = _lot_id(lot)
        if not lot_id:
            continue
        out[lot_id] = out.get(lot_id, 0.0) + float(lot["quantity"])
    return out


def _lots_diverge(intended: Dict[str, float], actual: Dict[str, float]) -> bool:
    ids = set(intended) | set(actual)
    for lot_id in ids:
        if abs(intended.get(lot_id, 0.0) - actual.get(lot_id, 0.0)) > 1e-4:
            return True
    return False


def reconcile_execution(
    tax_lots: Sequence[Dict[str, Any]],
    prices: Sequence[Dict[str, Any]],
    current_date: str,
    tax_rates: Optional[Any],
    trades: Sequence[Dict[str, Any]],
    disposal_method: str,
    use_specified_tax_lots: bool = True,
) -> Dict[str, Any]:
    """Compare Oracle lot-level sells to Robinhood execution outcomes.

    When ``use_specified_tax_lots`` is true (default), the broker path matches
    Oracle's chosen lots via ``place_equity_order.tax_lots`` (``open_lot_id`` from
    ``get_equity_tax_lots``). When false, or when a sell row lacks a lot id,
    simulates default FIFO disposal for comparison.
    """
    method = normalize_disposal_method(disposal_method)
    if not use_specified_tax_lots and method != "FIFO":
        raise ValueError(
            "Robinhood account-default disposal is FIFO. Use specified tax lots "
            "for LIFO/HIFO execution."
        )
    current = _as_date(current_date)
    rates = _rate_map(tax_rates)
    price_lookup = _price_map(prices)

    lots_by_symbol: Dict[str, List[Dict[str, Any]]] = {}
    for lot in tax_lots:
        symbol = _identifier(lot)
        if not symbol or symbol == CASH_CUSIP_ID or symbol == "CASH":
            continue
        lots_by_symbol.setdefault(symbol, []).append(dict(lot))

    sells_by_symbol: Dict[str, List[Dict[str, Any]]] = {}
    buys: List[Dict[str, Any]] = []
    for trade in trades:
        symbol = _identifier(trade)
        if not symbol or symbol == CASH_CUSIP_ID or symbol == "CASH":
            continue
        qty = float(trade.get("quantity") or 0)
        if qty <= _QTY_TOL:
            continue
        if _is_buy(trade):
            buys.append({"symbol": symbol, "side": "BUY", "quantity": qty})
        elif _is_sell(trade):
            sells_by_symbol.setdefault(symbol, []).append(dict(trade))

    sell_impacts: List[Dict[str, Any]] = []
    any_divergence = False
    for symbol, intended_trades in sorted(sells_by_symbol.items()):
        intended_qty = sum(float(t["quantity"]) for t in intended_trades)
        warnings_prefix = None
        intended_lots = [
            {
                "tax_lot_id": _lot_id(t),
                "open_lot_id": _lot_id(t),
                "quantity": float(t["quantity"]),
                "date": str(t.get("date") or t.get("date_acquired") or "")[:10] or None,
            }
            for t in intended_trades
            if _lot_id(t)
        ]
        # Enrich intended lots with cost basis from the holding when possible
        holdings = {_lot_id(l): l for l in lots_by_symbol.get(symbol, [])}
        for item in intended_lots:
            holding = holdings.get(item["tax_lot_id"])
            if holding is not None:
                item["cost_per_share"] = _cost_per_share(holding)
                item["date"] = str(holding.get("date") or holding.get("date_acquired"))[:10]

        all_lots_identified = len(intended_lots) == len(intended_trades) and bool(intended_lots)
        if use_specified_tax_lots and all_lots_identified:
            broker_lots = [
                {
                    "tax_lot_id": item["tax_lot_id"],
                    "open_lot_id": item["open_lot_id"],
                    "identifier": symbol,
                    "quantity": item["quantity"],
                    "cost_basis": item.get("cost_per_share", 0) * item["quantity"],
                    "cost_per_share": item.get("cost_per_share", 0),
                    "date": item.get("date") or current.isoformat(),
                }
                for item in intended_lots
            ]
            leftover = 0.0
            diverges = False
            execution_mode = "specified_lot"
        else:
            broker_lots, leftover = simulate_disposal(
                lots_by_symbol.get(symbol, []), intended_qty, "FIFO"
            )
            diverges = _lots_diverge(_qty_map(intended_lots), _qty_map(broker_lots))
            if leftover > _QTY_TOL:
                diverges = True
            execution_mode = "fifo_fallback"
            if use_specified_tax_lots and not all_lots_identified:
                warnings_prefix = (
                    f"{symbol}: missing open_lot_id on one or more sell rows — "
                    "cannot use specified-lot orders; compared against default FIFO."
                )
            else:
                warnings_prefix = None
        intended_tax = _tax_cost_of_lots(
            [
                {
                    **item,
                    "identifier": symbol,
                    "cost_basis": item.get("cost_per_share", 0) * item["quantity"],
                    "date": item.get("date") or current.isoformat(),
                }
                for item in intended_lots
            ],
            price_lookup,
            current,
            rates,
        )
        broker_tax = _tax_cost_of_lots(broker_lots, price_lookup, current, rates)
        if execution_mode != "specified_lot":
            if leftover > _QTY_TOL:
                diverges = True
        if diverges:
            any_divergence = True

        warnings: List[str] = []
        if warnings_prefix:
            warnings.append(warnings_prefix)
        if diverges and execution_mode == "fifo_fallback":
            warnings.append(
                f"{symbol}: Oracle selected different lots than {method} disposal will close. "
                "Pass tax_lots on place_equity_order (open_lot_id from get_equity_tax_lots) "
                "to match Oracle, or set enforce_disposal_order=true and re-run."
            )
        elif diverges:
            warnings.append(
                f"{symbol}: sell quantity {intended_qty} exceeds holdings; {leftover:.4f} shares unfilled."
            )
        if leftover > _QTY_TOL and not any(
            "exceeds holdings" in w for w in warnings
        ):
            warnings.append(
                f"{symbol}: sell quantity {intended_qty} exceeds holdings; {leftover:.4f} shares unfilled."
            )

        sell_impacts.append(
            {
                "symbol": symbol,
                "side": "SELL",
                "quantity": intended_qty,
                "diverges": diverges,
                "lots_oracle_intended": intended_lots,
                "lots_broker_will_dispose": [
                    {
                        "tax_lot_id": lot["tax_lot_id"],
                        "open_lot_id": lot.get("open_lot_id") or lot["tax_lot_id"],
                        "quantity": lot["quantity"],
                        "cost_per_share": lot["cost_per_share"],
                        "date": lot["date"],
                    }
                    for lot in broker_lots
                ],
                "execution_mode": execution_mode,
                "oracle_tax_cost": intended_tax["tax_cost"],
                "broker_tax_cost": broker_tax["tax_cost"],
                "tax_cost_delta": broker_tax["tax_cost"] - intended_tax["tax_cost"],
                "oracle_short_term_realized": intended_tax["short_term_realized"],
                "broker_short_term_realized": broker_tax["short_term_realized"],
                "warnings": warnings,
            }
        )

    return {
        "disposal_method": method,
        "use_specified_tax_lots": use_specified_tax_lots,
        "place_equity_order_supports_tax_lots": True,
        "note": (
            "Robinhood place_equity_order accepts optional tax_lots "
            "[{open_lot_id, quantity}, ...] on SELL orders (from get_equity_tax_lots). "
            "When omitted, Robinhood uses default FIFO. Fractional specified-lot sells "
            "require type=market and market_hours=regular_hours. Call review_equity_order "
            "first, then place_equity_order only after the user confirms the exact list."
        ),
        "any_divergence": any_divergence,
        "sells": sell_impacts,
        "buys": buys,
    }
