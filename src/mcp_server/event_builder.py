"""Build Oracle lambda-style events from agent-friendly, broker-normalized inputs.

The shapes accepted here are intentionally close to what a brokerage MCP
server (e.g. the Robinhood Trading MCP) returns from tools like
``get_equity_tax_lots``, ``get_equity_quotes``, ``get_portfolio`` and
``get_pnl_trade_history``, so an orchestrating agent only has to do light
field renaming rather than restructuring.
"""

from datetime import date
from typing import Any, Dict, List, Optional

# Default combined (federal + state) tax rates used when the caller does not
# supply their own. Agents should ask the user for their real rates.
DEFAULT_TAX_RATES = [
    {"gain_type": "long_term", "federal_rate": 0.20, "state_rate": 0.0, "total_rate": 0.20},
    {"gain_type": "short_term", "federal_rate": 0.37, "state_rate": 0.0, "total_rate": 0.37},
    {"gain_type": "qualified_dividend", "federal_rate": 0.15, "state_rate": 0.0, "total_rate": 0.15},
]

DEFAULT_SETTINGS = {
    "weight_tax": 1.0,
    "weight_drift": 1.0,
    "weight_transaction": 1.0,
    "weight_factor_model": 0.0,
    "weight_cash_drag": 0.0,
    "rebalance_threshold": None,
    "buy_threshold": 0.005,
    "holding_time_days": 1,
    "should_tlh": False,
    "tlh_min_loss_threshold": 0.015,
    "range_min_weight_multiplier": 0.5,
    "range_max_weight_multiplier": 2.0,
    "min_notional": 0,
    "rank_penalty_factor": 0.0,
    "trade_rounding": 4,
    "enforce_wash_sale_prevention": True,
}

STRATEGY_ID = "1"


def _norm_symbol(raw: Dict[str, Any]) -> str:
    symbol = raw.get("symbol") or raw.get("identifier") or raw.get("ticker")
    if not symbol:
        raise ValueError(f"Missing symbol/identifier in record: {raw}")
    return str(symbol).upper()


def _norm_tax_lot(raw: Dict[str, Any], index: int) -> Dict[str, Any]:
    quantity = raw.get("quantity")
    if quantity is None:
        quantity = raw.get("quantity_available")
    if quantity is None:
        raise ValueError(f"Tax lot missing 'quantity': {raw}")

    acquired = (
        raw.get("open_date")
        or raw.get("date_acquired")
        or raw.get("date")
        or raw.get("acquisition_date")
    )
    if not acquired:
        raise ValueError(f"Tax lot missing acquisition date ('open_date' / 'date_acquired'): {raw}")

    cost_basis = raw.get("tax_cost_basis") or raw.get("cost_basis") or raw.get("total_cost_basis")
    if cost_basis is None:
        per_share = raw.get("cost_basis_per_share") or raw.get("cost_per_share") or raw.get("average_cost")
        if per_share is None:
            raise ValueError(
                f"Tax lot missing cost basis (provide 'tax_cost_basis', 'cost_basis' total, or 'cost_per_share'): {raw}"
            )
        cost_basis = float(per_share) * float(quantity)

    open_lot_id = raw.get("open_lot_id")
    internal_lot_id = raw.get("tax_lot_id") or raw.get("lot_id")
    lot_id = str(open_lot_id or internal_lot_id or f"lot_{index}")

    normalized = {
        "tax_lot_id": lot_id,
        "identifier": _norm_symbol(raw),
        "quantity": float(quantity),
        "cost_basis": float(cost_basis),
        "date": str(acquired)[:10],
    }
    if open_lot_id:
        normalized["open_lot_id"] = str(open_lot_id)
    return normalized


def _norm_target(raw: Dict[str, Any]) -> Dict[str, Any]:
    weight = raw.get("target_weight") if raw.get("target_weight") is not None else raw.get("weight")
    if weight is None:
        raise ValueError(f"Target missing 'target_weight': {raw}")

    identifiers = raw.get("identifiers")
    if identifiers:
        identifiers = [str(i).upper() for i in identifiers]
    else:
        identifiers = [_norm_symbol(raw)]

    asset_class = str(raw.get("asset_class") or identifiers[0]).upper()
    return {"asset_class": asset_class, "target_weight": float(weight), "identifiers": identifiers}


def _norm_price(raw: Dict[str, Any]) -> Dict[str, Any]:
    price = raw.get("price") if raw.get("price") is not None else raw.get("last_trade_price")
    if price is None:
        # Fall back to bid/ask midpoint if a quote payload was passed through.
        bid, ask = raw.get("bid_price"), raw.get("ask_price")
        if bid is not None and ask is not None:
            price = (float(bid) + float(ask)) / 2
        else:
            raise ValueError(f"Price record missing 'price': {raw}")
    return {"identifier": _norm_symbol(raw), "price": float(price)}


def _norm_closed_lot(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "identifier": _norm_symbol(raw),
        "quantity": float(raw["quantity"]),
        "cost_basis": float(raw["cost_basis"]),
        "proceeds": float(raw["proceeds"]),
        "date_acquired": str(raw["date_acquired"])[:10],
        "date_sold": str(raw["date_sold"])[:10],
        "realized_gain": float(
            raw.get("realized_gain", float(raw["proceeds"]) - float(raw["cost_basis"]))
        ),
    }


def _norm_tax_rates(tax_rates: Optional[Any]) -> List[Dict[str, Any]]:
    if not tax_rates:
        return DEFAULT_TAX_RATES
    if isinstance(tax_rates, dict):
        # Simple form: {"short_term": 0.35, "long_term": 0.2, "qualified_dividend": 0.15}
        rates = []
        for gain_type, default in (
            ("long_term", 0.20),
            ("short_term", 0.37),
            ("qualified_dividend", 0.15),
        ):
            rate = float(tax_rates.get(gain_type, default))
            rates.append(
                {"gain_type": gain_type, "federal_rate": rate, "state_rate": 0.0, "total_rate": rate}
            )
        return rates
    return list(tax_rates)


def build_event(
    tax_lots: List[Dict[str, Any]],
    prices: List[Dict[str, Any]],
    targets: List[Dict[str, Any]],
    cash: float,
    optimization_type: str = "TAX_AWARE",
    current_date: Optional[str] = None,
    tax_rates: Optional[Any] = None,
    recently_closed_lots: Optional[List[Dict[str, Any]]] = None,
    stock_restrictions: Optional[List[Dict[str, Any]]] = None,
    spreads: Optional[List[Dict[str, Any]]] = None,
    factor_model: Optional[List[Dict[str, Any]]] = None,
    withdrawal_amount: float = 0.0,
    settings: Optional[Dict[str, Any]] = None,
    percentage_protection_from_inadvertent_wash_sales: float = 0.003,
) -> Dict[str, Any]:
    """Build a complete Oracle event (the same shape ``Oracle.process_lambda_event``
    consumes) from normalized single-strategy inputs.
    """
    strategy_settings = dict(DEFAULT_SETTINGS)
    if settings:
        unknown = set(settings) - set(DEFAULT_SETTINGS)
        if unknown:
            raise ValueError(
                f"Unknown settings keys: {sorted(unknown)}. Valid keys: {sorted(DEFAULT_SETTINGS)}"
            )
        strategy_settings.update(settings)

    normalized_spreads = [
        {"identifier": _norm_symbol(s), "spread": float(s["spread"])} for s in (spreads or [])
    ]

    event = {
        "oracle": {
            "current_date": current_date or date.today().isoformat(),
            "recently_closed_lots": [_norm_closed_lot(l) for l in (recently_closed_lots or [])],
            "stock_restrictions": [
                {
                    "identifier": _norm_symbol(r),
                    "can_buy": bool(r.get("can_buy", True)),
                    "can_sell": bool(r.get("can_sell", True)),
                }
                for r in (stock_restrictions or [])
            ],
            "tax_rates": _norm_tax_rates(tax_rates),
            "percentage_protection_from_inadvertent_wash_sales": percentage_protection_from_inadvertent_wash_sales,
            "strategies": {
                STRATEGY_ID: {
                    "strategy_id": int(STRATEGY_ID),
                    "label": f"{STRATEGY_ID}:MCP:{optimization_type.upper()}",
                    "optimization_type": optimization_type.upper(),
                    "tax_lots": [_norm_tax_lot(l, i) for i, l in enumerate(tax_lots or [])],
                    "targets": [_norm_target(t) for t in (targets or [])],
                    "prices": [_norm_price(p) for p in (prices or [])],
                    "cash": float(cash),
                    "spreads": normalized_spreads or None,
                    "factor_model": factor_model or None,
                    "withdrawal_amount": float(withdrawal_amount),
                }
            },
        },
        "settings": {"strategies": {STRATEGY_ID: strategy_settings}},
    }
    return event
