"""Oracle MCP server: portfolio optimization tools over the Model Context Protocol.

Local-only stdio server. Clone the repo, install, and register with your MCP client:

    python -m src.mcp_server
    # or, after `pip install -e .`:
    oracle-mcp
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from src.mcp_server.event_builder import DEFAULT_SETTINGS, STRATEGY_ID, build_event
from src.mcp_server.guide import ROBINHOOD_WORKFLOW_GUIDE
from src.mcp_server.index_targets import (
    DEFAULT_SPY_HOLDINGS_URL,
    fetch_etf_holdings_targets as _fetch_etf_holdings_targets,
)
from src.mcp_server.factors import (
    build_factor_model_from_fundamentals as _build_factor_model_from_fundamentals,
)
from src.mcp_server.robinhood import (
    build_robinhood_equity_orders as _build_robinhood_equity_orders,
    flatten_robinhood_tax_lots,
)
from src.mcp_server.snapshot import fetch_robinhood_snapshot as _fetch_robinhood_snapshot
from src.mcp_server.snapshot import load_snapshot
from src.mcp_server.wash_sale_history import build_wash_sale_history as _build_wash_sale_history
from src.service.helpers.disposal import (
    normalize_disposal_method,
    reconcile_execution,
)
from src.service.oracle import Oracle

logger = logging.getLogger(__name__)

SERVER_INSTRUCTIONS = """\
Oracle is a tax-aware portfolio optimizer (rebalancing, tax-loss harvesting,
direct indexing). It RECOMMENDS trades; it never executes them. When paired
with the Robinhood Trading MCP, operate dry-run by default: compute trades,
call review_equity_order for each payload in robinhood_orders, and only
place_equity_order after the user explicitly confirms the exact trade list.

Robinhood place_equity_order supports optional tax_lots [{open_lot_id, quantity}]
on SELL orders (from get_equity_tax_lots). Pass open_lot_id through Oracle's
tax_lots input; optimize_portfolio returns ready-made review/place payloads
when account_number is set. Omit tax_lots on the RH order only for intentional
FIFO disposal. Fractional specified-lot sells require type=market and
market_hours=regular_hours.
"""

mcp = MCPServer(
    "oracle-portfolio-optimizer",
    instructions=SERVER_INSTRUCTIONS,
)


def _resolve_artifact_path(path: str, field_name: str) -> Path:
    root = Path(
        os.environ.get("ORACLE_ARTIFACT_DIR", str(Path.cwd() / "artifacts"))
    ).expanduser().resolve()
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate = candidate.resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(
            f"{field_name} must be inside ORACLE_ARTIFACT_DIR ({root})"
        )
    return candidate


def _configure_stdio_logging() -> None:
    """Send logs to stderr so they cannot corrupt the stdio MCP transport."""
    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    for handler in list(root.handlers):
        stream = getattr(handler, "stream", None)
        if isinstance(handler, logging.StreamHandler) and stream in (sys.stdout, sys.__stdout__):
            root.removeHandler(handler)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root.addHandler(stderr_handler)


def _normalize_trades(
    trades: List[Dict[str, Any]],
    broker_lot_ids: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    broker_lot_ids = broker_lot_ids or {}
    normalized = []
    for trade in trades or []:
        row = dict(trade)
        action = str(row.get("action") or row.get("trade_type") or "").lower()
        if action:
            row["action"] = action
            row["trade_type"] = action.upper()
        identifier = row.get("identifier") or row.get("symbol")
        if identifier:
            row["identifier"] = identifier
            row["symbol"] = identifier
        tax_lot_id = row.get("tax_lot_id")
        open_lot_id = row.get("open_lot_id")
        if tax_lot_id:
            row["tax_lot_id"] = str(tax_lot_id)
        if open_lot_id:
            row["open_lot_id"] = str(open_lot_id)
        elif tax_lot_id and str(tax_lot_id) in broker_lot_ids:
            row["open_lot_id"] = broker_lot_ids[str(tax_lot_id)]
        normalized.append(row)
    return normalized


def _run_oracle_event(event: Dict[str, Any]) -> Dict[str, Any]:
    result = Oracle.process_lambda_event(event)
    event_strategies = (event.get("oracle") or {}).get("strategies") or {}
    all_broker_lot_ids: Dict[str, str] = {}
    for strategy_id, strategy in result.get("results", {}).items():
        source_strategy = event_strategies.get(str(strategy_id)) or event_strategies.get(strategy_id) or {}
        broker_lot_ids = {
            str(lot["tax_lot_id"]): str(lot["open_lot_id"])
            for lot in source_strategy.get("tax_lots") or []
            if lot.get("tax_lot_id") and lot.get("open_lot_id")
        }
        all_broker_lot_ids.update(broker_lot_ids)
        if isinstance(strategy, dict) and "trades" in strategy:
            strategy["trades"] = _normalize_trades(
                strategy.get("trades") or [],
                broker_lot_ids,
            )
    result["netted_trades"] = _normalize_trades(
        result.get("netted_trades") or [],
        all_broker_lot_ids,
    )
    return json.loads(json.dumps(result, default=str))


def _lot_level_trades(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    trades: List[Dict[str, Any]] = []
    for strategy in result.get("results", {}).values():
        trades.extend(strategy.get("trades") or [])
    return trades


def _merge_snapshot_inputs(
    *,
    snapshot_path: Optional[str],
    tax_lots: Optional[List[Dict[str, Any]]],
    prices: Optional[List[Dict[str, Any]]],
    targets: Optional[List[Dict[str, Any]]],
    cash: Optional[float],
    factor_model: Optional[List[Dict[str, Any]]],
    recently_closed_lots: Optional[List[Dict[str, Any]]],
    account_number: Optional[str],
) -> tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    float,
    Optional[List[Dict[str, Any]]],
    Optional[List[Dict[str, Any]]],
    Optional[str],
    float,
]:
    """Apply snapshot_path defaults; explicit arguments win."""
    reported_cash = 0.0
    snap_account = account_number
    if snapshot_path:
        snap = load_snapshot(snapshot_path)
        tax_lots = tax_lots if tax_lots is not None else snap.get("tax_lots") or []
        prices = prices if prices is not None else snap.get("prices") or []
        targets = targets if targets is not None else snap.get("targets") or []
        if cash is None:
            reported_cash = float(snap.get("cash") or 0.0)
            cash = reported_cash
        if factor_model is None:
            factor_model = snap.get("factor_model")
        if recently_closed_lots is None:
            recently_closed_lots = snap.get("recently_closed_lots")
        if snap_account is None:
            snap_account = snap.get("account_number")
    else:
        tax_lots = tax_lots or []
        prices = prices or []
        targets = targets or []
        if cash is None:
            raise ValueError("cash is required when snapshot_path is not provided")
        reported_cash = float(cash)

    if not targets:
        raise ValueError("targets are required (directly or via snapshot_path)")
    if not prices:
        raise ValueError("prices are required (directly or via snapshot_path)")

    return (
        tax_lots,
        prices,
        targets,
        float(cash),
        factor_model,
        recently_closed_lots,
        snap_account,
        reported_cash,
    )


def _summarize_optimize_result(result: Dict[str, Any]) -> Dict[str, Any]:
    strategy = result.get("results", {}).get(STRATEGY_ID) or {}
    return {
        "should_trade": strategy.get("should_trade"),
        "status": strategy.get("status"),
        "netted_trades": len(result.get("netted_trades") or []),
        "lot_level_trades": len(strategy.get("trades") or []),
        "used_default_tax_rates": result.get("used_default_tax_rates"),
    }


@mcp.tool()
def optimize_portfolio(
    tax_lots: Annotated[
        Optional[List[Dict[str, Any]]],
        Field(
            description=(
                "Current holdings as tax lots. Each: {symbol, quantity, cost_basis "
                "(TOTAL dollars; or cost_per_share / tax_cost_basis), open_date or "
                "date_acquired (YYYY-MM-DD), open_lot_id (from get_equity_tax_lots; "
                "stored as tax_lot_id). Empty list for cash-only. Omit when using "
                "snapshot_path."
            )
        ),
    ] = None,
    prices: Annotated[
        Optional[List[Dict[str, Any]]],
        Field(
            description=(
                "Real-time prices for EVERY symbol held or targeted. Each: {symbol, price} "
                "(bid_price/ask_price also accepted; midpoint is used). From Robinhood: "
                "get_equity_quotes. Omit when using snapshot_path."
            )
        ),
    ] = None,
    targets: Annotated[
        Optional[List[Dict[str, Any]]],
        Field(
            description=(
                "Target allocation. Each: {symbol, target_weight} with weights summing to "
                "~1.0. Optionally {asset_class, identifiers: [primary, alternate], "
                "target_weight}. Omit when using snapshot_path."
            )
        ),
    ] = None,
    cash: Annotated[
        Optional[float],
        Field(
            description=(
                "Available cash / buying power in dollars. From Robinhood: get_portfolio. "
                "Omit when using snapshot_path."
            )
        ),
    ] = None,
    snapshot_path: Annotated[
        Optional[str],
        Field(
            description=(
                "Path to a snapshot JSON from fetch_robinhood_snapshot. Loads tax_lots, "
                "prices, targets, cash, factor_model, recently_closed_lots, and "
                "account_number unless overridden by explicit arguments."
            )
        ),
    ] = None,
    out_path: Annotated[
        Optional[str],
        Field(
            description=(
                "If set, write the full optimize result JSON to this path and return a "
                "compact summary plus out_path."
            )
        ),
    ] = None,
    investable_amount: Annotated[
        Optional[float],
        Field(
            description=(
                "Override effective cash for optimization (e.g. 500 for a dry-run buy-in). "
                "Replaces total cash, not incremental deployment. Response includes "
                "cash_override with reported_cash vs investable_amount; robinhood_orders "
                "are tagged simulated when this differs from reported cash."
            )
        ),
    ] = None,
    optimization_type: Annotated[
        str,
        Field(
            description=(
                "One of: TAX_AWARE (default; rebalance minimizing tax+costs), TAX_UNAWARE, "
                "BUY_ONLY (deploy cash, no sells), PAIRS_TLH (tax-loss harvesting with pair "
                "alternatives; set settings.should_tlh=true), DIRECT_INDEX (direct indexing; "
                "supports TLH and factor_model), HOLD (no trades)."
            )
        ),
    ] = "TAX_AWARE",
    current_date: Annotated[
        Optional[str],
        Field(description="Valuation date YYYY-MM-DD. Defaults to today."),
    ] = None,
    tax_rates: Annotated[
        Optional[Dict[str, float]],
        Field(
            description=(
                "User's combined federal+state rates, e.g. {'short_term': 0.35, "
                "'long_term': 0.20, 'qualified_dividend': 0.15}. Defaults: 37% ST / 20% LT — "
                "tell the user if you rely on defaults."
            )
        ),
    ] = None,
    recently_closed_lots: Annotated[
        Optional[List[Dict[str, Any]]],
        Field(
            description=(
                "Sells from the last ~35 days for wash-sale protection. Each: {symbol, "
                "quantity, cost_basis, proceeds, date_acquired, date_sold, realized_gain}. "
                "Build from Robinhood via the build_wash_sale_history tool "
                "(get_pnl_trade_history + get_equity_orders)."
            )
        ),
    ] = None,
    stock_restrictions: Annotated[
        Optional[List[Dict[str, Any]]],
        Field(description="Per-symbol trade restrictions: {symbol, can_buy, can_sell}."),
    ] = None,
    spreads: Annotated[
        Optional[List[Dict[str, Any]]],
        Field(description="Optional bid-ask spreads as fractions: {symbol, spread}, e.g. 0.001 = 10bps."),
    ] = None,
    factor_model: Annotated[
        Optional[List[Dict[str, Any]]],
        Field(description="Optional factor-model rows for DIRECT_INDEX. Pass through as-is."),
    ] = None,
    withdrawal_amount: Annotated[
        float,
        Field(description="Dollars to raise tax-efficiently (0 for none). Not valid with HOLD/BUY_ONLY."),
    ] = 0.0,
    settings: Annotated[
        Optional[Dict[str, Any]],
        Field(
            description=(
                "Optimizer overrides. Keys and defaults: "
                + json.dumps(DEFAULT_SETTINGS)
                + ". Notable: should_tlh + tlh_min_loss_threshold (TLH), holding_time_days "
                "(min holding period), min_notional (min trade $), weight_tax/weight_drift/"
                "weight_transaction/weight_cash_drag (objective weights), trade_rounding "
                "(decimal places for quantities; use 0 to force whole shares)."
            )
        ),
    ] = None,
    disposal_method: Annotated[
        str,
        Field(
            description=(
                "Default FIFO/LIFO/HIFO disposal when comparing against account-default "
                "sells (tax_lots omitted) or when enforce_disposal_order=true. "
                "When use_specified_tax_lots=true (default), pass tax_lots on RH orders "
                "instead of relying on account disposal."
            )
        ),
    ] = "FIFO",
    use_specified_tax_lots: Annotated[
        bool,
        Field(
            description=(
                "If true (default), build Robinhood SELL orders with tax_lots "
                "[{open_lot_id, quantity}] from Oracle's lot-level sells. If false, "
                "omit tax_lots (Robinhood uses default FIFO) and execution_impact "
                "compares against simulated FIFO."
            )
        ),
    ] = True,
    account_number: Annotated[
        Optional[str],
        Field(
            description=(
                "Agentic account_number (agentic_allowed=true from get_accounts). "
                "When set, response includes robinhood_orders with review_equity_order "
                "and place_equity_order payloads."
            )
        ),
    ] = None,
    order_type: Annotated[
        str,
        Field(description="Robinhood order type. Only market is currently supported."),
    ] = "market",
    market_hours: Annotated[
        str,
        Field(description="Robinhood market_hours for robinhood_orders. Default regular_hours."),
    ] = "regular_hours",
    time_in_force: Annotated[
        str,
        Field(description="Robinhood time_in_force for robinhood_orders. Default gfd."),
    ] = "gfd",
    enforce_disposal_order: Annotated[
        bool,
        Field(
            description=(
                "If true, constrain the optimizer so sells within a symbol follow "
                "disposal_method (for workflows that omit tax_lots on RH orders). "
                "Default false when using specified-lot execution."
            )
        ),
    ] = False,
) -> Dict[str, Any]:
    """Compute recommended trades for a single portfolio (dry-run: nothing is executed).

    Returns per-strategy results, netted_trades, execution_impact, and (when
    account_number is set) robinhood_orders ready for review_equity_order /
    place_equity_order. Do not place orders until the user confirms the list.
    """
    if snapshot_path:
        snapshot_path = str(_resolve_artifact_path(snapshot_path, "snapshot_path"))
    if out_path:
        out_path = str(_resolve_artifact_path(out_path, "out_path"))

    (
        tax_lots,
        prices,
        targets,
        effective_cash,
        factor_model,
        recently_closed_lots,
        resolved_account,
        reported_cash,
    ) = _merge_snapshot_inputs(
        snapshot_path=snapshot_path,
        tax_lots=tax_lots,
        prices=prices,
        targets=targets,
        cash=cash,
        factor_model=factor_model,
        recently_closed_lots=recently_closed_lots,
        account_number=account_number,
    )
    account_number = resolved_account or account_number

    cash_override: Optional[Dict[str, Any]] = None
    simulated_orders = False
    if investable_amount is not None:
        cash_override = {
            "applied": True,
            "reported_cash": reported_cash,
            "investable_amount": float(investable_amount),
        }
        if float(investable_amount) != float(reported_cash):
            simulated_orders = True
        effective_cash = float(investable_amount)

    method = normalize_disposal_method(disposal_method)
    event = build_event(
        tax_lots=tax_lots,
        prices=prices,
        targets=targets,
        cash=effective_cash,
        optimization_type=optimization_type,
        current_date=current_date,
        tax_rates=tax_rates,
        recently_closed_lots=recently_closed_lots,
        stock_restrictions=stock_restrictions,
        spreads=spreads,
        factor_model=factor_model,
        withdrawal_amount=withdrawal_amount,
        settings=settings,
    )
    if enforce_disposal_order:
        event["settings"]["strategies"][STRATEGY_ID]["disposal_method"] = method
        event["settings"]["strategies"][STRATEGY_ID]["enforce_disposal_order"] = True

    result = _run_oracle_event(event)
    result["used_default_tax_rates"] = tax_rates is None
    result["disposal_method"] = method
    result["use_specified_tax_lots"] = use_specified_tax_lots
    lot_trades = _lot_level_trades(result)
    result["execution_impact"] = reconcile_execution(
        tax_lots=event["oracle"]["strategies"][STRATEGY_ID]["tax_lots"],
        prices=event["oracle"]["strategies"][STRATEGY_ID]["prices"],
        current_date=event["oracle"]["current_date"],
        tax_rates=event["oracle"]["tax_rates"],
        trades=lot_trades,
        disposal_method=method,
        use_specified_tax_lots=use_specified_tax_lots,
    )
    if account_number:
        rh_orders = _build_robinhood_equity_orders(
            account_number=account_number,
            trades=lot_trades,
            order_type=order_type,
            market_hours=market_hours,
            time_in_force=time_in_force,
            use_specified_tax_lots=use_specified_tax_lots,
            include_ref_id=True,
        )
        if simulated_orders:
            rh_orders["simulated"] = True
            rh_orders["warning"] = (
                "Orders exceed real buying power; investable_amount was used for optimization."
            )
        result["robinhood_orders"] = rh_orders
    if cash_override is not None:
        result["cash_override"] = cash_override
    if out_path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        summary = _summarize_optimize_result(result)
        summary["out_path"] = str(out)
        return summary
    return result


@mcp.tool()
def build_robinhood_equity_orders(
    account_number: Annotated[
        str,
        Field(description="Agentic account_number (agentic_allowed=true)."),
    ],
    trades: Annotated[
        List[Dict[str, Any]],
        Field(
            description=(
                "Lot-level Oracle trades (results[].trades), not netted_trades. "
                "Each sell should include open_lot_id/tax_lot_id when using specified lots."
            )
        ),
    ],
    use_specified_tax_lots: Annotated[
        bool,
        Field(description="Include tax_lots on SELL orders (default true)."),
    ] = True,
    order_type: Annotated[str, Field(description="Order type. Only market is currently supported.")] = "market",
    market_hours: Annotated[str, Field(description="regular_hours (default), extended_hours, all_day_hours.")] = "regular_hours",
    time_in_force: Annotated[str, Field(description="gfd (default) or gtc.")] = "gfd",
) -> Dict[str, Any]:
    """Build review_equity_order / place_equity_order payloads from Oracle lot-level trades.

    Returns review_equity_orders (call these first) and place_equity_orders
    (with ref_id UUIDs — reuse ref_id only on retries of the same logical order).
    """
    return _build_robinhood_equity_orders(
        account_number=account_number,
        trades=trades,
        order_type=order_type,
        market_hours=market_hours,
        time_in_force=time_in_force,
        use_specified_tax_lots=use_specified_tax_lots,
        include_ref_id=True,
    )


@mcp.tool()
def build_factor_model_from_fundamentals(
    fundamentals: Annotated[
        List[Any],
        Field(
            description=(
                "One or more get_equity_fundamentals MCP responses "
                "({data: {results: [...], not_found: [...]}}) or bare result row lists."
            )
        ),
    ],
    prices: Annotated[
        Optional[List[Dict[str, Any]]],
        Field(
            description=(
                "Optional prices from get_equity_quotes: [{symbol, price}, ...]. "
                "When omitted, uses session (high+low)/2 from each fundamentals row."
            )
        ),
    ] = None,
    include_sectors: Annotated[
        bool,
        Field(
            description=(
                "If true (default), add sector_* one-hot columns from each row's "
                "sector field (FactSet taxonomy from Robinhood)."
            )
        ),
    ] = True,
    factors: Annotated[
        Optional[List[str]],
        Field(
            description=(
                "Style factors to include. Default: value, momentum, size, "
                "dividend_yield, low_volatility, liquidity."
            )
        ),
    ] = None,
    winsorize: Annotated[
        float,
        Field(description="Clip cross-sectional z-scores to +/- this value (default 3.0)."),
    ] = 3.0,
) -> Dict[str, Any]:
    """Build an Oracle factor_model from Robinhood get_equity_fundamentals payloads.

    Returns factor_model rows plus coverage and diagnostics. Use with
    optimization_type=DIRECT_INDEX and settings.weight_factor_model > 0.
    """
    return _build_factor_model_from_fundamentals(
        fundamentals,
        prices=prices,
        include_sectors=include_sectors,
        factors=factors,
        winsorize=winsorize,
    )


@mcp.tool()
def normalize_robinhood_tax_lots(
    tax_lot_responses: Annotated[
        List[Dict[str, Any]],
        Field(
            description=(
                "One or more get_equity_tax_lots MCP responses "
                "({data: {symbol, tax_lots: [...]}}) or bare lot dict lists."
            )
        ),
    ],
) -> List[Dict[str, Any]]:
    """Flatten Robinhood get_equity_tax_lots responses into optimize_portfolio tax_lots."""
    return flatten_robinhood_tax_lots(tax_lot_responses)


@mcp.tool()
def fetch_etf_holdings_targets(
    url: Annotated[
        str,
        Field(
            description=(
                "Holdings file URL. Defaults to SSGA's daily SPY XLSX "
                "(S&P 500 via SPY constituents)."
            )
        ),
    ] = DEFAULT_SPY_HOLDINGS_URL,
    cash_target_weight: Annotated[
        float,
        Field(
            description=(
                "Optional cash buffer as a fraction (0–1). Adds a CASH target row and "
                "scales equity weights to fill the remainder. Default 0 (100% invested)."
            )
        ),
    ] = 0.0,
    min_weight_pct: Annotated[
        Optional[float],
        Field(
            description=(
                "Drop constituents below this index weight in percent "
                "(e.g. 0.01 = 0.01%). Remaining names are renormalized."
            )
        ),
    ] = None,
    top_n: Annotated[
        Optional[int],
        Field(description="Keep only the top N holdings by index weight, then renormalize."),
    ] = None,
) -> Dict[str, Any]:
    """Download an SSGA daily holdings XLSX and return Oracle ``targets`` rows.

    Default URL is SPY (https://www.ssga.com/.../holdings-daily-us-en-spy.xlsx).
    Each target includes {symbol, target_weight, asset_class, identifiers, name,
    weight_pct_raw}. Pass the ``targets`` list directly to ``optimize_portfolio``.
    Weights are normalized to sum to 1.0 (minus cash_target_weight). Fund cash
    (US DOLLAR) inside the index file is excluded by default.
    """
    return _fetch_etf_holdings_targets(
        url=url,
        cash_target_weight=cash_target_weight,
        min_weight_pct=min_weight_pct,
        top_n=top_n,
    )


@mcp.tool()
def compute_optimal_trades(
    event: Annotated[
        Dict[str, Any],
        Field(
            description=(
                "Full Oracle event: {oracle: {current_date, recently_closed_lots, "
                "stock_restrictions, tax_rates, percentage_protection_from_inadvertent_wash_sales, "
                "strategies: {id: {tax_lots, targets, prices, cash, spreads, factor_model, "
                "optimization_type, withdrawal_amount}}}, settings: {strategies: {id: {...}}}, "
                "max_withdrawal_amount_settings (optional)}. See the repo README for the schema. "
                "To constrain sells to a broker disposal method, set "
                "settings.strategies.<id>.enforce_disposal_order=true and "
                "settings.strategies.<id>.disposal_method to FIFO, LIFO, or HIFO."
            )
        ),
    ],
) -> Dict[str, Any]:
    """Advanced: run Oracle on a raw multi-strategy event (the AWS Lambda input format).

    Use optimize_portfolio for the common single-portfolio case; use this when you need
    multiple strategies netted together, factor models, or max-withdrawal calculations.
    """
    return _run_oracle_event(event)


@mcp.tool()
def build_wash_sale_history(
    pnl_trade_history: Annotated[
        Optional[List[Any]],
        Field(
            description=(
                "One or more get_pnl_trade_history MCP responses "
                "({data: {trades: [...], next_cursor}}) or bare trade lists. "
                "PRIMARY source for realized gain/loss. Pass account_number with the "
                "rhs_account_number value from get_accounts (not the agentic account_number "
                "when they differ)."
            )
        ),
    ] = None,
    equity_orders: Annotated[
        Optional[List[Any]],
        Field(
            description=(
                "One or more get_equity_orders MCP responses "
                "({data: {orders: [...], next}}) or bare order lists. "
                "Fallback/cross-check when PnL rows are missing or have null "
                "realized_gain. Fetch with state=filled and "
                "created_at_gte=<today-31d>; paginate using the cursor query "
                "param from next."
            )
        ),
    ] = None,
    tax_lots: Annotated[
        Optional[List[Any]],
        Field(
            description=(
                "Optional get_equity_tax_lots responses for wash_sale_preview only "
                "(recent purchases / is_selectable=false). One symbol per call."
            )
        ),
    ] = None,
    current_date: Annotated[
        Optional[str],
        Field(description="Valuation date YYYY-MM-DD. Defaults to today."),
    ] = None,
    window_days: Annotated[
        int,
        Field(description="Lookback window in days (default 31)."),
    ] = 31,
    cost_basis_hints: Annotated[
        Optional[Dict[str, Any]],
        Field(
            description=(
                "Optional cost basis overrides when Robinhood has no basis. Keys: "
                "SYMBOL or SYMBOL:YYYY-MM-DD -> total cost_basis dollars."
            )
        ),
    ] = None,
    unknown_basis_policy: Annotated[
        str,
        Field(
            description=(
                "When cost basis is unknown: conservative (default) assumes a "
                "marginal loss so the symbol is buy-restricted; exclude drops "
                "the sale with a warning."
            )
        ),
    ] = "conservative",
) -> Dict[str, Any]:
    """Build Oracle recently_closed_lots from Robinhood order and PnL payloads.

    The agent fetches raw JSON from Robinhood and passes it here unchanged.
    Returns recently_closed_lots (for optimize_portfolio), wash_sale_preview,
    needs_review, warnings, and coverage metadata.
    """
    return _build_wash_sale_history(
        equity_orders=equity_orders,
        pnl_trade_history=pnl_trade_history,
        tax_lots=tax_lots,
        current_date=current_date,
        window_days=window_days,
        cost_basis_hints=cost_basis_hints,
        unknown_basis_policy=unknown_basis_policy,
    )


@mcp.tool()
def fetch_robinhood_snapshot(
    account_number: Annotated[
        Optional[str],
        Field(
            description=(
                "Agentic account_number. When omitted, auto-selects the first "
                "agentic_allowed=true account from get_accounts."
            )
        ),
    ] = None,
    out_path: Annotated[
        str,
        Field(description="Write the full snapshot artifact JSON to this path."),
    ] = "artifacts/rh_snapshot.json",
    top_n: Annotated[
        Optional[int],
        Field(description="Optional: keep only top N SPY holdings (default: full index)."),
    ] = None,
    include_fundamentals: Annotated[
        bool,
        Field(description="Fetch get_equity_fundamentals and build factor_model (default true)."),
    ] = True,
    include_wash_sale_history: Annotated[
        bool,
        Field(description="Fetch PnL/orders and build recently_closed_lots (default true)."),
    ] = True,
    cache_dir: Annotated[
        str,
        Field(description="Per-batch cache directory for resumable fetches."),
    ] = "artifacts/.rh_cache",
    max_seconds: Annotated[
        float,
        Field(
            description=(
                "Time budget for this invocation (default 90s). Returns status=partial "
                "when exceeded; call again to resume from cache."
            )
        ),
    ] = 90.0,
) -> Dict[str, Any]:
    """Fetch Robinhood account state + SPY targets + quotes + fundamentals (read-only).

    Uses Oracle's authenticated Robinhood MCP client (run oracle-rh-login once).
    Returns counts and out_path only — not the full payload. For large universes,
    call repeatedly until status=complete.
    """
    resolved_out = _resolve_artifact_path(out_path, "out_path")
    resolved_cache = _resolve_artifact_path(cache_dir, "cache_dir")
    return _fetch_robinhood_snapshot(
        account_number=account_number,
        out_path=resolved_out,
        top_n=top_n,
        include_fundamentals=include_fundamentals,
        include_wash_sale_history=include_wash_sale_history,
        cache_dir=resolved_cache,
        max_seconds=max_seconds,
    )


@mcp.tool()
def get_workflow_guide() -> str:
    """Get the step-by-step guide for using Oracle with the Robinhood Trading MCP.

    Covers which Robinhood tools to fetch data from, field mappings into
    optimize_portfolio, all optimization types, disposal-method handling, and
    the mandatory dry-run / user-confirmation safety workflow for order placement.
    """
    return ROBINHOOD_WORKFLOW_GUIDE


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Oracle local MCP server (stdio). Register the absolute path to this script with your MCP client."
    )
    parser.parse_args()
    _configure_stdio_logging()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
