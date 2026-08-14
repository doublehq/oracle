import json

import pytest

from src.mcp_server.event_builder import build_event
from src.mcp_server.robinhood import build_robinhood_equity_orders, flatten_robinhood_tax_lots
from src.mcp_server.server import (
    build_robinhood_equity_orders as build_rh_orders_tool,
    compute_optimal_trades,
    get_workflow_guide,
    normalize_robinhood_tax_lots,
    optimize_portfolio,
)


TAX_LOTS = [
    # Big winner: 10 AAPL bought cheap long ago (total basis $500)
    {"symbol": "AAPL", "quantity": 10, "cost_basis": 500.0, "date_acquired": "2023-01-15"},
    # Loser lot: 20 SNAP at $15/share, now $8 (per-share basis form)
    {"symbol": "SNAP", "quantity": 20, "cost_basis_per_share": 15.0, "date_acquired": "2025-01-10"},
]

PRICES = [
    {"symbol": "AAPL", "price": 200.0},
    {"symbol": "SNAP", "price": 8.0},
    {"symbol": "MSFT", "price": 400.0},
]

TARGETS = [
    {"symbol": "AAPL", "target_weight": 0.4},
    {"symbol": "SNAP", "target_weight": 0.3},
    {"symbol": "MSFT", "target_weight": 0.3},
]


def test_build_event_normalizes_robinhood_open_lot_id():
    event = build_event(
        tax_lots=[
            {
                "symbol": "AAPL",
                "open_lot_id": "rh-lot-uuid-1",
                "open_date": "2024-06-01",
                "quantity": "10",
                "tax_cost_basis": "1500.00",
            }
        ],
        prices=[{"symbol": "AAPL", "price": 200.0}],
        targets=[{"symbol": "AAPL", "target_weight": 1.0}],
        cash=0.0,
    )
    lot = event["oracle"]["strategies"]["1"]["tax_lots"][0]
    assert lot["tax_lot_id"] == "rh-lot-uuid-1"
    assert lot["open_lot_id"] == "rh-lot-uuid-1"
    assert lot["cost_basis"] == pytest.approx(1500.0)


def test_build_event_keeps_internal_lot_id_distinct_from_broker_open_lot_id():
    event = build_event(
        tax_lots=[
            {
                "symbol": "AAPL",
                "tax_lot_id": "internal-lot-1",
                "open_date": "2024-06-01",
                "quantity": "10",
                "tax_cost_basis": "1500.00",
            }
        ],
        prices=[{"symbol": "AAPL", "price": 200.0}],
        targets=[{"symbol": "AAPL", "target_weight": 1.0}],
        cash=0.0,
    )

    lot = event["oracle"]["strategies"]["1"]["tax_lots"][0]
    assert lot["tax_lot_id"] == "internal-lot-1"
    assert "open_lot_id" not in lot


def test_flatten_robinhood_tax_lots_response():
    lots = flatten_robinhood_tax_lots(
        [
            {
                "data": {
                    "symbol": "AAPL",
                    "tax_lots": [
                        {
                            "open_lot_id": "lot-a",
                            "open_date": "2024-01-01",
                            "quantity": "5",
                            "cost_per_share": "100",
                        }
                    ],
                }
            }
        ]
    )
    assert len(lots) == 1
    assert lots[0]["open_lot_id"] == "lot-a"
    assert lots[0]["cost_basis"] == pytest.approx(500.0)


def test_build_robinhood_equity_orders_specified_lot_sell():
    orders = build_robinhood_equity_orders(
        account_number="TEST_ACCOUNT_001",
        trades=[
            {
                "symbol": "AAPL",
                "action": "sell",
                "quantity": 12.5,
                "open_lot_id": "lot-1",
            },
            {
                "symbol": "AAPL",
                "action": "sell",
                "quantity": 2.5,
                "open_lot_id": "lot-2",
            },
        ],
        use_specified_tax_lots=True,
        include_ref_id=True,
    )
    review = orders["review_equity_orders"][0]
    assert review["account_number"] == "TEST_ACCOUNT_001"
    assert review["side"] == "sell"
    assert review["type"] == "market"
    assert review["market_hours"] == "regular_hours"
    assert review["quantity"] == "15"
    assert review["tax_lots"] == [
        {"open_lot_id": "lot-1", "quantity": "12.5"},
        {"open_lot_id": "lot-2", "quantity": "2.5"},
    ]
    assert "ref_id" in orders["place_equity_orders"][0]


def test_build_robinhood_equity_orders_splits_large_lot_sets_without_duplicating_quantity():
    orders = build_robinhood_equity_orders(
        account_number="TEST_ACCOUNT_001",
        trades=[
            {
                "symbol": "AAPL",
                "action": "sell",
                "quantity": 1,
                "open_lot_id": f"lot-{index}",
            }
            for index in range(31)
        ],
        use_specified_tax_lots=True,
    )

    reviews = orders["review_equity_orders"]
    assert [order["quantity"] for order in reviews] == ["30", "1"]
    assert [
        sum(float(lot["quantity"]) for lot in order["tax_lots"])
        for order in reviews
    ] == [30.0, 1.0]


def test_build_robinhood_equity_orders_does_not_send_internal_lot_ids_to_broker():
    orders = build_robinhood_equity_orders(
        account_number="TEST_ACCOUNT_001",
        trades=[
            {
                "symbol": "AAPL",
                "action": "sell",
                "quantity": 2,
                "tax_lot_id": "internal-lot-1",
            }
        ],
        use_specified_tax_lots=True,
    )

    review = orders["review_equity_orders"][0]
    assert "tax_lots" not in review
    assert any("missing open_lot_id" in warning for warning in orders["warnings"])


def test_build_robinhood_equity_orders_rejects_unsupported_non_market_orders():
    with pytest.raises(ValueError, match="only supports market orders"):
        build_robinhood_equity_orders(
            account_number="TEST_ACCOUNT_001",
            trades=[{"symbol": "AAPL", "action": "buy", "quantity": 1}],
            order_type="limit",
        )


def test_optimize_portfolio_includes_robinhood_orders_when_account_set():
    result = optimize_portfolio(
        tax_lots=[
            {
                "symbol": "AAPL",
                "open_lot_id": "lot-1",
                "quantity": 10,
                "cost_basis": 500.0,
                "date_acquired": "2020-01-01",
            }
        ],
        prices=[{"symbol": "AAPL", "price": 200.0}, {"symbol": "CASH", "price": 1.0}],
        targets=[{"symbol": "AAPL", "target_weight": 0.5}, {"symbol": "CASH", "target_weight": 0.5}],
        cash=0.0,
        optimization_type="TAX_AWARE",
        current_date="2026-08-12",
        account_number="TEST_ACCOUNT_001",
    )
    assert "robinhood_orders" in result
    sells = [t for t in result["results"]["1"]["trades"] if t["trade_type"] == "SELL"]
    if sells:
        assert result["robinhood_orders"]["review_equity_orders"]
        assert result["execution_impact"]["place_equity_order_supports_tax_lots"] is True


def test_normalize_robinhood_tax_lots_tool():
    out = normalize_robinhood_tax_lots(
        [{"data": {"symbol": "MSFT", "tax_lots": [{"open_lot_id": "x", "open_date": "2024-01-01", "quantity": "1", "tax_cost_basis": "300"}]}}]
    )
    assert out[0]["symbol"] == "MSFT"
    assert out[0]["open_lot_id"] == "x"


def test_build_robinhood_equity_orders_tool_wrapper():
    out = build_rh_orders_tool(
        account_number="TEST_ACCOUNT_001",
        trades=[{"symbol": "MSFT", "action": "buy", "quantity": 1}],
    )
    assert out["review_equity_orders"][0]["side"] == "buy"


def test_build_event_normalizes_robinhood_shapes():
    event = build_event(
        tax_lots=TAX_LOTS,
        prices=PRICES,
        targets=TARGETS,
        cash=1000.0,
        tax_rates={"short_term": 0.35, "long_term": 0.20},
        recently_closed_lots=[
            {
                "symbol": "TSLA",
                "quantity": 1,
                "cost_basis": 300.0,
                "proceeds": 250.0,
                "date_acquired": "2025-06-01",
                "date_sold": "2025-08-01",
            }
        ],
    )

    strategy = event["oracle"]["strategies"]["1"]
    # Per-share cost basis converted to lot total
    snap_lot = next(l for l in strategy["tax_lots"] if l["identifier"] == "SNAP")
    assert snap_lot["cost_basis"] == pytest.approx(300.0)
    assert strategy["cash"] == 1000.0
    assert event["oracle"]["recently_closed_lots"][0]["realized_gain"] == pytest.approx(-50.0)
    rates = {r["gain_type"]: r["total_rate"] for r in event["oracle"]["tax_rates"]}
    assert rates["short_term"] == 0.35 and rates["long_term"] == 0.20
    # Event must be JSON-serializable as-is
    json.dumps(event)


def test_build_event_rejects_unknown_settings():
    with pytest.raises(ValueError, match="Unknown settings keys"):
        build_event(TAX_LOTS, PRICES, TARGETS, cash=0.0, settings={"weight_taxx": 1})


def test_build_event_price_midpoint_from_bid_ask():
    event = build_event(
        tax_lots=[],
        prices=[{"symbol": "AAPL", "bid_price": 99.0, "ask_price": 101.0}],
        targets=[{"symbol": "AAPL", "target_weight": 1.0}],
        cash=100.0,
    )
    assert event["oracle"]["strategies"]["1"]["prices"][0]["price"] == pytest.approx(100.0)


def test_optimize_portfolio_tax_aware_rebalance():
    result = optimize_portfolio(
        tax_lots=TAX_LOTS,
        prices=PRICES,
        targets=TARGETS,
        cash=1000.0,
        optimization_type="TAX_AWARE",
        current_date="2026-08-12",
        tax_rates={"short_term": 0.35, "long_term": 0.20},
    )

    assert "netted_trades" in result and "results" in result
    strategy_result = result["results"]["1"]
    assert strategy_result["status"] is not None

    trades = result["netted_trades"]
    # Portfolio is 100% short of MSFT (target 30%) -> must recommend buying MSFT
    msft_buys = [t for t in trades if t["identifier"] == "MSFT" and t["trade_type"] == "BUY"]
    assert msft_buys, f"expected a MSFT buy, got: {trades}"
    assert "execution_impact" in result
    assert result["disposal_method"] == "FIFO"
    assert result["used_default_tax_rates"] is False
    json.dumps(result)


def test_optimize_portfolio_buy_only_never_sells():
    result = optimize_portfolio(
        tax_lots=TAX_LOTS,
        prices=PRICES,
        targets=TARGETS,
        cash=5000.0,
        optimization_type="BUY_ONLY",
        current_date="2026-08-12",
    )
    sells = [t for t in result["netted_trades"] if t["trade_type"] == "SELL"]
    assert sells == []
    assert result["execution_impact"]["sells"] == []
    assert result["used_default_tax_rates"] is True


def test_optimize_portfolio_hold_returns_no_trades():
    result = optimize_portfolio(
        tax_lots=TAX_LOTS,
        prices=PRICES,
        targets=TARGETS,
        cash=1000.0,
        optimization_type="HOLD",
        current_date="2026-08-12",
    )
    assert result["netted_trades"] == []


def test_compute_optimal_trades_raw_event_passthrough():
    event = build_event(
        tax_lots=TAX_LOTS,
        prices=PRICES,
        targets=TARGETS,
        cash=1000.0,
        current_date="2026-08-12",
    )
    result = compute_optimal_trades(event)
    assert "netted_trades" in result


def test_workflow_guide_mentions_safety_and_robinhood_tools():
    guide = get_workflow_guide()
    for needle in [
        "review_equity_order",
        "place_equity_order",
        "get_equity_tax_lots",
        "open_lot_id",
        "DRY-RUN",
        "tax_lots",
        "specified_lot",
    ]:
        assert needle.lower() in guide.lower() or needle in guide


def test_optimize_portfolio_rejects_invalid_disposal_method():
    with pytest.raises(ValueError, match="Invalid disposal_method"):
        optimize_portfolio(
            tax_lots=TAX_LOTS,
            prices=PRICES,
            targets=TARGETS,
            cash=1000.0,
            disposal_method="AVERAGE_COST",
        )
