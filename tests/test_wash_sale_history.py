import pytest

from src.mcp_server.event_builder import build_event
from src.mcp_server.server import build_wash_sale_history, get_workflow_guide, optimize_portfolio
from src.mcp_server.wash_sale_history import (
    build_wash_sale_history as build_wash_sale_history_fn,
    extract_next_cursor,
    flatten_equity_orders,
    flatten_pnl_trade_history,
)
from src.service.initializers.closed_lots import initialize_closed_lots

# All brokerage-shaped records in this file are synthetic.
FRACTIONAL_BND_SELL = {
    "id": "test-order-fractional-sell",
    "symbol": "BND",
    "side": "sell",
    "type": "market",
    "state": "filled",
    "quantity": "0.5",
    "cumulative_quantity": "0.5",
    "average_price": "80.00",
    "fees": "0.00",
    "created_at": "2023-05-05T14:01:01.206676Z",
    "last_transaction_at": "2023-05-05T14:01:02.043407Z",
    "executions": [
        {
            "id": "test-execution-fractional-sell",
            "price": "80.00",
            "quantity": "0.5",
            "timestamp": "2023-05-05T14:01:01.518Z",
            "fees": "0.000000",
        }
    ],
}

MULTI_EXEC_VOO_SELL = {
    "id": "test-order-multi-execution",
    "symbol": "VOO",
    "side": "sell",
    "type": "market",
    "state": "filled",
    "quantity": "10",
    "cumulative_quantity": "10",
    "average_price": "300.00",
    "fees": "0.25",
    "last_transaction_at": "2020-12-23T14:30:03.912779Z",
    "executions": [
        {"price": "300.00", "quantity": "1", "timestamp": "2020-12-23T14:30:00Z"},
        {"price": "300.00", "quantity": "2", "timestamp": "2020-12-23T14:30:01Z"},
        {"price": "300.00", "quantity": "3", "timestamp": "2020-12-23T14:30:02Z"},
        {"price": "300.00", "quantity": "4", "timestamp": "2020-12-23T14:30:03Z"},
    ],
}

CANCELLED_BUY = {
    "symbol": "COIN",
    "side": "buy",
    "state": "cancelled",
    "quantity": "1",
    "executions": [],
}

CONFIRMED_BUY = {
    "symbol": "RVII",
    "side": "buy",
    "state": "confirmed",
    "quantity": "1",
    "executions": [],
}

PARTIAL_BUY = {
    "symbol": "CMI",
    "side": "buy",
    "state": "partially_filled_rest_cancelled",
    "cumulative_quantity": "10.000000",
    "last_transaction_at": "2022-08-18T00:00:06.839Z",
    "executions": [
        {
            "price": "230.280000",
            "quantity": "10.000000",
            "timestamp": "2022-08-17T22:43:38.276Z",
        }
    ],
}


def test_extract_next_cursor():
    url = "http://example.test/orders/?account_number=TEST_ACCOUNT_001&cursor=cD0yMDE5LTA1LTIw"
    assert extract_next_cursor(url) == "cD0yMDE5LTA1LTIw"
    assert extract_next_cursor(None) is None
    assert extract_next_cursor("http://example.com/no-cursor") is None


def test_flatten_equity_orders_fractional_sell():
    rows, meta, warnings = flatten_equity_orders([{"data": {"orders": [FRACTIONAL_BND_SELL]}}])
    assert len(rows) == 1
    row = rows[0]
    assert row["symbol"] == "BND"
    assert row["quantity"] == pytest.approx(0.5)
    assert row["proceeds"] == pytest.approx(0.5 * 80.0)
    assert row["date_sold"] == "2023-05-05"
    assert row["realized_gain"] is None
    assert meta["orders_kept"] == 1
    assert warnings == []


def test_flatten_equity_orders_ignores_cancelled_and_confirmed():
    rows, _, _ = flatten_equity_orders(
        [{"data": {"orders": [CANCELLED_BUY, CONFIRMED_BUY, FRACTIONAL_BND_SELL]}}]
    )
    assert len(rows) == 1
    assert rows[0]["symbol"] == "BND"


def test_flatten_equity_orders_ignores_non_sell_partial():
    rows, _, _ = flatten_equity_orders([{"data": {"orders": [PARTIAL_BUY]}}])
    assert rows == []


def test_flatten_equity_orders_multi_execution_proceeds():
    rows, _, _ = flatten_equity_orders([{"data": {"orders": [MULTI_EXEC_VOO_SELL]}}])
    assert len(rows) == 1
    expected_qty = 10.0
    assert rows[0]["quantity"] == pytest.approx(expected_qty)
    assert rows[0]["proceeds"] == pytest.approx(expected_qty * 300.0)


def test_flatten_pnl_trade_history_null_gain_is_unknown():
    rows, meta, warnings = flatten_pnl_trade_history(
        [
            {
                "data": {
                    "trades": [
                        {
                            "symbol": "BND",
                            "quantity": "0.5",
                            "price": "80.00",
                            "realized_gain": None,
                            "closed_at": "2023-05-05T14:01:01Z",
                        }
                    ],
                    "next_cursor": "",
                }
            }
        ]
    )
    assert len(rows) == 1
    assert rows[0]["realized_gain"] is None
    assert meta["trades_kept"] == 1
    assert any("null realized_gain" in w for w in warnings)


def test_flatten_pnl_trade_history_skips_options():
    rows, meta, _ = flatten_pnl_trade_history(
        [
            {
                "data": {
                    "trades": [
                        {
                            "symbol": "AAPL",
                            "quantity": "1",
                            "price": "100",
                            "realized_gain": "-10",
                            "closed_at": "2025-08-01T12:00:00Z",
                        },
                        {
                            "symbol": "AAPL",
                            "instrument_type": "option",
                            "quantity": "1",
                            "price": "1",
                            "realized_gain": "-1",
                        },
                    ]
                }
            }
        ]
    )
    assert len(rows) == 1
    assert rows[0]["symbol"] == "AAPL"
    assert meta["trades_kept"] == 1


def test_build_wash_sale_history_empty_responses():
    result = build_wash_sale_history_fn(
        equity_orders=[{"data": {"orders": [], "next": None}}],
        pnl_trade_history=[{"data": {"trades": [], "next_cursor": ""}}],
        current_date="2026-08-12",
    )
    assert result["recently_closed_lots"] == []
    assert result["needs_review"] == []
    assert result["coverage"]["merged_sells_in_window"] == 0


def test_build_wash_sale_history_conservative_unknown_basis():
    result = build_wash_sale_history_fn(
        equity_orders=[{"data": {"orders": [FRACTIONAL_BND_SELL]}}],
        current_date="2023-06-05",
        window_days=31,
        unknown_basis_policy="conservative",
    )
    assert len(result["recently_closed_lots"]) == 1
    lot = result["recently_closed_lots"][0]
    assert lot["realized_gain"] == pytest.approx(-0.01)
    assert lot["cost_basis"] == pytest.approx(lot["proceeds"] + 0.01)
    assert lot["basis_source"] == "assumed_loss"
    assert len(result["needs_review"]) == 1
    assert result["wash_sale_preview"]["buy_restricted_symbols"]


def test_build_wash_sale_history_exclude_unknown_basis():
    result = build_wash_sale_history_fn(
        equity_orders=[{"data": {"orders": [FRACTIONAL_BND_SELL]}}],
        current_date="2023-06-05",
        unknown_basis_policy="exclude",
    )
    assert result["recently_closed_lots"] == []
    assert any("excluded sell" in w for w in result["warnings"])


def test_build_wash_sale_history_pnl_row_preferred_on_dedupe():
    pnl = {
        "data": {
            "trades": [
                {
                    "symbol": "BND",
                    "quantity": "0.5",
                    "price": "80.00",
                    "realized_gain": "-12.34",
                    "date_acquired": "2022-01-01",
                    "closed_at": "2023-05-05T14:01:01Z",
                }
            ]
        }
    }
    result = build_wash_sale_history_fn(
        equity_orders=[{"data": {"orders": [FRACTIONAL_BND_SELL]}}],
        pnl_trade_history=[pnl],
        current_date="2023-06-05",
    )
    lot = result["recently_closed_lots"][0]
    assert lot["realized_gain"] == pytest.approx(-12.34)
    assert lot["date_acquired"] == "2022-01-01"
    assert lot["basis_source"] == "pnl_trade_history"


def test_build_wash_sale_history_cost_basis_hint():
    result = build_wash_sale_history_fn(
        equity_orders=[{"data": {"orders": [FRACTIONAL_BND_SELL]}}],
        current_date="2023-06-05",
        cost_basis_hints={"BND:2023-05-05": 50.0},
    )
    lot = result["recently_closed_lots"][0]
    assert lot["cost_basis"] == pytest.approx(50.0)
    assert lot["realized_gain"] == pytest.approx(lot["proceeds"] - 50.0)
    assert lot["basis_source"] == "cost_basis_hint"


def test_build_wash_sale_history_window_filter():
    result = build_wash_sale_history_fn(
        equity_orders=[{"data": {"orders": [FRACTIONAL_BND_SELL]}}],
        current_date="2023-05-05",
        window_days=31,
        cost_basis_hints={"BND": 100.0},
    )
    assert len(result["recently_closed_lots"]) == 1

    outside = build_wash_sale_history_fn(
        equity_orders=[{"data": {"orders": [FRACTIONAL_BND_SELL]}}],
        current_date="2023-07-01",
        window_days=31,
        cost_basis_hints={"BND": 100.0},
    )
    assert outside["recently_closed_lots"] == []


def test_build_wash_sale_history_preview_flags():
    result = build_wash_sale_history_fn(
        equity_orders=[{"data": {"orders": [FRACTIONAL_BND_SELL]}}],
        tax_lots=[
            {
                "data": {
                    "symbol": "AAPL",
                    "tax_lots": [
                        {
                            "open_lot_id": "lot-1",
                            "open_date": "2026-08-10",
                            "quantity": "5",
                            "tax_cost_basis": "500",
                            "is_selectable": False,
                        }
                    ],
                }
            }
        ],
        current_date="2026-08-12",
        window_days=31,
        cost_basis_hints={"BND:2023-05-05": 50.0},
    )
    preview = result["wash_sale_preview"]
    assert preview["recent_open_lots"]
    assert preview["not_selectable_lots"][0]["open_lot_id"] == "lot-1"


def test_build_wash_sale_history_unread_next_warning():
    result = build_wash_sale_history_fn(
        equity_orders=[
            {
                "data": {
                    "orders": [FRACTIONAL_BND_SELL],
                    "next": "http://example/orders?cursor=abc123",
                }
            }
        ],
        current_date="2023-06-05",
    )
    assert result["coverage"]["equity_orders_unread_next"] is True
    assert result["coverage"]["equity_orders_next_cursor"] == "abc123"
    assert any("unread pages" in w for w in result["warnings"])


def test_recently_closed_lots_passes_initialize_closed_lots():
    result = build_wash_sale_history_fn(
        pnl_trade_history=[
            {
                "data": {
                    "trades": [
                        {
                            "symbol": "TSLA",
                            "quantity": "1",
                            "price": "250",
                            "realized_gain": "-50",
                            "date_acquired": "2025-06-01",
                            "closed_at": "2025-08-01",
                        }
                    ]
                }
            }
        ],
        current_date="2025-08-12",
    )
    event = build_event(
        tax_lots=[],
        prices=[{"symbol": "TSLA", "price": 240.0}],
        targets=[{"symbol": "TSLA", "target_weight": 1.0}],
        cash=0.0,
        recently_closed_lots=result["recently_closed_lots"],
        current_date="2025-08-12",
    )
    import pandas as pd

    df = pd.DataFrame(event["oracle"]["recently_closed_lots"])
    validated = initialize_closed_lots(df)
    assert validated is not None
    assert validated.iloc[0]["realized_gain"] == pytest.approx(-50.0)


def test_build_wash_sale_history_end_to_end_optimize_portfolio():
    result = build_wash_sale_history_fn(
        pnl_trade_history=[
            {
                "data": {
                    "trades": [
                        {
                            "symbol": "SNAP",
                            "quantity": "10",
                            "price": "8",
                            "realized_gain": "-70",
                            "date_acquired": "2025-07-01",
                            "closed_at": "2025-08-01",
                        }
                    ]
                }
            }
        ],
        current_date="2025-08-12",
    )
    opt = optimize_portfolio(
        tax_lots=[
            {
                "symbol": "SNAP",
                "quantity": 10,
                "cost_basis": 150.0,
                "date_acquired": "2025-07-15",
            }
        ],
        prices=[{"symbol": "SNAP", "price": 8.0}, {"symbol": "MSFT", "price": 400.0}],
        targets=[{"symbol": "SNAP", "target_weight": 0.0}, {"symbol": "MSFT", "target_weight": 1.0}],
        cash=1000.0,
        optimization_type="TAX_AWARE",
        current_date="2025-08-12",
        recently_closed_lots=result["recently_closed_lots"],
    )
    assert "netted_trades" in opt


def test_build_wash_sale_history_mcp_tool_wrapper():
    out = build_wash_sale_history(
        pnl_trade_history=[{"data": {"trades": [], "next_cursor": ""}}],
        current_date="2026-08-12",
    )
    assert "recently_closed_lots" in out
    assert "coverage" in out


def test_workflow_guide_mentions_wash_sale_history():
    guide = get_workflow_guide()
    for needle in [
        "build_wash_sale_history",
        "rhs_account_number",
        "realized_gain: null",
        "created_at_gte",
    ]:
        assert needle in guide
