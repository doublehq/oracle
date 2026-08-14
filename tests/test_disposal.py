"""Disposal-method simulation and execution-impact tests."""

import inspect

import pytest

from src.mcp_server.server import optimize_portfolio
from src.service.oracle_strategy import OracleStrategy
from src.service.helpers.disposal import (
    lots_in_disposal_order,
    normalize_disposal_method,
    reconcile_execution,
    simulate_disposal,
)


LOTS = [
    {
        "tax_lot_id": "old_cheap",
        "identifier": "AAPL",
        "quantity": 10,
        "cost_basis": 500.0,  # $50/share
        "date": "2020-01-15",
    },
    {
        "tax_lot_id": "new_expensive",
        "identifier": "AAPL",
        "quantity": 10,
        "cost_basis": 2500.0,  # $250/share
        "date": "2026-06-01",
    },
]


def test_new_disposal_options_do_not_shift_existing_positional_parameters():
    parameter_names = list(inspect.signature(OracleStrategy.compute_optimal_trades).parameters)
    assert parameter_names.index("debug") < parameter_names.index("disposal_method")
    assert parameter_names.index("log_time") < parameter_names.index("disposal_method")


def test_normalize_disposal_method_aliases_and_default():
    assert normalize_disposal_method(None) == "FIFO"
    assert normalize_disposal_method("hifo") == "HIFO"
    assert normalize_disposal_method("highest_cost") == "HIFO"
    with pytest.raises(ValueError):
        normalize_disposal_method("AVERAGE")


def test_fifo_sells_oldest_lot_first():
    ordered = lots_in_disposal_order(LOTS, "FIFO")
    assert [lot["tax_lot_id"] for lot in ordered] == ["old_cheap", "new_expensive"]
    consumed, leftover = simulate_disposal(LOTS, 10, "FIFO")
    assert leftover == 0
    assert len(consumed) == 1
    assert consumed[0]["tax_lot_id"] == "old_cheap"
    assert consumed[0]["quantity"] == pytest.approx(10)


def test_lifo_sells_newest_lot_first():
    consumed, leftover = simulate_disposal(LOTS, 10, "LIFO")
    assert leftover == 0
    assert consumed[0]["tax_lot_id"] == "new_expensive"


def test_hifo_sells_highest_cost_first():
    consumed, leftover = simulate_disposal(LOTS, 10, "HIFO")
    assert leftover == 0
    assert consumed[0]["tax_lot_id"] == "new_expensive"


def test_partial_then_next_lot():
    consumed, leftover = simulate_disposal(LOTS, 15, "FIFO")
    assert leftover == 0
    assert consumed[0]["tax_lot_id"] == "old_cheap"
    assert consumed[0]["quantity"] == pytest.approx(10)
    assert consumed[1]["tax_lot_id"] == "new_expensive"
    assert consumed[1]["quantity"] == pytest.approx(5)


def test_reconcile_flags_divergence_when_oracle_picks_the_other_lot():
    # Oracle harvests the high-cost loser; FIFO would close the old cheap winner.
    oracle_trades = [
        {
            "identifier": "AAPL",
            "action": "sell",
            "quantity": 10,
            "tax_lot_id": "new_expensive",
        }
    ]
    impact = reconcile_execution(
        tax_lots=LOTS,
        prices=[{"identifier": "AAPL", "price": 200.0}],
        current_date="2026-08-12",
        tax_rates={"short_term": 0.37, "long_term": 0.20},
        trades=oracle_trades,
        disposal_method="FIFO",
        use_specified_tax_lots=False,
    )
    assert impact["any_divergence"] is True
    sell = impact["sells"][0]
    assert sell["diverges"] is True
    assert sell["lots_broker_will_dispose"][0]["tax_lot_id"] == "old_cheap"
    assert sell["tax_cost_delta"] != 0
    assert any("different lots" in w.lower() or "fifo" in w.lower() for w in sell["warnings"])


def test_reconcile_specified_lot_mode_matches_oracle():
    oracle_trades = [
        {
            "identifier": "AAPL",
            "action": "sell",
            "quantity": 10,
            "open_lot_id": "new_expensive",
            "tax_lot_id": "new_expensive",
        }
    ]
    impact = reconcile_execution(
        tax_lots=LOTS,
        prices=[{"identifier": "AAPL", "price": 200.0}],
        current_date="2026-08-12",
        tax_rates={"short_term": 0.37, "long_term": 0.20},
        trades=oracle_trades,
        disposal_method="FIFO",
        use_specified_tax_lots=True,
    )
    assert impact["any_divergence"] is False
    assert impact["sells"][0]["execution_mode"] == "specified_lot"


def test_tax_term_uses_calendar_anniversary_across_leap_year():
    impact = reconcile_execution(
        tax_lots=[
            {
                "tax_lot_id": "leap-boundary",
                "open_lot_id": "leap-boundary",
                "identifier": "AAPL",
                "quantity": 1,
                "cost_basis": 100.0,
                "date": "2023-03-01",
            }
        ],
        prices=[{"identifier": "AAPL", "price": 120.0}],
        current_date="2024-03-01",
        tax_rates={"short_term": 0.37, "long_term": 0.20},
        trades=[
            {
                "identifier": "AAPL",
                "action": "sell",
                "quantity": 1,
                "open_lot_id": "leap-boundary",
                "tax_lot_id": "leap-boundary",
            }
        ],
        disposal_method="FIFO",
        use_specified_tax_lots=True,
    )

    sell = impact["sells"][0]
    assert sell["oracle_short_term_realized"] == pytest.approx(20.0)


def test_reconcile_no_divergence_when_oracle_matches_fifo():
    oracle_trades = [
        {
            "identifier": "AAPL",
            "action": "sell",
            "quantity": 10,
            "tax_lot_id": "old_cheap",
        }
    ]
    impact = reconcile_execution(
        tax_lots=LOTS,
        prices=[{"identifier": "AAPL", "price": 200.0}],
        current_date="2026-08-12",
        tax_rates={"short_term": 0.37, "long_term": 0.20},
        trades=oracle_trades,
        disposal_method="FIFO",
    )
    assert impact["any_divergence"] is False
    assert impact["sells"][0]["diverges"] is False
    assert impact["sells"][0]["tax_cost_delta"] == pytest.approx(0.0)


def test_reconcile_rejects_non_fifo_broker_fallback():
    with pytest.raises(ValueError, match="disposal is FIFO"):
        reconcile_execution(
            tax_lots=LOTS,
            prices=[{"identifier": "AAPL", "price": 200.0}],
            current_date="2026-08-12",
            tax_rates={"short_term": 0.37, "long_term": 0.20},
            trades=[
                {
                    "identifier": "AAPL",
                    "action": "sell",
                    "quantity": 10,
                    "tax_lot_id": "new_expensive",
                }
            ],
            disposal_method="LIFO",
            use_specified_tax_lots=False,
        )


def test_disposal_constraints_accept_lot_ids_that_sanitize_to_same_text():
    result = optimize_portfolio(
        tax_lots=[
            {
                "symbol": "AAPL",
                "tax_lot_id": "lot-a",
                "quantity": 5,
                "cost_basis": 500.0,
                "date_acquired": "2020-01-15",
            },
            {
                "symbol": "AAPL",
                "tax_lot_id": "lot_a",
                "quantity": 5,
                "cost_basis": 500.0,
                "date_acquired": "2021-01-15",
            },
        ],
        prices=[{"symbol": "AAPL", "price": 100.0}],
        targets=[
            {"symbol": "AAPL", "target_weight": 0.5},
            {"symbol": "CASH", "target_weight": 0.5},
        ],
        cash=0.0,
        optimization_type="TAX_AWARE",
        current_date="2026-08-12",
        enforce_disposal_order=True,
    )

    assert result["results"]["1"]["status"] is not None


def test_enforce_disposal_order_prevents_selling_later_lot_first():
    """With FIFO enforced, Oracle cannot harvest the new high-cost lot without
    first exhausting the old cheap lot — so a loss-only sell of the new lot
    should not appear as a standalone recommendation.
    """
    tax_lots = [
        {
            "symbol": "AAPL",
            "tax_lot_id": "old_cheap",
            "quantity": 10,
            "cost_basis": 500.0,
            "date_acquired": "2020-01-15",
        },
        {
            "symbol": "AAPL",
            "tax_lot_id": "new_expensive",
            "quantity": 10,
            "cost_basis": 2500.0,
            "date_acquired": "2026-06-01",
        },
    ]
    prices = [{"symbol": "AAPL", "price": 200.0}]
    # 50% cash target forces a ~10-share sell. Tax-aware prefers the losing new lot;
    # FIFO would close the old cheap winner first.
    targets = [
        {"symbol": "AAPL", "target_weight": 0.5},
        {"symbol": "CASH", "target_weight": 0.5},
    ]

    unconstrained = optimize_portfolio(
        tax_lots=tax_lots,
        prices=prices,
        targets=targets,
        cash=0.0,
        optimization_type="TAX_AWARE",
        current_date="2026-08-12",
        tax_rates={"short_term": 0.37, "long_term": 0.20},
        disposal_method="FIFO",
        enforce_disposal_order=False,
        use_specified_tax_lots=False,
    )
    unconstrained_sells = [
        t for t in unconstrained["results"]["1"]["trades"] if t["trade_type"] == "SELL"
    ]

    constrained = optimize_portfolio(
        tax_lots=tax_lots,
        prices=prices,
        targets=targets,
        cash=0.0,
        optimization_type="TAX_AWARE",
        current_date="2026-08-12",
        tax_rates={"short_term": 0.37, "long_term": 0.20},
        disposal_method="FIFO",
        enforce_disposal_order=True,
    )
    constrained_sells = [
        t for t in constrained["results"]["1"]["trades"] if t["trade_type"] == "SELL"
    ]

    # Unconstrained tax-aware should prefer the losing new lot if it sells at all.
    if unconstrained_sells:
        sold_ids = {t["tax_lot_id"] for t in unconstrained_sells}
        # Divergence vs FIFO is the interesting unconstrained case
        if "new_expensive" in sold_ids and "old_cheap" not in sold_ids:
            assert unconstrained["execution_impact"]["any_divergence"] is True

    # Constrained sells must follow FIFO: cannot sell new_expensive unless old_cheap is fully sold.
    sold_by_id = {t["tax_lot_id"]: t["quantity"] for t in constrained_sells}
    new_qty = sold_by_id.get("new_expensive", 0.0)
    old_qty = sold_by_id.get("old_cheap", 0.0)
    if new_qty > 1e-6:
        assert old_qty == pytest.approx(10.0, abs=1e-4)
    assert constrained["execution_impact"]["any_divergence"] is False
    assert constrained["use_specified_tax_lots"] is True
