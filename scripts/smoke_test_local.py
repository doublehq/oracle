#!/usr/bin/env python3
"""Local smoke test: Oracle MCP tools + Robinhood-shaped inputs.

Run from repo root with the project venv active:

    source .venv/bin/activate
    python scripts/smoke_test_local.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mcp_server.server import (  # noqa: E402
    build_robinhood_equity_orders,
    get_workflow_guide,
    normalize_robinhood_tax_lots,
    optimize_portfolio,
)


AGENTIC_ACCOUNT = "TEST_ACCOUNT_001"

RH_TAX_LOTS_RESPONSE = {
    "data": {
        "symbol": "AAPL",
        "tax_lots": [
            {
                "open_lot_id": "11111111-aaaa-bbbb-cccc-000000000001",
                "open_date": "2020-01-15",
                "quantity": "10",
                "tax_cost_basis": "500.00",
                "term": "long",
                "is_selectable": True,
            },
            {
                "open_lot_id": "11111111-aaaa-bbbb-cccc-000000000002",
                "open_date": "2026-06-01",
                "quantity": "10",
                "tax_cost_basis": "2500.00",
                "term": "short",
                "is_selectable": True,
            },
        ],
    }
}


def main() -> None:
    print("=== Oracle local smoke test ===\n")

    tax_lots = normalize_robinhood_tax_lots([RH_TAX_LOTS_RESPONSE])
    print("1) Normalized Robinhood tax lots:")
    print(json.dumps(tax_lots, indent=2))

    result = optimize_portfolio(
        tax_lots=tax_lots,
        prices=[{"symbol": "AAPL", "price": 200.0}],
        targets=[
            {"symbol": "AAPL", "target_weight": 0.5},
            {"symbol": "CASH", "target_weight": 0.5},
        ],
        cash=0.0,
        optimization_type="TAX_AWARE",
        current_date="2026-08-12",
        tax_rates={"short_term": 0.37, "long_term": 0.20},
        account_number=AGENTIC_ACCOUNT,
    )

    print("\n2) Oracle optimize_portfolio (summary):")
    print(f"   should_trade: {result['results']['1']['should_trade']}")
    print(f"   netted_trades: {len(result['netted_trades'])}")
    print(f"   lot-level trades: {len(result['results']['1']['trades'])}")
    print(f"   execution_mode: {result['execution_impact'].get('sells', [{}])[0].get('execution_mode', 'n/a') if result['execution_impact']['sells'] else 'no sells'}")

    lot_trades = result["results"]["1"]["trades"]
    if lot_trades:
        print("\n3) Lot-level sells:")
        for t in lot_trades:
            if t.get("trade_type") == "SELL":
                print(f"   {t['symbol']} qty={t['quantity']} open_lot_id={t.get('open_lot_id')}")

    rh = result.get("robinhood_orders") or build_robinhood_equity_orders(
        account_number=AGENTIC_ACCOUNT,
        trades=lot_trades,
    )
    print("\n4) Robinhood review_equity_order payloads (dry-run):")
    print(json.dumps(rh["review_equity_orders"], indent=2))

    assert "open_lot_id" in get_workflow_guide()
    assert rh["review_equity_orders"], "expected at least one review order when sells exist"
    first = rh["review_equity_orders"][0]
    if first["side"] == "sell":
        assert "tax_lots" in first, "sell should include specified tax_lots"
        assert all("open_lot_id" in lot for lot in first["tax_lots"])

    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
