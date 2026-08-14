import pandas as pd

from src.service.helpers.trade_applier import apply_trades_to_portfolio
from src.service.helpers.trade_netting import net_trades_across_strategies


def test_net_trades_preserves_sell_direction_with_pandas_copy_on_write():
    sell = pd.DataFrame(
        [
            {
                "identifier": "AAPL",
                "action": "sell",
                "quantity": 2,
                "price": 100.0,
                "tax_lot_id": "lot-1",
                "gain_loss": {"realized_gain": -20.0, "gain_type": "short_term"},
            }
        ]
    )

    netted = net_trades_across_strategies(
        {1: (1, True, {}, sell)},
        trade_rounding=4,
    )

    assert netted.to_dict(orient="records")[0]["action"] == "sell"
    assert netted.to_dict(orient="records")[0]["quantity"] == 2.0


def test_apply_fractional_sell_to_integer_quantity_portfolio():
    tax_lots = pd.DataFrame(
        [
            {
                "tax_lot_id": "lot-1",
                "identifier": "AAPL",
                "quantity": 10,
                "cost_basis": 500,
                "date": pd.Timestamp("2024-01-01"),
            }
        ]
    )
    trades = pd.DataFrame(
        [
            {
                "identifier": "AAPL",
                "action": "sell",
                "quantity": 1.5,
                "price": 100.0,
                "tax_lot_id": "lot-1",
            }
        ]
    )

    updated_lots, updated_cash, _ = apply_trades_to_portfolio(
        tax_lots,
        trades,
        cash=0.0,
        current_date=pd.Timestamp("2026-08-12"),
    )

    assert updated_lots.iloc[0]["quantity"] == 8.5
    assert updated_cash == 150.0
