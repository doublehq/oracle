"""Test TAX_AWARE optimization specifically for Tax-Loss Harvesting."""
import unittest
from datetime import date
import pandas as pd
import numpy as np
import json # Import json for pretty printing
from src.service.oracle import Oracle
from src.service.oracle_strategy import OracleStrategy
from src.service.helpers.constants import CASH_CUSIP_ID
from src.service.helpers.enums import OracleOptimizationType
import pulp

class TestTaxAware(unittest.TestCase):
    def setUp(self):
        """Setup a portfolio mostly on target but with one significant loss position.
        
        Initial State:
        - STOCK_A: $50k position (25%), cost basis $50k (Flat)
        - STOCK_B: $50k position (25%), cost basis $50k (Flat)
        - STOCK_TLH: $40k position (20%), cost basis $60k (Significant Loss)
        - STOCK_GAIN: $55k position (27.5%), cost basis $50k (Small Gain, slightly overweight)
        - CASH: $5k (2.5%)
        Total Portfolio: $200k
        Target Weights: A=25%, B=25%, TLH=25%, GAIN=25%, CASH=0%
        Drift: TLH is underweight, GAIN is overweight. A & B are on target.
        """
        self.current_date = date(2024, 4, 2)
        self.tax_rates = pd.DataFrame([
            {'gain_type': 'short_term', 'federal_rate': 0.35, 'state_rate': 0.06, 'total_rate': 0.41},
            {'gain_type': 'long_term', 'federal_rate': 0.20, 'state_rate': 0.06, 'total_rate': 0.26},
            {"gain_type": "qualified_dividend", "federal_rate": 0.15, "state_rate": 0.06, "total_rate": 0.21}
        ])
        self.oracle = Oracle(current_date=self.current_date, tax_rates=self.tax_rates)
        
        self.tax_lots = pd.DataFrame([
            {'tax_lot_id': 'lot_a', 'identifier': 'STOCK_A', 'quantity': 1000, 'cost_basis': 50000, 'date': '2024-01-01'},
            {'tax_lot_id': 'lot_b', 'identifier': 'STOCK_B', 'quantity': 500, 'cost_basis': 50000, 'date': '2024-01-01'},
            {'tax_lot_id': 'lot_tlh', 'identifier': 'STOCK_TLH', 'quantity': 1000, 'cost_basis': 60000, 'date': '2024-01-01'}, # Cost $60/share
            {'tax_lot_id': 'lot_gain', 'identifier': 'STOCK_GAIN', 'quantity': 500, 'cost_basis': 50000, 'date': '2024-01-01'} # Cost $100/share
        ])
        self.targets = pd.DataFrame([
            {'asset_class': 'STOCK_A', 'identifiers': ['STOCK_A'], 'target_weight': 0.25},
            {'asset_class': 'STOCK_B', 'identifiers': ['STOCK_B'], 'target_weight': 0.25}, 
            {'asset_class': 'STOCK_TLH', 'identifiers': ['STOCK_TLH'], 'target_weight': 0.25},
            {'asset_class': 'STOCK_GAIN', 'identifiers': ['STOCK_GAIN'], 'target_weight': 0.25},
            {'asset_class': CASH_CUSIP_ID, 'identifiers': [CASH_CUSIP_ID], 'target_weight': 0.0}
        ])
        self.prices = pd.DataFrame([
            {'identifier': 'STOCK_A', 'price': 50.0},
            {'identifier': 'STOCK_B', 'price': 100.0},
            {'identifier': 'STOCK_TLH', 'price': 40.0}, # Current price $40, significant loss
            {'identifier': 'STOCK_GAIN', 'price': 110.0},# Current price $110, small gain
            {'identifier': CASH_CUSIP_ID, 'price': 1.0}
        ])
        self.spreads = pd.DataFrame([
            {'identifier': 'STOCK_A', 'spread': 0.001},
            {'identifier': 'STOCK_B', 'spread': 0.001},
            {'identifier': 'STOCK_TLH', 'spread': 0.001},
            {'identifier': 'STOCK_GAIN', 'spread': 0.001},
            {'identifier': CASH_CUSIP_ID, 'spread': 0.0}
        ])
        
        self.base_strategy_data = {
            'tax_lots': self.tax_lots,
            'targets': self.targets,
            'prices': self.prices,
            'spreads': self.spreads,
            'cash': 5000.0 # Start with $5k cash
        }

    def test_tax_aware_prioritizes_tlh(self):
        """Verify TAX_AWARE sells the significant loss position for TLH."""
        strategy = OracleStrategy(
            **self.base_strategy_data,
            optimization_type=OracleOptimizationType.TAX_AWARE
        )
        strategy.set_oracle(self.oracle)
        self.oracle.strategies = [strategy]

        # Run optimization
        status, should_trade, trade_summary, trades = strategy.compute_optimal_trades(
            weight_tax=1.0,
            weight_drift=0.0001,
            weight_transaction=0.0,
            weight_cash_drag=100,
            debug=True,
        )
        
        # Verify optimization completed successfully
        self.assertEqual(pulp.LpStatusOptimal, status)
        self.assertTrue(should_trade)

        self.assertTrue(len(trades) > 0, "Expected trades to occur for TLH")

        # Find TLH sells
        tlh_sells = trades[
            (trades['identifier'] == 'STOCK_TLH') &
            (trades['action'] == 'sell')
        ]

        # Find gain sells
        gain_sells = trades[
            (trades['identifier'] == 'STOCK_GAIN') &
            (trades['action'] == 'sell')
        ]

        # Calculate sell values
        tlh_sell_value = tlh_sells.apply(lambda x: x['quantity'] * x['price'], axis=1).sum()
        gain_sell_value = gain_sells.apply(lambda x: x['quantity'] * x['price'], axis=1).sum()

        # Verify TLH sells are larger than gain sells
        self.assertGreater(tlh_sell_value, gain_sell_value)

        # Verify buys match sells
        buy_trades = trades[trades['action'] == 'buy']

        # Calculate total buy value
        total_buy_value = buy_trades.apply(lambda x: x['quantity'] * x['price'], axis=1).sum()
        self.assertTrue(len(buy_trades) > 0, "Expected buy trades to reallocate proceeds or reduce drift") 
        
        # Check if cash was deployed
        initial_cash = self.base_strategy_data['cash']
        total_sell_value = tlh_sell_value + gain_sell_value
        net_trade_cash_flow = total_sell_value - total_buy_value
        final_cash_estimate = initial_cash + net_trade_cash_flow
        
        print(f"Initial Cash: {initial_cash:.2f}")
        print(f"Total Sell Value: {total_sell_value:.2f}")
        print(f"Total Buy Value: {total_buy_value:.2f}")
        print(f"Estimated Final Cash: {final_cash_estimate:.2f}")
        
        # Assert that most of the available cash (initial + sell proceeds) was used
        # Allow for a small residual cash balance due to trade-offs/costs
        # Let's assert final cash is less than 1% of initial portfolio value or initial cash, whichever is smaller
        portfolio_value = sum(self.prices[self.prices['identifier'] != CASH_CUSIP_ID]['price'] * self.tax_lots.set_index('identifier').loc[self.prices[self.prices['identifier'] != CASH_CUSIP_ID]['identifier']]['quantity']) + initial_cash
        tolerance = min(initial_cash, portfolio_value * 0.01) 
        self.assertLess(final_cash_estimate, tolerance, 
                        f"Expected final cash ({final_cash_estimate:.2f}) to be near zero (tolerance: {tolerance:.2f}), indicating cash deployment.") 
        

    def test_empty_portfolio_deploys_cash(self):
        """Verify that an empty portfolio with only cash buys into targets."""
        # Override setup for this test: empty lots, full cash
        initial_cash_value = 200000.0
        empty_tax_lots = pd.DataFrame(columns=self.tax_lots.columns)
        
        strategy = OracleStrategy(
            tax_lots=empty_tax_lots,
            targets=self.targets, # Use targets from setup
            prices=self.prices,   # Use prices from setup
            spreads=self.spreads, # Use spreads from setup
            cash=initial_cash_value,
            optimization_type=OracleOptimizationType.TAX_AWARE # Or TAX_UNAWARE
        )
        strategy.set_oracle(self.oracle)
        self.oracle.strategies = [strategy]

        # Run optimization
        status, should_trade, trade_summary, trades = strategy.compute_optimal_trades(
            weight_tax=1.0,
            weight_drift=1.0,
            weight_transaction=1.0,
            weight_cash_drag=1.0,
            debug=True,
        )
        
        # Verify optimization completed successfully
        self.assertEqual(pulp.LpStatusOptimal, status)
        self.assertTrue(should_trade)

        self.assertTrue(len(trades) > 0, "Expected trades to deploy initial cash")

        # Get sell and buy trades
        sell_trades = trades[trades['action'] == 'sell']
        buy_trades = trades[trades['action'] == 'buy']

        # Get identifiers that were bought
        bought_identifiers = set(buy_trades['identifier'])

        # Verify we bought back TLH stock
        self.assertIn('STOCK_TLH', bought_identifiers)

        # Calculate total buy value
        total_buy_value = buy_trades.apply(lambda x: x['quantity'] * x['price'], axis=1).sum()

        # Check if cash was deployed
        final_cash_estimate = initial_cash_value - total_buy_value
        
        print(f"Initial Cash: {initial_cash_value:.2f}")
        print(f"Total Buy Value: {total_buy_value:.2f}")
        print(f"Estimated Final Cash: {final_cash_estimate:.2f}")

        # Assert cash is near zero (allowing for transaction costs/small residuals)
        tolerance = initial_cash_value * 0.01 # Allow 1% residual cash 
        self.assertLess(final_cash_estimate, tolerance,
                        f"Expected final cash ({final_cash_estimate:.2f}) to be near zero (tolerance: {tolerance:.2f}) after deploying initial cash.") 