"""Test optimization behavior for different OracleOptimizationType settings."""
import unittest
from datetime import date
import pandas as pd
import numpy as np
from src.service.oracle import Oracle
from src.service.oracle_strategy import OracleStrategy
from src.service.helpers.constants import CASH_CUSIP_ID
from src.service.helpers.enums import OracleOptimizationType
import pulp

class TestOptimizationTypes(unittest.TestCase):
    def setUp(self):
        """Setup a two-stock portfolio requiring rebalancing and with tax implications.
        
        Initial State:
        - STOCK_A: $120k position (60%), cost basis $100k (Gain)
        - STOCK_B: $80k position (40%), cost basis $100k (Loss)
        Total Portfolio: $200k
        Target Weights: 50-50
        Cash: $0
        """
        self.current_date = date(2024, 4, 2)
        self.tax_rates = pd.DataFrame([
            {'gain_type': 'short_term', 'federal_rate': 0.35, 'state_rate': 0.06, 'total_rate': 0.41},
            {'gain_type': 'long_term', 'federal_rate': 0.20, 'state_rate': 0.06, 'total_rate': 0.26},
            {"gain_type": "qualified_dividend", "federal_rate": 0.15, "state_rate": 0.06, "total_rate": 0.21}
        ])
        self.oracle = Oracle(current_date=self.current_date, tax_rates=self.tax_rates)
        
        self.tax_lots = pd.DataFrame([
            {'tax_lot_id': 'lot_a1', 'identifier': 'STOCK_A', 'quantity': 1000, 'cost_basis': 100000, 'date': '2024-01-01'},
            {'tax_lot_id': 'lot_b1', 'identifier': 'STOCK_B', 'quantity': 1000, 'cost_basis': 100000, 'date': '2024-01-01'}
        ])
        self.targets = pd.DataFrame([
            {'asset_class': 'STOCK_A', 'identifiers': ['STOCK_A'], 'target_weight': 0.5},
            {'asset_class': 'STOCK_B', 'identifiers': ['STOCK_B'], 'target_weight': 0.5},
            {'asset_class': CASH_CUSIP_ID, 'identifiers': [CASH_CUSIP_ID], 'target_weight': 0.0}
        ])
        self.prices = pd.DataFrame([
            {'identifier': 'STOCK_A', 'price': 120.0},
            {'identifier': 'STOCK_B', 'price': 80.0},
            {'identifier': CASH_CUSIP_ID, 'price': 1.0}
        ])
        self.spreads = pd.DataFrame([
            {'identifier': 'STOCK_A', 'spread': 0.001},
            {'identifier': 'STOCK_B', 'spread': 0.001},
            {'identifier': CASH_CUSIP_ID, 'spread': 0.0}
        ])
        
        # Base strategy created here, optimization_type modified in each test
        self.base_strategy_data = {
            'tax_lots': self.tax_lots,
            'targets': self.targets,
            'prices': self.prices,
            'spreads': self.spreads,
            'cash': 0.0
        }
        self.oracle.strategies = [] # Reset strategies for each test
        self.compute_args = {
            'weight_tax': 1.0,
            'weight_drift': 1.0,
            'weight_transaction': 1.0,
            'weight_cash_drag': 0.5,
            'rebalance_threshold': 0.0001,
            'debug': False # Keep debug off for cleaner test output unless needed
        }

    def _create_and_run_strategy(self, opt_type: OracleOptimizationType, **kwargs):
        """Helper to create strategy with specific type and run optimization."""
        strategy = OracleStrategy(
            **self.base_strategy_data,
            optimization_type=opt_type
        )
        strategy.set_oracle(self.oracle)
        self.oracle.strategies = [strategy]
        
        # Default weights can be overridden by kwargs
        compute_args = {
            'weight_tax': 1.0,
            'weight_drift': 1.0,
            'weight_transaction': 1.0, # Small cost 
            'weight_cash_drag': 0.5,
            'rebalance_threshold': 0.0001,
            'debug': False # Keep debug off for cleaner test output unless needed
        }
        compute_args.update(kwargs)
        # Return the full tuple from compute_optimal_trades
        status, should_trade, trade_summary, trades = strategy.compute_optimal_trades(**compute_args)
        
        # For HOLD type, verify no trades
        if opt_type == OracleOptimizationType.HOLD:
            self.assertFalse(should_trade)
            self.assertTrue(trades.empty)
            return status, should_trade, trade_summary, trades
            
        # For other types, verify trades exist if should_trade is True
        if should_trade:
            self.assertFalse(trades.empty)
        
        return status, should_trade, trade_summary, trades

    def test_hold_type_no_trades(self):
        """Verify HOLD optimization type results in zero trades."""
        # Unpack tuple
        status, should_trade, trade_summary, trades = self._create_and_run_strategy(OracleOptimizationType.HOLD)
        self.assertEqual(len(trades), 0, "HOLD strategy should generate no trades")

    def test_buy_only_type_no_sells(self):
        """Verify BUY_ONLY optimization type generates no sell trades."""
        # Need cash to actually buy
        self.base_strategy_data['cash'] = 20000 
        # Unpack tuple
        status, should_trade, trade_summary, trades = self._create_and_run_strategy(OracleOptimizationType.BUY_ONLY)
        
        # Check status indicates success (if trades were made)
        if not trades.empty:
            self.assertEqual(status, pulp.LpStatusOptimal, "Optimization status should be Optimal if buys occurred")
        else:
            # If no trades, status could be Optimal or potentially another state if setup prevents buys
            pass # Allow flexibility if no trades are possible
            
        sell_trades = trades[trades['action'] == 'sell']
        self.assertEqual(len(sell_trades), 0, "BUY_ONLY strategy should generate no sell trades")
        
        # It should still buy to reduce underweight if cash allows
        buy_trades = trades[trades['action'] == 'buy']
        # In this setup, STOCK_B is underweight, so we expect buys
        self.assertTrue(len(buy_trades) > 0, "BUY_ONLY should still generate buy trades if possible and beneficial")
        stock_b_buys = buy_trades[buy_trades['identifier'] == 'STOCK_B']
        self.assertTrue(len(stock_b_buys) > 0, "BUY_ONLY should buy the underweight asset (STOCK_B)")

    def test_tax_unaware_type_ignores_tax(self):
        """Verify TAX_UNAWARE ignores tax implications and rebalances aggressively."""
        # Unpack tuple
        status, should_trade, trade_summary, trades = self._create_and_run_strategy(OracleOptimizationType.TAX_UNAWARE)
        self.assertEqual(status, pulp.LpStatusOptimal, "Optimization status should be Optimal")
        
        # Should sell STOCK_A (overweight with gain) despite the tax cost
        stock_a_sells = trades[
            (trades['identifier'] == 'STOCK_A') & 
            (trades['action'] == 'sell')
        ]
        self.assertTrue(len(stock_a_sells) > 0, "TAX_UNAWARE should sell overweight STOCK_A even with gains")
        
        # Should buy STOCK_B (underweight)
        stock_b_buys = trades[
            (trades['identifier'] == 'STOCK_B') & 
            (trades['action'] == 'buy')
        ]
        self.assertTrue(len(stock_b_buys) > 0, "TAX_UNAWARE should buy underweight STOCK_B")
