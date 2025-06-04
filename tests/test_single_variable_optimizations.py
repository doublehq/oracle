"""Test optimization behavior with a simple two-stock portfolio."""
import unittest
from datetime import date
import pandas as pd
import numpy as np
from src.service.oracle import Oracle
from src.service.oracle_strategy import OracleStrategy
from src.service.helpers.constants import CASH_CUSIP_ID
from src.service.helpers.enums import OracleOptimizationType
import pulp

class TestSingleVariableOptimizations(unittest.TestCase):
    def setUp(self):
        """Setup a two-stock portfolio with gains and losses.
        
        Initial State:
        - STOCK_A: $120k position (60%), cost basis $80k
            - Currently at gain ($40k unrealized gain)
            - Price: $100/share, 1200 shares
            - Cost basis: $80/share
            
        - STOCK_B: $80k position (40%), cost basis $120k
            - Currently at loss ($40k unrealized loss)
            - Price: $100/share, 800 shares
            - Cost basis: $120/share
            
        Total Portfolio: $200k
        Target Weights: 50-50
        """
        # Setup test data
        self.current_date = date(2024, 4, 2)
        
        # Create tax rates data
        self.tax_rates = pd.DataFrame([
            {
                'gain_type': 'short_term',
                'federal_rate': 0.35,
                'state_rate': 0.06,
                'total_rate': 0.41
            },
            {
                'gain_type': 'long_term',
                'federal_rate': 0.20,
                'state_rate': 0.06,
                'total_rate': 0.26
            },
            {
                "gain_type": "qualified_dividend",
                "federal_rate": 0.15,
                "state_rate": 0.06,
                "total_rate": 0.21
            }   
        ])
        
        # Initialize Oracle
        self.oracle = Oracle(
            current_date=self.current_date,
            tax_rates=self.tax_rates
        )
        
        # Create strategy data
        self.tax_lots = pd.DataFrame([
            {
                'tax_lot_id': 'lot_a1',
                'identifier': 'STOCK_A',
                'quantity': 1000,  
                'cost_basis': 100000,  
                'date': '2024-01-01'
            },
            {
                'tax_lot_id': 'lot_b1',
                'identifier': 'STOCK_B',
                'quantity': 1000,  
                'cost_basis': 100000, 
                'date': '2024-01-01'
            }
        ])
        
        self.targets = pd.DataFrame([
            {
                'asset_class': 'ASSET_CLASS_A',
                'identifiers': ['STOCK_A'],
                'target_weight': 0.5
            },
            {
                'asset_class': 'ASSET_CLASS_B',
                'identifiers': ['STOCK_B'],
                'target_weight': 0.5
            }
        ])
        
        self.prices = pd.DataFrame([
            {
                'identifier': 'STOCK_A',
                'price': 120.0,
            },
            {
                'identifier': 'STOCK_B',
                'price': 80.0,
            },
            {
                'identifier': CASH_CUSIP_ID,
                'price': 1.0,
            }
        ])
        
        self.spreads = pd.DataFrame([
            {
                'identifier': 'STOCK_A',
                'spread': 0.001  # 10 basis points
            },
            {
                'identifier': 'STOCK_B',
                'spread': 0.001  # 10 basis points
            },
            {
                'identifier': CASH_CUSIP_ID,
                'spread': 0.0  # No spread for cash
            }
        ])
        
        # Create OracleStrategy instance
        self.strategy = OracleStrategy(
            tax_lots=self.tax_lots,
            targets=self.targets,
            prices=self.prices,
            spreads=self.spreads,
            cash=0.0,
            optimization_type=OracleOptimizationType.TAX_AWARE,
            deminimus_cash_target=0.0001
        )
        
        # Set Oracle reference and add strategy
        self.strategy.set_oracle(self.oracle)
        self.oracle.strategies = [self.strategy]

    def test_tax_only_optimization(self):
        """Test optimization with only tax impact (should sell losing positions and re-deploy)."""
        status, should_trade, trade_summary, trades = self.strategy.compute_optimal_trades(
            weight_tax=1.0,
            weight_drift=0.0,
            weight_transaction=0.0,
            weight_cash_drag=1.0,
            debug=True
        )
        
        # Verify trades exist
        self.assertTrue(len(trades) > 0, "Expected at least one trade")
        
        # We expect to sell STOCK_B (the losing position) completely
        stock_b_sells = trades[
            (trades['identifier'] == 'STOCK_B') & 
            (trades['action'] == 'sell')
        ]
        
        self.assertTrue(len(stock_b_sells) > 0, "Expected sells for STOCK_B")
        total_stock_b_shares_sold = stock_b_sells['quantity'].sum()
        self.assertAlmostEqual(total_stock_b_shares_sold, 1000, delta=0.01, msg="Expected to sell all 800 shares of STOCK_B")
        
        # We don't expect any trades for STOCK_A
        stock_a_trades = trades[trades['identifier'] == 'STOCK_A']
        self.assertGreater(len(stock_a_trades), 0, "Expected trades for STOCK_A")

    def test_transaction_only_optimization(self):
        """Test optimization with only transaction costs (should do nothing)."""
        status, should_trade, trade_summary, trades = self.strategy.compute_optimal_trades(
            weight_tax=0.0,
            weight_drift=0.0,
            weight_transaction=1.0,
            rebalance_threshold=0.0001,
            buy_threshold=0.00005,
            debug=True
        )
        
        # Verify no trades are generated
        self.assertEqual(len(trades), 0, "Expected no trades when only considering transaction costs")

    def test_drift_only_optimization(self):
        """Test optimization with only drift impact (should rebalance to target weights)."""
        status, should_trade, trade_summary, trades = self.strategy.compute_optimal_trades(
            weight_tax=0.0,
            weight_drift=1.0,
            weight_transaction=0.0,
            debug=True
        )
        
        # Verify trades exist
        self.assertTrue(len(trades) > 0, "Expected at least one trade")
        
        # Calculate total value of sells for STOCK_A (overweight)
        stock_a_sells = trades[
            (trades['identifier'] == 'STOCK_A') & 
            (trades['action'] == 'sell')
        ]
        total_stock_a_sell_value = stock_a_sells["trade_value"].sum()
        
        # Calculate total value of buys for STOCK_B (underweight)
        stock_b_buys = trades[
            (trades['identifier'] == 'STOCK_B') & 
            (trades['action'] == 'buy')
        ]
        total_stock_b_buy_value = stock_b_buys["trade_value"].sum()
        
        # We expect approximately $20k of sells in STOCK_A (to reduce from 60% to 50%)
        self.assertAlmostEqual(total_stock_a_sell_value, 20000, delta=100)
        
        # We expect slightly less than $20k of buys in STOCK_B (due to transaction costs)
        self.assertAlmostEqual(total_stock_b_buy_value, 20000, delta=100)
        
        # Verify the sell amount is slightly higher than the buy amount
        self.assertGreater(total_stock_a_sell_value, total_stock_b_buy_value)

    def test_buy_only_threshold(self):
        """Test that buy-only trades occur when rebalance threshold is high but buy threshold is low."""
        self.strategy.cash = 10000.0  # Add cash to allow for buy trades
        status, should_trade, trade_summary, trades = self.strategy.compute_optimal_trades(
            weight_tax=1.0,
            weight_drift=1.0,
            weight_transaction=0.0,
            rebalance_threshold=1000.0,  # High threshold to prevent full rebalance
            buy_threshold=0.00005,       # Low threshold to allow buy-only trades
            debug=True
        )
        
        # Verify only buy trades are generated
        self.assertTrue(len(trades) > 0, "Expected at least one trade")
        sell_trades = trades[trades['action'] == 'sell']
        self.assertEqual(len(sell_trades), 0, "Expected no sell trades")
        
        buy_trades = trades[trades['action'] == 'buy']
        self.assertGreater(len(buy_trades), 0, "Expected at least one buy trade")

    def test_both_thresholds_prevent_trades(self):
        """Test that high thresholds for both rebalance and buy-only prevent all trades."""
        status, should_trade, trade_summary, trades = self.strategy.compute_optimal_trades(
            weight_tax=1.0,
            weight_drift=1.0,
            weight_transaction=0.0,
            rebalance_threshold=1000.0,  # High threshold
            buy_threshold=1000.0,        # High threshold
            debug=True
        )
        
        # Verify no trades are generated
        self.assertEqual(len(trades), 0, "Expected no trades with high thresholds")
        self.assertFalse(should_trade, "Expected should_trade to be False with high thresholds")