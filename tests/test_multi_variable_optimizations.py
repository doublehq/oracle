"""Test optimization behavior with multiple objective weights."""
import unittest
from datetime import date
import pandas as pd
import numpy as np
from src.service.oracle import Oracle
from src.service.oracle_strategy import OracleStrategy
from src.service.helpers.constants import CASH_CUSIP_ID
from src.service.helpers.enums import OracleOptimizationType

import pulp

class TestMultiVariableOptimizations(unittest.TestCase):
    def setUp_strategy(self):
        """Set up the OracleStrategy instance."""
        
        # Create OracleStrategy instance
        self.strategy = OracleStrategy(
            tax_lots=self.tax_lots,
            targets=self.targets,
            prices=self.prices,
            spreads=self.spreads,
            cash=self.cash,
            optimization_type=OracleOptimizationType.TAX_AWARE
        )
        
        # Set Oracle reference and add strategy
        self.strategy.set_oracle(self.oracle)
        self.oracle.strategies = [self.strategy]

    def setUp(self):
        """Setup a two-stock portfolio with gains and losses.
        
        Initial State:
        - STOCK_A: $120k position (60%), cost basis $100k
            - Currently at gain ($20k unrealized gain)
            - Price: $120/share, 1000 shares
            - Cost basis: $100/share
            
        - STOCK_B: $80k position (40%), cost basis $100k
            - Currently at loss ($20k unrealized loss)
            - Price: $80/share, 1000 shares
            - Cost basis: $100/share
            
        Total Portfolio: $200k
        Target Weights: 50-50
        """
        # Setup test data
        self.current_date = date(2024, 4, 2)
        self.cash = 0.0
        
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
                'quantity': 10,
                'cost_basis': 100,  # $10/share * 10 shares
                'date': '2024-01-01'
            },
            {
                'tax_lot_id': 'lot_b1',
                'identifier': 'STOCK_B',
                'quantity': 10,
                'cost_basis': 100,  # $10100/share * 10 shares
                'date': '2024-01-01'
            }
        ])
        self.targets = pd.DataFrame([
            {
                'asset_class': 'STOCK_A',
                'identifiers': ['STOCK_A'],
                'target_weight': 0.5
            },
            {
                'asset_class': 'STOCK_B', 
                'identifiers': ['STOCK_B'],
                'target_weight': 0.5
            },
            {
                'asset_class': CASH_CUSIP_ID,
                'identifiers': [CASH_CUSIP_ID],
                'target_weight': 0.0
            }
        ])
        
        self.prices = pd.DataFrame([
            {
                'identifier': 'STOCK_A',
                'price': 12,  # Stock has appreciated
            },
            {
                'identifier': 'STOCK_B',
                'price': 8.0,   # Stock has depreciated
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
                'spread': 0.0
            }
        ])
        self.setUp_strategy()
    

    def test_balanced_drift_and_tax(self):
        """Test optimization with balanced weights between drift and tax.
        
        Expected behavior:
        - Should partially rebalance towards target weights
        - Should avoid realizing all gains for tax efficiency
        - Final position should be between initial state (60-40) and target (50-50)
        """
        # Unpack the tuple
        initial_value = self.strategy.total_value()
        status, should_trade, trade_summary, trades = self.strategy.compute_optimal_trades(
            weight_tax=1.0,
            weight_drift=2.0,
            weight_transaction=0.0,
            weight_cash_drag=1.0,
            debug=True
        )
        
        # Find stock A sells
        stock_a_sells = trades[
            (trades['identifier'] == 'STOCK_A') &
            (trades['action'] == 'sell')
        ]

        # Find stock B buys
        stock_b_buys = trades[
            (trades['identifier'] == 'STOCK_B') &
            (trades['action'] == 'buy')
        ]

        # Calculate total values
        total_a_sell_value = stock_a_sells["trade_value"].sum()
        total_b_buy_value = stock_b_buys["trade_value"].sum()

        
        # Calculate final portfolio weights using actuals for initial values
        initial_a_value = self.strategy.actuals[self.strategy.actuals['identifier'] == 'STOCK_A']['market_value'].iloc[0]
        initial_b_value = self.strategy.actuals[self.strategy.actuals['identifier'] == 'STOCK_B']['market_value'].iloc[0]
        initial_a_weight = initial_a_value / initial_value
        initial_b_weight = initial_b_value / initial_value

        final_a_value = initial_a_value - total_a_sell_value
        final_b_value = initial_b_value + total_b_buy_value
        final_total = final_a_value + final_b_value
        final_a_weight = final_a_value / final_total
        
        # Should see partial rebalancing - weight should move from 60% towards 50%
        # but not all the way there due to tax considerations
        self.assertGreaterEqual(final_a_weight, 0.50,  # Should still be above target
                         msg="Weight should not fully rebalance to 50% due to tax impact")
        self.assertLess(final_a_weight, initial_a_weight,     # But should improve from initial 60%
                      msg="Weight should improve from initial 60%")
        
        post_weight_drift = self.strategy.post_trade_strategy.drift_report['drift_pct'].iloc[0]
        pre_weight_drift =  self.strategy.drift_report['drift_pct'].iloc[0]

        # Verify each trade improves drift
        self.assertLess(
            abs(post_weight_drift),
            abs(pre_weight_drift),
            "Each trade should improve drift"
        )

    def test_drift_vs_tax_large_gain(self):
        """Test optimization with balanced weights between drift and tax.
        
        Expected behavior:
        - Set current price of STOCK_A to $200, which is a large gain.
        - We should do nothing as a result.
        """

        
        self.prices = pd.DataFrame([
            {
                'identifier': 'STOCK_A',
                'price': 10.10,  # Stock has appreciated
            },
            {
                'identifier': 'STOCK_B',
                'price': 10.0,   # Stock is flat
            },
            {
                'identifier': CASH_CUSIP_ID,
                'price': 1.0,
            }
        ])

        self.tax_lots = pd.DataFrame([
            {
                'tax_lot_id': 'lot_a1',
                'identifier': 'STOCK_A',
                'quantity': 10,
                'cost_basis': 90, 
                'date': '2024-01-01'
            },
            {
                'tax_lot_id': 'lot_b1',
                'identifier': 'STOCK_B',
                'quantity': 10,
                'cost_basis': 100,  # $10100/share * 10 shares
                'date': '2024-01-01'
            }
        ])
        self.setUp_strategy()

        status, should_trade, trade_summary, trades = self.strategy.compute_optimal_trades(
            weight_tax=1.0,
            weight_drift=1.0,
            weight_transaction=0.0,
            weight_cash_drag=1.0,
            debug=True
        )
        # Verify that no trades are returned
        self.assertTrue(status, "Optimization should succeed")
        self.assertFalse(should_trade, "Should not recommend trading")
        self.assertEqual(len(trades), 0, "No trades should be generated")
        
        # Verify trade summary shows no activity
        self.assertEqual(trade_summary["execution"]['num_buys'], 0, "No buys should be recommended")
        self.assertEqual(trade_summary["execution"]['num_sells'], 0, "No sells should be recommended")
    
    def test_drift_vs_tax_tiny_gain(self):
        """Test optimization with balanced weights between drift and tax.
        
        Expected behavior:
        - Set current price of STOCK_A to $200, which is a large gain.
        - We should do nothing as a result.
        """

        
        self.prices = pd.DataFrame([
            {
                'identifier': 'STOCK_A',
                'price': 10.1,  # Stock has appreciated
            },
            {
                'identifier': 'STOCK_B',
                'price': 8.0,   # Stock has depreciated
            },
            {
                'identifier': CASH_CUSIP_ID,
                'price': 1.0,
            }
        ])
        self.setUp_strategy()

        status, should_trade, trade_summary, trades = self.strategy.compute_optimal_trades(
            weight_tax=1.0,
            weight_drift=1.0,
            weight_transaction=0.0,
            weight_cash_drag=1.0,
            debug=True
        )
        # Verify that trades are returned
        self.assertTrue(status, "Optimization should succeed")
        self.assertTrue(should_trade, "Should recommend trading")


        stock_a_sells = trades[
            (trades['identifier'] == 'STOCK_A') &
            (trades['action'] == 'sell')
        ]
        stock_b_buys = trades[
            (trades['identifier'] == 'STOCK_B') &
            (trades['action'] == 'buy')
        ]
        
        total_a_sell_value = stock_a_sells["trade_value"].sum()
        total_b_buy_value = stock_b_buys["trade_value"].sum()

        self.assertAlmostEqual(total_a_sell_value, 10.0, delta=2, msg="STOCK_A trade should be nearly $10")
        self.assertAlmostEqual(total_b_buy_value, 10.0, delta=2, msg="STOCK_B trade should be nearly $10")
    

    def test_uneven_transaction_costs(self):
        """Test optimization with uneven transaction costs.
        
        This test verifies that when transaction costs are uneven between securities,
        the optimizer will prefer trading the security with lower transaction costs.
        STOCK_A has a spread of 0.01 (1%) while STOCK_B has a spread of 0.001 (0.1%),
        so we expect the optimizer to prefer trading STOCK_B.
        """
    
        # Create empty tax_lots DataFrame with the correct columns
        self.tax_lots = pd.DataFrame(columns=[
            'tax_lot_id',
            'identifier',
            'quantity',
            'cost_basis',
            'date'
        ])
        self.prices = pd.DataFrame([
            {
                'identifier': 'STOCK_A',
                'price': 10.0,
            },
            {
                'identifier': 'STOCK_B',
                'price': 10.0,
            },
            {
                'identifier': CASH_CUSIP_ID,
                'price': 1.0,
            }
        ])
        self.spreads = pd.DataFrame([
            {
                'identifier': 'STOCK_A',
                'spread': 0.01  # 10x STOCK_B
            },
            {
                'identifier': 'STOCK_B',
                'spread': 0.001  # 10 basis points
            },
            {
                'identifier': CASH_CUSIP_ID,
                'spread': 0.0
            }
        ])
        self.cash = 1000.0
        self.setUp_strategy()

        status, should_trade, trade_summary, trades = self.strategy.compute_optimal_trades(
            weight_tax=1.0,
            weight_drift=1.0,
            weight_transaction=1.0,
            weight_cash_drag=1.0,
            debug=True
        )
        # Verify that trades are returned
        self.assertTrue(status, "Optimization should succeed")
        self.assertTrue(should_trade, "Should recommend trading")


        stock_a_buys = trades[
            (trades['identifier'] == 'STOCK_A') &
            (trades['action'] == 'buy')
        ]
        stock_b_buys = trades[
            (trades['identifier'] == 'STOCK_B') &
            (trades['action'] == 'buy')
        ]
        
        total_a_buy_value = stock_a_buys.apply(
            lambda x: x['quantity'] * x['price'], axis=1
        ).sum()
        total_b_buy_value = stock_b_buys.apply(
            lambda x: x['quantity'] * x['price'], axis=1
        ).sum()

        self.assertGreaterEqual(total_b_buy_value, total_a_buy_value, msg="Should buy more B due to mismatch spreads.")
        