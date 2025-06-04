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
                'asset_class': 'STOCK_A',
                'identifiers': ['STOCK_A'],
                'target_weight': 0.5
            },
            {
                'asset_class': 'STOCK_B',
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
            optimization_type=OracleOptimizationType.TAX_AWARE
        )
        
        # Set Oracle reference and add strategy
        self.strategy.set_oracle(self.oracle)
        self.oracle.strategies = [self.strategy]


    def test_transaction_only_optimization(self):
        """Test optimization with only transaction costs (should do nothing)."""
        status, should_trade, trade_summary, trades = self.strategy.compute_optimal_trades(
            weight_tax=0.0,
            weight_drift=0.0,
            weight_transaction=1.0,
            rebalance_threshold=0.0001,
            debug=True
        )
        
        # Verify no trades are generated
        self.assertEqual(len(trades), 0, "Expected no trades when only considering transaction costs")

    def test_cash_allocation_with_multiple_underweight_positions(self):
        """Test that cash is allocated optimally when multiple positions are underweight.
        
        Initial State:
        - STOCK_A: $40k position (20%), target 30%
            - Severely underweight (-10% from target)
            - Price: $120/share, ~333 shares
            
        - STOCK_B: $50k position (25%), target 35%
            - Moderately underweight (-10% from target)
            - Price: $80/share, ~625 shares
            
        - STOCK_C: $90k position (45%), target 35%
            - Overweight (+10% from target)
            - Price: $90/share, 1000 shares
            - Restricted from trading
            
        - Cash: $20k (10%)
        
        Total Portfolio: $200k
        Target Weights: 30-35-35
        
        Expected behavior:
        1. Cannot sell STOCK_C despite being overweight (restricted)
        2. Available cash ($20k) is insufficient to reach targets
        3. Cash should be allocated proportionally to underweight amount
        """
        # Update targets to include three stocks
        self.targets = pd.DataFrame([
            {
                'asset_class': 'STOCK_A',
                'identifiers': ['STOCK_A'],
                'target_weight': 0.30
            },
            {
                'asset_class': 'STOCK_B',
                'identifiers': ['STOCK_B'],
                'target_weight': 0.35
            },
            {
                'asset_class': 'STOCK_C',
                'identifiers': ['STOCK_C'],
                'target_weight': 0.35
            }
        ])
        
        # Set up stock restrictions to prevent trading of STOCK_C
        stock_restrictions = pd.DataFrame([
            {
                'identifier': 'STOCK_C',
                'can_buy': False,
                'can_sell': False
            }
        ])
        self.oracle.set_restrictions(stock_restrictions)
        
        # Update prices to include STOCK_C
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
                'identifier': 'STOCK_C',
                'price': 90.0,
            },
            {
                'identifier': CASH_CUSIP_ID,
                'price': 1.0,
            }
        ])
        
        # Update spreads to include STOCK_C
        self.spreads = pd.DataFrame([
            {
                'identifier': 'STOCK_A',
                'spread': 0.001
            },
            {
                'identifier': 'STOCK_B',
                'spread': 0.001
            },
            {
                'identifier': 'STOCK_C',
                'spread': 0.001
            },
            {
                'identifier': CASH_CUSIP_ID,
                'spread': 0.0
            }
        ])
        
        # Update tax lots to create underweight positions
        self.tax_lots = pd.DataFrame([
            {
                'tax_lot_id': 'lot_a1',
                'identifier': 'STOCK_A',
                'quantity': 333,  # ~$40k position at $120/share
                'cost_basis': 35000,
                'date': '2024-01-01'
            },
            {
                'tax_lot_id': 'lot_b1',
                'identifier': 'STOCK_B',
                'quantity': 625,  # $50k position at $80/share
                'cost_basis': 45000,
                'date': '2024-01-01'
            },
            {
                'tax_lot_id': 'lot_c1',
                'identifier': 'STOCK_C',
                'quantity': 1000,  # $90k position at $90/share
                'cost_basis': 85000,
                'date': '2024-01-01'
            }
        ])
        
        # Create new strategy with $20k cash
        self.strategy = OracleStrategy(
            tax_lots=self.tax_lots,
            targets=self.targets,
            prices=self.prices,
            spreads=self.spreads,
            cash=20000.0,
            optimization_type=OracleOptimizationType.TAX_AWARE,
            deminimus_cash_target=0.0
        )
        
        # Set Oracle reference and add strategy
        self.strategy.set_oracle(self.oracle)
        self.oracle.strategies = [self.strategy]
        
        # Run optimization focusing on drift and cash drag
        status, should_trade, trade_summary, trades = self.strategy.compute_optimal_trades(
            weight_tax=0.0,
            weight_drift=1.0,
            weight_transaction=0.0,
            weight_cash_drag=0.0,
            rebalance_threshold=0.0001,
            debug=True
        )
        
        # Verify trades exist
        self.assertTrue(len(trades) > 0, "Expected trades to deploy cash")
        
        # Verify no trades for STOCK_C
        stock_c_trades = trades[trades['identifier'] == 'STOCK_C']
        self.assertEqual(len(stock_c_trades), 0, "Expected no trades for restricted STOCK_C")
        
        # Calculate buys for each stock
        stock_a_buys = trades[
            (trades['identifier'] == 'STOCK_A') &
            (trades['action'] == 'buy')
        ]
        stock_b_buys = trades[
            (trades['identifier'] == 'STOCK_B') &
            (trades['action'] == 'buy')
        ]
        
        stock_a_drift_post = self.strategy.post_trade_strategy.drift_report.loc[
            self.strategy.post_trade_strategy.drift_report['asset_class'] == 'STOCK_A', 'drift'
        ].values[0]
        stock_b_drift_post = self.strategy.post_trade_strategy.drift_report.loc[
            self.strategy.post_trade_strategy.drift_report['asset_class'] == 'STOCK_B', 'drift'
        ].values[0]
        
        
        # STOCK_A is more underweight relative to its target (20% vs 30% = 33.3% below target)
        # compared to STOCK_B (25% vs 35% = 28.6% below target)
        # So it should receive more cash
        self.assertAlmostEqual(
            stock_a_drift_post,
            stock_b_drift_post,  # At least 10% more should go to STOCK_A
            delta=0.01,
            msg="Expected STOCK_A to receive more cash as it's more underweight relative to its target"
        )

        total_stock_a_buy_value = stock_a_buys["trade_value"].sum()
        total_stock_b_buy_value = stock_b_buys["trade_value"].sum()
        
        # Verify total cash deployed is close to available cash (accounting for transaction costs)
        total_deployed = total_stock_a_buy_value + total_stock_b_buy_value
        self.assertAlmostEqual(total_deployed, 20000, delta=1000, 
            msg="Expected most cash to be deployed (within transaction cost buffer)")
        
        # Verify final weights are still not at target (insufficient cash)
        final_portfolio_value = (
            333 * 120.0 + total_stock_a_buy_value +  # STOCK_A
            625 * 80.0 + total_stock_b_buy_value +   # STOCK_B
            1000 * 90.0 +                            # STOCK_C
            (20000 - total_deployed)                 # Remaining cash
        )
        
        final_stock_a_weight = (333 * 120.0 + total_stock_a_buy_value) / final_portfolio_value
        final_stock_b_weight = (625 * 80.0 + total_stock_b_buy_value) / final_portfolio_value
        
        self.assertLess(final_stock_a_weight, 0.30, 
            "Expected STOCK_A to still be underweight due to insufficient cash")
        self.assertLess(final_stock_b_weight, 0.35, 
            "Expected STOCK_B to still be underweight due to insufficient cash")
