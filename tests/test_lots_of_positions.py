"""Test optimization behavior with many positions and significant drift in a few."""
import unittest
from datetime import date
import pandas as pd
import numpy as np
from src.service.oracle import Oracle
from src.service.oracle_strategy import OracleStrategy
from src.service.helpers.constants import CASH_CUSIP_ID
from src.service.helpers.enums import OracleOptimizationType

class TestLotsOfPositions(unittest.TestCase):
    def setUp_strategy(self):
        """Set up the OracleStrategy instance."""
        self.strategy = OracleStrategy(
            tax_lots=self.tax_lots,
            targets=self.targets,
            prices=self.prices,
            spreads=self.spreads,
            cash=self.cash,
            optimization_type=OracleOptimizationType.TAX_AWARE
        )
        
        self.strategy.set_oracle(self.oracle)
        self.oracle.strategies = [self.strategy]

    def setUp(self):
        """Setup a portfolio with 500 positions where:
        - 490 positions are exactly on target (2% each)
        - 5 positions are significantly overweight (should be 2% but are at 4%)
        - 5 positions are significantly underweight (should be 2% but are at 0.5%)
        
        The 5 overweight positions are at a loss (cost basis higher than current price)
        The 5 underweight positions need to be bought
        All other positions are flat (cost basis = current price)
        """
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

        # Generate base data for 490 on-target positions
        base_positions = [{
            'tax_lot_id': f'lot_{i}',
            'identifier': f'STOCK_{i}',
            'quantity': 100,  # Each position has 100 shares
            'cost_basis': 100,  # Cost basis equals current price
            'date': '2024-01-01'
        } for i in range(490)]

        # Add 5 overweight positions at a loss
        overweight_positions = [{
            'tax_lot_id': f'lot_over_{i}',
            'identifier': f'OVER_{i}',
            'quantity': 100,  # Double quantity to be overweight
            'cost_basis': 100,  # Higher cost basis = loss
            'date': '2024-01-01'
        } for i in range(5)]

        # Add 5 underweight positions
        underweight_positions = [{
            'tax_lot_id': f'lot_under_{i}',
            'identifier': f'UNDER_{i}',
            'quantity': 100,  # Quarter quantity to be underweight
            'cost_basis': 100,
            'date': '2024-01-01'
        } for i in range(5)]

        # Combine all positions
        self.tax_lots = pd.DataFrame(base_positions + overweight_positions + underweight_positions)
        # Create targets (all positions should be 0.02 = 2%)
        base_targets = [{
            'asset_class': f'STOCK_{i}',
            'identifiers': [f'STOCK_{i}'],
            'target_weight': 0.002
        } for i in range(490)]

        overweight_targets = [{
            'asset_class': f'OVER_{i}',
            'identifiers': [f'OVER_{i}'],
            'target_weight': 0.002
        } for i in range(5)]

        underweight_targets = [{
            'asset_class': f'UNDER_{i}',
            'identifiers': [f'UNDER_{i}'],
            'target_weight': 0.002
        } for i in range(5)]

        self.targets = pd.DataFrame(base_targets + overweight_targets + underweight_targets)

        # Create prices
        base_prices = [{
            'identifier': f'STOCK_{i}',
            'price': 100.0  # Flat price
        } for i in range(490)]

        overweight_prices = [{
            'identifier': f'OVER_{i}',
            'price': 110.0  # Current price lower than cost basis
        } for i in range(5)]

        underweight_prices = [{
            'identifier': f'UNDER_{i}',
            'price': 90.0
        } for i in range(5)]

        self.prices = pd.DataFrame(base_prices + overweight_prices + underweight_prices + [
            {'identifier': CASH_CUSIP_ID, 'price': 1.0}
        ])

        # Create spreads (small spread for all positions)
        base_spreads = [{
            'identifier': f'STOCK_{i}',
            'spread': 0.001
        } for i in range(490)]

        overweight_spreads = [{
            'identifier': f'OVER_{i}',
            'spread': 0.001
        } for i in range(5)]

        underweight_spreads = [{
            'identifier': f'UNDER_{i}',
            'spread': 0.001
        } for i in range(5)]

        self.spreads = pd.DataFrame(base_spreads + overweight_spreads + underweight_spreads + [
            {'identifier': CASH_CUSIP_ID, 'spread': 0.0}
        ])

        self.setUp_strategy()

    def test_drift_optimization_large_portfolio(self):
        """Test that the optimizer focuses on fixing the most drifted positions when drift weight is high.
        
        Expected behavior:
        - Should prioritize fixing the 5 overweight and 5 underweight positions
        - Should largely ignore the 490 positions that are on target
        - Tax impact should be ignored since weight_tax = 0
        """
        initial_value = self.strategy.total_value()
        status, should_trade, trade_summary, trades = self.strategy.compute_optimal_trades(
            weight_tax=0.0,  # Ignore tax impact
            weight_drift=1.0,  # Focus on drift
            weight_transaction=0.0,  # Ignore transaction costs
            weight_cash_drag=0.0, 
            debug=True
        )

        self.assertTrue(status, "Optimization should succeed")
        self.assertTrue(should_trade, "Should recommend trading")

        # Filter trades by identifier type
        overweight_trades = trades[trades['identifier'].str.contains('OVER_')]
        underweight_trades = trades[trades['identifier'].str.contains('UNDER_')]
        base_trades = trades[trades['identifier'].str.contains('STOCK_')]

        # Verify we're mainly trading the drifted positions
        self.assertGreaterEqual(len(overweight_trades), 4, 
                              "Should trade most overweight positions")
        self.assertGreaterEqual(len(underweight_trades), 4, 
                              "Should trade most underweight positions")
        self.assertLess(len(base_trades), len(overweight_trades) + len(underweight_trades),
                       "Should trade fewer on-target positions than drifted positions")

        # Verify trade directions
        for _, trade in overweight_trades.iterrows():
            self.assertEqual(trade['action'], 'sell',
                           "Overweight positions should be sold")
            
        for _, trade in underweight_trades.iterrows():
            self.assertEqual(trade['action'], 'buy',
                           "Underweight positions should be bought")

        pre_trade_drift = self.strategy.drift_report['drift_pct'].iloc[0]
        post_trade_drift = self.strategy.post_trade_strategy.drift_report['drift_pct'].iloc[0]
        # Verify drift improvement
        self.assertLess(
            abs(post_trade_drift),
            abs(pre_trade_drift),
            "Each trade should improve drift"
        )