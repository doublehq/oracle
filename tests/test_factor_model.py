"""Test factor model rebalancing with Oracle."""
import unittest
from datetime import date
import pandas as pd
import numpy as np
from src.service.oracle import Oracle
from src.service.oracle_strategy import OracleStrategy
from src.service.helpers.constants import CASH_CUSIP_ID
from src.service.helpers.enums import OracleOptimizationType

import pulp
from src.service.helpers.trade_applier import apply_trades_to_portfolio

class TestFactorModel(unittest.TestCase):
    def setUp(self):
        """Setup a portfolio with three factor stocks in equal proportions,
        where one factor stock has significantly declined in value.
        
        Initial State:
        - GROWTH_STOCK: Originally 1/3 of portfolio, still at purchase price
        - MOMENTUM_STOCK: Originally 1/3 of portfolio, still at purchase price
        - MIN_VOL_STOCK: Originally 1/3 of portfolio, has dropped 40% in value
        
        Expected behavior: The optimizer should sell some of the GROWTH_STOCK and MOMENTUM_STOCK
        to buy more MIN_VOL_STOCK to bring it back to target allocation.
        """
        self.current_date = date(2024, 4, 2)
        self.tax_rates = pd.DataFrame([
            {'gain_type': 'short_term', 'federal_rate': 0.35, 'state_rate': 0.06, 'total_rate': 0.41},
            {'gain_type': 'long_term', 'federal_rate': 0.20, 'state_rate': 0.06, 'total_rate': 0.26},
            {"gain_type": "qualified_dividend", "federal_rate": 0.15, "state_rate": 0.06, "total_rate": 0.21}
        ])
        self.oracle = Oracle(current_date=self.current_date, tax_rates=self.tax_rates)
        
        # All stocks were purchased at the same time with equal dollar amounts
        self.tax_lots = pd.DataFrame([
            {'tax_lot_id': 'lot_growth', 'identifier': 'GROWTH_STOCK', 'quantity': 1000, 'cost_basis': 100000, 'date': '2024-01-01'},
            {'tax_lot_id': 'lot_momentum', 'identifier': 'MOMENTUM_STOCK', 'quantity': 1000, 'cost_basis': 100000, 'date': '2024-01-01'},
            {'tax_lot_id': 'lot_min_vol', 'identifier': 'MIN_VOL_STOCK', 'quantity': 1000, 'cost_basis': 100000, 'date': '2024-01-01'}
        ])
        
        # Target allocation is equal across all three factors
        self.targets = pd.DataFrame([
            {'asset_class': 'GROWTH_STOCK', 'identifiers': ['GROWTH_STOCK'], 'target_weight': 1/3},
            {'asset_class': 'MOMENTUM_STOCK', 'identifiers': ['MOMENTUM_STOCK'], 'target_weight': 1/3},
            {'asset_class': 'MIN_VOL_STOCK', 'identifiers': ['MIN_VOL_STOCK'], 'target_weight': 1/3},
            {'asset_class': CASH_CUSIP_ID, 'identifiers': [CASH_CUSIP_ID], 'target_weight': 0.0}
        ])
        
        # GROWTH and MOMENTUM are unchanged, MIN_VOL has dropped 40%
        self.prices = pd.DataFrame([
            {'identifier': 'GROWTH_STOCK', 'price': 100.0},  # Unchanged
            {'identifier': 'MOMENTUM_STOCK', 'price': 100.0},  # Unchanged
            {'identifier': 'MIN_VOL_STOCK', 'price': 60.0},   # 40% decline
            {'identifier': CASH_CUSIP_ID, 'price': 1.0}
        ])
        
        # Small spread for all stocks
        self.spreads = pd.DataFrame([
            {'identifier': 'GROWTH_STOCK', 'spread': 0.001},
            {'identifier': 'MOMENTUM_STOCK', 'spread': 0.001},
            {'identifier': 'MIN_VOL_STOCK', 'spread': 0.001},
            {'identifier': CASH_CUSIP_ID, 'spread': 0.0}
        ])
        
        # Create factor model with corresponding factors for each stock
        self.factor_model = pd.DataFrame([
            {'identifier': 'GROWTH_STOCK', 'growth': 1.0, 'momentum': 0.3, 'min_volatility': 0.1},
            {'identifier': 'MOMENTUM_STOCK', 'growth': 0.2, 'momentum': 1.0, 'min_volatility': 0.2},
            {'identifier': 'MIN_VOL_STOCK', 'growth': 0.1, 'momentum': 0.2, 'min_volatility': 1.0},
            {'identifier': CASH_CUSIP_ID, 'growth': 0.0, 'momentum': 0.0, 'min_volatility': 0.0}
        ])

        # Create OracleStrategy instance
        self.strategy = OracleStrategy(
            tax_lots=self.tax_lots,
            targets=self.targets,
            prices=self.prices,
            spreads=self.spreads,
            factor_model=self.factor_model,
            cash=0.0,
            optimization_type=OracleOptimizationType.DIRECT_INDEX
        )
        
        # Set Oracle reference and add strategy
        self.strategy.set_oracle(self.oracle)
        self.oracle.strategies = [self.strategy]

    def test_factor_model_only(self):
        """Test that the oracle correctly rebalances a portfolio where one factor has declined."""

        # Before optimization - calculate current weights
        total_value_before = self.strategy.total_value()
        target_factor_model = self.strategy.factor_model_target
        before_factor_model = self.strategy.factor_model_actual

        start_growth = before_factor_model['growth'].iloc[0]
        start_momentum = before_factor_model['momentum'].iloc[0]
        start_min_vol = before_factor_model['min_volatility'].iloc[0]
        
        
        # Verify initial factor exposures
        # Due to MIN_VOL_STOCK price drop (40%), we should be underweight min_volatility
        # and overweight growth and momentum factors
        
        # Check that we're overweight growth
        self.assertGreater(
            start_growth,
            target_factor_model['growth'].iloc[0],
            "Should be overweight growth factor due to MIN_VOL_STOCK decline"
        )
        
        # Check that we're overweight momentum
        self.assertGreater(
            start_momentum,
            target_factor_model['momentum'].iloc[0],
            "Should be overweight momentum factor due to MIN_VOL_STOCK decline"
        )
        
        # Check that we're underweight min_volatility
        self.assertLess(
            start_min_vol,
            target_factor_model['min_volatility'].iloc[0],
            "Should be underweight min_volatility factor due to MIN_VOL_STOCK decline"
        )
        
        # Run optimization
        # compute_optimal_trades returns (status, should_trade, trade_summary, trades)
        status, should_trade, trade_summary, trades = self.strategy.compute_optimal_trades(
            weight_factor_model=1.0,
            weight_transaction=0.0,
            weight_tax=0.0,
            weight_drift=0.0,
            weight_cash_drag=0.0
        )
        
        # Verify we got successful optimization
        self.assertIsNotNone(status)
        self.assertTrue(should_trade)
        self.assertFalse(trades.empty)
        
        # Apply trades to the portfolio using portfolio_updater
        updated_tax_lots, updated_cash, recently_closed_lots = apply_trades_to_portfolio(
            tax_lots=self.tax_lots,
            trades=trades,
            cash=0.0,  # Initial cash is 0
            current_date=self.current_date
        )
        
        # Create new strategy with updated portfolio
        updated_strategy = OracleStrategy(
            tax_lots=updated_tax_lots,
            targets=self.targets,
            prices=self.prices,
            spreads=self.spreads,
            factor_model=self.factor_model,
            cash=updated_cash,
            optimization_type=OracleOptimizationType.DIRECT_INDEX
        )
        updated_strategy.set_oracle(self.oracle)
        
        # Get new factor model exposures
        after_factor_model = updated_strategy.factor_model_actual
        
        end_growth = after_factor_model['growth'].iloc[0]
        end_momentum = after_factor_model['momentum'].iloc[0]
        end_min_vol = after_factor_model['min_volatility'].iloc[0]
        
        # Verify factor exposures have improved (closer to target)
        target_growth = target_factor_model['growth'].iloc[0]
        target_momentum = target_factor_model['momentum'].iloc[0]
        target_min_vol = target_factor_model['min_volatility'].iloc[0]
        
        # For each factor, verify the distance to target has decreased
        self.assertLess(
            abs(end_growth - target_growth),
            abs(start_growth - target_growth),
            "Growth factor exposure should be closer to target after trades"
        )
        
        self.assertLess(
            abs(end_momentum - target_momentum),
            abs(start_momentum - target_momentum),
            "Momentum factor exposure should be closer to target after trades"
        )
        
        self.assertLess(
            abs(end_min_vol - target_min_vol),
            abs(start_min_vol - target_min_vol),
            "Min volatility factor exposure should be closer to target after trades"
        )
        
        # Print factor exposure changes for debugging
        print("\nFactor Exposure Changes:")
        print(f"Growth: {start_growth:.3f} -> {end_growth:.3f} (target: {target_growth:.3f})")
        print(f"Momentum: {start_momentum:.3f} -> {end_momentum:.3f} (target: {target_momentum:.3f})")
        print(f"Min Vol: {start_min_vol:.3f} -> {end_min_vol:.3f} (target: {target_min_vol:.3f})")

    def test_factor_model_and_drift(self):
        """Test optimization when there's tension between factor model and drift objectives.
        
        Setup:
        - Create a new portfolio with 3 stocks exposed to 2 factors (growth & value)
        - PURE_GROWTH: 100% growth, 0% value
        - PURE_VALUE: 0% growth, 100% value
        - BALANCED: 75% growth, 25% value
        
        Initial state:
        - PURE_GROWTH is overweight (40% vs 33.3% target)
        - PURE_VALUE is underweight (25% vs 33.3% target)
        - BALANCED is at target weight (33.3%)
        
        If optimizing purely for drift:
          - Should sell PURE_GROWTH and buy PURE_VALUE to reach target weights
        
        If optimizing purely for factor model:
          - Should maintain factor exposures close to target
        
        When optimizing for both:
          - Should find a compromise that improves both objectives
          - Likely selling some PURE_GROWTH to buy both PURE_VALUE and BALANCED
        """
        # Create a new test portfolio with 2 factors and 3 stocks
        tax_lots = pd.DataFrame([
            {'tax_lot_id': 'lot_growth', 'identifier': 'PURE_GROWTH', 'quantity': 1200, 'cost_basis': 120000, 'date': '2024-01-01'},
            {'tax_lot_id': 'lot_value', 'identifier': 'PURE_VALUE', 'quantity': 750, 'cost_basis': 75000, 'date': '2024-01-01'},
            {'tax_lot_id': 'lot_balanced', 'identifier': 'BALANCED', 'quantity': 1000, 'cost_basis': 100000, 'date': '2024-01-01'}
        ])
        
        targets = pd.DataFrame([
            {'asset_class': 'PURE_GROWTH', 'identifiers': ['PURE_GROWTH'], 'target_weight': 1/3},
            {'asset_class': 'PURE_VALUE', 'identifiers': ['PURE_VALUE'], 'target_weight': 1/3},
            {'asset_class': 'BALANCED', 'identifiers': ['BALANCED'], 'target_weight': 1/3},
            {'asset_class': CASH_CUSIP_ID, 'identifiers': [CASH_CUSIP_ID], 'target_weight': 0.0}
        ])
        
        prices = pd.DataFrame([
            {'identifier': 'PURE_GROWTH', 'price': 100.0},
            {'identifier': 'PURE_VALUE', 'price': 100.0},
            {'identifier': 'BALANCED', 'price': 100.0},
            {'identifier': CASH_CUSIP_ID, 'price': 1.0}
        ])
        
        spreads = pd.DataFrame([
            {'identifier': 'PURE_GROWTH', 'spread': 0.001},
            {'identifier': 'PURE_VALUE', 'spread': 0.001},
            {'identifier': 'BALANCED', 'spread': 0.001},
            {'identifier': CASH_CUSIP_ID, 'spread': 0.0}
        ])
        
        # Create factor model with 2 factors
        factor_model = pd.DataFrame([
            {'identifier': 'PURE_GROWTH', 'growth': 1.0, 'value': 0.3, 'min_volatility': 0.0},
            {'identifier': 'PURE_VALUE', 'growth': 0.01, 'value': 1.0, 'min_volatility': 0.0},
            {'identifier': 'BALANCED', 'growth': 0.6, 'value': 0.4, 'min_volatility': 0.4},
            {'identifier': CASH_CUSIP_ID, 'growth': 0, 'value': 0, 'min_volatility': 0}
        ])

        # Create restrictions to limit trading on BALANCED
        restrictions = pd.DataFrame([
            {
                'identifier': 'BALANCED',
                'can_buy': False,     # Allow buying
                'can_sell': False     # Allow selling
            }
        ])
        self.oracle.set_restrictions(restrictions)
        
        # Create a fresh OracleStrategy for this test
        self.strategy = OracleStrategy(
            tax_lots=tax_lots,
            targets=targets,
            prices=prices,
            spreads=spreads,
            factor_model=factor_model,
            cash=0.0,  # No additional cash
            optimization_type=OracleOptimizationType.DIRECT_INDEX
        )
        
        self.strategy.set_oracle(self.oracle)
        
        # Calculate pre-optimization state
        total_value = self.strategy.total_value()
        target_factor_model = self.strategy.factor_model_target
        before_factor_model = self.strategy.factor_model_actual

        actuals_before = self.strategy.actuals
        before_growth_weight = actuals_before.loc[actuals_before['identifier'] == 'PURE_GROWTH', 'actual_weight'].iloc[0]
        before_value_weight = actuals_before.loc[actuals_before['identifier'] == 'PURE_VALUE', 'actual_weight'].iloc[0]
        before_balanced_weight = actuals_before.loc[actuals_before['identifier'] == 'BALANCED', 'actual_weight'].iloc[0]
        
        target_growth = target_factor_model['growth'].iloc[0]
        target_value= target_factor_model['value'].iloc[0]
        target_min_vol = target_factor_model['min_volatility'].iloc[0]
        
        start_growth = before_factor_model['growth'].iloc[0]
        start_value = before_factor_model['value'].iloc[0]
        start_min_vol = before_factor_model['min_volatility'].iloc[0]
        
        # Run optimization with equal weight on factor model and drift
        status, should_trade, trade_summary, trades = self.strategy.compute_optimal_trades(
            weight_factor_model=1.0,
            weight_drift=1.0,
            weight_transaction=0.1,  # Small penalty for transaction costs
            weight_tax=0.0,
            weight_cash_drag=0.0
        )
        
        # Verify we got successful optimization
        self.assertIsNotNone(status)
        self.assertTrue(should_trade)
        self.assertFalse(trades.empty)
        
        # Apply trades to the portfolio using portfolio_updater
        updated_tax_lots, updated_cash, recently_closed_lots = apply_trades_to_portfolio(
            tax_lots=tax_lots,
            trades=trades,
            cash=0.0,  # Initial cash
            current_date=self.oracle.current_date
        )
        
        # Create new strategy with updated portfolio
        updated_strategy = OracleStrategy(
            tax_lots=updated_tax_lots,
            targets=targets,
            prices=prices,
            spreads=spreads,
            factor_model=factor_model,
            cash=updated_cash,
            optimization_type=OracleOptimizationType.DIRECT_INDEX
        )
        updated_strategy.set_oracle(self.oracle)
        
        # Get actual weights after trades
        actuals_after = updated_strategy.actuals
        after_growth_weight = actuals_after.loc[actuals_after['identifier'] == 'PURE_GROWTH', 'actual_weight'].iloc[0]
        after_value_weight = actuals_after.loc[actuals_after['identifier'] == 'PURE_VALUE', 'actual_weight'].iloc[0]
        after_balanced_weight = actuals_after.loc[actuals_after['identifier'] == 'BALANCED', 'actual_weight'].iloc[0]
        
        # Calculate final factor exposures using updated strategy's factor model
        after_factor_model = updated_strategy.factor_model_actual
        final_growth = after_factor_model['growth'].iloc[0]
        final_value = after_factor_model['value'].iloc[0]
        final_min_volatility = after_factor_model['min_volatility'].iloc[0]
        
        # Verify drift improved (weights closer to target)
        self.assertLess(
            abs(after_growth_weight - 1/3),
            abs(before_growth_weight - 1/3),
            "Growth weight should be closer to target after trades"
        )
        self.assertLess(
            abs(after_value_weight - 1/3),
            abs(before_value_weight - 1/3),
            "Value weight should be closer to target after trades"
        )
        self.assertEqual(
            abs(after_balanced_weight - 1/3),
            abs(before_balanced_weight - 1/3),
            "Balanced weight should not change since we cannot trade it."
        )
        
        # Verify factor exposures improved (closer to target)
        self.assertLess(
            abs(final_growth - target_growth),
            abs(start_growth - target_growth),
            "Growth factor exposure should be closer to target after trades"
        )
        self.assertLess(
            abs(final_value - target_value),
            abs(start_value - target_value),
            "Momentum factor exposure should be closer to target after trades"
        )
        self.assertEqual(
            abs(final_min_volatility - target_min_vol),
            abs(start_min_vol - target_min_vol),
            "Min Vol factor exposure should the same since we cannot trade it."
        )
        
