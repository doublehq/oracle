import unittest
import pandas as pd
from datetime import date
from typing import cast
import pulp

# Assuming these are the correct locations - adjust if necessary
from src.service.oracle import Oracle
from src.service.oracle_strategy import OracleStrategy
from src.service.helpers.enums import OracleOptimizationType
from src.service.helpers.constants import CASH_CUSIP_ID
# Import the standalone apply_trades function
from src.service.helpers.trade_applier import apply_trades_to_portfolio

# Test Constants
STOCK_A = "STOCK_A"
STOCK_B = "STOCK_B"

class TestMaxWithdrawl(unittest.TestCase):
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
            {'tax_lot_id': 'lot_a2', 'identifier': 'STOCK_A', 'quantity': 1000, 'cost_basis': 50000, 'date': '2024-01-01'}, # Cost $50/share
            {'tax_lot_id': 'lot_gain', 'identifier': 'STOCK_B', 'quantity': 500, 'cost_basis': 50000, 'date': '2024-01-01'} # Cost $100/share
        ])
        self.targets = pd.DataFrame([
            {'asset_class': 'STOCK_A', 'identifiers': ['STOCK_A'], 'target_weight': 0.50},
            {'asset_class': 'STOCK_B', 'identifiers': ['STOCK_B'], 'target_weight': 0.50},
            {'asset_class': CASH_CUSIP_ID, 'identifiers': [CASH_CUSIP_ID], 'target_weight': 0.0}
        ])
        self.prices = pd.DataFrame([
            {'identifier': 'STOCK_A', 'price': 40.0}, # Current price $40, significant loss
            {'identifier': 'STOCK_B', 'price': 110.0},# Current price $110, small gain
            {'identifier': CASH_CUSIP_ID, 'price': 1.0}
        ])
        self.spreads = pd.DataFrame([
            {'identifier': 'STOCK_A', 'spread': 0.001},
            {'identifier': 'STOCK_B', 'spread': 0.001},
            {'identifier': CASH_CUSIP_ID, 'spread': 0.0}
        ])
        
        self.base_strategy_data = {
            'tax_lots': self.tax_lots,
            'targets': self.targets,
            'prices': self.prices,
            'spreads': self.spreads,
            'cash': 5000.0 # Start with $5k cash
        }

    def test_max_withdrawl(self):
        """ Test we sell everything with Max Withdrawl when nothing is in the wash sale restriction period."""
        # Create a strategy with Tax Aware optimization type
        strategy = OracleStrategy(
            **self.base_strategy_data,
            optimization_type=OracleOptimizationType.TAX_AWARE
        )
        strategy.min_notional = 0.0  # Set min notional to 0 for this test
        strategy.trade_rounding = 4
        # Set the oracle reference
        strategy.set_oracle(self.oracle)
        
        # Calculate the total portfolio value
        total_portfolio_value = strategy.total_value()
        self.assertGreater(total_portfolio_value, 0, "Portfolio should have positive value")
        
        # Calculate max withdrawal amount
        max_withdrawal, trades = strategy.calculate_max_withdrawal_amount(
            respect_wash_sales=True,
            preserve_targets=False,
            debug=True
        )
        
        # Verify max withdrawal equals total portfolio value (within small rounding tolerance)
        self.assertAlmostEqual(
            max_withdrawal,
            total_portfolio_value,
            delta=0.01,  # Allow for small numerical differences
            msg="Max withdrawal should equal total portfolio value when no wash sale restrictions"
        )
        
        # Verify we got trade recommendations
        self.assertFalse(trades.empty, "Should have trade recommendations")
        
        # Apply the trades to the portfolio and check the final state
        new_tax_lots, new_cash, recently_closed_lots = apply_trades_to_portfolio(
            tax_lots=strategy.tax_lots.copy(),
            trades=trades,
            cash=strategy.cash,
            current_date=self.current_date
        )
        
        # Verify all positions are sold (no securities left in tax lots)
        self.assertTrue(
            new_tax_lots.empty or new_tax_lots[new_tax_lots['identifier'] != CASH_CUSIP_ID].empty,
            "All securities should be sold"
        )
        
        # Verify all value is now in cash
        self.assertAlmostEqual(
            new_cash,
            total_portfolio_value,
            delta=0.01,  # Allow for small numerical differences
            msg="All value should be in cash after applying trades"
        )

    def test_max_withdrawl_with_restrictions(self):
        """ Test we sell a bunch but not everything with Max Withdrawl when something is in the wash sale restriction period."""
        # Start with a copy of the base tax lots
        modified_tax_lots = self.tax_lots.copy()
        
        # Add two recent buys (one for STOCK_A, one for STOCK_B) with dates close to current date
        # so they'll be in the wash sale restriction period
        recent_buys = pd.DataFrame([
            {
                'tax_lot_id': 'lot_a_recent', 
                'identifier': STOCK_A, 
                'quantity': 200, 
                'cost_basis': 12000,  # $60 per share - higher than current price
                'date': '2024-03-31'  # Just a few days before current date (2024-04-02)
            },
            {
                'tax_lot_id': 'lot_a_2nd_recent', 
                'identifier': STOCK_A, 
                'quantity': 200, 
                'cost_basis': 12000,  # $60 per share - higher than current price
                'date': '2024-03-30'  # Just a few days before current date
            }
        ])
        
        # Append the recent buys to the tax lots
        modified_tax_lots = pd.concat([modified_tax_lots, recent_buys])
        
        # Create Oracle with wash sale restrictions
        # Current date is 2024-04-02 from setUp
        oracle = Oracle(current_date=self.current_date, tax_rates=self.tax_rates)
        
        # Create a strategy with our modified tax lots
        strategy_data = self.base_strategy_data.copy()
        strategy_data['tax_lots'] = modified_tax_lots
        
        strategy = OracleStrategy(
            **strategy_data,
            optimization_type=OracleOptimizationType.TAX_AWARE
        )
        strategy.min_notional = 0.0  # Set min notional to 0 for this test
        strategy.trade_rounding = 4
        
        prevent_liquidation_strategy_data = strategy_data.copy()
        prevent_liquidation_strategy_data["tax_lots"] = pd.DataFrame([
            {
                'tax_lot_id': 'prevent_liquidation_lot_a_recent',
                'identifier': STOCK_A,
                'quantity': 1,
                'cost_basis': 120,  # $60 per share - higher than current price
                'date': '2024-03-31'  # Just a few days before current date (2024-04-02)
            }
        ])
        prevent_liquidation_strategy = OracleStrategy(
            **prevent_liquidation_strategy_data,
            optimization_type=OracleOptimizationType.TAX_AWARE
        )

        # Set the oracle reference
        strategy.set_oracle(oracle)
        prevent_liquidation_strategy.set_oracle(oracle)
        oracle.strategies = [strategy, prevent_liquidation_strategy]
        oracle.initialize_wash_sale_restrictions()
        
        # Calculate the total portfolio value
        total_portfolio_value = strategy.total_value()
        self.assertGreater(total_portfolio_value, 0, "Portfolio should have positive value")
        
        # Calculate max withdrawal amount with respect to wash sale restrictions
        max_withdrawal, trades = strategy.calculate_max_withdrawal_amount(
            respect_wash_sales=True,
            preserve_targets=False,
            debug=True
        )
        
        # Verify we got trade recommendations
        self.assertFalse(trades.empty, "Should have trade recommendations")
        
        # The max withdrawal should be less than the total portfolio value
        # because some positions can't be sold due to wash sale restrictions
        self.assertLess(
            max_withdrawal,
            total_portfolio_value,
            msg="Max withdrawal should be less than total portfolio value when wash sale restrictions exist"
        )
        
        # Check STOCK_A position - the older tax lot should be sold, but not the recent one (at a loss)
        stock_a_lots = modified_tax_lots[modified_tax_lots['identifier'] == STOCK_A]
        old_stock_a_quantity = stock_a_lots[stock_a_lots['tax_lot_id'] != 'lot_a_recent']['quantity'].sum()
        
        # Find sells for stock A
        sells_for_stock_a = trades[
            (trades['action'] == 'sell') & 
            (trades['identifier'] == STOCK_A)
        ]
        
        # Verify quantity sold
        self.assertEqual(
            len(sells_for_stock_a),
            0,
        )

        # Check STOCK_B position - should behave similarly to STOCK_A
        stock_b_lots = modified_tax_lots[modified_tax_lots['identifier'] == STOCK_B]
        old_stock_b_quantity = stock_b_lots[stock_b_lots['tax_lot_id'] != 'lot_b_recent']['quantity'].sum()
        
        # Find sells for stock B
        sells_for_stock_b = trades[
            (trades['action'] == 'sell') & 
            (trades['identifier'] == STOCK_B)
        ]
        
        # Verify quantity sold
        self.assertAlmostEqual(
            sells_for_stock_b['quantity'].sum(),
            old_stock_b_quantity,
            places=2
        )
        
        # Apply the trades to the portfolio and check the final state
        new_tax_lots, new_cash, recently_closed_lots = apply_trades_to_portfolio(
            tax_lots=modified_tax_lots.copy(),
            trades=trades,
            cash=strategy.cash,
            current_date=self.current_date
        )
        
        # Verify that we still have some securities left (the ones with wash sale restrictions)
        self.assertFalse(
            new_tax_lots.empty or new_tax_lots[new_tax_lots['identifier'] != CASH_CUSIP_ID].empty,
            "Should still have some securities after applying trades"
        )
        
    def test_max_withdrawl_preserve_targets(self):
        """Test max withdrawal behavior with preserve_targets flag for a large portfolio.
        
        Tests two scenarios:
        1. With preserve_targets=True: Should only sell down to 2.5% per position
        2. With preserve_targets=False: Should sell everything (to 0%)
        """
        # Create 20 different stocks with 5% target weight each
        identifiers = [f'STOCK_{i}' for i in range(1, 21)]
        target_weights = [0.05] * 20  # 5% each
        
        # Create tax lots for each stock
        tax_lots_data = []
        for i, identifier in enumerate(identifiers):
            # Each position starts at $100k (5% of $2M portfolio)
            tax_lots_data.append({
                'tax_lot_id': f'lot_{i}',
                'identifier': identifier,
                'quantity': 1000,  # 1000 shares
                'cost_basis': 100000,  # $100 per share
                'date': '2024-01-01'
            })
        
        self.tax_lots = pd.DataFrame(tax_lots_data)
        
        # Create targets DataFrame
        targets_data = []
        for identifier, weight in zip(identifiers, target_weights):
            targets_data.append({
                'asset_class': identifier,
                'identifiers': [identifier],
                'target_weight': weight
            })
        self.targets = pd.DataFrame(targets_data)
        
        # Create prices DataFrame - all stocks at $100
        prices_data = [{'identifier': identifier, 'price': 100.0} for identifier in identifiers]
        prices_data.append({'identifier': CASH_CUSIP_ID, 'price': 1.0})
        self.prices = pd.DataFrame(prices_data)
        
        # Create spreads DataFrame
        spreads_data = [{'identifier': identifier, 'spread': 0.001} for identifier in identifiers]
        spreads_data.append({'identifier': CASH_CUSIP_ID, 'spread': 0.0})
        self.spreads = pd.DataFrame(spreads_data)
        
        # Create factor model data
        factor_model_data = []
        for identifier in identifiers:
            # Add some basic factor exposures for each stock
            factor_model_data.append({
                'identifier': identifier,
                'momentum': 0.0,  # Neutral momentum
                'value': 0.0,  # Neutral value
                'quality': 0.0,  # Neutral quality
                'volatility': 0.15,  # 15% volatility
                'beta': 1.0  # Market beta of 1.0
            })
        
        # Add cash to factor model with zero factors
        factor_model_data.append({
            'identifier': CASH_CUSIP_ID,
            'momentum': 0.0,
            'value': 0.0,
            'quality': 0.0,
            'volatility': 0.0,
            'beta': 0.0
        })
        
        self.factor_model = pd.DataFrame(factor_model_data)
        
        # Update base strategy data
        self.base_strategy_data = {
            'tax_lots': self.tax_lots,
            'targets': self.targets,
            'prices': self.prices,
            'spreads': self.spreads,
            'cash': 0.0,  # Start with no cash
            'factor_model': self.factor_model  # Add validated factor model
        }
        
        # Test with preserve_targets=True
        strategy = OracleStrategy(
            **self.base_strategy_data,
            optimization_type=OracleOptimizationType.DIRECT_INDEX
        )
        strategy.min_notional = 0.0
        strategy.trade_rounding = 4
        strategy.range_min_weight_multiplier = 0.5
        strategy.range_max_weight_multiplier = 2.0
        strategy.set_oracle(self.oracle)
        
        # Calculate max withdrawal with preserve_targets=True
        max_withdrawal_preserve, trades_preserve = strategy.calculate_max_withdrawal_amount(
            respect_wash_sales=True,
            preserve_targets=True,
            debug=True
        )
        
        # Apply trades and check final positions
        new_tax_lots_preserve, new_cash_preserve, _ = apply_trades_to_portfolio(
            tax_lots=strategy.tax_lots.copy(),
            trades=trades_preserve,
            cash=strategy.cash,
            current_date=self.current_date
        )
        
        # Verify each position is at 2.5% (half of original 5%)
        for identifier in identifiers:
            position = new_tax_lots_preserve[new_tax_lots_preserve['identifier'] == identifier]
            position_value = position['quantity'].sum() * self.prices[self.prices['identifier'] == identifier]['price'].iloc[0]
            portfolio_value = strategy.total_value()
            position_weight = position_value / portfolio_value
            
            self.assertAlmostEqual(
                position_weight,
                0.025,  # Should be 2.5%
                places=3,
                msg=f"Position {identifier} should be at 2.5% when preserve_targets=True"
            )
        
        # Test with preserve_targets=False
        strategy = OracleStrategy(
            **self.base_strategy_data,
            optimization_type=OracleOptimizationType.DIRECT_INDEX
        )
        strategy.min_notional = 0.0
        strategy.trade_rounding = 4
        strategy.set_oracle(self.oracle)
        
        # Calculate max withdrawal with preserve_targets=False
        max_withdrawal_no_preserve, trades_no_preserve = strategy.calculate_max_withdrawal_amount(
            respect_wash_sales=True,
            preserve_targets=False,
            debug=True
        )
        
        # Apply trades and check final positions
        new_tax_lots_no_preserve, new_cash_no_preserve, _ = apply_trades_to_portfolio(
            tax_lots=strategy.tax_lots.copy(),
            trades=trades_no_preserve,
            cash=strategy.cash,
            current_date=self.current_date
        )
        
        # Verify all positions are sold (except cash)
        self.assertTrue(
            new_tax_lots_no_preserve.empty or 
            new_tax_lots_no_preserve[new_tax_lots_no_preserve['identifier'] != CASH_CUSIP_ID].empty,
            "All securities should be sold when preserve_targets=False"
        )
        
        # Verify all value is in cash
        total_portfolio_value = strategy.total_value()
        self.assertAlmostEqual(
            new_cash_no_preserve,
            total_portfolio_value,
            delta=0.01,
            msg="All value should be in cash when preserve_targets=False"
        )
        
