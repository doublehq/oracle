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

class TestMinNotional(unittest.TestCase):

    def setUp(self):
        """Set up test data for two consecutive days."""
        # Setup dates and initial cash
        self.day1 = date(2023, 1, 1)
        self.day2 = date(2023, 1, 3)
        self.initial_cash = 10.0

        # Prices - same for both days to test no-change scenario
        self.prices_data = {
            'identifier': [STOCK_A, STOCK_B, CASH_CUSIP_ID],
            'price': [50.0, 50.0, 1.0]
        }
        self.prices_day1 = pd.DataFrame(self.prices_data)
        self.prices_day2 = pd.DataFrame(self.prices_data)  # Same prices for day 2

        # Targets (Sum to 1.0)
        self.targets_data = {
            'asset_class': [STOCK_A, STOCK_B, CASH_CUSIP_ID],
            'identifiers': [[STOCK_A], [STOCK_B], [CASH_CUSIP_ID]],
            'target_weight': [0.60, 0.40, 0.00]
        }
        self.targets = pd.DataFrame(self.targets_data)

        # Empty initial tax lots
        self.initial_tax_lots = pd.DataFrame(columns=['identifier', 'quantity', 'cost_basis', 'date'])

        # Spreads (use defaults by passing None)
        self.spreads = None

        # Assumed Tax Rates (using defaults in OracleStrategy for simplicity)
        self.tax_rates = None

        # Initialize Day 1 Oracle and Strategy
        self.oracle_day1 = Oracle(
            current_date=self.day1,
            tax_rates=self.tax_rates
        )

        # Create Day 1 strategy
        self.strategy_day1 = OracleStrategy(
            tax_lots=self.initial_tax_lots.copy(),
            targets=self.targets.copy(),
            prices=self.prices_day1.copy(),
            spreads=self.spreads,
            cash=self.initial_cash,
            optimization_type=OracleOptimizationType.TAX_AWARE,
        )

        # Set up Oracle-Strategy relationship
        self.strategy_day1.set_oracle(self.oracle_day1)
        self.oracle_day1.strategies = [self.strategy_day1]
        self.oracle_day1.initialize_wash_sale_restrictions()

    def test_min_notional(self):
        """
        Test that trades respect the minimum notional constraint.
        
        Given:
        - $10 portfolio starting in cash
        - Target weights: 60% Stock A ($6), 40% Stock B ($4), 0% Cash ($0.0)
        - Stock A price = $50, Stock B price = $50
        - $5 minimum notional
        
        Expected:
        - Should buy exactly $5 of each stock (optimal solution)
        - Should not make any trades below $5 notional
        - Should still find a feasible solution
        - Final portfolio should be closer to targets while respecting min notional
        """
        # Run optimization with $5 min notional and a desire to invest.
        status_day1, should_trade_day1, trade_summary_day1, trades_day1 = self.strategy_day1.compute_optimal_trades(
            weight_tax=0.0,
            weight_drift=1.0,
            weight_transaction=0.0,
            min_notional=5.0,
            weight_cash_drag=1.0, 
            debug=False
        )
        
        # Verify optimization succeeded
        self.assertEqual(status_day1, pulp.LpStatusOptimal)
        self.assertTrue(should_trade_day1)
        
        # Verify we got exactly two trades (one for each stock)
        self.assertEqual(len(trades_day1), 2, "Should have exactly two trades, one for each stock")
        
        # Extract trades for each stock
        stock_a_trades = trades_day1[trades_day1['identifier'] == STOCK_A]
        stock_b_trades = trades_day1[trades_day1['identifier'] == STOCK_B]
        
        # Verify we have exactly one trade for each stock
        self.assertEqual(len(stock_a_trades), 1, "Should have exactly one trade for STOCK_A")
        self.assertEqual(len(stock_b_trades), 1, "Should have exactly one trade for STOCK_B")
        
        # Verify both trades are buys
        self.assertEqual(stock_a_trades['action'].iloc[0], 'buy', "STOCK_A trade should be a buy")
        self.assertEqual(stock_b_trades['action'].iloc[0], 'buy', "STOCK_B trade should be a buy")
        
        # Verify each trade is exactly $5
        self.assertAlmostEqual(stock_a_trades['trade_value'].sum(), 5.0, places=2, msg="STOCK_A trade should be exactly $5")
        self.assertAlmostEqual(stock_b_trades['trade_value'].sum(), 5.0, places=2, msg="STOCK_B trade should be exactly $5")
        
        # Check each trade meets minimum notional
        for _, trade in trades_day1.iterrows():
            notional = abs(trade['quantity'] * trade['price'])
            # If it's a real trade (not just a rounding error), it should meet min notional
            if notional > 0.01:  # Filter out tiny rounding error trades
                self.assertGreaterEqual(
                    notional, 
                    5.0, 
                    f"Trade notional {notional} is below minimum 5.0 for {trade['identifier']}"
                )
        
        # Apply the trades to see final portfolio state
        updated_tax_lots, updated_cash, recently_closed_lots = apply_trades_to_portfolio(
            tax_lots=self.initial_tax_lots.copy(),
            trades=trades_day1,
            cash=self.initial_cash,
            current_date=self.day1
        )
        
        # Verify we're not holding any positions with less than $5 market value
        # Group tax lots by identifier to get total position sizes
        if not updated_tax_lots.empty:
            position_sizes = updated_tax_lots.groupby('identifier').agg({
                'cost_basis': 'sum'
            }).reset_index()
            
            for _, position in position_sizes.iterrows():
                if position['identifier'] != CASH_CUSIP_ID:
                    self.assertGreaterEqual(
                        position['cost_basis'],
                        5.0,
                        f"Position size {position['cost_basis']} is below minimum 5.0 for {position['identifier']}"
                    )
                    
            # Verify final position sizes are exactly $5 each
            for _, position in position_sizes.iterrows():
                if position['identifier'] != CASH_CUSIP_ID:
                    self.assertAlmostEqual(
                        position['cost_basis'],
                        5.0,
                        places=2,
                        msg=f"Position size for {position['identifier']} should be exactly $5"
                    )
            
            # Verify remaining cash is exactly $0
            self.assertAlmostEqual(updated_cash, 0.0, places=2, msg="Should have no cash remaining")

    def test_min_notional_multiple_lots_less_than_min_notional(self):
        tax_lots = pd.DataFrame([
            {
                'identifier': STOCK_A,
                'quantity': 0.05,
                'cost_basis': 1.0,
                'date': self.day1,
            },
            {
                'identifier': STOCK_A,
                'quantity': 0.05,
                'cost_basis': 1.0,
                'date': self.day1,
            },
        ])

        targets = pd.DataFrame([
            {
                'asset_class': STOCK_A,
                'identifiers': [STOCK_A],
                'target_weight': 0,
            },
            {
                'asset_class': CASH_CUSIP_ID,
                'identifiers': [CASH_CUSIP_ID],
                'target_weight': 1,
            },
        ])

        strategy = OracleStrategy(
            tax_lots=tax_lots,
            targets=targets,
            prices=self.prices_day1.copy(),
            spreads=self.spreads,
            cash=self.initial_cash,
            optimization_type=OracleOptimizationType.TAX_AWARE,
        )

        strategy.set_oracle(self.oracle_day1)
        self.oracle_day1.strategies = [strategy]
        self.oracle_day1.initialize_wash_sale_restrictions()

        _, _, _, trades = strategy.compute_optimal_trades(
            weight_tax=0.0,
            weight_drift=1.0,
            weight_transaction=0.0,
            min_notional=5.0,
            weight_cash_drag=1.0, 
            debug=False
        )

        self.assertEqual(len(trades), 2)
        self.assertAlmostEqual(trades['trade_value'].sum(), 5.0, places=2)

        _, _, _, trades = strategy.compute_optimal_trades(
            weight_tax=0.0,
            weight_drift=1.0,
            weight_transaction=0.0,
            min_notional=5.1,
            weight_cash_drag=1.0, 
            debug=False
        )

        self.assertEqual(len(trades), 0)
