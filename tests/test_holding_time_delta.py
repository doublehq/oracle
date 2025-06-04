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

class TestHoldingTimeDelta(unittest.TestCase):

    def setUp(self):
        """Set up test data for two consecutive days."""
        # Setup dates and initial cash
        self.day1 = date(2025, 1, 1)
        self.initial_cash = 100000.0

        # Prices - same for both days to test no-change scenario
        self.prices_data = {
            'identifier': [STOCK_A, STOCK_B, CASH_CUSIP_ID],
            'price': [100.0, 50.0, 1.0]
        }
        self.prices_day1 = pd.DataFrame(self.prices_data)
        self.prices_day2 = pd.DataFrame(self.prices_data)  # Same prices for day 2

        # Targets (Sum to 1.0)
        self.targets_data = {
            'asset_class': [STOCK_A, STOCK_B, CASH_CUSIP_ID],
            'identifiers': [[STOCK_A], [STOCK_B], [CASH_CUSIP_ID]],
            'target_weight': [0.60, 0.39, 0.01]
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
            optimization_type=OracleOptimizationType.TAX_AWARE
        )

        # Set up Oracle-Strategy relationship
        self.strategy_day1.set_oracle(self.oracle_day1)
        self.oracle_day1.strategies = [self.strategy_day1]
        self.oracle_day1.initialize_wash_sale_restrictions()

    def test_holding_time_restriction(self):
        """Test holding time restrictions over four days.
        
        Scenario:
        Day 1: Buy initial positions (trades allowed)
        Day 2: Try to trade but blocked by holding time (no trades)
        Day 3: After holding time expires, trades allowed again
        Day 4: Try to trade but blocked by holding time (no trades)
        """
        from datetime import timedelta
        
        # === Day 1: Initial buy ===
        status_day1, should_trade_day1, trade_summary_day1, trades_day1 = self.strategy_day1.compute_optimal_trades(
            debug=False,
            rebalance_threshold=0.0,
            holding_time_days=1  # Set 1-day holding restriction
        )
        self.assertEqual(pulp.LpStatusOptimal, status_day1)
        self.assertTrue(should_trade_day1)
        self.assertEqual(2, len(trades_day1))
        
        # Apply Day 1 trades
        tax_lots_day1, cash_day1, recently_closed_day1 = apply_trades_to_portfolio(
            tax_lots=self.strategy_day1.tax_lots,
            trades=trades_day1,
            cash=self.strategy_day1.cash,
            current_date=self.day1
        )
        
        # === Day 2: Try to trade (should be restricted) ===
        day2 = date(2025, 1, 2)  # One day after initial trades
        
        # Create new targets that would normally trigger trades
        new_targets = pd.DataFrame({
            'asset_class': [STOCK_A, STOCK_B, CASH_CUSIP_ID],
            'identifiers': [[STOCK_A], [STOCK_B], [CASH_CUSIP_ID]],
            'target_weight': [0.50, 0.29, 0.21]  # Significantly different from initial targets
        })
        
        oracle_day2 = Oracle(
            current_date=day2,
            tax_rates=self.tax_rates,
            recently_closed_lots=recently_closed_day1
        )

        strategy_day2 = OracleStrategy(
            tax_lots=tax_lots_day1,
            targets=new_targets,
            prices=self.prices_day1.copy(),
            spreads=self.spreads,
            cash=0.0,
            optimization_type=OracleOptimizationType.TAX_AWARE
        )
        strategy_day2.set_oracle(oracle_day2)
        oracle_day2.strategies = [strategy_day2]
        oracle_day2.initialize_wash_sale_restrictions()
        
        status_day2, should_trade_day2, trade_summary_day2, trades_day2 = strategy_day2.compute_optimal_trades(
            debug=False,
            rebalance_threshold=0.0,
            holding_time_days=1
        )
        self.assertFalse(should_trade_day2)
        self.assertTrue(trades_day2.empty, "Day 2 should have no trades due to holding time restriction")
        
        # === Day 3: Holding time expired, trades allowed ===
        day3 = date(2025, 1, 3)  # Three days after initial trades
        
        oracle_day3 = Oracle(
            current_date=day3,
            tax_rates=self.tax_rates,
            recently_closed_lots=recently_closed_day1
        )

        # Use same targets as day 2 since we want to achieve those targets now that holding period has expired
        strategy_day3 = OracleStrategy(
            tax_lots=tax_lots_day1,
            targets=new_targets,  # Original targets, not adjusted, since we can now sell
            prices=self.prices_day1.copy(),
            spreads=self.spreads,
            cash=0.0,
            optimization_type=OracleOptimizationType.TAX_AWARE
        )
        strategy_day3.set_oracle(oracle_day3)
        oracle_day3.strategies = [strategy_day3]
        oracle_day3.initialize_wash_sale_restrictions()
        
        status_day3, should_trade_day3, trade_summary_day3, trades_day3 = strategy_day3.compute_optimal_trades(
            debug=False,
            rebalance_threshold=0.0,
            holding_time_days=1
        )
        self.assertEqual(pulp.LpStatusOptimal, status_day3)
        self.assertTrue(should_trade_day3)
        self.assertFalse(trades_day3.empty, "Day 3 should have trades as holding time has expired")
        
        # Apply Day 3 trades
        tax_lots_day3, cash_day3, recently_closed_day3 = apply_trades_to_portfolio(
            tax_lots=tax_lots_day1,
            trades=trades_day3,
            cash=cash_day1,
            current_date=day3
        )
        

        # === Day 4: Try to trade (should be restricted again) ===
        day4 = date(2025, 1, 6)  # One trading day after second set of trades
        
        # Create another set of targets
        final_targets = pd.DataFrame({
            'asset_class': [STOCK_A, STOCK_B, CASH_CUSIP_ID],
            'identifiers': [[STOCK_A], [STOCK_B], [CASH_CUSIP_ID]],
            'target_weight': [0.20, 0.19, 0.61]  # Different targets again
        })
        
        oracle_day4 = Oracle(
            current_date=day4,
            tax_rates=self.tax_rates,
            recently_closed_lots=recently_closed_day3
        )

        strategy_day4 = OracleStrategy(
            tax_lots=tax_lots_day3,
            targets=final_targets,
            prices=self.prices_day1.copy(),
            spreads=self.spreads,
            cash=0.0,
            optimization_type=OracleOptimizationType.TAX_AWARE
        )
        strategy_day4.set_oracle(oracle_day4)
        oracle_day4.strategies = [strategy_day4]
        oracle_day4.initialize_wash_sale_restrictions()
        
        status_day4, should_trade_day4, trade_summary_day4, trades_day4 = strategy_day4.compute_optimal_trades(
            debug=False,
            holding_time_days=1
        )
        self.assertEqual(pulp.LpStatusOptimal, status_day4)
        self.assertTrue(should_trade_day4)
        self.assertFalse(trades_day4.empty, "Day 4 should have trades as sells are not restricted by holding time, just buys and we bought these on the 1st.")
        
        # === Additional Assertions ===
        # Get buys from day 1
        buys_day1 = trades_day1[trades_day1['action'] == 'buy']
        buy_ids_day1 = set(buys_day1['identifier'])

        # Get identifiers from day 3 trades
        trades_day3_ids = set(trades_day3['identifier'])

        self.assertEqual(buy_ids_day1, {STOCK_A, STOCK_B}, "Day 1 buys should match target stocks")
        self.assertTrue(len(trades_day3_ids) > 0, "Day 3 should have trades for at least one stock")

if __name__ == '__main__':
    unittest.main()