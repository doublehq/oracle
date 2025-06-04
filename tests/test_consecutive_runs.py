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

class TestConsecutiveRuns(unittest.TestCase):

    def setUp(self):
        """Set up test data for two consecutive days."""
        # Setup dates and initial cash
        self.day1 = date(2023, 1, 1)
        self.day2 = date(2023, 1, 3)
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
            'asset_class': ['STOCK_A', 'STOCK_B', CASH_CUSIP_ID],
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
            optimization_type=OracleOptimizationType.TAX_AWARE,
        )

        # Set up Oracle-Strategy relationship
        self.strategy_day1.set_oracle(self.oracle_day1)
        self.oracle_day1.strategies = [self.strategy_day1]
        self.oracle_day1.initialize_wash_sale_restrictions()

    def test_consecutive_runs_start_cash(self):
        """Test running optimizer on day 1 (start cash), applying trades, then running on day 2."""
        # === Day 1: Run optimization ===
        status_day1, should_trade_day1, trade_summary_day1, trades_day1 = self.strategy_day1.compute_optimal_trades(debug=False)
        self.assertEqual(pulp.LpStatusOptimal, status_day1)
        self.assertTrue(should_trade_day1)
        self.assertEqual(2, len(trades_day1))

        # Get buys and sells from day 1
        buys_day1 = trades_day1[trades_day1['action'] == 'buy']
        sells_day1 = trades_day1[trades_day1['action'] == 'sell']
        
        # Get identifiers bought on day 1
        buy_ids_day1 = set(buys_day1['identifier'])

        # Get buys from day 1 and 2
        buys_day1 = trades_day1[trades_day1['action'] == 'buy'] if not trades_day1.empty else trades_day1

        # Get buys from day 1
        buys_day1 = trades_day1[trades_day1['action'] == 'buy'] if not trades_day1.empty else trades_day1
        


        self.assertFalse(buys_day1.empty, "Day 1 should have buy trades")
        self.assertTrue(sells_day1.empty, "Day 1 should have no sell trades (starting from cash)")
        self.assertEqual(buy_ids_day1, {STOCK_A, STOCK_B}, "Day 1 buys should match target stocks")

        # === Apply Trades (using standalone function) ===
        updated_tax_lots, updated_cash, new_recently_closed_lots = apply_trades_to_portfolio(
            tax_lots=self.strategy_day1.tax_lots,
            trades=trades_day1,
            cash=self.strategy_day1.cash,
            current_date=self.oracle_day1.current_date
        )

        # --- Assert intermediate state ---
        self.assertNotEqual(self.initial_cash, updated_cash, "Cash should have changed after Day 1 trades")
        self.assertFalse(updated_tax_lots.empty, "Tax lots should not be empty after Day 1 buys")
        self.assertTrue(all(col in updated_tax_lots.columns for col in ['identifier', 'quantity', 'cost_basis', 'date', 'tax_lot_id']), "Updated tax lots have correct columns")

        # === Day 2: Create new Oracle and Strategy with updated state ===
        oracle_day2 = Oracle(
            current_date=self.day2,
            tax_rates=self.tax_rates,
            recently_closed_lots=new_recently_closed_lots,
            stock_restrictions=self.oracle_day1.stock_restrictions,
        )

        strategy_day2 = OracleStrategy(
            tax_lots=updated_tax_lots,
            targets=self.targets.copy(),
            prices=self.prices_day2.copy(),  # Same prices as Day 1
            spreads=self.spreads,
            cash=updated_cash,
            optimization_type=OracleOptimizationType.TAX_AWARE
        )

        # Set up Oracle-Strategy relationship for Day 2
        strategy_day2.set_oracle(oracle_day2)
        oracle_day2.strategies = [strategy_day2]
        oracle_day2.initialize_wash_sale_restrictions()

        status_day2, should_trade_day2, trade_summary_day2, trades_day2 = strategy_day2.compute_optimal_trades(debug=False)
        self.assertEqual(pulp.LpStatusOptimal, status_day2)
        self.assertFalse(should_trade_day2)

        # Since prices haven't changed, we expect no trades on Day 2
        self.assertTrue(trades_day2.empty, "Day 2 should generate no trades since prices haven't changed")

        # Verify that cash and positions remain unchanged
        self.assertEqual(strategy_day2.cash, updated_cash, "Cash should remain unchanged on Day 2")
    
    def test_buy_buy_sell_wash_sale(self):
        """Test wash sale restrictions when buying twice then attempting to sell.
        
        Scenario:
        Day 1: Buy initial position in both stocks
        Day 2: Add cash and buy more of both stocks
        Day 3: Attempt to sell - should be restricted by wash sale rules
        """
        # === Day 1: Initial buy ===
        status_day1, should_trade_day1, trade_summary_day1, trades_day1 = self.strategy_day1.compute_optimal_trades(debug=False, rebalance_threshold=0.0, min_notional=0.1)
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
        
        # === Day 2: Add cash and buy more ===
        day2 = date(2023, 1, 2)
        additional_cash = 50000.0  # Add more cash for second purchase
        
        oracle_day2 = Oracle(
            current_date=day2,
            tax_rates=self.tax_rates,
            recently_closed_lots=recently_closed_day1
        )
        
        strategy_day2 = OracleStrategy(
            tax_lots=tax_lots_day1,
            targets=self.targets.copy(),
            prices=self.prices_day1.copy(),  # Keep same prices
            spreads=self.spreads,
            cash=cash_day1 + additional_cash,  # Add more cash
            optimization_type=OracleOptimizationType.TAX_AWARE
        )
        strategy_day2.set_oracle(oracle_day2)
        oracle_day2.strategies = [strategy_day2]
        oracle_day2.initialize_wash_sale_restrictions()
        
        status_day2, should_trade_day2, trade_summary_day2, trades_day2 = strategy_day2.compute_optimal_trades(debug=False, rebalance_threshold=0.0)
        self.assertEqual(pulp.LpStatusOptimal, status_day2)
        self.assertTrue(should_trade_day2)
        self.assertEqual(2, len(trades_day2))
        
        # Apply Day 2 trades
        tax_lots_day2, cash_day2, recently_closed_day2 = apply_trades_to_portfolio(
            tax_lots=tax_lots_day1,
            trades=trades_day2,
            cash=cash_day1 + additional_cash,
            current_date=day2
        )
        
        # === Day 3: Try to sell (should be restricted) ===
        day3 = date(2023, 1, 5)
        
        # Create new targets that would normally trigger sells
        sell_targets = pd.DataFrame({
            'asset_class': ['STOCK_A', 'STOCK_B', CASH_CUSIP_ID],
            'identifiers': [[STOCK_A], [STOCK_B], [CASH_CUSIP_ID]],
            'target_weight': [0.30, 0.20, 0.50]  # Much higher cash target to encourage selling
        })
        
        oracle_day3 = Oracle(
            current_date=day3,
            tax_rates=self.tax_rates,
            recently_closed_lots=recently_closed_day2
        )
        
        strategy_day3 = OracleStrategy(
            tax_lots=tax_lots_day2,
            targets=sell_targets,
            prices=self.prices_day1.copy(),  # Keep same prices
            spreads=self.spreads,
            cash=cash_day2,
            optimization_type=OracleOptimizationType.TAX_AWARE
        )
        strategy_day3.set_oracle(oracle_day3)
        oracle_day3.strategies = [strategy_day3]
        oracle_day3.initialize_wash_sale_restrictions()
        
        status_day3, should_trade_day3, trade_summary_day3, trades_day3 = strategy_day3.compute_optimal_trades(debug=False, rebalance_threshold=0.0)
        self.assertTrue(len(trades_day3) == 0)
        
        # === Assertions ===
        # Day 1 should have buys
        self.assertFalse(trades_day1.empty, "Day 1 should have trades")
        buys_day1 = trades_day1[trades_day1['action'] == 'buy'] if not trades_day1.empty else trades_day1
        self.assertFalse(buys_day1.empty, "Day 1 should have buy trades")
        
        # Day 2 should have additional buys
        self.assertFalse(trades_day2.empty, "Day 2 should have trades")
        buys_day2 = trades_day2[trades_day2['action'] == 'buy'] if not trades_day2.empty else trades_day2
        self.assertFalse(buys_day2.empty, "Day 2 should have buy trades")
        
        # Day 3 should have no trades due to wash sale restrictions
        if not trades_day3.empty:
            sells_day3 = trades_day3[trades_day3['action'] == 'sell']
            self.assertTrue(sells_day3.empty, "Day 3 should have no sell trades due to wash sale restrictions")
        else:
            self.assertTrue(trades_day3.empty, "Day 3 should have no trades at all due to wash sale restrictions")

    def test_buy_sell_buy_wash_sale(self):
        """Test wash sale restrictions when selling then attempting to buy back.
        
        Scenario:
        Day 1: Buy initial position in both stocks
        Day 2: Sell STOCK_A
        Day 3: Add cash and attempt to buy - STOCK_A should be restricted, but STOCK_B can be bought
        """
        # === Day 1: Initial buy ===
        status_day1, should_trade_day1, trade_summary_day1, trades_day1 = self.strategy_day1.compute_optimal_trades(debug=False, rebalance_threshold=0.0, min_notional=0.1)
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
        
        # === Day 2: Sell STOCK_A ===
        day2 = date(2023, 1, 3)
        
        # Create targets that will trigger selling STOCK_A
        sell_targets = pd.DataFrame({
            'asset_class': ['STOCK_A', 'STOCK_B', CASH_CUSIP_ID],
            'identifiers': [[STOCK_A], [STOCK_B], [CASH_CUSIP_ID]],
            'target_weight': [0.0, 0.40, 0.60] # Remove STOCK_A entirely
        })
        
        oracle_day2 = Oracle(
            current_date=day2,
            tax_rates=self.tax_rates,
            recently_closed_lots=recently_closed_day1
        )

        prices_data2 = pd.DataFrame({
            'identifier': [STOCK_A, STOCK_B, CASH_CUSIP_ID],
            'price': [99.99, 50.0, 1.0]
        })
        
        strategy_day2 = OracleStrategy(
            tax_lots=tax_lots_day1,
            targets=sell_targets,
            prices=prices_data2,  # Price must go down slightly for TLH sell to be restricted
            spreads=self.spreads,
            cash=cash_day1,
            optimization_type=OracleOptimizationType.TAX_AWARE  
        )
        strategy_day2.set_oracle(oracle_day2)
        oracle_day2.strategies = [strategy_day2]
        oracle_day2.initialize_wash_sale_restrictions()
        
        status_day2, should_trade_day2, trade_summary_day2, trades_day2 = strategy_day2.compute_optimal_trades(debug=False, rebalance_threshold=0.0, min_notional=0.1)
        self.assertEqual(pulp.LpStatusOptimal, status_day2)
        self.assertTrue(should_trade_day2)
        sells = trades_day2[trades_day2['action'] == 'sell']
        self.assertFalse(sells.empty, "Day 2 should have sell trades")
        self.assertEqual(len(sells), 1, "Day 2 should have one sell trade")        
        
        
        # Apply Day 2 trades
        tax_lots_day2, cash_day2, recently_closed_day2 = apply_trades_to_portfolio(
            tax_lots=tax_lots_day1,
            trades=trades_day2,
            cash=cash_day1,
            current_date=day2
        )
        
        # === Day 3: Try to buy back (STOCK_A should be restricted) ===
        day3 = date(2023, 1, 5)
        additional_cash = 50000.0  # Add more cash
        
        # Return to original targets
        oracle_day3 = Oracle(
            current_date=day3,
            tax_rates=self.tax_rates,
            recently_closed_lots=recently_closed_day2,
        )
        
        strategy_day3 = OracleStrategy(
            tax_lots=tax_lots_day2,
            targets=self.targets.copy(),  # Back to original targets
            prices=self.prices_day1.copy(),
            spreads=self.spreads,
            cash=cash_day2 + additional_cash,
            optimization_type=OracleOptimizationType.TAX_AWARE
        )
        strategy_day3.set_oracle(oracle_day3)
        oracle_day3.strategies = [strategy_day3]
        oracle_day3.initialize_wash_sale_restrictions()
        
        status_day3, should_trade_day3, trade_summary_day3, trades_day3 = strategy_day3.compute_optimal_trades(debug=False, rebalance_threshold=0.0)
        self.assertEqual(pulp.LpStatusOptimal, status_day3)
        self.assertTrue(should_trade_day3)
        self.assertEqual(1, len(trades_day3))
        
        # === Assertions ===
        # Day 1 should have buys
        self.assertFalse(trades_day1.empty, "Day 1 should have trades")
        buys_day1 = trades_day1[trades_day1['action'] == 'buy'] if not trades_day1.empty else trades_day1
        self.assertFalse(buys_day1.empty, "Day 1 should have buy trades")
        
        # Day 2 should have sells for STOCK_A
        self.assertFalse(trades_day2.empty, "Day 2 should have trades")
        sells_day2 = trades_day2[trades_day2['action'] == 'sell']
        self.assertFalse(sells_day2.empty, "Day 2 should have sell trades")
        sold_ids = set(sells_day2['identifier'])
        self.assertEqual(sold_ids, {STOCK_A}, "Only STOCK_A should be sold")
        
        # Day 3 should only have buys for STOCK_B (STOCK_A restricted)
        buys_day3 = trades_day3[trades_day3['action'] == 'buy']
        bought_ids = set(buys_day3['identifier'])
        self.assertEqual(bought_ids, {STOCK_B}, "Only STOCK_B should be bought (STOCK_A restricted)")

if __name__ == '__main__':
    unittest.main()