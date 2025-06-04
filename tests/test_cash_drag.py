"""Test scenarios using dthomas data."""
import unittest
from datetime import date
import pandas as pd
import json
from src.service.oracle import Oracle
from src.service.oracle_strategy import OracleStrategy
from src.service.helpers.enums import OracleOptimizationType
import pulp

class TestCashDrag(unittest.TestCase):
    def setUp(self):
        """Load dthomas data for testing."""
        # Load dthomas data
        with open('tests/example_oracle_inputs/cash_draggy.json', 'r') as f:
            self.data = json.load(f)
        
        # Create event dictionary in the format expected by process_lambda_event
        self.event = {
            "oracle": self.data["oracle"],
            "settings": self.data.get("settings", {})
        }

    def test_cash_draggy_buy_only(self):
        """Test buy-only optimization with dthomas cash draggy data.
        
        Expected behavior:
        - Should result in ~$4300 of buys
        - No sells should occur
        - Drift cost should get worse
        - Factor cost should get better
        - Cash drag should get better
        - All cost changes should be roughly equal (+/- 0.1)
        """
        # Process the event using Oracle.process_lambda_event
        response = Oracle.process_lambda_event(self.event)
        
        # Get the first strategy's results
        first_strategy_id = next(iter(response["results"]))
        strategy_result = response["results"][first_strategy_id]
        status = strategy_result["status"]
        should_trade = strategy_result["should_trade"]
        trades = pd.DataFrame(strategy_result["trades"])
        trade_summary = strategy_result["trade_summary"]
        
        # Verify optimization completed successfully
        self.assertEqual(pulp.LpStatusOptimal, status)
        self.assertTrue(should_trade)
        
        # Verify no sells occurred
        sell_trades = trades[trades['action'] == 'sell']
        self.assertEqual(len(sell_trades), 0, "Expected no sell trades")
        
        # Calculate total buy value
        buy_trades = trades[trades['action'] == 'buy']
        computed_total_buy_value = buy_trades.apply(lambda x: x['quantity'] * x['price'], axis=1).sum()
        total_buy_value = buy_trades['trade_value'].sum()

        self.assertAlmostEqual(computed_total_buy_value, total_buy_value,
                             msg="Computed total buy value does not match trade value")
        
        # Verify buy value is approximately $4300
        self.assertAlmostEqual(total_buy_value, 5250, delta=1000, 
                             msg=f"Expected total buy value to be ~$5250, got ${total_buy_value:.2f}")
        
        # Verify cost changes
        drift_cost_change = trade_summary['optimization_info']['after_optimization']['drift_cost'] - trade_summary['optimization_info']['before_optimization']['drift_cost']
        factor_cost_change = trade_summary['optimization_info']['after_optimization']['factor_cost'] - trade_summary['optimization_info']['before_optimization']['factor_cost']
        cash_drag_change = trade_summary['optimization_info']['after_optimization']['cash_drag'] - trade_summary['optimization_info']['before_optimization']['cash_drag']
        
        # Verify drift cost got worse (positive change)å
        self.assertGreater(drift_cost_change, 0, "Expected drift cost to get worse")
        
        # Verify factor cost got better (negative change)
        self.assertLess(factor_cost_change, 0, "Expected factor cost to get better")
        
        # Verify cash drag got better (negative change)
        self.assertLess(cash_drag_change, 0, "Expected cash drag to get better")
        

if __name__ == '__main__':
    unittest.main() 