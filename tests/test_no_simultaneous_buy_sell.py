"""Test that optimization does not result in buys and sells of the same security."""
import unittest
import pandas as pd
import json
from src.service.oracle import Oracle
import pulp

class TestNoSimultaneousBuySell(unittest.TestCase):
    def setUp(self):
        """Load dthomas data for testing."""
        # Load dthomas data
        with open('tests/example_oracle_inputs/duplicate_trades.json', 'r') as f:
            self.data = json.load(f)
        
        # Create event dictionary in the format expected by process_lambda_event
        self.event = {
            "oracle": self.data["oracle"],
            "settings": self.data.get("settings", {})
        }

    def test_no_simultaneous_buy_sell(self):
        """Test that no security is both bought and sold in the same optimization."""
        # Process the event using Oracle.process_lambda_event
        response = Oracle.process_lambda_event(self.event)
        
        # Get the first strategy's results
        first_strategy_id = next(iter(response["results"]))
        strategy_result = response["results"][first_strategy_id]
        status = strategy_result["status"]
        should_trade = strategy_result["should_trade"]
        trades = pd.DataFrame(strategy_result["trades"])
        
        # Verify optimization completed successfully or was feasible
        # (Allowing Feasible as well, in case the optimal solution isn't strictly found but is usable)
        self.assertEqual(status, pulp.LpStatusOptimal)

        if trades.empty:
            return

        # Group trades by security_id
        grouped_trades = trades.groupby('identifier')['action'].apply(set)

        # Check if any security has both 'buy' and 'sell' actions
        simultaneous_buy_sell = grouped_trades[grouped_trades.apply(lambda x: 'buy' in x and 'sell' in x)]

        self.assertTrue(simultaneous_buy_sell.empty, 
                        f"Found simultaneous buy and sell actions for securities: {list(simultaneous_buy_sell.index)}")

        print("Test passed: No security found with both buy and sell actions.")

if __name__ == '__main__':
    unittest.main() 