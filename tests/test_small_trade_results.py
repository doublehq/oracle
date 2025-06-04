import unittest
import json
import pandas as pd
from pathlib import Path
from decimal import Decimal

from src.service.oracle import Oracle
from src.service.oracle_strategy import OracleStrategy
from src.service.helpers.enums import OracleOptimizationType

class TestSmallTradeResults(unittest.TestCase):
    """Test that trades meet minimum notional and decimal place requirements."""

    def setUp(self):
        """Set up test data."""
        self.test_data_dir = Path('tests/example_oracle_inputs')
        self.min_notional = 5.1  # Minimum notional value for trades
        self.decimal_places = 4  # Required decimal places for rounding

    def test_min_notional_trades(self):
        """
        Test that trades from min_notional_trades.json:
        1. For buys: Each trade is greater than 5.1 min notional
        2. For sells: Sum of notional values for same identifier is greater than 5.1
        3. All trades are rounded to 4 decimal places
        """
        # Load test data
        json_file ='tests/example_oracle_inputs/min_notional_trades.json'

        with open(json_file, 'r') as f:
            self.test_data = json.load(f)

        self.event = {
            "oracle": self.test_data["oracle"],
            "settings": self.test_data.get("settings", {})
        }
        # Process the event using Oracle.process_lambda_event
        response = Oracle.process_lambda_event(self.event)

        # Get the first strategy's results
        first_strategy_id = next(iter(response["results"]))
        strategy_result = response["results"][first_strategy_id]
        status = strategy_result["status"]
        should_trade = strategy_result["should_trade"]
        trades = pd.DataFrame(strategy_result["trades"])
        trade_summary = strategy_result["trade_summary"]
        
        # Verify we got trades
        self.assertIsNotNone(trades)
        self.assertFalse(trades.empty)

        # Group sells by identifier to sum their notional values
        sells = trades[trades['action'] == 'sell'].copy()
        if not sells.empty:
            sells['notional'] = abs(sells['quantity'] * sells['price'])
            sell_totals = sells.groupby('identifier')['notional'].sum()
            
            # Check that each sell group meets minimum notional
            for identifier, total_notional in sell_totals.items():
                self.assertGreaterEqual(
                    total_notional,
                    self.min_notional,
                    f"Total sell notional {total_notional} for {identifier} is below minimum {self.min_notional}"
                )

        # Check buys individually
        buys = trades[trades['action'] == 'buy']
        for _, trade in buys.iterrows():
            notional = abs(float(trade['quantity']) * float(trade['price']))
            
            # Skip tiny rounding error trades
            if notional > 0.01:
                self.assertGreaterEqual(
                    notional,
                    self.min_notional,
                    f"Buy trade notional {notional} is below minimum {self.min_notional} for {trade['identifier']}"
                )

        # Check decimal places for all trades
        for _, trade in trades.iterrows():
            quantity_str = str(trade['quantity'])
            if '.' in quantity_str:
                decimal_part = quantity_str.split('.')[1]
                self.assertLessEqual(
                    len(decimal_part),
                    self.decimal_places,
                    f"Trade quantity {trade['quantity']} has more than {self.decimal_places} decimal places"
                )

            # Verify quantity is properly rounded
            rounded_quantity = round(float(trade['quantity']), self.decimal_places)
            self.assertEqual(
                float(trade['quantity']),
                rounded_quantity,
                f"Trade quantity {trade['quantity']} is not properly rounded to {self.decimal_places} decimal places"
            )
