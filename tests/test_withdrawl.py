import unittest
from datetime import date, datetime, timedelta
import pandas as pd
from decimal import Decimal
from src.service.helpers.enums import OracleOptimizationType
from src.service.oracle import Oracle
from src.service.helpers.constants import CASH_CUSIP_ID

class TestWithdrawl(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test case."""
        # Set a fixed date for tests
        self.test_date = date(2023, 1, 15)
        
        # Create Oracle instance
        self.oracle = Oracle(current_date=self.test_date)
        
        # Set tax rates for the test - must have this for optimization
        self.oracle.tax_rates = pd.DataFrame([
            {'gain_type': 'short_term', 'federal_rate': 0.35, 'state_rate': 0.06, 'total_rate': 0.41},
            {'gain_type': 'long_term', 'federal_rate': 0.20, 'state_rate': 0.06, 'total_rate': 0.26},
            {"gain_type": "qualified_dividend", "federal_rate": 0.15, "state_rate": 0.06, "total_rate": 0.21}
        ])
    
    def test_withdrawl(self):
        """Test that withdrawal amount works correctly."""
        # Create a simple portfolio with 2 stocks ($800) and some cash ($200)
        tax_lots = pd.DataFrame({
            'tax_lot_id': ['lot1', 'lot2'],
            'identifier': ['AAPL', 'MSFT'],
            'quantity': [10, 10],
            'date': [date(2022, 1, 1), date(2022, 1, 1)],  # Over 1 year ago (long-term)
            'cost_basis': [400.00, 400.00]
        })
        
        # Current prices - each lot is worth $400
        prices = pd.DataFrame({
            'identifier': ['AAPL', 'MSFT', CASH_CUSIP_ID],
            'price': [40.00, 40.00, 1.00]
        })
        
        # Target weights (equal weight for both stocks, some cash)
        targets = pd.DataFrame([
            {'asset_class': 'Tech1', 'identifiers': ['AAPL'], 'target_weight': 0.4},
            {'asset_class': 'Tech2', 'identifiers': ['MSFT'], 'target_weight': 0.4},
            {'asset_class': CASH_CUSIP_ID, 'identifiers': [CASH_CUSIP_ID], 'target_weight': 0.2}
        ])
        
        # Start with $200 in cash
        cash = 200.00
        
        # Create strategy with $500 withdrawal amount
        strategy = self.oracle.add_strategy(
            tax_lots=tax_lots,
            targets=targets,
            prices=prices,
            cash=cash,
            withdrawal_amount=500.00,
            optimization_type=OracleOptimizationType.TAX_AWARE
        )
        
        # Verify initial portfolio state
        self.assertEqual(strategy.total_value(), 1000.00)
        self.assertEqual(strategy.withdrawal_amount, 500.00)
        
        # Run optimization
        status, should_trade, trade_summary, trades = strategy.compute_optimal_trades(
            weight_tax=1.0,
            weight_drift=1.0,
            weight_transaction=0.1,
            debug=True  # Enable debug to see what's happening
        )
        
        # Verify optimization status
        self.assertIsNotNone(status)
        self.assertTrue(should_trade)
        
        # Check trades were generated
        self.assertFalse(trades.empty)

        # Get sell and buy trades
        sell_trades = trades[trades['action'] == 'sell']
        buy_trades = trades[trades['action'] == 'buy']
        total_sells = sell_trades.apply(lambda x: x['trade_value'], axis=1).sum()
        total_buys = buy_trades.apply(lambda x: x['trade_value'], axis=1).sum()
        
        new_cash = cash + total_sells - total_buys
        
        # Verify we have enough cash for withdrawal
        self.assertGreaterEqual(new_cash, 500.00)
        
        # Check that we actually have trades selling assets to fund the withdrawal
        self.assertGreater(total_sells, 1, "Should have 2 sells to fund the withdrawal")
        
        # Check that the withdrawal amount is respected in the final cash amount
        self.assertLessEqual(new_cash - 500.00, cash + 50.00, 
                            "Cash after withdrawal should be near original cash level")
        
        # Check that the cash deployment objective was not used due to withdrawal
        # This can be inferred indirectly by making sure we're explicitly selling 
        # enough to cover the withdrawal
        self.assertGreaterEqual(total_sells, 300.00)  # Need at least $300 more than initial $200
        
        # Print the trade summary for debugging
        print("\nTrade Summary for Withdrawal Test:")
        print(f"Initial cash: ${cash:.2f}")
        print(f"Withdrawal amount: ${strategy.withdrawal_amount:.2f}")
        print(f"Total sells: ${total_sells:.2f}")
        print(f"Total buys: ${total_buys:.2f}")
        print(f"Final cash: ${new_cash:.2f}")
        print(f"Cash after withdrawal: ${new_cash - 500.00:.2f}")
        
        # Verify the trade summary shows the right values
        if trade_summary:
            print("\nOptimization Summary:")
            for key, value in trade_summary.items():
                print(f"  {key}: {value}")
        
        # Test a withdrawal exceeding portfolio value (should raise ValueError)
        with self.assertRaises(ValueError):
            self.oracle.add_strategy(
                tax_lots=tax_lots,
                targets=targets,
                prices=prices,
                cash=cash,
                withdrawal_amount=1500.00,  # More than $1000 portfolio value
                optimization_type=OracleOptimizationType.TAX_AWARE
            )
            
    def test_withdrawal_incompatible_strategies(self):
        """Test that withdrawal correctly fails with incompatible strategy types."""
        # Create a simple portfolio with 2 stocks and some cash
        tax_lots = pd.DataFrame({
            'tax_lot_id': ['lot1', 'lot2'],
            'identifier': ['AAPL', 'MSFT'],
            'quantity': [2, 1],
            'date': [date(2022, 1, 1), date(2022, 1, 1)],
            'cost_basis': [200.00, 400.00]
        })
        
        prices = pd.DataFrame({
            'identifier': ['AAPL', 'MSFT', CASH_CUSIP_ID],
            'price': [200.00, 400.00, 1.00]
        })
        
        targets = pd.DataFrame([
            {'asset_class': 'Tech1', 'identifiers': ['AAPL'], 'target_weight': 0.4},
            {'asset_class': 'Tech2', 'identifiers': ['MSFT'], 'target_weight': 0.4},
            {'asset_class': CASH_CUSIP_ID, 'identifiers': [CASH_CUSIP_ID], 'target_weight': 0.2}
        ])
        
        cash = 200.00
        
        # Test with BUY_ONLY strategy
        strategy_buy_only = self.oracle.add_strategy(
            tax_lots=tax_lots,
            targets=targets,
            prices=prices,
            cash=cash,
            withdrawal_amount=500.00,
            optimization_type=OracleOptimizationType.BUY_ONLY
        )
        
        # This should fail during compute_optimal_trades
        with self.assertRaises(ValueError):
            strategy_buy_only.compute_optimal_trades(debug=True)
            
        # Test with HOLD strategy  
        strategy_hold = self.oracle.add_strategy(
            tax_lots=tax_lots,
            targets=targets,
            prices=prices,
            cash=cash,
            withdrawal_amount=500.00,
            optimization_type=OracleOptimizationType.HOLD
        )
        
        # HOLD strategy should just return early with no trades, not raise an error
        status, should_trade, trade_summary, trades = strategy_hold.compute_optimal_trades(debug=True)
        
        # Verify that no trades were returned
        self.assertFalse(should_trade, "HOLD strategy should return should_trade=False")
        self.assertEqual(len(trades), 0, "HOLD strategy should return empty trades DataFrame")