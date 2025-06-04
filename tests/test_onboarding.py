from datetime import date
import pandas as pd
import unittest
from src.service.initializers import (
    initialize_targets,
    initialize_tax_lots,
    initialize_prices,
    initialize_spreads,
    initialize_factor_model
)
from src.service.oracle import Oracle
from src.service.oracle_strategy import OracleStrategy
from src.service.helpers.enums import OracleOptimizationType
from src.service.helpers.constants import CASH_CUSIP_ID

class TestBasicPortfolioOptimization(unittest.TestCase):
    """Test basic portfolio optimization functionality."""

    def setUp(self):
        """Set up test data for basic portfolio optimization.
        
        Initial State:
        - STOCK_A: $10k position (50%)
            - Price: $100/share, 100 shares
            - Cost basis: $100/share
            
        - STOCK_B: $10k position (50%)
            - Price: $100/share, 100 shares
            - Cost basis: $100/share
            
        Total Portfolio: $20k + $10k cash
        Target Weights: 40-40-20 (STOCK_A, STOCK_B, CASH)
        """
        self.current_date = date(2024, 4, 20)
        
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

        # Define test data based on documentation example
        self.targets_df = pd.DataFrame({
            'asset_class': ['STOCK_A', 'STOCK_B', 'CASH'],
            'target_weight': [0.4, 0.4, 0.2],
            'identifiers': [['STOCK_A'], ['STOCK_B'], [CASH_CUSIP_ID]]
        })

        self.tax_lots_df = pd.DataFrame({
            'tax_lot_id': ['lot_a1', 'lot_b1'],
            'identifier': ['STOCK_A', 'STOCK_B'],
            'quantity': [100, 100],
            'cost_basis': [100, 100],
            'date': ['2024-01-01', '2024-01-01']
        })

        self.prices_df = pd.DataFrame([
            {
                'identifier': 'STOCK_A',
                'price': 100.0,
            },
            {
                'identifier': 'STOCK_B',
                'price': 100.0,
            },
            {
                'identifier': CASH_CUSIP_ID,
                'price': 1.0,
            }
        ])

        self.spreads_df = pd.DataFrame([
            {
                'identifier': 'STOCK_A',
                'spread': 0.001  # 10 basis points
            },
            {
                'identifier': 'STOCK_B',
                'spread': 0.001  # 10 basis points
            },
            {
                'identifier': CASH_CUSIP_ID,
                'spread': 0.0  # No spread for cash
            }
        ])

        # Create Oracle instance
        self.oracle = Oracle(
            current_date=self.current_date,
            recently_closed_lots=pd.DataFrame(),
            stock_restrictions=pd.DataFrame(),
            tax_rates=self.tax_rates
        )

        # Create and configure strategy
        self.strategy = OracleStrategy(
            strategy_id="STRATEGY_1",
            tax_lots=self.tax_lots_df,
            prices=self.prices_df,
            cash=10000.0,
            targets=self.targets_df,
            asset_class_targets=None,
            spreads=self.spreads_df,
            factor_model=None,
            optimization_type=OracleOptimizationType.TAX_AWARE,
            deminimus_cash_target=0.0001,
            withdrawal_amount=0.0,
            enforce_wash_sale_prevention=True
        )

        self.strategy.set_oracle(self.oracle)
        self.oracle.strategies = [self.strategy]
        self.oracle.initialize_wash_sale_restrictions(percentage_protection_from_inadvertent_wash_sales=0.003)

    def test_optimization_response_format(self):
        """Test that the optimization returns the expected response format."""
        # Run optimization with settings from documentation
        results, netted_trades = self.oracle.compute_optimal_trades_for_all_strategies(
            settings={
                "strategies": {
                    "STRATEGY_1": {
                        "weight_tax": 1.0,
                        "weight_drift": 1.0,
                        "weight_transaction": 1.0,
                        "weight_factor_model": 0.0,
                        "weight_cash_drag": 0.0,
                        "rebalance_threshold": 0.001,
                        "buy_threshold": 0.0005,
                        "holding_time_days": 0,
                        "should_tlh": True,
                        "tlh_min_loss_threshold": 0.015,
                        "range_min_weight_multiplier": 0.5,
                        "range_max_weight_multiplier": 2.0,
                        "min_notional": 0,
                        "rank_penalty_factor": 0.0,
                        "trade_rounding": 4
                    }
                }
            }
        )

        # Verify response format and basic expectations
        self.assertIsInstance(results, dict, "Results should be a dictionary")
        self.assertIsInstance(netted_trades, pd.DataFrame, "Netted trades should be a DataFrame")

        # Verify we have exactly 2 buy trades
        buy_trades = netted_trades[netted_trades["action"] == "buy"]
        self.assertEqual(len(buy_trades), 2, "Expected exactly 2 buy trades")
        

if __name__ == '__main__':
    unittest.main()
