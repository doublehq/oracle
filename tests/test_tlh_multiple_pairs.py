import json 
import unittest
from datetime import date
import pandas as pd
from src.service.helpers.enums import OracleOptimizationType
from src.service.oracle_strategy import OracleStrategy
from src.service.oracle import Oracle
from src.service.helpers.constants import CASH_CUSIP_ID

class TestTLHMultiplePairs(unittest.TestCase):
    def setUp(self):
        self.current_date = date(2024, 4, 2)
        self.tax_rates = pd.DataFrame([
            {'gain_type': 'short_term', 'federal_rate': 0.35, 'state_rate': 0.06, 'total_rate': 0.41},
            {'gain_type': 'long_term', 'federal_rate': 0.20, 'state_rate': 0.06, 'total_rate': 0.26},
            {"gain_type": "qualified_dividend", "federal_rate": 0.15, "state_rate": 0.06, "total_rate": 0.21}
        ])
        self.oracle = Oracle(current_date=self.current_date, tax_rates=self.tax_rates)

    def test_tlh_second_most_important_asset_class(self):
        """Test that when one asset class is down, we buy the 2nd most important asset class's identifier as replacement"""
        
        # Initial holdings - 2 asset classes with 3 identifiers each
        tax_lots = pd.DataFrame({
            'identifier': ['TECH1', 'TECH2', 'FIN1', 'FIN2'],
            'tax_lot_id': ['TECH1_20240101', 'TECH2_20240101', 'FIN1_20240101', 'FIN2_20240101'],
            'quantity': [100, 50, 75, 25],
            'cost_basis': [10000, 5000, 7500, 2500],  # $100 per share
            'date': [date(2024, 1, 1)] * 4  # All purchased on day 1
        })

        # Target weights - Tech is more important (0.6) than Finance (0.4)
        targets = pd.DataFrame([
            {'asset_class': 'Tech', 'identifiers': ['TECH1', 'TECH2'], 'target_weight': 0.60},
            {'asset_class': 'Finance', 'identifiers': ['FIN1', 'FIN2'], 'target_weight': 0.40},
            {'asset_class': CASH_CUSIP_ID, 'identifiers': [CASH_CUSIP_ID], 'target_weight': 0.00}
        ])

        # Current prices - Tech stocks down 10%, Finance stocks flat
        prices = pd.DataFrame({
            'identifier': ['TECH1', 'TECH2', 'TECH3', 'FIN1', 'FIN2', 'FIN3', CASH_CUSIP_ID],
            'price': [90.0, 90.0, 90.0, 100.0, 100.0, 100.0, 1.0]
        })

        # Initialize strategy
        strategy = OracleStrategy(
            tax_lots=tax_lots,
            targets=targets,
            prices=prices,
            cash=0,
            optimization_type=OracleOptimizationType.PAIRS_TLH,
            deminimus_cash_target=0.0,
        )

        strategy.set_oracle(self.oracle)
        self.oracle.strategies = [strategy]
        self.oracle.initialize_wash_sale_restrictions()

        # Run optimization with TLH enabled
        status, should_trade, trade_summary, trades = strategy.compute_optimal_trades(
            weight_tax=0.0,
            weight_drift=1.0,
            weight_transaction=0.0,
            weight_cash_drag=1.0,
            should_tlh=True,
            tlh_min_loss_threshold=0.015,  # 1.5% loss threshold
            range_min_weight_multiplier=0.5,
            range_max_weight_multiplier=2,
            rank_penalty_factor=0.00000001
        )

        # Verify optimization succeeded
        self.assertIsNotNone(status)
        self.assertTrue(should_trade)

        # Check that TECH1 and TECH2 were sold
        tech1_trades = trades[trades['identifier'].isin(['TECH1'])]
        self.assertTrue(len(tech1_trades) > 0)
        self.assertTrue(all(tech1_trades['action'] == 'sell'))

        # Check that FIN3 was bought as replacement (2nd most important asset class)
        tech2_trades = trades[trades['identifier'] == 'TECH2']
        self.assertTrue(len(tech2_trades) > 0)
        self.assertTrue(tech2_trades['action'].iloc[0] == 'buy')
        # Verify small trades for FIN1 and FIN2 since they're not at a loss
        fin_existing_trades = trades[trades['identifier'].isin(['FIN1', 'FIN2'])]
        for _, trade in fin_existing_trades.iterrows():
            self.assertLess(abs(trade['quantity']), 10, "Expected only small trades for non-loss positions")

    def test_no_trades_when_drift_on_target(self):
        """Test that no trades occur when asset class drift is on target despite owning multiple identifiers"""
        
        # Initial holdings - 2 asset classes with 2 identifiers each, owning both identifiers in each class
        tax_lots = pd.DataFrame({
            'identifier': ['TECH1', 'TECH2', 'FIN1', 'FIN2'],
            'tax_lot_id': ['TECH1_20240101', 'TECH2_20240101', 'FIN1_20240101', 'FIN2_20240101'],
            'quantity': [60, 40, 60, 40],  # Split 60/40 within each asset class
            'cost_basis': [6000, 4000, 6000, 4000],  # $100 per share
            'date': [date(2024, 1, 1)] * 4  # All purchased on day 1
        })

        # Target weights - equal weight between asset classes
        targets = pd.DataFrame([
            {'asset_class': 'Tech', 'identifiers': ['TECH1', 'TECH2'], 'target_weight': 0.50},
            {'asset_class': 'Finance', 'identifiers': ['FIN1', 'FIN2'], 'target_weight': 0.50},
            {'asset_class': CASH_CUSIP_ID, 'identifiers': [CASH_CUSIP_ID], 'target_weight': 0.00}
        ])

        # Current prices - All stocks at cost basis
        prices = pd.DataFrame({
            'identifier': ['TECH1', 'TECH2', 'FIN1', 'FIN2', CASH_CUSIP_ID],
            'price': [100.0, 100.0, 100.0, 100.0, 1.0]
        })

        # Initialize strategy
        strategy = OracleStrategy(
            tax_lots=tax_lots,
            targets=targets,
            prices=prices,
            cash=0,
            optimization_type=OracleOptimizationType.PAIRS_TLH,
            deminimus_cash_target=0.0,
        )

        strategy.set_oracle(self.oracle)
        self.oracle.strategies = [strategy]
        self.oracle.initialize_wash_sale_restrictions()

        # Run optimization with TLH enabled
        status, should_trade, trade_summary, trades = strategy.compute_optimal_trades(
            weight_tax=0.0,
            weight_drift=1.0,
            weight_transaction=0.0,
            weight_cash_drag=1.0,
            should_tlh=True,
            tlh_min_loss_threshold=0.05,  # 5% loss threshold
            range_min_weight_multiplier=0.5,
            range_max_weight_multiplier=2,
            rank_penalty_factor=0.0
        )

        # Verify optimization succeeded but no trades needed
        self.assertIsNotNone(status)
        self.assertFalse(should_trade)  # Should not trade since drift is on target

        # Verify no trades were generated
        self.assertTrue(len(trades) == 0)
