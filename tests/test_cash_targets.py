import json 
import unittest
from datetime import date
import pandas as pd
from src.service.helpers.enums import OracleOptimizationType
from src.service.oracle_strategy import OracleStrategy
from src.service.oracle import Oracle
from src.service.helpers.constants import CASH_CUSIP_ID

class TestCashTargets(unittest.TestCase):
    def setUp(self):
        self.current_date = date(2024, 4, 2)
        self.tax_rates = pd.DataFrame([
            {'gain_type': 'short_term', 'federal_rate': 0.35, 'state_rate': 0.06, 'total_rate': 0.41},
            {'gain_type': 'long_term', 'federal_rate': 0.20, 'state_rate': 0.06, 'total_rate': 0.26},
            {"gain_type": "qualified_dividend", "federal_rate": 0.15, "state_rate": 0.06, "total_rate": 0.21}
        ])
        self.oracle = Oracle(current_date=self.current_date, tax_rates=self.tax_rates)

    def test_all_cash_to_target_allocation(self):
        """Test that starting from all cash, each optimization type can reach target allocation of 45-45-10"""
        
        # Initial holdings - all cash ($100,000)
        tax_lots = pd.DataFrame({
            'identifier': [],
            'tax_lot_id': [],
            'quantity': [],
            'cost_basis': [],
            'date': []
        })

        # Target weights - 45% Tech, 45% Finance, 10% Cash
        targets = pd.DataFrame([
            {'asset_class': 'Tech', 'identifiers': ['TECH1', 'TECH2'], 'target_weight': 0.45},
            {'asset_class': 'Finance', 'identifiers': ['FIN1', 'FIN2'], 'target_weight': 0.45},
            {'asset_class': CASH_CUSIP_ID, 'identifiers': [CASH_CUSIP_ID], 'target_weight': 0.10}
        ])

        # Current prices for all securities
        prices = pd.DataFrame({
            'identifier': ['TECH1', 'TECH2', 'FIN1', 'FIN2', CASH_CUSIP_ID],
            'price': [100.0, 100.0, 100.0, 100.0, 1.0]
        })

        # Test each optimization type
        optimization_types = [
            OracleOptimizationType.TAX_AWARE,
            OracleOptimizationType.TAX_UNAWARE,
            OracleOptimizationType.BUY_ONLY,
            OracleOptimizationType.PAIRS_TLH
        ]  # Removed DIRECT_INDEX as it requires a factor model

        for opt_type in optimization_types:
            with self.subTest(optimization_type=opt_type.value):
                # Initialize strategy with $100,000 cash
                strategy = OracleStrategy(
                    tax_lots=tax_lots,
                    targets=targets,
                    prices=prices,
                    cash=100000.0,
                    optimization_type=opt_type,
                    deminimus_cash_target=0.0,
                )

                strategy.set_oracle(self.oracle)
                self.oracle.strategies = [strategy]
                self.oracle.initialize_wash_sale_restrictions()

                # Run optimization
                status, should_trade, trade_summary, trades = strategy.compute_optimal_trades(
                    weight_tax=1.0,
                    weight_drift=1.0,
                    weight_transaction=1.0,
                    weight_cash_drag=1.0,
                    should_tlh=False,
                    range_min_weight_multiplier=0.5,
                    range_max_weight_multiplier=2.0,
                    rank_penalty_factor=0.0
                )

                # Verify optimization succeeded
                self.assertIsNotNone(status)
                self.assertTrue(should_trade)

                # Calculate post-trade asset class weights
                post_trade_holdings = strategy.post_trade_strategy.actuals
                total_value = post_trade_holdings['market_value'].sum()

                # Group by asset class and calculate weights
                asset_class_weights = {}
                for _, target in targets.iterrows():
                    asset_class = target['asset_class']
                    identifiers = target['identifiers']
                    
                    if asset_class == CASH_CUSIP_ID:
                        # For cash, use the cash value from post-trade strategy
                        asset_class_weights[asset_class] = strategy.post_trade_strategy.cash / total_value
                    else:
                        # For other asset classes, sum up the market values of their identifiers
                        class_holdings = post_trade_holdings[post_trade_holdings['identifier'].isin(identifiers)]
                        asset_class_weights[asset_class] = class_holdings['market_value'].sum() / total_value

                # Verify each asset class is within 5% of target
                for _, target in targets.iterrows():
                    asset_class = target['asset_class']
                    target_weight = target['target_weight']
                    actual_weight = asset_class_weights[asset_class]
                    
                    self.assertAlmostEqual(
                        actual_weight,
                        target_weight,
                        delta=0.05,  # Allow 5% deviation
                        msg=f"For {opt_type.value}, {asset_class} weight {actual_weight:.2%} not within 5% of target {target_weight:.2%}"
                    )

                # Verify total portfolio value remains close to initial value
                self.assertAlmostEqual(
                    total_value,
                    100000.0,
                    delta=1.0,  # Allow $1 deviation for rounding
                    msg=f"For {opt_type.value}, total value {total_value:.2f} not close to initial 100000.0"
                )

    def test_direct_index_cash_to_target_allocation(self):
        """Test that DIRECT_INDEX optimization type can reach target allocation with factor model"""
        
        # Initial holdings - all cash ($100,000)
        tax_lots = pd.DataFrame({
            'identifier': [],
            'tax_lot_id': [],
            'quantity': [],
            'cost_basis': [],
            'date': []
        })

        # Target weights - 45% Tech, 45% Finance, 10% Cash
        targets = pd.DataFrame([
            {'asset_class': 'Tech', 'identifiers': ['TECH1', 'TECH2'], 'target_weight': 0.45},
            {'asset_class': 'Finance', 'identifiers': ['FIN1', 'FIN2'], 'target_weight': 0.45},
            {'asset_class': CASH_CUSIP_ID, 'identifiers': [CASH_CUSIP_ID], 'target_weight': 0.10}
        ])

        # Current prices for all securities
        prices = pd.DataFrame({
            'identifier': ['TECH1', 'TECH2', 'FIN1', 'FIN2', CASH_CUSIP_ID],
            'price': [100.0, 100.0, 100.0, 100.0, 1.0]
        })

        factor_model = pd.DataFrame([
            {'identifier': 'TECH1', 'growth': 1.0, 'value': 0.3, 'min_volatility': 0.0},
            {'identifier': 'TECH2', 'growth': 0.01, 'value': 1.0, 'min_volatility': 0.0},
            {'identifier': 'FIN1', 'growth': 0.6, 'value': 0.4, 'min_volatility': 0.4},
            {'identifier': 'FIN2', 'growth': 0.6, 'value': 0.4, 'min_volatility': 0.4},
            {'identifier': CASH_CUSIP_ID, 'growth': 0, 'value': 0, 'min_volatility': 0}
        ])

        # Initialize strategy with $100,000 cash
        strategy = OracleStrategy(
            tax_lots=tax_lots,
            targets=targets,
            prices=prices,
            cash=100000.0,
            optimization_type=OracleOptimizationType.DIRECT_INDEX,
            deminimus_cash_target=0.0,
            factor_model=factor_model
        )

        strategy.set_oracle(self.oracle)
        self.oracle.strategies = [strategy]
        self.oracle.initialize_wash_sale_restrictions()

        # Run optimization
        status, should_trade, trade_summary, trades = strategy.compute_optimal_trades(
            weight_tax=1.0,
            weight_drift=1.0,
            weight_transaction=0.0,
            weight_factor_model=1.0,
            weight_cash_drag=1.0,
            should_tlh=False,
            range_min_weight_multiplier=0.5,
            range_max_weight_multiplier=2.0,
            rank_penalty_factor=0.0
        )

        # Verify optimization succeeded
        self.assertIsNotNone(status)
        self.assertTrue(should_trade)

        # Calculate post-trade asset class weights
        post_trade_holdings = strategy.post_trade_strategy.actuals
        total_value = post_trade_holdings['market_value'].sum()

        # Group by asset class and calculate weights
        asset_class_weights = {}
        for _, target in targets.iterrows():
            asset_class = target['asset_class']
            identifiers = target['identifiers']
            
            if asset_class == CASH_CUSIP_ID:
                # For cash, use the cash value from post-trade strategy
                asset_class_weights[asset_class] = strategy.post_trade_strategy.cash / total_value
            else:
                # For other asset classes, sum up the market values of their identifiers
                class_holdings = post_trade_holdings[post_trade_holdings['identifier'].isin(identifiers)]
                asset_class_weights[asset_class] = class_holdings['market_value'].sum() / total_value

        # Verify each asset class is within 5% of target
        for _, target in targets.iterrows():
            asset_class = target['asset_class']
            target_weight = target['target_weight']
            actual_weight = asset_class_weights[asset_class]
            
            self.assertAlmostEqual(
                actual_weight,
                target_weight,
                delta=0.05,  # Allow 5% deviation
                msg=f"For DIRECT_INDEX, {asset_class} weight {actual_weight:.2%} not within 5% of target {target_weight:.2%}"
            )

        # Verify total portfolio value remains close to initial value
        self.assertAlmostEqual(
            total_value,
            100000.0,
            delta=1.0,  # Allow $1 deviation for rounding
            msg=f"For DIRECT_INDEX, total value {total_value:.2f} not close to initial 100000.0"
        )
