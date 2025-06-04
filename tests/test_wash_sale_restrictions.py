import unittest
from datetime import date, timedelta
import pandas as pd
from pathlib import Path
from src.service.constraints.restriction.wash_sale_restrictions import WashSaleRestrictions
from src.service.helpers.constants import CASH_CUSIP_ID
from src.service.helpers.enums import OracleOptimizationType
from src.service.oracle import Oracle
from src.service.oracle_strategy import OracleStrategy

class TestWashSaleRestrictions(unittest.TestCase):
    def test_buy_sell_buy_case(self):
        """Test that we can't buy a stock within 30 days after selling it at a loss"""
        current_date = date(2024, 1, 3)  # day 3
        
        # Initial holdings on day 1
        tax_lots = pd.DataFrame({
            'identifier': ['STOCK1', 'STOCK2'],
            'tax_lot_id': ['STOCK1_20240101', 'STOCK2_20240101'],
            'quantity': [100, 100],
            'cost_basis': [10000, 10000],  # $100 per share
            'date': [date(2024, 1, 1), date(2024, 1, 1)]  # day 1
        })
        
        # Current prices - STOCK1 at $90, STOCK2 at $100
        prices = pd.DataFrame({
            'identifier': ['STOCK1', 'STOCK2'],
            'price': [90.0, 100.0]
        })
        
        # Sold STOCK1 at a loss on day 2
        recently_closed_lots = pd.DataFrame({
            'identifier': ['STOCK1'],
            'tax_lot_id': ['STOCK1_20240101'],
            'quantity': [100],
            'cost_basis': [10000],
            'date_acquired': [date(2024, 1, 1)],
            'date_sold': [date(2024, 1, 2)],
            'proceeds': [9000],  # Sold at $90 per share = loss
            'realized_gain': [-1000]
        })
        
        # Initialize wash sale restrictions
        wash_sale_restrictions = WashSaleRestrictions(
            current_date=current_date,
            prices=prices,
            recently_closed_lots=recently_closed_lots,
            all_tax_lots=tax_lots
        )
        
        # STOCK1 should be restricted from buying
        self.assertTrue(wash_sale_restrictions.is_restricted_from_buying('STOCK1'))
        # STOCK2 should not be restricted
        self.assertFalse(wash_sale_restrictions.is_restricted_from_buying('STOCK2'))
    
    def test_buy_buy_sell_case(self):
        """Test that we can't sell a stock until 30 days after the second purchase"""
        current_date = date(2024, 1, 3)  # day 3
        
        # Initial holdings - bought STOCK1 twice
        tax_lots = pd.DataFrame({
            'identifier': ['STOCK1', 'STOCK1', 'STOCK2'],
            'tax_lot_id': ['STOCK1_20240101', 'STOCK1_20240102', 'STOCK2_20240101'],
            'quantity': [100, 50, 100],
            'cost_basis': [10000, 5000, 10000],
            'date': [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 1)]  # bought on day 1 and day 2
        })
        
        # Current prices - STOCK1 at $90 (loss), STOCK2 at $100 (no loss)
        prices = pd.DataFrame({
            'identifier': ['STOCK1', 'STOCK2'],
            'price': [90.0, 100.0]
        })
        
        # Initialize wash sale restrictions
        wash_sale_restrictions = WashSaleRestrictions(
            current_date=current_date,
            prices=prices,
            all_tax_lots=tax_lots
        )
        
        # Both STOCK1 lots should be restricted from selling
        self.assertTrue(wash_sale_restrictions.is_lot_restricted_from_selling(
            'STOCK1', 
            f"STOCK1_{date(2024, 1, 1).strftime('%Y%m%d')}"
        ))
        self.assertTrue(wash_sale_restrictions.is_lot_restricted_from_selling(
            'STOCK1', 
            f"STOCK1_{date(2024, 1, 2).strftime('%Y%m%d')}"
        ))
        # STOCK2 should not be restricted
        self.assertFalse(wash_sale_restrictions.is_lot_restricted_from_selling(
            'STOCK2',
            f"STOCK2_{date(2024, 1, 1).strftime('%Y%m%d')}"
        ))
    
    def test_timeout_on_buy_sell_buy_case(self):
        """Test that buy restriction expires after 30 days"""
        # Start on day 3
        current_date = date(2024, 1, 3)
        
        tax_lots = pd.DataFrame({
            'identifier': ['STOCK1', 'STOCK2'],
            'tax_lot_id': ['STOCK1_20240101', 'STOCK2_20240101'],
            'quantity': [100, 100],
            'cost_basis': [10000, 10000],
            'date': [date(2024, 1, 1), date(2024, 1, 1)]
        })
        
        # Current prices - STOCK1 at $90 (loss), STOCK2 at $100 (no loss)
        prices = pd.DataFrame({
            'identifier': ['STOCK1', 'STOCK2'],
            'price': [90.0, 100.0]
        })
        
        recently_closed_lots = pd.DataFrame({
            'identifier': ['STOCK1'],
            'tax_lot_id': ['STOCK1_20240101'],
            'quantity': [100],
            'cost_basis': [10000],
            'date_acquired': [date(2024, 1, 1)],
            'date_sold': [date(2024, 1, 2)],
            'proceeds': [9000],
            'realized_gain': [-1000]
        })
        
        # Check restriction on day 3
        wash_sale_restrictions = WashSaleRestrictions(
            current_date=current_date,
            prices=prices,
            recently_closed_lots=recently_closed_lots,
            all_tax_lots=tax_lots
        )
        self.assertTrue(wash_sale_restrictions.is_restricted_from_buying('STOCK1'))

        # Check restriction on day 29
        wash_sale_restrictions = WashSaleRestrictions(
            current_date=date(2024, 1, 31),
            prices=prices,
            recently_closed_lots=recently_closed_lots,
            all_tax_lots=tax_lots
        )
        self.assertTrue(wash_sale_restrictions.is_restricted_from_buying('STOCK1'))
        
        # Check restriction expires on day 32
        wash_sale_restrictions = WashSaleRestrictions(
            current_date=date(2024, 2, 1),  # 30 days after sale
            prices=prices,
            recently_closed_lots=recently_closed_lots,
            all_tax_lots=tax_lots
        )
        self.assertFalse(wash_sale_restrictions.is_restricted_from_buying('STOCK1'))

    def test_timeout_on_buy_buy_sell_case(self):
        """Test that sell restriction expires 30 days after second purchase"""
        # Start on day 3
        current_date = date(2024, 1, 3)
        
        tax_lots = pd.DataFrame({
            'identifier': ['STOCK1', 'STOCK1', 'STOCK2'],
            'tax_lot_id': ['STOCK1_20240101', 'STOCK1_20240102', 'STOCK2_20240101'],
            'quantity': [100, 50, 100],
            'cost_basis': [10000, 5000, 10000],
            'date': [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 1)]
        })
        
        # Current prices - STOCK1 at $90 (loss), STOCK2 at $100 (no loss)
        prices = pd.DataFrame({
            'identifier': ['STOCK1', 'STOCK2'],
            'price': [90.0, 100.0]
        })
        
        # Check restriction on day 3
        wash_sale_restrictions = WashSaleRestrictions(
            current_date=current_date,
            prices=prices,
            all_tax_lots=tax_lots
        )
        self.assertTrue(wash_sale_restrictions.is_lot_restricted_from_selling(
            'STOCK1',
            f"STOCK1_{date(2024, 1, 1).strftime('%Y%m%d')}"
        ))
        
        # Check restriction expires >30 days after second purchase
        wash_sale_restrictions = WashSaleRestrictions(
            current_date=date(2024, 2, 2),  # >30 days after second purchase
            prices=prices,
            all_tax_lots=tax_lots
        )
        self.assertFalse(wash_sale_restrictions.is_lot_restricted_from_selling(
            'STOCK1',
            f"STOCK1_{date(2024, 1, 1).strftime('%Y%m%d')}"
        ))
    
    def test_gain_can_always_be_sold(self):
        """Test that stocks sold for a gain don't trigger wash sale restrictions"""
        current_date = date(2024, 1, 3)  # day 3
        
        tax_lots = pd.DataFrame({
            'identifier': ['STOCK1', 'STOCK2'],
            'tax_lot_id': ['STOCK1_20240101', 'STOCK2_20240101'],
            'quantity': [100, 100],
            'cost_basis': [10000, 10000],
            'date': [date(2024, 1, 1), date(2024, 1, 1)]
        })
        
        # Current prices - STOCK1 at $110 (gain), STOCK2 at $100 (no change)
        prices = pd.DataFrame({
            'identifier': ['STOCK1', 'STOCK2'],
            'price': [110.0, 100.0]
        })
        
        # Sold STOCK1 at a gain on day 2
        recently_closed_lots = pd.DataFrame({
            'identifier': ['STOCK1'],
            'tax_lot_id': ['STOCK1_20240101'],
            'quantity': [100],
            'cost_basis': [10000],
            'date_acquired': [date(2024, 1, 1)],
            'date_sold': [date(2024, 1, 2)],
            'proceeds': [11000],  # Sold at $110 per share = gain
            'realized_gain': [1000]
        })
        
        wash_sale_restrictions = WashSaleRestrictions(
            current_date=current_date,
            prices=prices,
            recently_closed_lots=recently_closed_lots,
            all_tax_lots=tax_lots
        )
        
        # Should be allowed to buy STOCK1 since it was sold for a gain
        self.assertFalse(wash_sale_restrictions.is_restricted_from_buying('STOCK1'))

    def test_wash_sale_liquidation(self):
        """Test that we can't sell a stock until 30 days after the second purchase"""
        current_date = date(2024, 1, 3)  # day 3
        
        # Initial holdings - bought STOCK1 twice
        tax_lots = pd.DataFrame({
            'identifier': ['STOCK1', 'STOCK1', 'STOCK2'],
            'tax_lot_id': ['STOCK1_20240101', 'STOCK1_20240102', 'STOCK2_20240101'],
            'quantity': [100, 50, 100],
            'cost_basis': [10000, 4000, 10000],
            'date': [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 1)]  # bought on day 1 and day 2
        })

        # Current prices - STOCK1 at $90 (one loss, one gain), STOCK2 at $100 (no loss)
        prices = pd.DataFrame({
            'identifier': ['STOCK1', 'STOCK2'],
            'price': [90.0, 100.0]
        })

        targets = pd.DataFrame([
            {
                'asset_class': 'STOCK1',
                'identifiers': ['STOCK1'],
                'target_weight': 0,
            },
            {
                'asset_class': CASH_CUSIP_ID,
                'identifiers': [CASH_CUSIP_ID],
                'target_weight': 1,
            },
        ])

        strategy1 = OracleStrategy(
            tax_lots=tax_lots,
            targets=targets,
            prices=prices,
            cash=0,
            optimization_type=OracleOptimizationType.TAX_AWARE,
        )

        oracle1 = Oracle(
            current_date=current_date,
            tax_rates=None,
        )

        strategy1.set_oracle(oracle1)
        oracle1.strategies = [strategy1]
        oracle1.initialize_wash_sale_restrictions()

        self.assertTrue(oracle1.wash_sale_restrictions.is_lot_restricted_from_selling('STOCK1', 'STOCK1_20240101'))
        self.assertFalse(oracle1.wash_sale_restrictions.is_lot_restricted_from_selling('STOCK1', 'STOCK1_20240102'))

        _, _, _, trades = strategy1.compute_optimal_trades(
            weight_tax=0.0,
            weight_drift=1.0,
            weight_transaction=0.0,
            debug=False
        )

        self.assertCountEqual(
            trades[["identifier", "tax_lot_id", "action", "quantity"]].to_dict(orient="records"),
            [
                { "identifier": "STOCK1", "tax_lot_id": "STOCK1_20240101", "action": "sell", "quantity": 100 },
                { "identifier": "STOCK1", "tax_lot_id": "STOCK1_20240102", "action": "sell", "quantity": 50 },
            ]
        )

        strategy2 = OracleStrategy(
            tax_lots=tax_lots,
            targets=targets,
            prices=prices,
            cash=0,
            optimization_type=OracleOptimizationType.TAX_AWARE,
        )

        oracle2 = Oracle(
            current_date=current_date,
            tax_rates=None,
        )

        strategy2.set_oracle(oracle2)
        oracle2.strategies = [strategy1, strategy2]
        oracle2.initialize_wash_sale_restrictions()

        _, _, _, trades = strategy2.compute_optimal_trades(
            weight_tax=0.0,
            weight_drift=1.0,
            weight_transaction=0.0,
            debug=False
        )

        self.assertCountEqual(
            trades[["identifier", "tax_lot_id", "action", "quantity"]].to_dict(orient="records"),
            [
                { "identifier": "STOCK1", "tax_lot_id": "STOCK1_20240102", "action": "sell", "quantity": 50 },
            ]
        )
