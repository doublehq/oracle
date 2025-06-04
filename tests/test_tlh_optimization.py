import json 
import unittest
from datetime import date, timedelta
import pandas as pd
import numpy as np
from src.service.constraints.restriction.wash_sale_restrictions import WashSaleRestrictions
from src.service.helpers.enums import OracleOptimizationType
from src.service.oracle_strategy import OracleStrategy
from src.service.oracle import Oracle
from src.service.helpers.constants import CASH_CUSIP_ID

class TestTLHOptimization(unittest.TestCase):
    def setUp(self):
        self.current_date = date(2024, 4, 2)
        self.tax_rates = pd.DataFrame([
            {'gain_type': 'short_term', 'federal_rate': 0.35, 'state_rate': 0.06, 'total_rate': 0.41},
            {'gain_type': 'long_term', 'federal_rate': 0.20, 'state_rate': 0.06, 'total_rate': 0.26},
            {"gain_type": "qualified_dividend", "federal_rate": 0.15, "state_rate": 0.06, "total_rate": 0.21}
        ])
        self.oracle = Oracle(current_date=self.current_date, tax_rates=self.tax_rates)
        
    def test_pairs_tlh_basic(self):
        """Test that we can harvest tax losses in pairs mode with a simple setup"""
        current_date = date(2024, 1, 3)  # day 3
        
        # Initial holdings - STOCK1 down 2%, STOCK2 flat
        tax_lots = pd.DataFrame({
            'identifier': ['STOCK1', 'STOCK2'],
            'tax_lot_id': ['STOCK1_20240101', 'STOCK2_20240101'],
            'quantity': [100, 100],
            'cost_basis': [10000, 10000],  # $100 per share
            'date': [date(2024, 1, 1), date(2024, 1, 1)]  # day 1
        })
        # Target weights - equal weight plus cash
        targets = pd.DataFrame([
            {'asset_class': 'Tech', 'identifiers': ['STOCK1', 'STOCK1_ALT'], 'target_weight': 0.45},
            {'asset_class': 'Finance', 'identifiers': ['STOCK2'], 'target_weight': 0.55},
            {'asset_class': CASH_CUSIP_ID, 'identifiers': [CASH_CUSIP_ID], 'target_weight': 0.00}
        ])
        
        # Current prices - STOCK1 at $98 (2% loss), STOCK2 at $100 (flat), STOCK1_ALT at $100
        prices = pd.DataFrame({
            'identifier': ['STOCK1', 'STOCK2', 'STOCK1_ALT', CASH_CUSIP_ID],
            'price': [90.0, 110.0, 90.0, 1.0]
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
            rank_penalty_factor=0.5
        )
        
        # Verify optimization succeeded
        self.assertIsNotNone(status)
        self.assertTrue(should_trade)

        print("Trade Summary:", json.dumps(trade_summary, indent=4))
        
        # Check that STOCK1 was sold and STOCK1_ALT was bought
        stock1_trades = trades[trades['identifier'] == 'STOCK1']
        stock1_alt_trades = trades[trades['identifier'] == 'STOCK1_ALT']
        
        self.assertTrue(len(stock1_trades) > 0)
        self.assertTrue(len(stock1_alt_trades) > 0)
        self.assertTrue(stock1_trades['action'].iloc[0] == 'sell')  # Selling STOCK1
        self.assertTrue(stock1_alt_trades['action'].iloc[0] == 'buy')  # Buying STOCK1_ALT
        
        # Verify the trade amounts match (selling all STOCK1 and buying equivalent STOCK1_ALT)
        self.assertAlmostEqual(
            abs(stock1_trades['quantity'].iloc[0]), 
            stock1_alt_trades['quantity'].iloc[0],
            places=2
        )

        # Verify STOCK2 wasn't traded since it's not at a loss
        stock2_trades = trades[trades['identifier'] == 'STOCK2']
        self.assertTrue(len(stock2_trades) == 0)
        
    def test_direct_index_tlh(self):
        """Test that we can harvest tax losses in direct index mode with multiple stocks"""
        current_date = date(2024, 1, 3)  # day 3
        
        # Generate 100 stocks with random prices and weights
        np.random.seed(42)  # For reproducibility
        num_stocks = 100
        
        # Generate identifiers
        identifiers = [f'STOCK{i+1}' for i in range(num_stocks)]
        
        tax_lots_data = []
        for stock in identifiers:
            tax_lots_data.append({
                'identifier': stock,
                'tax_lot_id': f'{stock}_20240101',
                'quantity': 100,
                'cost_basis': 10000,  # $100 per share
                'date': date(2024, 1, 1)
            })
        tax_lots = pd.DataFrame(tax_lots_data)
        
        # Generate target weights - all stocks have some target
        target_weights = 1/(num_stocks)
        targets_data = []
        for i, stock in enumerate(identifiers):
            targets_data.append({
                'identifiers': [stock],
                'target_weight': target_weights,
                'asset_class': stock  # 10 different sectors
            })
        # Add cash target
        targets_data.append({
            'identifiers': [CASH_CUSIP_ID],
            'target_weight': 0,
            'asset_class': CASH_CUSIP_ID
        })
        targets = pd.DataFrame(targets_data)
        
        # Generate prices - some stocks down >1.5%
        prices_data = []
        for i, stock in enumerate(identifiers):
            if i < 5:
                price = 80.0  # 20% down
            elif i < 10:
                price = 120.0  # 20% up
            else:
                price = 100.0
            prices_data.append({
                'identifier': stock,
                'price': price
            })
        # Add cash price
        prices_data.append({
            'identifier': CASH_CUSIP_ID,
            'price': 1.0
        })
        prices = pd.DataFrame(prices_data)
        
        # Generate factor model data
        num_factors = 5
        factor_data = []
        for stock in identifiers:
            exposures = np.random.normal(0, 1, num_factors)
            factor_data.append({
                'identifier': stock,
                'factor1': exposures[0],
                'factor2': exposures[1],
                'factor3': exposures[2],
                'factor4': exposures[3],
                'factor5': exposures[4]
            })
        # Add cash with zero factor exposures
        factor_data.append({
            'identifier': CASH_CUSIP_ID,
            'factor1': 0,
            'factor2': 0,
            'factor3': 0,
            'factor4': 0,
            'factor5': 0
        })
        factor_model = pd.DataFrame(factor_data)
        
        # Initialize strategy
        strategy = OracleStrategy(
            tax_lots=tax_lots,
            targets=targets,
            prices=prices,
            cash=0,
            factor_model=factor_model,
            optimization_type=OracleOptimizationType.DIRECT_INDEX
        )

        strategy.set_oracle(self.oracle)
        self.oracle.strategies = [strategy]
        self.oracle.initialize_wash_sale_restrictions()
        
        # Run optimization with TLH enabled
        status, should_trade, trade_summary, trades = strategy.compute_optimal_trades(
            weight_tax=1.0,
            weight_drift=1.0,
            weight_transaction=1.0,
            weight_factor_model=100.0,
            weight_cash_drag=1.0,
            should_tlh=True,
            tlh_min_loss_threshold=0.015,  
            range_min_weight_multiplier=0.5,
            range_max_weight_multiplier=2.0
        )
        
        # Verify optimization succeeded
        self.assertIsNotNone(status)
        self.assertTrue(should_trade)
        
        # Check that we have both sells and buys
        sells = trades[trades['action'] == 'sell']
        buys = trades[trades['action'] == 'buy']
        
        self.assertTrue(len(sells) > 0)
        self.assertTrue(len(buys) > 0)
        # Verify that TLH sells are for stocks that were down >1.5%
        tlh_sells = sells[sells['gain_loss'].apply(lambda x: x['is_tlh_trade'])]
        for _, sell in tlh_sells.iterrows():
            stock_price = prices[prices['identifier'] == sell['identifier']]['price'].iloc[0]
            original_cost = tax_lots[tax_lots['identifier'] == sell['identifier']]['cost_basis'].iloc[0] / \
                          tax_lots[tax_lots['identifier'] == sell['identifier']]['quantity'].iloc[0]
            loss_pct = (original_cost - stock_price) / original_cost
            self.assertGreater(loss_pct, 0.015)  # Verify loss is greater than 1.5%
        
        # Verify that the buys maintain factor exposure
        if len(strategy.factor_model_target) > 0:
            # Calculate pre and post trade factor exposures
            pre_trade_exposures = strategy.factor_model_actual
            post_trade_exposures = strategy.post_trade_strategy.factor_model_actual
            targets = strategy.factor_model_target

            #Check that overall we've improved the factor model.
            change = 0
            for factor in pre_trade_exposures.columns:
                pre_exposure = pre_trade_exposures[factor].iloc[0]
                post_exposure = post_trade_exposures[factor].iloc[0]
                target = targets[factor].iloc[0]
                # Check that across all trades we are closer to the target
                change += abs(post_exposure - target) - abs(pre_exposure - target)
            self.assertLess(change, 0, "Factor model should improve after trades")

        # Check trades["gain_loss"] for our sells have is_tlh_trade == True
        tlh_sells = trades[(trades['action'] == 'sell') & (trades["gain_loss"].apply(lambda x: x['is_tlh_trade'] == True))]
        self.assertTrue(len(tlh_sells) == 5, "Expected TLH trades to be marked")
            
    def test_pairs_tlh_realistic_portfolio(self):
        """Test TLH with a realistic 7 ETF portfolio covering major asset classes.
        Some ETFs have TLH pairs, others don't. Most positions are at a loss."""
        current_date = date(2024, 1, 3)  # day 3
        
        # Initial holdings - 7 ETFs with $100,000 portfolio
        tax_lots = pd.DataFrame({
            'identifier': [
                'VTI',      # US Total Market
                'VXUS',     # International Stocks
                'AGG',      # US Aggregate Bonds
                'VNQ',      # US REITs
                'GLD',      # Gold
                'TIP',      # TIPS
                'VTIP'      # Short-term TIPS
            ],
            'tax_lot_id': [
                'VTI_20240101', 'VXUS_20240101', 'AGG_20240101',
                'VNQ_20240101', 'GLD_20240101', 'TIP_20240101',
                'VTIP_20240101'
            ],
            'quantity': [
                200, 200, 400,   # $40k in stocks, $40k bonds
                100, 50, 100, 100  # $10k each in alternatives
            ],
            'cost_basis': [
                40000, 40000, 40000,  # $200/share initial cost
                10000, 10000, 10000, 10000  # $100-200/share initial costs
            ],
            'date': [date(2024, 1, 1)] * 7  # All purchased on day 1
        })

        # Target weights - including TLH alternates where available
        targets = pd.DataFrame([
            {'asset_class': 'US_Equity', 'identifiers': ['VTI', 'ITOT'], 'target_weight': 0.40},  # US stocks with alternates
            {'asset_class': 'Intl_Equity', 'identifiers': ['VXUS', 'IXUS'], 'target_weight': 0.20},  # Int'l with alternate
            {'asset_class': 'US_Bond', 'identifiers': ['AGG'], 'target_weight': 0.15},  # No alternate
            {'asset_class': 'REIT', 'identifiers': ['VNQ', 'SCHH'], 'target_weight': 0.10},  # REITs with alternate
            {'asset_class': 'Gold', 'identifiers': ['GLD'], 'target_weight': 0.05},  # No alternate
            {'asset_class': 'TIPS', 'identifiers': ['TIP', 'SCHP'], 'target_weight': 0.05},  # TIPS with alternate
            {'asset_class': 'Short_TIPS', 'identifiers': ['VTIP'], 'target_weight': 0.05},  # No alternate
            {'asset_class': CASH_CUSIP_ID, 'identifiers': [CASH_CUSIP_ID], 'target_weight': 0.00}
        ])
        
        # Current prices - Most positions down 5-15%, VTIP and GLD at cost basis
        prices = pd.DataFrame({
            'identifier': [
                'VTI', 'ITOT', 'SCHB',  # US stock and alternates
                'VXUS', 'IXUS',         # International and alternate
                'AGG',                  # Bonds (no alternate)
                'VNQ', 'SCHH',          # REITs and alternate
                'GLD',                  # Gold (no alternate)
                'TIP', 'SCHP',          # TIPS and alternate
                'VTIP',                 # Short TIPS (no alternate)
                CASH_CUSIP_ID
            ],
            'price': [
                180.0, 180.0, 180.0,    # US stocks down 10%
                170.0, 170.0,           # International down 15%
                95.0,                   # Bonds down 5%
                85.0, 85.0,             # REITs down 15%
                150.0,                  # Gold down 
                90.0, 90.0,             # TIPS down 10%
                73.0,                   # VTIP down 
                1.0
            ]
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
            tlh_min_loss_threshold=0.03,  # 3% loss threshold
            range_min_weight_multiplier=0.5,
            range_max_weight_multiplier=2,
            rank_penalty_factor=0.5
        )
        
        # Verify optimization succeeded
        self.assertIsNotNone(status)
        self.assertTrue(should_trade)

        print("\nTrade Summary for Realistic Portfolio:", json.dumps(trade_summary, indent=4))
        print("\nDetailed Trades:", trades.to_string())
        
        # Verify that we harvested losses where possible
        sells = trades[trades['action'] == 'sell']
        tlh_sells = sells[sells['gain_loss'].apply(lambda x: x['is_tlh_trade'])]
        buys = trades[trades['action'] == 'buy']
        tlh_buys = buys[buys['gain_loss'].apply(lambda x: x['is_tlh_trade'])]

        # Check that we sold positions with losses and TLH pairs
        self.assertTrue('VTI' in sells['identifier'].values)  # Should sell VTI (down 10%, has alternate)
        self.assertTrue('VXUS' in sells['identifier'].values)  # Should sell VXUS (down 15%, has alternate)
        self.assertTrue('VNQ' in sells['identifier'].values)  # Should sell VNQ (down 15%, has alternate)
        self.assertTrue('TIP' in sells['identifier'].values)  # Should sell TIP (down 10%, has alternate)
        
        # Check that we bought the corresponding alternates
        self.assertTrue(any(x in buys['identifier'].values for x in ['ITOT', 'SCHB']))  # Should buy VTI alternate
        self.assertTrue('IXUS' in buys['identifier'].values)  # Should buy VXUS alternate
        self.assertTrue('SCHH' in buys['identifier'].values)  # Should buy VNQ alternate
        self.assertTrue('SCHP' in buys['identifier'].values)  # Should buy TIP alternate
        
        # Verify we didn't sell positions without alternates
        self.assertFalse('GLD' in sells['identifier'].values)  # Shouldn't sell GLD (no alternate)
        self.assertFalse('VTIP' in sells['identifier'].values)  # Shouldn't sell VTIP (no alternate)
        
        # Verify all sells are actually at a loss
        for _, sell in sells.iterrows():
            stock_price = prices[prices['identifier'] == sell['identifier']]['price'].iloc[0]
            original_cost = tax_lots[tax_lots['identifier'] == sell['identifier']]['cost_basis'].iloc[0] / \
                          tax_lots[tax_lots['identifier'] == sell['identifier']]['quantity'].iloc[0]
            loss_pct = (original_cost - stock_price) / original_cost
            self.assertGreater(loss_pct, 0.03)  # Verify loss is greater than threshold
            
        # Verify that the total portfolio value stays roughly the same
        total_tlh_sells = (tlh_sells['trade_value']).sum()
        total_tlh_buys = (tlh_buys['trade_value']).sum()
        self.assertAlmostEqual(total_tlh_sells, total_tlh_buys, delta=1.0)  # Allow for small rounding differences
            