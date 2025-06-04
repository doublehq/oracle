import unittest
import pandas as pd
import numpy as np
from src.service.initializers import initialize_targets
from src.service.helpers.constants import CASH_CUSIP_ID

class TestInitializeTargets(unittest.TestCase):
    def test_basic_asset_class_targets(self):
        """Test basic initialization of asset class targets with valid data."""
        targets = pd.DataFrame({
            'asset_class': ['US_EQUITY', 'INTL_EQUITY', 'FIXED_INCOME'],
            'target_weight': [0.6, 0.2, 0.2],
            'identifiers': [
                ['VTI', 'ITOT'],  # US_EQUITY options
                ['VXUS', 'IXUS'],        # INTL_EQUITY options
                ['AGG', 'BND']           # FIXED_INCOME options
            ]
        })
        
        result = initialize_targets(targets)
        
        # Check that CASH was added
        self.assertIn(CASH_CUSIP_ID, result['asset_class'].values)
        
        # Check that weights sum to 1
        self.assertAlmostEqual(result['target_weight'].sum(), 1.0)
        
        # Check that original asset classes were preserved
        original_classes = set(['US_EQUITY', 'INTL_EQUITY', 'FIXED_INCOME'])
        self.assertTrue(original_classes.issubset(set(result['asset_class'])))
        
        # Check that identifiers were converted to uppercase
        cash_row = result[result['asset_class'] == CASH_CUSIP_ID].iloc[0]
        self.assertEqual(cash_row['identifiers'], [CASH_CUSIP_ID])

    def test_cash_handling_with_withdrawal(self):
        """Test that cash targets are properly handled with withdrawal targets."""
        targets = pd.DataFrame({
            'asset_class': ['US_EQUITY', 'INTL_EQUITY',CASH_CUSIP_ID],
            'target_weight': [0.5, 0.5, 0.00],  # Initial cash below withdrawal target
            'identifiers': [
                ['VTI', 'ITOT'],
                ['VXUS', 'IXUS'],
                [CASH_CUSIP_ID]
            ]
        })
        
        withdraw_target = 0.05  # 5% withdrawal target
        result = initialize_targets(targets, withdraw_target=withdraw_target)
        
        # Check that cash weight matches withdrawal target
        cash_weight = result[result['asset_class'] == CASH_CUSIP_ID]['target_weight'].iloc[0]
        self.assertEqual(cash_weight, withdraw_target)
        

    def test_invalid_inputs(self):
        """Test that invalid inputs raise appropriate errors."""
        # Test missing required columns
        with self.assertRaises(ValueError):
            initialize_targets(pd.DataFrame({
                'asset_class': ['US_EQUITY'],
                'target_weight': [1.0]
                # Missing identifiers column
            }))
        
        # Test empty identifier list
        with self.assertRaises(ValueError):
            initialize_targets(pd.DataFrame({
                'asset_class': ['US_EQUITY', 'INTL_EQUITY'],
                'target_weight': [0.6, 0.4],
                'identifiers': [[], ['VXUS']]  # Empty list for US_EQUITY
            }))
        

    def test_deminimus_cash_handling(self):
        """Test that deminimus cash requirements are properly handled."""
        targets = pd.DataFrame({
            'asset_class': ['US_EQUITY', 'INTL_EQUITY'],
            'target_weight': [0.7, 0.3],
            'identifiers': [
                ['VTI', 'ITOT'],
                ['VXUS', 'IXUS']
            ]
        })
        
        deminimus_cash = 0.02  # 2% minimum cash
        result = initialize_targets(targets, deminimus_cash_target=deminimus_cash)
        
        # Check that cash was added with correct weight
        cash_weight = result[result['asset_class'] == CASH_CUSIP_ID]['target_weight'].iloc[0]
        self.assertEqual(cash_weight, deminimus_cash)
        
        # Check that other weights were scaled properly
        us_weight = result[result['asset_class'] == 'US_EQUITY']['target_weight'].iloc[0]
        intl_weight = result[result['asset_class'] == 'INTL_EQUITY']['target_weight'].iloc[0]
        
        # Original ratio was 0.7/0.3, should be preserved
        self.assertAlmostEqual(us_weight/intl_weight, 0.7/0.3)
        
        # Sum should still be 1
        self.assertAlmostEqual(result['target_weight'].sum(), 1.0) 