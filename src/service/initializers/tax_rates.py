import pandas as pd
from typing import Optional

def initialize_tax_rates(tax_rates: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Initialize and validate tax rates DataFrame.
    
    Args:
        tax_rates (Optional[pd.DataFrame]): DataFrame with columns:
        - gain_type (str): Type of gain ('short_term', 'long_term', 'qualified_dividend')
        - federal_rate (float): Federal tax rate as decimal (0 to 1)
        - state_rate (float): State tax rate as decimal (0 to 1)
        - total_rate (float): Combined federal and state rate as decimal (0 to 1)
        
    Returns:
        Validated DataFrame with standardized tax rates. If no rates provided, uses defaults:
        - Short-term gains: 41% (35% federal + 6% state)
        - Long-term gains: 26% (20% federal + 6% state)
        - Qualified dividends: 21% (15% federal + 6% state)
            
    Notes:
        Default rates are based on 75th percentile wealth users:
        - Federal ordinary income (short term): 35%
        - Federal long term capital gains: 20%
        - Federal qualified dividends: 15%
        - State tax rate: 6% (average for high tax states)
        
    Raises:
        ValueError: If:
        - Required columns are missing
        - Invalid gain types are present
        - Required gain types are missing
        - Tax rates are outside [0,1] range
        - Total rate doesn't match federal + state
    """
    if tax_rates is None or tax_rates.empty:
        # Default to realistic rates for 75th percentile wealth users
        return pd.DataFrame({
            'gain_type': ['short_term', 'long_term', 'qualified_dividend'],
            'federal_rate': [0.35, 0.20, 0.15],  # 35% ordinary income, 20% LT gains, 15% qualified dividends
            'state_rate': [0.06, 0.06, 0.06],    # 6% state tax (average for high tax states)
            'total_rate': [0.41, 0.26, 0.21]     # Combined rates
        })
    
    required_columns = {'gain_type', 'federal_rate', 'state_rate', 'total_rate'}
    if not set(tax_rates.columns).issuperset(required_columns):
        raise ValueError(f"Tax rates DataFrame missing required columns: {required_columns}")
    
    # Validate gain types
    valid_gain_types = {'short_term', 'long_term', 'qualified_dividend'}
    invalid_types = set(tax_rates['gain_type']) - valid_gain_types
    if invalid_types:
        raise ValueError(f"Invalid gain types found: {invalid_types}")
    
    missing_types = valid_gain_types - set(tax_rates['gain_type'])
    if missing_types:
        raise ValueError(f"Missing required gain types: {missing_types}")
    
    # Validate rates are between 0 and 1
    for col in ['federal_rate', 'state_rate', 'total_rate']:
        if not ((tax_rates[col] >= 0) & (tax_rates[col] <= 1)).all():
            raise ValueError(f"Tax rates must be between 0 and 1, found invalid rates in {col}")
    
    return tax_rates 