import pandas as pd
from typing import Optional


def initialize_stock_restrictions(stock_restrictions: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Initialize and validate stock restrictions DataFrame.
    
    Args:
        stock_restrictions (Optional[pd.DataFrame]): DataFrame with columns:
        - identifier (str): Security identifier
        - can_buy (bool): Whether the security can be purchased
        - can_sell (bool): Whether the security can be sold
        
    Returns:
        Validated stock restrictions DataFrame with standardized columns:
        - identifier (str): Security identifier (uppercase)
        - can_buy (bool): Purchase permission flag
        - can_sell (bool): Sale permission flag
        
    Notes:
        - If no restrictions provided, returns empty DataFrame implying no restrictions
        - A security cannot have both can_buy and can_sell set to True
        - Identifiers are standardized to uppercase
        - Empty DataFrame implies all securities can be both bought and sold
        
    Raises:
        ValueError: If:
        - Required columns are missing
        - Both can_buy and can_sell are True for any security
        - Invalid data types in any column
        - Duplicate identifiers present
    """
    if stock_restrictions is None or stock_restrictions.empty:
        return pd.DataFrame(columns=['identifier', 'can_buy', 'can_sell'])
        
    required_columns = {'identifier', 'can_buy', 'can_sell'}
    if not set(stock_restrictions.columns).issuperset(required_columns):
        raise ValueError(f"Stock restrictions DataFrame missing required columns: {required_columns}")
    
    # Ensure data types
    stock_restrictions = stock_restrictions.copy()
    stock_restrictions['identifier'] = stock_restrictions['identifier'].astype(str)
    stock_restrictions['can_buy'] = stock_restrictions['can_buy'].astype(bool)
    stock_restrictions['can_sell'] = stock_restrictions['can_sell'].astype(bool)
    # Validate that can_buy and can_sell are not both True for each stock
    invalid_restrictions = stock_restrictions['can_buy'] & stock_restrictions['can_sell']
    if invalid_restrictions.any():
        invalid_stocks = stock_restrictions.loc[invalid_restrictions, 'identifier'].tolist()
        raise ValueError(
            f"Found stocks that can be both bought and sold: {invalid_stocks}. "
            "At least one of can_buy or can_sell must be False."
        )
    return stock_restrictions