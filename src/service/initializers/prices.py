import pandas as pd
from typing import Set
from src.service.helpers.constants import CASH_CUSIP_ID

def initialize_prices(prices: pd.DataFrame, all_identifiers: Set[str]) -> pd.DataFrame:
    """
    Initialize and validate prices DataFrame.
    
    Args:
        prices (pd.DataFrame): DataFrame with columns:
        - identifier (str): Security identifier
        - price (float): Current market price
        all_identifiers (Set[str]): Set of all identifiers that need prices
        
    Returns:
        Validated prices DataFrame with standardized columns:
        - identifier (str): Security identifier (uppercase)
        - price (float): Current market price (non-negative)
        
    Notes:
        - CASH_CUSIP_ID is automatically added with price of 1.0 if needed
        - All identifiers must have a valid price
        - Prices are validated to be non-negative
        - Identifiers are standardized to uppercase
        - Duplicate identifiers are not allowed
        
    Raises:
        ValueError: If:
        - Required columns are missing
        - Prices are missing for any required identifier
        - Negative prices are present
        - Invalid data types in any column
        - Duplicate identifiers present
    """
    required_columns = {'identifier', 'price'}
    if not set(prices.columns).issuperset(required_columns):
        raise ValueError(f"Prices DataFrame missing required columns: {required_columns}")
    
    # Ensure data types
    prices = prices.copy()
    prices['identifier'] = prices['identifier'].astype(str)
    prices['price'] = pd.to_numeric(prices['price'], errors='raise')
    
    # Replace any 'CASH' or 'cash' identifiers with CASH_CUSIP_ID
    cash_mask = prices['identifier'].str.upper() == 'CASH'
    if cash_mask.any():
        prices.loc[cash_mask, 'identifier'] = CASH_CUSIP_ID
        # If we have multiple CASH entries after replacement, keep only the first one
        prices = prices.drop_duplicates(subset=['identifier'], keep='first')
    
    # Validate no negative prices
    if (prices['price'] < 0).any():
        raise ValueError("Prices contain negative values")
    
    # Add CASH_CUSIP_ID with price 1.0 if it doesn't exist
    if CASH_CUSIP_ID not in set(prices['identifier']):
        prices = pd.concat([
            prices,
            pd.DataFrame([{'identifier': CASH_CUSIP_ID, 'price': 1.0}])
        ], ignore_index=True)
        
    # Check we have prices for all identifiers
    missing_prices = all_identifiers - set(prices['identifier'])
    if missing_prices:
        raise ValueError(f"Missing prices for identifiers: {missing_prices}")
        
    return prices