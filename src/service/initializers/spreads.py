import pandas as pd
from typing import Set, Optional
from src.service.helpers.constants import CASH_CUSIP_ID

def initialize_spreads(spreads: pd.DataFrame | None, all_identifiers: Set[str], prices: pd.DataFrame) -> pd.DataFrame:
    """
    Initialize spreads DataFrame. If no spreads provided, create default DataFrame
    with 0.0001 (1bps) spread for all identifiers.
    
    Args:
        spreads (pd.DataFrame | None): Optional DataFrame with columns:
        - identifier (str): Security identifier
        - spread (float): Bid-ask spread as decimal (e.g., 0.0001 for 1bps)
        all_identifiers (Set[str]): Set of all identifiers that need spreads
        prices (pd.DataFrame): DataFrame with columns:
        - identifier (str): Security identifier
        - price (float): Current price used to calculate per_share_cost
        
    Returns:
        DataFrame with spreads for all identifiers:
        - identifier (str): Security identifier
        - spread (float): Bid-ask spread as decimal
        - per_share_cost (float): Pre-calculated transaction cost per share (price * spread)
        
    Notes:
        - Default spread of 0.0001 (1bps) is used for any missing identifiers
        - CASH_CUSIP_ID is automatically added with 0 spread if needed
        - All spreads are validated to be non-negative
        
    Raises:
        ValueError: If:
        - Spreads DataFrame is provided but missing required columns
        - Contains negative spread values
        - Contains duplicate identifiers
    """
    if spreads is not None and not spreads.empty:
        required_columns = {'identifier', 'spread'}
        if not set(spreads.columns).issuperset(required_columns):
            raise ValueError(f"Spreads DataFrame missing required columns: {required_columns}")
        
        # Ensure data types
        spreads = spreads.copy()
        spreads['identifier'] = spreads['identifier'].astype(str)
        spreads['spread'] = pd.to_numeric(spreads['spread'], errors='raise')
        
        # Validate no negative spreads
        if (spreads['spread'] < 0).any():
            raise ValueError("Spreads contain negative values")
        
        # Ensure all required identifiers have spreads
        missing_identifiers = all_identifiers - set(spreads['identifier'])
        if missing_identifiers:
            # Add default spreads for missing identifiers
            default_spreads = pd.DataFrame({
                'identifier': list(missing_identifiers),
                'spread': 0.0001  # Default 1bps spread
            })
            spreads = pd.concat([spreads, default_spreads], ignore_index=True)
    else:
        # Create default spreads DataFrame if none provided
        spreads = pd.DataFrame({
            'identifier': list(all_identifiers),
            'spread': 0.0003  # Default 3bps spread
        })
    
    # Join with prices to calculate per_share_cost
    spreads = spreads.merge(
        prices[['identifier', 'price']], 
        on='identifier', 
        how='left',
        validate='one_to_one'
    )
    
    # Calculate per_share_cost
    spreads['per_share_cost'] = spreads['price'] * spreads['spread']
    
    # Drop the price column as it's no longer needed
    spreads = spreads.drop(columns=['price'])
    
    return spreads