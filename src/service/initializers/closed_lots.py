import pandas as pd
from typing import Optional

def initialize_closed_lots(closed_lots: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
    """
    Initialize and validate closed lots DataFrame.
    
    Args:
        closed_lots (Optional[pd.DataFrame]): DataFrame with columns:
        - identifier (str): Security identifier
        - quantity (float): Number of shares/units sold
        - cost_basis (float): Original purchase cost of the lot
        - date_acquired (datetime): Original purchase date
        - date_sold (datetime): Date the lot was sold
        - proceeds (float): Amount received from sale
        - realized_gain (float): Proceeds minus cost basis
        
    Returns:
        Optional[pd.DataFrame]: Validated closed lots DataFrame with standardized:
        - Data types for all columns
        - Uppercase identifiers
        - ISO format dates
        - Validated calculations
        Returns None if no closed lots provided
        
    Notes:
    - All numeric values must be non-negative
    - date_sold must be after date_acquired
    - realized_gain must equal proceeds minus cost_basis
    - quantity must be greater than 0
        
    Raises:
        ValueError: If:
        - Required columns are missing
        - Invalid data types in any column
        - Negative values in quantity/cost/proceeds
        - Invalid date relationships
        - Inconsistent gain calculations
    """
    if closed_lots is None or closed_lots.empty:
        return None
        
    required_columns = {
        'identifier', 'quantity', 'cost_basis', 'date_acquired', 
        'date_sold', 'proceeds', 'realized_gain'
    }
    if not set(closed_lots.columns).issuperset(required_columns):
        raise ValueError(f"Closed lots DataFrame missing required columns: {required_columns}")
    
    # Ensure data types
    closed_lots = closed_lots.copy()
    closed_lots['identifier'] = closed_lots['identifier'].astype(str)
    closed_lots['quantity'] = pd.to_numeric(closed_lots['quantity'], errors='raise')
    closed_lots['cost_basis'] = pd.to_numeric(closed_lots['cost_basis'], errors='raise')
    closed_lots['proceeds'] = pd.to_numeric(closed_lots['proceeds'], errors='raise')
    closed_lots['realized_gain'] = pd.to_numeric(closed_lots['realized_gain'], errors='raise')
    closed_lots['date_acquired'] = pd.to_datetime(closed_lots['date_acquired'], errors='raise')
    closed_lots['date_sold'] = pd.to_datetime(closed_lots['date_sold'], errors='raise')
    
    # Validate no negative quantities
    if (closed_lots['quantity'] < 0).any():
        raise ValueError("Closed lots contain negative quantities")
        
    # Validate no negative cost basis
    if (closed_lots['cost_basis'] < 0).any():
        raise ValueError("Closed lots contain negative cost basis")
        
    # Validate date_acquired is before date_sold
    if (closed_lots['date_acquired'] > closed_lots['date_sold']).any():
        raise ValueError("Found closed lots where acquisition date is after sale date")
        
    # Validate realized_gain calculation
    calculated_gain = closed_lots['proceeds'] - closed_lots['cost_basis']
    if not (abs(calculated_gain - closed_lots['realized_gain']) < 1e-6).all():
        raise ValueError("Realized gain values do not match proceeds minus cost basis")
        
    return closed_lots