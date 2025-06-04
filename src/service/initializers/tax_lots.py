import pandas as pd
from src.service.helpers.constants import CASH_CUSIP_ID

def initialize_tax_lots(tax_lots: pd.DataFrame) -> pd.DataFrame:
    """
    Initialize and validate tax lots DataFrame.
    
    Args:
        tax_lots: DataFrame with columns:
        - tax_lot_id (str, optional): Unique identifier for each tax lot
        - identifier (str): Security identifier
        - quantity (float): Number of shares/units
        - cost_basis (float): Total cost basis of the lot
        - date or date_acquired (datetime): Purchase date of the lot
        
    Returns:
        Validated tax lots DataFrame with standardized columns and types:
        - tax_lot_id (str): Unique identifier for each tax lot (generated if not provided)
        - identifier (str): Security identifier (standardized to uppercase)
        - quantity (float): Number of shares/units (validated as non-negative)
        - cost_basis (float): Total cost basis of the lot (validated as non-negative)
        - date (datetime): Acquisition date (standardized column name from date or date_acquired)
        
    Raises:
        ValueError: If tax lots DataFrame is missing required columns or contains invalid data
    """
    if tax_lots is None or tax_lots.empty:
        return pd.DataFrame(columns=['tax_lot_id', 'identifier', 'quantity', 'cost_basis', 'date'])

    # Check for either date or date_acquired
    if 'date_acquired' in tax_lots.columns and 'date' not in tax_lots.columns:
        tax_lots = tax_lots.rename(columns={'date_acquired': 'date'})
    elif 'date' not in tax_lots.columns:
        raise ValueError("Tax lots DataFrame must have either 'date' or 'date_acquired' column")

    required_columns = {'identifier', 'quantity', 'cost_basis', 'date'}
    if not set(tax_lots.columns).issuperset(required_columns):
        raise ValueError(f"Tax lots DataFrame missing required columns: {required_columns}")
    
    # Ensure data types
    tax_lots = tax_lots.copy()
    tax_lots['identifier'] = tax_lots['identifier'].astype(str)
    
    # Replace any 'CASH' or 'cash' identifiers with CASH_CUSIP_ID
    cash_mask = tax_lots['identifier'].str.upper() == 'CASH'
    if cash_mask.any():
        tax_lots.loc[cash_mask, 'identifier'] = CASH_CUSIP_ID
    
    tax_lots['quantity'] = pd.to_numeric(tax_lots['quantity'], errors='raise')
    tax_lots['cost_basis'] = pd.to_numeric(tax_lots['cost_basis'], errors='raise')
    tax_lots['date'] = pd.to_datetime(tax_lots['date'], errors='raise')
    
    # Handle tax lot IDs: preserve existing, fill nulls with unique IDs, or create new if column doesn't exist
    if 'tax_lot_id' not in tax_lots.columns:
        tax_lots['tax_lot_id'] = [f"lot_{i}_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S%f')}" for i in range(len(tax_lots))]
    else:
        # Fill any null values with unique IDs
        null_mask = tax_lots['tax_lot_id'].isnull()
        if null_mask.any():
            timestamp = pd.Timestamp.now().strftime('%Y%m%d%H%M%S%f')
            null_indices = null_mask[null_mask].index
            tax_lots.loc[null_indices, 'tax_lot_id'] = [
                f"lot_{i}_{timestamp}" for i in range(len(null_indices))
            ]
    
    # Validate that all tax lot IDs are unique
    if tax_lots['tax_lot_id'].duplicated().any():
        duplicated_ids = tax_lots.loc[tax_lots['tax_lot_id'].duplicated(), 'tax_lot_id'].tolist()
        raise ValueError(f"Found duplicate tax lot IDs: {duplicated_ids}. All tax lot IDs must be unique.")
    
    # Validate no negative quantities
    if (tax_lots['quantity'] < 0).any():
        raise ValueError("Tax lots contain negative quantities")
        
    # Validate no negative cost basis
    if (tax_lots['cost_basis'] < 0).any():
        raise ValueError("Tax lots contain negative cost basis")
        
    return tax_lots