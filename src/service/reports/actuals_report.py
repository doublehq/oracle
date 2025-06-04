import pandas as pd
from src.service.helpers.constants import CASH_CUSIP_ID

def generate_actuals_report(
    tax_lots: pd.DataFrame,
    prices: pd.DataFrame,
    cash: float
) -> pd.DataFrame:
    """
    Calculate actual portfolio weights based on current tax lots and prices.
    
    Args:
        tax_lots: DataFrame of current tax lots
        prices: DataFrame of current prices
        cash: Current cash balance
        
    Returns:
        DataFrame with columns:
        - identifier: Security identifier
        - market_value: Current total value (sum of quantity * price across tax lots)
        - actual_weight: Current portfolio weight
        - quantity: Total quantity held (sum across tax lots)
    """
    if tax_lots.empty and cash == 0:
        return pd.DataFrame(columns=['identifier', 'market_value', 'actual_weight', 'quantity'])
        
    # Calculate market value for each tax lot
    holdings = tax_lots.merge(
        prices[['identifier', 'price']], 
        on='identifier',
        how='left',
        validate='many_to_one'
    )
    holdings['market_value'] = holdings['quantity'] * holdings['price']
    
    # Sum market values and quantities by identifier
    actuals = holdings.groupby('identifier').agg({
        'market_value': 'sum',
        'quantity': 'sum'
    }).reset_index()
    
    # Add cash position (use quantity=1 for cash since it's a single position)
    actuals = pd.concat([
        actuals,
        pd.DataFrame([{
            'identifier': CASH_CUSIP_ID,
            'market_value': cash,
            'quantity': 1.0
        }])
    ], ignore_index=True)
    
    # Calculate weights
    total_value = actuals['market_value'].sum()
    actuals['actual_weight'] = actuals['market_value'] / total_value if total_value > 0 else 0
    
    return actuals[['identifier', 'market_value', 'actual_weight', 'quantity']] 