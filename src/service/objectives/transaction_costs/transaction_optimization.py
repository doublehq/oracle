import pulp
import pandas as pd
from typing import Dict

def get_buy_cost(
    quantity: float,
    spread: float,
    total_value: float,
    transaction_normalization: float = 1.0
) -> float:
    """
    Calculate the buy cost component of the objective function.
    
    Args:
        identifier: String identifier of the security
        spreads: DataFrame with bid-ask spreads
        
    Returns:
        The total buy cost as a percentage of portfolio value
    """
    
    return quantity * spread / total_value * transaction_normalization if total_value > 0 else 0

def get_sell_cost(
    quantity: float,
    spread: float,
    total_value: float,
    transaction_normalization: float = 1.0
) -> float:
    """
    Calculate the sell cost component of the objective function.
    
    Args:
        identifier: String identifier of the security
        spreads: DataFrame with bid-ask spreads
        
    Returns:
        The total sell cost as a percentage of portfolio value
    """    
    return quantity * spread / total_value * transaction_normalization if total_value > 0 else 0


def calculate_transaction_costs(
    buys: Dict[str, pulp.LpVariable],
    sells: Dict[str, pulp.LpVariable],    
    total_value: float,
    spreads: pd.DataFrame,
    transaction_normalization: float
) -> tuple[pulp.LpAffineExpression, float]:
    """
    Calculate the transaction costs component of the objective function.
    Uses pre-calculated per_share_cost from spreads DataFrame.
    
    Args:
        buys: Dictionary of buy variables
        sells: Dictionary of sell variables
        drift: DataFrame with drift report
        gain_loss: DataFrame with gain/loss report
        total_value: Total portfolio value
        prices: DataFrame with current prices
        spreads: DataFrame with bid-ask spreads
        transaction_normalization: Normalization factor for transaction costs
        
    Returns:
        The normalized transaction costs expression
    """

    transaction_impact = (pulp.lpSum([
        buys[key] * get_buy_cost(1, spreads.loc[spreads['identifier'] == key, 'per_share_cost'].iloc[0], total_value, transaction_normalization)
        for key, value in buys.items()
        if key in spreads['identifier'].values
    ] + [
        sells[key] * get_sell_cost(1, spreads.loc[spreads['identifier'] == key, 'per_share_cost'].iloc[0], total_value, transaction_normalization)
        for key, value in sells.items()
        if key in spreads['identifier'].values
    ])) 
    
    # Apply normalization multiplier to transaction costs
    return transaction_impact