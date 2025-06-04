import pulp
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional

from src.service.helpers.constants import logger

def _create_buy_dataframe(
    buys: Dict[str, pulp.LpVariable],
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a standardized DataFrame from buy variables.
    
    Args:
        buys: Dictionary of buy variables
        include_prices: Whether to merge with price data
        exclude_cash: Whether to exclude cash positions
        
    Returns:
        DataFrame with buy variables and optional price data
    """
    # Create base DataFrame from buys dictionary
    buy_df = pd.DataFrame([
        {'identifier': id, 'buy_var': var} 
        for id, var in buys.items() 
    ])
    
    # Return empty DataFrame if no buys
    if buy_df.empty:
        return buy_df

    buy_df = buy_df.merge(
        prices[['identifier', 'price']], 
        on='identifier', 
        how='left'
    )
        
    return buy_df
    
def _create_sell_dataframe(
    sells: Dict[str, pulp.LpVariable],
    gain_loss_report: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a standardized DataFrame from sell variables.
    
    Args:
        sells: Dictionary of sell variables
        include_prices: Whether to merge with price data
        exclude_cash: Whether to exclude cash positions
        
    Returns:
        DataFrame with sell variables and optional price data
    """
    # Create base DataFrame from sells dictionary
    sell_df = pd.DataFrame([
        {'tax_lot_id': tid, 'sell_var': var} 
        for tid, var in sells.items()
    ])
    
    # Return empty DataFrame if no sells
    if sell_df.empty:
        return sell_df
        
    # Merge with gain_loss to get identifiers
    sell_df = sell_df.merge(
        gain_loss_report[['tax_lot_id', 'identifier']], 
        on='tax_lot_id', 
        how='inner'
    )
        
    # Optionally merge with prices
    sell_df = sell_df.merge(
        prices[['identifier', 'price']], 
        on='identifier', 
        how='left'
    )
        
    return sell_df

def create_decision_variables(
    buy_identifiers: list[str],
    gain_loss: pd.DataFrame,
    prices: pd.DataFrame,
    debug: bool = False
) -> tuple[dict, dict, dict]:
    """
    Create decision variables for the optimization problem.
    
    Args:
        drift: DataFrame with drift report
        gain_loss: DataFrame with gain/loss report
        debug: Enable debug logging
        
    Returns:
        Tuple of (buys, sells) dictionaries
    """
    buys = {}
    sells = {}
    
    # Buy variables - one per security
    for identifier in buy_identifiers:
        buys[identifier] = pulp.LpVariable(
            f"buy_{identifier}",
            lowBound=0,
            cat='Continuous'
        )
        if debug:
            # logger.info for debug steps
            logger.info(f"Created buy variable for {identifier}")
    
    # Sell variables - one per tax lot
    for _, lot in gain_loss.iterrows():
        tax_lot_id = lot['tax_lot_id']
        sells[tax_lot_id] = pulp.LpVariable(
            f"sell_{tax_lot_id}",
            lowBound=0,
            upBound=lot['quantity'],
            cat='Continuous'
        )
        sells[tax_lot_id]
        if debug:
            # logger.info for debug steps
            logger.info(f"Created sell variable for lot {tax_lot_id} ({lot['identifier']}) - max {lot['quantity']} shares")
    
    buy_df = _create_buy_dataframe(buys, prices)
    sell_df = _create_sell_dataframe(sells, gain_loss, prices)
    
    return buys, sells, buy_df, sell_df
