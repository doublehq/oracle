import pulp
import pandas as pd
from typing import Dict
from src.service.helpers.constants import CASH_CUSIP_ID
from src.service.helpers.constants import logger

def calculate_cash_deployment_objective(
    prob: pulp.LpProblem,
    buys: Dict[str, pulp.LpVariable],
    sells: Dict[str, pulp.LpVariable],
    drift: pd.DataFrame,
    gain_loss: pd.DataFrame,
    total_value: float,
    prices: pd.DataFrame,
    cash_normalization: float,
    debug: bool = True,
) -> pulp.LpAffineExpression:
    """
    Calculate a penalty for holding excess cash above target.
    Only penalizes cash when it's above target, not when below.
    
    Args:
        prob: The optimization problem to add constraints to
        buys: Dictionary of buy variables
        sells: Dictionary of sell variables
        drift: DataFrame with drift report
        gain_loss: DataFrame with gain/loss report
        total_value: Total portfolio value
        prices: DataFrame with current prices
        debug: Enable debug logging
        cash_normalization: Normalization factor for cash penalty
        
    Returns:
        The cash penalty expression
    """
    # Get cash information from drift report
    cash_row = drift[drift['asset_class'] == CASH_CUSIP_ID]
    if cash_row.empty:
        if debug:
            logger.warning("CASH_CUSIP_ID not found in drift report for cash penalty calculation.")
        return 0  # No cash position defined
        
    cash_target = cash_row['target_weight'].iloc[0]
    current_cash = cash_row['actual_weight'].iloc[0]
    
    if debug:
        logger.info("Calculating Cash Deployment Component ===")
        logger.info(f"  Current cash weight: {current_cash:.4%}")
    
    # Calculate total buys and sells in dollar terms
    total_buys = 0
    for identifier, buy_var in buys.items():
        if identifier != CASH_CUSIP_ID:
            price = prices.loc[prices['identifier'] == identifier, 'price'].iloc[0]
            total_buys += buy_var * price
            
    total_sells = 0
    for _, lot in gain_loss.iterrows():
        tax_lot_id = lot['tax_lot_id']
        if tax_lot_id in sells:
            price = prices.loc[prices['identifier'] == lot['identifier'], 'price'].iloc[0]
            total_sells += sells[tax_lot_id] * price
    
    # Calculate new cash weight after trades
    new_cash_dollars = (current_cash * total_value) + total_sells - total_buys
    new_cash_weight = new_cash_dollars / total_value
    
    # Calculate initial excess cash
    initial_excess_cash = max(0, current_cash - cash_target)
    
    # Create variable for excess cash after trades
    excess_cash = pulp.LpVariable(f"excess_cash", lowBound=0)
    prob += new_cash_weight - cash_target <= excess_cash, "cash_excess_constr"
    
    if debug:
        logger.info(f"Added cash deployment component (minimizing new_cash_weight)")
        logger.info(f"  Initial excess cash: {initial_excess_cash:.4%}")
    
    # Create list of terms to sum
    cash_terms = [(excess_cash - initial_excess_cash) * cash_normalization]
    
    # Return the sum of all cash terms
    return pulp.lpSum(cash_terms)


def calculate_max_withdrawal_objective(
    self,
    prob: pulp.LpProblem,
    buys: Dict[str, pulp.LpVariable],
    sells: Dict[str, pulp.LpVariable],
    gain_loss: pd.DataFrame,
    debug: bool = True
) -> pulp.LpAffineExpression:
    """
    Calculate an objective that maximizes cash withdrawal.
    Creates an objective that maximizes the cash generated from selling securities.
    
    Args:
        prob: PuLP optimization problem
        buys: Dictionary of buy variables
        sells: Dictionary of sell variables
        gain_loss: DataFrame with gain/loss report
        debug: Enable debug logging
        
    Returns:
        Cash withdrawal maximization objective term (negative since we're minimizing)
    """
    if debug:
        logger.info("Calculating max withdrawal objective")
        
    # Create a cash generation objective (for maximizing withdrawal)
    # We want to maximize cash, which means minimizing the negative of cash
    total_cash_generated = 0
    for _, lot in gain_loss.iterrows():
        tax_lot_id = lot['tax_lot_id']
        identifier = lot['identifier']
        price = self.strategy.prices.loc[self.strategy.prices['identifier'] == identifier, 'price'].iloc[0]
        total_cash_generated += sells[tax_lot_id] * price
    
    # In a minimization problem, we minimize the negative of what we want to maximize
    return -1 * total_cash_generated
    