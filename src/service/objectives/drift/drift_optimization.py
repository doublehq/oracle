import pulp
import pandas as pd
import time
from typing import Dict, List
from src.service.helpers.constants import CASH_CUSIP_ID
from src.service.helpers.constants import logger
from src.service.helpers.piecewise_linear import create_piecewise_deviation_variable
from src.service.reports.drift_report import PositionStatus

def get_buy_weight_change(
    quantity: float,
    price: float,
    total_value: float,
) -> float:
    """
    Calculate the weight change from buying a security.
    
    Args:
        quantity: Quantity of the security to buy
        prices: DataFrame with current prices
        total_value: Total portfolio value
        
    Returns:
        The weight change from buying the security
    """
    return (quantity * price) / total_value if total_value > 0 else 0


def get_sell_weight_change(
    quantity: float,
    price: float,
    total_value: float,
) -> float:
    """
    Calculate the weight change from buying a security.
    
    Args:
        quantity: Quantity of the security to buy
        prices: DataFrame with current prices
        total_value: Total portfolio value
        
    Returns:
        The weight change from buying the security
    """
    return (quantity * price) / total_value if total_value > 0 else 0

def calculate_drift_impact(
    prob: pulp.LpProblem,
    buys: Dict[str, pulp.LpVariable],
    sells: Dict[str, pulp.LpVariable],
    drift: pd.DataFrame,
    gain_report: pd.DataFrame,
    total_value: float,
    prices: pd.DataFrame,
    absolute_drift_normalization: float = 1.0,
    use_piecewise: bool = True,
    rank_penalty_factor: float = 0.0,
    debug: bool = True
) -> pulp.LpAffineExpression:
    """
    Calculate both absolute and relative drift impacts.
    - Absolute drift measures deviation as % of portfolio (comparable to tax impact)
    - Relative drift measures proportional deviation from target (e.g. 50% off target vs 10% off target)
    Both components are weighted equally to ensure we consider both the size of the position
    and how far it is from its target in relative terms.
    
    Args:
        prob: The optimization problem to add constraints to
        buys: Dictionary of buy variables
        sells: Dictionary of sell variables
        drift: DataFrame with drift report
        gain_report: DataFrame with gain/loss report
        total_value: Total portfolio value
        prices: DataFrame with current prices
        debug: Enable debug logging
        absolute_drift_normalization: Normalization factor for absolute drift (default 1.0)
        use_piecewise: Whether to use piecewise linear approximation (True) or absolute deviation (False)
        rank_penalty_factor: Factor to penalize non-primary securities within an asset class (default 0.1)
    
    Returns:
        Combined drift impact expression incorporating both absolute and relative components with equal weights
    """
    drift = drift.copy()

    if debug:
        logger.info("Calculating Drift Impact ===")
        logger.info(f"  Total portfolio value: ${total_value:,.2f}")
        logger.info(f"  Using {'piecewise' if use_piecewise else 'linear'} deviation")
        logger.info(f"  Rank penalty factor: {rank_penalty_factor}")

    drift_impacts = []
    
    # Handle each position's drift at asset class level
    for _, row in drift.iterrows():
        asset_class = row['asset_class']
        if asset_class == CASH_CUSIP_ID:
            continue
        
        target_weight = row['target_weight']
        actual_weight = row['actual_weight']
        identifiers = row['identifiers']
        
        # Calculate total weight changes for all identifiers in this asset class
        total_buy_weight_change = 0
        total_sell_weight_change = 0
        
        for identifier in identifiers:
            price = prices.loc[prices['identifier'] == identifier, 'price'].iloc[0]

            # Add buy weight changes for this identifier
            if identifier in buys:
                total_buy_weight_change += buys[identifier] * get_buy_weight_change(1, price, total_value)
            
            # Add sell weight changes for all tax lots of this identifier
            for tax_lot_id, value in sells.items():
                # Check if this tax lot belongs to this identifier
                if gain_report.loc[gain_report['tax_lot_id'] == tax_lot_id, 'identifier'].iloc[0] == identifier:
                    total_sell_weight_change += sells[tax_lot_id] * get_sell_weight_change(1, price, total_value)
        
        # Calculate new weight for entire asset class after all trades
        new_weight = actual_weight + total_buy_weight_change - total_sell_weight_change
        
        # Calculate deviation at asset class level
        absolute_deviation = new_weight - target_weight
        
        if use_piecewise:
            avg_target_weight = drift['target_weight'].mean()
            absolute_drift = create_piecewise_deviation_variable(
                prob=prob,
                deviation=absolute_deviation,
                variable_name=f"absolute_drift_{asset_class}",  # Use asset_class instead of identifier
                normalization=absolute_drift_normalization,
            )
        else:
            # Create absolute deviation variable using linear method
            abs_deviation = pulp.LpVariable(f"abs_drift_{asset_class}", lowBound=0)
            prob += abs_deviation >= absolute_deviation, f"abs_drift_pos_{asset_class}"
            prob += abs_deviation >= -absolute_deviation, f"abs_drift_neg_{asset_class}"
            absolute_drift = abs_deviation * absolute_drift_normalization
            
        drift_impacts.append(absolute_drift)

    # Add rank penalties if enabled
    if rank_penalty_factor > 0:
        for _, row in drift.iterrows():
            asset_class = row['asset_class']
            if asset_class == CASH_CUSIP_ID:
                continue
                
            identifiers = row['identifiers']
            
            # Add penalties/rewards based on rank within asset class
            for rank, identifier in enumerate(identifiers):
                if rank == 0:  # Skip primary security
                    continue
                    
                price = prices.loc[prices['identifier'] == identifier, 'price'].iloc[0]
                
                # Penalize buys of non-primary securities
                if identifier in buys:
                    buy_weight_change = buys[identifier] * get_buy_weight_change(1, price, total_value)
                    rank_penalty = rank_penalty_factor * rank * buy_weight_change
                    drift_impacts.append(rank_penalty)
                
                # Reward sells of non-primary securities
                for tax_lot_id, value in sells.items():
                    if gain_report.loc[gain_report['tax_lot_id'] == tax_lot_id, 'identifier'].iloc[0] == identifier:
                        sell_weight_change = sells[tax_lot_id] * get_sell_weight_change(1, price, total_value)
                        rank_reward = -rank_penalty_factor * rank * sell_weight_change
                        drift_impacts.append(rank_reward)

    # Sum all impacts
    total_drift = pulp.lpSum(drift_impacts)
    
    if debug:
        logger.info(f"\nTotal drift impact terms calculated: {len(drift_impacts)}")
    
    return total_drift


def calculate_withdrawal_objective(
    prob: pulp.LpProblem,
    buys: Dict[str, pulp.LpVariable],
    sells: Dict[str, pulp.LpVariable],
    drift: pd.DataFrame,
    gain_loss: pd.DataFrame,
    total_value: float,
    prices: pd.DataFrame,
    withdrawal_amount: float,
    debug: bool = True
) -> pulp.LpAffineExpression:
    """
    Calculate constraints and objective components for a portfolio withdrawal.
    
    Args:
        prob: The optimization problem to add constraints to
        buys: Dictionary of buy variables
        sells: Dictionary of sell variables
        drift: DataFrame with drift report
        gain_loss: DataFrame with gain/loss report
        total_value: Total portfolio value
        prices: DataFrame with current prices
        withdrawal_amount: Amount to be withdrawn from the portfolio
        debug: Enable debug logging
        
    Returns:
        The withdrawal penalty expression (usually 0 since this adds constraints)
    """
    if withdrawal_amount <= 0:
        return 0  # No withdrawal, no constraints needed
    
    # Get cash information from drift report
    cash_row = drift[drift['asset_class'] == CASH_CUSIP_ID]
    if cash_row.empty:
        if debug:
            logger.warning("CASH_CUSIP_ID not found in drift report for withdrawal calculation.")
        return 0  # No cash position defined
        
    current_cash = cash_row['actual_weight'].iloc[0] * total_value
    
    if debug:
        logger.info("Calculating Withdrawal Constraints ===")
        logger.info(f"  Current cash: ${current_cash:.2f}")
        logger.info(f"  Withdrawal amount: ${withdrawal_amount:.2f}")
    
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
    
    # Calculate new cash after trades and withdrawal
    new_cash = (current_cash + total_sells - total_buys - withdrawal_amount)
    
    # Constraint: Ensure new cash is non-negative
    prob += new_cash >= 0, "withdrawal_cash_constraint"
    
    if debug:
        logger.info(f"Added withdrawal constraint: new_cash >= 0")
    
    # No penalty term for the objective - this just adds constraints
    return 0

def calculate_drift_impact_vectorized(
    prob: pulp.LpProblem,
    buy_df: pd.DataFrame,  # DataFrame with columns: identifier, buy_var, price
    sell_df: pd.DataFrame,  # DataFrame with columns: tax_lot_id, sell_var, identifier, price
    drift: pd.DataFrame,    # DataFrame with columns: asset_class, target_weight, actual_weight, identifiers
    total_value: float,
    absolute_drift_normalization: float = 1.0,
    use_piecewise: bool = True,
    rank_penalty_factor: float = 0.0,
    debug: bool = True,
    log_time: bool = True
) -> pulp.LpAffineExpression:
    """
    Vectorized version of calculate_drift_impact that avoids nested loops.
    
    Key vectorization steps:
    1. Process all buys at once by grouping buy_df by identifier
    2. Process all sells at once by grouping sell_df by identifier
    3. Merge weight changes into drift DataFrame for efficient processing
    
    Args:
        prob: The optimization problem to add constraints to
        buy_df: DataFrame containing buy variables and prices
        sell_df: DataFrame containing sell variables, tax lots and prices
        drift: DataFrame with drift report containing asset class groupings
        total_value: Total portfolio value
        absolute_drift_normalization: Normalization factor for absolute drift
        use_piecewise: Whether to use piecewise linear approximation
        rank_penalty_factor: Factor to penalize non-primary securities
        debug: Enable debug logging
        log_time: Whether to log timing information for each step
    
    Returns:
        Combined drift impact expression
    """
    start_time = time.time()
    
    if debug:
        logger.info("Calculating Vectorized Drift Impact ===")
        logger.info(f"Total portfolio value: ${total_value:,.2f}")
        logger.info(f"Using {'piecewise' if use_piecewise else 'linear'} deviation")
        logger.info(f"Rank penalty factor: {rank_penalty_factor}")
    
    drift_impacts = []
    drift = drift.copy()
    
    # Step 1: Calculate buy weight changes
    step1_start = time.time()
    if not buy_df.empty:
        # Calculate weight change for each buy
        buy_df['weight_change'] = buy_df.apply(
            lambda row: row['buy_var'] * get_buy_weight_change(1, row['price'], total_value),
            axis=1
        )
        # Group by identifier to get total buy weight changes
        buy_weight_changes = buy_df.groupby('identifier')['weight_change'].sum()
    else:
        buy_weight_changes = pd.Series(dtype=float)
    if log_time:
        logger.info(f"Step 1 - Buy weight calculations: {time.time() - step1_start:.3f}s")
    
    # Step 2: Calculate sell weight changes
    step2_start = time.time()
    if not sell_df.empty:
        # Calculate weight change for each sell
        sell_df['weight_change'] = sell_df.apply(
            lambda row: row['sell_var'] * get_sell_weight_change(1, row['price'], total_value),
            axis=1
        )
        # Group by identifier to get total sell weight changes
        sell_weight_changes = sell_df.groupby('identifier')['weight_change'].sum()
    else:
        sell_weight_changes = pd.Series(dtype=float)
    if log_time:
        logger.info(f"Step 2 - Sell weight calculations: {time.time() - step2_start:.3f}s")
    
    # Step 3: Process drift by asset class
    step3_start = time.time()
    for _, row in drift.iterrows():
        asset_class = row['asset_class']
        if asset_class == CASH_CUSIP_ID:
            continue
            
        target_weight = row['target_weight']
        actual_weight = row['actual_weight']
        identifiers = row['identifiers']
        
        # Get total buy and sell changes for all identifiers in this asset class
        total_buy_change = sum(buy_weight_changes.get(id, 0) for id in identifiers)
        total_sell_change = sum(sell_weight_changes.get(id, 0) for id in identifiers)
        
        # Calculate new weight and deviation
        new_weight = actual_weight + total_buy_change - total_sell_change
        absolute_deviation = new_weight - target_weight
        
        # Create deviation variable (piecewise or linear)
        if use_piecewise:
            absolute_drift = create_piecewise_deviation_variable(
                prob=prob,
                deviation=absolute_deviation,
                variable_name=f"absolute_drift_{asset_class}",
                normalization=absolute_drift_normalization
            )
        else:
            abs_deviation = pulp.LpVariable(f"abs_drift_{asset_class}", lowBound=0)
            prob += abs_deviation >= absolute_deviation, f"abs_drift_pos_{asset_class}"
            prob += abs_deviation >= -absolute_deviation, f"abs_drift_neg_{asset_class}"
            absolute_drift = abs_deviation * absolute_drift_normalization
            
        drift_impacts.append(absolute_drift)
    if log_time:
        logger.info(f"Step 3 - Process drift by asset class: {time.time() - step3_start:.3f}s")
    
    # Step 4: Handle rank penalties if enabled
    step4_start = time.time()
    if rank_penalty_factor > 0:
        for _, row in drift.iterrows():
            asset_class = row['asset_class']
            if asset_class == CASH_CUSIP_ID:
                continue
                
            identifiers = row['identifiers']
            
            # Process all non-primary securities at once
            for rank, identifier in enumerate(identifiers[1:], start=1):  # Skip primary security
                # Add buy penalties
                if identifier in buy_weight_changes:
                    buy_penalty = rank_penalty_factor * rank * buy_weight_changes[identifier]
                    drift_impacts.append(buy_penalty)
                
                # Add sell rewards
                if identifier in sell_weight_changes:
                    sell_reward = -rank_penalty_factor * rank * sell_weight_changes[identifier]
                    drift_impacts.append(sell_reward)
    if log_time:
        logger.info(f"Step 4 - Handle rank penalties: {time.time() - step4_start:.3f}s")
    
    # Step 5: Combine all impacts
    step5_start = time.time()
    total_drift = pulp.lpSum(drift_impacts)
    if log_time:
        logger.info(f"Step 5 - Combine impacts: {time.time() - step5_start:.3f}s")
        logger.info(f"Total execution time: {time.time() - start_time:.3f}s")
    
    if debug:
        logger.info(f"\nTotal drift impact terms calculated: {len(drift_impacts)}")
    
    return total_drift