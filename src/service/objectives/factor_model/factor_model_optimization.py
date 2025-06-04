import time 
import pulp
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from src.service.helpers.constants import CASH_CUSIP_ID
from src.service.helpers.constants import logger
from src.service.helpers.piecewise_linear import create_piecewise_deviation_variable

def calculate_factor_model_impact_vectorized(
    prob: pulp.LpProblem,
    buy_df: pd.DataFrame,  # DataFrame with columns: identifier, buy_var, price
    sell_df: pd.DataFrame,  # DataFrame with columns: tax_lot_id, sell_var, identifier, price
    total_value: float,
    factor_model: pd.DataFrame,
    target_factors: pd.DataFrame,
    actual_factors: pd.DataFrame,
    factor_normalization: float,
    debug: bool = False,
    use_piecewise: bool = True
) -> pulp.LpAffineExpression:
    """
    Vectorized version of calculate_factor_model_impact that avoids nested loops.
    Uses pandas operations for efficient calculation of factor exposures.
    
    Args:
        prob: The optimization problem to add constraints to
        buy_df: DataFrame containing buy variables and prices
        sell_df: DataFrame containing sell variables, tax lots and prices
        drift: DataFrame with drift report
        total_value: Total portfolio value
        factor_model: DataFrame with factor exposures for each security
        target_factors: DataFrame with target factor exposures
        actual_factors: DataFrame with actual factor exposures
        debug: Enable debug logging
        factor_normalization: Normalization factor for factor impact
        use_piecewise: Whether to use piecewise linear approximation
        
    Returns:
        The total factor model impact expression to minimize
    """
    if debug:
        logger.info("Calculating Vectorized Factor Model Impact ===")
        logger.info(f"Total portfolio value: ${total_value:,.2f}")
        logger.info(f"Using {'piecewise' if use_piecewise else 'linear'} deviation")
    
    factor_impacts = []
    
    # Get list of factor columns (excluding identifier)
    factor_cols = [col for col in factor_model.columns if col != 'identifier']
    
    # Step 1: Calculate buy weight changes and merge with factor exposures
    if not buy_df.empty:
        # Calculate weight changes for buys
        buy_df['weight_change'] = buy_df['buy_var'] * buy_df['price'] / total_value
        
        # Merge with factor model to get exposures
        buy_exposures = buy_df.merge(
            factor_model,
            on='identifier',
            how='left'
        )
        
        # Calculate exposure changes for each factor (weight_change * factor_exposure)
        buy_factor_changes = {}
        for factor in factor_cols:
            buy_factor_changes[factor] = (
                buy_exposures['weight_change'] * buy_exposures[factor]
            ).sum()
    else:
        buy_factor_changes = {factor: 0 for factor in factor_cols}
    
    # Step 2: Calculate sell weight changes and merge with factor exposures
    if not sell_df.empty:
        # Calculate weight changes for sells
        sell_df['weight_change'] = sell_df['sell_var'] * sell_df['price'] / total_value
        
        # Merge with factor model to get exposures
        sell_exposures = sell_df.merge(
            factor_model,
            on='identifier',
            how='left'
        )
        # Calculate exposure changes for each factor using vectorized operations
        sell_factor_changes = {}
        if not sell_df.empty:
            # Pure numpy operations for maximum performance
            weight_changes = sell_exposures['weight_change'].to_numpy()
            factor_matrix = sell_exposures[factor_cols].to_numpy()
            # Add debug logging for matrix shapes
            if debug:
                logger.info(f"weight_changes shape: {weight_changes.shape}")
                logger.info(f"factor_matrix shape: {factor_matrix.shape}")
            
            # Calculate factor sums directly
            factor_sums = {factor: 0 for factor in factor_cols}
            start_time = time.time()
            
            # Calculate sum for each factor
            for j, factor in enumerate(factor_cols):
                factor_sum = pulp.lpSum(
                    weight_changes[k] * factor_matrix[k, j]
                    for k in range(len(weight_changes))
                )
                factor_sums[factor] = factor_sum
            
            if debug:
                logger.info(f"Factor sums calculation time: {time.time() - start_time:.3f} seconds")
            # Create dictionary of results  
            sell_factor_changes = factor_sums
        else:
            sell_factor_changes = {factor: 0 for factor in factor_cols}
    else:
        sell_factor_changes = {factor: 0 for factor in factor_cols}
    
    # Step 3: Calculate new exposures and deviations for each factor
    for factor in factor_cols:
        current_exposure = actual_factors[factor].iloc[0]
        target_exposure = target_factors[factor].iloc[0]
        
        # Calculate new exposure after trades
        new_exposure = (
            current_exposure + 
            buy_factor_changes[factor] - 
            sell_factor_changes[factor]
        )
        
        # Calculate deviation from target
        deviation = new_exposure - target_exposure
        
        if debug:
            logger.info(f"\nProcessing factor {factor}:")
            logger.info(f"  Current exposure: {current_exposure:.4f}")
            logger.info(f"  Target exposure: {target_exposure:.4f}")
        
        # Create deviation variable (piecewise or linear)
        if use_piecewise:
            factor_impact = create_piecewise_deviation_variable(
                prob=prob,
                deviation=deviation,
                variable_name=f"factor_{factor}",
                normalization=factor_normalization
            )
        else:
            abs_deviation = pulp.LpVariable(f"abs_factor_{factor}", lowBound=0)
            prob += abs_deviation >= deviation, f"abs_factor_pos_{factor}"
            prob += abs_deviation >= -deviation, f"abs_factor_neg_{factor}"
            factor_impact = abs_deviation * factor_normalization
            
        factor_impacts.append(factor_impact)
    
    # Step 4: Combine all impacts
    total_factor_impact = pulp.lpSum(factor_impacts)
    
    if debug:
        logger.info(f"\nTotal factor impact terms calculated: {len(factor_impacts)}")
    
    return total_factor_impact
