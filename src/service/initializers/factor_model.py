import pandas as pd
import numpy as np
from typing import Tuple
from src.service.helpers.constants import CASH_CUSIP_ID, logger

def normalize_factor_model(factor_model: pd.DataFrame, 
                         preserve_range: float = 0.75,
                         scale_factor: float = 0.1) -> pd.DataFrame:
    """
    Normalize factor model values using a piecewise approach that preserves values
    in the [-preserve_range, preserve_range] range and applies tanh compression beyond that.
    Uses a global maximum across all factors to ensure consistent scaling.
    
    Args:
        factor_model: DataFrame with factor exposures
        preserve_range: Range to preserve without compression (default 0.75)
        scale_factor: Scaling factor for tanh compression beyond preserve_range.
                     Lower values make the transformation more gradual.
                     Higher values make it more aggressive.
                     Default 0.1 provides good coverage for typical factor values.
        
    Returns:
        Normalized factor model DataFrame
    """
    # Make a copy to avoid modifying the original
    normalized = factor_model.copy()
    
    # Get factor columns (excluding identifier)
    factor_cols = [col for col in factor_model.columns if col != 'identifier']
    
    # Find the global maximum absolute value across all factors
    global_max_abs = np.max([np.max(np.abs(normalized[col])) for col in factor_cols])
    
    # Calculate scaling factor to map global_max_abs to preserve_range
    scale = preserve_range / global_max_abs
    
    # Only normalize if we have values outside [-1, 1]
    if global_max_abs > 1:
        for col in factor_cols:
            # Apply piecewise transformation
            mask_inside = np.abs(normalized[col]) <= preserve_range
            mask_outside = ~mask_inside
            
            # For values that would be inside preserve_range after scaling, just scale them
            normalized.loc[mask_inside, col] = normalized[col][mask_inside] * scale
            
            # For values that would be outside preserve_range after scaling
            if mask_outside.any():
                # Calculate the offset to make tanh continuous at preserve_range
                offset = preserve_range - np.tanh(scale_factor * preserve_range)
                
                # Apply tanh compression with offset
                normalized.loc[mask_outside, col] = (
                    np.tanh(scale_factor * normalized[col][mask_outside] * scale) + offset
                )
            
            # Ensure values are within [-1, 1]
            normalized.loc[normalized[col] > 1, col] = 1.0
            normalized.loc[normalized[col] < -1, col] = -1.0
    
    return normalized

def initialize_factor_model(factor_model: pd.DataFrame, targets: pd.DataFrame, actuals: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Initialize and validate a factor model DataFrame, and compute weighted average factors.
    
    Args:
        factor_model (pd.DataFrame): DataFrame with columns:
        - identifier (str): Security identifier
        - [factor_name] (float): One or more factor exposure columns
        targets (pd.DataFrame): DataFrame with columns:
        - asset_class (str): Asset class identifier
        - target_weight (float): Target portfolio weight
        - identifiers (list[str]): List of valid identifiers for each asset class
        actuals (pd.DataFrame): DataFrame with columns:
        - identifier (str): Security identifier
        - actual_weight (float): Current portfolio weight
    
    Returns:
        Tuple containing:
        - factor_model (pd.DataFrame): Normalized factor exposures for each security
        - target_exposures (pd.DataFrame): Target portfolio's weighted average factor exposures
        - actual_exposures (pd.DataFrame): Current portfolio's weighted average factor exposures
        
    Notes:
        - Factor values are normalized using tanh-based compression
        - Weighted averages are computed using target and actual weights
        - Missing factors are handled gracefully with appropriate warnings
        - Factors are validated for numerical stability
        
    Raises:
        ValueError: If:
        - Factor model is missing required columns
        - No numeric factor columns are present
        - Non-numeric data in factor columns
        - Missing entries for required identifiers
        - Null values in any column
    """
        
    # Validate factor model structure
    if 'identifier' not in factor_model.columns:
        raise ValueError("Factor model must have 'identifier' column")
    
    # Check that we have at least one numeric column
    if len(factor_model.columns) <= 1:  # Only identifier column
        raise ValueError("Factor model must have at least one numeric factor column besides identifier")
    
    factor_model = factor_model.copy()
    factor_model['identifier'] = factor_model['identifier'].astype(str)
    
    # Check for null values in any column
    null_counts = factor_model.isnull().sum()
    columns_with_nulls = null_counts[null_counts > 0]
    if not columns_with_nulls.empty:
        raise ValueError(f"Found null values in the following columns: {dict(columns_with_nulls)}")
    
    # Verify all non-identifier columns are numeric
    non_numeric_cols = factor_model.drop(columns=['identifier']).select_dtypes(exclude=['float64', 'int64']).columns
    if len(non_numeric_cols) > 0:
        raise ValueError(f"All factor columns must be numeric. Non-numeric columns found: {list(non_numeric_cols)}")
    
    # Normalize the factor model
    factor_model = normalize_factor_model(factor_model)
    
    # Check if CASH_CUSIP_ID exists in targets
    cash_exists = CASH_CUSIP_ID in set(factor_model['identifier'])
    
    # If cash doesn't exist, add it with zero factors
    if not cash_exists:
        # Add cash with zero factors
        cash_row = pd.DataFrame({
            'identifier': [CASH_CUSIP_ID],
            **{col: [0.0] for col in factor_model.columns if col != 'identifier'},
        })
        factor_model = pd.concat([factor_model, cash_row], ignore_index=True)
    
    # Get all identifiers from targets
    all_target_identifiers = set([id_ for ids in targets['identifiers'] for id_ in ids])
    
    # Check that we have factor model entries for all required identifiers
    if all_target_identifiers:
        missing_identifiers = all_target_identifiers - set(factor_model['identifier'])
        if missing_identifiers:
            raise ValueError(f"Missing factor model entries for identifiers: {missing_identifiers}")
    
    # Create a mapping of identifier to target weight by distributing asset class weights
    # This only makes sense because the identifiers in a asset_class are length 1
    identifier_target_weights = {}
    for _, row in targets.iterrows():
        asset_class_weight = row['target_weight']
        identifiers = row['identifiers']
        # Distribute asset class weight equally among identifiers
        if identifiers:  # Skip if empty list
            weight_per_identifier = asset_class_weight / len(identifiers)
            for identifier in identifiers:
                identifier_target_weights[identifier] = weight_per_identifier
    
    # Create a DataFrame with identifier weights
    identifier_weights = pd.DataFrame({
        'identifier': list(identifier_target_weights.keys()),
        'target_weight': list(identifier_target_weights.values())
    })
    
    # Compute weighted average factors for targets
    # First merge factor model with identifier weights
    weighted_factors = factor_model.merge(
        identifier_weights,
        on='identifier',
        how='left'
    )
    weighted_factors['target_weight'] = weighted_factors['target_weight'].fillna(0.0)
    
    # For each factor column, multiply by weight and sum
    factor_cols = [col for col in factor_model.columns if col != 'identifier']
    weighted_sums = {}
    for col in factor_cols:
        weighted_sums[col] = (weighted_factors[col] * weighted_factors['target_weight']).sum()
    
    # Create DataFrame with weighted averages for targets
    weighted_average_factors = pd.DataFrame([weighted_sums])

    # Compute weighted average factors for actuals
    weighted_actual_factors = factor_model.merge(
        actuals[['identifier', 'actual_weight']], 
        on='identifier',
        how='left'
    )
    weighted_actual_factors['actual_weight'] = weighted_actual_factors['actual_weight'].fillna(0.0)
    
    # For each factor column, multiply by actual weight and sum
    actual_weighted_sums = {}
    for col in factor_cols:
        actual_weighted_sums[col] = (weighted_actual_factors[col] * weighted_actual_factors['actual_weight']).sum()
    
    # Create DataFrame with weighted averages for actuals
    weighted_average_actuals_factors = pd.DataFrame([actual_weighted_sums])

    # Check for missing actual identifiers and add them with weighted average factors
    all_actual_identifiers = set(actuals['identifier'])
    missing_actual_identifiers = all_actual_identifiers - set(factor_model['identifier'])
    
    if missing_actual_identifiers:
        # Create rows for missing identifiers using weighted average factors
        missing_rows = []
        for identifier in missing_actual_identifiers:
            new_row = {'identifier': identifier}
            # Add weighted average factors for each factor column
            for col in factor_cols:
                new_row[col] = weighted_average_factors[col].iloc[0]
            missing_rows.append(new_row)
        
        # Add the missing rows to the factor model
        if missing_rows:
            missing_df = pd.DataFrame(missing_rows)
            factor_model = pd.concat([factor_model, missing_df], ignore_index=True)
            logger.info(f"Added {len(missing_actual_identifiers)} missing identifiers to factor model with weighted average factors")
    
    return factor_model, weighted_average_factors, weighted_average_actuals_factors
