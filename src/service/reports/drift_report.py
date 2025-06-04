import pandas as pd
from enum import Enum
from typing import Tuple

# Threshold for considering a position on target (used for position_status)
DRIFT_THRESHOLD = 0.001

class PositionStatus(str, Enum):
    """Position status relative to target weight."""
    ON_TARGET = 'ON_TARGET'
    OVERWEIGHT = 'OVERWEIGHT'
    UNDERWEIGHT = 'UNDERWEIGHT'
    NON_TARGET_INSTRUMENT = 'NON_TARGET_INSTRUMENT'

def generate_drift_report(
    targets: pd.DataFrame,
    actuals: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate a drift report comparing actual vs target weights at the asset class level.
    
    Args:
        targets: DataFrame with asset class target weights with columns:
            - asset_class: Asset class identifier
            - target_weight: Target portfolio weight for the asset class
            - identifiers: List of valid security identifiers in this asset class
        actuals: DataFrame with actual weights and market values with columns:
            - identifier: Security identifier
            - actual_weight: Current portfolio weight
            - market_value: Current market value
        
    Returns:
        DataFrame with columns:
        - asset_class: Asset class identifier
        - target_weight: Target portfolio weight for the asset class
        - actual_weight: Current portfolio weight for the asset class
        - market_value: Current market value for the asset class
        - drift: Difference between actual and target weights (actual - target)
        - drift_pct: Percentage drift relative to target ((actual - target) / target)
        - drift_dollars: Dollar value of drift (market_value - target_weight * total_value)
        - position_status: PositionStatus enum value (OVERWEIGHT, UNDERWEIGHT, ON_TARGET, or NON_TARGET_INSTRUMENT)
    """
    # Create mapping of identifier to asset class
    id_to_asset_class = {}
    for _, row in targets.iterrows():
        for identifier in row['identifiers']:
            id_to_asset_class[identifier] = row['asset_class']
    
    # Add asset class to actuals
    actuals_with_ac = actuals.copy()
    actuals_with_ac['asset_class'] = actuals_with_ac['identifier'].map(id_to_asset_class)
    
    # Aggregate actuals by asset class
    asset_class_actuals = actuals_with_ac.groupby('asset_class').agg({
        'market_value': 'sum',
        'actual_weight': 'sum'
    }).reset_index()
    
    # Start with all asset classes from targets
    drift = targets[['asset_class', 'target_weight', 'identifiers']].copy()
    
    # Merge with aggregated actuals using full outer join
    drift = drift.merge(
        asset_class_actuals[['asset_class', 'market_value', 'actual_weight']], 
        on='asset_class', 
        how='outer'
    )
    
    # Fill missing values
    drift['actual_weight'] = drift['actual_weight'].fillna(0.0)
    drift['market_value'] = drift['market_value'].fillna(0.0)
    drift['target_weight'] = drift['target_weight'].fillna(0.0)
    
    # Normalize target weights to sum to 1
    total_target_weight = drift['target_weight'].sum()
    if total_target_weight > 0:
        drift['target_weight'] = drift['target_weight'] / total_target_weight
    
    # Calculate total portfolio value
    total_value = drift['market_value'].sum()
    
    # Calculate drift metrics
    drift['drift'] = drift['actual_weight'] - drift['target_weight']
    drift['drift_pct'] = (
        (drift['actual_weight'] - drift['target_weight']) / 
        drift['target_weight'].where(drift['target_weight'] != 0, 1.0)
    )
    
    # Calculate dollar-based drift
    drift['drift_dollars'] = drift['market_value'] - (drift['target_weight'] * total_value)
    
    # Add position status
    drift['position_status'] = PositionStatus.ON_TARGET
    drift.loc[drift['drift'] > DRIFT_THRESHOLD, 'position_status'] = PositionStatus.OVERWEIGHT
    drift.loc[drift['drift'] < -DRIFT_THRESHOLD, 'position_status'] = PositionStatus.UNDERWEIGHT
    drift.loc[drift['target_weight'] == 0, 'position_status'] = PositionStatus.NON_TARGET_INSTRUMENT
    
    # Order columns logically
    column_order = [
        'asset_class',
        'target_weight',
        'actual_weight',
        'market_value',
        'drift',
        'drift_pct',
        'drift_dollars',
        'position_status',
        'identifiers'
    ]
    
    return drift[column_order].sort_values('drift', ascending=False)
