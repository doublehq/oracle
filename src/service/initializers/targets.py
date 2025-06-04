import pandas as pd
from src.service.helpers.constants import CASH_CUSIP_ID

NUM_DECIMALS = 6

POSSIBLE_CASH_ASSET_CLASSES = [
    CASH_CUSIP_ID,
    CASH_CUSIP_ID + '_AC',  # Alternate cash identifier
    'CASH',
    'CASH_AC'
]

def _handle_cash_targets(targets: pd.DataFrame, deminimus_cash_target: float, withdraw_target: float) -> pd.DataFrame:
    """
    Handle cash targets in the targets DataFrame.
    
    Args:
        targets: DataFrame with columns ['asset_class', 'target_weight', 'identifiers']
        deminimus_cash_target: Minimum cash target percentage
        withdraw_target: Target cash percentage for withdrawals
        
    Returns:
        Updated targets DataFrame with proper cash handling
    """
    # Check if cash target exists
    cash_exists = CASH_CUSIP_ID in set(targets['asset_class'])
    
    # Error if multiple cash targets
    cash_targets = targets[targets['asset_class'].isin(POSSIBLE_CASH_ASSET_CLASSES)]
    if len(cash_targets) > 1:
        raise ValueError("Multiple cash targets found. Only one cash target is allowed.")
    non_withdraw_cash_target = max(deminimus_cash_target, cash_targets["target_weight"].sum())
    post_withdraw_cash_target = (1 - withdraw_target) * non_withdraw_cash_target
    cash_target = post_withdraw_cash_target + withdraw_target
    
    if not cash_exists:
        # Add CASH asset class with minimum target weight
        cash_row = pd.DataFrame({
            'asset_class': [CASH_CUSIP_ID],
            'target_weight': [cash_target],
            'identifiers': [[CASH_CUSIP_ID]]
        })
        targets = pd.concat([targets, cash_row], ignore_index=True)
        
        # Scale non-cash weights to make total sum to 1
        non_cash_scale_factor = (1 - cash_target) / targets[targets['asset_class'] != CASH_CUSIP_ID]['target_weight'].sum()
        targets.loc[targets['asset_class'] != CASH_CUSIP_ID, 'target_weight'] *= non_cash_scale_factor
        targets['target_weight'] = targets['target_weight'].round(NUM_DECIMALS)
    elif targets[targets['asset_class'] == CASH_CUSIP_ID]['target_weight'].iloc[0] < cash_target:
        # Calculate how much we need to increase cash by
        current_cash_weight = targets[targets['asset_class'] == CASH_CUSIP_ID]['target_weight'].iloc[0]
        
        # Update cash target to minimum
        targets.loc[targets['asset_class'] == CASH_CUSIP_ID, 'target_weight'] = cash_target
        
        # Scale down non-cash weights proportionally to make room for increased cash
        non_cash_mask = targets['asset_class'] != CASH_CUSIP_ID
        non_cash_scale_factor = (1 - cash_target) / (1 - current_cash_weight)
        targets.loc[non_cash_mask, 'target_weight'] *= non_cash_scale_factor
        targets.loc[non_cash_mask, 'target_weight'] = targets.loc[non_cash_mask, 'target_weight'].round(NUM_DECIMALS)
    
    return targets

def initialize_targets(targets: pd.DataFrame, withdraw_target: float = 0.0, deminimus_cash_target: float = 0.000) -> pd.DataFrame:
    """
    Initialize and validate targets DataFrame with asset class structure.
    
    Args:
        targets: DataFrame with columns:
        - asset_class (str): Asset class identifier
        - target_weight (float): Target portfolio weight (0 to 1)
        - identifiers (list[str]): List of valid identifiers for each asset class
        withdraw_target (float): Target cash percentage for withdrawals (0.0 to 1.0, default: 0.0)
        deminimus_cash_target (float): Minimum cash target percentage (0.0 to 1.0, default: 0.0)
        
    Returns:
        Validated targets DataFrame with:
        - Standardized asset class names (CASH_CUSIP_ID for cash)
        - Validated target weights (sum to 1.0)
        - Standardized identifiers (uppercase)
        - Cash target adjusted for withdrawals and minimum requirements
        
    Raises:
        ValueError: If:
        - DataFrame is missing required columns
        - Target weights don't sum to 1.0
        - Target weights are outside [0,1] range
        - Asset classes are duplicated
        - Identifiers are not in list format
    """
    required_columns = {'asset_class', 'target_weight', 'identifiers'}
    if not set(targets.columns).issuperset(required_columns):
        raise ValueError(f"Targets DataFrame missing required columns: {required_columns}")
    
    # Ensure data types
    targets = targets.copy()
    targets['asset_class'] = targets['asset_class'].astype(str)
    targets['target_weight'] = pd.to_numeric(targets['target_weight'], errors='raise')
    targets['target_weight'] = targets['target_weight'].round(NUM_DECIMALS)

    # Standardize CASH identifiers to CASH_CUSIP_ID
    targets.loc[targets['asset_class'].str.upper().isin(POSSIBLE_CASH_ASSET_CLASSES), 'asset_class'] = CASH_CUSIP_ID
    targets['identifiers'] = targets['identifiers'].apply(
        lambda ids: [CASH_CUSIP_ID if str(id_).upper() in POSSIBLE_CASH_ASSET_CLASSES else str(id_).upper() for id_ in ids]
    )
    
    # Validate identifiers list
    if not all(isinstance(ids, list) for ids in targets['identifiers']):
        raise ValueError("All entries in 'identifiers' column must be lists")
    
    # Validate asset classes are unique
    duplicate_asset_classes = targets['asset_class'].duplicated()
    if duplicate_asset_classes.any():
        duplicates = targets.loc[duplicate_asset_classes, 'asset_class'].tolist()
        raise ValueError(f"Found duplicate asset classes: {duplicates}. Asset classes must be unique.")
    
    # Validate identifier lists are not longer than 2 elements
    long_lists = targets['identifiers'].apply(lambda x: len(x) > 2)
    if long_lists.any():
        invalid_asset_classes = targets.loc[long_lists, 'asset_class'].tolist()
        raise ValueError(
            f"Found asset classes with more than 2 identifiers: {invalid_asset_classes}. "
            "Currently only up to 2 identifiers per asset class are supported."
        )
    
    # Validate no empty identifier lists (except for CASH)
    empty_lists = targets[targets['asset_class'] != CASH_CUSIP_ID]['identifiers'].apply(lambda x: len(x) == 0)
    if empty_lists.any():
        empty_identifiers = targets.loc[empty_lists, 'identifiers'].tolist()
        raise ValueError(f"Found empty identifier lists for asset classes: {empty_identifiers}")
    
    # Ensure all identifiers are strings
    targets['identifiers'] = targets['identifiers'].apply(
        lambda ids: [str(id_).upper() for id_ in ids]
    )
    
    # Handle CASH asset class
    targets = _handle_cash_targets(targets, deminimus_cash_target, withdraw_target)

    # Validate weights are between 0 and 1
    if not all((targets['target_weight'] >= 0) & (targets['target_weight'] <= 1)):
        raise ValueError("Target weights must be between 0 and 1")
    
    # Validate weights sum to 1 (within rounding error)
    total_weight = targets['target_weight'].sum()
    if not abs(total_weight - 1.0) < 1e-2:  # Slightly more lenient due to rounding
        raise ValueError(f"Target weights must sum to 1, got {total_weight}")
    
    # Ensure CASH identifiers list contains CASH_CUSIP_ID
    cash_mask = targets['asset_class'] == CASH_CUSIP_ID
    if cash_mask.any():
        cash_identifiers = targets.loc[cash_mask, 'identifiers'].iloc[0]
        if CASH_CUSIP_ID not in cash_identifiers:
            targets.loc[cash_mask, 'identifiers'] = [[CASH_CUSIP_ID]]
    
    return targets