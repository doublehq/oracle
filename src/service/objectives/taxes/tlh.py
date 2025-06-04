import pulp
import pandas as pd
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from src.service.constraints.restriction.wash_sale_restrictions import WashSaleRestrictions
from src.service.constraints.constraints_manager import ConstraintsManager
from src.service.helpers.enums import OracleOptimizationType
from src.service.reports.drift_report import generate_drift_report, PositionStatus
from src.service.helpers.constants import CASH_CUSIP_ID

@dataclass
class TLHTrade:
    """Represents a tax loss harvesting opportunity for a specific lot"""
    tax_lot_id: str
    identifier: str
    quantity: float
    cost_basis: float
    current_value: float
    loss_percentage: float
    potential_tax_savings: float
    target_weight: float  # Target weight for the position
    current_weight: float  # Current weight of the position
    harvest_quantity: float  # Quantity to harvest based on sizing function
    lot_priority: float  # Priority score for harvesting this lot
    replacement_buys: Dict[str, float] = None  # Added field for replacement securities and quantities

def _calculate_harvest_quantities(
    lots: pd.DataFrame,
    current_weight: float,
    target_weight: float,
    min_weight_multiplier: float,
    max_weight_multiplier: float,
    total_portfolio_value: float,
    prices: pd.DataFrame,
    min_notional: float,
    trade_rounding: float,
    min_harvest_basis_points: float = 50.0,
    soft_limit_percentage: float = 0.90  # New parameter for soft limit (90% by default)
) -> Dict[str, float]:
    """
    Calculate harvest quantities for a group of lots of the same security,
    ensuring we stay within weight bounds.
    
    Args:
        lots: DataFrame of lots for a single security, sorted by priority
        current_weight: Current weight of the security
        target_weight: Target weight for the security
        min_weight_multiplier: Minimum weight multiplier
        max_weight_multiplier: Maximum weight multiplier
        total_portfolio_value: Total portfolio value
        prices: DataFrame with security prices
        min_notional: Minimum notional value for a trade in dollars
        trade_rounding: Rounding increment for trade quantities
        min_harvest_basis_points: Minimum harvest size in basis points (default 50 = 0.5%)
        soft_limit_percentage: Percentage of the way to hard limit for soft limit (default 90%)
        
    Returns:
        Dict mapping tax_lot_id to harvest quantity
    """
    # If no target weight is set, we can't calculate min/max allowed weights
    if target_weight == 0.0:
        return {}
        
    # Calculate hard minimum allowed weight
    hard_min_allowed_weight = target_weight * min_weight_multiplier
    
    # Calculate soft minimum allowed weight (90% of the way to hard minimum)
    # Example: If target=10%, hard_min=5%, then soft_min=5.5%
    # Because 10% - (90% * (10% - 5%)) = 5.5%
    weight_delta = target_weight - hard_min_allowed_weight
    soft_min_allowed_weight = target_weight - (soft_limit_percentage * weight_delta)
    
    max_allowed_weight = target_weight * max_weight_multiplier
    
    # If we're already below soft minimum weight, don't harvest anything
    if current_weight <= soft_min_allowed_weight:
        return {}
        
    # Calculate how much value we can harvest while staying above soft minimum weight
    max_harvest_value = (current_weight - soft_min_allowed_weight) * total_portfolio_value
    
    # Calculate minimum harvest value (50 bps = 0.5% of current position value)
    current_position_value = current_weight * total_portfolio_value
    min_harvest_value = max(
        (min_harvest_basis_points / 10000.0) * current_position_value,
        min_notional  # Ensure we meet minimum notional requirement
    )
    
    # If max possible harvest is less than minimum size, don't harvest
    if max_harvest_value < min_harvest_value:
        return {}
    
    # Allocate harvesting across lots in priority order
    harvest_quantities = {}
    remaining_harvest_value = max_harvest_value
    total_harvested_value = 0.0  # Track total value harvested
    
    #Lots have already been but into order but lets make sure. 
    lots = lots.sort_values('priority', ascending=True)
    for _, lot in lots.iterrows():
        if lot['identifier'] == "EYPT_30233G209":
            print("ahhhh")
        if remaining_harvest_value <= 0:
            break
            
        price = prices[prices['identifier'] == lot['identifier']]['price'].iloc[0]
        lot_value = lot['quantity'] * price
        
        # Harvest the minimum of:
        # 1. The entire lot
        # 2. The remaining harvest value we can do
        harvest_value = min(lot_value, remaining_harvest_value)
        harvest_qty = harvest_value / price
        
        # If we're harvesting the entire lot, don't round
        if harvest_value >= lot_value:
            harvest_qty = lot['quantity']
        else:
            # For partial harvests, floor to the specified increment
            harvest_qty = (harvest_qty // trade_rounding) * trade_rounding
        
        # Recalculate harvest value after potential rounding
        harvest_value = harvest_qty * price
        
        # Check if this harvest would violate the hard minimum weight limit
        new_position_value = current_position_value - harvest_value
        new_weight = new_position_value / total_portfolio_value
        
        if new_weight < hard_min_allowed_weight:
            # Skip this lot if it would violate hard limit
            continue
        
        if harvest_qty > 0:
            # Check if this lot would meet minimum notional requirement
            if harvest_value >= min_notional:
                # Only proceed if this would make total harvest at least min_harvest_value
                if total_harvested_value + harvest_value >= min_harvest_value:
                    harvest_quantities[lot['tax_lot_id']] = harvest_qty
                    remaining_harvest_value -= harvest_value
                    total_harvested_value += harvest_value
                else:
                    # If adding this lot wouldn't reach minimum size, skip it
                    continue
            
    # If total harvest is less than minimum size, don't harvest anything
    if total_harvested_value < min_harvest_value:
        return {}
            
    return harvest_quantities

def _identify_direct_index_tlh_opportunities(
    gain_loss_report: pd.DataFrame,
    prices: pd.DataFrame,
    tax_rates: pd.DataFrame,
    constraints_manager,
    target_weights: pd.DataFrame,
    total_portfolio_value: float,
    min_weight_multiplier: float,
    max_weight_multiplier: float,
    min_notional: float,
    trade_rounding: float,
    min_loss_threshold: float = 0.015
) -> List[TLHTrade]:
    """
    Identify tax loss harvesting opportunities for Direct Indexing strategy.
    Uses asset class target weights to determine harvesting opportunities.
    
    Args:
        gain_loss_report: DataFrame with current gain/loss information for all lots
        prices: DataFrame with current security prices
        tax_rates: DataFrame with tax rates for different gain types
        constraints_manager: ConstraintsManager instance to check restrictions
        target_weights: DataFrame with target weights for each security
        total_portfolio_value: Total portfolio value for weight calculations
        min_weight_multiplier: Minimum weight multiplier from strategy type
        max_weight_multiplier: Maximum weight multiplier from strategy type
        min_notional: Minimum notional value for a trade in dollars
        trade_rounding: Rounding increment for trade quantities
        min_loss_threshold: Minimum loss percentage to consider for TLH (default 1.5%)
        
    Returns:
        List of TLHTrade objects representing potential TLH opportunities
    """
    tlh_opportunities = []
    
    # Calculate current weights by security
    grouped_lots = gain_loss_report.groupby('identifier').agg({
        'quantity': 'sum',
        'cost_basis': 'sum'
    }).reset_index().merge(prices)
    
    grouped_lots["current_value"] = grouped_lots["quantity"] * grouped_lots["price"]
    grouped_lots['current_weight'] = grouped_lots['current_value'] / total_portfolio_value
    
    # Process each security's lots
    for identifier in grouped_lots['identifier'].unique():
        # Get security-level info
        current_weight = 0.0
        target_weight = 0.0
        
        # Safely get current weight
        current_weight_series = grouped_lots[grouped_lots['identifier'] == identifier]['current_weight']
        if not current_weight_series.empty:
            current_weight = current_weight_series.iloc[0]
        
        if current_weight == 0.0:
            # No current weight, skip this security
            continue
            
        # Get target weight from asset class
        for _, row in target_weights.iterrows():
            if identifier in row['identifiers']:
                target_weight = row['target_weight']
                break
        
        # Get all lots for this security
        security_lots = gain_loss_report[gain_loss_report['identifier'] == identifier].copy()
        
        # Skip lots that can't be sold due to restrictions using ConstraintsManager
        security_lots = security_lots[
            ~security_lots['tax_lot_id'].apply(
                lambda x: constraints_manager.is_restricted_from_selling_pre_trade(identifier, x)[0]
            )
        ]
        
        if len(security_lots) == 0:
            continue
            
        # Calculate loss percentage and priority for each lot
        security_lots['current_value'] = security_lots['quantity'] * prices[
            prices['identifier'] == identifier
        ]['price'].iloc[0]
        security_lots = security_lots[security_lots['tax_gain_loss_percentage'] < -abs(min_loss_threshold)]
        if len(security_lots) == 0:
            continue
            
        # Rank lots by per_share_tax_liability (smaller values get higher priority)
        security_lots['priority'] = security_lots['per_share_tax_liability'].rank(ascending=True)
        
        # Sort by priority (highest first) and calculate harvest quantities
        security_lots = security_lots.sort_values('priority', ascending=True)
        if len(security_lots) == 0:
            continue
            
        harvest_quantities = _calculate_harvest_quantities(
            lots=security_lots,
            current_weight=current_weight,
            target_weight=target_weight,
            min_weight_multiplier=min_weight_multiplier,
            max_weight_multiplier=max_weight_multiplier,
            total_portfolio_value=total_portfolio_value,
            prices=prices,
            min_notional=min_notional,
            trade_rounding=trade_rounding
        )
        
        # Create TLHTrade objects for lots we want to harvest
        for _, lot in security_lots.iterrows():
            harvest_qty = harvest_quantities.get(lot['tax_lot_id'], 0.0)
            if harvest_qty > 0:
                tax_rate = tax_rates[tax_rates['gain_type'] == lot['gain_type']]['total_rate'].iloc[0]
                potential_savings = (lot['cost_basis'] - lot['current_value']) * tax_rate
                
                tlh_opportunities.append(TLHTrade(
                    tax_lot_id=lot['tax_lot_id'],
                    identifier=lot['identifier'],
                    quantity=lot['quantity'],
                    cost_basis=lot['cost_basis'],
                    current_value=lot['current_value'],
                    loss_percentage=lot['tax_gain_loss_percentage'],
                    potential_tax_savings=potential_savings,
                    target_weight=target_weight,
                    current_weight=current_weight,
                    harvest_quantity=harvest_qty,
                    lot_priority=lot['priority']
                ))
    
    return tlh_opportunities

def _identify_pairs_tlh_opportunities(
    drift_report: pd.DataFrame,
    gain_loss_report: pd.DataFrame,
    prices: pd.DataFrame,
    tax_rates: pd.DataFrame,
    constraints_manager,
    target_weights: pd.DataFrame,
    total_portfolio_value: float,
    min_notional: float,
    trade_rounding: float,
    min_loss_threshold: float = 0.015,
    min_weight_multiplier: float = 0.5,  # Added parameter for minimum weight multiplier
    soft_limit_percentage: float = 0.90  # New parameter for soft limit
) -> List[TLHTrade]:
    """
    Identify tax loss harvesting opportunities for Pairs TLH strategy.
    For each asset class, finds the security with the best tax benefit that can be sold
    and pairs it with a valid replacement security from the same asset class.
    
    Args:
        gain_loss_report: DataFrame with current gain/loss information for all lots
        prices: DataFrame with current security prices
        tax_rates: DataFrame with tax rates for different gain types
        constraints_manager: ConstraintsManager instance to check restrictions
        target_weights: DataFrame with target weights for each security
        total_portfolio_value: Total portfolio value for weight calculations
        min_notional: Minimum notional value for a trade in dollars
        trade_rounding: Rounding increment for trade quantities
        min_loss_threshold: Minimum loss percentage to consider for TLH (default 1.5%)
        min_weight_multiplier: Minimum weight multiplier (default 0.5)
        soft_limit_percentage: Percentage of the way to hard limit for soft limit (default 90%)
        
    Returns:
        List of TLHTrade objects representing potential TLH opportunities with replacement buys
    """
    tlh_opportunities = []
    
    # Calculate current weights by security
    grouped_lots = gain_loss_report.groupby('identifier').agg({
        'quantity': 'sum',
        'cost_basis': 'sum'
    }).reset_index().merge(prices)
    
    grouped_lots["current_value"] = grouped_lots["quantity"] * grouped_lots["price"]
    grouped_lots['current_weight'] = grouped_lots['current_value'] / total_portfolio_value
    
    # Process each asset class
    for _, asset_class in target_weights.iterrows():
        if asset_class['asset_class'] == CASH_CUSIP_ID:
            continue
            
        identifiers = asset_class['identifiers']
        best_sell_identifier = None
        best_tax_benefit = 0
        best_lots = None
        
        # Find the security with the best tax benefit that we can sell
        for identifier in identifiers:
            if identifier == "J_46982L108":
                print("ahh")
            # Get all lots for this security
            security_lots = gain_loss_report[gain_loss_report['identifier'] == identifier].copy()
            if len(security_lots) == 0:
                continue
                
            # Check if we can sell any lots
            can_sell_any = False
            for _, lot in security_lots.iterrows():
                is_restricted, _ = constraints_manager.is_restricted_from_selling_pre_trade(identifier, lot['tax_lot_id'])
                if not is_restricted:
                    can_sell_any = True
                    break
                    
            if not can_sell_any:
                continue
            
            # Filter to lots with sufficient loss
            loss_lots = security_lots[security_lots['tax_gain_loss_percentage'] < -abs(min_loss_threshold)].copy()
            if len(loss_lots) == 0:
                continue
                
            # Calculate total potential tax benefit for this security
            total_tax_benefit = 0
            for _, lot in loss_lots.iterrows():
                is_restricted, _ = constraints_manager.is_restricted_from_selling_pre_trade(identifier, lot['tax_lot_id'])
                if not is_restricted:
                    total_tax_benefit += abs(lot['tax_liability'])
            
            if total_tax_benefit > best_tax_benefit:
                best_tax_benefit = total_tax_benefit
                best_sell_identifier = identifier
                best_lots = loss_lots
                
        if best_sell_identifier is None or best_lots is None:
            continue
            
        # Find a valid replacement security from the same asset class
        replacement_security = None
        for potential_replacement in identifiers:
            if potential_replacement != best_sell_identifier:
                is_restricted, _ = constraints_manager.is_restricted_from_buying_pre_trade(potential_replacement)
                if not is_restricted:
                    replacement_security = potential_replacement
                    break
                    
        if replacement_security is None:
            continue
            
        # Rank lots by per_share_tax_liability (smaller values get higher priority)
        best_lots['priority'] = best_lots['per_share_tax_liability'].rank(ascending=True)
        best_lots = best_lots.sort_values('priority', ascending=True)

        # Calculate target weight for each security in the asset class
        target_weight = asset_class['target_weight'] / len(identifiers)
        
        # Calculate hard and soft minimum weights
        hard_min_allowed_weight = target_weight * min_weight_multiplier
        weight_delta = target_weight - hard_min_allowed_weight
        soft_min_allowed_weight = target_weight - (soft_limit_percentage * weight_delta)

        # Calculate current weight
        current_weight = grouped_lots[grouped_lots['identifier'] == best_sell_identifier]['current_weight'].iloc[0]
        
        # If already below soft minimum, skip this opportunity
        if current_weight <= soft_min_allowed_weight:
            continue

        # Calculate how much value we can harvest while staying above soft minimum
        max_harvest_value = (current_weight - soft_min_allowed_weight) * total_portfolio_value
        
        # Calculate harvest quantities respecting both soft and hard limits
        total_harvest_value = 0.0
        harvest_quantities = {}
        
        # Calculate minimum harvest value (50 bps = 0.5% of current position value)
        current_position_value = best_lots['market_value'].sum()
        min_harvest_value = max(
            (50.0 / 10000.0) * current_position_value,  # 50 bps = 0.5%
            min_notional  # Ensure we meet minimum notional requirement
        )

        # Process each lot in priority order
        for _, lot in best_lots.iterrows():
            price = prices[prices['identifier'] == lot['identifier']]['price'].iloc[0]
            lot_value = lot['quantity'] * price
            
            # Calculate maximum harvestable value for this lot
            harvest_value = min(lot_value, max_harvest_value - total_harvest_value)
            
            # Calculate harvest quantity
            harvest_qty = (harvest_value / price) // trade_rounding * trade_rounding
            actual_harvest_value = harvest_qty * price
            
            # Check if this harvest would violate the hard minimum weight limit
            new_position_value = current_position_value - actual_harvest_value
            new_weight = new_position_value / total_portfolio_value
            
            if new_weight < hard_min_allowed_weight:
                # Skip this lot if it would violate hard limit
                continue
            
            # Only include if it meets minimum notional
            if actual_harvest_value >= min_notional:
                harvest_quantities[lot['tax_lot_id']] = harvest_qty
                total_harvest_value += actual_harvest_value

        # If total harvest value doesn't meet minimum, skip this opportunity
        if total_harvest_value < min_harvest_value:
            continue

        # Calculate buy quantity for the replacement security
        replacement_price = prices[prices['identifier'] == replacement_security]['price'].iloc[0]
        buy_qty = (total_harvest_value / replacement_price) // trade_rounding * trade_rounding
        
        # Only proceed if replacement buy meets minimum notional
        if buy_qty * replacement_price < min_notional:
            continue

        buy_quantities = {replacement_security: buy_qty}
            
        # Create TLHTrade objects for lots we want to harvest
        for _, lot in best_lots.iterrows():
            harvest_qty = harvest_quantities.get(lot['tax_lot_id'], 0.0)
            if harvest_qty > 0:
                tax_rate = tax_rates[tax_rates['gain_type'] == lot['gain_type']]['total_rate'].iloc[0]
                potential_savings = abs(lot['tax_liability'])
                
                tlh_opportunities.append(TLHTrade(
                    tax_lot_id=lot['tax_lot_id'],
                    identifier=lot['identifier'],
                    quantity=lot['quantity'],
                    cost_basis=lot['cost_basis'],
                    current_value=lot['market_value'],
                    loss_percentage=lot['tax_gain_loss_percentage'],
                    potential_tax_savings=potential_savings,
                    target_weight=target_weight,
                    current_weight=current_weight,
                    harvest_quantity=harvest_qty,
                    lot_priority=lot['priority'],
                    replacement_buys=buy_quantities
                ))
    
    return tlh_opportunities

def _identify_tlh_opportunities(
    drift_report: pd.DataFrame,
    gain_loss_report: pd.DataFrame,
    prices: pd.DataFrame,
    tax_rates: pd.DataFrame,
    constraints_manager,
    target_weights: pd.DataFrame,
    total_portfolio_value: float,
    min_weight_multiplier: float,
    max_weight_multiplier: float,
    min_notional: float,
    trade_rounding: float,
    min_loss_threshold: float = 0.015,
    optimization_type: Optional[OracleOptimizationType] = None
) -> List[TLHTrade]:
    """
    Helper function to identify tax loss harvesting opportunities.
    Routes to the appropriate TLH identification function based on optimization type.
    
    Args:
        gain_loss_report: DataFrame with current gain/loss information for all lots
        prices: DataFrame with current security prices
        tax_rates: DataFrame with tax rates for different gain types
        constraints_manager: ConstraintsManager instance to check restrictions
        target_weights: DataFrame with target weights for each security
        total_portfolio_value: Total portfolio value for weight calculations
        min_weight_multiplier: Minimum weight multiplier from strategy type
        max_weight_multiplier: Maximum weight multiplier from strategy type
        min_notional: Minimum notional value for a trade in dollars
        trade_rounding: Rounding increment for trade quantities
        min_loss_threshold: Minimum loss percentage to consider for TLH (default 1.5%)
        optimization_type: The optimization type (OracleOptimizationType)
        
    Returns:
        List of TLHTrade objects representing potential TLH opportunities
    """
    if optimization_type == OracleOptimizationType.PAIRS_TLH:
        return _identify_pairs_tlh_opportunities(
            drift_report=drift_report,
            gain_loss_report=gain_loss_report,
            prices=prices,
            tax_rates=tax_rates,
            constraints_manager=constraints_manager,
            target_weights=target_weights,
            total_portfolio_value=total_portfolio_value,
            min_notional=min_notional,
            trade_rounding=trade_rounding,
            min_loss_threshold=min_loss_threshold
        )
    elif optimization_type == OracleOptimizationType.DIRECT_INDEX:
        return _identify_direct_index_tlh_opportunities(
            gain_loss_report=gain_loss_report,
            prices=prices,
            tax_rates=tax_rates,
            constraints_manager=constraints_manager,
            target_weights=target_weights,
            total_portfolio_value=total_portfolio_value,
            min_weight_multiplier=min_weight_multiplier,
            max_weight_multiplier=max_weight_multiplier,
            min_notional=min_notional,
            trade_rounding=trade_rounding,
            min_loss_threshold=min_loss_threshold
        )
    else:
        return []

def calculate_tlh_impact(
    prob: pulp.LpProblem,
    buys: Dict[str, pulp.LpVariable],
    sells: Dict[str, pulp.LpVariable],
    drift_report: pd.DataFrame,
    gain_loss_report: pd.DataFrame,
    prices: pd.DataFrame,
    tax_rates: pd.DataFrame,
    constraints_manager: ConstraintsManager,
    target_weights: pd.DataFrame,
    total_portfolio_value: float,
    min_weight_multiplier: float,
    max_weight_multiplier: float,
    min_notional: float,
    trade_rounding: float,
    min_loss_threshold: float = 0.015,
    objective_manager = None,  # Add ObjectiveManager parameter
    optimization_type: Optional[OracleOptimizationType] = None  # Updated type hint
) -> Tuple[List[TLHTrade], Dict[str, float], Dict[str, float], Optional[Dict]]:  # Updated return type
    """
    Identify tax loss harvesting opportunities and determine sell quantities.
    Also calculates baseline objective values after TLH trades.
    For pairs trading, identifies and constrains both sells and replacement buys.
    
    Args:
        prob: The optimization problem
        buys: Dictionary of buy variables
        sells: Dictionary of sell variables
        gain_loss_report: DataFrame with current gain/loss information for all lots
        prices: DataFrame with current security prices
        tax_rates: DataFrame with tax rates for different gain types
        constraints_manager: ConstraintsManager instance to check restrictions
        target_weights: DataFrame with target weights for each security
        total_portfolio_value: Total portfolio value for weight calculations
        min_weight_multiplier: Minimum weight multiplier from strategy type
        max_weight_multiplier: Maximum weight multiplier from strategy type
        min_notional: Minimum notional value for a trade in dollars
        trade_rounding: Rounding increment for trade quantities
        min_loss_threshold: Minimum loss percentage to consider for TLH (default 1.5%)
        objective_manager: ObjectiveManager instance to calculate objective values
        optimization_type: The optimization type (OracleOptimizationType)
        
    Returns:
        Tuple of:
        - List of TLHTrade objects representing identified opportunities
        - Dictionary mapping tax_lot_id to sell quantities
        - Dictionary mapping identifier to buy quantities (for pairs trading)
        - Dictionary of baseline objective component values after TLH trades (or None if no TLH trades)
    """
    # Identify TLH opportunities with quantities
    tlh_opportunities = _identify_tlh_opportunities(
        drift_report=drift_report,
        gain_loss_report=gain_loss_report,
        prices=prices,
        tax_rates=tax_rates,
        constraints_manager=constraints_manager,
        target_weights=target_weights,
        total_portfolio_value=total_portfolio_value,
        min_weight_multiplier=min_weight_multiplier,
        max_weight_multiplier=max_weight_multiplier,
        min_notional=min_notional,
        trade_rounding=trade_rounding,
        min_loss_threshold=min_loss_threshold,
        optimization_type=optimization_type
    )
    
    # Create dictionaries of sell and buy quantities
    sell_quantities = {}
    buy_quantities = {}

    sell_identifiers = []
    sold_identifiers = []
    
    for opportunity in tlh_opportunities:
        if opportunity.tax_lot_id in sells:
            # Add constraint to sell exactly the harvest quantity
            if opportunity.harvest_quantity > opportunity.quantity:
                raise ValueError("TLH Harvest quantity exceeds available quantity.")
            prob += sells[opportunity.tax_lot_id] == opportunity.harvest_quantity, f"tlh_sell_{opportunity.tax_lot_id}"
            sell_identifiers.append(opportunity.identifier)
            sell_quantities[opportunity.tax_lot_id] = opportunity.harvest_quantity
            # When selling a security, prevent buying it at the same time
            if opportunity.identifier in buys and opportunity.identifier not in sold_identifiers:
                prob += buys[opportunity.identifier] == 0, f"tlh_buy_{opportunity.identifier}"
                sold_identifiers.append(opportunity.identifier)
            
            # For pairs trading, add buy constraints for replacement securities
            if optimization_type == OracleOptimizationType.PAIRS_TLH and opportunity.replacement_buys:
                for replacement_id, buy_qty in opportunity.replacement_buys.items():
                    if replacement_id in buys: # and replacement_id not in sell_identifiers:
                        prob += buys[replacement_id] == buy_qty, f"tlh_buy_{replacement_id}"
                        buy_quantities[replacement_id] = buy_qty
                        all_sell_lots = gain_loss_report[gain_loss_report['identifier'] == replacement_id]
                        for _, lot in all_sell_lots.iterrows():
                            prob += sells[lot['tax_lot_id']] == 0, f"tlh_sell_{lot['tax_lot_id']}"
    
    # If we have TLH trades and an objective manager, calculate baseline values
    baseline_components = None
    if (sell_quantities or buy_quantities) and objective_manager:
        # Extract component values
        baseline_components = objective_manager.extract_component_values(prob)
    
    return tlh_opportunities, sell_quantities, buy_quantities, baseline_components