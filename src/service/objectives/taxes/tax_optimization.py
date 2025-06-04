import pulp
import pandas as pd
from typing import Dict

def get_tax_cost(
    quantity: float,
    per_share_tax_liability: float,
    total_value: float,
    tax_normalization: float = 1.0
) -> float:
    """
    Calculate the tax cost component of the objective function.
    
    Args:
        quantity: Quantity of the security to sell
        per_share_tax_liability: Tax liability per share
        total_value: Total portfolio value
        tax_normalization: Normalization factor for tax impact
    Returns:
        The total tax cost as a percentage of portfolio value
    """

    return (quantity * per_share_tax_liability) / total_value * tax_normalization if total_value > 0 else 0

def calculate_tax_impact(
    prob: pulp.LpProblem,
    sells: Dict[str, pulp.LpVariable],
    gain_loss: pd.DataFrame,
    total_value: float,
    tax_normalization: float = 1.0,
    enforce_wash_sale_prevention: bool = True
) -> tuple[pulp.LpAffineExpression, float]:
    """
    Calculate the tax impact component of the objective function and current tax score.
    Uses pre-calculated per_share_tax_liability from gain_loss report.
    Tax impact is expressed as a percentage of portfolio value to be comparable with drift.
    
    For each tax lot, we track:
    - Current tax liability = quantity * per_share_tax_liability
    - New tax liability = (quantity - sells) * per_share_tax_liability
    - The difference represents realized tax cost
    - Negative tax_realized values represent beneficial tax loss harvesting
    
    Args:
        prob: The optimization problem to add constraints to
        sells: Dictionary of sell variables
        gain_loss: DataFrame with gain/loss report
        total_value: Total portfolio value
        tax_normalization: Normalization factor for tax impact
        
    Returns:
        Tuple of:
        - Tax impact expression representing the change in tax liability
        - Current tax score (sum of unrealized gains as percentage of portfolio)
    """
    tax_impacts = []
    current_tax_score = 0
    
    # Calculate tax impact for each lot
    for _, lot in gain_loss.iterrows():
        tax_lot_id = lot['tax_lot_id']
        if tax_lot_id not in sells:
            continue
            
        quantity = lot['quantity']
        per_share_tax = lot['per_share_tax_liability']
        if per_share_tax < 0:
            # When wash sale prevention is enabled, reduce negative tax liability by 1/5th
            # to make tax loss harvesting less attractive. Otherwise ignore negative tax liability
            # to prevent any tax loss harvesting.
            if enforce_wash_sale_prevention:
                per_share_tax = per_share_tax / 5
            else:
                per_share_tax = 0
        if per_share_tax == 0:
            continue
        
        # Calculate current tax liability for this lot
        current_lot_tax = quantity * per_share_tax
        current_tax_score += current_lot_tax / total_value
        
        # Create variable for realized tax (can be negative for tax loss harvesting)
        tax_realized = pulp.LpVariable(f"tax_realized_{tax_lot_id}")
        
        # Constraint: realized tax equals reduction in tax liability
        # new_tax_liability = (quantity - sells[tax_lot_id]) * per_share_tax
        # tax_realized = current_tax_liability - new_tax_liability
        # Scale the constraint by total_value to match units with drift
        prob += tax_realized == sells[tax_lot_id] * per_share_tax / total_value, f"tax_realized_{tax_lot_id}"
        
        # Add to total tax impact (no need to divide by total_value again since constraint is now scaled)
        tax_impacts.append(tax_realized * tax_normalization)
    
    # Sum all tax impacts
    total_tax_impact = pulp.lpSum(tax_impacts)
    
    return total_tax_impact
