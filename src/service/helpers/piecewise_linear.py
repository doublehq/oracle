import pulp
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import logging

from src.service.helpers.constants import logger

def create_piecewise_linear_variables(
    prob: pulp.LpProblem,
    x: float,
    variable_name: str,
    breakpoints: List[float],
    values: List[float]
) -> Tuple[pulp.LpVariable, List[pulp.LpVariable]]:
    """
    Create variables for piecewise linear approximation using the convex combination method.
    For values beyond the upper breakpoint, extends linearly based on the slope at the endpoint.
    
    Args:
        prob: PuLP problem to add constraints to
        x: The value to approximate
        variable_name: Base name for variables
        breakpoints: List of x-coordinates for breakpoints
        values: List of y-coordinates for breakpoints
        
    Returns:
        Tuple of (y, lambda_vars) where:
        - y is the approximated output value
        - lambda_vars are the convex combination coefficients
    """
    n = len(breakpoints)
    
    # Calculate slope at right endpoint for extrapolation
    right_slope = (values[-1] - values[-2]) / (breakpoints[-1] - breakpoints[-2])
    
    # Create variable for the linear portion beyond upper breakpoint
    right_excess = pulp.LpVariable(f"right_excess_{variable_name}", 0)
    
    # Create lambda variables for convex combination within breakpoint range
    lambda_vars = [
        pulp.LpVariable(f"lambda_{variable_name}_{i}", 0, 1)
        for i in range(n)
    ]
    
    # Convex combination constraint
    prob += pulp.lpSum(lambda_vars) <= 1, f"sum_lambda_{variable_name}"
    
    # Split x into within-range and excess components
    prob += (pulp.lpSum(b * l for b, l in zip(breakpoints, lambda_vars)) + 
            breakpoints[-1] * right_excess) == x, f"x_conv_{variable_name}"
    
    # Calculate y as combination of:
    # 1. Convex combination within breakpoint range
    # 2. Linear extrapolation above last breakpoint
    y = (pulp.lpSum(v * l for v, l in zip(values, lambda_vars)) +
         (values[-1] + right_slope * (breakpoints[-1] - breakpoints[-1])) * right_excess)
    
    # SOS2 constraint - at most two adjacent lambda variables can be non-zero
    # This is handled implicitly by the solver in most cases
    return y, lambda_vars

def get_piecewise_breakpoints() -> Tuple[List[float], List[float]]:
    """
    Get the breakpoints for piecewise linear approximation.
    Uses 7 exponentially spaced points between 0 and 0.10,
    followed by linear spacing to 1.0.
    The y values use x^1.5 to create a less aggressive exponential relationship
    compared to the previous x^2.
    
    Returns:
        Tuple of (x_points, y_points) for the piecewise linear function
    """
    # Generate exponential points from 0 to 0.10
    # exp_x = list(np.geomspace(0.001, 1, 7))  # Start at small non-zero for geomspace
    # exp_y = [round(x**1.5, 8) for x in exp_x]  # Using x^1.5 instead of x^2 to reduce exponential growth
    # exp_x = [x/10 for x in exp_x]  # Round again to ensure precision
    # exp_y = [y/10 for y in exp_y]  # Round again to ensure precision
    # exp_x = [round(x, 8) for x in exp_x]  # Round x values for stability
    # exp_y = [round(y, 8) for y in exp_y]  # Round y values for stability
    # exp_x = [0.0] + exp_x + [1.0] # Add 0 and 1 manually since geomspace can't handle it
    # exp_y = [0.0] + exp_y + [1.0] # Add 0 and 1 manually since geomspace can't handle it
    
    # # Combine points
    # x_points = exp_x
    # y_points = exp_y
    # Define piecewise linear breakpoints and their corresponding y-values
    # Each point represents (x, y) where:
    # - x is the deviation amount (0.0 to 0.10)
    # - y is the penalty value, with increasing severity for larger deviations
    data = [
        (0.0, 0.0),              # No deviation = no penalty
        (0.0001, 0.0001/1000),   # Tiny deviation (<0.01%) gets minimal penalty
        (0.001, 0.001/100),      # Small deviation (0.1%) gets very small penalty
        (0.005, 0.005/25),       # Minor deviation (0.5%) starts increasing penalty
        (0.01, 0.01/10),         # Moderate deviation (1%) has noticeable penalty
        (0.05, 0.01/2.5),         # Moderate deviation (1%) has noticeable penalty
        (0.10, 0.10),            # Maximum deviation (10%) has full linear penalty
    ]

    # data = [
    #     (0.0, 0.0),
    #     (0.0001, 0.000001),  # Enough to be zero
    #     (0.008891706, 0.0000977496),  # Early curve
    #     (0.024812704, 0.007078037),  # Start of steeper increase
    #     (0.044597184, 0.019923271),  # Mid-low range
    #     (0.08249175, 0.051704922),   # Acceleration point
    #     (0.12, 0.12),                # Key inflection point
    # ]
    x_points, y_points = zip(*data)

    # Log the breakpoint ranges
    # logger.info(f"Piecewise breakpoints range: [{min(x_points)}, {max(x_points)}]")
    
    return list(x_points), list(y_points)

def create_piecewise_deviation_variable(
    prob: pulp.LpProblem,
    deviation: pulp.LpAffineExpression,
    variable_name: str,
    normalization: float = 1.0
) -> pulp.LpVariable:
    """
    Create a piecewise linear approximation for a deviation variable.
    This is used for both drift and factor model deviations.
    
    Args:
        prob: PuLP problem to add constraints to
        deviation: The deviation expression to approximate
        variable_name: Base name for variables
        normalization: Normalization factor to apply to the output
        
    Returns:
        The approximated deviation variable
    """
    # Get breakpoints for the piecewise approximation
    x_points, y_points = get_piecewise_breakpoints()
    
    # Create positive and negative deviation variables
    pos_dev = pulp.LpVariable(f"pos_dev_{variable_name}", 0)
    neg_dev = pulp.LpVariable(f"neg_dev_{variable_name}", 0)
    
    # Deviation = pos_dev - neg_dev
    prob += deviation == pos_dev - neg_dev, f"dev_split_{variable_name}"
    
    # Create piecewise approximation for positive deviation
    pos_impact, _ = create_piecewise_linear_variables(
        prob, pos_dev, f"pos_{variable_name}",
        x_points, [y * normalization for y in y_points]
    )
    
    # Create piecewise approximation for negative deviation
    neg_impact, _ = create_piecewise_linear_variables(
        prob, neg_dev, f"neg_{variable_name}",
        x_points, [y * normalization for y in y_points]
    )
    
    # Total impact is sum of positive and negative impacts
    return pos_impact + neg_impact 