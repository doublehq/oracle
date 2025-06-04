import pulp
import copy
import pandas as pd
from typing import Dict, Optional, Tuple, List, TYPE_CHECKING
from datetime import datetime

from src.service.helpers.constants import CASH_CUSIP_ID
from src.service.helpers.enums import OracleOptimizationType
from src.service.objectives.taxes.tlh import TLHTrade
from src.service.helpers.constants import logger


def _calculate_total_cash_generated(sells, gain_loss, prices):
    """
    Helper function to calculate the total cash generated from sells.
    
    Args:
        sells: Dictionary of sell variables with their values
        gain_loss: DataFrame with gain/loss information for tax lots
        prices: DataFrame with security prices
        
    Returns:
        Total cash value generated from selling securities
    """
    total_cash = 0
    for _, lot in gain_loss.iterrows():
        tax_lot_id = lot['tax_lot_id']
        identifier = lot['identifier']
        price = prices.loc[prices['identifier'] == identifier, 'price'].iloc[0]
        if tax_lot_id in sells and pulp.value(sells[tax_lot_id]) is not None:
            total_cash += pulp.value(sells[tax_lot_id]) * price
    return total_cash


def calculate_max_withdrawal(
    strategy,
    debug: bool = False,
    respect_wash_sales: bool = True,
    preserve_targets: bool = True
) -> Tuple[float, Dict, pd.DataFrame]:
    """
    Calculate the maximum possible withdrawal amount from a portfolio.
    
    This function creates a special optimization problem that attempts to
    liquidate as much of the portfolio as possible while respecting constraints
    like wash sale restrictions. It leverages the existing OracleStrategy methods
    for solving optimization problems.
    
    Args:
        strategy: The OracleStrategy instance containing portfolio data
        debug: Whether to print debug information
        respect_wash_sales: Whether to respect wash sale restrictions
        
    Returns:
        Tuple containing:
        - max_withdrawal_amount: The maximum amount that can be withdrawn
        - trade_summary: Summary statistics of the trades
        - trades: DataFrame of trades required to achieve the maximum withdrawal
    """
    # Import OracleStrategy here to avoid circular imports
    from src.service.oracle_strategy import OracleStrategy
    
    # Get the current portfolio state
    total_value = strategy.actuals['market_value'].sum()
    
    if debug:
        logger.info(f"Calculating maximum withdrawal from portfolio with value: ${total_value:.2f}")
    
    # Early exit if portfolio is empty
    if total_value == 0:
        if debug:
            logger.info("Portfolio is empty. Maximum withdrawal is 0.")
        return 0.0, {}, pd.DataFrame()
    liquidation_strategy = copy.deepcopy(strategy)
    
    # Create liquidation targets (all securities at 0%, cash at 100%)
    liquidation_targets = strategy.targets.copy()
    if not preserve_targets:
        liquidation_targets.loc[liquidation_targets['asset_class'] != CASH_CUSIP_ID, 'target_weight'] = 0.0
        liquidation_targets.loc[liquidation_targets['asset_class'] == CASH_CUSIP_ID, 'target_weight'] = 1.0    
        liquidation_strategy.targets = liquidation_targets
        liquidation_strategy.min_cash = 0
    else:
        liquidation_strategy.min_cash = liquidation_strategy.min_cash_amount()
    # Set the oracle reference to get tax rates and other settings
    liquidation_strategy.set_oracle(strategy.oracle)
    
    # Get all necessary data for the new strategy
    gain_loss = liquidation_strategy.gain_loss_report
    drift = liquidation_strategy.drift_report
    
    total_value = liquidation_strategy.total_value()
    
    # Create optimization problem for withdrawal (use minimize with negated objective)
    prob = pulp.LpProblem("Max_Withdrawal", pulp.LpMinimize)
    
    # Set the optimization problem reference
    liquidation_strategy.optimization_problem = prob
    
    # Create decision variables
    buys, sells, buy_df, sell_df = liquidation_strategy._create_decision_variables(liquidation_strategy.target_identifiers, gain_loss, debug)
    
    if debug:
        logger.info("Building Max Withdrawal Optimization Problem")
    
    # Create a cash generation objective using the objective manager
    max_withdrawal_objective = liquidation_strategy.objective_manager.calculate_max_withdrawal_objective(
        prob=prob,
        buys=buys,
        sells=sells,
        gain_loss=gain_loss,
        debug=debug
    )
    
    # Set the withdrawal maximization as the objective function
    prob += max_withdrawal_objective, "Max_Withdrawal_Value"
    
    # Add constraint: No buying of securities (since we're liquidating)
    # Use the constraints manager to add no buy constraints
    liquidation_strategy.constraints_manager.add_no_buy_constraints(
        prob=prob,
        buys=buys,
        exclude_cash=True  # Don't constrain cash buys
    )
    
    # Add constraints using the constraints manager
    liquidation_strategy.constraints_manager.add_constraints(
        prob=prob,
        buys=buys,
        buy_df=buy_df,
        sells=sells,
        sell_df=sell_df,
        prices=liquidation_strategy.prices,
        min_notional=strategy.min_notional,
        min_cash_amount=liquidation_strategy.min_cash,
        gain_loss=gain_loss,
        tax_lots=liquidation_strategy.tax_lots,
        holding_time_delta=None,
        enforce_wash_sale_prevention=strategy.enforce_wash_sale_prevention,
        range_max_weight_multiplier=getattr(strategy, 'range_max_weight_multiplier', None),
        range_min_weight_multiplier=getattr(strategy, 'range_min_weight_multiplier', None),
        log_time=True
    )
    
    # Set initial values for warm start
    liquidation_strategy._set_initial_values(buys, sells, gain_loss, debug)
    
    # Solve the optimization problem
    status, optimized_value = liquidation_strategy._solve_optimization(prob, debug)
    
    # Handle solution status
    if status is None or status != pulp.LpStatusOptimal:
        status_str = pulp.LpStatus[status] if status is not None else 'Failure before/during solve'
        logger.warning(f"Max withdrawal optimization failed (Status: {status_str}).")
        return 0.0, pd.DataFrame()
    
    # Calculate the maximum withdrawal amount by summing the values of all sells
    max_withdrawal = _calculate_total_cash_generated(sells, gain_loss, liquidation_strategy.prices) + liquidation_strategy.cash - liquidation_strategy.min_cash
    
    if debug:
        logger.info(f"Maximum withdrawal amount: ${max_withdrawal:.2f}")
    
    # Use the strategy's trade extraction to get the trades
    from src.service.helpers.trade_extractor import extract_trades
    trades = extract_trades(
        buys=buys,
        sells=sells,
        gain_loss=gain_loss,
        total_value=total_value,
        prices=liquidation_strategy.prices,
        spreads=liquidation_strategy.spreads,
        tlh_trades=[],  # No TLH trades for withdrawal
        tax_normalization=liquidation_strategy.TAX_NORMALIZATION * 0.0,
        transaction_normalization=liquidation_strategy.TRANSACTION_NORMALIZATION * 0.0,
        min_notional=strategy.min_notional,
        trade_rounding=strategy.trade_rounding,
    )
    
    return max_withdrawal, trades
    