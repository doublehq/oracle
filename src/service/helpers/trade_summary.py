import pandas as pd
from typing import Dict, Any, Tuple
from src.service.types import OracleStrategy
from src.service.helpers.enums import OracleOptimizationType
import time
import logging

from src.service.helpers.constants import logger

def generate_explanation_from_context(explanation_context: Dict[str, Any]) -> str:
    """
    Generate a detailed explanation based on the optimization context.
    
    Args:
        explanation_context: Dictionary containing:
            - case_type: Type of case (e.g. 'buy_only_failed', 'optimization_failed')
            - improvements: Dict of improvements with their thresholds
            - optimization_status: Status of optimization
            - additional_info: Any additional context-specific information
            
    Returns:
        A detailed explanation of what happened during optimization
    """
    case_type = explanation_context.get('case_type')
    if not case_type:
        return "No trades were executed due to an unknown issue."
        
    parts = []
    improvements = explanation_context.get('improvements', {})
    
    def format_improvement_message(improvement_info):
        """Helper to format improvement message consistently"""
        value = improvement_info.get('value', 0)
        threshold = improvement_info.get('threshold', 0)
        return f"the improvement of {value:.4f} was less than the required threshold of {threshold:.4f}"
    
    if case_type == 'buy_only_failed':
        if 'rebalance' in improvements:
            parts.append(f"We tried to rebalance, but {format_improvement_message(improvements['rebalance'])}.")
        parts.append("We then tried a buy-only optimization, but it failed to find a feasible solution.")
        
    elif case_type == 'buy_only_below_threshold':
        if 'rebalance' in improvements:
            parts.append(f"We tried to rebalance, but {format_improvement_message(improvements['rebalance'])}.")
        if 'buy_only' in improvements:
            parts.append(f"We tried a buy-only optimization, but {format_improvement_message(improvements['buy_only'])}.")
        
    elif case_type == 'optimization_failed':
        status = explanation_context.get('optimization_status')
        status_str = f" (Status: {status})" if status else ""
        parts.append(f"The optimization problem could not be solved{status_str}. This may be due to conflicting constraints or numerical issues.")
        
    elif case_type == 'hold_strategy':
        parts.append("No trades were executed as this is a HOLD strategy.")
        
    elif case_type == 'empty_portfolio':
        parts.append("No trades were executed as the portfolio is empty and there are no non-cash targets.")
        
    elif case_type == 'no_trade_failed':
        parts.append("The no-trades scenario failed to solve, indicating a problem with the optimization setup.")
        
    elif case_type == 'not_enough_cash_to_buy_only':
        actual_cash = explanation_context.get('additional_info', {}).get('actual_cash', 0)
        min_cash = explanation_context.get('additional_info', {}).get('min_cash', 0)
        
        if 'rebalance' in improvements:
            parts.append(f"We tried to rebalance, but {format_improvement_message(improvements['rebalance'])}.")
        parts.append(f"We then considered a buy-only optimization, but there wasn't enough cash (${actual_cash:,.2f}) to meet the minimum requirement (${min_cash:,.2f}).")
    
    # Add any additional context information
    additional_info = explanation_context.get('additional_info', {})
    if additional_info:
        for key, value in additional_info.items():
            # if key not in ['actual_cash', 'min_cash']:  # Skip already used info
            parts.append(f"{key}: {value}")
    
    return " ".join(parts)

def _generate_explanation(
    trades: pd.DataFrame,
    execution_summary: Dict[str, float],
    before_optimization: Dict[str, Any],
    after_optimization: Dict[str, Any],
    component_improvements: Dict[str, Dict[str, float]],
    is_2nd_buy_only_optimization: bool = False,
    additional_info: Dict[str, Any] = None
) -> str:
    """
    Generate a human-readable explanation of the trades and their impact.
    
    Args:
        trades: DataFrame of trades executed
        execution_summary: Summary of trade execution details
        before_optimization: Pre-optimization metrics
        after_optimization: Post-optimization metrics
        component_improvements: Calculated improvements in various metrics
        is_2nd_buy_only_optimization: Whether this is a second pass buy-only optimization
        
    Returns:
        A string explaining the trades and their impact
    """
    if trades.empty:
        if is_2nd_buy_only_optimization:
            return "No trades were executed as we decided to skip rebalancing and found no beneficial buy opportunities."
        return "No trades were executed as no beneficial trades were identified."

    # Start with trade summary
    explanation_parts = []

    if is_2nd_buy_only_optimization:
        explanation_parts.append("After deciding to skip rebalancing, we identified the following buy opportunities:")

    # Add trade value information
    summary_parts = []
    if execution_summary['num_buys'] > 0:
        summary_parts.append(f"buying ${execution_summary['total_buys_value']:,.2f} worth of {execution_summary['num_buys']} securities")
    if execution_summary['num_sells'] > 0 and not is_2nd_buy_only_optimization:
        summary_parts.append(f"selling ${execution_summary['total_sells_value']:,.2f} worth of {execution_summary['num_sells']} securities")

    if summary_parts:
        explanation_parts.append(f"{' and '.join(summary_parts)}.")

    # Add individual trade details
    buy_details = []
    sell_details = []

    for _, trade in trades[trades['action'] == 'buy'].iterrows():
        trade_value = trade['quantity'] * trade['price']
        buy_details.append(f"${trade_value:,.0f} {trade['identifier']}")
    if not is_2nd_buy_only_optimization:
        # Aggregate sell trades by identifier
        sell_trades = trades[trades['action'] == 'sell']
        if not sell_trades.empty:
            # Group by identifier and sum quantities and values
            sell_aggregates = {}
            for _, trade in sell_trades.iterrows():
                identifier = trade['identifier']
                trade_value = trade['quantity'] * trade['price']
                if identifier not in sell_aggregates:
                    sell_aggregates[identifier] = {'value': 0, 'lots': 0}
                sell_aggregates[identifier]['value'] += trade_value
                sell_aggregates[identifier]['lots'] += 1
            
            # Create sell details with aggregated values
            for identifier, data in sell_aggregates.items():
                sell_details.append(f"${data['value']:,.0f} {identifier} ({data['lots']} lots)")

    if buy_details:
        explanation_parts.append(f"Buying {', '.join(buy_details)}.")
    if sell_details:
        explanation_parts.append(f"Selling {', '.join(sell_details)}.")

    # Add gain/loss summary if there are sells
    if execution_summary['num_sells'] > 0 and not is_2nd_buy_only_optimization:
        # Count tax lots with gains and losses
        lots_with_gains = sum(1 for _, trade in trades[trades['action'] == 'sell'].iterrows() 
                            if trade['gain_loss']['realized_gain'] > 0)
        lots_with_losses = sum(1 for _, trade in trades[trades['action'] == 'sell'].iterrows() 
                             if trade['gain_loss']['realized_gain'] <= 0)
        tlh_lots = sum(1 for _, trade in trades[trades['action'] == 'sell'].iterrows() 
                      if trade['gain_loss'].get('is_tlh_trade', False))
        total_lots = lots_with_gains + lots_with_losses
        
        if total_lots > 0:
            total_gains = sum(trade['gain_loss']['realized_gain'] 
                            for _, trade in trades[trades['action'] == 'sell'].iterrows() 
                            if trade['gain_loss']['realized_gain'] > 0)
            total_losses = abs(sum(trade['gain_loss']['realized_gain'] 
                                 for _, trade in trades[trades['action'] == 'sell'].iterrows() 
                                 if trade['gain_loss']['realized_gain'] <= 0))
            tlh_losses = abs(sum(trade['gain_loss']['realized_gain']
                               for _, trade in trades[trades['action'] == 'sell'].iterrows()
                               if trade['gain_loss'].get('is_tlh_trade', False)))
            
            explanation = f"\nOut of {total_lots} tax lots to sell"
            if lots_with_losses > 0:
                explanation += f", {lots_with_losses} are at a total loss of ${total_losses:,.0f}"
            if lots_with_gains > 0:
                explanation += f", {lots_with_gains} are at a total gain of ${total_gains:,.0f}"
            explanation += "."
            explanation_parts.append(explanation)
            if tlh_lots > 0:
                explanation_parts.append(f"Of these, {tlh_lots} are tax-loss harvesting trades with realized losses of ${tlh_losses:,.0f}.")

    # Add improvement information
    improvements = []
    for metric, values in component_improvements.items():
        if values['absolute'] != 0:
            if metric != 'overall':  # Skip overall as it will be summarized differently
                direction = "got better" if values['absolute'] > 0 else "got worse"
                improvements.append(f"{metric.replace('_', ' ')} {direction} by {abs(values['absolute']):,.4f}")

    if improvements:
        explanation_parts.append(", ".join(improvements))

    # Add overall impact
    if 'overall' in component_improvements:
        overall_impact = component_improvements['overall']['absolute']
        if overall_impact > 0:
            explanation_parts.append(f"\nOverall, these trades will improve the portfolio by {abs(overall_impact):,.2f}.")
        elif overall_impact < 0:
            explanation_parts.append(f"\nThese trades will cost {abs(overall_impact):,.2f}.")
    
    if additional_info:
        for key, value in additional_info.items():
            explanation_parts.append(f"{key}: {value}")

    return " ".join(explanation_parts)

def generate_trade_summary_from_strategies(
    pre_strategy: OracleStrategy,
    is_2nd_buy_only_optimization: bool,
    trades: pd.DataFrame,
    total_value: float,
    before_optimization: Dict[str, Any],
    after_optimization: Dict[str, Any] = None,
    explanation_context: Dict[str, Any] = None,
    log_time: bool = False
) -> Dict[str, Dict[str, float]]:
    """
    Generate a trade summary by comparing pre and post trade strategies.
    
    Args:
        pre_strategy: The strategy before trades
        trades: DataFrame of trades executed
        total_value: Total portfolio value
        before_optimization: Dictionary of pre-optimization metrics
        after_optimization: Dictionary of post-optimization metrics (optional)
        explanation_context: Dictionary containing context for generating detailed explanations
        log_time: Whether to log timing information
        
    Returns:
        Dictionary containing execution, gain/loss, drift, factor model and optimization summaries
    """
    timing_data = {}
    if log_time:
        start_time = time.time()

    if trades.empty:
        # Generate explanation based on context if provided
        explanation = generate_explanation_from_context(explanation_context) if explanation_context else "No trades were executed as no beneficial trades were identified."
        
        return {
            'execution': {
                'num_buys': 0,
                'num_sells': 0,
                'total_buys_value': 0.0,
                'total_sells_value': 0.0,
                'total_trades': 0,
                'total_value': 0.0
            },
            'gain_loss': {
                'short_term_gains': 0.0,
                'short_term_losses': 0.0,
                'long_term_gains': 0.0,
                'long_term_losses': 0.0,
                'total_gains': 0.0,
                'total_losses': 0.0,
                'net_gain_loss': 0.0
            },
            'drift': {},
            'factor_model': {},
            'optimization_info': {
                'before_optimization': before_optimization,
                'after_optimization': after_optimization,
                'component_improvements': {}
            },
            'explanation': explanation
        }

    if log_time:
        execution_start = time.time()

    # Execution Summary
    buys = trades[trades['action'] == 'buy']
    sells = trades[trades['action'] == 'sell']
    
    buy_values = buys.apply(lambda x: x['quantity'] * x['price'], axis=1).sum()
    sell_values = sells.apply(lambda x: x['quantity'] * x['price'], axis=1).sum()
    
    execution_summary = {
        'num_buys': len(buys),
        'num_sells': len(sells),
        'total_buys_value': buy_values,
        'total_sells_value': sell_values,
        'total_trades': len(trades),
        'total_value': buy_values + sell_values
    }

    if log_time:
        timing_data['execution_summary'] = time.time() - execution_start
        gain_loss_start = time.time()

    # Gain/Loss Summary
    short_term_gains = 0.0
    short_term_losses = 0.0
    long_term_gains = 0.0
    long_term_losses = 0.0
    
    for _, trade in sells.iterrows():
        gain_loss = trade['gain_loss']
        realized_gain = gain_loss['realized_gain']
        gain_type = gain_loss['gain_type']
        
        if realized_gain > 0:
            if gain_type == 'short_term':
                short_term_gains += realized_gain
            else:
                long_term_gains += realized_gain
        else:
            if gain_type == 'short_term':
                short_term_losses += abs(realized_gain)
            else:
                long_term_losses += abs(realized_gain)
                
    total_gains = short_term_gains + long_term_gains
    total_losses = short_term_losses + long_term_losses
    
    gain_loss_summary = {
        'short_term_gains': short_term_gains,
        'short_term_losses': short_term_losses,
        'long_term_gains': long_term_gains,
        'long_term_losses': long_term_losses,
        'total_gains': total_gains,
        'total_losses': total_losses,
        'net_gain_loss': total_gains - total_losses
    }

    if log_time:
        timing_data['gain_loss_summary'] = time.time() - gain_loss_start
        drift_start = time.time()

    # Drift Summary
    drift_summary = {}
    
    if pre_strategy.post_trade_strategy is None:
        raise ValueError("Post trade strategy is not set.")
    _, drift_stats = pre_strategy.compare_drift()
    drift_summary = drift_stats

    if log_time:
        timing_data['drift_summary'] = time.time() - drift_start
        factor_model_start = time.time()

    # Factor Model Summary
    factor_model_summary = {}
    if pre_strategy.post_trade_strategy is None:
        raise ValueError("Post trade strategy is not set.")
    
    if pre_strategy.optimization_type == OracleOptimizationType.DIRECT_INDEX:
        _, factor_stats = pre_strategy.compare_factor_model()
        factor_model_summary = factor_stats

    if log_time:
        timing_data['factor_model_summary'] = time.time() - factor_model_start
        optimization_start = time.time()

    # Use provided after_optimization or calculate it
    if after_optimization is None:
        # Calculate from trade data
        spread_costs = trades['transaction'].apply(lambda x: x['transaction_cost']).sum()
        tax_cost = trades['gain_loss'].apply(lambda x: x.get('tax_cost', 0.0)).sum()

        after_optimization = {
            'tax_cost': tax_cost,
            'spread_costs': spread_costs,
            'overall': tax_cost + spread_costs
        }
    
    # Calculate component improvements
    component_improvements = {}
    if before_optimization and after_optimization:
        for key in before_optimization.keys():
            if key in after_optimization:
                before_val = before_optimization.get(key, 0)
                after_val = after_optimization.get(key, 0)
                improvement = before_val - after_val
                percent_improvement = improvement / abs(before_val) * 100 if before_val != 0 else 0
                component_improvements[key] = {
                    'absolute': improvement,
                    'percent': percent_improvement
                }

    if log_time:
        timing_data['optimization_summary'] = time.time() - optimization_start
        explanation_start = time.time()

    # Generate explanation
    explanation = _generate_explanation(
        trades,
        execution_summary,
        before_optimization,
        after_optimization,
        component_improvements,
        is_2nd_buy_only_optimization,
        {
            "pre_strategy_cash": round(pre_strategy.cash, 0),
            "min_cash": round(pre_strategy.min_cash, 0),
            "post_strategy_cash": round(pre_strategy.post_trade_strategy.cash, 0)
        }
    )

    if log_time:
        timing_data['explanation_generation'] = time.time() - explanation_start
        timing_data['total_time'] = time.time() - start_time
        logger.info("=== Trade Summary Generation Timing Breakdown ===")
        for section, duration in timing_data.items():
            logger.info(f"{section}: {duration:.3f} seconds")

    return {
        'execution': execution_summary,
        'gain_loss': gain_loss_summary,
        'drift': drift_summary,
        'factor_model': factor_model_summary,
        'optimization_info': {
            'before_optimization': before_optimization,
            'after_optimization': after_optimization,
            'component_improvements': component_improvements
        },
        'explanation': explanation
    }
