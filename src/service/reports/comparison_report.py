from typing import Tuple, Dict
import pandas as pd
from src.service.helpers.enums import OracleOptimizationType
import numpy as np

def generate_drift_comparison_report(
    strategy,
    post_trade_strategy
) -> Tuple[pd.DataFrame, Dict]:
    """
    Compare drift between current strategy and post-trade strategy.
    For pairs-based strategies, compares based on asset class drift instead of individual security drift.
    
    Args:
        strategy: The pre-trade strategy object
        post_trade_strategy: The post-trade strategy object
        
    Returns:
        Tuple containing:
        - comparison_df: DataFrame showing drift comparison for each position
        - summary_stats: Dictionary of summary statistics about the drift improvement
    """
        # Use security-level drift for regular strategies
    pre_drift = strategy.drift_report
    post_drift = post_trade_strategy.drift_report
    
    # Join the two drift reports on identifier
    comparison_df = pd.merge(
        pre_drift, 
        post_drift,
        on=['asset_class'], 
        how='outer',
        suffixes=('_pre', '_post')
    )
    
    # Fill NaN values with 0 (for securities that might be in one but not the other)
    comparison_df = comparison_df.fillna(0)
    
    # Calculate drift deltas and improvements
    comparison_df['drift_delta'] = comparison_df['drift_post'] - comparison_df['drift_pre']
    comparison_df['drift_improvement'] = comparison_df['drift_pre'].abs() - comparison_df['drift_post'].abs()
    comparison_df['absolute_improvement'] = comparison_df['drift_improvement'] > 0
    
    # Create derived columns for analysis
    comparison_df['pre_overweight'] = comparison_df['drift_pre'] > 0
    comparison_df['pre_underweight'] = comparison_df['drift_pre'] < 0
    comparison_df['post_overweight'] = comparison_df['drift_post'] > 0
    comparison_df['post_underweight'] = comparison_df['drift_post'] < 0
    
    # Generate summary statistics
    summary_stats = {
        'total_drift_improvement': comparison_df['drift_improvement'].sum(),
        'average_drift_improvement': comparison_df['drift_improvement'].mean(),
        'median_drift_improvement': comparison_df['drift_improvement'].median(),
        'max_drift_improvement': comparison_df['drift_improvement'].max(),
        'positions_improved': int((comparison_df['drift_improvement'] > 0).sum()),
        'positions_worsened': int((comparison_df['drift_improvement'] < 0).sum()),
        'positions_unchanged': int((comparison_df['drift_improvement'] == 0).sum()),
        'pre_num_overweight': int(comparison_df['pre_overweight'].sum()),
        'pre_num_underweight': int(comparison_df['pre_underweight'].sum()),
        'post_num_overweight': int(comparison_df['post_overweight'].sum()),
        'post_num_underweight': int(comparison_df['post_underweight'].sum()),
        'pre_drift_magnitude': comparison_df['drift_pre'].abs().sum(),
        'post_drift_magnitude': comparison_df['drift_post'].abs().sum(),
        'overall_drift_reduction': 1 - (
            comparison_df['drift_post'].abs().sum() /
            comparison_df['drift_pre'].abs().sum() if comparison_df['drift_pre'].abs().sum() > 0 else 0
        ),
    }
    
    # # Add weighted metrics (by target weight)
    if 'target_weight_pre' in comparison_df.columns:
        summary_stats.update({
            'weighted_drift_reduction': 1 - (
                (comparison_df['drift_post'].abs() * comparison_df['target_weight_pre']).sum() /
                (comparison_df['drift_pre'].abs() * comparison_df['target_weight_pre']).sum()
                if (comparison_df['drift_pre'].abs() * comparison_df['target_weight_pre']).sum() > 0 else 0
            )
        })
    
    return comparison_df, summary_stats

def generate_factor_model_comparison_report(
    strategy,
    post_trade_strategy
) -> Tuple[pd.DataFrame, Dict]:
    """
    Compare factor model exposures between current strategy and post-trade strategy.
    Only applicable for DIRECT_INDEX optimization type.
    
    Args:
        strategy: The pre-trade strategy object
        post_trade_strategy: The post-trade strategy object
        
    Returns:
        Tuple containing:
        - comparison_df: DataFrame showing factor exposure comparison
        - summary_stats: Dictionary of summary statistics about factor exposure improvements
    """
    if strategy.optimization_type != OracleOptimizationType.DIRECT_INDEX:
        raise ValueError("Factor model comparison is only available for DIRECT_INDEX optimization type.")
        
    if strategy.factor_model is None or post_trade_strategy.factor_model is None:
        raise ValueError("Factor model data not available for comparison.")
        
    # Extract factor model data
    pre_factor = strategy.factor_model_actual
    post_factor = post_trade_strategy.factor_model_actual
    target_factor = strategy.factor_model_target
    
    # Transpose the factor dataframes before converting to DataFrames
    pre_df = pd.DataFrame(pd.Series(pre_factor) if np.ndim(pre_factor) > 2 else pre_factor).T.reset_index()
    pre_df.columns = ['factor', 'exposure_pre']
    
    post_df = pd.DataFrame(pd.Series(post_factor) if np.ndim(post_factor) > 2 else post_factor).T.reset_index()
    post_df.columns = ['factor', 'exposure_post']
    
    target_df = pd.DataFrame(pd.Series(target_factor) if np.ndim(target_factor) > 2 else target_factor).T.reset_index()
    target_df.columns = ['factor', 'exposure_target']
    
    # Merge all three dataframes on factor name
    comparison_df = pd.merge(
        pd.merge(
            pre_df,
            post_df,
            on=['factor'],
            how='outer'
        ),
        target_df,
        on=['factor'],
        how='outer'
    )
    
    # Fill NaN values with 0 (for factors that might be in one but not the other)
    comparison_df = comparison_df.fillna(0)
    
    # Calculate drift from target for pre and post
    comparison_df['drift_pre'] = comparison_df['exposure_pre'] - comparison_df['exposure_target']
    comparison_df['drift_post'] = comparison_df['exposure_post'] - comparison_df['exposure_target']
    
    # Calculate improvement metrics
    comparison_df['drift_delta'] = comparison_df['drift_post'] - comparison_df['drift_pre']
    comparison_df['drift_improvement'] = comparison_df['drift_pre'].abs() - comparison_df['drift_post'].abs()
    comparison_df['absolute_improvement'] = comparison_df['drift_improvement'] > 0
    
    # Create derived columns for analysis
    comparison_df['pre_overexposed'] = comparison_df['drift_pre'] > 0
    comparison_df['pre_underexposed'] = comparison_df['drift_pre'] < 0
    comparison_df['post_overexposed'] = comparison_df['drift_post'] > 0
    comparison_df['post_underexposed'] = comparison_df['drift_post'] < 0
    
    # Generate summary statistics
    summary_stats = {
        'total_factor_improvement': comparison_df['drift_improvement'].sum(),
        'average_factor_improvement': comparison_df['drift_improvement'].mean(),
        'median_factor_improvement': comparison_df['drift_improvement'].median(),
        'max_factor_improvement': comparison_df['drift_improvement'].max(),
        'factors_improved': int((comparison_df['drift_improvement'] > 0).sum()),
        'factors_worsened': int((comparison_df['drift_improvement'] < 0).sum()),
        'factors_unchanged': int((comparison_df['drift_improvement'] == 0).sum()),
        'pre_num_overexposed': int(comparison_df['pre_overexposed'].sum()),
        'pre_num_underexposed': int(comparison_df['pre_underexposed'].sum()),
        'post_num_overexposed': int(comparison_df['post_overexposed'].sum()),
        'post_num_underexposed': int(comparison_df['post_underexposed'].sum()),
        'pre_factor_tracking_error': comparison_df['drift_pre'].abs().sum(),
        'post_factor_tracking_error': comparison_df['drift_post'].abs().sum(),
        'overall_factor_improvement': 1 - (
            comparison_df['drift_post'].abs().sum() / 
            comparison_df['drift_pre'].abs().sum() if comparison_df['drift_pre'].abs().sum() > 0 else 0
        ),
    }
    
    # Add weighted metrics if factor importance is available
    if 'importance' in comparison_df.columns:
        summary_stats.update({
            'weighted_factor_improvement': 1 - (
                (comparison_df['drift_post'].abs() * comparison_df['importance']).sum() /
                (comparison_df['drift_pre'].abs() * comparison_df['importance']).sum()
                if (comparison_df['drift_pre'].abs() * comparison_df['importance']).sum() > 0 else 0
            )
        })
        
        # Add top 5 factor improvements (weighted by importance)
        top_improvements = comparison_df.sort_values(
            by='drift_improvement', 
            ascending=False
        ).head(5)
        
        summary_stats['top_factor_improvements'] = {
            row['factor']: {
                'improvement': row['drift_improvement'],
                'pre_drift': row['drift_pre'],
                'post_drift': row['drift_post']
            }
            for _, row in top_improvements.iterrows()
        }
    
    return comparison_df, summary_stats 