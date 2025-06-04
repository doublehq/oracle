import pandas as pd
import numpy as np
import pulp
from src.service.objectives.taxes.tlh import TLHTrade
from src.service.objectives.transaction_costs.transaction_optimization import (
    get_buy_cost,
    get_sell_cost
)
from src.service.objectives.taxes.tax_optimization import get_tax_cost

from typing import Optional
from src.service.helpers.enums import OracleOptimizationType

def smart_round_trades(trades_df: pd.DataFrame, trade_rounding: int, min_notional: float) -> pd.DataFrame:
    """
    Alternative implementation of smart rounding that:
    1. First rounds all trades DOWN
    2. Identifies trades below min_notional and rounds those UP
    3. Distributes remaining value difference to maintain cash balance
    
    Args:
        trades_df: DataFrame of trades before rounding
        trade_rounding: Number of decimal places to round to
        min_notional: Minimum trade value allowed. If 0 or None, no minimum notional checks will be performed.
        
    Returns:
        DataFrame with rounded trades that preserve total value and min notional requirements
    """
    if trades_df.empty:
        return trades_df
    
    # Create working copy and remove zero quantity trades
    working_df = trades_df.copy()
    working_df = working_df[working_df['quantity'] > 0]
    
    if working_df.empty:
        return working_df
    
    # Calculate original total value
    original_total_value = (working_df['quantity'] * working_df['price']).sum()
    
    # Calculate minimum increment
    min_increment = 1.0 / 10**trade_rounding
    
    # First pass: Round all trades down
    working_df['quantity'] = np.floor(working_df['quantity'] * 10**trade_rounding) / 10**trade_rounding
    working_df['trade_value'] = working_df['quantity'] * working_df['price']
    
    # Second pass: Round up trades below min_notional
    if min_notional and min_notional > 0:
        # Group trades by identifier and action to check combined value
        grouped_values = working_df.groupby(['identifier', 'action'])['trade_value'].transform('sum')
        
        # Identify trades below min_notional
        below_min = grouped_values < min_notional
        
        # Round up these trades
        for idx in working_df[below_min].index:
            price = working_df.loc[idx, 'price']
            current_quantity = working_df.loc[idx, 'quantity']
            
            # Calculate how many increments needed to reach min_notional
            current_value = current_quantity * price
            needed_value = min_notional - current_value
            needed_increments = int(np.ceil(needed_value / (price * min_increment)))
            
            # Round up
            new_quantity = current_quantity + (needed_increments * min_increment)
            working_df.loc[idx, 'quantity'] = np.round(new_quantity, trade_rounding)
            working_df.loc[idx, 'trade_value'] = new_quantity * price
    
    # Calculate new total value and difference
    working_df['trade_value'] = working_df['quantity'] * working_df['price']
    new_total_value = working_df['trade_value'].sum()
    value_difference = original_total_value - new_total_value
    
    # Third pass: Distribute remaining value difference
    if abs(value_difference) > 0:
        # Sort by trade value to ensure deterministic distribution
        working_df = working_df.sort_values('trade_value', ascending=False)
        
        # Calculate minimum possible increment value
        min_increment_value = working_df['price'].min() * min_increment
        
        # Only proceed if remaining value is at least as large as minimum increment
        if abs(value_difference) >= min_increment_value:
            # Determine if we need to add or subtract value
            add_value = value_difference > 0
            
            for idx in working_df.index:
                if abs(value_difference) <= 0:
                    break
                    
                price = working_df.loc[idx, 'price']
                value_per_unit = price * min_increment
                
                if value_per_unit <= abs(value_difference):
                    if add_value:
                        working_df.loc[idx, 'quantity'] = np.round(working_df.loc[idx, 'quantity'] + min_increment, trade_rounding)
                    else:
                        working_df.loc[idx, 'quantity'] = np.round(working_df.loc[idx, 'quantity'] - min_increment, trade_rounding)
                    value_difference = round(value_difference - (value_per_unit if add_value else -value_per_unit), trade_rounding)
                    working_df.loc[idx, 'trade_value'] = working_df.loc[idx, 'quantity'] * price
    
    # Remove 0 quantity trades
    working_df = working_df[working_df['quantity'] > 0]
    
    # Remove trades below min_notional if min_notional is specified
    if min_notional and min_notional > 0:
        working_df = working_df[working_df.groupby(['identifier', 'action'])['trade_value'].transform('sum') >= min_notional]
    
    return working_df.sort_values('trade_value', ascending=False)

def extract_trades(
    buys: dict,
    sells: dict,
    gain_loss: pd.DataFrame,
    total_value: float,
    prices: pd.DataFrame,
    spreads: pd.DataFrame,
    tlh_trades: list[TLHTrade],
    tax_normalization: float,
    transaction_normalization: float,
    trade_rounding: int = 4,
    min_notional: Optional[float] = None,
) -> pd.DataFrame:
    """
    Extract trade results from the optimization solution.
    
    Args:
        buys: Dictionary of buy variables {identifier: LpVariable}
        sells: Dictionary of sell variables {tax_lot_id: LpVariable}
        gain_loss: DataFrame with gain/loss report
        total_value: Total portfolio value for calculating drift impact
        prices: DataFrame with current prices
        spreads: DataFrame with bid-ask spreads
        tlh_trades: List of tax-loss harvesting trades
        tax_normalization: Normalization factor for tax impact
        transaction_normalization: Normalization factor for transaction costs
        trade_rounding: Number of decimal places to round trades to (default 4)
        
    Returns:
        DataFrame with trade details including metadata about optimization
    """
    trades = []
    
    # Process buys
    for identifier, var in buys.items():
        # Skip if value is None 
        if var.value() is None:
            continue

        price = prices.loc[prices['identifier'] == identifier, 'price'].iloc[0]
        quantity = var.value()
        trade_value = quantity * price
        
        # Safely get spread (default to 0 if not found)
        spread = 0.0
        if spreads is not None and not spreads.empty and identifier in spreads['identifier'].values:
            spread = spreads.loc[spreads['identifier'] == identifier, 'spread'].iloc[0]
        buy_cost = get_buy_cost(quantity, spread, total_value, transaction_normalization)
        is_tlh_trade = any(hasattr(trade, 'replacement_buys') and trade.replacement_buys is not None and identifier in trade.replacement_buys.keys() for trade in tlh_trades)

        trades.append({
            'identifier': identifier,
            'tax_lot_id': None,
            'action': 'buy',
            'trade_value': trade_value,
            'quantity': var.value(),
            'lot_quantity': var.value(),
            'price': price,
            'gain_loss': {
                'cost_basis': trade_value,  # Cost basis for new purchases
                'realized_gain': 0.0,  # No realized gain/loss for buys
                'gain_type': None,  # Gain type is None for new buys
                'is_tlh_trade': is_tlh_trade,
                'tax_cost': 0,  # Buys do not incur tax costs
            },
            'transaction': {
                'spread': spread,
                'transaction_cost': buy_cost,
            }
        })
            
    # Process sells
    for tax_lot_id, var in sells.items():
        # Skip if value is None
        if var.value() is None:
            continue
            
        lot = gain_loss[gain_loss['tax_lot_id'] == tax_lot_id].iloc[0]
        identifier = lot['identifier']
        price = prices.loc[prices['identifier'] == identifier, 'price'].iloc[0]
        quantity = var.value()
        trade_value = quantity * price
        
        # Safely get spread (default to 0 if not found)
        spread = 0.0
        if spreads is not None and not spreads.empty and identifier in spreads['identifier'].values:
            spread = spreads.loc[spreads['identifier'] == identifier, 'spread'].iloc[0]
        sell_cost = get_sell_cost(quantity, spread, total_value, transaction_normalization)

        # Calculate realized gain/loss
        per_share_tax_liability = (gain_loss.loc[gain_loss['identifier'] == identifier, 'per_share_tax_liability'].iloc[0] 
                                 if 'per_share_tax_liability' in gain_loss.columns else 0.0)
        realized_gain = quantity * (lot['current_price'] - lot['cost_per_share'])
        tax_cost = get_tax_cost(quantity, per_share_tax_liability, total_value, tax_normalization)
        is_tlh_trade = any(trade.tax_lot_id == tax_lot_id for trade in tlh_trades)
        
        trades.append({           
            'identifier': lot['identifier'],
            'tax_lot_id': tax_lot_id,
            'action': 'sell',
            'trade_value': trade_value,
            'quantity': quantity,
            'lot_quantity': lot['quantity'],
            'price': lot['current_price'],
            'gain_loss': {
                'cost_basis': quantity * lot['cost_per_share'],
                'realized_gain': realized_gain,
                'gain_type': lot['gain_type'],
                'is_tlh_trade': is_tlh_trade,
                'tax_cost': tax_cost,
            },
            'transaction': {
                'spread': spread,
                'transaction_cost': sell_cost
            }
        })
    
    # Create DataFrame
    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df

    # Apply smart rounding
    trades_df = smart_round_trades(trades_df, trade_rounding, min_notional)
    
    return trades_df.sort_values(by=['trade_value'], ascending=False)