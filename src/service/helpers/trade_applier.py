"""Functions to apply trades to portfolios and strategies."""
import pandas as pd
from datetime import date, datetime
import time
from typing import Tuple, Optional, Union
from src.service.types import OracleStrategy, ApplyTradesReturn
from src.service.helpers.constants import logger

def apply_trades_to_portfolio(
    tax_lots: pd.DataFrame,
    trades: pd.DataFrame,
    cash: float,
    current_date: Union[date, datetime],
    strategy: Optional[OracleStrategy] = None,
    log_time: bool = False
) -> ApplyTradesReturn:
    """
    Unified function to apply trades to either a raw portfolio or a strategy.
    
    This function combines the previous portfolio_updater.apply_trades and 
    strategy_updater.apply_trades_to_strategy into a single function that can
    handle both use cases.

    Args:
        tax_lots: DataFrame of current tax lots before trades
        trades: DataFrame of trades executed
        cash: Current cash balance before trades
        current_date: The date these trades are considered executed
        strategy: Optional OracleStrategy instance. If provided, updates will be
                 applied to a new strategy instance instead of returning raw components.
        log_time: Whether to log timing information

    Returns:
        If strategy is None:
            A tuple containing:
            - updated_tax_lots: DataFrame of tax lots after applying trades
            - updated_cash: Cash balance after applying trades
            - recently_closed_lots: DataFrame of lots closed in this update
        If strategy is provided:
            A new OracleStrategy instance with trades applied
    """
    timing_data = {}
    if log_time:
        start_time = time.time()

    if trades.empty:
        if strategy is None:
            return tax_lots, cash, pd.DataFrame(columns=[
                'identifier', 'quantity', 'cost_basis', 'date_acquired', 
                'date_sold', 'proceeds', 'realized_gain'
            ])
        return strategy

    # Standardize trades DataFrame format
    if log_time:
        standardize_start = time.time()
        
    trades_for_update = pd.DataFrame([{
        'action': trade['action'],
        'identifier': trade['identifier'],
        'quantity': trade['quantity'],
        'price': trade['price'],
        'tax_lot_id': trade.get('tax_lot_id')
    } for _, trade in trades.iterrows()])

    if log_time:
        timing_data['standardize_trades'] = time.time() - standardize_start
        process_start = time.time()

    updated_tax_lots = tax_lots.copy()
    updated_cash = cash
    new_lots_list = []
    closed_lots_list = []

    for _, trade in trades_for_update.iterrows():
        action = trade['action']
        identifier = trade['identifier']
        quantity = trade['quantity']
        price = trade['price']
        trade_value = quantity * price

        if action == 'sell':
            tax_lot_id = trade['tax_lot_id']
            if tax_lot_id is None:
                logger.warning(f"Sell trade missing tax_lot_id for {identifier}. Skipping.")
                continue
                
            matching_lots = updated_tax_lots[updated_tax_lots['tax_lot_id'] == tax_lot_id]
            
            if matching_lots.empty:
                logger.warning(f"Tax lot {tax_lot_id} not found for selling {identifier}. Skipping trade.")
                continue
                
            if len(matching_lots) > 1:
                raise ValueError(f"Multiple tax lots found with ID {tax_lot_id}. Should not be possible")
                
            lot = matching_lots.iloc[0]
            current_quantity = lot['quantity']
            
            proceeds = trade_value
            cost_basis_per_share = lot['cost_basis'] / current_quantity
            sold_cost_basis = cost_basis_per_share * quantity
            realized_gain = proceeds - sold_cost_basis
            
            closed_lots_list.append({
                'identifier': identifier,
                'quantity': quantity,
                'cost_basis': sold_cost_basis,
                'date_acquired': lot['date'],
                'date_sold': pd.Timestamp(current_date),
                'proceeds': proceeds,
                'realized_gain': realized_gain
            })
            
            new_quantity = current_quantity - quantity
            if new_quantity < 1e-6:  # Consider quantity effectively zero
                updated_tax_lots = updated_tax_lots.drop(index=lot.name)
            else:
                updated_tax_lots.loc[lot.name, 'quantity'] = new_quantity
                updated_tax_lots.loc[lot.name, 'cost_basis'] = lot['cost_basis'] * (new_quantity / current_quantity)
                
            updated_cash += trade_value

        elif action == 'buy':
            cost_basis = trade_value
            new_lot_id = f"lot_{identifier}_{int(time.time()*1e6)}_{len(new_lots_list)}"
            
            new_lot_data = {
                'tax_lot_id': new_lot_id,
                'identifier': identifier,
                'quantity': quantity,
                'cost_basis': cost_basis,
                'date': pd.Timestamp(current_date)
            }
            new_lots_list.append(new_lot_data)
            updated_cash -= cost_basis
            
        else:
            logger.warning(f"Unknown trade action '{action}'. Skipping trade.")

    if log_time:
        timing_data['process_trades'] = time.time() - process_start
        concat_start = time.time()

    if new_lots_list:
        new_lots_df = pd.DataFrame(new_lots_list)
        updated_tax_lots = pd.concat([updated_tax_lots, new_lots_df], ignore_index=True)

    recently_closed_lots = pd.DataFrame(closed_lots_list) if closed_lots_list else pd.DataFrame(columns=[
        'identifier', 'quantity', 'cost_basis', 'date_acquired', 
        'date_sold', 'proceeds', 'realized_gain'
    ])

    if log_time:
        timing_data['concat_and_finalize'] = time.time() - concat_start
        timing_data['total_time'] = time.time() - start_time
        logger.info("=== Apply Trades Timing Breakdown ===")
        for section, duration in timing_data.items():
            logger.info(f"{section}: {duration:.3f} seconds")

    if strategy is None:
        return updated_tax_lots, updated_cash, recently_closed_lots
    
    # If strategy provided, return new strategy instance with updates
    # Import OracleStrategy here to avoid circular import
    from src.service.oracle_strategy import OracleStrategy
    return OracleStrategy(
        tax_lots=updated_tax_lots,
        targets=strategy.targets.copy(),
        prices=strategy.prices.copy(),
        cash=updated_cash,
        spreads=strategy.spreads.copy() if strategy.spreads is not None else None,
        factor_model=strategy.factor_model.copy() if strategy.factor_model is not None else None,
        strategy_id=strategy.strategy_id,
        optimization_type=strategy.optimization_type,
        withdrawal_amount=strategy.withdrawal_amount
    ) 