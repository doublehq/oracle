import unittest
from datetime import date, timedelta
import pandas as pd
import json
from pathlib import Path
import numpy as np
from src.service.helpers.enums import OracleOptimizationType
from src.service.oracle import Oracle, OracleStrategy
from src.service.initializers import (
    initialize_tax_lots,
    initialize_targets,
    initialize_prices,
    initialize_spreads,
    initialize_closed_lots,
    initialize_stock_restrictions,
    initialize_tax_rates,
    initialize_factor_model
)

def compare_trades(actual_trade, expected_trade, tolerance=0.05):
    """
    Compare if an actual trade matches an expected trade within tolerance.
    
    Args:
        actual_trade: Dictionary containing actual trade info
        expected_trade: Dictionary containing expected trade info
        tolerance: Percentage tolerance for quantity comparison (default 5%)
        
    Returns:
        bool: True if trades match within tolerance
    """
    if actual_trade['identifier'] != expected_trade['identifier']:
        return False
        
    actual_qty = float(actual_trade['quantity'])
    expected_qty = float(expected_trade['quantity'])
    
    # Handle zero quantity edge case
    if expected_qty == 0:
        return abs(actual_qty) < 1e-6
        
    percent_diff = abs(actual_qty - expected_qty) / abs(expected_qty)
    return percent_diff <= tolerance

def analyze_trade_patterns(actual_trades_df, expected_trades):
    """
    Analyze patterns between actual and expected trades and determine if they match sufficiently.
    
    Args:
        actual_trades_df: DataFrame containing actual trades
        expected_trades: List of dictionaries containing expected trades
        
    Returns:
        tuple: (bool, dict) where:
            - bool: True if trades match sufficiently, False otherwise
            - dict: Analysis results containing:
                - overbuying: List of securities we're buying too much of
                - overselling: List of securities we're selling too much of
                - missing_trades: Expected trades we didn't make
                - unexpected_trades: Trades we made that weren't expected
                - matching_trades: Trades that match within tolerance
                Each category includes both count and dollar amount metrics
    """
    results = {
        'overbuying': [],
        'overselling': [],
        'missing_trades': [],
        'unexpected_trades': [],
        'matching_trades': [],
        'total_amounts': {
            'overbuying_amount': 0.0,
            'overselling_amount': 0.0,
            'missing_trades_amount': 0.0,
            'unexpected_trades_amount': 0.0,
            'matching_trades_amount': 0.0
        }
    }
    
    # Handle empty actual trades DataFrame
    if actual_trades_df is None or actual_trades_df.empty or 'identifier' not in actual_trades_df.columns:
        # All expected trades are missing in this case
        for expected_trade in expected_trades:
            if 'price' in expected_trade:
                price = float(expected_trade['price'])
            else:
                price = 0.0  # No price available
                
            expected_amount = abs(float(expected_trade['quantity']) * price)
            results['missing_trades'].append({
                'identifier': expected_trade['identifier'],
                'expected_qty': expected_trade['quantity'],
                'price': price,
                'amount': expected_amount
            })
            results['total_amounts']['missing_trades_amount'] += expected_amount
        return len(expected_trades) == 0, results
    
    # Handle empty expected trades
    if not expected_trades:
        # All actual trades are unexpected in this case
        for _, actual_trade in actual_trades_df.iterrows():
            actual_amount = abs(float(actual_trade['quantity']) * float(actual_trade['price']))
            results['unexpected_trades'].append({
                'identifier': actual_trade['identifier'],
                'quantity': actual_trade['quantity'],
                'price': actual_trade['price'],
                'amount': actual_amount
            })
            results['total_amounts']['unexpected_trades_amount'] += actual_amount
        # Fail if we have trades when none were expected
        return len(actual_trades_df) == 0, results
    
    # Convert expected trades to DataFrame for easier comparison
    expected_df = pd.DataFrame(expected_trades)
    
    # Find matching trades
    for _, actual_trade in actual_trades_df.iterrows():
        found_match = False
        actual_amount = abs(float(actual_trade['quantity']) * float(actual_trade['price']))
        
        for _, expected_trade in expected_df.iterrows():
            if compare_trades(actual_trade, expected_trade):
                results['matching_trades'].append({
                    'identifier': actual_trade['identifier'],
                    'actual_qty': actual_trade['quantity'],
                    'expected_qty': expected_trade['quantity'],
                    'price': actual_trade['price'],
                    'amount': actual_amount
                })
                results['total_amounts']['matching_trades_amount'] += actual_amount
                found_match = True
                break
                
        if not found_match:
            # Check if this security exists in expected trades
            expected_for_security = expected_df[expected_df['identifier'] == actual_trade['identifier']]
            if len(expected_for_security) > 0:
                # Trade exists but quantity is off
                expected_qty = float(expected_for_security.iloc[0]['quantity'])
                expected_amount = abs(expected_qty * float(actual_trade['price']))
                diff_amount = abs(actual_amount - expected_amount)
                
                if actual_trade['quantity'] > expected_qty:
                    results['overbuying'].append({
                        'identifier': actual_trade['identifier'],
                        'actual_qty': actual_trade['quantity'],
                        'expected_qty': expected_qty,
                        'price': actual_trade['price'],
                        'amount': actual_amount,
                        'expected_amount': expected_amount,
                        'diff_amount': diff_amount
                    })
                    results['total_amounts']['overbuying_amount'] += diff_amount
                else:
                    results['overselling'].append({
                        'identifier': actual_trade['identifier'],
                        'actual_qty': actual_trade['quantity'],
                        'expected_qty': expected_qty,
                        'price': actual_trade['price'],
                        'amount': actual_amount,
                        'expected_amount': expected_amount,
                        'diff_amount': diff_amount
                    })
                    results['total_amounts']['overselling_amount'] += diff_amount
            else:
                # Completely unexpected trade
                results['unexpected_trades'].append({
                    'identifier': actual_trade['identifier'],
                    'quantity': actual_trade['quantity'],
                    'price': actual_trade['price'],
                    'amount': actual_amount
                })
                results['total_amounts']['unexpected_trades_amount'] += actual_amount
    
    # Find missing trades
    for _, expected_trade in expected_df.iterrows():
        actual_for_security = actual_trades_df[actual_trades_df['identifier'] == expected_trade['identifier']]
        if len(actual_for_security) == 0:
            # Use the price from actual_trades_df for the same security if available, otherwise use expected price
            if 'price' in expected_trade:
                price = float(expected_trade['price'])
            else:
                # Since we already validated actual_trades_df has 'identifier' column above, this is safe
                matching_actual = actual_trades_df[actual_trades_df['identifier'] == expected_trade['identifier']]
                price = float(matching_actual['price'].iloc[0]) if not matching_actual.empty else 0.0
                
            expected_amount = abs(float(expected_trade['quantity']) * price)
            results['missing_trades'].append({
                'identifier': expected_trade['identifier'],
                'expected_qty': expected_trade['quantity'],
                'price': price,
                'amount': expected_amount
            })
            results['total_amounts']['missing_trades_amount'] += expected_amount
    
    # Calculate total expected and actual trade amounts
    total_expected_amount = (results['total_amounts']['matching_trades_amount'] + 
                           results['total_amounts']['missing_trades_amount'])
    total_actual_amount = (results['total_amounts']['matching_trades_amount'] + 
                          results['total_amounts']['overbuying_amount'] + 
                          results['total_amounts']['overselling_amount'] + 
                          results['total_amounts']['unexpected_trades_amount'])
    
    # Define pass/fail criteria
    # 1. No unexpected trades
    # 2. No missing trades
    # 3. Total trade amount difference within 5%
    # 4. At least 90% of trades match within tolerance
    passes = (
        len(results['unexpected_trades']) == 0 and
        len(results['missing_trades']) == 0 and
        (total_expected_amount == 0 or 
         abs(total_actual_amount - total_expected_amount) / total_expected_amount <= 0.05) and
        (len(expected_trades) == 0 or 
         len(results['matching_trades']) / len(expected_trades) >= 0.9)
    )
    
    return passes, results

class TestOptimizationVars(unittest.TestCase):
    """Test optimization variables and trade generation."""
    
    def setUp(self):
        self.test_data_dir = Path(__file__).parent / 'oracle_test_dumps'

    def _create_oracle_from_json(self, json_data):
        """Create an Oracle instance from JSON test data using initializers."""
        current_date = date.fromisoformat(json_data.get('oracle', {}).get('current_date', '2025-01-01'))
        
        # Initialize tax rates
        tax_rates_df = pd.DataFrame(json_data.get('oracle', {}).get('tax_rates', []))
        tax_rates = initialize_tax_rates(tax_rates_df)

        # Initialize stock restrictions
        stock_restrictions_df = pd.DataFrame(json_data.get('oracle', {}).get('stock_restrictions', []))
        stock_restrictions = initialize_stock_restrictions(stock_restrictions_df)
        
        # Initialize recently closed lots
        closed_lots_df = pd.DataFrame(json_data.get('oracle', {}).get('closed_lots', []))
        recently_closed_lots = initialize_closed_lots(closed_lots_df)

        # Create and return Oracle instance
        oracle = Oracle(
            current_date=current_date,
            recently_closed_lots=recently_closed_lots,
            stock_restrictions=stock_restrictions,
            tax_rates=tax_rates
        )
        
        # Process strategies
        strategies = []
        for strat_data in json_data.get('strategies', []):
            strategy_id = strat_data.get('strategy_id', None)
            optimization_mode = strat_data['optimization_mode']
            if "QUARTERLY_REBALANCING" in optimization_mode:
                optimization_mode = "BUY_ONLY"
            optimization_mode = OracleOptimizationType(optimization_mode)
            # Initialize tax lots
            tax_lots_df = pd.DataFrame(strat_data.get('tax_lots', []))
            tax_lots = initialize_tax_lots(tax_lots_df)
            
            # Initialize targets
            targets_df = pd.DataFrame(strat_data.get('targets', []))
            targets = initialize_targets(targets_df)
            
            # Get all identifiers needed for prices and spreads
            all_identifiers = set(tax_lots['identifier']) | set(targets['identifier'])
            
            # Initialize prices
            prices_df = pd.DataFrame(strat_data.get('prices', []))
            prices = initialize_prices(prices_df, all_identifiers)
            
            # Initialize spreads
            spreads_df = pd.DataFrame(strat_data.get('spreads', []))
            spreads = initialize_spreads(spreads_df, all_identifiers, prices)
            
            # Initialize factor model if present
            factor_model_data = strat_data.get('factor_model', [])
            if factor_model_data and optimization_mode == OracleOptimizationType.DIRECT_INDEX:
                factor_model_df = pd.DataFrame(factor_model_data)
            else: 
                factor_model_df = None

            # Create strategy
            strategy = OracleStrategy(
                strategy_id=strategy_id,
                optimization_type=optimization_mode,
                cash=strat_data['cash'],
                tax_lots=tax_lots,
                targets=targets,
                prices=prices,
                spreads=spreads,
                factor_model=factor_model_df,
                deminimus_cash_target=OracleStrategy.DEMINIMUS_CASH_TARGET_PERCENT
            )
            strategy.set_oracle(oracle)
            strategies.append(strategy)
        oracle.strategies = strategies
        oracle.initialize_wash_sale_restrictions()
        return oracle
    
    def test_buy_only(self):
        """Test trade generation and normalization values."""
        results = []
        all_tests_pass = True
        
        # for json_file in sorted(self.test_data_dir.glob('oracle_data_id*.json'), reverse=True):
        for json_file in sorted(self.test_data_dir.glob('oracle_data_id_20948.json'), reverse=True):
            print(f"\nProcessing {json_file.name}")
        
            with open(json_file, 'r') as f:
                test_data = json.load(f)

            expected_trades = test_data.get('output_strategy_trades', [])
            expected_explanations = test_data.get('output_strategy_explanations', '')
            
            # Create Oracle instance
            oracle = self._create_oracle_from_json(test_data)
            
            if not oracle.strategies:
                print(f"No strategies in {json_file.name}, skipping")
                continue
            
            # results, netted_trades = oracle.compute_optimal_trades_for_all_strategies(min_notional=5.10)
            # pass
            for strategy in oracle.strategies:
                # if strategy.optimization_type != OracleOptimizationType.BUY_ONLY:
                #     print(f"Skipping strategy {strategy.strategy_id} with optimization type {strategy.optimization_type.value}")
                #     continue
                if strategy.strategy_id != 496:
                    print(f"Skipping strategy {strategy.strategy_id}")
                    continue
                
                # Compute optimal trades
                should_tlh = strategy.optimization_type in (OracleOptimizationType.PAIRS_TLH, OracleOptimizationType.DIRECT_INDEX)
                status, should_trade, trade_summary, trades = strategy.compute_optimal_trades(
                    weight_tax=1.0,
                    weight_drift=1.0,
                    weight_transaction=1.0,
                    weight_factor_model=1.0,
                    weight_cash_drag=1.0,
                    should_tlh=should_tlh,
                    tlh_min_loss_threshold=0.015,
                    min_notional=5.0,
                    buy_threshold=None,
                    rebalance_threshold=None,
                    trade_rounding=5,
                    debug=False
                ) 

                # Get expected trades and explanation from test data
                # Filter expected trades to only include those matching the current strategy ID
                expected_strategy_trades = [
                    trade for trade in expected_trades 
                    if 'strategy_id' in trade and trade['strategy_id'] == strategy.strategy_id
                ]
                expected_explanation = next(
                    (explanation['explanation'] for explanation in expected_explanations 
                     if 'strategy_id' in explanation and explanation['strategy_id'] == strategy.strategy_id),
                    None
                )

                trade_summary = trade_summary or {}
                explanation = trade_summary.get('explanation', "No explanation provided")
                print(f"Testing strategy type: {strategy.optimization_type.value}")
                print(f"strategy_id: {strategy.strategy_id} Explanation: {explanation}")
                print(f"strategy_id: {strategy.strategy_id} ROWBOAT Explanation: {expected_explanation}")
                
                # Analyze trade patterns
                passes, trade_analysis = analyze_trade_patterns(trades, expected_strategy_trades)
                all_tests_pass = all_tests_pass and passes
                
                # Print detailed analysis if test fails
                if not passes:
                    print(f"\nTest failed for strategy {strategy.strategy_id} in {json_file.name}")
                    print(f"Matching trades: {len(trade_analysis['matching_trades'])}")
                    print(f"Overbuying trades: {len(trade_analysis['overbuying'])}")
                    print(f"Overselling trades: {len(trade_analysis['overselling'])}")
                    print(f"Missing trades: {len(trade_analysis['missing_trades'])}")
                    print(f"Unexpected trades: {len(trade_analysis['unexpected_trades'])}")
                    print(f"Total amounts:")
                    for key, amount in trade_analysis['total_amounts'].items():
                        print(f"  {key}: ${amount:,.2f}")
                
                results.append({
                    'file': json_file.name,
                    'passed': passes,
                    'oracle_explanation': explanation,
                    'rowboat_explanation': expected_explanation,
                    'strategy_id': strategy.strategy_id,
                    'optimization_type': strategy.optimization_type.value,
                    'analysis': trade_analysis,
                    'matching_trades_count': len(trade_analysis['matching_trades']),
                    'overbuying_count': len(trade_analysis['overbuying']),
                    'overselling_count': len(trade_analysis['overselling']),
                    'missing_trades_count': len(trade_analysis['missing_trades']),
                    'unexpected_trades_count': len(trade_analysis['unexpected_trades']),
                    'matching_trades_amount': trade_analysis['total_amounts']['matching_trades_amount'],
                    'overbuying_amount': trade_analysis['total_amounts']['overbuying_amount'],
                    'overselling_amount': trade_analysis['total_amounts']['overselling_amount'],
                    'missing_trades_amount': trade_analysis['total_amounts']['missing_trades_amount'],
                    'unexpected_trades_amount': trade_analysis['total_amounts']['unexpected_trades_amount']
                })
        
        self.assertTrue(all_tests_pass, "One or more trade pattern tests failed")
        return results
    