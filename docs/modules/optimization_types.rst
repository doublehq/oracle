Optimization Types
==================

Oracle supports several optimization types for any given strategy through the ``OracleOptimizationType`` enum.
Each type has specific behaviors and use cases.

HOLD
----
The ``HOLD`` optimization type prevents any trading. It's useful when you want to temporarily freeze a portfolio:

.. code-block:: python

   # Example: Create a HOLD strategy
   strategy = OracleStrategy(
       tax_lots=tax_lots_df,
       targets=targets_df,
       prices=prices_df,
       cash=0.0,
       optimization_type=OracleOptimizationType.HOLD
   )
   
   # No trades will be generated regardless of drift or other conditions
   status, should_trade, trade_summary, trades = strategy.compute_optimal_trades()
   # should_trade will be False, trades will be empty

BUY_ONLY
--------
The ``BUY_ONLY`` type only allows buy trades, no sells permitted. Useful for initial portfolio setup or adding new cash:

.. code-block:: python

   # Example: Create a BUY_ONLY strategy with $20,000 cash
   strategy = OracleStrategy(
       tax_lots=tax_lots_df,
       targets=pd.DataFrame([
           {'asset_class': 'STOCK_A', 'identifiers': ['STOCK_A'], 'target_weight': 0.5},
           {'asset_class': 'STOCK_B', 'identifiers': ['STOCK_B'], 'target_weight': 0.5},
           {'asset_class': 'CASH', 'identifiers': ['CASH'], 'target_weight': 0.0}
       ]),
       prices=pd.DataFrame([
           {'identifier': 'STOCK_A', 'price': 100.0},
           {'identifier': 'STOCK_B', 'price': 80.0},
           {'identifier': 'CASH', 'price': 1.0}
       ]),
       cash=20000.0,
       optimization_type=OracleOptimizationType.BUY_ONLY
   )
   
   # Only buy trades will be generated
   status, should_trade, trade_summary, trades = strategy.compute_optimal_trades()
   # trades will only contain 'buy' actions

TAX_UNAWARE
-----------
The ``TAX_UNAWARE`` type rebalances towards targets ignoring tax implications:

.. code-block:: python

   # Example: Create a TAX_UNAWARE strategy
   strategy = OracleStrategy(
       tax_lots=pd.DataFrame([
           {'tax_lot_id': 'lot_a1', 'identifier': 'STOCK_A', 'quantity': 1000, 
            'cost_basis': 100000, 'date': '2024-01-01'},  # $100/share cost
           {'tax_lot_id': 'lot_b1', 'identifier': 'STOCK_B', 'quantity': 1000, 
            'cost_basis': 100000, 'date': '2024-01-01'}   # $100/share cost
       ]),
       targets=pd.DataFrame([
           {'asset_class': 'STOCK_A', 'identifiers': ['STOCK_A'], 'target_weight': 0.5},
           {'asset_class': 'STOCK_B', 'identifiers': ['STOCK_B'], 'target_weight': 0.5},
           {'asset_class': 'CASH', 'identifiers': ['CASH'], 'target_weight': 0.0}
       ]),
       prices=pd.DataFrame([
           {'identifier': 'STOCK_A', 'price': 120.0},  # Up 20%
           {'identifier': 'STOCK_B', 'price': 80.0},   # Down 20%
           {'identifier': 'CASH', 'price': 1.0}
       ]),
       optimization_type=OracleOptimizationType.TAX_UNAWARE
   )
   
   # Will sell STOCK_A (despite gains) and buy STOCK_B to rebalance
   status, should_trade, trade_summary, trades = strategy.compute_optimal_trades(
       weight_tax=1.0,
       weight_drift=1.0,
       weight_transaction=1.0
   )

TAX_AWARE
---------
The ``TAX_AWARE`` type (default) rebalances considering tax implications:

.. code-block:: python

   # Example: Create a TAX_AWARE strategy with tax-loss harvesting opportunity
   strategy = OracleStrategy(
       tax_lots=pd.DataFrame([
           {'tax_lot_id': 'lot_tlh', 'identifier': 'STOCK_TLH', 'quantity': 100, 
            'cost_basis': 10000, 'date': '2024-01-01'},  # $100/share cost
           {'tax_lot_id': 'lot_gain', 'identifier': 'STOCK_GAIN', 'quantity': 100, 
            'cost_basis': 8000, 'date': '2024-01-01'}    # $80/share cost
       ]),
       targets=pd.DataFrame([
           {'asset_class': 'STOCK_TLH', 'identifiers': ['STOCK_TLH'], 'target_weight': 0.5},
           {'asset_class': 'STOCK_GAIN', 'identifiers': ['STOCK_GAIN'], 'target_weight': 0.5},
           {'asset_class': 'CASH', 'identifiers': ['CASH'], 'target_weight': 0.0}
       ]),
       prices=pd.DataFrame([
           {'identifier': 'STOCK_TLH', 'price': 85.0},    # Down 15%
           {'identifier': 'STOCK_GAIN', 'price': 100.0},  # Up 25%
           {'identifier': 'CASH', 'price': 1.0}
       ]),
       optimization_type=OracleOptimizationType.TAX_AWARE
   )
   
   # Will prioritize selling STOCK_TLH for tax loss harvesting
   status, should_trade, trade_summary, trades = strategy.compute_optimal_trades(
       weight_tax=1.0,
       weight_drift=0.0001,  # Low weight on drift to prioritize tax
       weight_transaction=0.0
   )

PAIRS_TLH
---------
The ``PAIRS_TLH`` type is specifically designed for tax-loss harvesting with pairs trading:

.. code-block:: python

   # Example: Create a PAIRS_TLH strategy with ETF pairs
   strategy = OracleStrategy(
       tax_lots=pd.DataFrame([
           {'identifier': 'VTI', 'tax_lot_id': 'VTI_lot', 'quantity': 200, 
            'cost_basis': 40000, 'date': '2024-01-01'},  # $200/share cost
           {'identifier': 'VXUS', 'tax_lot_id': 'VXUS_lot', 'quantity': 200, 
            'cost_basis': 40000, 'date': '2024-01-01'}   # $200/share cost
       ]),
       targets=pd.DataFrame([
           {'asset_class': 'US_Equity', 'identifiers': ['VTI', 'ITOT', 'SCHB'], 
            'target_weight': 0.40},  # US stocks with alternates
           {'asset_class': 'Intl_Equity', 'identifiers': ['VXUS', 'IXUS'], 
            'target_weight': 0.20},  # Int'l with alternate
           {'asset_class': 'CASH', 'identifiers': ['CASH'], 'target_weight': 0.0}
       ]),
       prices=pd.DataFrame([
           {'identifier': 'VTI', 'price': 180.0},    # Down 10%
           {'identifier': 'ITOT', 'price': 180.0},
           {'identifier': 'SCHB', 'price': 180.0},
           {'identifier': 'VXUS', 'price': 170.0},   # Down 15%
           {'identifier': 'IXUS', 'price': 170.0},
           {'identifier': 'CASH', 'price': 1.0}
       ]),
       optimization_type=OracleOptimizationType.PAIRS_TLH
   )
   
   # Will sell VTI and VXUS at a loss and buy their alternates
   status, should_trade, trade_summary, trades = strategy.compute_optimal_trades(
       weight_tax=0.0,
       weight_drift=1.0,
       weight_transaction=0.0,
       weight_cash_drag=1.0,
       should_tlh=True,
       tlh_min_loss_threshold=0.03,  # 3% loss threshold
       range_min_weight_multiplier=0.5,
       range_max_weight_multiplier=2.0,
       rank_penalty_factor=0.5  # Prefer earlier alternates in the identifiers list
   )

DIRECT_INDEX
------------
The ``DIRECT_INDEX`` type supports direct indexing with factor model considerations:

.. code-block:: python

   # Example: Create a DIRECT_INDEX strategy with factor model
   # Generate factor model data
   num_stocks = 100
   num_factors = 5
   identifiers = [f'STOCK{i+1}' for i in range(num_stocks)]
   
   # Create factor model DataFrame
   factor_data = []
   for stock in identifiers:
       exposures = np.random.normal(0, 1, num_factors)
       factor_data.append({
           'identifier': stock,
           'factor1': exposures[0],
           'factor2': exposures[1],
           'factor3': exposures[2],
           'factor4': exposures[3],
           'factor5': exposures[4]
       })
   # Add cash with zero factor exposures
   factor_data.append({
       'identifier': 'CASH',
       'factor1': 0, 'factor2': 0, 'factor3': 0, 'factor4': 0, 'factor5': 0
   })
   factor_model = pd.DataFrame(factor_data)
   
   strategy = OracleStrategy(
       tax_lots=pd.DataFrame([
           {'identifier': stock, 'tax_lot_id': f'{stock}_lot', 
            'quantity': 100, 'cost_basis': 10000, 'date': '2024-01-01'}
           for stock in identifiers
       ]),
       targets=pd.DataFrame([
           {'identifiers': [stock], 'target_weight': 1.0/num_stocks, 
            'asset_class': stock} for stock in identifiers
       ] + [{'identifiers': ['CASH'], 'target_weight': 0, 'asset_class': 'CASH'}]),
       prices=pd.DataFrame([
           {'identifier': stock, 
            'price': 80.0 if i < 5 else 120.0 if i < 10 else 100.0}
           for i, stock in enumerate(identifiers)
       ] + [{'identifier': 'CASH', 'price': 1.0}]),
       factor_model=factor_model,
       optimization_type=OracleOptimizationType.DIRECT_INDEX
   )
   
   # Will optimize for both tax-loss harvesting and factor model alignment
   status, should_trade, trade_summary, trades = strategy.compute_optimal_trades(
       weight_tax=1.0,
       weight_drift=1.0,
       weight_transaction=1.0,
       weight_factor_model=100.0,  # High weight on factor model alignment
       weight_cash_drag=1.0,
       should_tlh=True,
       tlh_min_loss_threshold=0.015,
       range_min_weight_multiplier=0.5,
       range_max_weight_multiplier=2.0
   ) 