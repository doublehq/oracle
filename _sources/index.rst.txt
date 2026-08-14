.. Oracle Optimizer documentation master file, created by
   sphinx-quickstart on Tue May 20 17:11:36 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Oracle Optimizer
================================

Overview
--------

Oracle, built by `Double Finance <https://double.finance>`_, is a sophisticated portfolio optimization engine that tax loss harvests and intelligently rebalances portfolios while considering tax implications, trading costs, and various constraints placed upon portfolios (holding time restrictions, etc). It's designed to help investors maintain their target asset allocations while minimizing tax burdens and transaction costs.

We use this in production at `Double Finance <https://double.finance>`_ for daily Tax Loss Harvesting and Automated Rebalancing. Oracle simply returns optimal trades to make for a portfolio - it does not execute anything. You must bring your own price and target/index data. Oracle does not fetch anything from the internet. It allows for customization and control regarding how aggressive you want to be along pretty much every axis.

It provides the basics of a complete portfolio optimization engine similar to:

* Rowboat Advisors (Acquired by Betterment May 2025)
* `Smartleaf <https://www.smartleaf.com/>`_
* `AlphaThena <https://alphathena.com/>`_
* `BlackRock's Tax-Managed Equity SMA by Aperio <https://www.blackrock.com/us/financial-professionals/investments/tax-managed-equity-sma-aperio>`_
* `AdvisorArch (Acquired by Apex 2024) <https://apexfintechsolutions.com/news-resources/press-releases/apex-fintech-solutions-announces-acquisition-of-advisorarch-bringing-a-suite-of-technology-driven-portfolio-management-solutions/>`_
* `Parti Pris <https://partipris-invest.com/>`_

Features
--------

* **Tax Loss Harvesting**: Ability to specify a threshold at which to TLH a given security, while respecting wash sale rules.
* **Tax-Aware Rebalancing**: Optimizes trades to minimize tax impact while reducing drift
* **Multi-Asset Support**: Can handle any asset type (stocks, etfs, mutual funds, crypto)
* **Consider Trading Costs**: Considers spreads/trading costs while recommending trades.
* **Factor Model Rebalancing Integration**: Supports considering a factor model for use the Direct Indexed based TLH
* **Tax Lot Management**: Tracks and optimizes individual tax lots.
* **Wash Sale Prevention**: Built-in protection against wash sales
* **Stock Restrictions**: Can restrict buying/selling of specific securities
* **Holding Time Restrictions**: Enforces minimum holding periods before selling tax lots

Not currently supported
----------------------

* **ESG Tilts**: No support for ESG scoring or ESG-based portfolio tilts
* **Household Level Optimization**: Cannot optimize across multiple accounts in a household
* **Sector Constraints**: No support for sector exposure limits or constraints

Documentation Contents
--------------------

.. toctree::
   :maxdepth: 4
   :caption: Contents:
   :glob:
   :hidden:

   self
   modules/initializers
   modules/optimization_types
   modules/oracle
   modules/oracle_strategy
   modules/objectives
   modules/constraints
   modules/helpers

Installation
-----------

.. code-block:: bash

   # Install CBC solver (required for optimization)
   brew install coin-or-tools/coinor/cbc

   # Set up Python environment
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

Quick Start Guide
---------------

Initialize an Oracle Strategy
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import pandas as pd
   from service.oracle import Oracle
   from service.oracle_strategy import OracleStrategy
   from service.helpers.enums import OracleOptimizationType

    current_date = date(2024, 4, 20)
    
    # Create tax rates data
    tax_rates = pd.DataFrame([
        {
            'gain_type': 'short_term',
            'federal_rate': 0.35,
            'state_rate': 0.06,
            'total_rate': 0.41
        },
        {
            'gain_type': 'long_term',
            'federal_rate': 0.20,
            'state_rate': 0.06,
            'total_rate': 0.26
        },
        {
            "gain_type": "qualified_dividend",
            "federal_rate": 0.15,
            "state_rate": 0.06,
            "total_rate": 0.21
        }   
    ])

    # Define test data based on documentation example
    targets_df = pd.DataFrame({
        'asset_class': ['STOCK_A', 'STOCK_B', 'CASH'],
        'target_weight': [0.4, 0.4, 0.2],
        'identifiers': [['STOCK_A'], ['STOCK_B'], [CASH_CUSIP_ID]]
    })

    tax_lots_df = pd.DataFrame({
        'tax_lot_id': ['lot_a1', 'lot_b1'],
        'identifier': ['STOCK_A', 'STOCK_B'],
        'quantity': [100, 100],
        'cost_basis': [100, 100],
        'date': ['2024-01-01', '2024-01-01']
    })

    prices_df = pd.DataFrame([
        {
            'identifier': 'STOCK_A',
            'price': 100.0,
        },
        {
            'identifier': 'STOCK_B',
            'price': 100.0,
        },
        {
            'identifier': CASH_CUSIP_ID,
            'price': 1.0,
        }
    ])

    spreads_df = pd.DataFrame([
        {
            'identifier': 'STOCK_A',
            'spread': 0.001  # 10 basis points
        },
        {
            'identifier': 'STOCK_B',
            'spread': 0.001  # 10 basis points
        },
        {
            'identifier': CASH_CUSIP_ID,
            'spread': 0.0  # No spread for cash
        }
    ])

    # Create Oracle instance
    oracle = Oracle(
        current_date=current_date,
        recently_closed_lots=pd.DataFrame(),
        stock_restrictions=pd.DataFrame(),
        tax_rates=tax_rates
    )

    # Create and configure strategy
    strategy = OracleStrategy(
        strategy_id="STRATEGY_1",
        tax_lots=tax_lots_df,
        prices=prices_df,
        cash=10000.0,
        targets=targets_df,
        asset_class_targets=None,
        spreads=spreads_df,
        factor_model=None,
        optimization_type=OracleOptimizationType.TAX_AWARE,
        deminimus_cash_target=0.0001,
        withdrawal_amount=0.0,
        enforce_wash_sale_prevention=True
    )

    strategy.set_oracle(oracle)
    oracle.strategies = [strategy]
    oracle.initialize_wash_sale_restrictions(percentage_protection_from_inadvertent_wash_sales=0.003)

Run the Optimizer
^^^^^^^^^^^^^^^

.. code-block:: python

   # Compute optimal trades
   results, netted_trades = oracle.compute_optimal_trades_for_all_strategies(
       settings={
           "strategies": {
               "STRATEGY_1": {
                   "weight_tax": 1.0,
                   "weight_drift": 1.0,
                   "weight_transaction": 1.0,
                   "weight_factor_model": 0.0,
                   "weight_cash_drag": 0.0,
                   "rebalance_threshold": 0.001,
                   "buy_threshold": 0.0005,
                   "holding_time_days": 0,
                   "should_tlh": True,
                   "tlh_min_loss_threshold": 0.015,
                   "range_min_weight_multiplier": 0.5,
                   "range_max_weight_multiplier": 2.0,
                   "min_notional": 0,
                   "rank_penalty_factor": 0.0,
                   "trade_rounding": 4
               }
           }
       }
   )

Response Format
^^^^^^^^^^^^^

The ``compute_optimal_trades_for_all_strategies`` method returns a tuple of ``(results, netted_trades)``:

The return value consists of two components:

1. ``results`` - A dictionary with strategy IDs as keys
2. ``netted_trades`` - A pandas DataFrame containing the final trades

Results Dictionary Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    {
        'STRATEGY_1': (
            1,  # Status code
            True,  # Success flag
            {
                'drift': {
                    'average_drift_improvement': 0.08755555555555557,
                    'max_drift_improvement': 0.1313333333333333,
                    'median_drift_improvement': 0.0656666666666667,
                    'overall_drift_reduction': 0.985,
                    'positions_improved': 3,
                    'positions_unchanged': 0,
                    'positions_worsened': 0,
                    'post_drift_magnitude': 0.0040000000000000036,
                    'post_num_overweight': 1,
                    'post_num_underweight': 2,
                    'pre_drift_magnitude': 0.2666666666666667,
                    'pre_num_overweight': 1,
                    'pre_num_underweight': 2,
                    'total_drift_improvement': 0.2626666666666667,
                    'weighted_drift_reduction': 0.985
                },
                'execution': {
                    'num_buys': 2,
                    'num_sells': 0,
                    'total_buys_value': 3940.0,
                    'total_sells_value': 0.0,
                    'total_trades': 2,
                    'total_value': 3940.0
                },
                'explanation': 'buying $3,940.00 worth of 2 securities. '
                             'Buying $1,970 STOCK_B, $1,970 STOCK_A. drift '
                             'cost got better by 4.1313, spread costs got '
                             'worse by 0.1576\n'
                             'Overall, these trades will improve the '
                             'portfolio by 3.97. pre_strategy_cash: 10000.0 '
                             'min_cash: 5850.0 post_strategy_cash: 6060.0',
                'factor_model': {},
                'gain_loss': {
                    'long_term_gains': 0.0,
                    'long_term_losses': 0.0,
                    'net_gain_loss': 0.0,
                    'short_term_gains': 0.0,
                    'short_term_losses': 0.0,
                    'total_gains': 0.0,
                    'total_losses': 0.0
                },
                'optimization_info': {
                    'after_optimization': {
                        'cash_drag': 0,
                        'drift_cost': 0.002,
                        'factor_cost': 0,
                        'overall': 0.1596,
                        'spread_costs': 0.1576,
                        'tax_cost': 0.0
                    },
                    'before_optimization': {
                        'cash_drag': 0,
                        'drift_cost': 4.1333334,
                        'factor_cost': 0,
                        'overall': 4.1333334,
                        'spread_costs': 0.0,
                        'tax_cost': 0.0
                    },
                    'component_improvements': {
                        'cash_drag': {'absolute': 0, 'percent': 0},
                        'drift_cost': {'absolute': 4.1313334, 'percent': 99.95161290400625},
                        'factor_cost': {'absolute': 0, 'percent': 0},
                        'overall': {'absolute': 3.9737333999999995, 'percent': 96.13870973969823},
                        'spread_costs': {'absolute': -0.1576, 'percent': 0},
                        'tax_cost': {'absolute': 0.0, 'percent': 0}
                    }
                }
            }
        )
    }

The results dictionary contains:

* **Status code** (``1``): Indicates the optimization status
* **Success flag** (``True/False``): Whether the optimization succeeded
* **Detailed results dictionary** containing:

  * ``drift``: Metrics about portfolio drift improvements
  * ``execution``: Summary of trades (counts and values)
  * ``explanation``: Human-readable description of the changes
  * ``factor_model``: Factor model related data (if used)
  * ``gain_loss``: Tax implications of the trades
  * ``optimization_info``: Detailed metrics before and after optimization

Netted Trades DataFrame
~~~~~~~~~~~~~~~~~~~~~

The netted trades DataFrame shows the actual trades to be executed:

.. code-block:: python

    identifier action  quantity  price tax_lot_id  short_term_gain  short_term_loss  long_term_gain  long_term_loss
    STOCK_A    buy      19.7  100.0        NaN             NaN             NaN            NaN            NaN
    STOCK_B    buy      19.7  100.0        NaN             NaN             NaN            NaN            NaN

Each row represents a single trade with the following columns:

* ``identifier``: The security identifier
* ``action``: The trade action (buy/sell)
* ``quantity``: Number of units to trade
* ``price``: Trade price
* ``tax_lot_id``: Associated tax lot (if applicable)
* ``short_term_gain``: Short-term capital gains (if applicable)
* ``short_term_loss``: Short-term capital losses (if applicable)
* ``long_term_gain``: Long-term capital gains (if applicable)
* ``long_term_loss``: Long-term capital losses (if applicable)

For detailed information about optimization types, parameters, constraints, and more, please refer to the specific sections in the documentation.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
