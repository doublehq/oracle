Constraints
===========

.. currentmodule:: service.constraints

The constraints module handles all optimization constraints used in Oracle's portfolio optimization.

ConstraintsManager
------------------

.. autoclass:: service.constraints.constraints_manager.ConstraintsManager
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Constraint Types
----------------

Cash Constraints
~~~~~~~~~~~~~~~~
- Minimum cash balance maintenance
- Cash flow validation for trades
- Withdrawal requirements
- Non-negative cash position enforcement

Trade Constraints
~~~~~~~~~~~~~~~~~
- Minimum notional amount for trades
- No simultaneous buys/sells of same security
- Buy-only or sell-only restrictions based on strategy type
- Trade size rounding requirements
- Maximum position size limits

Holding Time Constraints
~~~~~~~~~~~~~~~~~~~~~~~~~
- Minimum holding period enforcement
- Tax lot sale restrictions
- Tax-aware trading constraints
- Wash sale prevention rules

Stock Restrictions
~~~~~~~~~~~~~~~~~~
- Security-specific trading restrictions
- Asset class constraints
- Position limits
- Concentration limits

Factor Model Constraints
~~~~~~~~~~~~~~~~~~~~~~~~
- Factor exposure limits
- Tracking error constraints
- Risk model alignment
- Portfolio characteristic constraints

Optimization Types
------------------

.. autoclass:: service.helpers.enums.OracleOptimizationType
   :members:
   :undoc-members:
   :show-inheritance:

Each optimization type enforces different constraints:

- **HOLD**: No trading allowed
- **BUY_ONLY**: Only buy trades permitted
- **TAX_UNAWARE**: No tax-specific constraints
- **TAX_AWARE**: Full tax awareness with wash sale prevention
- **PAIRS_TLH**: Tax loss harvesting with paired replacements
- **DIRECT_INDEX**: Factor model based optimization 