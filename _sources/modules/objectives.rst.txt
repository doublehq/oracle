Objectives
==========

.. currentmodule:: service.objectives

The objectives module handles all optimization objective terms used in Oracle's portfolio optimization.

ObjectiveManager
----------------

.. autoclass:: service.objectives.objective_manager.ObjectiveManager
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Objective Components
--------------------

Tax Impact
~~~~~~~~~~
.. automodule:: service.objectives.taxes.tax_optimization
   :members:
   :undoc-members:

Tax Loss Harvesting
~~~~~~~~~~~~~~~~~~~
.. automodule:: service.objectives.taxes.tlh
   :members:
   :undoc-members:

Drift Impact
~~~~~~~~~~~~
.. automodule:: service.objectives.drift.drift_optimization
   :members:
   :undoc-members:

Transaction Costs
~~~~~~~~~~~~~~~~~
.. automodule:: service.objectives.transaction_costs.transaction_optimization
   :members:
   :undoc-members:

Factor Model Impact
~~~~~~~~~~~~~~~~~~~
.. automodule:: service.objectives.factor_model.factor_model_optimization
   :members:
   :undoc-members:

Cash Deployment
~~~~~~~~~~~~~~~
.. automodule:: service.objectives.cash_deployment.cash_deployment
   :members:
   :undoc-members:

Normalization Constants
-----------------------

The following normalization constants are used to scale different objective components:

.. code-block:: python

   TAX_NORMALIZATION = 800  # Scale tax impact
   DRIFT_NORMALIZATION = 1.0  # Scale drift impact
   TRANSACTION_NORMALIZATION = 1.0  # Scale transaction costs
   FACTOR_MODEL_NORMALIZATION = 1.0  # Scale factor model impact
   CASH_DRAG_NORMALIZATION = 1.0  # Scale cash drag impact 