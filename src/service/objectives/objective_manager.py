import pulp
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
import time

from src.service.objectives.drift.drift_optimization import calculate_drift_impact_vectorized
from src.service.objectives.cash_deployment.cash_deployment import calculate_cash_deployment_objective, calculate_max_withdrawal_objective
from src.service.objectives.taxes.tax_optimization import calculate_tax_impact, get_tax_cost
from src.service.objectives.transaction_costs.transaction_optimization import calculate_transaction_costs
from src.service.objectives.factor_model.factor_model_optimization import calculate_factor_model_impact_vectorized
from src.service.helpers.constants import CASH_CUSIP_ID
from src.service.helpers.enums import OracleOptimizationType

from src.service.helpers.constants import logger

class ObjectiveManager:
    """Manager class for handling all optimization objective terms."""
    
    def __init__(self, oracle_strategy):
        """
        Initialize ObjectiveManager with reference to OracleStrategy.
        
        Args:
            oracle_strategy: Reference to the OracleStrategy instance
        """
        self.strategy = oracle_strategy
        self.objective_components = {}  # Store individual objective components
        
        
    def calculate_objectives(
        self,
        buys: Dict[str, pulp.LpVariable],
        sells: Dict[str, pulp.LpVariable],
        drift: pd.DataFrame,
        gain_loss: pd.DataFrame,
        total_value: float,
        weight_tax: float = 1.0,
        weight_drift: float = 1.0,
        weight_transaction: float = 1.0,
        weight_factor_model: float = 0.0,
        weight_cash_drag: float = 0.0,
        rank_penalty_factor: float = 0.00,
        buy_df: Optional[pd.DataFrame] = None,
        sell_df: Optional[pd.DataFrame] = None,
        enforce_wash_sale_prevention: bool = True,
        debug: bool = True,
        log_time: bool = False
    ) -> pulp.LpAffineExpression:
        """
        Calculate all objective terms and return the combined objective.
        
        Args:
            buys: Dictionary of buy variables
            sells: Dictionary of sell variables
            drift: DataFrame with drift report
            tax_lot_drift: DataFrame with tax lot level drift
            gain_loss: DataFrame with gain/loss report
            total_value: Total portfolio value
            weight_tax: Weight for tax component
            weight_drift: Weight for drift component
            weight_transaction: Weight for transaction cost component
            weight_factor_model: Weight for factor model component
            weight_cash_drag: Weight for cash deployment component
            rank_penalty_factor: Factor to control the strength of rank penalty (default 0.05)
            debug: Enable debug logging
            log_time: Whether to log timing information
            
        Returns:
            Combined objective function expression
        """
        timing_data = {}
        if log_time:
            start_time = time.time()

        # Apply strategy-specific weight adjustments
        weight_tax, weight_drift, weight_transaction, weight_factor_model, weight_cash_drag = \
            self.strategy.optimization_type.adjust_weights(
                weight_tax, weight_drift, weight_transaction, weight_factor_model, weight_cash_drag
            )
            
        objective_terms = []
        # Reset objective components for this calculation
        self.objective_components = {}
        
        # Calculate tax impact
        if weight_tax > 0:
            if log_time:
                tax_start = time.time()
            tax_impact = calculate_tax_impact(
                prob=self.strategy.optimization_problem,
                sells=sells,
                gain_loss=gain_loss,
                total_value=self.strategy.total_value(),
                tax_normalization=self.strategy.TAX_NORMALIZATION * weight_tax,
                enforce_wash_sale_prevention=enforce_wash_sale_prevention,
            )
            objective_terms.append(tax_impact)
            self.objective_components['tax'] = tax_impact
            if log_time:
                timing_data['tax_calculation'] = time.time() - tax_start
            
        # Calculate drift impact
        if weight_drift > 0:
            if log_time:
                drift_start = time.time()
            drift_impact  = calculate_drift_impact_vectorized(
                prob=self.strategy.optimization_problem,
                buy_df=buy_df,
                sell_df=sell_df,
                drift=drift,
                total_value=total_value,
                debug=debug,
                absolute_drift_normalization=self.strategy.DRIFT_NORMALIZATION * weight_drift,
                rank_penalty_factor=rank_penalty_factor,
            )
            objective_terms.append(drift_impact)
            self.objective_components['drift'] = drift_impact
            if log_time:
                timing_data['drift_calculation'] = time.time() - drift_start
            
        # Calculate transaction costs
        if weight_transaction > 0:
            if log_time:
                transaction_start = time.time()

            transaction_impact = calculate_transaction_costs(
                buys=buys,
                sells=sells,
                total_value=self.strategy.total_value(),
                spreads=self.strategy.spreads,
                transaction_normalization=self.strategy.TRANSACTION_NORMALIZATION * weight_transaction
            )
            objective_terms.append(transaction_impact)
            self.objective_components['transaction'] = transaction_impact
            if log_time:
                timing_data['transaction_calculation'] = time.time() - transaction_start
            
        # Calculate factor model impact
        if weight_factor_model > 0 and self.strategy.factor_model is not None:
            if log_time:
                factor_start = time.time()
            factor_impact = calculate_factor_model_impact_vectorized(
                prob=self.strategy.optimization_problem,
                buy_df=buy_df,
                sell_df=sell_df,
                total_value=total_value,
                factor_model=self.strategy.factor_model,
                target_factors=self.strategy.factor_model_target,
                actual_factors=self.strategy.factor_model_actual,
                debug=debug,
                factor_normalization=self.strategy.FACTOR_MODEL_NORMALIZATION * weight_factor_model,
                use_piecewise=True
            )
            objective_terms.append(factor_impact)
            self.objective_components['factor_model'] = factor_impact
            if log_time:
                timing_data['factor_model_calculation'] = time.time() - factor_start
            
        # Calculate cash deployment impact (only if no withdrawal)
        if weight_cash_drag > 0 and self.strategy.withdrawal_amount <= 0:
            if log_time:
                cash_start = time.time()
            cash_impact = self.calculate_cash_deployment_objective(
                prob=self.strategy.optimization_problem,
                buys=buys,
                sells=sells,
                drift=drift,
                total_value=total_value,
                factor_normalization=self.strategy.CASH_DRAG_NORMALIZATION * weight_cash_drag,
                debug=debug
            )
            objective_terms.append(cash_impact)
            self.objective_components['cash_drag'] = cash_impact
            if log_time:
                timing_data['cash_drag_calculation'] = time.time() - cash_start
            
        # Combine all terms
        combined_objective = pulp.lpSum(objective_terms)
        self.objective_components['total'] = combined_objective

        if log_time:
            timing_data['total_time'] = time.time() - start_time
            logger.info("=== Objective Calculation Timing Breakdown ===")
            for section, duration in timing_data.items():
                logger.info(f"{section}: {duration:.3f} seconds")

        return combined_objective
        
    def calculate_cash_deployment_objective(
        self,
        prob: pulp.LpProblem,
        buys: Dict[str, pulp.LpVariable],
        sells: Dict[str, pulp.LpVariable],
        drift: pd.DataFrame,
        total_value: float,
        factor_normalization: float,
        debug: bool = True
    ) -> pulp.LpAffineExpression:
        """
        Calculate the cash deployment objective term.
        
        Args:
            prob: The optimization problem to add constraints to
            buy_df: DataFrame with buy variables and prices
            sell_df: DataFrame with sell variables and prices
            drift: DataFrame with drift report
            total_value: Total portfolio value
            debug: Enable debug logging
            
        Returns:
            Cash deployment objective term
        """
        if debug:
            logger.info("Calculating cash deployment objective")
            
        if self.strategy.withdrawal_amount > 0:
            return 0  # No cash deployment objective when withdrawing
            
        # Call the existing cash deployment calculation
        cash_impact = calculate_cash_deployment_objective(
            prob=prob,
            buys=buys,
            sells=sells,
            drift=drift,
            gain_loss=self.strategy.gain_loss_report,
            total_value=total_value,
            prices=self.strategy.prices,
            cash_normalization=factor_normalization,
            debug=debug
        )
        
        # Apply normalization
        return cash_impact
    
    def calculate_max_withdrawal_objective(
        self,
        prob: pulp.LpProblem,
        buys: Dict[str, pulp.LpVariable],
        sells: Dict[str, pulp.LpVariable],
        gain_loss: pd.DataFrame,
        debug: bool = True
    ) -> pulp.LpAffineExpression:
        """
        Calculate an objective that maximizes cash withdrawal.
        Creates an objective that maximizes the cash generated from selling securities.
        
        Args:
            prob: PuLP optimization problem
            buys: Dictionary of buy variables
            sells: Dictionary of sell variables
            gain_loss: DataFrame with gain/loss report
            debug: Enable debug logging
            
        Returns:
            Cash withdrawal maximization objective term (negative since we're minimizing)
        """
        
        return calculate_max_withdrawal_objective(
            self=self,
            prob=prob,
            buys=buys,
            sells=sells,
            gain_loss=gain_loss,
            debug=debug
        )

    def extract_component_values(self, prob: pulp.LpProblem) -> dict:
        """
        Extract the values of each objective component from a solved problem.
        
        Args:
            prob: A solved PuLP problem
            
        Returns:
            Dictionary containing component values
        """
        component_values = {}
        for component_name, component_expr in self.objective_components.items():
            try:
                if isinstance(component_expr, pulp.LpAffineExpression):
                    # Calculate the value of this component with the current variable values
                    component_value = pulp.value(component_expr)
                    component_values[component_name] = component_value
                else:
                    # Handle non-expression components
                    logger.warning(f"Component {component_name} is not an LpAffineExpression")
                    component_values[component_name] = None
            except Exception as e:
                logger.warning(f"Could not evaluate component {component_name}: {str(e)}")
                component_values[component_name] = None
        
        # Compare total with actual optimization value
        actual_obj_value = pulp.value(prob.objective)
        computed_total = component_values.get('total')
        if computed_total is not None and abs(actual_obj_value - computed_total) > 1e-6:
            logger.warning(f"Discrepancy detected between optimization value ({actual_obj_value}) "
                         f"and sum of components ({computed_total})")
            logger.warning("Component breakdown:")
            for name, value in component_values.items():
                if name != 'total':
                    logger.warning(f"  {name}: {value}")
                
        return component_values 


