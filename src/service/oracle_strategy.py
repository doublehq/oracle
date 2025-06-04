import pandas as pd
from datetime import date, datetime
from functools import cached_property
from typing import Optional, Dict, Tuple, List
import pulp
from datetime import timedelta
import json
import os
from pathlib import Path
from src.service.helpers.constants import logger
import numpy as np
import time

from src.service.initializers import (
    initialize_tax_lots,
    initialize_targets,
    initialize_prices,
    initialize_spreads,
    initialize_factor_model,
)
from src.service.helpers.constants import CASH_CUSIP_ID
from src.service.helpers.trade_extractor import extract_trades
from src.service.constraints import ConstraintsManager
from src.service.objectives import factor_model
from src.service.objectives.objective_manager import ObjectiveManager
from src.service.helpers import calculate_max_withdrawal
from src.service.reports import (
    generate_gain_loss_report,
    generate_actuals_report,
    generate_drift_report,
)
from src.service.reports.comparison_report import (
    generate_drift_comparison_report,
    generate_factor_model_comparison_report
)
from src.service.helpers.create_decision_vars import create_decision_variables
from src.service.helpers.enums import OracleOptimizationType
from src.service.objectives.taxes.tlh import calculate_tlh_impact, TLHTrade
from src.service.helpers.trade_applier import apply_trades_to_portfolio
from src.service.helpers.trade_summary import generate_trade_summary_from_strategies


class OracleStrategy:
    """
    A sophisticated portfolio optimization strategy class that handles various types of portfolio rebalancing.

    This class implements complex portfolio optimization strategies including tax-aware rebalancing,
    tax-loss harvesting, direct indexing, and withdrawal optimization. It uses linear programming
    to find optimal trade solutions while considering multiple objectives and constraints.

    Class Attributes:
        TAX_NORMALIZATION (float): Normalization factor for tax objective (800)
        DRIFT_NORMALIZATION (float): Normalization factor for drift objective (100)
        TRANSACTION_NORMALIZATION (float): Normalization factor for transaction costs (1200)
        FACTOR_MODEL_NORMALIZATION (float): Normalization factor for factor model objective (60)
        CASH_DRAG_NORMALIZATION (float): Normalization factor for cash deployment objective (50)
        DEMINIMUS_CASH_TARGET_PERCENT (float): Minimum possible cash target (0.03%)
        _next_strategy_id (int): Counter for generating unique strategy IDs
    """
    # Add normalization multipliers as class constants
    TAX_NORMALIZATION = 800
    DRIFT_NORMALIZATION = 100
    TRANSACTION_NORMALIZATION = 1200
    FACTOR_MODEL_NORMALIZATION = 60
    CASH_DRAG_NORMALIZATION = 50 # Multiplier for the cash deployment objective term
    DEMINIMUS_CASH_TARGET_PERCENT = 0.03 / 100 # 30 Basis Points for the lowest possible cash target
    
    # Counter for generating unique strategy IDs
    _next_strategy_id = 1

    def __init__(
        self, 
        tax_lots: pd.DataFrame,        
        prices: pd.DataFrame,
        cash: float,
        targets: Optional[pd.DataFrame] = None,
        asset_class_targets: Optional[pd.DataFrame] = None,
        spreads: Optional[pd.DataFrame] = None,
        factor_model: Optional[pd.DataFrame] = None,
        strategy_id: Optional[int] = None,
        optimization_type: OracleOptimizationType | str = OracleOptimizationType.TAX_AWARE,
        deminimus_cash_target: float = 0.0,
        withdrawal_amount: float = 0.0,
        enforce_wash_sale_prevention: bool = True,
    ):
        """
        Initialize an OracleStrategy instance with portfolio data and optimization parameters.
        
        Args:
            tax_lots (pd.DataFrame): Current portfolio tax lots with columns including 'identifier',
                'tax_lot_id', 'quantity', 'cost_basis', and 'date'
            prices (pd.DataFrame): Current security prices with columns 'identifier' and 'price'
            cash (float): Current portfolio cash balance
            targets (pd.DataFrame, optional): Target portfolio weights with columns 'identifier',
                'target_weight', and 'identifiers' (list of equivalent securities)
            asset_class_targets (pd.DataFrame, optional): Asset class level targets, required if
                targets is not provided
            spreads (pd.DataFrame, optional): Bid-ask spreads for securities
            factor_model (pd.DataFrame, optional): Factor exposures for direct indexing,
                required if optimization_type is DIRECT_INDEX
            strategy_id (int, optional): Unique identifier for the strategy. Auto-generated if None
            optimization_type (OracleOptimizationType | str): Type of optimization strategy to use
            deminimus_cash_target (float): Minimum cash percentage target (default: 0.0)
            withdrawal_amount (float): Amount to withdraw from portfolio (default: 0.0)
            enforce_wash_sale_prevention (bool): Whether to enforce wash sale rules (default: True)

        Raises:
            ValueError: If neither targets nor asset_class_targets is provided, or if factor_model
                is required but not provided
            TypeError: If optimization_type is not a valid OracleOptimizationType or string
        """
        # Initialize basic data first
        self.oracle = None  # Will be set by Oracle when strategy is added
        self.cash = cash
        self.withdrawal_amount = max(0, withdrawal_amount)  # Ensure non-negative
        self.deminimus_cash_target = deminimus_cash_target
        self.enforce_wash_sale_prevention = enforce_wash_sale_prevention
        
        # Ensure optimization_type is the correct Enum type
        if isinstance(optimization_type, str):
            try:
                self.optimization_type = OracleOptimizationType.from_string(optimization_type)
            except ValueError as e:
                raise ValueError(f"Invalid optimization_type string provided: {optimization_type}") from e
        elif isinstance(optimization_type, OracleOptimizationType):
            self.optimization_type = optimization_type
        else:
            raise TypeError(f"optimization_type must be an OracleOptimizationType enum or a valid string, not {type(optimization_type)}")
        
        # Validate that at least one of targets or asset_class_targets is provided
        if targets is None and asset_class_targets is None:
            raise ValueError("At least one of targets or asset_class_targets must be provided.")            
        
        # Generate unique strategy_id if not provided
        if strategy_id is None:
            strategy_id = OracleStrategy._next_strategy_id
            OracleStrategy._next_strategy_id += 1
        self.strategy_id = strategy_id
        
        self.tax_lots = initialize_tax_lots(tax_lots)
        #Create targets assuming we do not have a withdrawal amount
        # If we have a withdrawal amount, we need to re-create the targets with the new cash target later.
        self.targets = initialize_targets(targets, 0, deminimus_cash_target)
        # Get all identifiers from targets where target_weight > 0
        self.target_identifiers = []
        for _, row in self.targets.iterrows():
            if row['target_weight'] > 0:
                self.target_identifiers.extend([id for id in row['identifiers']])
        self.target_identifiers = list(set(self.target_identifiers))  # Remove duplicates
        self.owned_identifiers = self.tax_lots['identifier'].unique() if not self.tax_lots.empty else []
        self.owned_tax_lots = self.tax_lots['tax_lot_id'].unique() if not self.tax_lots.empty else []
        
        self.all_identifiers = set(self.owned_identifiers).union(set(self.target_identifiers))
        
        # Initialize prices and spreads (which depend on all_identifiers)
        self.prices = initialize_prices(prices, self.all_identifiers)
        self.spreads = initialize_spreads(spreads, self.all_identifiers, self.prices)
        if self.optimization_type == OracleOptimizationType.DIRECT_INDEX:
            if factor_model is None or factor_model.empty:
                raise ValueError("Factor model is required for DIRECT_INDEX optimization type.")
            self.factor_model, self.factor_model_target, self.factor_model_actual = initialize_factor_model(factor_model, self.targets, self.actuals)
        else:
            self.factor_model = None
            self.factor_model_target = None
        
        
        # Initialize constraints manager
        self.constraints_manager = ConstraintsManager(self)
        
        # Initialize objective manager
        self.objective_manager = ObjectiveManager(self)
        
        # Get the appropriate optimization strategy
        # Note: optimization_type already stores this information, no need for a separate attribute
        
        # Validate withdrawal amount
        if self.withdrawal_amount > 0:
            total_portfolio_value = self.total_value()
            remaining_portfolio_value = abs(total_portfolio_value - self.withdrawal_amount)
            if 0 < remaining_portfolio_value < 0.00001:
                self.withdrawal_amount = total_portfolio_value
            if self.withdrawal_amount > total_portfolio_value:
                raise ValueError(f"Withdrawal amount (${self.withdrawal_amount:.2f}) exceeds total portfolio value (${total_portfolio_value:.2f})")
            #Here we need to re-create the targets with the new cash target
            withdraw_target = min(1.0, withdrawal_amount / total_portfolio_value) if total_portfolio_value > 0 else 0
            self.targets = initialize_targets(targets, withdraw_target, deminimus_cash_target)

        self.optimization_problem = None  # Will be set when optimization is performed
    
    def set_oracle(self, oracle) -> None:
        """
        Set the Oracle reference for this strategy.

        This method is called by Oracle when the strategy is added to establish
        a bidirectional relationship between Oracle and Strategy instances.

        Args:
            oracle: The Oracle instance this strategy belongs to

        Raises:
            ValueError: If attempting to access Oracle reference before it's set
        """
        self.oracle = oracle
        
    @property
    def current_date(self) -> date:
        """
        Get the current date from the Oracle instance.

        Returns:
            date: The current date set in the Oracle instance

        Raises:
            ValueError: If Oracle reference is not set
        """
        if self.oracle is None:
            raise ValueError("Oracle reference not set")
        return self.oracle.current_date
    
    @cached_property
    def gain_loss_report(self) -> pd.DataFrame:
        """
        Generate a gain/loss report for all tax lots using current prices.

        This cached property calculates realized and unrealized gains/losses
        for all tax lots in the portfolio based on current market prices.

        Returns:
            pd.DataFrame: DataFrame containing gain/loss information for each tax lot
                including columns for cost basis, market value, and gain/loss amounts
        """
        return generate_gain_loss_report(
            tax_lots=self.tax_lots,
            prices=self.prices,
            current_date=self.current_date,
            tax_rates=self.oracle.tax_rates
        )
    

    @cached_property
    def actuals(self) -> pd.DataFrame:
        """
        Calculate actual portfolio weights based on current tax lots and prices.

        This cached property computes the current portfolio allocation including
        both security positions and cash, with weights calculated as percentages
        of total portfolio value.

        Returns:
            pd.DataFrame: DataFrame containing actual portfolio weights and market values
        """
        return generate_actuals_report(
            tax_lots=self.tax_lots,
            prices=self.prices,
            cash=self.cash
        )
    
    def total_value(self) -> float:
        """
        Calculate the total market value of the portfolio.

        Sums the market value of all positions including cash to determine
        the total portfolio value.

        Returns:
            float: Total portfolio market value in base currency
        """
        return self.actuals['market_value'].sum()
    
    def min_cash_amount(self) -> float:
        """
        Calculate the minimum cash amount required to meet target weights.

        The minimum cash amount is calculated using a hard limit that is:
        1. At least self.deminimus_cash_target (3 bps)
        2. No more than 97.5% of the target cash weight
        3. No more than the current actual cash percentage
        4. Capped at 100%
        5. Rounded to 8 decimal places

        Returns:
            float: Minimum required cash amount in base currency
        """
        total_value = self.total_value()
        
        # Get target cash weight
        cash_target = self.targets[self.targets['asset_class'] == CASH_CUSIP_ID]['target_weight'].iloc[0]
        
        # Get current actual cash percentage
        current_cash_percentage = self.cash / total_value if total_value > 0 else 0
        
        # Calculate hard limit as a percentage (between 0 and 100)
        hard_limit = round(min(100 * max(self.deminimus_cash_target, (float(cash_target) * 0.975)), 100), 8)
        
        # Ensure hard limit doesn't exceed current actual cash percentage
        if hard_limit >= current_cash_percentage * 100:
            hard_limit = current_cash_percentage * 100
            
        # Convert percentage back to decimal for final calculation
        return (hard_limit / 100) * total_value

    @cached_property
    def drift_report(self) -> pd.DataFrame:
        """
        Generate a report comparing current portfolio weights to target weights.

        This cached property calculates the drift between actual and target weights
        for all positions in the portfolio, including cash.

        Returns:
            pd.DataFrame: DataFrame containing drift information including columns
                for actual weights, target weights, and drift amounts
        """
        return generate_drift_report(
            targets=self.targets,
            actuals=self.actuals,
        )

    def _create_decision_variables(
        self,
        buy_identifiers: list[str],
        gain_loss: pd.DataFrame,
        debug: bool = False
    ) -> tuple[dict, dict, dict]:
        """
        Create decision variables for the optimization problem.

        Creates PuLP variables for buy and sell decisions that will be used
        in the optimization problem constraints and objective function.

        Args:
            buy_identifiers (list[str]): List of identifiers that can be bought
            gain_loss (pd.DataFrame): Gain/loss report for current positions
            debug (bool): Whether to print debug information

        Returns:
            tuple[dict, dict, dict]: Tuple containing:
                - Dictionary of buy variables
                - Dictionary of sell variables
                - Dictionary of buy dataframes
        """
        return create_decision_variables(buy_identifiers=buy_identifiers, gain_loss=gain_loss, prices=self.prices, debug=debug)

    def _set_initial_values(
        self,
        buys: Dict[str, pulp.LpVariable],
        sells: Dict[str, pulp.LpVariable],
        gain_loss: pd.DataFrame,
        debug: bool = False
    ) -> None:
        """
        Set initial values for optimization variables based on current portfolio state.

        Provides a warm start for the solver by initializing buy and sell variables
        with reasonable starting values based on the current portfolio state.

        Args:
            buys (Dict[str, pulp.LpVariable]): Dictionary of buy decision variables
            sells (Dict[str, pulp.LpVariable]): Dictionary of sell decision variables
            gain_loss (pd.DataFrame): Gain/loss report for current positions
            debug (bool): Whether to print debug information
        """
        # Initialize all buys to 0 (no initial purchases)
        for identifier, buy_var in buys.items():
            buy_var.setInitialValue(0)
            
        # Initialize sells to 0 quantities (no initial sells)
        for _, lot in gain_loss.iterrows():
            if lot['tax_lot_id'] in sells:
                sells[lot['tax_lot_id']].setInitialValue(0)
        
        if debug:
            logger.info("Set initial values for optimization variables:")
            logger.info(f"  Initialized {len(buys)} buy variables to 0")
            logger.info(f"  Initialized {len(sells)} sell variables to 0")

    def _solve_optimization(
        self,
        prob: pulp.LpProblem,
        debug: bool = False
    ) -> tuple[int, float]:
        """
        Solve the optimization problem using the CBC solver.

        Attempts to solve the given linear programming problem and handles
        various solution statuses and potential failures.

        Args:
            prob (pulp.LpProblem): The PuLP optimization problem to solve
            debug (bool): Whether to print debug information

        Returns:
            tuple[int, float]: Tuple containing:
                - Solution status code (e.g., pulp.LpStatusOptimal)
                - Optimized objective value (or None if optimization failed)
        """
        
        from src.solvers import solve_optimization_problem
        
        status, optimized_value = solve_optimization_problem(prob)
        
        if debug:
            # logger.info for debug steps
            logger.info(f"Solution status: {pulp.LpStatus[status]}")
            logger.info("Checking individual constraints for feasibility...")
            for name, constraint in prob.constraints.items():
                try:
                    constraint_value = pulp.value(constraint)
                    # Log constraint value at debug level
                    logger.debug(f"Constraint {name}: {constraint} -> Value: {constraint_value}") 
                except:
                    # logger.warning if constraint evaluation fails
                    logger.warning(f"Could not evaluate constraint {name}: {constraint}")
        
        if status != pulp.LpStatusOptimal:
            # Use logger.warning for non-optimal solutions
            logger.warning(f"Optimization finished with status: {pulp.LpStatus[status]}")
            return status, None

        return status, optimized_value

    def to_dict(self) -> dict:
        """
        Convert the strategy's state to a dictionary for serialization.

        Converts all strategy data, including DataFrames and complex objects,
        into a JSON-serializable dictionary format for storage or transmission.

        Returns:
            dict: Dictionary containing all strategy data with properly formatted dates
                and serializable values
        """
        # Convert DataFrames to dictionaries with date handling
        tax_lots_dict = self.tax_lots.copy()
        if not tax_lots_dict.empty and 'date' in tax_lots_dict.columns:
            # Handle null dates and ensure proper datetime conversion
            tax_lots_dict['date'] = pd.to_datetime(tax_lots_dict['date'], errors='coerce')
            tax_lots_dict['date'] = tax_lots_dict['date'].dt.strftime('%Y-%m-%d')
        tax_lots_dict = tax_lots_dict.to_dict(orient="records")
        
        return {
            "strategy_id": self.strategy_id,
            "optimization_type": self.optimization_type,
            "cash": self.cash,
            "withdrawal_amount": self.withdrawal_amount,
            "tax_lots": tax_lots_dict,
            "targets": self.targets.to_dict(orient="records"),
            "prices": self.prices.to_dict(orient="records"),
            "spreads": self.spreads.to_dict(orient="records") if self.spreads is not None else None
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'OracleStrategy':
        """
        Create a new OracleStrategy instance from a dictionary.

        Factory method to reconstruct an OracleStrategy instance from a serialized
        dictionary format, typically created by to_dict().

        Args:
            data (dict): Dictionary containing strategy data

        Returns:
            OracleStrategy: New instance initialized with the provided data

        Raises:
            ValueError: If required data is missing or invalid
        """
        # Convert date strings back to datetime
        tax_lots_df = pd.DataFrame(data["tax_lots"])
        if not tax_lots_df.empty:
            tax_lots_df['date'] = pd.to_datetime(tax_lots_df['date'])
        
        # Get optimization type from string, default to TAX_AWARE if not present
        optimization_type_str = data.get("optimization_type", OracleOptimizationType.TAX_AWARE.value)
        try:
            optimization_type = OracleOptimizationType.from_string(optimization_type_str)
        except ValueError:
            # Use logger.warning for invalid input with fallback
            logger.warning(f"Invalid optimization_type '{optimization_type_str}' in data. Defaulting to TAX_AWARE.")
            optimization_type = OracleOptimizationType.TAX_AWARE
            
        return cls(
            tax_lots=tax_lots_df,
            targets=pd.DataFrame(data["targets"]),
            prices=pd.DataFrame(data["prices"]),
            spreads=pd.DataFrame(data["spreads"]) if data["spreads"] is not None else None,
            factor_model=pd.DataFrame(data["factor_model"]) if data["factor_model"] is not None else None,
            cash=data["cash"],
            strategy_id=data.get("strategy_id"),
            optimization_type=optimization_type,
            withdrawal_amount=data.get("withdrawal_amount", 0.0)
        )
    
    def _check_numerical_stability(self, value, name):
        """
        Check for potential numerical stability issues in optimization values.

        Monitors for values that might cause numerical instability in the optimization,
        such as very large or very small numbers.

        Args:
            value: The numerical value to check
            name (str): Name of the component being checked for logging purposes

        Note:
            Logs warnings for values with absolute value > 1e8 or < 1e-8 (except 0)
        """
        if abs(value) > 1e8:
             # Use logger.warning for potential stability issues
            logger.warning(f"Large value detected in {name}: {value}")
        if abs(value) < 1e-8 and value != 0:
             # Use logger.warning for potential stability issues
            logger.warning(f"Very small value detected in {name}: {value}")

    def _calculate_trade_values(
        self,
        buys: dict,
        sells: dict,
        drift: pd.DataFrame,
        gain_loss: pd.DataFrame,
        cash_identifier: str
    ) -> tuple[dict, dict, float, float]:
        """
        Calculate the total buy and sell values for all securities.

        Computes the market value of all buy and sell trades, taking into account
        current prices and quantities.

        Args:
            buys (dict): Dictionary of buy variables
            sells (dict): Dictionary of sell variables
            drift (pd.DataFrame): Drift report
            gain_loss (pd.DataFrame): Gain/loss report
            cash_identifier (str): Identifier for cash position

        Returns:
            tuple[dict, dict, float, float]: Tuple containing:
                - Dictionary of buy values by identifier
                - Dictionary of sell values by identifier
                - Total buy value
                - Total sell value
        """
        # Calculate total buy value
        buy_values = {}
        total_buy_value = 0
        for _, row in drift.iterrows():
            identifier = row['identifier']
            if identifier == cash_identifier or identifier not in buys:
                continue
            price = self.prices.loc[self.prices['identifier'] == identifier, 'price'].iloc[0]
            buy_values[identifier] = buys[identifier] * price
            total_buy_value += buy_values[identifier]
        
        # Calculate total sell value
        sell_values = {}
        total_sell_value = 0
        for identifier in self.all_identifiers:
            if identifier == cash_identifier:
                continue
            identifier_lots = gain_loss[gain_loss['identifier'] == identifier]
            if identifier_lots.empty:
                continue
            price = self.prices.loc[self.prices['identifier'] == identifier, 'price'].iloc[0]
            sell_values[identifier] = pulp.lpSum([
                sells[lot['tax_lot_id']] * price
                for _, lot in identifier_lots.iterrows()
                if lot['tax_lot_id'] in sells
            ])
            total_sell_value += sell_values[identifier]
            
        return buy_values, sell_values, total_buy_value, total_sell_value

    def _solve_no_trades_scenario(
        self,
        prob: pulp.LpProblem,
        buys: Dict[str, pulp.LpVariable],
        sells: Dict[str, pulp.LpVariable],
        debug: bool = False
    ) -> tuple[Optional[int], Optional[float], Optional[Dict]]:
        """
        Solve the optimization problem with all trades forced to zero.

        Creates and solves a modified version of the optimization problem where
        all trading is prohibited, to establish a baseline for comparison.

        Args:
            prob (pulp.LpProblem): The original optimization problem
            buys (Dict[str, pulp.LpVariable]): Dictionary of buy variables
            sells (Dict[str, pulp.LpVariable]): Dictionary of sell variables
            debug (bool): Whether to print debug information

        Returns:
            tuple[Optional[int], Optional[float], Optional[Dict]]: Tuple containing:
                - Solution status code
                - Optimized value for no-trades scenario
                - Dictionary of objective component values
        """
        # Create a copy for no-trades scenario
        prob_no_trades = prob.copy()
        
        # Add constraints to force all variables to zero in no-trades copy
        for buy_var in buys.values():
            prob_no_trades += buy_var == 0, f"no_trade_buy_{buy_var.name}"
        for sell_var in sells.values():
            prob_no_trades += sell_var == 0, f"no_trade_sell_{sell_var.name}"
            
        if debug:
            logger.info("Solving no-trades scenario first...")
        
        status, optimized_value = self._solve_optimization(prob_no_trades, debug)
        
        # Extract component values if solve was successful
        component_values = None
        if status == pulp.LpStatusOptimal:
            component_values = self.objective_manager.extract_component_values(prob_no_trades)
            if debug:
                logger.info("No-trades objective components:")
                for name, value in component_values.items():
                    logger.info(f"  {name}: {value}")
        
        return status, optimized_value, component_values

    def _handle_tlh_optimization(
        self,
        prob: pulp.LpProblem,
        buys: Dict[str, pulp.LpVariable],
        sells: Dict[str, pulp.LpVariable],
        drift_report: pd.DataFrame,
        gain_loss: pd.DataFrame,
        total_value: float,
        min_notional: float,
        debug: bool = False,
        log_time: bool = False
    ) -> Tuple[List[TLHTrade], Optional[Dict]]:
        """
        Handle tax-loss harvesting (TLH) optimization if enabled.

        Identifies tax-loss harvesting opportunities and modifies the optimization
        problem to incorporate TLH objectives and constraints.

        Args:
            prob (pulp.LpProblem): The optimization problem
            buys (Dict[str, pulp.LpVariable]): Dictionary of buy variables
            sells (Dict[str, pulp.LpVariable]): Dictionary of sell variables
            drift_report (pd.DataFrame): Current drift report
            gain_loss (pd.DataFrame): Gain/loss report
            total_value (float): Total portfolio value
            min_notional (float): Minimum notional amount for any trade
            debug (bool): Whether to print debug information
            log_time (bool): Whether to log timing information

        Returns:
            Tuple[List[TLHTrade], Optional[Dict]]: Tuple containing:
                - List of identified TLH trade opportunities
                - Dictionary of TLH baseline components (or None if TLH not enabled)

        Raises:
            ValueError: If wash sale restrictions are not set but TLH is enabled
        """
        timing_data = {}
        if log_time:
            start_time = time.time()

        tlh_trades = []
        tlh_baseline_components = None
        
        if not (self.optimization_type.should_tlh() and self.should_tlh):
            return tlh_trades, tlh_baseline_components
            
        if self.oracle.wash_sale_restrictions is None:
            raise ValueError("Wash sale restrictions not set in Oracle but we're trying to TLH.")
            
        if log_time:
            tlh_impact_start = time.time()

        # Add TLH impact to objective function
        tlh_opportunities, sell_quantities, buy_quantities, tlh_baseline_components = calculate_tlh_impact(
            prob=prob,
            buys=buys,
            sells=sells,
            drift_report=drift_report,
            gain_loss_report=gain_loss,
            prices=self.prices,
            tax_rates=self.oracle.tax_rates,
            constraints_manager=self.constraints_manager,
            target_weights=self.targets,
            total_portfolio_value=total_value,
            min_weight_multiplier=self.range_min_weight_multiplier,
            max_weight_multiplier=self.range_max_weight_multiplier,
            min_notional=min_notional,
            trade_rounding=self.trade_rounding,
            min_loss_threshold=self.tlh_min_loss_threshold,
            objective_manager=self.objective_manager,
            optimization_type=self.optimization_type,
        )

        if log_time:
            timing_data['tlh_impact_calculation'] = time.time() - tlh_impact_start
        
        # Store TLH trades for later use
        tlh_trades = tlh_opportunities
        
        if debug and tlh_opportunities:
            logger.info(f"Found {len(tlh_opportunities)} TLH opportunities:")
            for trade in tlh_opportunities:
                logger.info(f"  {trade.identifier}: {trade.loss_percentage:.1%} loss, "
                          f"potential savings ${trade.potential_tax_savings:.2f}")
            
            if tlh_baseline_components:
                logger.info("TLH baseline objective components:")
                for name, value in tlh_baseline_components.items():
                    logger.info(f"  {name}: {value}")

        if log_time:
            timing_data['total_time'] = time.time() - start_time
            logger.info("=== TLH Optimization Timing Breakdown ===")
            for section, duration in timing_data.items():
                logger.info(f"{section}: {duration:.3f} seconds")
                    
        return tlh_trades, tlh_baseline_components

    def _try_buy_only_optimization(
        self,
        buy_only_prob: pulp.LpProblem,
        buys: Dict[str, pulp.LpVariable],
        sells: Dict[str, pulp.LpVariable],
        drift: pd.DataFrame,
        gain_loss: pd.DataFrame,
        total_value: float,
        min_cash: float,
        no_trade_optimized_value: float,
        buy_threshold: float,
        holding_time_delta: Optional[datetime],
        min_notional: float,
        actual_cash: float,
        improvement: float,
        rebalance_threshold: float,
        buy_df: Optional[pd.DataFrame] = None,
        sell_df: Optional[pd.DataFrame] = None,
        debug: bool = False
    ) -> Tuple[Optional[int], Optional[float], pulp.LpProblem, bool, Optional[Dict]]:
        """
        Try buy-only optimization if full rebalance doesn't meet improvement threshold.

        Attempts a more conservative optimization approach that only allows purchases,
        useful when a full rebalance would not provide sufficient improvement to
        justify the trading costs.

        Args:
            buy_only_prob (pulp.LpProblem): The optimization problem for buy-only scenario
            buys (Dict[str, pulp.LpVariable]): Dictionary of buy variables
            sells (Dict[str, pulp.LpVariable]): Dictionary of sell variables
            drift (pd.DataFrame): Current drift report
            gain_loss (pd.DataFrame): Gain/loss report
            total_value (float): Total portfolio value
            min_cash (float): Minimum required cash amount
            no_trade_optimized_value (float): Optimized value from no-trade scenario
            buy_threshold (float): Minimum improvement threshold for buy-only trades
            holding_time_delta (Optional[datetime]): Optional holding time constraint
            min_notional (float): Minimum notional amount for any trade
            actual_cash (float): Current cash position
            improvement (float): Current improvement from full rebalance
            rebalance_threshold (float): Threshold for full rebalance
            buy_df (Optional[pd.DataFrame]): Buy decisions dataframe
            sell_df (Optional[pd.DataFrame]): Sell decisions dataframe
            debug (bool): Whether to print debug information

        Returns:
            Tuple[Optional[int], Optional[float], pulp.LpProblem, bool, Optional[Dict]]:
                - Solution status code
                - Optimized value for buy-only scenario
                - Modified problem object
                - Whether this is a buy-only optimization
                - Failure context dictionary (if optimization failed or didn't meet threshold)
        """
        # First check if we have enough cash
        if actual_cash < min_cash:
            if debug:
                logger.info(f"Not enough cash (${actual_cash:,.2f}) to meet minimum requirement (${min_cash:,.2f}). Skipping buy-only optimization.")
            failure_context = {
                'case_type': 'not_enough_cash_to_buy_only',
                'improvements': {
                    'rebalance': {
                        'value': improvement,
                        'threshold': rebalance_threshold
                    }
                },
                'additional_info': {
                    'actual_cash': actual_cash,
                    'min_cash': min_cash
                }
            }
            return None, None, buy_only_prob, True, failure_context

        # Create a new problem for buy-only optimization
        self.optimization_problem = buy_only_prob
        
        # Force all sells to zero
        for sell_var in sells.values():
            buy_only_prob += sell_var == 0, f"force_no_sell_{sell_var.name}"
        
        # Solve buy-only optimization
        status, buy_only_optimized_value = self._solve_optimization(buy_only_prob, debug)
        
        if status is None or status != pulp.LpStatusOptimal:
            if debug:
                logger.info("Buy-only optimization failed.")
            failure_context = {
                'case_type': 'buy_only_failed',
                'improvements': {
                    'rebalance': {
                        'value': improvement,
                        'threshold': rebalance_threshold
                    }
                },
                'optimization_status': pulp.LpStatus[status] if status is not None else 'Failed before solve',
                'additional_info': {'total_value': total_value, 'min_cash': min_cash, 'actual_cash': actual_cash}
            }
            return status, None, buy_only_prob, True, failure_context
        
        # Check improvement against buy threshold
        buy_only_improvement = no_trade_optimized_value - buy_only_optimized_value
        if buy_threshold is not None and buy_only_improvement < buy_threshold:
            if debug:
                logger.info(f"Buy-only improvement ({buy_only_improvement}) is below threshold ({buy_threshold}).")
            failure_context = {
                'case_type': 'buy_only_below_threshold',
                'improvements': {
                    'rebalance': {
                        'value': improvement,
                        'threshold': rebalance_threshold
                    },
                    'buy_only': {
                        'value': buy_only_improvement,
                        'threshold': buy_threshold
                    }
                },
                'additional_info': {'total_value': round(total_value, 2), 'min_cash': round(min_cash, 2), 'actual_cash': round(actual_cash, 2)}
            }
            return status, None, buy_only_prob, True, failure_context
            
        return status, buy_only_optimized_value, buy_only_prob, True, None

    def compute_optimal_trades(
        self,
        weight_tax: float = 1,
        weight_drift: float = 1,
        weight_transaction: float = 1,
        weight_factor_model: float = 0.0,
        weight_cash_drag: float = 0.0,  
        rebalance_threshold: float = None,  
        buy_threshold: float = None,  
        holding_time_days: int = 0,
        should_tlh: bool = False,
        tlh_min_loss_threshold: float = 0.015,  
        range_min_weight_multiplier: float = 0.5,  
        range_max_weight_multiplier: float = 2.0,  
        min_notional: float = 0,  
        rank_penalty_factor: float = 0.0,  
        trade_rounding: int = 4,
        enforce_wash_sale_prevention: bool = True,
        debug: bool = True,
        dump: bool = False,
        log_time: bool = True
    ) -> tuple[Optional[int], bool, Dict, pd.DataFrame]:
        """
        Compute optimal trades using linear optimization.

        This is the main optimization method that attempts to find the optimal set
        of trades to achieve the portfolio's objectives while respecting all constraints.
        It can handle various optimization types including tax-aware rebalancing,
        tax-loss harvesting, and direct indexing.

        Args:
            weight_tax (float): Weight for tax objective component (default: 1)
            weight_drift (float): Weight for drift objective component (default: 1)
            weight_transaction (float): Weight for transaction cost component (default: 1)
            weight_factor_model (float): Weight for factor model objective (default: 0.0)
            weight_cash_drag (float): Weight for cash penalty component (default: 0.0)
            rebalance_threshold (float, optional): Minimum improvement for full rebalance
            buy_threshold (float, optional): Minimum improvement for buy-only trades
            holding_time_days (int): Minimum holding period in days (default: 0)
            should_tlh (bool): Whether to enable tax-loss harvesting (default: False)
            tlh_min_loss_threshold (float): Minimum loss % for TLH (default: 1.5%)
            range_min_weight_multiplier (float): Lower bound for weight ranges (default: 0.5)
            range_max_weight_multiplier (float): Upper bound for weight ranges (default: 2.0)
            min_notional (float): Minimum notional amount for trades (default: 0)
            rank_penalty_factor (float): Factor for rank penalty (default: 0.0)
            trade_rounding (int): Decimal places for trade rounding (default: 4)
            enforce_wash_sale_prevention (bool): Whether to enforce wash sale rules (default: True)
            debug (bool): Whether to print debug information (default: True)
            dump (bool): Whether to dump strategy state (default: False)
            log_time (bool): Whether to log execution times (default: True)

        Returns:
            tuple[Optional[int], bool, Dict, pd.DataFrame]: Tuple containing:
                - Solution status code (or None if optimization failed)
                - Whether trades should be executed
                - Trade summary dictionary with optimization statistics
                - DataFrame of recommended trades

        Raises:
            ValueError: If optimization parameters are invalid or incompatible
        """
        if log_time:
            start_time = time.time()
            timing_data = {}

        # Store optimization parameters
        self.weight_tax = weight_tax
        self.weight_drift = weight_drift
        self.weight_transaction = weight_transaction
        self.weight_factor_model = weight_factor_model
        self.weight_cash_drag = weight_cash_drag
        self.rebalance_threshold = rebalance_threshold
        self.buy_threshold = buy_threshold
        self.holding_time_delta = timedelta(days=holding_time_days)
        self.min_notional = min_notional
        self.rank_penalty_factor = rank_penalty_factor
        self.tlh_min_loss_threshold = tlh_min_loss_threshold
        self.should_tlh = should_tlh
        self.trade_rounding = trade_rounding
        self.range_min_weight_multiplier = range_min_weight_multiplier
        self.range_max_weight_multiplier = range_max_weight_multiplier
        self.enforce_wash_sale_prevention = enforce_wash_sale_prevention

        if dump and self.oracle is not None:
            pass
            #TODO: DUMP STATE

        # Early exit for HOLD strategy
        if self.optimization_type == OracleOptimizationType.HOLD:
            if debug:
                logger.info(f"Optimization Type: {self.optimization_type.value}. Returning no trades.")
            empty_summary = generate_trade_summary_from_strategies(
                pre_strategy=self,
                is_2nd_buy_only_optimization=False,
                trades=pd.DataFrame(),
                total_value=0,
                before_optimization={},
                after_optimization={},
                explanation_context={'case_type': 'hold_strategy'}
            )
            return None, False, empty_summary, pd.DataFrame()

        # Early validation for withdrawal compatibility
        if self.withdrawal_amount > 0:
            data_context = {}
            if not self.optimization_type.can_handle_withdrawal():
                raise ValueError(f"Optimization type {self.optimization_type.value} is not compatible with withdrawals")
            if debug:
                logger.info(f"Withdrawal amount: ${self.withdrawal_amount:.2f} with optimization type: {self.optimization_type.value}")

        if log_time:
            timing_data['initialization'] = time.time() - start_time
            logger.info(f"Initialization time: {timing_data['initialization']:.3f} seconds")
            report_start = time.time()

        # Initialize and get current portfolio state
        gain_loss = self.gain_loss_report
        drift = self.drift_report
        total_value = self.actuals['market_value'].sum()

        if log_time:
            timing_data['report_generation'] = time.time() - report_start
            logger.info(f"Report generation time: {timing_data['report_generation']:.3f} seconds")
            problem_setup_start = time.time()

        # Early exit if portfolio is empty
        if total_value == 0 and self.targets[self.targets['asset_class'] != CASH_CUSIP_ID]['target_weight'].sum() == 0:
            if debug:
                logger.info("Portfolio is empty and no non-cash targets. Returning no trades.")
            empty_summary = generate_trade_summary_from_strategies(
                pre_strategy=self,
                is_2nd_buy_only_optimization=False,
                trades=pd.DataFrame(),
                total_value=0,
                before_optimization={},
                after_optimization={},
                explanation_context={'case_type': 'empty_portfolio'}
            )
            return None, False, empty_summary, pd.DataFrame()
        
        # Create optimization problem
        prob = pulp.LpProblem("Portfolio_Rebalance", pulp.LpMinimize)
        self.optimization_problem = prob
        
        self.buys, self.sells, self.buy_df, self.sell_df = self._create_decision_variables(self.target_identifiers, gain_loss, debug)
        
        if debug:
            logger.info(f"Building Optimization Problem ({self.optimization_type.value}) ===")
            if self.withdrawal_amount > 0:
                logger.info(f"Withdrawal amount: ${self.withdrawal_amount:.2f}")
        
        # Apply strategy-specific setup
        self.optimization_type.setup_optimization(prob, self.buys, self.sells)

        if log_time:
            timing_data['problem_setup'] = time.time() - problem_setup_start
            logger.info(f"Problem setup time: {timing_data['problem_setup']:.3f} seconds")
            objective_start = time.time()
        
        # Use ObjectiveManager to calculate and set the objective function
        total_objective = self.objective_manager.calculate_objectives(
            buys=self.buys,
            sells=self.sells,
            drift=drift,
            gain_loss=gain_loss,
            total_value=total_value,
            weight_tax=weight_tax,
            weight_drift=weight_drift,
            weight_transaction=weight_transaction,
            weight_factor_model=weight_factor_model,
            weight_cash_drag=weight_cash_drag,
            rank_penalty_factor=rank_penalty_factor,
            buy_df=self.buy_df,
            sell_df=self.sell_df,
            enforce_wash_sale_prevention=enforce_wash_sale_prevention,
            debug=debug,
            log_time=log_time
        )

        prob += total_objective

        if log_time:
            timing_data['objective_calculation'] = time.time() - objective_start
            logger.info(f"Objective calculation time: {timing_data['objective_calculation']:.3f} seconds")
            no_trade_start = time.time()
        
        # Solve no-trades scenario BEFORE adding constraints
        no_trade_status, no_trade_optimized_value, no_trade_components = self._solve_no_trades_scenario(prob, self.buys, self.sells, debug)
        if no_trade_optimized_value is None:
            no_trade_optimized_value = 0
            logger.warning("No-trades scenario failed to solve.")
            raise ValueError("Failed to solve no-trades scenario. Check the problem setup.")

        if log_time:
            timing_data['no_trade_scenario'] = time.time() - no_trade_start
            logger.info(f"No trade scenario time: {timing_data['no_trade_scenario']:.3f} seconds")
            constraints_start = time.time()
        
        # Handle withdrawal if applicable
        if self.withdrawal_amount > 0:
            self.constraints_manager.add_withdrawal_constraints(
                prob=prob,
                buys=self.buys,
                sells=self.sells,
                drift=drift,
                gain_loss=gain_loss,
                total_value=total_value,
                withdrawal_amount=self.withdrawal_amount,
                debug=debug
            )
        
        # Store the no-trade component breakdown for reference
        self.no_trade_component_values = no_trade_components
        
        # Add general constraints
        self.min_cash = self.min_cash_amount()
        self.constraints_manager.add_constraints(
            prob=prob,
            buys=self.buys,
            sells=self.sells,
            prices=self.prices,
            gain_loss=gain_loss,
            tax_lots=self.tax_lots,
            min_cash_amount=self.min_cash,
            holding_time_delta=self.holding_time_delta,
            min_notional=min_notional,
            buy_df=self.buy_df,
            sell_df=self.sell_df,
            enforce_wash_sale_prevention=self.enforce_wash_sale_prevention
        )

        if log_time:
            timing_data['constraints_setup'] = time.time() - constraints_start
            logger.info(f"Constraints setup time: {timing_data['constraints_setup']:.3f} seconds")
            tlh_start = time.time()
        #Before we do any TLH, we need to set the buy_only_prob to be the same as prob. This is what we use later in the buy only case.
        buy_only_prob = prob.copy()

        # Handle TLH if enabled
        tlh_trades, tlh_baseline_components = self._handle_tlh_optimization(
            prob=prob,
            buys=self.buys,
            sells=self.sells,
            drift_report=drift,
            gain_loss=gain_loss,
            total_value=total_value,
            min_notional=min_notional,
            debug=debug,
            log_time=log_time
        )

        if log_time:
            timing_data['tlh_optimization'] = time.time() - tlh_start
            logger.info(f"TLH optimization time: {timing_data['tlh_optimization']:.3f} seconds")
            solve_start = time.time()

        # Set initial values for warm start
        self._set_initial_values(self.buys, self.sells, gain_loss, debug)
        
        # Solve the optimization problem
        status, optimized_value = self._solve_optimization(prob, debug)

        if log_time:
            timing_data['main_solve'] = time.time() - solve_start
            logger.info(f"Main solve time: {timing_data['main_solve']:.3f} seconds")
        
        # Handle solution status
        if status is None or status != pulp.LpStatusOptimal:
            status_str = pulp.LpStatus[status] if status is not None else 'Failure before/during solve'
            logger.warning(f"Optimization did not find an optimal solution (Status: {status_str}). Returning empty trades.")
            empty_summary = generate_trade_summary_from_strategies(
                pre_strategy=self,
                is_2nd_buy_only_optimization=False,
                trades=pd.DataFrame(),
                total_value=total_value,
                before_optimization=no_trade_components if no_trade_components else {},
                after_optimization={},
                explanation_context={
                    'case_type': 'optimization_failed',
                    'optimization_status': status_str,
                    'additional_info': {'total_value': total_value}
                }
            )
            return status, False, empty_summary, pd.DataFrame()
        
        # Check for improvement against rebalance threshold
        no_trade_optimized_value = tlh_baseline_components.get('overall', no_trade_optimized_value) if tlh_baseline_components else no_trade_optimized_value
        improvement = no_trade_optimized_value - optimized_value
        is_2nd_buy_only_optimization = False
        should_trade = True

        if no_trade_status == pulp.LpStatusOptimal and rebalance_threshold is not None and improvement < rebalance_threshold:
            if debug:
                logger.info(f"Rebalance improvement ({improvement}) is below threshold ({rebalance_threshold}). Trying buy-only optimization.")
            
            if log_time:
                buy_only_start = time.time()
            # Calculate actual cash position
            actual_cash = self.actuals[self.actuals['identifier'] == CASH_CUSIP_ID]['market_value'].sum()

            status, buy_only_optimized_value, prob, is_2nd_buy_only_optimization, failure_context = self._try_buy_only_optimization(
                buy_only_prob=buy_only_prob,
                buys=self.buys,
                sells=self.sells,
                drift=drift,
                gain_loss=gain_loss,
                total_value=total_value,
                min_cash=self.min_cash,
                no_trade_optimized_value=no_trade_optimized_value,
                buy_threshold=buy_threshold,
                holding_time_delta=self.holding_time_delta,
                min_notional=min_notional,
                actual_cash=actual_cash,
                improvement=improvement,
                rebalance_threshold=rebalance_threshold,
                buy_df=self.buy_df,
                sell_df=self.sell_df,
                debug=debug
            )
            
            if log_time:
                timing_data['buy_only_optimization'] = time.time() - buy_only_start
            
            if buy_only_optimized_value is None:
                buy_only_optimized_value = self.objective_manager.extract_component_values(prob)
                self.optimized_component_values = buy_only_optimized_value
                empty_summary = generate_trade_summary_from_strategies(
                    pre_strategy=self,
                    is_2nd_buy_only_optimization=True,
                    trades=pd.DataFrame(),
                    total_value=total_value,
                    before_optimization=no_trade_components if no_trade_components else {},
                    after_optimization={
                        'tax_cost': buy_only_optimized_value.get('tax', 0) if buy_only_optimized_value else 0,
                        'drift_cost': buy_only_optimized_value.get('drift', 0) if buy_only_optimized_value else 0,
                        'spread_costs': buy_only_optimized_value.get('transaction', 0) if buy_only_optimized_value else 0,
                        'factor_cost': buy_only_optimized_value.get('factor_model', 0) if buy_only_optimized_value else 0,
                        'cash_drag': buy_only_optimized_value.get('cash_drag', 0) if buy_only_optimized_value else 0,
                        'overall': buy_only_optimized_value
                    },
                    explanation_context=failure_context
                )
                return status, False, empty_summary, pd.DataFrame()
                
            optimized_value = buy_only_optimized_value
            improvement = no_trade_optimized_value - buy_only_optimized_value

        if log_time:
            post_process_start = time.time()

        # Extract optimized component values
        optimized_components = self.objective_manager.extract_component_values(prob)
        self.optimized_component_values = optimized_components

        # Extract trades
        trades = extract_trades(
            buys=self.buys,
            sells=self.sells,
            gain_loss=gain_loss,
            total_value=total_value,
            prices=self.prices,
            spreads=self.spreads,
            tlh_trades=tlh_trades,
            tax_normalization=self.TAX_NORMALIZATION * weight_tax,
            transaction_normalization=self.TRANSACTION_NORMALIZATION * weight_transaction,
            trade_rounding=trade_rounding,
            min_notional=min_notional,
        )
        if len(trades) == 0:
            should_trade = False
        
        if log_time:
            apply_trades_start = time.time()
            
        # Apply trades to get post-trade strategy
        self.post_trade_strategy = apply_trades_to_portfolio(
            tax_lots=self.tax_lots,
            trades=trades,
            cash=self.cash,
            current_date=self.current_date,
            strategy=self,
            log_time=log_time
        )
        
        if log_time:
            timing_data['apply_trades'] = time.time() - apply_trades_start
            logger.info(f"Apply trades time: {timing_data['apply_trades']:.3f} seconds")
            trade_summary_start = time.time()
        
        # Generate trade summary by comparing pre and post strategies
        trade_summary = generate_trade_summary_from_strategies(
            pre_strategy=self,
            is_2nd_buy_only_optimization=is_2nd_buy_only_optimization,
            trades=trades,
            total_value=total_value,
            before_optimization={
                'tax_cost': tlh_baseline_components.get('tax', no_trade_components.get('tax', 0)) if tlh_baseline_components else no_trade_components.get('tax', 0),
                'drift_cost': tlh_baseline_components.get('drift', no_trade_components.get('drift', 0)) if tlh_baseline_components else no_trade_components.get('drift', 0),
                'spread_costs': tlh_baseline_components.get('transaction', no_trade_components.get('transaction', 0)) if tlh_baseline_components else no_trade_components.get('transaction', 0),
                'factor_cost': tlh_baseline_components.get('factor_model', no_trade_components.get('factor_model', 0)) if tlh_baseline_components else no_trade_components.get('factor_model', 0),
                'cash_drag': tlh_baseline_components.get('cash_drag', no_trade_components.get('cash_drag', 0)) if tlh_baseline_components else no_trade_components.get('cash_drag', 0),
                'overall': no_trade_optimized_value
            },
            after_optimization={
                'tax_cost': optimized_components.get('tax', 0) if optimized_components else 0,
                'drift_cost': optimized_components.get('drift', 0) if optimized_components else 0,
                'spread_costs': optimized_components.get('transaction', 0) if optimized_components else 0,
                'factor_cost': optimized_components.get('factor_model', 0) if optimized_components else 0,
                'cash_drag': optimized_components.get('cash_drag', 0) if optimized_components else 0,
                'overall': optimized_value
            },
            log_time=log_time
        )

        if log_time:
            timing_data['trade_summary_generation'] = time.time() - trade_summary_start
            logger.info(f"Trade summary generation time: {timing_data['trade_summary_generation']:.3f} seconds")
            timing_data['post_processing'] = time.time() - post_process_start
            logger.info(f"Post processing time: {timing_data['post_processing']:.3f} seconds")
            timing_data['total_time'] = time.time() - start_time
            
            # Log final timing information
            logger.info("\n=== Final Optimization Timing Breakdown ===")
            for section, duration in timing_data.items():
                logger.info(f"{section}: {duration:.3f} seconds")
        
        return status, should_trade, trade_summary, trades

    def calculate_max_withdrawal_amount(
        self,
        respect_wash_sales: bool = True,
        preserve_targets: bool = True,
        debug: bool = False
    ) -> tuple[float, Dict, pd.DataFrame]:
        """
        Calculate the maximum amount that can be withdrawn from the portfolio.

        Creates a special optimization problem that attempts to liquidate as much
        of the portfolio as possible while respecting specified constraints.

        Args:
            respect_wash_sales (bool): Whether to respect wash sale restrictions (default: True)
            preserve_targets (bool): Whether to maintain target allocations (default: True)
            debug (bool): Whether to print debug information (default: False)

        Returns:
            tuple[float, Dict, pd.DataFrame]: Tuple containing:
                - Maximum withdrawal amount possible
                - Trade summary with optimization statistics
                - DataFrame of trades required to achieve the withdrawal

        Note:
            The actual withdrawal amount may be less than the total portfolio value
            due to various constraints like wash sales or target preservation.
        """
        # Call the helper function to do the calculation
        return calculate_max_withdrawal(
            strategy=self,
            debug=debug,
            respect_wash_sales=respect_wash_sales,
            preserve_targets=preserve_targets
        )

    def compare_drift(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Compare drift between current strategy and post-trade strategy.

        For pairs-based strategies, compares based on asset class drift instead
        of individual security drift.

        Returns:
            Tuple[pd.DataFrame, Dict]: Tuple containing:
                - DataFrame showing drift comparison for each position
                - Dictionary of summary statistics about drift improvement

        Raises:
            ValueError: If post-trade strategy is not available
        """
        if not hasattr(self, 'post_trade_strategy'):
            raise ValueError("No post-trade strategy available. Run compute_optimal_trades first.")
            
        return generate_drift_comparison_report(self, self.post_trade_strategy)

    def compare_factor_model(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Compare factor model exposures between current and post-trade strategy.

        Only applicable for DIRECT_INDEX optimization type. Analyzes changes in
        factor exposures resulting from the proposed trades.

        Returns:
            Tuple[pd.DataFrame, Dict]: Tuple containing:
                - DataFrame showing factor exposure comparison
                - Dictionary of summary statistics about factor exposure improvements

        Raises:
            ValueError: If post-trade strategy is not available
        """
        if not hasattr(self, 'post_trade_strategy'):
            raise ValueError("No post-trade strategy available. Run compute_optimal_trades first.")
            
        return generate_factor_model_comparison_report(self, self.post_trade_strategy)