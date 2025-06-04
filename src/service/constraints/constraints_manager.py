import pulp
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import timedelta
import time

from src.service.helpers.constants import logger
from src.service.constraints.cash.cash_validator import CashValidator
from src.service.constraints.trade.min_notional_validator import MinNotionalValidator
from src.service.constraints.trade.no_simultaneous_trade_validator import NoSimultaneousTradeValidator
from src.service.constraints.restriction.restriction_validator import RestrictionValidator
from src.service.constraints.holding_time.holding_time_validator import HoldingTimeValidator
from src.service.constraints.cash.withdrawal_validator import WithdrawalValidator
from src.service.constraints.trade.no_buy_validator import NoBuyValidator
from src.service.constraints.drift.drift_validator import DriftValidator
from src.service.helpers.constants import CASH_CUSIP_ID

class ConstraintsManager:
    """Manager class for handling all optimization constraints."""
    
    def __init__(self, oracle_strategy):
        """
        Initialize ConstraintsManager with reference to OracleStrategy.
        
        Args:
            oracle_strategy: Reference to the OracleStrategy instance
        """
        self.strategy = oracle_strategy
        self.validators = {}
        self.pre_trade_validators = {}  # Validators that must be checked before any trade
        self.post_trade_validators = {}  # Validators that are checked after trade decisions
    
    def add_constraints(
        self,
        prob: pulp.LpProblem,
        buys: Dict[str, pulp.LpVariable],
        sells: Dict[str, pulp.LpVariable],
        prices: pd.DataFrame,
        gain_loss: pd.DataFrame,
        tax_lots: pd.DataFrame,
        min_cash_amount: float,
        holding_time_delta: timedelta = None,
        min_notional: float = 0,
        range_min_weight_multiplier: float = 0.5,  # Default to 50% below target
        range_max_weight_multiplier: float = 2.0,   # Default to 100% above target
        log_time: bool = False,
        buy_df: Optional[pd.DataFrame] = None,
        sell_df: Optional[pd.DataFrame] = None,
        enforce_wash_sale_prevention: bool = True,
        debug: bool = False,
    ) -> None:
        """
        Add all constraints to the optimization problem.
        
        Args:
            prob: PuLP optimization problem
            buys: Dictionary of buy variables
            sells: Dictionary of sell variables
            gain_loss: DataFrame with gain/loss report
            tax_lots: DataFrame with tax lot information
            min_cash_amount: Minimum cash amount to maintain
            holding_time_delta: Minimum holding time for tax lots
            min_notional: Minimum notional amount for any trade (in dollars)
            range_min_weight_multiplier: Multiplier for minimum weight (e.g. 0.5 means can't go below 50% of target)
            range_max_weight_multiplier: Multiplier for maximum weight (e.g. 2.0 means can't go above 200% of target)
            debug: Enable debug logging
            log_time: Whether to log timing information
            buy_df: Optional DataFrame with buy information for vectorized operations
            sell_df: Optional DataFrame with sell information for vectorized operations
        """
        timing_data = {}
        if log_time:
            start_time = time.time()
        
        # Initialize and use CashValidator if not already created
        if log_time:
            cash_start = time.time()
        if 'cash' not in self.post_trade_validators:
            self.post_trade_validators['cash'] = CashValidator(self.strategy, min_cash_amount)
        else:
            self.post_trade_validators['cash'].min_cash_amount = min_cash_amount
            
        # Add cash constraints using validator
        self.post_trade_validators['cash'].add_to_problem(
            prob=prob,
            buys=buys,
            sells=sells,
            gain_loss=gain_loss
        )
        if log_time:
            timing_data['cash_constraints'] = time.time() - cash_start
        
        # Initialize and use MinNotionalValidator if not already created
        if log_time:
            notional_start = time.time()
        if 'min_notional' not in self.post_trade_validators:
            self.post_trade_validators['min_notional'] = MinNotionalValidator(self.strategy, min_notional)
        else:
            self.post_trade_validators['min_notional'].min_notional = min_notional
            
        # Add minimum notional constraints using validator
        self.post_trade_validators['min_notional'].add_to_problem(
            prob=prob,
            buys=buys,
            sells=sells,
            prices=prices,
            tax_lots=tax_lots
        )
        if log_time:
            timing_data['notional_constraints'] = time.time() - notional_start
        
        # Initialize and use NoSimultaneousTradeValidator
        if log_time:
            simultaneous_start = time.time()
        if 'no_simultaneous_trade' not in self.post_trade_validators:
            self.post_trade_validators['no_simultaneous_trade'] = NoSimultaneousTradeValidator(self.strategy)
            
        # Add no simultaneous buy/sell constraints using validator
        self.post_trade_validators['no_simultaneous_trade'].add_to_problem(
            prob=prob,
            buys=buys,
            sells=sells,
            gain_loss=gain_loss,
            all_identifiers=self.strategy.all_identifiers
        )
        if log_time:
            timing_data['simultaneous_trade_constraints'] = time.time() - simultaneous_start
        
        # Initialize and use RestrictionValidator
        if log_time:
            restriction_start = time.time()
        if 'restriction' not in self.pre_trade_validators:
            self.pre_trade_validators['restriction'] = RestrictionValidator(self.strategy, enforce_wash_sale_prevention)
            
        # Add stock and wash sale restrictions using validator
        self.pre_trade_validators['restriction'].add_to_problem(
            prob=prob,
            buys=buys,
            sells=sells,
            gain_loss=gain_loss,
            stock_restrictions=self.strategy.oracle.stock_restrictions,
            wash_sale_restrictions=self.strategy.oracle.wash_sale_restrictions,
            all_identifiers=self.strategy.all_identifiers
        )
        if log_time:
            timing_data['restriction_constraints'] = time.time() - restriction_start
        
        # Initialize and use HoldingTimeValidator if needed
        if holding_time_delta is not None and holding_time_delta > timedelta(days=0):
            if log_time:
                holding_time_start = time.time()
            if 'holding_time' not in self.pre_trade_validators:
                self.pre_trade_validators['holding_time'] = HoldingTimeValidator(self.strategy, holding_time_delta)
            else:
                self.pre_trade_validators['holding_time'].holding_time_delta = holding_time_delta
                
            # Add holding time constraints using validator
            self.pre_trade_validators['holding_time'].add_to_problem(
                prob=prob,
                sells=sells,
                tax_lots=tax_lots,
                current_date=self.strategy.oracle.current_date
            )
            if log_time:
                timing_data['holding_time_constraints'] = time.time() - holding_time_start
            
        # Add DriftValidator for PAIRS_TLH and DIRECT_INDEX optimization types
        if self.strategy.optimization_type in {self.strategy.optimization_type.PAIRS_TLH, 
                                                    self.strategy.optimization_type.DIRECT_INDEX}:
            if log_time:
                drift_start = time.time()
            if 'drift' not in self.post_trade_validators:
                self.post_trade_validators['drift'] = DriftValidator(
                    self.strategy,
                    range_min_weight_multiplier=range_min_weight_multiplier,
                    range_max_weight_multiplier=range_max_weight_multiplier
                )
            else:
                self.post_trade_validators['drift'].range_min_weight_multiplier = range_min_weight_multiplier
                self.post_trade_validators['drift'].range_max_weight_multiplier = range_max_weight_multiplier
                
            # Add drift constraints using validator with vectorized DataFrames
            self.post_trade_validators['drift'].add_to_problem(
                prob=prob,
                buys=buys,
                sells=sells,
                drift=self.strategy.drift_report,
                buy_df=buy_df,
                sell_df=sell_df
            )
            if log_time:
                timing_data['drift_constraints'] = time.time() - drift_start
        
        # Keep validators dict in sync with pre and post trade validators
        self.validators = {**self.pre_trade_validators, **self.post_trade_validators}

        if log_time:
            timing_data['total_time'] = time.time() - start_time
            logger.info("=== Constraints Setup Timing Breakdown ===")
            for section, duration in timing_data.items():
                logger.info(f"{section}: {duration:.3f} seconds")
            
    def add_withdrawal_constraints(
        self,
        prob: pulp.LpProblem,
        buys: Dict[str, pulp.LpVariable],
        sells: Dict[str, pulp.LpVariable],
        drift: pd.DataFrame,
        gain_loss: pd.DataFrame,
        total_value: float,
        withdrawal_amount: float,
        debug: bool = True
    ) -> None:
        """
        Add withdrawal-related constraints to the optimization problem.
        
        Args:
            prob: PuLP optimization problem
            buys: Dictionary of buy variables
            sells: Dictionary of sell variables
            drift: DataFrame with drift report
            gain_loss: DataFrame with gain/loss report
            total_value: Total portfolio value
            withdrawal_amount: Amount to withdraw from the portfolio
            debug: Enable debug logging
        """
        if withdrawal_amount <= 0:
            return  # No withdrawal, no constraints needed
            
        # Initialize and use WithdrawalValidator if not already created
        if 'withdrawal' not in self.post_trade_validators:
            self.post_trade_validators['withdrawal'] = WithdrawalValidator(self.strategy, withdrawal_amount)
        else:
            self.post_trade_validators['withdrawal'].withdrawal_amount = withdrawal_amount
            
        # Add withdrawal constraints using validator
        self.post_trade_validators['withdrawal'].add_to_problem(
            prob=prob,
            buys=buys,
            sells=sells,
            drift=drift,
            gain_loss=gain_loss,
            total_value=total_value,
            debug=debug
        )
        
        # Keep validators dict in sync
        self.validators = {**self.pre_trade_validators, **self.post_trade_validators}
        
    def add_no_buy_constraints(
        self,
        prob: pulp.LpProblem,
        buys: Dict[str, pulp.LpVariable],
        exclude_cash: bool = True
    ) -> None:
        """
        Add constraints to prevent buying any securities (typically used for liquidation).
        
        Args:
            prob: PuLP optimization problem
            buys: Dictionary of buy variables
            exclude_cash: Whether to exclude cash from the no-buy constraint (default True)
        """
        # Initialize and use NoBuyValidator if not already created
        if 'no_buy' not in self.post_trade_validators:
            self.post_trade_validators['no_buy'] = NoBuyValidator(self.strategy, exclude_cash)
        else:
            self.post_trade_validators['no_buy'].exclude_cash = exclude_cash
            
        # Add no-buy constraints using validator
        self.post_trade_validators['no_buy'].add_to_problem(
            prob=prob,
            buys=buys,
            exclude_cash=exclude_cash
        )
        
        # Keep validators dict in sync
        self.validators = {**self.pre_trade_validators, **self.post_trade_validators}
        
    def is_restricted_from_selling_pre_trade(self, identifier: str, tax_lot_id: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a security is restricted from buying or selling.
        
        Args:
            identifier: The security identifier to check
            tax_lot_id: Optional tax lot ID if checking sell restrictions
            
        Returns:
            Tuple of (is_restricted, reason)
            - is_restricted: True if restricted, False otherwise
            - reason: None if not restricted, otherwise a string explaining why it's restricted
        """
        # Check sell restrictions
        for validator in self.pre_trade_validators.values():
            try:
                is_allowed, reason = validator.validate_sell(tax_lot_id, 0)
                if not is_allowed:
                    return True, reason
            except NotImplementedError:
                # Skip validators that don't implement individual trade validation
                continue
        return False, None
        
    def is_restricted_from_buying_pre_trade(self, identifier: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a security is restricted from buying or selling.
        
        Args:
            identifier: The security identifier to check
            tax_lot_id: Optional tax lot ID if checking sell restrictions
            
        Returns:
            Tuple of (is_restricted, reason)
            - is_restricted: True if restricted, False otherwise
            - reason: None if not restricted, otherwise a string explaining why it's restricted
        """
        for validator in self.validators.values():
            try:
                is_allowed, reason = validator.validate_buy(identifier, 0)
                if not is_allowed:
                    return True, reason
            except NotImplementedError:
                # Skip validators that don't implement individual trade validation
                continue
        return False, None
        