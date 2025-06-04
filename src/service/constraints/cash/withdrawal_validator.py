from typing import Optional, Tuple
import pandas as pd
import pulp

from src.service.constraints.base_validator import BaseValidator
from src.service.helpers.constants import CASH_CUSIP_ID, logger

class WithdrawalValidator(BaseValidator):
    """
    Validator for withdrawal-related constraints.
    Note: Individual trade validation is not supported as withdrawal validation requires knowledge of all trades.
    """
    
    def __init__(self, oracle_strategy, withdrawal_amount: float):
        """
        Initialize WithdrawalValidator.
        
        Args:
            oracle_strategy: Reference to the OracleStrategy instance
            withdrawal_amount: Amount to withdraw from the portfolio
        """
        super().__init__(oracle_strategy)
        self.withdrawal_amount = withdrawal_amount
        
    def validate_buy(self, identifier: str, quantity: float) -> Tuple[bool, Optional[str]]:
        """Cannot validate individual buys without knowledge of all trades."""
        raise NotImplementedError("Withdrawal validation requires knowledge of all trades")
        
    def validate_sell(self, tax_lot_id: str, quantity: float) -> Tuple[bool, Optional[str]]:
        """Cannot validate individual sells without knowledge of all trades."""
        raise NotImplementedError("Withdrawal validation requires knowledge of all trades")
        
    def add_to_problem(
        self,
        prob: pulp.LpProblem,
        buys: dict,
        sells: dict,
        drift: pd.DataFrame,
        gain_loss: pd.DataFrame,
        total_value: float,
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
            debug: Enable debug logging
        """
        if self.withdrawal_amount <= 0:
            return  # No withdrawal, no constraints needed
            
        # Get cash information from drift report
        cash_row = drift[drift['asset_class'] == CASH_CUSIP_ID]
        if cash_row.empty:
            if debug:
                logger.warning("CASH_CUSIP_ID not found in drift report for withdrawal calculation.")
            return  # No cash position defined
            
        current_cash = cash_row['actual_weight'].iloc[0] * total_value
        
        if debug:
            logger.info("Calculating Withdrawal Constraints ===")
            logger.info(f"  Current cash: ${current_cash:.2f}")
            logger.info(f"  Withdrawal amount: ${self.withdrawal_amount:.2f}")
        
        # Calculate total buys and sells in dollar terms
        total_buys = 0
        for identifier, buy_var in buys.items():
            if identifier != CASH_CUSIP_ID:
                price = self.strategy.prices.loc[
                    self.strategy.prices['identifier'] == identifier, 'price'
                ].iloc[0]
                total_buys += buy_var * price
                
        total_sells = 0
        for _, lot in gain_loss.iterrows():
            tax_lot_id = lot['tax_lot_id']
            if tax_lot_id in sells:
                price = self.strategy.prices.loc[
                    self.strategy.prices['identifier'] == lot['identifier'], 'price'
                ].iloc[0]
                total_sells += sells[tax_lot_id] * price
        
        # Calculate new cash after trades and withdrawal
        new_cash = (current_cash + total_sells - total_buys - self.withdrawal_amount)
        
        # Constraint: Ensure new cash is non-negative
        prob += new_cash >= 0, "withdrawal_cash_constraint"
        
        if debug:
            logger.info(f"Added withdrawal constraint: new_cash >= 0") 