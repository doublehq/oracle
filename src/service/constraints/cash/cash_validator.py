from typing import Optional, Tuple
import pandas as pd
import pulp

from src.service.constraints.base_validator import BaseValidator
from src.service.helpers.constants import CASH_CUSIP_ID

class CashValidator(BaseValidator):
    """
    Validator for cash-related constraints.
    Note: Individual trade validation is not supported as cash validation requires knowledge of all trades.
    """
    
    def __init__(self, oracle_strategy, min_cash_amount: float):
        """
        Initialize CashValidator.
        
        Args:
            oracle_strategy: Reference to the OracleStrategy instance
            min_cash_amount: Minimum cash amount to maintain
        """
        super().__init__(oracle_strategy)
        self.min_cash_amount = min_cash_amount
        
    def validate_buy(self, identifier: str, quantity: float) -> Tuple[bool, Optional[str]]:
        """Cash validation requires knowledge of all trades."""
        raise NotImplementedError("Cash validation requires knowledge of all trades")
        
    def validate_sell(self, tax_lot_id: str, quantity: float) -> Tuple[bool, Optional[str]]:
        """Cash validation requires knowledge of all trades."""
        raise NotImplementedError("Cash validation requires knowledge of all trades")
        
    def add_to_problem(
        self,
        prob: pulp.LpProblem,
        buys: dict,
        sells: dict,
        gain_loss: pd.DataFrame
    ) -> None:
        """
        Add cash constraints to the optimization problem.
        This preserves the original add_cash_constraint functionality.
        
        Args:
            prob: PuLP optimization problem
            buys: Dictionary of buy variables
            sells: Dictionary of sell variables
            gain_loss: DataFrame with gain/loss report
        """
        # Never buy CASH
        if CASH_CUSIP_ID in buys:
            prob += (buys[CASH_CUSIP_ID] == 0, "no_cash_buy")
            
        # Calculate total buy cost
        total_buy_cost = pulp.lpSum(
            buy_var * self.strategy.prices.loc[self.strategy.prices['identifier'] == identifier, 'price'].iloc[0]
            for identifier, buy_var in buys.items()
        )
            
        # Calculate total sell proceeds using current_price from gain_loss
        total_sell_proceeds = pulp.lpSum(
            sells[lot['tax_lot_id']] * lot['current_price']
            for _, lot in gain_loss.iterrows()
        )
            
        # Ensure we don't exceed available cash plus proceeds from sales when buying
        prob += (
            total_buy_cost <= self.strategy.cash + total_sell_proceeds,
            "cash_balance"
        )
        
        # Add the minimum cash floor constraint
        prob += (
            self.strategy.cash + total_sell_proceeds - total_buy_cost >= self.min_cash_amount,
            "min_cash_floor"
        ) 