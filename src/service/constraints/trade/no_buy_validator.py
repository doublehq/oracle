from typing import Optional, Tuple
import pulp

from src.service.constraints.base_validator import BaseValidator
from src.service.helpers.constants import CASH_CUSIP_ID

class NoBuyValidator(BaseValidator):
    """Validator for preventing buying of securities (typically used for liquidation)."""
    
    def __init__(self, oracle_strategy, exclude_cash: bool = True):
        """
        Initialize NoBuyValidator.
        
        Args:
            oracle_strategy: Reference to the OracleStrategy instance
            exclude_cash: Whether to exclude cash from the no-buy constraint (default True)
        """
        super().__init__(oracle_strategy)
        self.exclude_cash = exclude_cash
        
    def validate_buy(self, identifier: str, quantity: float) -> Tuple[bool, Optional[str]]:
        """Check if buying a security is allowed."""
        if self.exclude_cash and identifier == CASH_CUSIP_ID:
            return True, None
        return False, f"Buying {identifier} is not allowed during liquidation"
        
    def validate_sell(self, tax_lot_id: str, quantity: float) -> Tuple[bool, Optional[str]]:
        """Selling is always allowed during liquidation."""
        return True, None
        
    def add_to_problem(
        self,
        prob: pulp.LpProblem,
        buys: dict,
        exclude_cash: bool = True
    ) -> None:
        """
        Add constraints to prevent buying any securities.
        
        Args:
            prob: PuLP optimization problem
            buys: Dictionary of buy variables
            exclude_cash: Whether to exclude cash from the no-buy constraint (default True)
        """
        for identifier, buy_var in buys.items():
            # Skip cash if exclude_cash is True
            if exclude_cash and identifier == CASH_CUSIP_ID:
                continue
            # Add constraint to force buy variable to zero
            prob += buy_var == 0, f"No_Buy_{identifier}" 