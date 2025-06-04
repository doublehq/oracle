from typing import Optional, Tuple, List
import pandas as pd
import pulp

from src.service.constraints.base_validator import BaseValidator

class NoSimultaneousTradeValidator(BaseValidator):
    """
    Validator to prevent buying and selling the same security simultaneously.
    Note: Individual trade validation is not supported as this requires knowledge of all trades.
    """
    
    def validate_buy(self, identifier: str, quantity: float) -> Tuple[bool, Optional[str]]:
        """Cannot validate individual buys without knowledge of all trades."""
        raise NotImplementedError("No simultaneous trade validation requires knowledge of all trades")
        
    def validate_sell(self, tax_lot_id: str, quantity: float) -> Tuple[bool, Optional[str]]:
        """Cannot validate individual sells without knowledge of all trades."""
        raise NotImplementedError("No simultaneous trade validation requires knowledge of all trades")
        
    def add_to_problem(
        self,
        prob: pulp.LpProblem,
        buys: dict,
        sells: dict,
        gain_loss: pd.DataFrame,
        all_identifiers: List[str]
    ) -> None:
        """Add no simultaneous buy/sell constraints to the optimization problem."""
        # For each identifier, create a constraint that prevents buying and selling at the same time
        for identifier in all_identifiers:
            # Get all tax lots for this identifier
            identifier_lots = gain_loss[gain_loss['identifier'] == identifier]
            
            if identifier_lots.empty or identifier not in buys:
                continue
                
            # Sum up all sells for this identifier
            total_sells = pulp.lpSum(
                sells[lot['tax_lot_id']]
                for _, lot in identifier_lots.iterrows()
                if lot['tax_lot_id'] in sells
            )
            
            # Add binary variable to indicate if we're buying
            is_buying = pulp.LpVariable(f"is_buying_{identifier}", cat='Binary')
            
            # If is_buying is 1, we can buy any amount up to a large number M
            # If is_buying is 0, we must buy 0
            M = 1e6  # A large number that's bigger than any reasonable trade
            prob += buys[identifier] <= M * is_buying, f"Buy_Indicator_{identifier}"
            
            # If is_buying is 1, we cannot sell (total_sells must be 0)
            # If is_buying is 0, we can sell any amount
            prob += total_sells <= M * (1 - is_buying), f"No_Simultaneous_{identifier}" 