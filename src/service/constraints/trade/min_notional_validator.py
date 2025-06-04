from collections import defaultdict
from typing import Optional, Tuple
import pandas as pd
import pulp

from src.service.constraints.base_validator import BaseValidator

class MinNotionalValidator(BaseValidator):
    """Validator for minimum notional trade constraints."""
    
    def __init__(self, oracle_strategy, min_notional: float):
        """
        Initialize MinNotionalValidator.
        
        Args:
            oracle_strategy: Reference to the OracleStrategy instance
            min_notional: Minimum notional amount for any trade (in dollars)
        """
        super().__init__(oracle_strategy)
        self.min_notional = min_notional
        
    def validate_buy(self, identifier: str, quantity: float) -> Tuple[bool, Optional[str]]:
        """Validate if a buy trade meets minimum notional requirements."""
        if self.min_notional <= 0:
            return True, None
            
        price = self.strategy.prices.loc[self.strategy.prices['identifier'] == identifier, 'price'].iloc[0]
        notional = quantity * price
        
        if notional < self.min_notional:
            return False, f"Trade notional ({notional:.2f}) below minimum ({self.min_notional})"
            
        return True, None
        
    def validate_sell(self, tax_lot_id: str, quantity: float) -> Tuple[bool, Optional[str]]:
        """Validate if a sell trade meets minimum notional requirements."""
        if self.min_notional <= 0:
            return True, None
            
        # Get the identifier from tax lots
        lot_info = self.strategy.tax_lots[self.strategy.tax_lots['tax_lot_id'] == tax_lot_id].iloc[0]
        identifier = lot_info['identifier']
        
        price = self.strategy.prices.loc[self.strategy.prices['identifier'] == identifier, 'price'].iloc[0]
        notional = quantity * price
        
        if notional < self.min_notional:
            return False, f"Trade notional ({notional:.2f}) below minimum ({self.min_notional})"
            
        return True, None
        
    def add_to_problem(
        self,
        prob: pulp.LpProblem,
        buys: dict,
        sells: dict,
        prices: pd.DataFrame,
        tax_lots: pd.DataFrame
    ) -> None:
        """Add minimum notional constraints to the optimization problem."""
        if self.min_notional <= 0:
            return
            
        # Add minimum notional constraints for buys
        for identifier, buy_var in buys.items():
            price = prices.loc[prices['identifier'] == identifier, 'price'].iloc[0]
            
            # Add binary variable to track if trade happens
            trade_happens = pulp.LpVariable(f"buy_happens_{identifier}", cat='Binary')
            
            # If trade_happens is 1, buy must be >= min_notional/price
            # If trade_happens is 0, buy must be 0
            prob += buy_var <= trade_happens * 1e6, f"Buy_Upper_{identifier}"  # Big M constraint
            prob += buy_var >= (self.min_notional / price) * trade_happens, f"Buy_Min_Notional_{identifier}"
            
        sells_by_identifier = defaultdict(list)

        # Add minimum notional constraints for sells
        for tax_lot_id, sell_var in sells.items():
            identifier = tax_lots.loc[tax_lots['tax_lot_id'] == tax_lot_id, "identifier"].iloc[0]
            sells_by_identifier[identifier].append(sell_var)

        for identifier, sell_vars in sells_by_identifier.items():
            price = prices.loc[prices['identifier'] == identifier, 'price'].iloc[0]
            max_sell = tax_lots.loc[tax_lots['identifier'] == identifier, "quantity"].sum()
            
            # Add binary variable to track if trade happens
            trade_happens = pulp.LpVariable(f"sell_happens_{identifier}", cat='Binary')
            
            sell_sum = pulp.lpSum(sell_vars)
            # If trade_happens is 1, sell must be >= min_notional/price
            # If trade_happens is 0, sell must be 0
            prob += sell_sum <= trade_happens * max_sell, f"Sell_Upper_{identifier}"
            prob += sell_sum >= (self.min_notional / price) * trade_happens, f"Sell_Min_Notional_{identifier}"
