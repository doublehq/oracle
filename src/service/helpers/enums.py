from enum import Enum
import pulp
from typing import Dict, Any, Tuple, Optional

class OracleOptimizationType(Enum):
    """Defines the optimization types available for OracleStrategy."""
    HOLD = "HOLD"  # No trades allowed
    BUY_ONLY = "BUY_ONLY"  # Only buys allowed, no sells
    TAX_UNAWARE = "TAX_UNAWARE"  # Rebalance towards targets ignoring tax implications
    TAX_AWARE = "TAX_AWARE"  # Rebalance considering tax implications (default)
    PAIRS_TLH = "PAIRS_TLH"  # Tax-Loss Harvesting for pairs
    DIRECT_INDEX = "DIRECT_INDEX"  # Direct Indexing strategy
    # Add other types as needed, e.g., TLH, DIRECT_INDEX
    
    @classmethod
    def from_string(cls, type_str: str) -> 'OracleOptimizationType':
        """Get enum member from string, case-insensitive."""
        for member in cls:
            if member.value.upper() == type_str.upper():
                return member
        raise ValueError(f"Invalid OracleOptimizationType: {type_str}") 
    
    def allows_sells(self) -> bool:
        """Check if the optimization type allows sell trades."""
        return self not in {self.BUY_ONLY, self.HOLD}
    
    def should_tlh(self) -> bool:
        """
        Determine if the optimization type supports Tax Loss Harvesting (TLH).
        
        Different optimization types handle TLH differently:
        - PAIRS_TLH: Specifically designed for tax-loss harvesting with pairs trading
        - DIRECT_INDEX: Supports TLH as part of direct indexing strategy
        - Other types (HOLD, BUY_ONLY, TAX_UNAWARE, TAX_AWARE): Never support TLH
        
        Returns:
            bool: True if the optimization type supports TLH operations
        """

        # PAIRS_TLH and DIRECT_INDEX always support TLH
        if self in {self.PAIRS_TLH, self.DIRECT_INDEX}:
            return True
        return False
    
    def setup_optimization(self, prob: pulp.LpProblem, buys: Dict, sells: Dict) -> None:
        """
        Setup optimization problem based on strategy.
        
        Args:
            prob: The optimization problem
            buys: Dictionary of buy variables
            sells: Dictionary of sell variables
            data_context: Dictionary with optimization context data
        """

        if self == self.BUY_ONLY:
            # Force sells to zero
            for sell_var in sells.values():
                prob += sell_var == 0, f"constraint_buy_only_{sell_var.name}"
        
        elif self == self.HOLD:
            # Force all variables to zero
            for buy_var in buys.values():
                prob += buy_var == 0, f"hold_buy_{buy_var.name}"
            for sell_var in sells.values():
                prob += sell_var == 0, f"hold_sell_{sell_var.name}"
    
    def adjust_weights(self, 
                      weight_tax: float, 
                      weight_drift: float, 
                      weight_transaction: float, 
                      weight_factor_model: float,
                      weight_cash_drag: float) -> Tuple[float, float, float, float, float]:
        """
        Adjust weights based on strategy type.
        
        Args:
            weight_tax: Weight for tax component
            weight_drift: Weight for drift component
            weight_transaction: Weight for transaction cost component
            weight_factor_model: Weight for factor model component
            weight_cash_drag: Weight for cash deployment component
            
        Returns:
            Tuple of adjusted weights
        """
        if self == self.TAX_UNAWARE:
            # Override tax weight to zero
            return 0, weight_drift, weight_transaction, 0, weight_cash_drag
        
        elif self == self.HOLD:
            # Weights don't matter for HOLD
            return 0, 0, 0, 0, 0
        
        elif self == self.DIRECT_INDEX:
            # Direct index might adjust weights to prioritize factor model
            return weight_tax, weight_drift, weight_transaction, weight_factor_model, weight_cash_drag
        
        else:
            # Default weights for TAX_AWARE, BUY_ONLY, PAIRS_TLH
            return weight_tax, weight_drift, weight_transaction, 0, weight_cash_drag
    
    def can_handle_withdrawal(self) -> bool:
        """
        Check if the strategy can handle withdrawals.
        
        Returns:
            True if the strategy can handle withdrawals, False otherwise
        """
        if self in {self.HOLD, self.BUY_ONLY}:
            return False
        return True