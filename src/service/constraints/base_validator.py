from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

class BaseValidator(ABC):
    """Base class for all constraint validators."""
    
    def __init__(self, oracle_strategy):
        """Initialize validator with reference to OracleStrategy."""
        self.strategy = oracle_strategy
    
    @abstractmethod
    def validate_buy(self, identifier: str, quantity: float) -> Tuple[bool, Optional[str]]:
        """
        Validate if a buy trade is allowed.
        
        Args:
            identifier: The security identifier
            quantity: The quantity to buy
            
        Returns:
            Tuple of (is_allowed, reason)
            - is_allowed: True if trade is allowed, False otherwise
            - reason: None if allowed, otherwise a string explaining why it's not allowed
        """
        pass
        
    @abstractmethod
    def validate_sell(self, tax_lot_id: str, quantity: float) -> Tuple[bool, Optional[str]]:
        """
        Validate if a sell trade is allowed.
        
        Args:
            tax_lot_id: The tax lot identifier
            quantity: The quantity to sell
            
        Returns:
            Tuple of (is_allowed, reason)
            - is_allowed: True if trade is allowed, False otherwise
            - reason: None if allowed, otherwise a string explaining why it's not allowed
        """
        pass 