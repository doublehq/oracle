from typing import Optional, Tuple
import pandas as pd
import pulp
from datetime import timedelta

from src.service.constraints.base_validator import BaseValidator
from src.service.constraints.holding_time.trading_day_lookup import TradingDayLookup

class HoldingTimeValidator(BaseValidator):
    """Validator for minimum holding time requirements."""
    
    def __init__(self, oracle_strategy, holding_time_delta: timedelta):
        """
        Initialize HoldingTimeValidator.
        
        Args:
            oracle_strategy: Reference to the OracleStrategy instance
            holding_time_delta: Minimum holding time required
        """
        super().__init__(oracle_strategy)
        self.holding_time_delta = holding_time_delta
        self.trading_day_lookup = TradingDayLookup()
        self._last_current_date = self.strategy.oracle.current_date
        self._before_date = self._calculate_before_date(self._last_current_date)
        
    def _calculate_before_date(self, current_date: pd.Timestamp) -> pd.Timestamp:
        """
        Calculate the before date for holding time calculations.
        
        Args:
            current_date: The current date to calculate from
            
        Returns:
            The effective before date for holding time calculations
        """
        target_date = current_date - self.holding_time_delta
        
        # Get the trading day information
        trading_day_info = self.trading_day_lookup.get_trading_day(target_date)
        
        # If no trading day info is found, use the target date directly
        if trading_day_info is None:
            return target_date
            
        # If the date is not a trading day, find the nearest trading day before the current date
        if trading_day_info['date'] != trading_day_info['nearest_trading_day']:
            return pd.to_datetime(trading_day_info['backward_trading_day'])
        
        return pd.to_datetime(trading_day_info['date'])
        
    def _get_before_date(self, current_date: pd.Timestamp) -> pd.Timestamp:
        """
        Get the before date for holding time calculations, with caching.
        
        Args:
            current_date: The current date to calculate from
            
        Returns:
            The effective before date for holding time calculations
        """
        # If we've already calculated for this current_date, return cached value
        if self._last_current_date == current_date and self._before_date is not None:
            return self._before_date
            
        # Calculate and cache new value
        self._last_current_date = current_date
        self._before_date = self._calculate_before_date(current_date)
        return self._before_date
        
    def validate_buy(self, identifier: str, quantity: float) -> Tuple[bool, Optional[str]]:
        """Buying is always allowed with respect to holding time."""
        return True, None
        
    def validate_sell(self, tax_lot_id: str, quantity: float) -> Tuple[bool, Optional[str]]:
        """Check if a tax lot has been held long enough to sell."""
        if self.holding_time_delta is None or self.holding_time_delta <= timedelta(days=0):
            return True, None
            
        # Get the lot information
        lot_info = self.strategy.tax_lots[self.strategy.tax_lots['tax_lot_id'] == tax_lot_id].iloc[0]
        purchase_date = lot_info['date'].date()
        
        # Get the before date using the cached method
        before_date = self._get_before_date(self.strategy.oracle.current_date)
        
        # If the purchase date is after or equal to the before_date, the lot cannot be sold
        if purchase_date >= before_date.date():
            days_remaining = (self.holding_time_delta - (self.strategy.oracle.current_date - purchase_date)).days
            return False, f"Tax lot must be held for {days_remaining + 1} more days"
            
        return True, None
        
    def add_to_problem(
        self,
        prob: pulp.LpProblem,
        sells: dict,
        tax_lots: pd.DataFrame,
        current_date: pd.Timestamp
    ) -> None:
        """Add holding time constraints to the optimization problem."""
        if self.holding_time_delta is None or self.holding_time_delta <= timedelta(days=0):
            return
        
        # Get the before date using the cached method
        before_date = self._get_before_date(current_date)
        
        # Find tax lots acquired within the holding time window
        recently_bought_lots = tax_lots[
            pd.to_datetime(tax_lots["date"]).dt.date >= before_date.date()
        ]
        
        # Add constraint to prevent selling these lots
        for _, lot in recently_bought_lots.iterrows():
            tax_lot_id = lot['tax_lot_id']
            if tax_lot_id in sells:
                prob += (
                    sells[tax_lot_id] == 0,
                    f"No_sell_recently_bought_{tax_lot_id}"
                )