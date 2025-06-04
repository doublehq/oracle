from typing import Optional, Tuple
import pandas as pd
import pulp
from src.service.helpers.constants import CASH_CUSIP_ID
from src.service.constraints.base_validator import BaseValidator

class DriftValidator(BaseValidator):
    """
    Validator for drift-related constraints at the asset class level.
    Note: Individual trade validation is not supported as drift validation requires knowledge of all trades.
    """
    
    def __init__(self, oracle_strategy, range_min_weight_multiplier: float = 0.5, range_max_weight_multiplier: float = 2.0):
        """
        Initialize DriftValidator.
        
        Args:
            oracle_strategy: Reference to the OracleStrategy instance
            range_min_weight_multiplier: Minimum weight multiplier for range constraints
            range_max_weight_multiplier: Maximum weight multiplier for range constraints
        """
        super().__init__(oracle_strategy)
        self.range_min_weight_multiplier = range_min_weight_multiplier
        self.range_max_weight_multiplier = range_max_weight_multiplier
        
    def validate_buy(self, identifier: str, quantity: float) -> Tuple[bool, Optional[str]]:
        """Cannot validate individual buys without knowledge of all trades."""
        raise NotImplementedError("Drift validation requires knowledge of all trades")
        
    def validate_sell(self, tax_lot_id: str, quantity: float) -> Tuple[bool, Optional[str]]:
        """Cannot validate individual sells without knowledge of all trades."""
        raise NotImplementedError("Drift validation requires knowledge of all trades")
        
    def add_to_problem(
        self,
        prob: pulp.LpProblem,
        buys: dict,
        sells: dict,
        drift: pd.DataFrame,
        buy_df: Optional[pd.DataFrame] = None,
        sell_df: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Add drift range constraints to the optimization problem at the asset class level.
        
        Args:
            prob: The optimization problem to add constraints to
            buys: Dictionary of buy variables
            sells: Dictionary of sell variables
            drift: DataFrame with drift report containing asset class information
            buy_df: Optional DataFrame with buy information for vectorized operations
            sell_df: Optional DataFrame with sell information for vectorized operations
        """
        total_value = self.strategy.total_value()

        # Process each asset class
        for _, row in drift.iterrows():
            asset_class = row['asset_class']
            if asset_class == CASH_CUSIP_ID:
                continue
                
            target_weight = row['target_weight']
            actual_weight = row['actual_weight']
            identifiers = row['identifiers']
            
            # Calculate total weight changes for all identifiers in this asset class using vectorized operations
            total_buy_weight_change = 0
            total_sell_weight_change = 0
            
            if buy_df is not None and not buy_df.empty:
                # Filter buys for this asset class's identifiers
                asset_buys = buy_df[buy_df['identifier'].isin(identifiers)]
                if not asset_buys.empty:
                    # Calculate total buy weight change using vectorized operations
                    total_buy_weight_change = sum(
                        buys[identifier] * price / total_value 
                        for identifier, price in zip(asset_buys['identifier'], asset_buys['price'])
                    )
            else:
                # Fallback to non-vectorized method for buys
                for identifier in identifiers:
                    price = self.strategy.prices.loc[self.strategy.prices['identifier'] == identifier, 'price'].iloc[0]
                    if identifier in buys:
                        total_buy_weight_change += buys[identifier] * price / total_value

            if sell_df is not None and not sell_df.empty:
                # Filter sells for this asset class's identifiers
                asset_sells = sell_df[sell_df['identifier'].isin(identifiers)]
                if not asset_sells.empty:
                    # Calculate total sell weight change using vectorized operations
                    total_sell_weight_change = sum(
                        sells[tax_lot_id] * price / total_value
                        for tax_lot_id, price in zip(asset_sells['tax_lot_id'], asset_sells['price'])
                    )
            else:
                # Fallback to non-vectorized method for sells
                for tax_lot_id, value in sells.items():
                    # Check if this tax lot belongs to this identifier
                    if self.strategy.gain_loss_report.loc[self.strategy.gain_loss_report['tax_lot_id'] == tax_lot_id, 'identifier'].iloc[0] in identifiers:
                        price = self.strategy.prices.loc[self.strategy.prices['identifier'] == self.strategy.gain_loss_report.loc[self.strategy.gain_loss_report['tax_lot_id'] == tax_lot_id, 'identifier'].iloc[0], 'price'].iloc[0]
                        total_sell_weight_change += sells[tax_lot_id] * price / total_value
            
            # Calculate new weight for entire asset class after all trades
            new_weight = actual_weight + total_buy_weight_change - total_sell_weight_change
            
            # Calculate min/max weight constraints at asset class level
            min_weight_multiplier = getattr(self, 'range_min_weight_multiplier', None)
            max_weight_multiplier = getattr(self, 'range_max_weight_multiplier', None)
            
            min_weight = min_weight_multiplier * target_weight if min_weight_multiplier is not None else None
            max_weight = max_weight_multiplier * target_weight if max_weight_multiplier is not None else None
            # Only apply constraints if the asset class has a position and target
            if actual_weight > 0 and target_weight > 0:
                # Skip if current weight is already outside bounds (to avoid infeasible problems)
                if actual_weight > 0 and min_weight and actual_weight < min_weight:
                    # Make it so we can only buy the thing, not sell it more
                    prob += total_sell_weight_change == 0, f"no_sells_below_min_{asset_class}"
                elif actual_weight > 0 and max_weight and actual_weight > max_weight:
                    # Make it so we can only sell the thing, not buy it more
                    prob += total_buy_weight_change == 0, f"no_buys_above_max_{asset_class}"
                else:
                    if min_weight is not None:
                        # Add minimum weight constraint for the asset class
                        prob += new_weight >= min_weight, f"min_weight_{asset_class}"
                    if max_weight is not None:
                        # Add maximum weight constraint for the asset class
                        prob += new_weight <= max_weight, f"max_weight_{asset_class}"

    