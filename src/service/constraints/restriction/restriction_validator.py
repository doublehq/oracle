from typing import Optional, Tuple, List
import pandas as pd
import pulp

from src.service.constraints.base_validator import BaseValidator

class RestrictionValidator(BaseValidator):
    """Validator for stock and wash sale restrictions."""
    
    def __init__(self, strategy, enforce_wash_sale_prevention: bool = True):
        """Initialize the validator.
        
        Args:
            strategy: The strategy object containing portfolio information
            enforce_wash_sales: If False, wash sale restrictions will be ignored
        """
        super().__init__(strategy)
        self.enforce_wash_sale_prevention = enforce_wash_sale_prevention

    def validate_buy(self, identifier: str, quantity: float) -> Tuple[bool, Optional[str]]:
        """Check if buying a security is allowed by restrictions."""
        # Check stock restrictions
        if self.strategy.oracle.stock_restrictions is not None:
            restrictions = self.strategy.oracle.stock_restrictions
            if identifier in restrictions["identifier"].values:
                if not restrictions.loc[identifier, 'can_buy']:
                    return False, f"Security {identifier} is restricted from buying"
                    
        # Check wash sale restrictions
        if self.enforce_wash_sale_prevention and self.strategy.oracle.wash_sale_restrictions is not None:
            if self.strategy.oracle.wash_sale_restrictions.is_restricted_from_buying(identifier):
                return False, f"Security {identifier} is restricted due to wash sale rules"
                
        return True, None
        
    def validate_sell(self, tax_lot_id: str, quantity: float) -> Tuple[bool, Optional[str]]:
        """Check if selling a tax lot is allowed by restrictions."""
        # Get the identifier for this tax lot
        lot_info = self.strategy.tax_lots[self.strategy.tax_lots['tax_lot_id'] == tax_lot_id].iloc[0]
        identifier = lot_info['identifier']
        
        # Check stock restrictions
        if self.strategy.oracle.stock_restrictions is not None:
            restrictions = self.strategy.oracle.stock_restrictions
            if identifier in restrictions["identifier"].values:
                if not restrictions.loc[restrictions["identifier"] == identifier, 'can_sell'].iloc[0]:
                    return False, f"Security {identifier} is restricted from selling"
                    
        # Check wash sale restrictions
        if self.enforce_wash_sale_prevention and self.strategy.oracle.wash_sale_restrictions is not None:
            restricted_lots = self.strategy.oracle.wash_sale_restrictions.get_restricted_lots(identifier)
            if not restricted_lots.empty:
                for _, lot in restricted_lots.iterrows():
                    if lot['tax_lot_id'] == tax_lot_id:
                        return False, f"Tax lot {tax_lot_id} is restricted due to wash sale rules"
                
        return True, None
        
    def add_to_problem(
        self,
        prob: pulp.LpProblem,
        buys: dict,
        sells: dict,
        gain_loss: pd.DataFrame,
        stock_restrictions: pd.DataFrame,
        wash_sale_restrictions,
        all_identifiers: List[str]
    ) -> None:
        """Add stock and wash sale restrictions to the optimization problem."""
        # Add stock restrictions
        if stock_restrictions is not None:
            for _, row in stock_restrictions.iterrows():
                identifier = row['identifier']
                if not row['can_buy']:
                    if identifier in buys:
                        prob += (buys[identifier] == 0), f"no_buy_{identifier}"
                if not row['can_sell']:
                    for _, lot in gain_loss[gain_loss['identifier'] == identifier].iterrows():
                        if lot['tax_lot_id'] in sells:
                            prob += (sells[lot['tax_lot_id']] == 0), f"no_sell_{lot['tax_lot_id']}"
                                
        # Wash sale restrictions
        if self.enforce_wash_sale_prevention and wash_sale_restrictions is not None:
            for identifier in all_identifiers:
                # Check buy restrictions
                if wash_sale_restrictions.is_restricted_from_buying(identifier):
                    if identifier in buys:
                        prob += (buys[identifier] == 0), f"wash_sale_buy_{identifier}"
                
                # Check sell restrictions - get all restricted lots for this identifier
                restricted_lots = wash_sale_restrictions.get_restricted_lots(identifier)
                if not restricted_lots.empty:
                    # Add binary variable to track if liquidating
                    liquidate = pulp.LpVariable(f"liquidate_{identifier}", cat='Binary')

                    all_tax_lots = self.strategy.oracle.all_tax_lots
                    all_quantity = all_tax_lots.loc[all_tax_lots['identifier'] == identifier, 'quantity'].sum()

                    identifier_sells = []
                    for _, lot in gain_loss[gain_loss['identifier'] == identifier].iterrows():
                        if lot['tax_lot_id'] in sells:
                            identifier_sells.append(sells[lot['tax_lot_id']])

                    prob += (pulp.lpSum(identifier_sells) >= (all_quantity * liquidate)), f"wash_sale_liquidate_{identifier}"

                    for _, lot in restricted_lots.iterrows():
                        if lot['tax_lot_id'] in sells:
                            prob += (sells[lot['tax_lot_id']] == (lot["quantity"] * liquidate)), f"wash_sale_sell_{lot['tax_lot_id']}"
