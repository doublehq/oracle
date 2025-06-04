import pandas as pd
from datetime import date, timedelta
from typing import Optional, List, Dict, Set, Literal
from enum import Enum

class WashSaleReason(str, Enum):
    """Enum for different types of wash sale restrictions"""
    BUY_SELL_BUY = "buy-sell-buy"  # Bought, sold at loss, then bought again
    BUY_BUY_SELL = "buy-buy-sell"  # Bought, bought more, then sold first lot at loss

class WashSaleRestrictions:
    """
    Class to handle wash sale restrictions for tax-aware portfolio optimization.
    
    Wash sale rules prevent claiming a loss on a security if a "substantially identical"
    security is purchased within 30 days before or after the sale (inclusive).
    
    The class tracks two types of restrictions:
    1. Buy restrictions: Identifiers that cannot be bought due to wash sale rules
    2. Sell restrictions: Specific tax lots that cannot be sold due to wash sale rules
    """
    
    def __init__(
        self,
        current_date: date,
        all_tax_lots: pd.DataFrame,
        prices: pd.DataFrame,
        recently_closed_lots: Optional[pd.DataFrame] = None,
        wash_window_days: int = 30,
        percentage_protection_from_inadvertent_wash_sales: float = 0.001,
    ):
        """
        Initialize wash sale restrictions.
        
        Args:
            current_date: The date to use for determining active restrictions
            prices: DataFrame with columns ['identifier', 'price'] containing current prices
            recently_closed_lots: DataFrame with columns ['identifier', 'quantity', 'cost_basis', 
                                'date_acquired', 'date_sold', 'proceeds', 'realized_gain']
            all_tax_lots: DataFrame containing all current tax lots across all strategies
            wash_window_days: Number of days to consider for wash sale restrictions (inclusive, default: 30)
        """
        self.current_date = current_date
        self.wash_window_days = wash_window_days
        self.percentage_protection_from_inadvertent_wash_sales = percentage_protection_from_inadvertent_wash_sales
        # Validate prices DataFrame has required columns
        if not {'identifier', 'price'}.issubset(prices.columns):
            raise ValueError("prices DataFrame must have columns ['identifier', 'price']")
        self.prices = prices
        
        # Initialize restricted_from_buying DataFrame
        self.restricted_from_buying = pd.DataFrame(columns=[
            'identifier',
            'reason',  # WashSaleReason enum
            'restriction_ends_after'  # date after which restriction ends (exclusive)
        ])
        
        # Initialize restricted_from_selling DataFrame (lot-level)
        self.restricted_from_selling = pd.DataFrame(columns=[
            'identifier',
            'tax_lot_id',  # Unique identifier for the tax lot
            'quantity',
            'cost_basis',
            'date_acquired',  # date
            'reason',  # WashSaleReason enum
            'restriction_ends_after',  # date after which restriction ends (exclusive)
            'price'  # current price of the security
        ])
        
        if recently_closed_lots is None:
            recently_closed_lots = pd.DataFrame(columns=[
                'identifier', 'quantity', 'cost_basis', 
                'date_acquired', 'date_sold', 'proceeds', 'realized_gain'
            ])
        self._process_closed_lots(recently_closed_lots, all_tax_lots)
    
    def _identify_buy_restrictions(
        self,
        closed_lots: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Identify securities that cannot be bought due to recent loss sales.
        Uses vectorized operations for efficiency. For each identifier, gets the
        furthest restriction_ends_after from all its loss sales.
        
        Args:
            closed_lots: DataFrame of closed tax lots
            
        Returns:
            DataFrame of buy restrictions with columns [identifier, reason, restriction_ends_after],
            where restriction_ends_after is the latest date for each identifier
        """
        # Filter to just loss sales
        loss_sales = closed_lots[closed_lots['realized_gain'] < 0].copy()
        if loss_sales.empty:
            return pd.DataFrame(columns=['identifier', 'reason', 'restriction_ends_after'])
            
        # Calculate restriction end dates (inclusive 30 days)
        loss_sales['restriction_ends_after'] = (
            pd.to_datetime(loss_sales['date_sold'])
            .dt.normalize() + pd.Timedelta(days=self.wash_window_days)
        ).dt.date
        
        # Filter to active restrictions (inclusive today)
        active_restrictions = loss_sales[loss_sales['restriction_ends_after'] > self.current_date]
        if active_restrictions.empty:
            return pd.DataFrame(columns=['identifier', 'reason', 'restriction_ends_after'])
        
        # Group by identifier and get the maximum restriction_ends_after
        buy_restrictions = (
            active_restrictions
            .groupby('identifier')
            .agg({'restriction_ends_after': 'max'})
            .reset_index()
        )
        
        # Add the reason column
        buy_restrictions['reason'] = WashSaleReason.BUY_SELL_BUY
        
        return buy_restrictions[['identifier', 'reason', 'restriction_ends_after']].sort_values(
            by='identifier', ascending=False)

    def _identify_sell_restrictions(
        self,
        all_tax_lots: pd.DataFrame,
        closed_lots: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Identify tax lots that cannot be sold due to multiple recent purchases.
        A lot is restricted from selling only if there are multiple purchases
        of the same security within the wash window (inclusive 30 days).
        The restriction expires after the second earliest purchase moves out of the window,
        leaving only one purchase within the window.
        
        Args:
            all_tax_lots: DataFrame containing all current tax lots across all strategies
            
        Returns:
            DataFrame of sell restrictions with columns [identifier, tax_lot_id, quantity, 
            cost_basis, date_acquired, reason, restriction_ends_after, price]
        """
        if all_tax_lots.empty:
            return pd.DataFrame(columns=['identifier', 'tax_lot_id', 'quantity', 'cost_basis', 'date_acquired', 'reason', 'restriction_ends_after', 'price'])
        all_lots_copy = all_tax_lots.copy()
        all_lots_copy['date'] = pd.to_datetime(all_lots_copy['date']).dt.date
        all_lots_copy.rename(columns={'date': 'date_acquired'}, inplace=True)

        # Find lots purchased within the wash window (inclusive)
        recent_purchases = all_tax_lots[
            pd.to_datetime(all_tax_lots['date']).dt.date >= self.current_date - timedelta(days=self.wash_window_days)
        ].copy()
        
        if recent_purchases.empty:
            return pd.DataFrame(columns=['identifier', 'tax_lot_id', 'quantity', 'cost_basis', 'date_acquired', 'reason', 'restriction_ends_after', 'price'])
        
        # Convert date_acquired to datetime.date
        recent_purchases['date'] = pd.to_datetime(recent_purchases['date']).dt.date
        # Rename date to date_acquired for consistency
        recent_purchases.rename(columns={'date': 'date_acquired'}, inplace=True)

        lots_with_recent_purchases = all_lots_copy.merge(recent_purchases, how="inner", on="identifier", suffixes=("", "_recent_purchase"))
        lots_with_recent_purchases = lots_with_recent_purchases[lots_with_recent_purchases["tax_lot_id"] != lots_with_recent_purchases["tax_lot_id_recent_purchase"]]

        restriction_ends_after = lots_with_recent_purchases.groupby("tax_lot_id")["date_acquired_recent_purchase"].max() + timedelta(days=self.wash_window_days)
        potentially_restricted_lots = all_lots_copy.merge(restriction_ends_after.rename("restriction_ends_after"), how="inner", on="tax_lot_id")


        # Add current prices
        restricted_lots = potentially_restricted_lots.merge(
            self.prices[['identifier', 'price']],
            on='identifier',
            how='left'
        )
        restricted_lots["current_value"] = round((restricted_lots['price'] * restricted_lots['quantity']),2)
        restricted_lots["adjusted_current_value"] = round(restricted_lots["current_value"] * (1-self.percentage_protection_from_inadvertent_wash_sales),2)
        # Only restrict lots where the current market value is lower than the cost basis
        # (we only care about potential losses for wash sale purposes)
        restricted_lots = restricted_lots[restricted_lots['adjusted_current_value'] <= restricted_lots['cost_basis'] * (1 + 0.0000001)]
        restricted_identifiers = restricted_lots['identifier'].unique()
        # If we've filtered out all lots, return empty DataFrame
        if restricted_lots.empty:
            return pd.DataFrame(columns=['identifier', 'tax_lot_id', 'quantity', 'cost_basis', 'date_acquired', 'reason', 'restriction_ends_after', 'price'])
        
        # Add reason for sell restriction
        restricted_lots['reason'] = WashSaleReason.BUY_BUY_SELL
        
        # Select and order columns
        return restricted_lots[[
            'identifier', 'tax_lot_id', 'quantity', 'cost_basis', 
            'date_acquired', 'reason', 'restriction_ends_after', 'price'
        ]].sort_values(by='identifier', ascending=False)

    def _process_closed_lots(
        self,
        closed_lots: pd.DataFrame,
        all_tax_lots: pd.DataFrame
    ) -> None:
        """
        Process closed lots to identify wash sale restrictions.
        Also considers all current tax lots across strategies.
        
        Args:
            closed_lots: DataFrame with closed tax lots
            all_tax_lots: DataFrame containing all current tax lots across all strategies
        """
        required_columns = {
            'identifier', 'quantity', 'cost_basis', 'date_acquired',
            'date_sold', 'proceeds', 'realized_gain'
        }
        if not set(closed_lots.columns).issuperset(required_columns):
            raise ValueError(f"Closed lots DataFrame missing required columns: {required_columns}")
        
        # Identify securities that cannot be bought (due to recent loss sales)
        buy_restrictions = self._identify_buy_restrictions(closed_lots)
        self.restricted_from_buying = pd.concat(
            [self.restricted_from_buying, buy_restrictions],
            ignore_index=True
        ).drop_duplicates()
        
        # Identify tax lots that cannot be sold (due to recent purchases)
        sell_restrictions = self._identify_sell_restrictions(all_tax_lots, closed_lots)
        self.restricted_from_selling = pd.concat(
            [self.restricted_from_selling, sell_restrictions],
            ignore_index=True
        ).drop_duplicates()
    
    def is_restricted_from_buying(self, identifier: str) -> bool:
        """
        Check if a security is restricted from buying.
        
        Args:
            identifier: Security identifier to check
            
        Returns:
            True if the security cannot be bought, False otherwise
        """
        # Check if identifier exists and restriction hasn't ended
        if identifier not in self.restricted_from_buying['identifier'].values:
            return False
            
        restriction = self.restricted_from_buying[
            self.restricted_from_buying['identifier'] == identifier
        ].iloc[0]
        return self.current_date <= restriction['restriction_ends_after']
    
    def get_buy_restriction_reason(self, identifier: str) -> Optional[WashSaleReason]:
        """
        Get the reason why a security is restricted from buying.
        
        Args:
            identifier: Security identifier to check
            
        Returns:
            WashSaleReason enum value if restricted, None otherwise
        """
        if not self.is_restricted_from_buying(identifier):
            return None
            
        restriction = self.restricted_from_buying[
            (self.restricted_from_buying['identifier'] == identifier) &
            (self.restricted_from_buying['restriction_ends_after'] >= self.current_date)
        ].iloc[0]
        return restriction['reason']
    
    def is_lot_restricted_from_selling(self, identifier: str, tax_lot_id: str) -> bool:
        """
        Check if a specific tax lot is restricted from selling.
        
        Args:
            identifier: Security identifier
            tax_lot_id: Unique identifier for the tax lot
            
        Returns:
            True if the lot cannot be sold, False otherwise
        """
        matching_lots = self.restricted_from_selling[
            (self.restricted_from_selling['identifier'] == identifier) &
            (self.restricted_from_selling['tax_lot_id'] == tax_lot_id)
        ]
        
        if matching_lots.empty:
            return False
            
        # Check if the restriction is still active (current_date is within the restriction period)
        return self.current_date <= matching_lots.iloc[0]['restriction_ends_after']
    
    def get_restricted_lots(self, identifier: str) -> pd.DataFrame:
        """
        Get all restricted lots for a given security.
        Only returns lots where the restriction is still active.
        
        Args:
            identifier: Security identifier
            
        Returns:
            DataFrame containing all restricted lots for the security where
            the restriction is still active
        """
        return self.restricted_from_selling[
            (self.restricted_from_selling['identifier'] == identifier) &
            (self.restricted_from_selling['restriction_ends_after'] >= self.current_date)
        ].copy()

    def get_all_restricted_buys(self) -> Set[str]:
        """
        Get all identifiers that are currently restricted from buying.
        
        Returns:
            Set of identifiers that cannot be bought due to active wash sale restrictions
        """
        # Get identifiers with active buy restrictions
        active_buy_restrictions = self.restricted_from_buying[
            self.restricted_from_buying['restriction_ends_after'] >= self.current_date
        ]['identifier'].unique()
        
        return set(sorted(active_buy_restrictions))
    
    def get_all_restricted_sells(self) -> pd.DataFrame:
        """
        Get all specific tac lots that have any active sell restrictions.
        
        Returns:
            Dataframe of restricted lots with columns [identifier, tax_lot_id, quantity,
            cost_basis, date_acquired, reason, restriction_ends_after]
        """
        # Get identifiers with active sell restrictions
        active_sell_restrictions = self.restricted_from_selling[
            self.restricted_from_selling['restriction_ends_after'] >= self.current_date
        ]
        return active_sell_restrictions.sort_values(by=['identifier', 'tax_lot_id'], ascending=False)
    
    def get_restricted_identifiers(self) -> Set[str]:
        """
        Get all identifiers that have any active restrictions (buy or sell).
        
        Returns:
            Set of identifiers that are either restricted from buying or have
            lots restricted from selling
        """
        # Get active buy restrictions
        active_buy_restrictions = self.restricted_from_buying[
            self.restricted_from_buying['restriction_ends_after'] >= self.current_date
        ]['identifier'].unique()
        
        # Get active sell restrictions
        active_sell_restrictions = self.restricted_from_selling[
            self.restricted_from_selling['restriction_ends_after'] >= self.current_date
        ]['identifier'].unique()
        
        # Combine both sets of identifiers
        return set(active_buy_restrictions) | set(active_sell_restrictions)
