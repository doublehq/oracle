import os
import numpy as np
import pandas as pd
from datetime import date, datetime
from functools import cached_property
from typing import List, Optional, Dict, Tuple
from pathlib import Path
import json
import pulp

from src.service.initializers import (
    initialize_closed_lots,
    initialize_stock_restrictions,
    initialize_tax_rates,
)

from src.service.constraints.restriction.wash_sale_restrictions import WashSaleRestrictions
from src.service.oracle_strategy import OracleStrategy
from src.service.helpers.enums import OracleOptimizationType
from src.service.helpers.constants import logger
from src.service.helpers.trade_netting import net_trades_across_strategies

class Oracle:
    
    def __init__(
        self, 
        current_date: date,
        recently_closed_lots: pd.DataFrame = None,
        stock_restrictions: pd.DataFrame = None,
        tax_rates: pd.DataFrame = None,
        strategies: Optional[List[OracleStrategy]] = None
    ):
        """
        Initialize Oracle with portfolio data.
        
        Args:
            current_date: The current date to use for calculations and restrictions
            recently_closed_lots: Optional DataFrame of recently closed tax lots
            stock_restrictions: Optional DataFrame specifying which stocks can be bought/sold
            tax_rates: Optional DataFrame of tax rates with columns [type, rate]
            strategies: Optional list of OracleStrategy instances
        """
        # Initialize basic data first
        self.current_date = current_date
        self.strategies = strategies or []
        
        # Set Oracle reference in each strategy
        for strategy in self.strategies:
            strategy.set_oracle(self)

        # Initialize closed lots
        self.recently_closed_lots = initialize_closed_lots(recently_closed_lots)
        
        # Initialize stock restrictions and tax rates
        self.stock_restrictions = initialize_stock_restrictions(stock_restrictions)
        self.tax_rates = initialize_tax_rates(tax_rates)
        
        # Initialize wash sale restrictions as None - will be set up later
        self.wash_sale_restrictions = None

    def add_strategy(
        self, 
        tax_lots: pd.DataFrame,
        targets: pd.DataFrame,
        prices: pd.DataFrame,
        cash: float,
        spreads: Optional[pd.DataFrame] = None,
        factor_model: Optional[pd.DataFrame] = None,
        withdrawal_amount: float = 0.0,
        optimization_type: OracleOptimizationType = OracleOptimizationType.TAX_AWARE,
        strategy_id: Optional[int] = None
    ) -> OracleStrategy:
        """
        Create a new strategy and add it to the Oracle.
        
        Args:
            tax_lots: DataFrame of current tax lots
            targets: DataFrame of target weights
            prices: DataFrame of current prices
            cash: Current cash balance
            spreads: Optional DataFrame of bid-ask spreads
            factor_model: Optional DataFrame for factor model optimization
            withdrawal_amount: Amount to withdraw from the portfolio (default 0.0)
            optimization_type: The type of optimization strategy to use
            strategy_id: Optional unique identifier for the strategy
            
        Returns:
            The newly created OracleStrategy instance
        """
        # Create a new strategy instance
        strategy = OracleStrategy(
            tax_lots=tax_lots,
            targets=targets,
            prices=prices,
            cash=cash,
            spreads=spreads,
            factor_model=factor_model,
            withdrawal_amount=withdrawal_amount,
            optimization_type=optimization_type,
            strategy_id=strategy_id
        )
        
        # Set this Oracle instance as the strategy's oracle
        strategy.set_oracle(self)
        
        # Add the strategy to our list
        self.strategies.append(strategy)
        
        return strategy
    
    def set_restrictions(self, stock_restrictions: pd.DataFrame) -> None:
        """
        Set stock trading restrictions for the Oracle.
        
        This method updates the stock trading restrictions that will be applied across
        all strategies managed by this Oracle instance.
        
        Args:
            stock_restrictions (pd.DataFrame): DataFrame specifying which stocks can be
                bought/sold. The DataFrame should contain columns defining the trading
                restrictions for each security.
                
        Note:
            The restrictions are initialized using the initialize_stock_restrictions
            helper function, which ensures the data is in the correct format and
            contains all necessary information for enforcing trading restrictions.
        """
        self.stock_restrictions = initialize_stock_restrictions(stock_restrictions)

    def _gather_all_tax_lots(self) -> pd.DataFrame:
        """
        Gather all tax lots from all strategies into a single DataFrame.
        This is needed for proper wash sale restriction evaluation.
        
        Returns:
            DataFrame containing all tax lots across all strategies with columns:
            - identifier: Security identifier
            - quantity: Number of shares
            - cost_basis: Total cost basis
            - date_acquired: Date the lot was acquired
            - strategy_id: ID of the strategy owning this lot
        """
        if not self.strategies:
            return pd.DataFrame(columns=['identifier', 'quantity', 'cost_basis', 'date_acquired', 'strategy_id'])
            
        all_tax_lots = []
        for strategy in self.strategies:
            if hasattr(strategy, 'tax_lots') and not strategy.tax_lots.empty:
                # Add strategy_id to the tax lots
                strategy_lots = strategy.tax_lots.copy()
                strategy_lots['strategy_id'] = strategy.strategy_id if strategy.strategy_id is not None else -1
                all_tax_lots.append(strategy_lots)
        
        if not all_tax_lots:
            return pd.DataFrame(columns=['identifier', 'quantity', 'cost_basis', 'date_acquired', 'strategy_id'])
            
        return pd.concat(all_tax_lots, ignore_index=True)
    

    def _gather_all_prices(self) -> pd.DataFrame:
        """
        Gather all prices from all strategies into a single DataFrame.
        
        This method collects price information from all registered strategies while ensuring
        that duplicate prices for the same identifier are not included. It maintains price
        consistency across the entire portfolio.
        
        Returns:
            pd.DataFrame: A DataFrame containing unique prices across all strategies with columns:
                - identifier: Security identifier
                - price: Current price of the security
                
        Note:
            If no strategies are registered or no prices are available, returns an empty
            DataFrame with the appropriate columns. The method ensures price uniqueness by
            tracking seen identifiers and only including the first occurrence of each.
        """
        if not self.strategies:
            return pd.DataFrame(columns=['identifier', 'price'])
            
        all_prices = []
        seen_identifiers = set()
        
        for strategy in self.strategies:
            if hasattr(strategy, 'prices') and not strategy.prices.empty:
                # Filter out identifiers we've already seen
                strategy_prices = strategy.prices.copy()
                new_prices = strategy_prices[~strategy_prices['identifier'].isin(seen_identifiers)]
                
                # Update our set of seen identifiers
                seen_identifiers.update(new_prices['identifier'].tolist())
                
                # Add to our collection if we have any new prices
                if not new_prices.empty:
                    all_prices.append(new_prices)
        
        if not all_prices:
            return pd.DataFrame(columns=['identifier', 'price'])
            
        return pd.concat(all_prices, ignore_index=True)

    def initialize_wash_sale_restrictions(self, percentage_protection_from_inadvertent_wash_sales: float = 0.003) -> None:
        """
        Initialize wash sale restrictions using all tax lots from all strategies.
        This method should be called after strategies are set up and their tax lots are loaded.
        
        Args:
            percentage_protection_from_inadvertent_wash_sales (float): The percentage buffer to protect against
                inadvertent wash sales. Default is 0.3% (0.003). This adds a safety margin to wash sale calculations.
                
        Note:
            This method gathers all tax lots and prices from all strategies to create a comprehensive
            wash sale restriction system. It requires that strategies have been added to the Oracle
            and their tax lots are properly loaded.
        """
        # Gather all tax lots from all strategies
        self.all_tax_lots = self._gather_all_tax_lots()
        prices = self._gather_all_prices()
        
        # Initialize wash sale restrictions with all necessary data
        self.wash_sale_restrictions = WashSaleRestrictions(
            current_date=self.current_date,
            all_tax_lots=self.all_tax_lots,
            prices=prices,
            recently_closed_lots=self.recently_closed_lots,
            percentage_protection_from_inadvertent_wash_sales=percentage_protection_from_inadvertent_wash_sales
        )

    def compute_optimal_trades_for_all_strategies(
        self,
        settings: dict,
        debug: bool = True,
        dump: bool = False,
        **kwargs
    ) -> Tuple[Dict[int, Tuple[Optional[int], bool, Dict, pd.DataFrame]], pd.DataFrame]:
        """
        Compute optimal trades for all strategies managed by this Oracle instance.
        
        Args:
            settings (dict): A dictionary containing optimization settings for each strategy.
                           Must have a 'strategies' key with strategy-specific settings.
            debug (bool): If True, enables detailed logging of the optimization process.
                        Default is True.
            dump (bool): If True, dumps the optimization state for debugging.
                        Default is False.
            **kwargs: Additional keyword arguments passed to individual strategy optimizations.
            
        Returns:
            tuple: A tuple containing:
                - Dict[int, Tuple[Optional[int], bool, Dict, pd.DataFrame]]: A dictionary mapping
                  strategy IDs to their optimization results. Each result is a tuple containing:
                    * status: The optimization status (None if failed)
                    * should_trade: Boolean indicating if trades should be executed
                    * trade_summary: Dictionary containing trade summary information
                    * trades: DataFrame containing the detailed trades
                - pd.DataFrame: A DataFrame containing the netted trades across all strategies
                
        Note:
            This method aggregates trades across all strategies and performs trade netting
            to produce a final set of trades that considers the entire portfolio's needs.
            If debug is enabled, detailed logging will show the progress and results of
            each strategy's optimization.
        """
        if not self.strategies:
            return {}, pd.DataFrame()
            
        results = {}
        
        if debug:
            logger.info(f"Running compute_optimal_trades for {len(self.strategies)} strategies")
            
        for strategy in self.strategies:
            strategy_id = strategy.strategy_id
            if debug:
                logger.info(f"Computing optimal trades for strategy {strategy_id} ({strategy.optimization_type.value})")
                
            try:
                strategy_settings = settings["strategies"][str(strategy_id)]
                result = strategy.compute_optimal_trades(debug=debug, dump=dump, **strategy_settings)
                results[strategy_id] = result
                
                if debug:
                    status, should_trade, trade_summary, trades = result
                    status_str = pulp.LpStatus[status] if status is not None else 'None'
                    logger.info(f"Strategy {strategy_id} optimization completed:")
                    logger.info(f"  Status: {status_str}")
                    logger.info(f"  Should trade: {should_trade}")
                    logger.info(f"  Number of trades: {len(trades)}")
                    
            except Exception as e:
                logger.error(f"Error computing optimal trades for strategy {strategy_id}: {str(e)}")
                results[strategy_id] = (None, False, {}, pd.DataFrame())
                
        # Net trades across all strategies
        trade_rounding = min(strategy.trade_rounding for strategy in self.strategies)
        netted_trades = net_trades_across_strategies(results, trade_rounding=trade_rounding)
        
        if debug and not netted_trades.empty:
            logger.info(f"Generated {len(netted_trades)} netted trades across all strategies")
                
        return results, netted_trades

    def to_dict(self) -> dict:
        """
        Convert Oracle state to a dictionary for serialization.
        
        This method serializes the current state of the Oracle instance into a dictionary
        format suitable for storage or transmission. It handles the conversion of complex
        data types (like DataFrames and dates) into JSON-serializable formats.
        
        Returns:
            dict: A dictionary containing the Oracle state with the following keys:
                - current_date: The current date in ISO format
                - recently_closed_lots: List of dictionaries representing closed lots
                - stock_restrictions: List of dictionaries for stock trading restrictions
                - tax_rates: List of dictionaries containing tax rate information
                - strategies: List of dictionaries containing serialized strategy data
                - timestamp: Current timestamp in YYYYMMDD_HHMMSS format
                - identifier: Optional identifier (None by default)
                
        Note:
            Special handling is performed for date fields, converting them to appropriate
            string formats. DataFrames are converted to lists of dictionaries using the
            'records' orient.
        """
        # Convert tax_rates DataFrame to list of dictionaries
        tax_rates_list = []
        if self.tax_rates is not None:
            tax_rates_list = self.tax_rates.to_dict(orient="records")

        # Convert recently_closed_lots DataFrame to list of dictionaries with date strings
        recently_closed_lots_list = []
        if self.recently_closed_lots is not None and not self.recently_closed_lots.empty:
            # Convert timestamps to date strings
            df = self.recently_closed_lots.copy()
            df['date_acquired'] = df['date_acquired'].dt.date
            df['date_sold'] = df['date_sold'].dt.date
            recently_closed_lots_list = df.to_dict(orient="records")

        return {
            "current_date": self.current_date,
            "recently_closed_lots": recently_closed_lots_list,
            "stock_restrictions": self.stock_restrictions.to_dict(orient="records") if self.stock_restrictions is not None else None,
            "tax_rates": tax_rates_list,
            "strategies": [strategy.to_dict() for strategy in self.strategies],
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "identifier": None
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Oracle':
        """
        Create an Oracle instance from a dictionary representation.
        
        This class method deserializes an Oracle instance from a dictionary format,
        typically used when loading saved state or receiving data from an API.
        
        Args:
            data (dict): A dictionary containing Oracle state data with the following keys:
                - current_date: ISO format date string
                - recently_closed_lots: List of dictionaries representing closed lots
                - stock_restrictions: List of dictionaries for stock trading restrictions
                - tax_rates: List of dictionaries containing tax rate information
                - strategies: List of dictionaries containing strategy data
                - percentage_protection_from_inadvertent_wash_sales: Float value for wash sale protection
                
        Returns:
            Oracle: A new Oracle instance initialized with the provided data
            
        Note:
            The method handles date conversions from ISO format strings and properly
            initializes all components including strategies. It's the counterpart to
            the to_dict() method used for serialization.
        """
        recently_closed_lots = pd.DataFrame(data["recently_closed_lots"])
        if not recently_closed_lots.empty:
            recently_closed_lots["date_acquired"] = recently_closed_lots["date_acquired"].map(date.fromisoformat)
            recently_closed_lots["date_sold"] = recently_closed_lots["date_sold"].map(date.fromisoformat)

        oracle = cls(
            current_date=date.fromisoformat(data["current_date"]),
            recently_closed_lots=pd.DataFrame(recently_closed_lots),
            stock_restrictions=pd.DataFrame(data["stock_restrictions"]),
            tax_rates=pd.DataFrame(data["tax_rates"]),
            strategies=[OracleStrategy.from_dict(strategy) for strategy in (data["strategies"].values() if isinstance(data["strategies"], dict) else data["strategies"])],
        )

        # fixme: __init__
        oracle.initialize_wash_sale_restrictions(data["percentage_protection_from_inadvertent_wash_sales"])

        return oracle

    @classmethod
    def process_lambda_event(cls, event):
        """
        Process an AWS Lambda event to compute optimal trades and handle withdrawal amounts.
        
        This class method handles the processing of Lambda events, creating an Oracle instance
        from the event data, computing optimal trades, and optionally calculating maximum
        withdrawal amounts.
        
        Args:
            event (dict): The Lambda event dictionary containing:
                - oracle: Dictionary containing Oracle configuration
                - settings: Dictionary containing optimization settings
                - max_withdrawal_amount_settings: Optional dictionary for withdrawal calculations
                
        Returns:
            dict: A response dictionary containing:
                - version: Current version of the service
                - results: Dictionary mapping strategy IDs to their optimization results
                - netted_trades: List of netted trades across all strategies
                - max_withdrawal_amount_results: Optional dictionary with withdrawal calculations
                
        Note:
            The method handles both regular trade optimization and optional maximum withdrawal
            amount calculations. It ensures all trades are properly formatted and NaN values
            are replaced with None for JSON serialization.
        """
        oracle = cls.from_dict(event["oracle"])
        results, netted_trades = oracle.compute_optimal_trades_for_all_strategies(settings=event["settings"])

        for strategy_id, result in results.items():
            status, should_trade, trade_summary, trades = result
            results[strategy_id] = {
                "label": event["oracle"]["strategies"][str(strategy_id)].get("label"),
                "status": status,
                "should_trade": should_trade,
                "trades": trades.replace(np.nan, None).to_dict(orient="records"),
                "trade_summary": trade_summary,
            }

        response = {
            "version": os.environ.get("VERSION", "test"),
            "results": results,
            "netted_trades": netted_trades.replace(np.nan, None).to_dict(orient="records"),
        }

        # `compute_optimal_trades_for_all_strategies`/`compute_optimal_trades` needs to run before this
        # so that optimization parameters (`min_notional`, `should_tlh`, etc) get set on the strategy objects
        if "max_withdrawal_amount_settings" in event:
            max_withdrawal_amount_results = {}

            for strategy_id, settings in event["max_withdrawal_amount_settings"]["strategies"].items():
                strategy = next(strategy for strategy in oracle.strategies if strategy.strategy_id == int(strategy_id))
                max_withdrawal, trades = strategy.calculate_max_withdrawal_amount(**settings)
                max_withdrawal_amount_results[strategy.strategy_id] = {
                    "max_withdrawal": max_withdrawal,
                    "trades": trades.replace(np.nan, None).to_dict(orient="records"),
                }

            response["max_withdrawal_amount_results"] = max_withdrawal_amount_results

        return response
