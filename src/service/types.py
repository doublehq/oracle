"""Type definitions for the service package."""
from typing import TypeVar, Union, Tuple
import pandas as pd
from datetime import date, datetime

# Create a type variable for OracleStrategy
OracleStrategy = TypeVar('OracleStrategy', bound='service.oracle_strategy.OracleStrategy')

# Define the return type for apply_trades_to_portfolio
ApplyTradesReturn = Union[
    Tuple[pd.DataFrame, float, pd.DataFrame],
    OracleStrategy
] 