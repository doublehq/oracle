import pandas as pd
from datetime import date
from pathlib import Path
import json
from typing import Dict, Optional

class TradingDayLookup:
    """Class to handle trading day lookups from JSON data."""
    
    def __init__(self, trading_days_df: pd.DataFrame = None):
        """
        Initialize TradingDayLookup with either a DataFrame or load from JSON.
        
        Args:
            trading_days_df: Optional DataFrame containing trading day data
        """
        self.trading_days_df = trading_days_df
        self._lookup_cache: Dict[str, pd.Series] = {}
        if trading_days_df is None:
            self._load_from_json()
            
    def _load_from_json(self):
        """Load trading days data from JSON file."""
        json_path = Path(__file__).parent / 'trading_day.json'
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        # Convert JSON to DataFrame
        self.trading_days_df = pd.DataFrame(data)
        # Convert date columns to datetime
        date_columns = ['nearest_trading_day', 'forward_trading_day', 'backward_trading_day', 'date']
        for col in date_columns:
            self.trading_days_df[col] = pd.to_datetime(self.trading_days_df[col])
            
    def get_trading_day(self, target_date: date) -> Optional[pd.Series]:
        """
        Get trading day information for a given date.
        
        Args:
            target_date: Date to lookup
            
        Returns:
            Series containing trading day information or None if not found
        """
        if self.trading_days_df is None:
            return None
            
        # Convert target_date to datetime for comparison and get string key for cache
        target_date_ts = pd.to_datetime(target_date)
        cache_key = target_date_ts.strftime('%Y-%m-%d')
        
        # Check cache first
        if cache_key in self._lookup_cache:
            return self._lookup_cache[cache_key]
            
        # Find the matching row
        matching_days = self.trading_days_df[self.trading_days_df['date'] == target_date_ts]
        
        if matching_days.empty:
            self._lookup_cache[cache_key] = None
            return None
            
        # Cache and return result
        result = matching_days.iloc[0]
        self._lookup_cache[cache_key] = result
        return result 