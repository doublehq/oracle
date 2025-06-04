import pandas as pd
from datetime import date
from typing import Optional

def generate_gain_loss_report(
    tax_lots: pd.DataFrame,
    prices: pd.DataFrame,
    current_date: date,
    tax_rates: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate a gain/loss report for all tax lots using current prices.
    
    Args:
        tax_lots: DataFrame of tax lots
        prices: DataFrame of current prices
        current_date: Current date for calculating holding period
        tax_rates: DataFrame with tax rates
        
    Returns:
        DataFrame with columns:
        - tax_lot_id: Unique identifier for the tax lot
        - identifier: Security identifier
        - quantity: Number of shares
        - cost_basis: Total cost basis for the lot
        - cost_per_share: Cost basis per share
        - current_price: Current price per share
        - market_value: Current total value (quantity * current_price)
        - unrealized_gain: Market value - cost basis
        - unrealized_gain_pct: Unrealized gain as percentage of cost basis
        - holding_period_days: Number of days position has been held
        - is_long_term: Whether the position qualifies for long-term capital gains (held > 365 days)
        - gain_type: Type of gain for tax purposes (short_term or long_term)
        - federal_tax_rate: Applicable federal tax rate
        - state_tax_rate: Applicable state tax rate
        - total_tax_rate: Combined federal and state tax rate
        - tax_liability: Estimated tax liability if sold at current price
        - per_share_tax_liability: Tax liability per share
        - tax_gain_loss_percentage: Tax gain/loss percentage (gain/loss percentage divided by tax rate)
    """
    # Define column order upfront for both empty and non-empty cases
    column_order = [
        'tax_lot_id',
        'identifier',
        'quantity',
        'cost_basis',
        'cost_per_share',
        'current_price',
        'market_value',
        'unrealized_gain',
        'unrealized_gain_pct',
        'holding_period_days',
        'is_long_term',
        'gain_type',
        'federal_tax_rate',
        'state_tax_rate',
        'total_tax_rate',
        'tax_liability',
        'per_share_tax_liability',
        'tax_gain_loss_percentage'
    ]

    # Return empty DataFrame with correct columns if tax_lots is empty
    if tax_lots.empty:
        return pd.DataFrame(columns=column_order)

    # Create copy to avoid modifying inputs
    report = tax_lots.copy()
    
    # Join with prices
    report = report.merge(
        prices[['identifier', 'price']], 
        on='identifier', 
        how='left',
        validate='many_to_one'
    )
    
    # Calculate derived values
    report['cost_per_share'] = report['cost_basis'] / report['quantity']
    report['market_value'] = report['quantity'] * report['price']
    report['unrealized_gain'] = report['market_value'] - report['cost_basis']
    report['unrealized_gain_pct'] = report['unrealized_gain'] / report['cost_basis']
    report['holding_period_days'] = (pd.Timestamp(current_date) - report['date']).dt.days
    report['is_long_term'] = report['holding_period_days'] > 365
    
    # Determine gain type for tax purposes
    report['gain_type'] = report['is_long_term'].map({True: 'long_term', False: 'short_term'})
    
    # Join with tax rates based on gain type
    report = report.merge(
        tax_rates[['gain_type', 'federal_rate', 'state_rate', 'total_rate']],
        on='gain_type',
        how='left',
        validate='many_to_one'
    )
    
    # Calculate tax impact
    # - Positive tax_liability means taxes owed on gains (unrealized_gain > 0)
    # - Negative tax_liability means potential tax benefit from losses (unrealized_gain < 0)
    #   which can offset other gains when realized
    report['tax_liability'] = report['unrealized_gain'] * report['total_rate']
    
    # Calculate per-share tax liability for optimization
    report['per_share_tax_liability'] = report['tax_liability'] / report['quantity']

    # Calculate tax gain/loss percentage (gain/loss percentage divided by tax rate)
    # Note: Handle division by zero by replacing 0 tax rates with NaN
    safe_total_rate = report['total_rate'].replace(0, float('nan'))
    report['tax_gain_loss_percentage'] = report['unrealized_gain_pct'] * safe_total_rate
    
    # Rename columns for clarity
    report = report.rename(columns={
        'price': 'current_price',
        'federal_rate': 'federal_tax_rate',
        'state_rate': 'state_tax_rate',
        'total_rate': 'total_tax_rate'
    })
    
    return report[column_order] 