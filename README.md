# 🔮 Oracle - Intelligent Portfolio Optimizer

Oracle, built by [Double Finance](https://double.finance), is a sophisticated portfolio optimization engine that tax loss harvests and intelligently rebalances portfolios while considering tax implications, trading costs, and various constraints placed upon portfolios (holding time restrictions, etc). It's designed to help investors maintain their target asset allocations while minimizing tax burdens and transaction costs.

We use this in production at [Double Finance](https://double.finance) for daily Tax Loss Harvesting and Automated Rebalancing. Oracle simply returns optimal trades to make for a portfolio - it does not execute anything. You must bring your own price and target/index data. Oracle does not fetch anything from the internet. It allows for customization and control regarding how aggressive you want to be along pretty much every axis.

It provides the basics of a complete portfolio optimization engine similar to:

- Rowboat Advisors (Acquired by Betterment May 2025)
- [Smartleaf](https://www.smartleaf.com/)
- [AlphaThena](https://alphathena.com/) 
- [BlackRock's Tax-Managed Equity SMA by Aperio](https://www.blackrock.com/us/financial-professionals/investments/tax-managed-equity-sma-aperio)
- [AdvisorArch (Acquired by Apex 2024)](https://apexfintechsolutions.com/news-resources/press-releases/apex-fintech-solutions-announces-acquisition-of-advisorarch-bringing-a-suite-of-technology-driven-portfolio-management-solutions/)
- [Parti Pris](https://partipris-invest.com/)

## Features

- **Tax Loss Harvesting**: Ability to specify a threshold at which to TLH a given security, while respecting wash sale rules.
- **Tax-Aware Rebalancing**: Optimizes trades to minimize tax impact while reducing drift
- **Multi-Asset Support**: Can handle any asset type (stocks, etfs, mutual funds, crypto)
- **Consider Trading Costs**: Considers spreads/trading costs while recommending trades.
- **Factor Model Rebalancing Integration**: Supports considering a factor model for use the Direct Indexed based TLH
- **Tax Lot Management**: Tracks and optimizes individual tax lots.
- **Wash Sale Prevention**: Built-in protection against wash sales
- **Stock Restrictions**: Can restrict buying/selling of specific securities
- **Holding Time Restrictions**: Enforces minimum holding periods before selling tax lots

## Not currently supported

- **ESG Tilts**: No support for ESG scoring or ESG-based portfolio tilts
- **Household Level Optimization**: Cannot optimize across multiple accounts in a household
- **Sector Constraints**: No support for sector exposure limits or constraints

## Installation

```shell
# Install CBC solver (required for optimization)
brew install coin-or-tools/coinor/cbc

# Set up Python environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

### 1. Initialize an Oracle Strategy

```python
import pandas as pd
from service.initializers import (
    initialize_targets,
    initialize_tax_lots,
    initialize_prices,
    initialize_spreads,
    initialize_factor_model
)
from service.oracle import Oracle
from service.oracle_strategy import OracleStrategy
from service.helpers.enums import OracleOptimizationType

# Define your target allocations
targets_df = pd.DataFrame({
    'asset_class': ['STOCK_A', 'STOCK_B', 'CASH'],
    'target_weight': [0.4, 0.4, 0.2],
    'identifiers': [['STOCK_A'], ['STOCK_B'], ['CASH']]
})

# Define your current positions
tax_lots_df = pd.DataFrame({
    'tax_lot_id': ['lot_a1', 'lot_b1'],
    'identifier': ['STOCK_A', 'STOCK_B'],
    'quantity': [100, 100],
    'cost_basis': [100, 100],
    'date': ['2024-01-01', '2024-01-01']
})

# Set current prices
prices_df = pd.DataFrame({
    'identifier': ['STOCK_A', 'STOCK_B'],
    'price': [100, 100]
})

# Create Oracle instance
oracle = Oracle(
    current_date='2024-04-20',
    recently_closed_lots=pd.DataFrame(),
    stock_restrictions=pd.DataFrame(),
    tax_rates=pd.DataFrame()
)

# Add strategy to Oracle
strategy = OracleStrategy(
    # Required parameters
    tax_lots=tax_lots_df,
    prices=prices_df,
    cash=10000.0,  # Current cash balance
    # Optional parameters with their defaults
    targets=targets_df,  # Either targets or asset_class_targets must be provided
    asset_class_targets=None,  # Alternative to targets
    spreads=None,  # Optional DataFrame for bid-ask spreads
    factor_model=None,  # Required for DIRECT_INDEX optimization type
    strategy_id=None,  # If None, auto-generated
    optimization_type=OracleOptimizationType.TAX_AWARE,  # Default optimization type
    deminimus_cash_target=0.0,  # Minimum cash percentage target
    withdrawal_amount=0.0,  # Amount to withdraw from portfolio
    enforce_wash_sale_prevention=True  # Whether to enforce wash sale prevention
)

# Set this Oracle instance as the strategy's oracle
strategy.set_oracle(self)
oracle.initialize_wash_sale_restrictions(percentage_protection_from_inadvertent_wash_sales=0.003)


```

### 2. Run the Optimizer

```python
# Compute optimal trades
results, netted_trades = oracle.compute_optimal_trades_for_all_strategies(
    settings={
        "strategies": {
            "STRATEGY_1": {
                "weight_tax": 1.0,
                "weight_drift": 1.0,
                "weight_transaction": 1.0,
                "weight_factor_model": 0.0,
                "weight_cash_drag": 0.0,
                "rebalance_threshold": 0.001,
                "buy_threshold": 0.0005,
                "holding_time_days": 0,
                "should_tlh": True,
                "tlh_min_loss_threshold": 0.015,
                "range_min_weight_multiplier": 0.5,
                "range_max_weight_multiplier": 2.0,
                "min_notional": 0,
                "rank_penalty_factor": 0.0,
                "trade_rounding": 4
            }
        }
    }
)
```

## Optimization Types

Oracle supports several optimization types for any given strategy through the `OracleOptimizationType` enum:

- `HOLD`: No trading allowed, maintains current positions
- `BUY_ONLY`: Only allows buy trades, no sells permitted
- `TAX_UNAWARE`: Rebalances towards targets ignoring tax implications
- `TAX_AWARE`: Rebalances considering tax implications (default)
- `PAIRS_TLH`: Tax-Loss Harvesting specifically for pairs trading
- `DIRECT_INDEX`: Direct Indexing strategy with factor model considerations

Each optimization type has specific behaviors:

- **TLH Support**: Only `PAIRS_TLH` and `DIRECT_INDEX` support tax-loss harvesting
- **Sell Restrictions**: `BUY_ONLY` and `HOLD` types don't allow sell trades
- **Withdrawal Support**: All types except `HOLD` and `BUY_ONLY` can handle portfolio withdrawals
- **Weight Adjustments**: 
  - `TAX_UNAWARE`: Ignores tax implications
  - `HOLD`: Weights don't matter as no trading occurs
  - `DIRECT_INDEX`: Can prioritize factor model considerations
  - Others: Use standard tax-aware optimization

## Response Format

The `compute_optimal_trades_for_all_strategies` method returns a tuple of `(results, netted_trades)`:

```python
{
    "results": {
        "0": {  # Strategy ID
            "label": "Strategy Name",
            "status": "Optimal",  # PuLP status
            "should_trade": True,
            "trades": [
                {
                    "identifier": "STOCK_A",
                    "quantity": 50,
                    "tax_lot_id": "lot_a1",
                    "trade_type": "SELL"
                },
                {
                    "identifier": "STOCK_B",
                    "quantity": 100,
                    "trade_type": "BUY"
                }
            ],
            "trade_summary": {
                "tax_cost": 150.0,
                "drift_cost": 200.0,
                "spread_costs": 10.0,
                "factor_cost": 0.0,
                "cash_drag": 0.0,
                "overall": 360.0
            }
        }
    },
    "netted_trades": [
        {
            "identifier": "STOCK_A",
            "quantity": 50,
            "trade_type": "SELL"
        },
        {
            "identifier": "STOCK_B",
            "quantity": 100,
            "trade_type": "BUY"
        }
    ]
}
```

## Optimization Parameters

The optimizer uses several key parameters to control its behavior:

### Weight Parameters
These parameters control how much the optimizer "cares" about different aspects of the optimization:

- `weight_tax`: How much to prioritize tax efficiency (e.g., harvesting losses)
- `weight_drift`: How much to prioritize staying close to target weights
- `weight_transaction`: How much to prioritize minimizing transaction costs
- `weight_factor_model`: How much to prioritize factor model alignment (for DIRECT_INDEX)
- `weight_cash_drag`: How much to penalize excess cash positions

### Threshold Parameters
These control when trades are triggered:

- `rebalance_threshold`: Minimum deviation from target weight to trigger a rebalance (e.g., 0.001 = 0.1%)
- `buy_threshold`: Minimum deviation to trigger a buy-only trade (typically lower than rebalance_threshold)
- `tlh_min_loss_threshold`: Minimum loss percentage to consider for tax-loss harvesting (e.g., 0.015 = 1.5%)

### Trade Constraints
Parameters that control trade execution:

- `min_notional`: Minimum trade size in dollars (e.g., 5.0 = $5 minimum trade)
- `trade_rounding`: Number of decimal places to round trade quantities
- `range_min_weight_multiplier`: Minimum allowed weight as a fraction of target (e.g., 0.5 = 50% of target)
- `range_max_weight_multiplier`: Maximum allowed weight as a fraction of target (e.g., 2.0 = 200% of target)
- `rank_penalty_factor`: Penalty for deviating from factor model rankings (for PAIRS_TLH)

### Pairs TLH and Rank Penalty
When using the `PAIRS_TLH` optimization type, you can specify pairs of securities that can be used as tax-loss harvesting alternatives for each other. The pairs are specified in the `identifiers` array of your targets DataFrame, where the order matters:

```python
targets_df = pd.DataFrame({
    'asset_class': ['US_LARGE_CAP', 'US_LARGE_CAP'],
    'target_weight': [0.5, 0.5],
    'identifiers': [['SPY', 'IVV', 'VOO'], ['VTI', 'ITOT', 'SCHB']]
})
```

In this example:
- `['SPY', 'IVV', 'VOO']` forms one group of interchangeable securities
- `['VTI', 'ITOT', 'SCHB']` forms another group
- Within each group, earlier securities are preferred over later ones (controlled by `rank_penalty_factor`)

The `rank_penalty_factor` parameter (default 0.0) controls how strongly the optimizer prefers securities earlier in each identifiers list:
- `0.0`: No preference, treats all securities in a group equally
- `0.00001`: Mild preference for earlier securities
- `0.0001`: Strong preference for earlier securities
- `0.001`: Very strong preference for earlier securities

For example, with a high `rank_penalty_factor`:
- The optimizer will prefer to hold SPY over IVV or VOO
- When tax-loss harvesting SPY, it will prefer to switch to IVV before considering VOO
- The preference strength increases with the rank penalty factor

This ranking system helps maintain a preference hierarchy while still allowing flexibility for tax-loss harvesting opportunities.

### Example: Tax-Loss Harvesting Focus
```python
results, netted_trades = oracle.compute_optimal_trades_for_all_strategies(
    settings={
        "strategies": {
            "0": {
                "weight_tax": 1.0,        # Prioritize tax efficiency
                "weight_drift": 0.0001,   # Minimally care about drift
                "weight_transaction": 0.0, # Ignore trading costs
                "weight_cash_drag": 100,  # Strongly prefer deploying cash
                "should_tlh": True,
                "tlh_min_loss_threshold": 0.015  # Harvest losses > 1.5%
            }
        }
    }
)
```

### Example: Direct Indexing with Factor Model
```python
results, netted_trades = oracle.compute_optimal_trades_for_all_strategies(
    settings={
        "strategies": {
            "0": {
                "weight_tax": 1.0,
                "weight_drift": 1.0,
                "weight_transaction": 0.1,
                "weight_factor_model": 0.6,  # Consider factor model alignment
                "weight_cash_drag": 1.0,
                "should_tlh": True,
                "rank_penalty_factor": 0.000001  # Small penalty for deviating from factor rankings
            }
        }
    }
)
```

## Netted Trades

The optimizer returns both individual strategy trades and netted trades. This is important because:

1. Each strategy is optimized independently to maintain separation of concerns
2. Multiple strategies might want to trade the same security
3. Netted trades combine all strategy trades to show the final, consolidated trades needed

For example, if Strategy A wants to buy 100 shares of AAPL and Strategy B wants to sell 50 shares of AAPL, the netted trades would show a single buy of 50 shares of AAPL.

## Lambda Deployment

Oracle is designed to run as an AWS Lambda function. The deployment process:

1. Builds a Docker container with all dependencies
2. Packages the code and dependencies
3. Deploys to AWS Lambda

To deploy:
```shell
./deploy
```

The Lambda function expects input in the following format:
```json
{
  "oracle": {
    "current_date": "2024-04-20",
    "recently_closed_lots": [],
    "stock_restrictions": [],
    "tax_rates": [],
    "percentage_protection_from_inadvertent_wash_sales": 0.003,
    "strategies": {
      "0": {
        "tax_lots": [...],
        "targets": [...],
        "prices": [...],
        "cash": 0,
        "spreads": [],
        "factor_model": [],
        "tlh_pairs": [],
        "strategy_id": 0,
        "optimization_type": "TAX_AWARE"
      }
    }
  },
  "settings": {
    "strategies": {
      "0": {
        "weight_tax": 1.0,
        "weight_drift": 1.0,
        "weight_transaction": 1.0,
        "weight_factor_model": 0.0,
        "weight_cash_drag": 1.0,
        "rebalance_threshold": 0.001,
        "buy_threshold": 0.0005,
        "holding_time_days": 0,
        "should_tlh": true,
        "tlh_min_loss_threshold": 0.015,
        "range_min_weight_multiplier": 0.5,
        "range_max_weight_multiplier": 2.0,
        "min_notional": 0,
        "rank_penalty_factor": 0.0,
        "trade_rounding": 4
      }
    }
  }
}
```

## Objectives

Oracle uses multiple objective terms in its optimization, each weighted according to the strategy's needs:

### 1. Tax Impact (weight_tax)
- Minimizes realized capital gains/losses
- Considers short-term vs long-term tax implications
- Supports tax-loss harvesting (TLH) in PAIRS_TLH and DIRECT_INDEX modes
- Includes wash sale prevention logic

### 2. Drift Impact (weight_drift)
- Minimizes deviation from target asset allocations
- Uses vectorized calculations for performance
- Supports rank-based penalties for preferred securities

### 3. Transaction Costs (weight_transaction)
- Accounts for bid-ask spreads
- Minimizes trading costs
- Considers market impact
- Normalized by total portfolio value

### 4. Factor Model Impact (weight_factor_model)
- Used in DIRECT_INDEX optimization type
- Aligns portfolio with target factor exposures
- Supports piecewise linear approximation

### 5. Cash Deployment (weight_cash_drag)
- Minimizes cash drag when no withdrawal is planned
- Optimizes cash utilization
- Balances cash needs with investment opportunities
- Considers minimum cash requirements

### 6. Maximum Withdrawal Objective
- Special objective for withdrawal scenarios
- Optimizes tax efficiency of withdrawals
- Maintains target allocations during withdrawals
- Considers transaction costs in withdrawal execution

## Constraints

Oracle enforces various constraints to ensure portfolio validity and meet specific requirements:

### 1. Cash Constraints
- Maintains minimum cash balance
- Ensures sufficient cash for withdrawals
- Validates cash flow from trades
- Prevents negative cash positions

### 2. Trade Constraints
- Minimum notional amount for trades
- No simultaneous buys/sells of same security
- Buy-only or sell-only restrictions
- Trade size rounding requirements

### 3. Holding Time Constraints
- Enforces minimum holding periods
- Prevents premature tax lot sales
- Considers tax implications
- Supports tax-aware trading

### 4. Stock Restrictions
- Individual security trading restrictions
- Wash sale prevention rules
- Regulatory compliance checks
- Custom trading rules

### 5. Drift Range Constraints
- Minimum weight multiplier (e.g., 50% of target)
- Maximum weight multiplier (e.g., 200% of target)
- Asset class drift limits
- Special handling for PAIRS_TLH and DIRECT_INDEX

Each constraint can be customized through the settings dictionary when calling `compute_optimal_trades_for_all_strategies`:

```python
settings = {
    "strategies": {
        "0": {
            "min_notional": 10,  # Minimum trade size in dollars
            "holding_time_days": 31,  # Minimum holding period
            "range_min_weight_multiplier": 0.5,  # Can't go below 50% of target
            "range_max_weight_multiplier": 2.0,  # Can't go above 200% of target
            "rebalance_threshold": 0.1,  # Minimum improvement to trigger a rebalance
            "buy_threshold": 0.005,  # Minimum improvement to trigger buy-only trade
        }
    }
}
```

## License

MIT License. See [LICENSE.md](LICENSE.md) for details.
