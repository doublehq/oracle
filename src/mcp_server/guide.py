"""Agent-facing workflow guide for pairing Oracle with the Robinhood Trading MCP."""

ROBINHOOD_WORKFLOW_GUIDE = """\
# Oracle + Robinhood Agentic Trading workflow

Oracle is a local-only tax-aware portfolio optimizer. It recommends trades;
it never executes them. Pair this MCP server with the Robinhood Trading MCP
(https://agent.robinhood.com/mcp/trading) to fetch account data and, only
after explicit user confirmation, place orders.

## Safety rules (non-negotiable)

1. DRY-RUN BY DEFAULT. After Oracle returns recommended trades, present them
   to the user and run Robinhood's `review_equity_order` for each order to
   surface pre-trade warnings. Do NOT call `place_equity_order`.
2. Only call `place_equity_order` after the user explicitly confirms the
   specific trade list in this conversation. "Rebalance my portfolio" is a
   request to compute and review, not to execute.
3. Robinhood only allows agent trading in the user's dedicated Agentic
   account (`agentic_allowed=true`). Use `get_accounts` to find it.
4. Never invent prices, tax lots, or tax rates. Fetch real data or ask.
5. For SELL orders, prefer specified-lot execution: pass `tax_lots` on
   `review_equity_order` / `place_equity_order` using `open_lot_id` values
   from `get_equity_tax_lots`. Omit `tax_lots` only if you intentionally
   want Robinhood's default FIFO cost basis.

## Live `place_equity_order` schema (verified)

Required: `account_number`, `symbol`, `side`, `type`. All numeric fields are
**strings**. Extra fields are rejected (`additionalProperties: false`).

```json
{
  "account_number": "<agentic account_number>",
  "symbol": "AAPL",
  "side": "buy | sell",
  "type": "market | limit | stop_market | stop_limit",
  "quantity": "10.5",
  "dollar_amount": "100.00",
  "limit_price": "190.00",
  "stop_price": "185.00",
  "time_in_force": "gfd | gtc",
  "market_hours": "regular_hours | extended_hours | all_day_hours",
  "tax_lots": [
    { "open_lot_id": "<from get_equity_tax_lots>", "quantity": "5.0" }
  ],
  "ref_id": "<uuid, place_equity_order only>"
}
```

`review_equity_order` uses the same payload **without** `ref_id`.

### Tax-lot rules (sell only)

- Pass `{open_lot_id, quantity}` from `get_equity_tax_lots`; quantities must
  sum to order `quantity`. Max **30 lots** per order; US accounts only.
- **Not allowed** with `dollar_amount`, stop orders, `all_day_hours`, or
  fractional **limit** orders.
- Fractional shares: only `type=market` + `market_hours=regular_hours`.
- Provide **exactly one** of `quantity` or `dollar_amount` (`dollar_amount`
  requires `type=market`).

Oracle's `optimize_portfolio` (with `account_number`) returns ready-made
payloads in `robinhood_orders.review_equity_orders` and
`robinhood_orders.place_equity_orders` when lot ids are present.

## Connect the Robinhood Trading MCP (once)

- Cursor: Settings → Tools & MCPs → Connect
  `https://agent.robinhood.com/mcp/trading` (OAuth in a desktop browser).
- Claude Code: `claude mcp add robinhood-trading --transport http https://agent.robinhood.com/mcp/trading`
- Claude Desktop / Codex: add the same URL as a custom connector.

You can only open an Agentic account and authenticate on a desktop device.

## Oracle read-only Robinhood client (bulk fetch)

Oracle can fetch Robinhood account data server-side via its own OAuth client
(separate from Cursor's Robinhood MCP login). This is **read-only** — Oracle
cannot place orders; use the Robinhood Trading MCP for writes.

One-time setup:

```shell
oracle-rh-login
```

Then call `fetch_robinhood_snapshot` (or CLI `oracle-rh-snapshot`) to pull:

- account, portfolio cash, positions, tax lots
- SPY index targets (SSGA daily XLSX)
- batched quotes and fundamentals (~82 Robinhood MCP calls for full S&P 500)
- wash-sale history via `build_wash_sale_history`

Results are written to `artifacts/rh_snapshot.json` with per-batch caching in
`artifacts/.rh_cache`. If the MCP tool returns `"status": "partial"`, call it
again until `"status": "complete"`.

Pass the artifact to optimize via `snapshot_path` — the agent never needs to
hold 500 rows in context.

### investable_amount (dry-run buy-in)

`investable_amount` replaces **total** effective cash for optimization. Use it
to simulate deploying $500 even when real buying power differs (e.g. pending
deposits). When it differs from reported cash, `robinhood_orders` is tagged
`simulated: true`. This does **not** mean "deploy only $X on top of holdings" —
targets remain whole-portfolio weights.

## Step-by-step

### 1. Gather account data (Robinhood Trading MCP)

- `get_accounts` -> find the Agentic account (`agentic_allowed=true`).
- `get_portfolio` -> cash / buying power (Oracle's `cash` input).
- `get_equity_positions` -> symbols held.
- `get_equity_tax_lots` (one symbol per call) -> per-lot rows with
  `open_lot_id`, `open_date`, `quantity`, `tax_cost_basis` or
  `cost_per_share`, `term`, `is_selectable`. Pass responses to
  `normalize_robinhood_tax_lots`.
- `get_equity_quotes` -> prices for every symbol (held AND targets). Max 20
  symbols per call; batch.
- `get_equity_fundamentals` -> sector, valuation ratios, 52-week range, and
  volume for factor-model construction. Max **10 symbols per call**; batch
  (~50 calls for full S&P 500 — cache the result).
- **Wash-sale history** (see below) -> `recently_closed_lots` for wash sales.
- `get_equity_tradability` -> fractional vs whole-share; set
  `settings.trade_rounding = 0` for whole-share-only symbols.

#### Wash-sale history fetch recipe

Oracle needs realized **loss** sales from the last ~31 days for wash-sale buy
restrictions. Recent purchases are derived automatically from open `tax_lots`
(no extra fetch).

**Account-number trap:** `get_pnl_trade_history` and `get_realized_pnl` take
the **`rhs_account_number`** from `get_accounts`. Order and lot tools take
`account_number`. Passing the wrong one returns an empty list (not an error).

```
get_accounts()                                              # need BOTH numbers
get_pnl_trade_history(account_number=<rhs_account_number from get_accounts>, span="month")
  -> page while next_cursor is non-empty
get_equity_orders(account_number, state="filled", created_at_gte="<today-31d>")
  -> fallback/cross-check; page while next is present (cursor from next URL)
get_equity_tax_lots(account_number, symbol)                 # preview only; one symbol per call
```

Pass the raw responses to Oracle's `build_wash_sale_history` tool. It returns
`recently_closed_lots` ready for `optimize_portfolio`, plus `wash_sale_preview`,
`needs_review`, `warnings`, and `coverage`.

**Cost-basis gap:** Robinhood may report `realized_gain: null` on transferred-in
lots (no broker basis). A null gain is **unknown basis**, not zero gain.
`build_wash_sale_history` defaults to a conservative assumed loss so the symbol
is buy-restricted; review `needs_review` with the user. Pass `cost_basis_hints`
when the user supplies basis, or set `unknown_basis_policy=exclude` to drop
those sales (with a loud warning).

**Pagination:** If `coverage.equity_orders_unread_next` or
`coverage.pnl_trade_history_unread_next` is true, fetch more pages before
optimizing.

**Preview flags:** `wash_sale_preview.not_selectable_lots` lists open lots with
`is_selectable=false` (acquired today; cannot be used for specified-lot sells).

### 2. Establish targets and tax rates

- `targets`: {symbol, target_weight} summing to ~1.0. For S&P 500 direct
  indexing, call `fetch_etf_holdings_targets` (default SPY SSGA daily XLSX)
  and pass its `targets` list to `optimize_portfolio`.
- `tax_rates`: {"short_term": 0.35, "long_term": 0.20, ...} or defaults
  (37% ST / 20% LT — disclose when `used_default_tax_rates` is true).

### 3. Optimize (this server)

Call `optimize_portfolio` with:

- `tax_lots` including `open_lot_id` from Robinhood (mapped to Oracle's
  `tax_lot_id` automatically).
- `account_number` (Agentic) to receive `robinhood_orders` payloads.
- `use_specified_tax_lots=true` (default) so sells include `tax_lots` on
  the Robinhood order bodies.
- `optimization_type`: TAX_AWARE, TAX_UNAWARE, BUY_ONLY, PAIRS_TLH,
  DIRECT_INDEX, or HOLD.

For **S&P 500 direct indexing with a factor model** (recommended bulk path):

1. `fetch_robinhood_snapshot` -> `artifacts/rh_snapshot.json` (repeat if partial).
2. `optimize_portfolio(snapshot_path=..., optimization_type="DIRECT_INDEX",
   investable_amount=500, settings={"weight_factor_model": 0.25})`.

Manual path (agent fetches each Robinhood tool):

1. `fetch_etf_holdings_targets` -> `targets` (SPY constituents).
2. Batch `get_equity_fundamentals` (10 symbols per call) for all target symbols.
3. `build_factor_model_from_fundamentals(fundamentals, prices=...)` -> pass
   the returned `factor_model` list to `optimize_portfolio`.
4. Set `optimization_type="DIRECT_INDEX"`, `settings.should_tlh=true`, and
   **`settings.weight_factor_model > 0`** (default is 0 — factor drift is
   ignored otherwise). Factor weights apply **only** to DIRECT_INDEX; other
   optimization types force `weight_factor_model` to zero.

Default style factors (cross-sectional z-scores, clipped to +/-3):

| Factor | Source fields | Meaning |
|---|---|---|
| `value` | `pe_ratio`, `pb_ratio` | Higher = cheaper (earnings/book yield) |
| `momentum` | 52-week range + price | Higher = closer to 52-week high |
| `size` | `market_cap` | Higher = larger cap |
| `dividend_yield` | `dividend_yield` | Higher = higher yield (null -> 0) |
| `low_volatility` | 52-week range / price | Higher = narrower range (defensive) |
| `liquidity` | `average_volume_30_days` × price | Higher = more liquid |

When `include_sectors=true` (default), Robinhood's `sector` field becomes
`sector_*` one-hot columns (FactSet taxonomy, e.g. `sector_finance`). Align
symbol conventions with targets (`BRK.B` vs `BRK-B`).

Optional: `enforce_disposal_order=true` + `disposal_method` (FIFO/LIFO/HIFO)
constrains the optimizer when you plan to omit `tax_lots` and rely on
account default disposal instead.

### 4. Review with the user

Present `netted_trades`, lot-level `results[].trades`, and
`robinhood_orders.review_equity_orders`. Check `execution_impact`:

- `execution_mode: specified_lot` — Oracle lots match RH `tax_lots` payloads.
- `execution_mode: fifo_fallback` — missing `open_lot_id` or
  `use_specified_tax_lots=false`; compared against default FIFO.

### 5. Execute (only after explicit confirmation)

For each order in `robinhood_orders.review_equity_orders`:

1. `review_equity_order` with that exact payload.
2. After user approval, call `place_equity_order` with the matching entry from
   `place_equity_orders`. Reuse its `ref_id` only when retrying that same order.
3. Place SELLs before BUYs when buys depend on sale proceeds.
4. `get_equity_orders` to confirm; `cancel_equity_order` if needed.

## Field mapping (Robinhood -> optimize_portfolio)

| Robinhood (`get_equity_tax_lots`) | Oracle `tax_lots[]` |
|---|---|
| `open_lot_id` | `open_lot_id` / `tax_lot_id` |
| `open_date` | `date_acquired` |
| `quantity` | `quantity` |
| `tax_cost_basis` or `cost_per_share` × qty | `cost_basis` (total) |
| `data.symbol` or row context | `symbol` |

| Robinhood quote | Oracle `prices[]` |
|---|---|
| last/bid/ask | `price` or bid/ask midpoint |

| `get_portfolio` buying power | `cash` |

| `build_wash_sale_history` -> `recently_closed_lots[]` | wash-sale input |

Oracle cash identifier `_CASH_123` is internal only — not an order symbol.
"""
