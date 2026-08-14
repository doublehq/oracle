# Agent E2E: S&P 500 direct index with factor model

Run this prompt with `cursor-agent` (headless) or manually in Cursor chat with both MCP servers enabled.

## Prerequisites

1. `pip install -e .` and `pip install -r requirements.txt`
2. `oracle-rh-login` (one-time Robinhood OAuth for Oracle's read-only client)
3. `cursor-agent mcp enable oracle robinhood-trading`

## Prompt

You are running an end-to-end dry-run test of Oracle direct indexing into the S&P 500.

**Safety:** NEVER call `place_equity_order`. Do not execute trades.

### Step 1 — Fetch snapshot

Call Oracle's `fetch_robinhood_snapshot` with defaults (`out_path=artifacts/rh_snapshot.json`, full SPY universe).

If the response has `"status": "partial"`, call it again until `"status": "complete"`.

### Step 2 — Optimize

Call `optimize_portfolio` with:

- `snapshot_path`: `artifacts/rh_snapshot.json`
- `optimization_type`: `DIRECT_INDEX`
- `investable_amount`: `500`
- `settings`: `{"weight_factor_model": 0.25, "should_tlh": false}`
- `out_path`: `artifacts/e2e_result.json`

Do not pass tax_lots, prices, targets, or cash manually.

### Step 3 — Verify

Run from repo root:

```bash
python scripts/verify_e2e_run.py --result artifacts/e2e_result.json --snapshot artifacts/rh_snapshot.json
```

### Step 4 — Report

Print PASS or FAIL, the top 15 netted trades by notional, `cash_override`, and any warnings from the snapshot.
