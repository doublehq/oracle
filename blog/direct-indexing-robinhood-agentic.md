# Direct Indexing with Robinhood's Agentic Account

*How I direct-indexed the S&P 500 with $300, an AI agent, and an open-source portfolio optimizer.*

Direct indexing — owning the individual stocks in an index instead of an ETF wrapper — has historically been a rich-person product. Firms like Aperio and Parametric run it as separately managed accounts with six-figure minimums, because the hard parts (tracking hundreds of positions, harvesting losses lot-by-lot, dodging wash sales) require real portfolio-optimization software and a human to babysit it.

Two things just changed that math:

1. **Robinhood's Agentic account** — a dedicated brokerage account that AI agents can trade in via MCP, with fractional dollar-based orders.
2. **AI agents that can orchestrate real tools** — fetch your tax lots, run an optimizer, and stage orders for your review.

This post walks through wiring those together with [Oracle](https://github.com/double-finance/oracle), the open-source portfolio optimizer we built at [Double Finance](https://double.finance), to build a $300 S&P 500 direct index.

## What the Agentic account actually is

In May 2026, Robinhood became the first major US retail broker to ship an official MCP server. The design is sensible:

- You connect any MCP-capable agent (Claude, ChatGPT, Cursor, Codex, etc.) to `https://agent.robinhood.com/mcp/trading`. Auth runs through the Robinhood app — the agent never sees your password.
- The agent can **read** all your accounts: positions, balances, orders, tax lots, quotes, fundamentals.
- The agent can **trade** only in a separate **Agentic account** that you open and fund deliberately. Your main account is read-only to the agent. The amount you move in is the most the agent can ever touch.
- Orders go through a two-step flow: `review_equity_order` (a pre-trade review with quotes and broker alerts) before `place_equity_order`.

That isolation model is what makes it reasonable to let an agent near a brokerage account at all.

## Why direct indexing is the killer app for this

Most "AI trading" demos are stock-picking, which is exactly what you shouldn't let an LLM do. Direct indexing is the opposite: the *strategy* is boring and mechanical (match the index), but the *execution* is a genuinely hard constrained-optimization problem:

- Match ~500 cap-weighted positions with whatever cash you have
- Harvest tax losses at the individual-lot level without triggering wash sales
- Respect minimum notionals, holding periods, and transaction costs
- Use a factor model so that when you *can't* hold every name, the names you skip don't wreck your tracking error

That's not a job for a language model — it's a job for a linear-programming solver. The agent's job is orchestration: gather the data, call the optimizer, present the result, and place the orders you approve.

## The stack

Three pieces, all running from a chat window:

1. **Robinhood Trading MCP** (hosted) — account data, quotes, fundamentals, order review/placement in the Agentic account.
2. **Oracle MCP** (local, open source) — a stdio MCP server wrapping our production optimizer. It builds S&P 500 targets from SSGA's daily SPY holdings file, constructs a factor model from Robinhood's fundamentals data, and solves a multi-objective LP balancing tax cost, drift from target, transaction costs, and factor alignment. It never places orders.
3. **The agent** (Cursor, in my case) — glue.

The workflow the agent runs:

```text
fetch_robinhood_snapshot     → tax lots, cash, quotes, SPY holdings
build_factor_model           → from Robinhood fundamentals
optimize_portfolio           → DIRECT_INDEX mode, LP solve
build_robinhood_equity_orders→ dollar-based fractional order payloads
review_equity_order (×N)     → Robinhood pre-trade review, broker alerts
[human reviews the list]
place_equity_order (×N)      → only after explicit confirmation
```

## The $300 run

I gave the agent $300 of investable cash and asked for an S&P 500 direct index. The optimizer came back with **161 dollar-based fractional market buys totaling exactly $300.00**, and Robinhood's pre-trade review flagged **zero broker alerts**.

The top of the book looks exactly like you'd hope — cap-weighted:

| Symbol | Amount |
|---|---:|
| NVDA | $24.42 |
| AAPL | $20.06 |
| MSFT | $16.48 |
| AMZN | $11.64 |
| GOOGL | $9.09 |
| AVGO | $8.88 |
| GOOG | $7.27 |
| META | $5.83 |

...tapering down to ~110 names at the $1 minimum order size. With only $300, you can't hold all 500 names — the $1 minimum notional caps you at 161 positions. But because the S&P 500 is cap-weighted, those 161 names cover the overwhelming majority of the index's weight, and the factor model picks the *right* subset of the tail to minimize tracking error rather than just cutting off alphabetically.

The whole thing — snapshot, factor model, LP solve, 161 order reviews — ran end-to-end from a chat prompt in a few minutes.

## The guardrails matter more than the demo

A few design decisions I'd consider non-negotiable for anyone building on this:

- **The optimizer is deterministic; the LLM never picks stocks.** The agent moves data between tools. Weights come from SSGA's holdings file, trades come from a solver.
- **Dry-run by default.** Oracle's server instructions require the compute → review → explicit-human-confirmation → place sequence. The agent must show you the exact trade list before anything executes.
- **Blast radius is capped by funding.** The Agentic account only holds what you moved into it.
- **You're still responsible.** Robinhood's terms are clear that agents can make errors and you own every trade. Review the order list. Every time.

## Where this goes

Once positions exist, the same loop handles the ongoing work that made direct indexing an expensive managed product: daily tax-loss harvesting with wash-sale prevention, tax-aware rebalancing as the index drifts, and lot-level disposal when you withdraw. That's the part that historically justified the SMA fees — and it's now a cron job away.

Oracle is MIT-licensed and on GitHub if you want to run this yourself. Start with a small amount, keep dry-run mode on, and read every review before you confirm.

---

*Nothing here is investment advice. Robinhood Agentic Trading is in beta, equities-only, and agents can misbehave — fund accordingly.*
