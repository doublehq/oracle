#!/usr/bin/env python3
"""Run live e2e: snapshot (requires oracle-rh-login) -> optimize -> verify."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from src.mcp_server.rh_client import tokens_present
from src.mcp_server.server import optimize_portfolio
from src.mcp_server.snapshot import fetch_robinhood_snapshot


def _trade_side(trade: Dict[str, Any]) -> str:
    return str(trade.get("trade_type") or trade.get("action") or "").upper()


def _trade_notional(trade: Dict[str, Any]) -> float:
    qty = float(trade.get("quantity") or 0)
    price = float(trade.get("price") or 0)
    return abs(qty * price)


def summarize_netted_trades(
    result: Dict[str, Any],
    *,
    investable_amount: float | None = None,
    top_n: int = 15,
) -> Dict[str, Any]:
    """Build a compact summary from a full optimize result."""
    netted: List[Dict[str, Any]] = result.get("netted_trades") or []
    buys = [t for t in netted if _trade_side(t) == "BUY"]
    sells = [t for t in netted if _trade_side(t) == "SELL"]

    buy_notional = sum(_trade_notional(t) for t in buys)
    sell_notional = sum(_trade_notional(t) for t in sells)

    ranked = sorted(netted, key=_trade_notional, reverse=True)
    top = [
        {
            "symbol": t.get("symbol") or t.get("identifier"),
            "side": _trade_side(t),
            "quantity": round(float(t.get("quantity") or 0), 6),
            "price": round(float(t.get("price") or 0), 2),
            "notional": round(_trade_notional(t), 2),
        }
        for t in ranked[:top_n]
    ]

    summary: Dict[str, Any] = {
        "netted_trades_count": len(netted),
        "buys": len(buys),
        "sells": len(sells),
        "buy_notional": round(buy_notional, 2),
        "sell_notional": round(sell_notional, 2),
        "top_by_notional": top,
    }
    if investable_amount is not None:
        summary["investable_amount"] = investable_amount
        summary["buy_notional_pct_of_investable"] = round(
            100.0 * buy_notional / investable_amount, 1
        ) if investable_amount else None

    cash_override = result.get("cash_override")
    if cash_override:
        summary["cash_override"] = cash_override

    strategy = (result.get("results") or {}).get("1") or {}
    summary["should_trade"] = strategy.get("should_trade")
    summary["solver_status"] = strategy.get("status")

    return summary


def print_netted_trades_report(
    result: Dict[str, Any] | Path,
    *,
    investable_amount: float | None = None,
    top_n: int = 15,
    show_all: bool = False,
) -> None:
    if isinstance(result, Path):
        result = json.loads(result.read_text(encoding="utf-8"))
    summary = summarize_netted_trades(
        result, investable_amount=investable_amount, top_n=top_n
    )

    print("\n=== Netted trades summary ===")
    print(f"  Total: {summary['netted_trades_count']}  "
          f"(buys: {summary['buys']}, sells: {summary['sells']})")
    print(f"  Buy notional:  ${summary['buy_notional']:,.2f}")
    if summary["sells"]:
        print(f"  Sell notional: ${summary['sell_notional']:,.2f}")
    if summary.get("investable_amount") is not None:
        print(
            f"  vs investable: ${summary['investable_amount']:,.2f} "
            f"({summary.get('buy_notional_pct_of_investable')}%)"
        )
    if summary.get("cash_override"):
        co = summary["cash_override"]
        print(
            f"  Cash override: reported ${co.get('reported_cash')} → "
            f"investable ${co.get('investable_amount')} "
            f"(simulated orders: {result.get('robinhood_orders', {}).get('simulated', False)})"
        )
    print(f"  should_trade: {summary.get('should_trade')}  "
          f"solver_status: {summary.get('solver_status')}")

    print(f"\n  Top {top_n} by notional:")
    for i, row in enumerate(summary["top_by_notional"], 1):
        print(
            f"    {i:2d}. {row['symbol']:6s} {row['side']:4s}  "
            f"qty={row['quantity']:>10.4f}  @ ${row['price']:>8.2f}  "
            f"≈ ${row['notional']:>7.2f}"
        )

    if show_all:
        print("\n=== All netted trades ===")
        for t in result.get("netted_trades") or []:
            sym = t.get("symbol") or t.get("identifier")
            side = _trade_side(t)
            qty = float(t.get("quantity") or 0)
            price = float(t.get("price") or 0)
            print(f"  {sym:6s} {side:4s} qty={qty:.6f} @ ${price:.2f}  ≈ ${_trade_notional(t):.2f}")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Live SP500 direct-index e2e pipeline.")
    parser.add_argument("--snapshot", default="artifacts/rh_snapshot.json")
    parser.add_argument("--result", default="artifacts/e2e_result.json")
    parser.add_argument("--investable-amount", type=float, default=500.0)
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Number of largest netted trades to print (default 15).",
    )
    parser.add_argument(
        "--show-all-trades",
        action="store_true",
        help="Print every netted trade after the summary.",
    )
    args = parser.parse_args()

    if not args.skip_fetch:
        if not tokens_present():
            print(
                "No Robinhood OAuth tokens. Run once:\n\n  oracle-rh-login\n",
                file=sys.stderr,
            )
            raise SystemExit(1)
        while True:
            print("Fetching snapshot (resumable)...")
            status = fetch_robinhood_snapshot(
                out_path=args.snapshot,
                max_seconds=None,
            )
            print(json.dumps(status, indent=2))
            if status.get("status") == "complete":
                break
            if status.get("status") != "partial":
                raise SystemExit(1)

    print("Optimizing...")
    summary = optimize_portfolio(
        snapshot_path=args.snapshot,
        optimization_type="DIRECT_INDEX",
        investable_amount=args.investable_amount,
        settings={"weight_factor_model": 0.25, "should_tlh": False},
        out_path=args.result,
    )
    print(json.dumps(summary, indent=2))

    result_path = Path(args.result)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    trade_summary = summarize_netted_trades(
        result, investable_amount=args.investable_amount, top_n=args.top_n
    )
    result["netted_trades_summary"] = trade_summary
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print_netted_trades_report(
        result,
        investable_amount=args.investable_amount,
        top_n=args.top_n,
        show_all=args.show_all_trades,
    )

    print("Verifying...")
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/verify_e2e_run.py",
            "--result",
            args.result,
            "--snapshot",
            args.snapshot,
        ],
        cwd=Path(__file__).resolve().parents[1],
    )
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
