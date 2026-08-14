#!/usr/bin/env python3
"""Deterministic verifier for the SP500 direct-index agent e2e run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def _fail(errors: List[str], msg: str) -> None:
    errors.append(msg)


def verify(
    result_path: Path,
    *,
    expected_reported_cash: float | None = None,
) -> tuple[bool, List[str]]:
    errors: List[str] = []
    data = json.loads(result_path.read_text(encoding="utf-8"))

    targets = data.get("_snapshot_targets")
    if targets is None:
        # Full result file — infer from event if present; else skip target count
        pass

    cash_override = data.get("cash_override") or {}
    if not cash_override.get("applied"):
        _fail(errors, "cash_override.applied is not true")
    elif (
        expected_reported_cash is not None
        and float(cash_override.get("reported_cash", -1)) != expected_reported_cash
    ):
        _fail(
            errors,
            f"cash_override.reported_cash expected {expected_reported_cash}, "
            f"got {cash_override.get('reported_cash')}",
        )

    strategy = (data.get("results") or {}).get("1") or {}
    if not strategy.get("should_trade"):
        _fail(errors, "results['1'].should_trade is not true")
    status = strategy.get("status")
    if status not in (1, "Optimal", "optimal"):
        _fail(errors, f"solver status not optimal: {status!r}")

    trades = strategy.get("trades") or []
    netted = data.get("netted_trades") or []
    all_rows = trades or netted
    sells = [t for t in all_rows if str(t.get("trade_type") or t.get("action", "")).upper() == "SELL"]
    if sells:
        _fail(errors, f"expected no sells on empty portfolio, got {len(sells)}")

    investable = cash_override.get("investable_amount")
    if investable is not None:
        buy_notional = 0.0
        prices = {p["symbol"]: p["price"] for p in (data.get("_snapshot_prices") or [])}
        for t in all_rows:
            if str(t.get("trade_type") or t.get("action", "")).upper() != "BUY":
                continue
            sym = t.get("symbol") or t.get("identifier")
            qty = float(t.get("quantity") or 0)
            px = prices.get(sym) or float(t.get("price") or 0)
            buy_notional += qty * px
        if buy_notional > float(investable) * 1.05:
            _fail(errors, f"buy notional {buy_notional:.2f} exceeds investable_amount {investable}")

    rh = data.get("robinhood_orders") or {}
    if rh:
        if not rh.get("simulated"):
            _fail(errors, "robinhood_orders.simulated is not true")
        for order in rh.get("review_equity_orders") or []:
            if str(order.get("side", "")).lower() != "buy":
                _fail(errors, f"expected buy-only orders, got side={order.get('side')}")

    if data.get("used_default_tax_rates") is None:
        _fail(errors, "used_default_tax_rates not surfaced")

    return len(errors) == 0, errors


def verify_snapshot(snapshot_path: Path) -> tuple[bool, List[str]]:
    errors: List[str] = []
    snap = json.loads(snapshot_path.read_text(encoding="utf-8"))
    targets = snap.get("targets") or []
    if len(targets) < 400:
        _fail(errors, f"expected >= 400 targets, got {len(targets)}")
    weight_sum = sum(float(t.get("target_weight") or 0) for t in targets)
    if abs(weight_sum - 1.0) > 1e-4:
        _fail(errors, f"target weights sum to {weight_sum}, expected ~1.0")

    target_syms = {
        (t.get("symbol") or (t.get("identifiers") or [None])[0])
        for t in targets
        if (t.get("symbol") or "CASH") != "CASH"
    }
    prices = {p["symbol"] for p in (snap.get("prices") or [])}
    missing = [s for s in target_syms if s and s not in prices]
    if missing:
        _fail(errors, f"missing prices for {len(missing)} targets (e.g. {missing[:5]})")

    factor_model = snap.get("factor_model") or []
    if len(factor_model) < len(target_syms) * 0.95:
        _fail(errors, f"factor_model coverage low: {len(factor_model)} vs {len(target_syms)} targets")
    if factor_model:
        sample = factor_model[0]
        style_keys = {"value", "momentum", "size", "dividend_yield", "low_volatility", "liquidity"}
        if not style_keys.intersection(sample.keys()):
            _fail(errors, "factor_model missing style factor columns")
        if not any(k.startswith("sector_") for k in sample.keys()):
            _fail(errors, "factor_model missing sector_* columns")

    return len(errors) == 0, errors


def enrich_result_for_verify(result_path: Path, snapshot_path: Path) -> None:
    """Attach snapshot prices/targets to result file for notional checks."""
    result = json.loads(result_path.read_text(encoding="utf-8"))
    snap = json.loads(snapshot_path.read_text(encoding="utf-8"))
    result["_snapshot_prices"] = snap.get("prices") or []
    result["_snapshot_targets"] = snap.get("targets") or []
    result_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Verify SP500 direct-index e2e artifacts.")
    parser.add_argument("--result", default="artifacts/e2e_result.json")
    parser.add_argument("--snapshot", default="artifacts/rh_snapshot.json")
    parser.add_argument(
        "--expected-reported-cash",
        type=float,
        default=None,
        help="Expected broker-reported cash; defaults to the snapshot value.",
    )
    parser.add_argument("--skip-snapshot", action="store_true")
    args = parser.parse_args(argv)

    all_errors: List[str] = []
    result_path = Path(args.result)
    if not result_path.exists():
        print(f"FAIL: result file not found: {result_path}", file=sys.stderr)
        raise SystemExit(1)

    snapshot_path = Path(args.snapshot)
    expected_reported_cash = args.expected_reported_cash
    if snapshot_path.exists():
        enrich_result_for_verify(result_path, snapshot_path)
        if expected_reported_cash is None:
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            expected_reported_cash = float(snapshot.get("cash") or 0.0)

    if not args.skip_snapshot and snapshot_path.exists():
        ok, errs = verify_snapshot(snapshot_path)
        all_errors.extend(errs)
        print(f"snapshot: {'PASS' if ok else 'FAIL'}")
        for e in errs:
            print(f"  - {e}")

    ok, errs = verify(result_path, expected_reported_cash=expected_reported_cash)
    all_errors.extend(errs)
    print(f"optimize: {'PASS' if ok else 'FAIL'}")
    for e in errs:
        print(f"  - {e}")

    if all_errors:
        raise SystemExit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
