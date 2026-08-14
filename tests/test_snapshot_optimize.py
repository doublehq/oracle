"""Tests for snapshot_path / investable_amount optimize_portfolio integration."""

import json
import os
import time

import pytest

from src.mcp_server.server import optimize_portfolio
from src.mcp_server.snapshot import BatchCache, load_snapshot


@pytest.fixture
def mini_snapshot(tmp_path):
    path = tmp_path / "snap.json"
    payload = {
        "account_number": "TEST_ACCOUNT_001",
        "cash": 300.0,
        "tax_lots": [],
        "prices": [
            {"symbol": "AAPL", "price": 200.0},
            {"symbol": "MSFT", "price": 400.0},
        ],
        "targets": [
            {"symbol": "AAPL", "target_weight": 0.5},
            {"symbol": "MSFT", "target_weight": 0.5},
        ],
        "factor_model": [
            {"identifier": "AAPL", "value": 0.1, "momentum": 0.2, "size": 0.3,
             "dividend_yield": 0.0, "low_volatility": 0.1, "liquidity": 0.2,
             "sector_technology": 1.0},
            {"identifier": "MSFT", "value": 0.0, "momentum": 0.1, "size": 0.4,
             "dividend_yield": 0.1, "low_volatility": 0.0, "liquidity": 0.3,
             "sector_technology": 1.0},
        ],
        "recently_closed_lots": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_quotes_to_prices_handles_nested_robinhood_shape():
    from src.mcp_server.snapshot import _quotes_to_prices

    batches = [
        {
            "data": {
                "results": [
                    {
                        "quote": {
                            "symbol": "AAPL",
                            "last_trade_price": "200.50",
                        },
                        "close": {"symbol": "AAPL", "price": "199.00"},
                    }
                ]
            }
        }
    ]
    assert _quotes_to_prices(batches) == [{"symbol": "AAPL", "price": 200.5}]


def test_load_snapshot_requires_keys(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"cash": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="prices"):
        load_snapshot(bad)


def test_load_snapshot_rejects_partial_artifact(tmp_path):
    partial = tmp_path / "partial.json"
    partial.write_text(
        json.dumps(
            {
                "status": "partial",
                "cash": 1,
                "prices": [{"symbol": "AAPL", "price": 1}],
                "targets": [{"symbol": "AAPL", "target_weight": 1}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="partial"):
        load_snapshot(partial)


def test_batch_cache_can_be_cleared_after_completed_snapshot(tmp_path):
    cache = BatchCache(tmp_path / "cache")
    cache.put("quotes", {"data": [1]})
    assert cache.get("quotes") == {"data": [1]}

    cache.clear()

    assert cache.get("quotes") is None


def test_batch_cache_expires_stale_partial_run_data(tmp_path):
    cache = BatchCache(tmp_path / "cache", max_age_seconds=60)
    cache.put("quotes", {"data": [1]})
    cache_file = next((tmp_path / "cache").glob("*.json"))
    old = time.time() - 120
    os.utime(cache_file, (old, old))

    assert cache.get("quotes") is None


def test_optimize_rejects_output_outside_artifact_directory(
    mini_snapshot,
    tmp_path,
    monkeypatch,
):
    artifact_root = tmp_path / "allowed"
    monkeypatch.setenv("ORACLE_ARTIFACT_DIR", str(artifact_root))

    with pytest.raises(ValueError, match="ORACLE_ARTIFACT_DIR"):
        optimize_portfolio(
            snapshot_path=str(mini_snapshot),
            out_path=str(tmp_path / "outside.json"),
        )


def test_optimize_from_snapshot_with_investable_amount(mini_snapshot, tmp_path, monkeypatch):
    monkeypatch.setenv("ORACLE_ARTIFACT_DIR", str(tmp_path))
    out = tmp_path / "result.json"
    summary = optimize_portfolio(
        snapshot_path=str(mini_snapshot),
        optimization_type="DIRECT_INDEX",
        investable_amount=500.0,
        settings={"weight_factor_model": 0.25, "should_tlh": False},
        out_path=str(out),
    )
    assert summary["out_path"] == str(out)
    assert out.exists()
    full = json.loads(out.read_text(encoding="utf-8"))
    assert full["cash_override"]["applied"] is True
    assert full["cash_override"]["reported_cash"] == 300.0
    assert full["cash_override"]["investable_amount"] == 500.0
    assert full["robinhood_orders"]["simulated"] is True
    assert full["results"]["1"]["should_trade"] is True
