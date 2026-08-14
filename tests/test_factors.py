"""Tests for Robinhood fundamentals -> Oracle factor model builder."""

import pandas as pd
import pytest

from src.mcp_server.factors import (
    build_factor_model_from_fundamentals,
    flatten_fundamentals_responses,
)
from src.mcp_server.server import build_factor_model_from_fundamentals as build_factors_tool
from src.service.helpers.constants import CASH_CUSIP_ID
from src.service.initializers.factor_model import initialize_factor_model


def _synthetic_fundamental(symbol: str, index: int, sector: str) -> dict:
    price = 50.0 + index * 10.0
    return {
        "symbol": symbol,
        "high": str(price + 1.0),
        "low": str(price - 1.0),
        "pe_ratio": str(10.0 + index),
        "pb_ratio": str(1.0 + index / 10.0),
        "market_cap": str(1_000_000_000 * (index + 1)),
        "high_52_weeks": str(price * 1.2),
        "low_52_weeks": str(price * 0.7),
        "average_volume_30_days": str(1_000_000 + index * 100_000),
        "dividend_yield": None if symbol in {"BRK.B", "TSLA"} else str(index / 10.0),
        "sector": sector,
    }


@pytest.fixture
def fundamentals_payloads():
    rows = [
        _synthetic_fundamental("AAPL", 0, "Electronic Technology"),
        _synthetic_fundamental("JNJ", 1, "Health Technology"),
        _synthetic_fundamental("XOM", 2, "Energy Minerals"),
        _synthetic_fundamental("NEE", 3, "Utilities"),
        _synthetic_fundamental("BRK.B", 4, "Finance"),
        _synthetic_fundamental("JPM", 5, "Finance"),
        _synthetic_fundamental("WMT", 6, "Retail Trade"),
        _synthetic_fundamental("PG", 7, "Consumer Non-Durables"),
        _synthetic_fundamental("CAT", 8, "Producer Manufacturing"),
        _synthetic_fundamental("AMT", 9, "Finance"),
        _synthetic_fundamental("LIN", 10, "Process Industries"),
        _synthetic_fundamental("DIS", 11, "Consumer Services"),
        _synthetic_fundamental("MSFT", 12, "Technology Services"),
        _synthetic_fundamental("UNH", 13, "Health Services"),
        _synthetic_fundamental("TSLA", 14, "Consumer Durables"),
    ]
    return [
        {"data": {"results": rows[:5]}},
        {"data": {"results": rows[5:], "not_found": ["FAKE"]}},
    ]


def test_flatten_fundamentals_responses(fundamentals_payloads):
    rows, not_found = flatten_fundamentals_responses(fundamentals_payloads)
    assert len(rows) == 15
    assert not_found == ["FAKE"]
    symbols = {row["symbol"] for row in rows}
    assert {"AAPL", "TSLA"} <= symbols


def test_build_factor_model_no_nulls_and_floats(fundamentals_payloads):
    result = build_factor_model_from_fundamentals(fundamentals_payloads)
    model = result["factor_model"]
    assert len(model) == 15

    style_cols = {"value", "momentum", "size", "dividend_yield", "low_volatility", "liquidity"}
    for row in model:
        assert "identifier" in row
        for col in style_cols:
            assert isinstance(row[col], float)
            assert row[col] == row[col]

    sector_cols = [column for column in model[0] if column.startswith("sector_")]
    assert "sector_finance" in sector_cols
    assert "sector_electronic_technology" in sector_cols


def test_null_dividend_yield_becomes_zero_before_zscore(fundamentals_payloads):
    result = build_factor_model_from_fundamentals(fundamentals_payloads)
    by_symbol = {row["identifier"]: row for row in result["factor_model"]}
    assert isinstance(by_symbol["BRK.B"]["dividend_yield"], float)
    assert isinstance(by_symbol["TSLA"]["dividend_yield"], float)


def test_sector_one_hots(fundamentals_payloads):
    result = build_factor_model_from_fundamentals(fundamentals_payloads)
    jpm = next(row for row in result["factor_model"] if row["identifier"] == "JPM")
    assert jpm["sector_finance"] == 1.0
    assert jpm["sector_electronic_technology"] == 0.0
    assert "Finance" in result["sectors"]


def test_diagnostics_present(fundamentals_payloads):
    result = build_factor_model_from_fundamentals(fundamentals_payloads)
    assert result["diagnostics"]["value"]["min"] >= -3.0
    assert result["diagnostics"]["value"]["max"] <= 3.0


def test_passes_initialize_factor_model(fundamentals_payloads):
    result = build_factor_model_from_fundamentals(fundamentals_payloads)
    factor_df = pd.DataFrame(result["factor_model"])
    symbols = factor_df["identifier"].tolist()
    weight = 1.0 / len(symbols)
    targets = pd.DataFrame(
        {
            "asset_class": symbols + [CASH_CUSIP_ID],
            "target_weight": [weight] * len(symbols) + [0.0],
            "identifiers": [[symbol] for symbol in symbols] + [[CASH_CUSIP_ID]],
        }
    )
    actuals = pd.DataFrame(
        {"identifier": symbols[:3], "actual_weight": [0.33, 0.33, 0.34]}
    )

    normalized, target_exp, actual_exp = initialize_factor_model(
        factor_df,
        targets,
        actuals,
    )
    assert normalized.isnull().sum().sum() == 0
    assert CASH_CUSIP_ID in set(normalized["identifier"])
    assert len(target_exp.columns) > 0
    assert len(actual_exp.columns) > 0


def test_build_factor_model_tool_wrapper(fundamentals_payloads):
    out = build_factors_tool(fundamentals=fundamentals_payloads, include_sectors=True)
    assert len(out["factor_model"]) == 15
    assert out["coverage"]["count"] == 15


def test_exclude_sectors(fundamentals_payloads):
    result = build_factor_model_from_fundamentals(
        fundamentals_payloads,
        include_sectors=False,
    )
    assert not any(
        key.startswith("sector_")
        for key in result["factor_model"][0]
        if key != "identifier"
    )


def test_unknown_factor_raises(fundamentals_payloads):
    with pytest.raises(ValueError, match="Unknown style factors"):
        build_factor_model_from_fundamentals(
            fundamentals_payloads,
            factors=["value", "not_a_factor"],
        )
