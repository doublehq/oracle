"""Tests for SSGA ETF holdings download and target normalization."""

import io

import pandas as pd
import pytest

from src.mcp_server.index_targets import (
    DEFAULT_SPY_HOLDINGS_URL,
    fetch_etf_holdings_targets,
    parse_ssga_holdings_xlsx,
)


def _minimal_ssga_xlsx() -> bytes:
    """Build a tiny SSGA-shaped workbook in memory."""
    meta = pd.DataFrame(
        [
            ["Fund Name:", "Test Index ETF"],
            ["Ticker Symbol:", "TEST"],
            ["Holdings:", "As of 11-Aug-2026"],
            [None, None],
        ]
    )
    holdings = pd.DataFrame(
        [
            {
                "Name": "ALPHA INC",
                "Ticker": "AAA",
                "Identifier": "111",
                "SEDOL": "S1",
                "Weight": 60.0,
                "Sector": "Tech",
                "Shares Held": 1000,
                "Local Currency": "USD",
            },
            {
                "Name": "BETA INC",
                "Ticker": "BBB",
                "Identifier": "222",
                "SEDOL": "S2",
                "Weight": 30.0,
                "Sector": "-",
                "Shares Held": 500,
                "Local Currency": "USD",
            },
            {
                "Name": "US DOLLAR",
                "Ticker": "-",
                "Identifier": "999USDZ92",
                "SEDOL": "S3",
                "Weight": 10.0,
                "Sector": "-",
                "Shares Held": 1,
                "Local Currency": "USD",
            },
        ]
    )
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        meta.to_excel(writer, sheet_name="holdings", index=False, header=False, startrow=0)
        holdings.to_excel(writer, sheet_name="holdings", index=False, startrow=4)
    return buffer.getvalue()


def test_parse_ssga_holdings_normalizes_equity_weights():
    parsed = parse_ssga_holdings_xlsx(_minimal_ssga_xlsx())
    assert parsed["fund_ticker"] == "TEST"
    assert parsed["holdings_as_of"] == "2026-08-11"
    targets = parsed["targets"]
    assert len(targets) == 2
    assert targets[0]["symbol"] == "AAA"
    assert targets[1]["symbol"] == "BBB"
    assert targets[0]["target_weight"] == pytest.approx(0.66666667, rel=1e-4)
    assert targets[1]["target_weight"] == pytest.approx(0.33333333, rel=1e-4)
    assert sum(t["target_weight"] for t in targets) == pytest.approx(1.0)
    assert parsed["summary"]["fund_cash_pct_excluded"] == pytest.approx(10.0)


def test_parse_ssga_holdings_reserves_cash_buffer():
    parsed = parse_ssga_holdings_xlsx(_minimal_ssga_xlsx(), cash_target_weight=0.05)
    targets = parsed["targets"]
    cash = next(t for t in targets if t["symbol"] == "CASH")
    equities = [t for t in targets if t["symbol"] != "CASH"]
    assert cash["target_weight"] == pytest.approx(0.05)
    assert sum(t["target_weight"] for t in equities) == pytest.approx(0.95)
    assert sum(t["target_weight"] for t in targets) == pytest.approx(1.0)


def test_parse_ssga_holdings_can_include_fund_cash_row():
    parsed = parse_ssga_holdings_xlsx(
        _minimal_ssga_xlsx(),
        include_fund_cash=True,
    )

    cash = next(target for target in parsed["targets"] if target["symbol"] == "CASH")
    assert cash["weight_pct_raw"] == pytest.approx(10.0)
    assert sum(t["target_weight"] for t in parsed["targets"]) == pytest.approx(1.0)


def test_parse_ssga_holdings_top_n_and_min_weight():
    parsed = parse_ssga_holdings_xlsx(_minimal_ssga_xlsx(), top_n=1)
    assert len(parsed["targets"]) == 1
    assert parsed["targets"][0]["symbol"] == "AAA"
    assert parsed["targets"][0]["target_weight"] == pytest.approx(1.0)

    parsed2 = parse_ssga_holdings_xlsx(_minimal_ssga_xlsx(), min_weight_pct=50.0)
    assert len(parsed2["targets"]) == 1
    assert parsed2["targets"][0]["symbol"] == "AAA"


def test_parse_ssga_holdings_rejects_cusip_like_tickers():
    from src.mcp_server.index_targets import _normalize_ticker

    assert _normalize_ticker("2602335D") is None
    assert _normalize_ticker("NVDA") == "NVDA"
    assert _normalize_ticker("BRK.B") == "BRK.B"


@pytest.mark.integration
def test_fetch_spy_holdings_targets_live():
    """Hit the real SSGA SPY URL (requires network)."""
    result = fetch_etf_holdings_targets(url=DEFAULT_SPY_HOLDINGS_URL, top_n=10)
    assert result["fund_ticker"] == "SPY"
    assert result["holdings_as_of"]
    assert len(result["targets"]) == 10
    assert result["targets"][0]["symbol"]  # e.g. NVDA
    assert result["targets"][0]["target_weight"] > 0
    assert sum(t["target_weight"] for t in result["targets"]) == pytest.approx(1.0)
    assert result["source_url"] == DEFAULT_SPY_HOLDINGS_URL
