"""Download and parse ETF index holdings (e.g. SSGA SPY daily XLSX) into Oracle targets."""

from __future__ import annotations

import io
import re
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# SSGA daily holdings file for SPY (State Street SPDR S&P 500 ETF Trust).
DEFAULT_SPY_HOLDINGS_URL = (
    "https://www.ssga.com/library-content/products/fund-data/etfs/us/holdings-daily-us-en-spy.xlsx"
)

_HOLDINGS_SHEET = "holdings"
_HEADER_ROW = 4  # row index for pandas header=4 (Excel row 5: Name, Ticker, Weight, ...)
_AS_OF_RE = re.compile(r"As of\s+(\d{1,2}-[A-Za-z]{3}-\d{4})", re.IGNORECASE)
_NON_EQUITY_TICKERS = {"-", "", "NAN", "NONE"}
_TICKER_RE = re.compile(r"^[A-Z]{1,5}([.\-][A-Z]{1,2})?$")


def _download_bytes(url: str, timeout: float = 60.0) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "oracle-mcp/1.0 (portfolio-optimizer)"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.URLError as exc:
        raise ValueError(f"Failed to download holdings from {url}: {exc}") from exc


def _parse_as_of_date(meta: pd.DataFrame) -> Optional[str]:
    """Extract holdings as-of date from SSGA metadata rows."""
    for _, row in meta.iterrows():
        for cell in row:
            if cell is None or (isinstance(cell, float) and pd.isna(cell)):
                continue
            text = str(cell)
            match = _AS_OF_RE.search(text)
            if match:
                try:
                    return datetime.strptime(match.group(1), "%d-%b-%Y").date().isoformat()
                except ValueError:
                    return match.group(1)
    return None


def _parse_fund_ticker(meta: pd.DataFrame) -> Optional[str]:
    for _, row in meta.iterrows():
        label = str(row.iloc[0]) if len(row) else ""
        if "ticker" in label.lower() and len(row) > 1 and pd.notna(row.iloc[1]):
            return str(row.iloc[1]).strip().upper()
    return None


def _parse_fund_name(meta: pd.DataFrame) -> Optional[str]:
    for _, row in meta.iterrows():
        label = str(row.iloc[0]) if len(row) else ""
        if "fund name" in label.lower() and len(row) > 1 and pd.notna(row.iloc[1]):
            return str(row.iloc[1]).strip()
    return None


def _normalize_ticker(raw: Any) -> Optional[str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    ticker = str(raw).strip().upper()
    if ticker in _NON_EQUITY_TICKERS:
        return None
    if not _TICKER_RE.match(ticker):
        return None
    return ticker


def parse_ssga_holdings_xlsx(
    content: bytes,
    *,
    cash_target_weight: float = 0.0,
    min_weight_pct: Optional[float] = None,
    top_n: Optional[int] = None,
    include_fund_cash: bool = False,
) -> Dict[str, Any]:
    """Parse an SSGA-style daily holdings XLSX into Oracle ``targets`` rows.

    The Weight column is in **percent** (e.g. 7.92 = 7.92%). Returned
    ``target_weight`` values are fractions summing to ``1.0`` (with an optional
    CASH row when ``cash_target_weight`` > 0).

    By default fund cash (US DOLLAR row, ticker ``-``) is excluded and equity
    weights are renormalized to fill ``1 - cash_target_weight``.
    """
    if not 0.0 <= cash_target_weight < 1.0:
        raise ValueError("cash_target_weight must be in [0, 1)")

    buffer = io.BytesIO(content)
    meta = pd.read_excel(buffer, sheet_name=_HOLDINGS_SHEET, header=None, nrows=_HEADER_ROW)

    buffer.seek(0)
    df = pd.read_excel(buffer, sheet_name=_HOLDINGS_SHEET, header=_HEADER_ROW)

    required = {"Ticker", "Weight", "Name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Holdings sheet missing columns {sorted(missing)}; got {list(df.columns)}")

    rows: List[Dict[str, Any]] = []
    fund_cash_pct = 0.0

    for _, row in df.iterrows():
        name = str(row.get("Name") or "").strip()
        ticker = _normalize_ticker(row.get("Ticker"))
        weight_pct = pd.to_numeric(row.get("Weight"), errors="coerce")
        if weight_pct is None or pd.isna(weight_pct) or float(weight_pct) <= 0:
            continue

        is_cash = ticker is None or name.upper() in {"US DOLLAR", "CASH"}
        if is_cash:
            fund_cash_pct += float(weight_pct)
            if include_fund_cash and name.upper() in {"US DOLLAR", "CASH"}:
                rows.append(
                    {
                        "symbol": "CASH",
                        "name": name or "CASH",
                        "weight_pct_raw": float(weight_pct),
                        "sector": row.get("Sector"),
                        "identifier": row.get("Identifier"),
                        "shares_held": row.get("Shares Held"),
                    }
                )
            continue

        if min_weight_pct is not None and float(weight_pct) < min_weight_pct:
            continue

        rows.append(
            {
                "symbol": ticker,
                "name": name,
                "weight_pct_raw": float(weight_pct),
                "sector": None if str(row.get("Sector") or "").strip() in {"", "-"} else str(row.get("Sector")),
                "identifier": row.get("Identifier"),
                "sedol": row.get("SEDOL"),
                "shares_held": pd.to_numeric(row.get("Shares Held"), errors="coerce"),
            }
        )

    if not rows:
        raise ValueError("No equity holdings found in XLSX (after filters)")

    if top_n is not None:
        if top_n <= 0:
            raise ValueError("top_n must be positive")
        rows = sorted(rows, key=lambda r: r["weight_pct_raw"], reverse=True)[:top_n]

    raw_weight_pct_sum = sum(r["weight_pct_raw"] for r in rows)
    if raw_weight_pct_sum <= 0:
        raise ValueError("Holdings weights sum to zero")

    equity_budget = 1.0 - cash_target_weight
    scale = equity_budget / (raw_weight_pct_sum / 100.0)

    targets: List[Dict[str, Any]] = []
    for row in rows:
        if row["symbol"] == "CASH":
            target_weight = row["weight_pct_raw"] / 100.0 * scale
        else:
            target_weight = row["weight_pct_raw"] / 100.0 * scale
        targets.append(
            {
                "symbol": row["symbol"],
                "target_weight": round(target_weight, 8),
                "asset_class": row["symbol"],
                "identifiers": [row["symbol"]],
                "name": row["name"],
                "weight_pct_raw": row["weight_pct_raw"],
                **({k: v for k, v in row.items() if k in {"sector", "identifier", "sedol", "shares_held"} and v is not None}),
            }
        )

    if cash_target_weight > 0:
        targets.append(
            {
                "symbol": "CASH",
                "target_weight": round(cash_target_weight, 8),
                "asset_class": "CASH",
                "identifiers": ["CASH"],
                "name": "CASH",
            }
        )

    weight_sum = sum(t["target_weight"] for t in targets)

    return {
        "fund_name": _parse_fund_name(meta),
        "fund_ticker": _parse_fund_ticker(meta),
        "holdings_as_of": _parse_as_of_date(meta),
        "targets": targets,
        "summary": {
            "constituent_count": len([t for t in targets if t["symbol"] != "CASH"]),
            "raw_weight_pct_sum": round(raw_weight_pct_sum, 6),
            "fund_cash_pct_excluded": round(fund_cash_pct, 6) if not include_fund_cash else 0.0,
            "cash_target_weight": cash_target_weight,
            "normalized_target_weight_sum": round(weight_sum, 8),
            "top_n": top_n,
            "min_weight_pct": min_weight_pct,
        },
    }


def fetch_etf_holdings_targets(
    url: str = DEFAULT_SPY_HOLDINGS_URL,
    *,
    cash_target_weight: float = 0.0,
    min_weight_pct: Optional[float] = None,
    top_n: Optional[int] = None,
    include_fund_cash: bool = False,
    timeout: float = 60.0,
) -> Dict[str, Any]:
    """Download ``url`` and return normalized Oracle targets."""
    content = _download_bytes(url, timeout=timeout)
    parsed = parse_ssga_holdings_xlsx(
        content,
        cash_target_weight=cash_target_weight,
        min_weight_pct=min_weight_pct,
        top_n=top_n,
        include_fund_cash=include_fund_cash,
    )
    parsed["source_url"] = url
    return parsed
