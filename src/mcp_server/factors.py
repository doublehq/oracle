"""Build Oracle factor models from Robinhood ``get_equity_fundamentals`` payloads."""

from __future__ import annotations

import math
import re
import statistics
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

DEFAULT_STYLE_FACTORS = (
    "value",
    "momentum",
    "size",
    "dividend_yield",
    "low_volatility",
    "liquidity",
)

_EARNINGS_YIELD_CLIP = (-0.2, 0.5)


def _norm_symbol(raw: Dict[str, Any]) -> str:
    symbol = raw.get("symbol") or raw.get("identifier") or raw.get("ticker")
    if not symbol:
        raise ValueError(f"Missing symbol in fundamentals row: {raw}")
    return str(symbol).upper()


def _parse_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def _slugify_sector(sector: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", sector.strip().lower())
    slug = slug.strip("_")
    return f"sector_{slug}" if slug else "sector_unknown"


def flatten_fundamentals_responses(
    responses: Sequence[Any],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Flatten one or more ``get_equity_fundamentals`` MCP payloads.

    Accepts full envelopes ``{data: {results: [...], not_found: [...]}}``,
    bare ``{results: [...]}`` objects, or flat lists of row dicts.
    """
    rows: List[Dict[str, Any]] = []
    not_found: List[str] = []

    for response in responses or []:
        if isinstance(response, list):
            for row in response:
                if isinstance(row, dict):
                    rows.append(dict(row))
            continue

        if not isinstance(response, dict):
            continue

        data = response.get("data") or response
        for symbol in data.get("not_found") or []:
            not_found.append(str(symbol).upper())

        for row in data.get("results") or []:
            if isinstance(row, dict):
                rows.append(dict(row))

    deduped: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        symbol = _norm_symbol(row)
        deduped[symbol] = row
    return list(deduped.values()), sorted(set(not_found))


def _build_price_map(prices: Optional[Sequence[Dict[str, Any]]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not prices:
        return out
    for row in prices:
        if not isinstance(row, dict):
            continue
        symbol = row.get("symbol") or row.get("identifier")
        if not symbol:
            continue
        price = _parse_float(row.get("price") or row.get("last_trade_price") or row.get("last_price"))
        if price is not None and price > 0:
            out[str(symbol).upper()] = price
    return out


def _resolve_price(row: Dict[str, Any], price_map: Dict[str, float]) -> Optional[float]:
    symbol = _norm_symbol(row)
    if symbol in price_map:
        return price_map[symbol]
    high = _parse_float(row.get("high"))
    low = _parse_float(row.get("low"))
    if high is not None and low is not None and high > 0 and low > 0:
        return (high + low) / 2.0
    open_price = _parse_float(row.get("open"))
    if open_price is not None and open_price > 0:
        return open_price
    return None


def _median_fill(values: List[Optional[float]]) -> Tuple[List[float], List[int]]:
    """Replace None with cross-sectional median; return filled values and missing indices."""
    present = [v for v in values if v is not None]
    if not present:
        return [0.0] * len(values), list(range(len(values)))
    median = statistics.median(present)
    filled: List[float] = []
    missing_indices: List[int] = []
    for idx, value in enumerate(values):
        if value is None:
            filled.append(float(median))
            missing_indices.append(idx)
        else:
            filled.append(float(value))
    return filled, missing_indices


def _zscore(values: Sequence[float], clip: float) -> List[float]:
    if len(values) <= 1:
        return [0.0 for _ in values]
    mean = statistics.mean(values)
    try:
        stdev = statistics.pstdev(values)
    except statistics.StatisticsError:
        stdev = 0.0
    if stdev == 0.0:
        return [0.0 for _ in values]
    scored = [(v - mean) / stdev for v in values]
    if clip is not None:
        scored = [max(-clip, min(clip, s)) for s in scored]
    return scored


def _compute_raw_inputs(
    rows: Sequence[Dict[str, Any]],
    price_map: Dict[str, float],
) -> Tuple[List[str], Dict[str, List[Optional[float]]], List[str], Dict[str, Optional[str]]]:
    symbols: List[str] = []
    sectors: Dict[str, Optional[str]] = {}
    earnings_yield: List[Optional[float]] = []
    book_yield: List[Optional[float]] = []
    momentum_raw: List[Optional[float]] = []
    log_market_cap: List[Optional[float]] = []
    dividend_yield: List[Optional[float]] = []
    vol_proxy: List[Optional[float]] = []
    log_dollar_volume: List[Optional[float]] = []
    missing_price: List[str] = []

    for row in rows:
        symbol = _norm_symbol(row)
        symbols.append(symbol)
        sectors[symbol] = row.get("sector")

        pe = _parse_float(row.get("pe_ratio"))
        pb = _parse_float(row.get("pb_ratio"))
        market_cap = _parse_float(row.get("market_cap"))
        high_52 = _parse_float(row.get("high_52_weeks"))
        low_52 = _parse_float(row.get("low_52_weeks"))
        avg_vol = _parse_float(row.get("average_volume_30_days"))
        div_yield = _parse_float(row.get("dividend_yield"))

        ey: Optional[float] = None
        if pe is not None and pe != 0:
            ey = 1.0 / pe
            ey = max(_EARNINGS_YIELD_CLIP[0], min(_EARNINGS_YIELD_CLIP[1], ey))
        earnings_yield.append(ey)

        by: Optional[float] = None
        if pb is not None and pb != 0:
            by = 1.0 / pb
        book_yield.append(by)

        log_market_cap.append(math.log(market_cap) if market_cap and market_cap > 0 else None)

        # Null dividend yield is a real zero (no dividend), not missing data.
        dividend_yield.append(div_yield if div_yield is not None else 0.0)

        price = _resolve_price(row, price_map)
        if price is None or price <= 0:
            missing_price.append(symbol)
            momentum_raw.append(None)
            vol_proxy.append(None)
            log_dollar_volume.append(None)
            continue

        if high_52 is not None and low_52 is not None and high_52 != low_52:
            range_pos = 2.0 * (price - low_52) / (high_52 - low_52) - 1.0
        else:
            range_pos = 0.0
        momentum_raw.append(range_pos)

        if high_52 is not None and low_52 is not None:
            vol_proxy.append((high_52 - low_52) / price)
        else:
            vol_proxy.append(None)

        if avg_vol is not None and avg_vol > 0:
            log_dollar_volume.append(math.log(avg_vol * price))
        else:
            log_dollar_volume.append(None)

    raw = {
        "earnings_yield": earnings_yield,
        "book_yield": book_yield,
        "momentum_raw": momentum_raw,
        "log_market_cap": log_market_cap,
        "dividend_yield": dividend_yield,
        "vol_proxy": vol_proxy,
        "log_dollar_volume": log_dollar_volume,
    }
    return symbols, raw, missing_price, sectors


def build_factor_model_from_fundamentals(
    fundamentals: Sequence[Any],
    *,
    prices: Optional[Sequence[Dict[str, Any]]] = None,
    include_sectors: bool = True,
    factors: Optional[Sequence[str]] = None,
    winsorize: float = 3.0,
) -> Dict[str, Any]:
    """Build an Oracle-ready factor model from Robinhood fundamentals payloads.

    Returns ``{factor_model, coverage, sectors, diagnostics}``.
    """
    rows, not_found = flatten_fundamentals_responses(fundamentals)
    if not rows:
        raise ValueError("No fundamentals rows found in input")

    style_factors = list(factors) if factors is not None else list(DEFAULT_STYLE_FACTORS)
    unknown = set(style_factors) - set(DEFAULT_STYLE_FACTORS)
    if unknown:
        raise ValueError(f"Unknown style factors: {sorted(unknown)}")

    price_map = _build_price_map(prices)
    symbols, raw, missing_price, sector_by_symbol = _compute_raw_inputs(rows, price_map)

    median_filled: Dict[str, Tuple[List[float], List[int]]] = {}
    for key, values in raw.items():
        if key == "dividend_yield":
            median_filled[key] = ([float(v) for v in values], [])
        else:
            median_filled[key] = _median_fill(values)

    ey_filled, ey_missing = median_filled["earnings_yield"]
    by_filled, by_missing = median_filled["book_yield"]
    z_ey = _zscore(ey_filled, winsorize)
    z_by = _zscore(by_filled, winsorize)
    value_scores = [(a + b) / 2.0 for a, b in zip(z_ey, z_by)]

    style_scores: Dict[str, List[float]] = {
        "value": value_scores,
        "momentum": _zscore(median_filled["momentum_raw"][0], winsorize),
        "size": _zscore(median_filled["log_market_cap"][0], winsorize),
        "dividend_yield": _zscore(median_filled["dividend_yield"][0], winsorize),
        "low_volatility": [-s for s in _zscore(median_filled["vol_proxy"][0], winsorize)],
        "liquidity": _zscore(median_filled["log_dollar_volume"][0], winsorize),
    }

    sector_columns: List[str] = []
    if include_sectors:
        sector_names: Set[str] = set()
        for sector in sector_by_symbol.values():
            if sector:
                sector_names.add(_slugify_sector(str(sector)))
        sector_columns = sorted(sector_names)

    factor_model: List[Dict[str, Any]] = []
    for idx, symbol in enumerate(symbols):
        row: Dict[str, Any] = {"identifier": symbol}
        for name in style_factors:
            row[name] = float(style_scores[name][idx])
        if include_sectors:
            sector = sector_by_symbol.get(symbol)
            slug = _slugify_sector(str(sector)) if sector else "sector_unknown"
            for col in sector_columns:
                row[col] = 1.0 if col == slug else 0.0
        factor_model.append(row)

    all_factor_cols = list(style_factors) + sector_columns
    diagnostics: Dict[str, Dict[str, float]] = {}
    for col in all_factor_cols:
        values = [float(r[col]) for r in factor_model]
        diagnostics[col] = {
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
        }

    missing_inputs: Dict[str, List[str]] = {
        "price": missing_price,
        "earnings_yield": [symbols[i] for i in ey_missing],
        "book_yield": [symbols[i] for i in by_missing],
        "momentum_raw": [symbols[i] for i in median_filled["momentum_raw"][1]],
        "log_market_cap": [symbols[i] for i in median_filled["log_market_cap"][1]],
        "vol_proxy": [symbols[i] for i in median_filled["vol_proxy"][1]],
        "log_dollar_volume": [symbols[i] for i in median_filled["log_dollar_volume"][1]],
    }

    discovered_sectors = sorted(
        {str(s) for s in sector_by_symbol.values() if s}
    )

    return {
        "factor_model": factor_model,
        "coverage": {
            "symbols": symbols,
            "count": len(symbols),
            "not_found": not_found,
            "missing_inputs": missing_inputs,
        },
        "sectors": discovered_sectors,
        "diagnostics": diagnostics,
    }
