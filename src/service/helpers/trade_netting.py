from decimal import Decimal
from operator import itemgetter, methodcaller
import pandas as pd
from typing import Dict, Tuple, Optional
import numpy as np

NETTED_TRADES_COLUMNS = ["identifier", "action", "quantity", "price", "tax_lot_id", "short_term_gain", "short_term_loss", "long_term_gain", "long_term_loss"]

def net_trades_across_strategies(strategy_results: Dict[int, Tuple[Optional[int], bool, Dict, pd.DataFrame]], trade_rounding: int) -> pd.DataFrame:
    trades = pd.concat(strategy_trades for _, _, _, strategy_trades in strategy_results.values()).copy()
    if trades.empty:
        return pd.DataFrame(columns=NETTED_TRADES_COLUMNS)
    
    trades["quantity"] = trades["quantity"].map(lambda quantity: round(Decimal(quantity), trade_rounding))
    trades["quantity"].mask(trades["action"] == "sell", -trades["quantity"], inplace=True)

    net = trades.groupby("identifier").agg({
        "quantity": "sum",
        "price": "first",
    })
    net = net[net["quantity"] != 0]
    net["action"] = np.where(net["quantity"] > 0, "buy", "sell")

    trades.sort_values("quantity", ascending=False, inplace=True)
    trades["cumqty"] = trades.groupby("identifier")["quantity"].transform(methodcaller("cumsum"))
    sells = trades[trades["cumqty"] < 0]
    sells["netqty"] = sells["quantity"].clip(lower=sells["cumqty"])

    netpct = (sells["netqty"] / sells["quantity"]).astype(float)
    realized_gain = sells["gain_loss"].map(itemgetter("realized_gain")) * netpct
    gain_type = sells["gain_loss"].map(itemgetter("gain_type"))
    sells["short_term_gain"] = realized_gain.mask(gain_type != "short_term", 0).clip(lower=0)
    sells["short_term_loss"] = realized_gain.mask(gain_type != "short_term", 0).clip(upper=0).abs()
    sells["long_term_gain"] = realized_gain.mask(gain_type != "long_term", 0).clip(lower=0)
    sells["long_term_loss"] = realized_gain.mask(gain_type != "long_term", 0).clip(upper=0).abs()

    matching_lots = sells[["identifier", "tax_lot_id", "short_term_gain", "short_term_loss", "long_term_gain", "long_term_loss", "netqty"]]
    assert not matching_lots.duplicated(["identifier", "tax_lot_id"]).any(), "duplicate sell tax lots"

    net = net.merge(matching_lots, how="left", on="identifier")
    net["quantity"].mask(net["action"] == "sell", -net["netqty"], inplace=True)
    net["quantity"] = net["quantity"].astype(float)

    assert (net["quantity"] > 0).all(), "invalid netting"

    # Reset index to make identifier a column before returning
    return net[NETTED_TRADES_COLUMNS]
