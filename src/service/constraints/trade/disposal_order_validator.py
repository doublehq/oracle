"""Force sells within a symbol to follow a broker disposal method (FIFO/LIFO/HIFO)."""

from typing import Optional, Tuple

import pulp

from src.service.constraints.base_validator import BaseValidator
from src.service.helpers.constants import CASH_CUSIP_ID
from src.service.helpers.disposal import lots_in_disposal_order, normalize_disposal_method
from src.service.helpers.lp_names import safe_lp_name


class DisposalOrderValidator(BaseValidator):
    """Require that lot i+1 is only sold after lot i is fully exhausted, in disposal order."""

    def __init__(self, oracle_strategy, disposal_method: str = "FIFO"):
        super().__init__(oracle_strategy)
        self.disposal_method = normalize_disposal_method(disposal_method)

    def validate_buy(self, identifier: str, quantity: float) -> Tuple[bool, Optional[str]]:
        return True, None

    def validate_sell(self, tax_lot_id: str, quantity: float) -> Tuple[bool, Optional[str]]:
        # Ordering is a joint constraint across lots; individual sells are not rejected here.
        return True, None

    def add_to_problem(self, prob: pulp.LpProblem, sells: dict, tax_lots) -> None:
        if tax_lots is None or tax_lots.empty:
            return

        for identifier, group in tax_lots.groupby("identifier"):
            if identifier == CASH_CUSIP_ID:
                continue
            ordered = lots_in_disposal_order(group.to_dict(orient="records"), self.disposal_method)
            for current, nxt in zip(ordered, ordered[1:]):
                tid = current["tax_lot_id"]
                next_tid = nxt["tax_lot_id"]
                if tid not in sells or next_tid not in sells:
                    continue
                qty = float(current["quantity"])
                next_qty = float(nxt["quantity"])
                if qty <= 0 or next_qty <= 0:
                    continue
                exhausted = pulp.LpVariable(
                    f"disposal_exhausted_{safe_lp_name(tid)}",
                    cat="Binary",
                )
                # exhausted = 1 ⇒ current lot must be fully sold
                prob += (
                    sells[tid] >= qty * exhausted,
                    f"disposal_exhaust_{safe_lp_name(tid)}",
                )
                # next lot can only be sold if the previous lot is exhausted
                prob += (
                    sells[next_tid] <= next_qty * exhausted,
                    f"disposal_next_{safe_lp_name(next_tid)}",
                )
