"""
solver.py

Solver object for parts 1 and 2.
If holdings supplied, solves exit portfolio model.
Else, solves base arbitrage detection model.

Returns resulting objective and solution vector.
"""

import numpy as np
from scipy.optimize import linprog

from spx_arbitrage.model import build_base_model, build_portfolio_model

class Solver:
    """
    Detects arbitrage opportunities by solving the LP constructed in model.py.
    """

    def __init__(
        self,
        strikes,
        is_call,
        bid,
        ask,
        position_limit=1.0,
        holdings=None,
        rate=0.0,
        time_to_expiry=0.0,
    ):
        self.strikes = np.asarray(strikes, dtype=float)
        self.is_call = np.asarray(is_call, dtype=bool)
        self.bid = np.asarray(bid, dtype=float)
        self.ask = np.asarray(ask, dtype=float)
        self.position_limit = float(position_limit)
        
        self.holdings = np.asarray(holdings, dtype=float) if holdings is not None else None
        self.rate = float(rate)
        self.time_to_expiry = float(time_to_expiry)


    def main(self) -> tuple[float, np.ndarray]:
        """
        Builds the appropriate LP model and solves it. 
        Returns the resulting objective value and solution vector.
        
        Note: In the base model, a negative objective indicates an arbitrage opportunity.
        """

        if self.holdings is not None:
            lp_model = build_portfolio_model(
                strikes=self.strikes,
                is_call=self.is_call,
                bid=self.bid,
                ask=self.ask,
                holdings=self.holdings,
                rate=self.rate,
                time_to_expiry=self.time_to_expiry,
                position_limit=self.position_limit,
            )
        else:
            lp_model = build_base_model(
                strikes=self.strikes,
                is_call=self.is_call,
                bid=self.bid,
                ask=self.ask,
                position_limit=self.position_limit,
            )

        result = linprog(
            c=lp_model.c,
            A_ub=lp_model.A_ub,
            b_ub=lp_model.b_ub,
            A_eq=lp_model.A_eq,
            b_eq=lp_model.b_eq,
            bounds=lp_model.bounds,
            method="highs",
        )

        if not result.success:
            raise RuntimeError(f"LP solver failed: {result.message}")

        return result.fun, result.x