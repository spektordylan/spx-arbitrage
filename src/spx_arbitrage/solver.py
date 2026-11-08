"""
solver.py

Solver object for parts 1 and 2.
If holdings supplied, solves exit portfolio model.
Else, solves base arbitrage detection model:
    - Solve for strong arbitrage. 
    - If strong arbitrage objective is zero (meaning cannot be paid to enter), solve for weak arbitrage.
    - If weak arbitrage objective gives positive max total payoff, we have weak arbitrage.
    - Otherwise, no arbitrage. 

Returns resulting objective and solution vector.
For arbitrage detection, self.arbitrage_type is set to "strong", "weak" or "none".
"""

import numpy as np
from scipy.optimize import linprog

import spx_arbitrage.model as model

TOL = 1e-9

class Solver:
    """
    Detects arbitrage opportunities and/or constructs exit portfolio using LPs in model.py.
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
        arbitrage_type=None,
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

        If holdings are supplied, constructs exit portfolio.
        Else, performs arbitrage detection:
            - Solves strong arbitrage LP.
            - If strong objective zero, solves weak arbitrage LP.

        Returns the resulting objective value and solution vector.
        """

        if self.holdings is not None:
            lp_model = model.build_portfolio_model(
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
            lp_model = model.build_strong_arb_model(
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

        # if problem is exit portfolio, we are done
        if self.holdings is not None:
            return result.fun, result.x

        # if strong arbitrage was detected, we are done
        if result.fun < -TOL:
            self.arbitrage_type = "strong"
            return result.fun, result.x

        # otherwise strong arbitrage objective was zero, solve weak arb LP
        weak_model = model.build_weak_arb_model(lp_model)
 
        weak_result = linprog(
            c=weak_model.c,
            A_ub=weak_model.A_ub,
            b_ub=weak_model.b_ub,
            bounds=weak_model.bounds,
            method="highs",
        )

        if not weak_result.success:
            raise RuntimeError(f"LP solver failed: {weak_result.message}")

        # if negated max total payoff is strictly negative, we have weak arbitrage
        if weak_result.fun < -TOL:
            self.arbitrage_type = "weak"
            return weak_result.fun, weak_result.x

        # otherwise no arbitrage at all
        self.arbitrage_type = "none"
        return result.fun, result.x