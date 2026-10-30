"""
model.py

Builds the LP to be solved by arbitrage.py annd exit_portfolio.py

Decision vector (length 2n):  x = [L_1..L_n, S_1..S_n]
    L_i >= 0 : units of option i bought (at ask a_i)
    S_i >= 0 : units of option i sold   (at bid b_i)
 
Base model:
    minimize    sum_i a_i * L_i - b_i * S_i
    subject to  sum_i (L_i - S_i) * X_ij >= 0   for every state j (see below explanation for j = n + 1)
                0 <= L_i, S_i <= position_limit
 
    X_ij is the expiry payoff of option i in state j. (states correspond to possible spot prices)
    By piecewise linearity of option payoffs, considering the following set of states is sufficient:
        - $0
        - $K_i (every strike price)
        - any state above the largest strike

    To enforce nonnegative payoff for the infinite states above the largest strike, we form a final
    slope constraint by dividing the base constraint by the state price and taking the limit as it
    grows arbitrarily, resulting in a LHS of (L_i - S_i) for call options and 0 for put options, as expected.


Portfolio (exit) model: 
    same variables plus bond columns, x = [L, S, L_B, S_B]
        minimize    sum_i a_i * L_i - b_i * S_i + e^{-rT} * (L_B - S_B)
        subject to  sum_i (L_i - S_i) * X_ij + (L_B - S_B) = -sum_i h_i * X_ij  for all j
    i.e. the trades plus bond exactly cancel the payoff of the current position h in every state.
    The bond pays 1 unit in every state except for the slope state above the largest strike,
    in which case the same limit calculation as before yields 0 (which makes sense given constant payoff).
"""

from dataclasses import dataclass
 
import numpy as np


@dataclass(frozen=True)
class LPModel:
    """Dataclass with all fields needed to reconstruct and solve LP."""

    c: np.ndarray
    A_ub: np.ndarray | None
    b_ub: np.ndarray | None
    bounds: tuple[float, float | None]
    n_options: int
    A_eq: np.ndarray | None = None
    b_eq: np.ndarray | None = None


def payoff_matrix(strikes: np.ndarray, is_call: np.ndarray) -> np.ndarray:
    """Return payoff matrix X of shape (n_options, n_states)."""

    strikes = np.asarray(strikes, dtype=float)
    is_call = np.asarray(is_call, dtype=bool)
 
    states = np.concatenate(([0.0], np.unique(strikes)))
    S = states[None, :]
    K = strikes[:, None]
    X = np.where(is_call[:, None], np.maximum(S - K, 0.0), np.maximum(K - S, 0.0))
 
    above_max = is_call[:, None].astype(float)

    return np.hstack([X, above_max])


def build_base_model(
    strikes: np.ndarray, 
    is_call: np.ndarray, 
    bid: np.ndarray, 
    ask: np.ndarray, 
    position_limit: float = 1.0
) -> LPModel:
    """
    Base arbitrage-detection LP. A negative objective means arbitrage exists.
 
    position_limit caps the amount of long/short orders for each option to ensure boundedness.
    """
    
    bid = np.asarray(bid, dtype=float)
    ask = np.asarray(ask, dtype=float)
    n = len(bid)
 
    X = payoff_matrix(strikes, is_call)
 
    c = np.concatenate([ask, -bid])
 
    A_ub = np.hstack([-X.T, X.T])
    b_ub = np.zeros(X.shape[1])
 
    bounds = (0.0, position_limit)
    
    return LPModel(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, n_options=n)


def build_portfolio_model(
    strikes: np.ndarray,
    is_call: np.ndarray,
    bid: np.ndarray,
    ask: np.ndarray,
    holdings: np.ndarray,
    rate: float,
    time_to_expiry: float,
    position_limit: float = 1.0,
) -> LPModel:
    """
    Exit-portfolio LP, introduces zero-coupon bond.
 
    holdings[i] is the net position in option i.
    """

    holdings = np.asarray(holdings, dtype=float)
    n = len(holdings)

    # let position_limit match max holdings
    position_limit = max(position_limit, float(np.max(np.abs(holdings))))
 
    base = build_base_model(strikes, is_call, bid, ask, position_limit)
    X = payoff_matrix(strikes, is_call)
    m = X.shape[1]
 
    bond_price = np.exp(-rate * time_to_expiry)
 
    c = np.concatenate([base.c, [bond_price, -bond_price]])
 
    bond_col = np.ones(m)
    bond_col[-1] = 0.0
 
    A_eq = np.hstack([X.T, -X.T, bond_col[:, None], -bond_col[:, None]])
    b_eq = -(X.T @ holdings)
 
    return LPModel(c=c, A_ub=None, b_ub=None, bounds=base.bounds, n_options=n, A_eq=A_eq, b_eq=b_eq)