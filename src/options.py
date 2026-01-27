import numpy as np


def payoff_call(S_T: np.ndarray, K: float) -> np.ndarray:
    """
    European call payoff: max(S_T - K, 0)
    Vectorized over S_T.
    """
    S_T = np.asarray(S_T, dtype=float)
    K = float(K)
    return np.maximum(S_T - K, 0.0)


def payoff_put(S_T: np.ndarray, K: float) -> np.ndarray:
    """
    European put payoff: max(K - S_T, 0)
    Vectorized over S_T.
    """
    S_T = np.asarray(S_T, dtype=float)
    K = float(K)
    return np.maximum(K - S_T, 0.0)
