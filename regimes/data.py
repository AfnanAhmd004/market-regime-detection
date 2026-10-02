"""Synthetic two-regime market and causal features."""
from __future__ import annotations

import numpy as np
import pandas as pd


def two_regime_market(n: int = 2500, p_stay=(0.99, 0.97), mu=(0.0006, -0.0010), sigma=(0.008, 0.022), seed: int = 0):
    """Calm-bull / turbulent-bear Markov-switching returns. Returns (returns, true_states)."""
    rng = np.random.default_rng(seed)
    s = np.empty(n, dtype=int)
    s[0] = 0
    for t in range(1, n):
        s[t] = s[t - 1] if rng.random() < p_stay[s[t - 1]] else 1 - s[t - 1]
    r = rng.normal(np.take(mu, s), np.take(sigma, s))
    idx = pd.bdate_range("2014-01-01", periods=n)
    return pd.Series(r, index=idx, name="ret"), pd.Series(s, index=idx, name="state")


def features(returns: pd.Series, vol_window: int = 10) -> pd.DataFrame:
    """Return and short rolling volatility (log), both known at the close of day t."""
    f = pd.DataFrame({"ret": returns, "log_vol": np.log(returns.rolling(vol_window).std())})
    return f.dropna()
