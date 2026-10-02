"""Fit an HMM on the first half, then scale exposure by the filtered probability of the calm regime."""
import numpy as np

from regimes import GaussianHMM, features, two_regime_market

ret, truth = two_regime_market(seed=4)
f = features(ret)
split = len(f) // 2
X = (f - f.iloc[:split].mean()) / f.iloc[:split].std()  # scale with training statistics only

hmm = GaussianHMM(n_states=2, seed=0).fit(X.iloc[:split].values)
calm = int(np.argmin(hmm.mean_[:, 1]))  # the state with lower volatility
p_calm = hmm.filter(X.values)[:, calm]  # causal: uses data up to t only

test = slice(split, len(f) - 1)
r_next = ret.reindex(f.index).shift(-1).values[test]
strat = p_calm[test] * r_next  # exposure decided at t, earns t -> t+1

def sharpe(x):
    return x.mean() / x.std() * np.sqrt(252)

acc = (np.argmax(hmm.filter(X.values), 1) == calm).astype(int)
true_calm = (truth.reindex(f.index).values == 0).astype(int)
print(f"filtered regime accuracy (test): {np.mean(acc[test] == true_calm[test]):.2%}")
print(f"buy & hold Sharpe:               {sharpe(r_next):.2f}")
print(f"regime-scaled Sharpe:            {sharpe(strat):.2f}")
