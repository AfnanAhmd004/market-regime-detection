import numpy as np

from regimes import GaussianHMM, features, two_regime_market


def _fit():
    ret, truth = two_regime_market(n=1500, seed=1)
    f = features(ret)
    X = ((f - f.mean()) / f.std()).values
    return GaussianHMM(2, seed=0).fit(X), X, truth.reindex(f.index).values


def test_em_log_likelihood_is_non_decreasing():
    hmm, _, _ = _fit()
    assert np.all(np.diff(hmm.loglik_) > -1e-6)


def test_recovers_planted_regimes():
    hmm, X, truth = _fit()
    calm = int(np.argmin(hmm.mean_[:, 1]))
    pred = (hmm.viterbi(X) != calm).astype(int)
    assert (pred == truth).mean() > 0.85


def test_filter_is_causal():
    hmm, X, _ = _fit()
    p = hmm.filter(X)
    X2 = X.copy()
    X2[800:] += 5.0
    assert np.allclose(hmm.filter(X2)[:800], p[:800])


def test_probabilities_are_normalised():
    hmm, X, _ = _fit()
    assert np.allclose(hmm.filter(X).sum(1), 1) and np.allclose(hmm.smooth(X).sum(1), 1)
    assert np.allclose(hmm.trans_.sum(1), 1)
