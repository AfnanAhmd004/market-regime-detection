"""Gaussian hidden Markov model with diagonal covariances, written from scratch in NumPy.

Two ways to read regimes out of a fitted model:

* `filter` - P(state_t | x_1..x_t). Causal: safe to trade on.
* `smooth` / `viterbi` - use the whole sample, including the future. Fine for
  describing history, never for a backtest signal.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


class GaussianHMM:
    def __init__(self, n_states: int = 2, n_iter: int = 100, tol: float = 1e-4, seed: int = 0):
        self.k, self.n_iter, self.tol, self.seed = n_states, n_iter, tol, seed

    # ------------------------------------------------------------------ helpers
    def _log_emission(self, X: np.ndarray) -> np.ndarray:
        var = self.var_
        ll = -0.5 * (np.log(2 * np.pi * var)[None] + (X[:, None, :] - self.mean_[None]) ** 2 / var[None])
        return ll.sum(-1)  # (T, k)

    def _forward(self, logB):
        T = len(logB)
        la = np.empty((T, self.k))
        la[0] = np.log(self.start_) + logB[0]
        logA = np.log(self.trans_)
        for t in range(1, T):
            la[t] = logsumexp(la[t - 1][:, None] + logA, axis=0) + logB[t]
        return la

    def _backward(self, logB):
        T = len(logB)
        lb = np.zeros((T, self.k))
        logA = np.log(self.trans_)
        for t in range(T - 2, -1, -1):
            lb[t] = logsumexp(logA + logB[t + 1][None] + lb[t + 1][None], axis=1)
        return lb

    # ------------------------------------------------------------------ API
    def fit(self, X: np.ndarray) -> "GaussianHMM":
        X = np.asarray(X, float)
        rng = np.random.default_rng(self.seed)
        # initialise means on quantiles of the first feature so states start ordered
        order = np.argsort(X[:, 0])
        chunks = np.array_split(order, self.k)
        self.mean_ = np.array([X[c].mean(0) for c in chunks])
        self.var_ = np.array([X[c].var(0) + 1e-6 for c in chunks])
        self.start_ = np.full(self.k, 1 / self.k)
        self.trans_ = np.full((self.k, self.k), 0.05 / max(1, self.k - 1)) + np.eye(self.k) * (0.95 - 0.05 / max(1, self.k - 1))
        self.trans_ += rng.uniform(0, 1e-3, self.trans_.shape)
        self.trans_ /= self.trans_.sum(1, keepdims=True)
        self.loglik_ = []
        for _ in range(self.n_iter):
            logB = self._log_emission(X)
            la, lb = self._forward(logB), self._backward(logB)
            ll = logsumexp(la[-1])
            self.loglik_.append(float(ll))
            gamma = np.exp(la + lb - ll)
            logA = np.log(self.trans_)
            xi = np.exp(la[:-1, :, None] + logA[None] + logB[1:, None, :] + lb[1:, None, :] - ll).sum(0)
            self.start_ = gamma[0] + 1e-12
            self.trans_ = xi / xi.sum(1, keepdims=True)
            w = gamma.sum(0)
            self.mean_ = gamma.T @ X / w[:, None]
            self.var_ = np.maximum(gamma.T @ X**2 / w[:, None] - self.mean_**2, 1e-8)
            if len(self.loglik_) > 1 and abs(self.loglik_[-1] - self.loglik_[-2]) < self.tol:
                break
        return self

    def filter(self, X: np.ndarray) -> np.ndarray:
        la = self._forward(self._log_emission(np.asarray(X, float)))
        return np.exp(la - logsumexp(la, axis=1, keepdims=True))

    def smooth(self, X: np.ndarray) -> np.ndarray:
        logB = self._log_emission(np.asarray(X, float))
        la, lb = self._forward(logB), self._backward(logB)
        g = la + lb
        return np.exp(g - logsumexp(g, axis=1, keepdims=True))

    def viterbi(self, X: np.ndarray) -> np.ndarray:
        logB = self._log_emission(np.asarray(X, float))
        logA = np.log(self.trans_)
        T = len(logB)
        delta = np.log(self.start_) + logB[0]
        back = np.zeros((T, self.k), dtype=int)
        for t in range(1, T):
            scores = delta[:, None] + logA
            back[t] = scores.argmax(0)
            delta = scores.max(0) + logB[t]
        path = np.empty(T, dtype=int)
        path[-1] = delta.argmax()
        for t in range(T - 1, 0, -1):
            path[t - 1] = back[t, path[t]]
        return path
