"""Frisch-Waugh-Lovell (within) projection for the global static model.

Builds the nuisance design ``W`` = [country dummies | time dummies |
country-specific linear trend | country-specific quadratic trend] over the
OBSERVED cells, in the same row order the loss uses (``argwhere(~mask)``, i.e.
time-major then country-ascending -- identical to how ``Matrixize`` scatters and
``y_pred[~mask]`` reads back). It exposes the annihilator

    P v = v - W (WᵀW)⁺ Wᵀ v

as an *implicit* operator (O(n·p), never the dense n×n matrix), the OLS
recovery ``γ̂ = (WᵀW)⁺ Wᵀ r`` for reconstructing FE/trends post-fit, and
``rank(W)`` for the effective parameter count.

Used only when ``instance.within_projection`` is true. The pseudo-inverse makes
the build robust to the usual dummy collinearities (no manual reference-dropping).
Trend columns use the (centred) time index; the column space of
[FE, FE·t, FE·t²] is invariant to any affine reparameterisation of t, so the
projection does not depend on the exact time encoding.
"""

import numpy as np


class WithinProjector:
    def __init__(self, mask_TN, country_trends=True, quadratic_trends=True,
                 include_time=True):
        obs = ~np.asarray(mask_TN, dtype=bool)
        T, N = obs.shape
        idx = np.argwhere(obs)
        t_arr, n_arr = idx[:, 0], idx[:, 1]
        n_obs = idx.shape[0]
        self.t_arr = t_arr
        self.n_arr = n_arr
        self.n_obs = n_obs

        Wc = np.zeros((n_obs, N), dtype=np.float64)
        Wc[np.arange(n_obs), n_arr] = 1.0
        blocks = [Wc]
        if include_time:
            Wt = np.zeros((n_obs, T), dtype=np.float64)
            Wt[np.arange(n_obs), t_arr] = 1.0
            blocks.append(Wt)
        if country_trends:
            tc = t_arr.astype(np.float64)
            tc = tc - tc.mean()
            blocks.append(Wc * tc[:, None])
            if quadratic_trends:
                blocks.append(Wc * (tc ** 2)[:, None])
        W = np.concatenate(blocks, axis=1)
        self.W = W

        self.B = np.linalg.pinv(W.T @ W) @ W.T          # (p, n_obs)
        self.rank = int(np.linalg.matrix_rank(W))
        self.p = W.shape[1]

    def annihilate(self, v):
        """Within projection ``P v = v - W (WᵀW)⁺ Wᵀ v`` (implicit, O(n·p))."""
        v = np.asarray(v, dtype=np.float64)
        return v - self.W @ (self.B @ v)

    def recover_gamma(self, resid):
        """OLS FE/trend coefficients for residual r = y - f:  γ̂ = B r."""
        return self.B @ np.asarray(resid, dtype=np.float64)
