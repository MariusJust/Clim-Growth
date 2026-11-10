"""Frisch-Waugh-Lovell (within) projection for the global static model.

Builds the nuisance design ``W`` = [unit dummies | time dummies |
unit/country-specific linear trend | quadratic trend] over the OBSERVED cells,
in the same row order the loss uses (``argwhere(~mask)``, i.e. time-major then
unit-ascending -- identical to how ``Matrixize`` scatters and ``y_pred[~mask]``
reads back). It exposes the annihilator

    P v = v - W (WᵀW)⁺ Wᵀ v

as an *implicit* operator (O(n·p), never the dense n×n matrix), the OLS
recovery ``γ̂ = (WᵀW)⁺ Wᵀ r`` for reconstructing FE/trends post-fit, and
``rank(W)`` for the effective parameter count.

Used only when ``instance.within_projection`` is true. The pseudo-inverse makes
the build robust to the usual dummy collinearities (no manual reference-dropping).
Trend columns use the (centred) time index; the column space of
[FE, FE·t, FE·t²] is invariant to any affine reparameterisation of t, so the
projection does not depend on the exact time encoding.

Subnational ('ee') support -- three additions, all inert for the 'wb' path
--------------------------------------------------------------------------
``trend_groups``
    Map each unit to a group (country) so trends are estimated per country
    rather than per unit. On the admin-1 panel this takes p from
    2287 + 32 + 2·2287 = 6893 down to 2287 + 32 + 2·207 = 2733, which is what
    makes the projection fit in memory and the pseudo-inverse 16x cheaper.
    ``None`` (default) keeps one trend per unit, exactly as before.

``weights``
    Per-unit regression weights, expanded to observations. Estimation solves
    the weighted problem ``min Σ wᵢ (yᵢ - fᵢ - Wᵢγ)²``, implemented by the
    √w transform: with ``s = √w``, ``W̃ = s⊙W`` and the annihilator becomes
    ``P̃ v = s⊙v - W̃ (W̃ᵀW̃)⁺ W̃ᵀ (s⊙v)``. Both the network and the nuisance
    parameters are then fitted on the same weighted objective -- weighting only
    the network would leave the FE and trends fitted on a different sample.
    ``None`` (default) gives s = 1 and reproduces the unweighted operator.

no dense ``B``
    The old code stored ``B = (WᵀW)⁺Wᵀ``, a p×n matrix -- 1.49 GB in float64 on
    the ee panel, plus another 0.80 GB once TensorFlow held a float32 copy in
    the graph. We store ``M = (W̃ᵀW̃)⁺`` (p×p, 0.06 GB) instead and apply it as
    ``W̃ (M (W̃ᵀ v))``. Algebraically identical; the caller must use
    ``matvec(Ws, v, transpose_a=True)`` in place of ``matvec(B, v)``.

``rank`` comes from the eigenvalues of the p×p Gram matrix rather than an SVD of
W itself: identical answer, ~25x cheaper on the ee panel (a 73184×2733 SVD is
~5.5e11 flops).

``self.W`` remains the UNWEIGHTED design, because post-fit reconstruction
(``f_obs + W @ gamma``) must return fitted values in original units.
"""

import numpy as np


class WithinProjector:
    def __init__(self, mask_TN, country_trends=True, quadratic_trends=True,
                 include_time=True, trend_groups=None, weights=None):
        obs = ~np.asarray(mask_TN, dtype=bool)
        T, N = obs.shape
        idx = np.argwhere(obs)
        t_arr, n_arr = idx[:, 0], idx[:, 1]
        n_obs = idx.shape[0]
        self.t_arr = t_arr
        self.n_arr = n_arr
        self.n_obs = n_obs

        # Column layout: [unit dummies | time dummies | linear trend | quadratic
        # trend]. W is preallocated and filled in place: building the blocks
        # separately and np.concatenate-ing them peaks at ~2x the final size
        # (3.1 GB on the ee panel for a 1.49 GB result).
        if trend_groups is None:
            gi, n_groups = n_arr, N                      # one trend per unit
        else:
            g = np.asarray(trend_groups)
            if g.shape[0] != N:
                raise ValueError(
                    f"trend_groups has length {g.shape[0]}, expected {N} (one per unit)")
            _, gcode = np.unique(g, return_inverse=True)
            gi, n_groups = gcode[n_arr], int(gcode.max()) + 1   # one trend per group

        n_trend = (1 + int(bool(quadratic_trends))) if country_trends else 0
        p = N + (T if include_time else 0) + n_trend * n_groups
        W = np.zeros((n_obs, p), dtype=np.float64)
        rows = np.arange(n_obs)
        W[rows, n_arr] = 1.0
        off = N
        if include_time:
            W[rows, off + t_arr] = 1.0
            off += T
        if country_trends:
            tc = t_arr.astype(np.float64)
            tc = tc - tc.mean()
            W[rows, off + gi] = tc
            off += n_groups
            if quadratic_trends:
                W[rows, off + gi] = tc ** 2
                off += n_groups
        self.W = W                                       # UNWEIGHTED (reconstruction)
        self.p = p

        # ---- weights: solve the √w-transformed problem -----------------------
        # W̃ = s⊙W is never materialised: it would double peak memory (2 x 1.49 GB
        # on the ee panel). Everything is expressed through W and w instead, using
        #     W̃ᵀ(s⊙v) = Wᵀ(w⊙v)        and        W̃c = s⊙(Wc).
        if weights is None:
            self.w = np.ones(n_obs, dtype=np.float64)
            self.s = self.w
        else:
            wu = np.asarray(weights, dtype=np.float64).ravel()
            if wu.shape[0] == N:
                w = wu[n_arr]                            # per-unit -> per-observation
            elif wu.shape[0] == n_obs:
                w = wu
            else:
                raise ValueError(
                    f"weights has length {wu.shape[0]}, expected {N} (per unit) "
                    f"or {n_obs} (per observation)")
            if np.any(w <= 0) or not np.all(np.isfinite(w)):
                raise ValueError("weights must be finite and strictly positive")
            self.w = w
            self.s = np.sqrt(w)
        self.sum_w = float(self.w.sum())

        # G = W̃ᵀW̃ = Wᵀ diag(w) W, accumulated in chunks so the weighted copy of
        # W is never held whole (a full W*w temporary is 1.49 GB on the ee panel).
        G = np.zeros((self.p, self.p), dtype=np.float64)
        step = max(1, int(2e8 // max(self.p, 1)))
        for i in range(0, n_obs, step):
            Wi = W[i:i + step]
            G += Wi.T @ (Wi * self.w[i:i + step, None])
        self.M = np.linalg.pinv(G)
        # rank(W̃) = rank(W̃ᵀW̃); eigenvalues of the p×p Gram are enough.
        ev = np.linalg.eigvalsh(G)
        tol = max(G.shape) * np.finfo(np.float64).eps * (ev.max() if ev.size else 0.0)
        self.rank = int(np.sum(ev > tol))

    @property
    def B(self):
        """Legacy dense ``(W̃ᵀW̃)⁺W̃ᵀ`` (p x n), materialised on demand.

        The global/within path no longer uses this -- it applies ``M`` via
        ``transpose_a`` instead, which is what keeps the ee panel inside memory
        (this matrix is 1.49 GB there, plus 0.80 GB once TF holds a float32 copy).
        It is retained because the regional model and the NIC code still take the
        dense form, and both run on the country-sized 'wb' panel where p is small.
        Do NOT touch it on a subnational panel.
        """
        return (self.M @ self.W.T) * self.s[None, :]

    def annihilate(self, v):
        """Weighted within projection ``P̃ v = s⊙v - W̃ (M (W̃ᵀ (s⊙v)))``.

        Takes the UNWEIGHTED vector and returns the weighted, annihilated one,
        so the caller's sum of squares is the weighted SSE. With unit weights
        this is exactly ``v - W (WᵀW)⁺ Wᵀ v`` as before.
        """
        v = np.asarray(v, dtype=np.float64)
        return self.s * (v - self.W @ (self.M @ (self.W.T @ (self.w * v))))

    def recover_gamma(self, resid):
        """Weighted FE/trend coefficients for residual r = y - f: γ̂ = M Wᵀ(w⊙r)."""
        r = np.asarray(resid, dtype=np.float64)
        return self.M @ (self.W.T @ (self.w * r))
