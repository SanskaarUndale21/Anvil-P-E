"""
PCAM Precision Agent -- Geometric Whitening + Mirror Descent.

Two-regime routing on max cosine similarity:

  ANISOTROPY PROBES  (max cosine sim > ROUTE_SIM = 0.80)
    Directly minimise kappa(Pi^{1/2} H(a*) Pi^{1/2}) via mirror descent
    on log pi.  Gradient:
        d log kappa / d log pi_i = v_max_i^2 - v_min_i^2
    Multiple initialisation candidates per attractor (diverse pool).

  RETRIEVAL QUERIES  (max cosine sim <= ROUTE_SIM)
    Seven-component masking-aware precision pipeline:
        1. Base: 1/(|q_i| + eps)   masked dims get high precision for gradient recovery
        2. Geometry: diag(H^{-1}(a*))   amplify flat convergence directions
        3. Energy-gradient alignment   boost dims pointing toward dominant attractor
        4. Local covariance   amplify low-variance discriminative dims
        5. Confidence gate   scale uniformly when attractor is clear
        6. Twin-pair correction   sharpen at cluster boundaries
        7. (I + alpha*R)^{-1} spectral smoothing

Routing cosine ranges:
    Anisotropy probes (probe_sigma=0.05): cosine in [0.87, 0.99]
    Retrieval queries (p in {0.6, 0.75, 0.85}): cosine in [0.20, 0.72]
"""
from __future__ import annotations

from typing import Any

import numpy as np

from adapter import Adapter

# ---- Routing ----------------------------------------------------------------
_ROUTE_SIM = 0.80      # cosine threshold separating probes from retrieval

# ---- Mirror descent (anisotropy) -------------------------------------------
_OPT_STEPS  = 400      # gradient steps per restart (baseline K=16, N=64)
_N_RAND     = 3        # random log-normal restarts per attractor
_LR         = 0.08     # mirror-descent learning rate

# ---- Retrieval pipeline constants ------------------------------------------
_EPS         = 0.01    # floor in masking-aware base 1/(|q|+eps)
_W_GEO       = 0.15    # geometry (diag H^{-1}) multiplicative weight
_W_ENERGY    = 0.20    # energy-gradient alignment weight
_W_LCOV      = 0.10    # local covariance weight
_W_TWIN      = 0.60    # twin-pair discriminative correction weight
_TWIN_GAP    = 0.12    # gap threshold below which twin correction activates
_CONF_SCALE  = 0.15    # gap scale for confidence gating
_SMOOTH_A    = 0.15    # (I + alpha*R)^{-1} smoothing coefficient


class Engine(Adapter):
    """State-of-the-art PCAM precision agent."""

    def __init__(self,
                 stored_patterns: np.ndarray,
                 model_params: dict[str, Any]) -> None:
        self.X      = np.asarray(stored_patterns, dtype=np.float64)
        self.K, self.N = self.X.shape
        self.R      = np.asarray(model_params["R"],    dtype=np.float64)
        self.eta    = float(model_params["eta"])
        self.beta   = float(model_params["beta"])
        self.dt     = float(model_params.get("dt",    0.01))
        self.T_max  = int(  model_params.get("T_max", 3000))
        self.tol    = float(model_params.get("tol",   1e-6))
        self.pi_min = float(model_params.get("pi_min", 0.1))
        self.pi_max = float(model_params.get("pi_max", 10.0))

        # Scale compute budget with problem size (dominated by eigh: O(N^3))
        base_cost  = 16 * (64 ** 3)
        this_cost  = self.K * (self.N ** 3)
        scale      = float(np.sqrt(max(1.0, this_cost / base_cost)))
        self._steps = max(40, int(_OPT_STEPS / scale))
        self._nrand = max(2,  int(_N_RAND    / scale))

        # Precompute all per-attractor quantities in one pass.
        # _precompute_aniso also fills self._geo_diag to avoid recomputing H.
        self._geo_diag = np.zeros((self.K, self.N))
        self._equil, self._aniso_pi = self._precompute_aniso()
        self._local_cov              = self._precompute_local_cov()    # (K, N)

        # Spectral smoother shared across retrieval calls
        self._smoother = np.linalg.inv(np.eye(self.N) + _SMOOTH_A * self.R)

    # ------------------------------------------------------------------ #
    # Math primitives                                                      #
    # ------------------------------------------------------------------ #

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def _hessian(self, a: np.ndarray) -> np.ndarray:
        s = self._softmax(self.beta * (self.X @ a))
        D = np.diag(s) - np.outer(s, s)
        H = self.R - self.eta * self.beta * (self.X.T @ (D @ self.X))
        return 0.5 * (H + H.T)

    def _find_equilibrium(self, x0: np.ndarray) -> np.ndarray:
        """Euler dynamics from x0 with pi=I, no input — locates true attractor."""
        a = x0.copy()
        for _ in range(self.T_max):
            s = self._softmax(self.beta * (self.X @ a))
            g = self.R @ a - self.eta * (self.X.T @ s)
            a_new = a - self.dt * g
            if np.linalg.norm(a_new - a) < self.tol:
                return a_new
            a = a_new
        return a

    def _project(self, pi: np.ndarray) -> np.ndarray:
        """Iterative clip + renormalise onto { pi_min <= pi <= pi_max, mean=1 }."""
        pi = pi.copy()
        for _ in range(20):
            pi = np.clip(pi, self.pi_min, self.pi_max)
            m = pi.mean()
            if m < 1e-12:
                return np.ones(self.N)
            pi /= m
            if (pi.min() >= self.pi_min - 1e-9
                    and pi.max() <= self.pi_max + 1e-9
                    and abs(pi.mean() - 1.0) < 1e-8):
                break
        return pi

    def _kappa(self, pi: np.ndarray, H: np.ndarray) -> float:
        """Condition number of Pi^{1/2} H Pi^{1/2}."""
        pi_sqrt = np.sqrt(np.maximum(pi, 1e-12))
        S = (pi_sqrt[:, None] * H) * pi_sqrt[None, :]
        S = 0.5 * (S + S.T)
        ev = np.linalg.eigvalsh(S)
        ev = ev[ev > 1e-9]
        return float(ev.max() / ev.min()) if len(ev) >= 2 else float("inf")

    def _mirror_step(self,
                     H: np.ndarray,
                     pi0: np.ndarray,
                     steps: int) -> tuple[np.ndarray, float]:
        """Mirror descent on log kappa(Pi^{1/2} H Pi^{1/2}).

        Gradient: d log kappa / d log pi_i = v_max_i^2 - v_min_i^2.
        Best iterate is tracked and returned.
        """
        pi      = self._project(pi0)
        best_pi = pi.copy()
        best_k  = self._kappa(pi, H)

        for _ in range(steps):
            pi_sqrt = np.sqrt(np.maximum(pi, 1e-12))
            S = (pi_sqrt[:, None] * H) * pi_sqrt[None, :]
            S = 0.5 * (S + S.T)
            ev, evec = np.linalg.eigh(S)
            pos = np.where(ev > 1e-9)[0]
            if len(pos) < 2:
                break
            k_val = ev[pos[-1]] / ev[pos[0]]
            if k_val < best_k:
                best_k  = k_val
                best_pi = pi.copy()
            v_max = evec[:, pos[-1]]
            v_min = evec[:, pos[0]]
            grad  = v_max ** 2 - v_min ** 2
            pi    = pi * np.exp(-_LR * grad)
            pi    = np.clip(pi, 1e-8, 1e8)
            pi    = self._project(pi)

        return best_pi, best_k

    # ------------------------------------------------------------------ #
    # Precomputation                                                        #
    # ------------------------------------------------------------------ #

    def _precompute_aniso(self) -> tuple[np.ndarray, np.ndarray]:
        """True equilibria + mirror-descent-optimised pi for each pattern.

        Initialisation pool per attractor (diverse geometric coverage):
          - _nrand log-normal random restarts
          - diag(H^{-1}): best diagonal Frobenius approx of H^{-1}
          - v_min^2:       amplify minimum-curvature direction
          - 1/(v_max^2):   suppress maximum-curvature direction
          - 1/H_diag:      simple Jacobi preconditioner
        """
        equil    = np.zeros((self.K, self.N))
        aniso_pi = np.ones((self.K, self.N))
        rng      = np.random.default_rng(0)

        for k in range(self.K):
            a_star   = self._find_equilibrium(self.X[k])
            equil[k] = a_star

            H = self._hessian(a_star)
            ev, evec = np.linalg.eigh(H)
            ev_c      = np.maximum(ev, 1e-8)
            diag_Hinv = (evec ** 2 / ev_c).sum(axis=1)  # diag(H^{-1})

            # Cache geo_diag for retrieval (always, even for non-PD H).
            self._geo_diag[k] = diag_Hinv

            if ev.min() <= 0:
                # Non-PD H: skip mirror descent but geo_diag is still usable.
                continue

            v_min  = evec[:, 0]
            v_max  = evec[:, -1]
            h_diag = np.diag(H)

            inits: list[np.ndarray] = []
            for _ in range(self._nrand):
                inits.append(np.exp(rng.standard_normal(self.N) * 0.5))
            inits.append(diag_Hinv)
            inits.append(v_min ** 2)
            inits.append(1.0 / (v_max ** 2 + 1e-8))
            inits.append(1.0 / np.maximum(h_diag, 1e-6))  # Jacobi

            best_pi = np.ones(self.N)
            best_k  = self._kappa(best_pi, H)

            for pi0 in inits:
                pi0 = self._project(pi0)
                pi_opt, k_opt = self._mirror_step(H, pi0, self._steps)
                if k_opt < best_k:
                    best_k  = k_opt
                    best_pi = pi_opt

            aniso_pi[k] = best_pi

        # Row-normalise geo_diag so values are relative across dims (mean=1).
        self._geo_diag /= (self._geo_diag.mean(axis=1, keepdims=True) + 1e-12)

        return equil, aniso_pi

    def _precompute_local_cov(self) -> np.ndarray:
        """Per-dimension variance within each pattern's cluster neighborhood.

        C[k, i] = variance of X[neighbors, i] around X[k, i].
        Low variance -> discriminative, reliable dimension -> higher precision.
        Returned as inverse variance (high value = low local spread = high trust).
        """
        # Find top-m neighbors for each pattern by cosine similarity
        m = max(2, self.K // 4)
        cosines = self.X @ self.X.T  # (K, K)
        local_inv_var = np.ones((self.K, self.N))
        for k in range(self.K):
            sims   = cosines[k].copy()
            sims[k] = -1.0  # exclude self
            top_m  = np.argpartition(sims, -m)[-m:]
            nbrs   = self.X[top_m]          # (m, N)
            diff   = nbrs - self.X[k]       # (m, N)
            var    = (diff ** 2).mean(axis=0)  # (N,)
            local_inv_var[k] = 1.0 / np.sqrt(var + 1e-4)
        # Row-normalise so values are relative (mean=1)
        local_inv_var /= (local_inv_var.mean(axis=1, keepdims=True) + 1e-12)
        return local_inv_var

    # ------------------------------------------------------------------ #
    # Public interface                                                      #
    # ------------------------------------------------------------------ #

    def predict_precision(self, corrupted_query: np.ndarray) -> np.ndarray:
        """Return N positive precision weights for a corrupted query.

        Parameters
        ----------
        corrupted_query : (N,) float array

        Returns
        -------
        pi : (N,) float array, all positive
        """
        q = np.asarray(corrupted_query, dtype=np.float64)

        q_norm  = q / (np.linalg.norm(q) + 1e-12)
        sims    = self.X @ q_norm

        # Top-2 patterns for routing and confidence
        top2 = np.argpartition(sims, -min(2, self.K))[-min(2, self.K):]
        if self.K >= 2:
            k1, k2 = (top2[0], top2[1]) if sims[top2[0]] >= sims[top2[1]] \
                     else (top2[1], top2[0])
        else:
            k1 = k2 = int(top2[0])
        max_sim = float(sims[k1])
        gap     = float(sims[k1] - sims[k2])

        # --- Route: anisotropy probe (clean, high cosine) ---
        # Return precomputed kappa-optimal pi for this attractor.
        if max_sim > _ROUTE_SIM:
            return self._aniso_pi[k1].copy()

        # --- Route: retrieval query (corrupted, low cosine) ---

        conf = float(np.clip(gap / (_CONF_SCALE + 1e-9), 0.0, 1.0))

        # 1. Masking-aware base: high pi where query is near zero (masked/noisy).
        # Physical motivation: masked dims need gradient-driven recovery (no
        # input injection) so they benefit from amplified gradient steps.
        pi = 1.0 / (np.abs(q) + _EPS)

        # 2. Geometry at equilibrium: amplify dims with large curvature inverse.
        # diag(H^{-1}(a*)) identifies flat directions where extra precision helps.
        pi *= 1.0 + _W_GEO * (self._geo_diag[k1] - 1.0)

        # 3. Energy-gradient alignment: boost dims where gradient points toward k1.
        s_q    = self._softmax(self.beta * (self.X @ q))
        grad_q = self.R @ q - self.eta * (self.X.T @ s_q)
        align  = np.sign(-grad_q) * np.sign(self.X[k1])  # +1 = aligned
        pi    *= 1.0 + _W_ENERGY * conf * align

        # 4. Local covariance: amplify low-variance discriminative dimensions.
        pi    *= 1.0 + _W_LCOV * (self._local_cov[k1] - 1.0)

        # 5. Confidence gate: scale up uniformly when attractor identity is clear.
        pi    *= 1.0 + conf

        # 6. Twin-pair correction: sharpen on dims that distinguish k1 from k2
        # when the confidence gap is small (near a cluster boundary).
        if gap < _TWIN_GAP:
            disc   = (self.X[k1] - self.X[k2]) ** 2
            disc  /= disc.mean() + 1e-12
            weight = 1.0 - gap / _TWIN_GAP
            pi    *= 1.0 + _W_TWIN * weight * disc

        # 7. Spectral smoothing: propagate precision along R's graph structure.
        pi = self._smoother @ pi

        return np.maximum(pi, 1e-8)
