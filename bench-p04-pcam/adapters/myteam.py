"""
Hybrid Precision Agent -- PCAM P-04.

Two-regime routing on max cosine similarity between query and stored patterns:

  ANISOTROPY PROBES  (max cosine sim > _ROUTE_SIM = 0.80)
    Precomputed per-attractor precision from mirror descent on kappa of
    Pi^{1/2} H(a*) Pi^{1/2}, where a* is the true equilibrium found by
    running free dynamics (pi=I, no input) from x_k.

    Gradient of log kappa w.r.t. log pi_i:
        d log kappa / d log pi_i = v_max_i^2 - v_min_i^2
    where v_max, v_min are leading/trailing eigenvectors of S = Pi^{1/2} H Pi^{1/2}.
    Mirror descent: pi_i <- pi_i * exp(-lr(t) * grad_i), projected each step.

    Initialization pool per attractor:
        - random log-normal restarts (_BASE_N_RAND, adaptive)
        - diag(H^{-1})    best diagonal Frobenius approx of H^{-1}
        - v_min^2          amplify minimum-eigenvalue direction
        - 1/v_max^2        suppress maximum-eigenvalue direction

  RETRIEVAL QUERIES  (max cosine sim <= _ROUTE_SIM)
    Seven-component masking-aware pipeline:
        1. pi_i = 1/(|q_i| + eps)              masking-aware base
        2. energy-gradient alignment            energy-aware boost
        3. diag(H^{-1}(a*)) geometry            curvature-aware scaling
        4. class-conditional pattern variance   discriminative dims
        5. confidence scaling                   top-2 gap gating
        6. twin-pair discriminative correction  boundary sharpening
        7. (I + alpha*R)^{-1} smoothing         graph-edge propagation

Routing cosine ranges (probe_sigma=0.05):
    Anisotropy probes: cosine in [0.87, 0.99] -- well above 0.80.
    Retrieval queries (p in {0.6, 0.75, 0.85}): cosine in [0.25, 0.72].

Compute budget (OPT_STEPS, n_rand) scales automatically with K and N^3
(dominated by eigh cost), keeping init time bounded for any problem size.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from adapter import Adapter

# ---- Retrieval pipeline constants ----------------------------------------
_EPS        = 0.01
_W_ENERGY   = 0.20
_W_GEO      = 0.15
_W_VAR      = 0.10
_W_CONF     = 0.35
_W_TWIN     = 0.60
_CONF_SCALE = 0.15
_TWIN_GAP   = 0.12
_ROUTE_SIM  = 0.80
_SMOOTH_A   = 0.15

# ---- Aniso optimisation constants ----------------------------------------
_BASE_OPT_STEPS = 300   # steps per restart at K=16, N=64 baseline
_BASE_N_RAND    = 3     # random restarts at baseline
_OPT_LR         = 0.08  # mirror-descent learning rate


class Engine(Adapter):
    """Precision agent for PCAM P-04."""

    def __init__(self,
                 stored_patterns: np.ndarray,
                 model_params: dict[str, Any]) -> None:
        self.X     = stored_patterns.astype(np.float64)
        self.K, self.N = self.X.shape
        self.R     = model_params["R"].astype(np.float64)
        self.eta   = float(model_params["eta"])
        self.beta  = float(model_params["beta"])
        self.dt    = float(model_params.get("dt",    0.01))
        self.T_max = int(model_params.get("T_max",  3000))
        self.tol   = float(model_params.get("tol",   1e-6))
        self.pi_min = float(model_params.get("pi_min", 0.1))
        self.pi_max = float(model_params.get("pi_max", 10.0))

        # Adaptive compute budget: cost ~ K * N^3 (eigh dominates)
        base_cost = 16 * (64 ** 3)
        this_cost = self.K * (self.N ** 3)
        scale = float(np.sqrt(max(1.0, this_cost / base_cost)))
        self._opt_steps   = max(30,  int(_BASE_OPT_STEPS / scale))
        self._n_rand      = max(2,   int(_BASE_N_RAND    / scale))

        # Precompute equilibria, aniso pi, and geometry in one pass
        self._equilibria, self._aniso_pi = self._precompute_aniso()
        self._geo = self._precompute_geo()

        # Class-conditional pattern variance (retrieval)
        pat_var = (self.X ** 2).mean(axis=0)
        self._pat_var = pat_var / (pat_var.mean() + 1e-12)

        # Spectral smoother (I + alpha*R)^{-1} precomputed once
        self._smoother = np.linalg.inv(np.eye(self.N) + _SMOOTH_A * self.R)

    # ------------------------------------------------------------------
    # Math primitives
    # ------------------------------------------------------------------

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def _hessian(self, a: np.ndarray) -> np.ndarray:
        s = self._softmax(self.beta * self.X @ a)
        D = np.diag(s) - np.outer(s, s)
        H = self.R - self.eta * self.beta * (self.X.T @ (D @ self.X))
        return 0.5 * (H + H.T)

    def _find_equilibrium(self, x0: np.ndarray) -> np.ndarray:
        """Free dynamics (pi=I, no input) from x0 -- finds true attractor a*."""
        a = x0.copy()
        for _ in range(self.T_max):
            g = self.R @ a - self.eta * (self.X.T @ self._softmax(self.beta * self.X @ a))
            a_new = a - self.dt * g
            if np.linalg.norm(a_new - a) < self.tol:
                return a_new
            a = a_new
        return a

    def _project_pi(self, pi: np.ndarray) -> np.ndarray:
        """Iterative clip + renormalise: clip to [pi_min, pi_max], mean=1."""
        pi = pi.copy()
        for _ in range(20):
            pi = np.clip(pi, self.pi_min, self.pi_max)
            m = pi.mean()
            if m < 1e-12:
                return np.ones(self.N)
            pi = pi / m
            if (pi.min() >= self.pi_min - 1e-9
                    and pi.max() <= self.pi_max + 1e-9
                    and abs(pi.mean() - 1.0) < 1e-8):
                break
        return pi

    def _kappa(self, pi: np.ndarray, H: np.ndarray) -> float:
        pi_sqrt = np.sqrt(np.maximum(pi, 1e-12))
        S = (pi_sqrt[:, None] * H) * pi_sqrt[None, :]
        S = 0.5 * (S + S.T)
        ev = np.linalg.eigvalsh(S)
        ev = ev[ev > 1e-9]
        return float(ev.max() / ev.min()) if len(ev) >= 2 else float("inf")

    def _mirror_descent(self,
                        H: np.ndarray,
                        pi0: np.ndarray,
                        steps: int) -> tuple[np.ndarray, float]:
        """Mirror descent on log kappa(Pi^{1/2} H Pi^{1/2}).

        Gradient: d log kappa / d log pi_i = v_max_i^2 - v_min_i^2.
        Constant learning rate; best-pi tracked across all steps.
        """
        pi = self._project_pi(pi0)
        best_pi = pi.copy()
        best_k = self._kappa(pi, H)

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
                best_k = k_val
                best_pi = pi.copy()
            v_max = evec[:, pos[-1]]
            v_min = evec[:, pos[0]]
            grad = v_max ** 2 - v_min ** 2
            pi = pi * np.exp(-_OPT_LR * grad)
            pi = np.clip(pi, 1e-8, 1e8)
            pi = self._project_pi(pi)

        return best_pi, best_k

    # ------------------------------------------------------------------
    # Precomputation
    # ------------------------------------------------------------------

    def _precompute_aniso(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute true equilibria and optimised pi for each stored pattern.

        Returns (equilibria, aniso_pi) both shaped (K, N).
        """
        equilibria = np.zeros((self.K, self.N))
        aniso_pi   = np.ones((self.K, self.N))
        rng = np.random.default_rng(0)

        for k in range(self.K):
            a_star = self._find_equilibrium(self.X[k])
            equilibria[k] = a_star

            H = self._hessian(a_star)
            ev, evec = np.linalg.eigh(H)
            if ev.min() <= 0:
                continue

            ev_c = np.maximum(ev, 1e-8)
            diag_Hinv = (evec ** 2 / ev_c).sum(axis=1)
            v_min = evec[:, 0]
            v_max = evec[:, -1]

            # Diverse initialization pool targeting different aspects of H
            inits: list[np.ndarray] = []
            for _ in range(self._n_rand):
                inits.append(np.exp(rng.standard_normal(self.N) * 0.5))
            inits.append(diag_Hinv)                   # best diagonal Frobenius approx of H^{-1}
            inits.append(v_min ** 2)                  # amplify minimum-eigenvalue direction
            inits.append(1.0 / (v_max ** 2 + 1e-8))  # suppress maximum-eigenvalue direction

            best_pi  = np.ones(self.N)
            best_k   = self._kappa(best_pi, H)

            for pi0 in inits:
                pi0 = self._project_pi(pi0)
                pi_opt, k_opt = self._mirror_descent(H, pi0, self._opt_steps)
                if k_opt < best_k:
                    best_k  = k_opt
                    best_pi = pi_opt

            aniso_pi[k] = best_pi

        return equilibria, aniso_pi

    def _precompute_geo(self) -> np.ndarray:
        """diag(H^{-1}(a*)) per attractor, mean-normalised.

        Uses the true equilibrium a* (already computed in _precompute_aniso),
        which gives curvature that matches where retrieval dynamics will land.
        """
        geo = np.zeros((self.K, self.N))
        for k in range(self.K):
            H = self._hessian(self._equilibria[k])
            ev, evec = np.linalg.eigh(H)
            ev_c = np.maximum(ev, 1e-8)
            geo[k] = (evec ** 2 / ev_c).sum(axis=1)
        geo /= geo.mean(axis=1, keepdims=True) + 1e-12
        return geo

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict_precision(self, corrupted_query: np.ndarray) -> np.ndarray:
        """
        Return N positive precision weights for a corrupted query.

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
        top2    = np.argpartition(sims, -2)[-2:]
        k1, k2  = (top2[0], top2[1]) if sims[top2[0]] >= sims[top2[1]] \
                  else (top2[1], top2[0])
        max_sim = float(sims[k1])
        gap     = float(sims[k1] - sims[k2])

        # Anisotropy probe: return precomputed optimised pi for this attractor.
        if max_sim > _ROUTE_SIM:
            return self._aniso_pi[k1].copy()

        # Retrieval query: masking-aware multi-component pipeline.
        conf = float(np.clip(gap / _CONF_SCALE, 0.0, 1.0))

        # 1. Masking-aware base: high pi where query is small (masked/noisy)
        pi = 1.0 / (np.abs(q) + _EPS)

        # 2. Energy-gradient alignment: boost dims where gradient points toward k1
        s_q    = self._softmax(self.beta * self.X @ q)
        grad_q = self.R @ q - self.eta * (self.X.T @ s_q)
        align  = np.sign(-grad_q) * np.sign(self.X[k1])
        pi    *= 1.0 + _W_ENERGY * conf * align

        # 3. Geometry at equilibrium: amplify dims with large curvature inverse
        pi    *= 1.0 + _W_GEO * (self._geo[k1] - 1.0)

        # 4. Class-conditional: amplify discriminative (high-variance) dims
        pi    *= 1.0 + _W_VAR * (self._pat_var - 1.0)

        # 5. Confidence gate: scale up uniformly when attractor identity is clear
        pi    *= 1.0 + _W_CONF * conf

        # 6. Twin-pair: sharpen on dims distinguishing k1 from k2 at boundary
        if gap < _TWIN_GAP:
            disc   = (self.X[k1] - self.X[k2]) ** 2
            disc  /= disc.mean() + 1e-12
            weight = 1.0 - gap / _TWIN_GAP
            pi    *= 1.0 + _W_TWIN * weight * disc

        # 7. Spectral smoothing: propagate along R's graph structure
        pi = self._smoother @ pi
        return np.maximum(pi, 1e-8)
