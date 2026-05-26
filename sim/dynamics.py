"""
dynamics.py
-----------
Continuous-time stochastic dynamics of the N-cell cluster, integrated
by reflected Euler-Maruyama. Matches eqs. (queue_dyn) and (energy_dyn)
in main.tex:

  dq_i = [lambda_i(m_t) - mu(q_i, e_i, u_i, m_t)] dt
         + sigma_q sqrt(lambda_i + mu) dW_i^q
         reflected at q_i = 0

  de_i = u_i dt + sigma_e dW_i^e
         reflected at e_i in {0, 1}

The control u_i is supplied by an external policy. The mean-field
statistic alpha(t) is computed at every slot from the per-cell e values
(empirical mean field, exact in the symmetric MFG limit).
"""
from __future__ import annotations
import numpy as np

from .config import SimCfg
from .mean_field import (
    awake_density, diurnal_modulation,
    lambda_load_shift, service_rate,
)


class MFGEnv:
    """N-cell MFG simulator with reflected SDE dynamics.

    State per cell: (q_i, e_i) in R_+ x [0, 1].
    Mean field summarized by scalar alpha = mean(e).

    Per-step output of `step(u)`:
        info dict with keys
          energy_W   : per-cell power vector (length N), in W (or relative units)
          loss       : per-slot loss vector (length N), summary metric
          q          : queue vector after step
          e          : energy mode vector after step
          alpha      : scalar awake density
          mu         : service rate vector
          lambda_i   : per-cell load-shifted arrival
          t_s        : current simulated time, s
          n_toggles  : count of e crossings of 0.5 during the step (proxy
                       for sleep-state transitions per slot)
    """

    def __init__(self, cfg: SimCfg, positions: np.ndarray, seed: int = 0,
                 arrival_factor: np.ndarray | None = None):
        self.cfg = cfg
        self.N = positions.shape[0]
        self.pos = positions
        self.rng = np.random.default_rng(seed)
        # Per-cell per-slot diurnal multiplier from a real trace (optional).
        # Shape (N, n_slots); used to weight lambda_0 each slot per cell.
        self.arrival_factor = arrival_factor
        # State
        self.q = np.zeros(self.N)
        self.e = np.ones(self.N)        # start all awake
        self.t_s = 0.0
        self.slot_k = 0
        self.n_toggles = 0

    # -----------------------------------------------------------------
    def reset(self, q0: np.ndarray | None = None,
              e0: np.ndarray | None = None) -> dict:
        self.q = np.zeros(self.N) if q0 is None else np.array(q0)
        self.e = np.ones(self.N) if e0 is None else np.array(e0)
        self.t_s = 0.0
        self.slot_k = 0
        self.n_toggles = 0
        return self._observe()

    def _observe(self) -> dict:
        return {
            "q": self.q.copy(),
            "e": self.e.copy(),
            "alpha": awake_density(self.e),
            "t_s": self.t_s,
        }

    # -----------------------------------------------------------------
    def step(self, u: np.ndarray) -> dict:
        """One Euler-Maruyama step of size dt = cfg.time.dt_s.

        u : np.ndarray of length N, sleep/wake control per cell, in
            [-u_bar, +u_bar].
        """
        cfg = self.cfg
        dt = cfg.time.dt_s
        u = np.clip(u, -cfg.energy.u_bar, +cfg.energy.u_bar)
        # Mean field at start of step (broadcast statistic)
        alpha = awake_density(self.e)
        # Per-cell baseline arrival rate. Real-trace data, if provided,
        # supplies a per-cell per-slot multiplier; otherwise we use the
        # homogeneous synthetic diurnal sinusoid.
        if self.arrival_factor is not None:
            k = min(self.slot_k, self.arrival_factor.shape[1] - 1)
            per_cell_factor = self.arrival_factor[:, k]
            base = cfg.arr.lambda0_pdu_per_s * per_cell_factor
        else:
            base = cfg.arr.lambda0_pdu_per_s * diurnal_modulation(self.t_s, cfg)
        lam_i = lambda_load_shift(base, alpha, cfg.chan.epsilon_lambda)
        # Per-cell service rate
        mu_i = service_rate(self.q, self.e, alpha,
                             cfg.chan.mu_max_pdu_per_s, cfg.chan.eta_I)
        # Drifts
        drift_q = lam_i - mu_i
        drift_e = u
        # Diffusions
        sigma_q_eff = cfg.arr.sigma_q * np.sqrt(np.maximum(lam_i + mu_i, 1e-6))
        sigma_e_eff = cfg.energy.sigma_e * np.ones(self.N)
        # Brownian increments
        dW_q = self.rng.standard_normal(self.N) * np.sqrt(dt)
        dW_e = self.rng.standard_normal(self.N) * np.sqrt(dt)
        # Euler-Maruyama
        e_prev = self.e.copy()
        self.q = self.q + drift_q * dt + sigma_q_eff * dW_q
        self.e = self.e + drift_e * dt + sigma_e_eff * dW_e
        # Reflection
        np.maximum(self.q, 0.0, out=self.q)
        np.clip(self.e, 0.0, 1.0, out=self.e)
        # Toggle count: count cells that crossed the 0.5 threshold
        crossings = ((e_prev > 0.5) != (self.e > 0.5)).sum()
        self.n_toggles += int(crossings)
        self.t_s += dt
        self.slot_k += 1
        # Energy and loss accounting
        p_per_cell = self._power(u)              # length N (per-cell W)
        loss = self._loss()                      # length N (per-slot loss)
        return {
            "q": self.q.copy(), "e": self.e.copy(),
            "alpha": awake_density(self.e),
            "mu": mu_i, "lambda_i": lam_i,
            "t_s": self.t_s,
            "energy_W": p_per_cell,
            "loss": loss,
            "n_toggles": int(crossings),
        }

    def _power(self, u: np.ndarray) -> np.ndarray:
        """Per-cell power
            p(e, u) = P_sleep + (P0 - P_sleep) e + P1 e util(q) + P2 u^2,
        a smoothed EARTH-style mapping. At e=0 (deep sleep) p = P_sleep;
        at e=1 with full load p = P0 + P1 + P2 u^2."""
        cfg = self.cfg
        mu_i = service_rate(self.q, self.e, awake_density(self.e),
                             cfg.chan.mu_max_pdu_per_s, cfg.chan.eta_I)
        util = mu_i / cfg.chan.mu_max_pdu_per_s
        return (cfg.energy.P_sleep_W
                 + (cfg.energy.P0_W - cfg.energy.P_sleep_W) * self.e
                 + cfg.energy.P1_W * self.e * util
                 + cfg.energy.P2_W * u ** 2)

    def _loss(self) -> np.ndarray:
        """Per-slot loss summary: backlog overshoot relative to baseline
        target. Used as a service-quality proxy that we time-average."""
        # Simple choice: loss_i = q_i (delay proxy by Little's law)
        return self.q.copy()
