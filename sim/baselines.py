"""
baselines.py
------------
Reference controllers compared against the proposed MFG policy.

Naming matches §IX of main.tex:

  ALWAYS-ON    : every cell awake, no control action
  INDEP-TH     : per-cell hysteresis threshold; sleeps when q below low
                 threshold, wakes when q above high threshold
  CENT-SOC     : centralized social-optimum surrogate; tractable only
                 for moderate N. We implement an aggregate-throughput
                 LP relaxation: at each slot, pick the awake set that
                 minimizes total power subject to sum-rate covering
                 sum-arrival, with hysteresis.
  MF-RL-NAIVE  : vanilla actor-critic without HJB grounding, parameterized
                 over local (q, e, alpha). Implemented in solvers/mfac.py
                 with a flag.
"""
from __future__ import annotations
import numpy as np

from .config import SimCfg


# ---------------------------------------------------------------------------
class AlwaysOnController:
    """All cells permanently awake; u = 0 everywhere."""
    name = "ALWAYS-ON"

    def __init__(self, cfg: SimCfg, N: int):
        self.cfg = cfg
        self.N = N

    def reset(self, *args, **kwargs):
        pass

    def act(self, q: np.ndarray, e: np.ndarray, alpha: float,
            t_s: float = 0.0) -> np.ndarray:
        # Push e toward 1 (controlled by u > 0) until reflected
        return self.cfg.energy.u_bar * np.ones(self.N)


# ---------------------------------------------------------------------------
class IndepThController:
    """Per-cell hysteretic threshold controller."""
    name = "INDEP-TH"

    def __init__(self, cfg: SimCfg, N: int):
        self.cfg = cfg
        self.N = N
        self._target = np.ones(self.N)   # target e in {0, 1}

    def reset(self, *args, **kwargs):
        self._target = np.ones(self.N)

    def act(self, q: np.ndarray, e: np.ndarray, alpha: float,
            t_s: float = 0.0) -> np.ndarray:
        cfg = self.cfg.indep_th
        sleep_now = q < cfg.q_th_low_pdu
        wake_now = q > cfg.q_th_high_pdu
        self._target[sleep_now] = 0.0
        self._target[wake_now] = 1.0
        # Slew toward target
        u = np.sign(self._target - e) * self.cfg.energy.u_bar
        return u


# ---------------------------------------------------------------------------
class CentSocController:
    """Centralized social-optimum surrogate.

    Implements a myopic aggregate-rate LP heuristic that picks an
    awake set minimizing total power subject to the aggregate-service-rate
    constraint sum_i e_i * psi(alpha) * mu_max >= sum_i lambda_i.

    Greedy ordering by q (highest-queue cells are kept awake first)
    yields an O(N log N) approximation that matches the LP optimum when
    arrivals are homogeneous.
    """
    name = "CENT-SOC"

    def __init__(self, cfg: SimCfg, N: int):
        self.cfg = cfg
        self.N = N

    def reset(self, *args, **kwargs):
        pass

    def act(self, q: np.ndarray, e: np.ndarray, alpha: float,
            t_s: float = 0.0) -> np.ndarray:
        cfg = self.cfg
        # Estimate aggregate arrival rate
        from .mean_field import lambda_load_shift, psi_interference
        lam0 = cfg.arr.lambda0_pdu_per_s
        lam_aggr = self.N * lambda_load_shift(lam0, alpha, cfg.chan.epsilon_lambda)
        # Each awake cell delivers approximately psi(alpha) * mu_max
        mu_each = psi_interference(alpha, cfg.chan.eta_I) * cfg.chan.mu_max_pdu_per_s
        n_needed = int(np.ceil(lam_aggr / max(mu_each, 1e-3)))
        n_needed = min(max(n_needed, 1), self.N)
        # Keep the top n_needed cells (ranked by q) awake
        order = np.argsort(-q)   # high q first
        target = np.zeros(self.N)
        target[order[:n_needed]] = 1.0
        u = np.sign(target - e) * self.cfg.energy.u_bar
        return u
