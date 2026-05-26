"""
solvers/graphon.py
------------------
Multi-population (block-graphon) mean-field game solver.

Generalizes the scalar FB-PDE solver: the N cells are partitioned into
B blocks, each block b carries its own baseline arrival lambda_0^(b),
and a block-a cell couples to the LOCAL field

    abar_a(t) = sum_b W[a,b] alpha_b(t)

through a row-stochastic block-interaction kernel W. The graphon MFE is
a PROFILE (alpha_1^*(t), ..., alpha_B^*(t)), one awake-density trajectory
per block. With B=1 (or a uniform W) this reduces exactly to the scalar
FB-PDE solver.

Each block's best response is one scalar HJB backward + FP forward sweep
(reusing fb_pde._hamiltonian_min and fb_pde._fp_forward_step) at the
block's local field abar_a and its own lambda_0^(b).
"""
from __future__ import annotations
import copy
import numpy as np

from ..config import SimCfg
from .fb_pde import _hamiltonian_min, _fp_forward_step, _bilinear


def solve_graphon_mfg(cfg: SimCfg, block_lambda0: np.ndarray,
                       W: np.ndarray, alpha_init: float = 0.7,
                       verbose: bool = False) -> dict:
    """Solve the block-graphon MFG.

    Args:
        block_lambda0 : (B,) per-block baseline arrival rate (PDU/s)
        W             : (B,B) row-stochastic block-interaction kernel
        alpha_init    : uniform initial awake density per block

    Returns dict with:
        alpha_blocks  : (B, n_t) MFE awake-density profile per block
        policy_tables : (B, n_t, N_q, N_e) per-block optimal control
        q_grid, e_grid, t_grid
        n_iters, delta_final
    """
    sc = cfg.solver
    B = len(block_lambda0)
    q_grid = np.linspace(0.0, sc.q_max_pdu, sc.N_q)
    e_grid = np.linspace(0.0, 1.0, sc.N_e)
    n_t = sc.N_t
    t_grid = np.linspace(0.0, cfg.time.T_horizon_s, n_t)
    dt_grid = cfg.time.T_horizon_s / max(n_t - 1, 1)

    alpha_blocks = np.full((B, n_t), alpha_init)
    history = [alpha_blocks.copy()]
    delta = np.inf
    n_iters = 0

    # Per-block cfg copies carrying the block's lambda_0
    cfg_b = []
    for b in range(B):
        c = copy.deepcopy(cfg)
        c.arr.lambda0_pdu_per_s = float(block_lambda0[b])
        cfg_b.append(c)

    for outer in range(sc.K_max_outer):
        # Local fields abar[a,t] = sum_b W[a,b] alpha_b(t)
        abar = W @ alpha_blocks                        # (B, n_t)
        policy_tables = np.zeros((B, n_t, sc.N_q, sc.N_e))
        alpha_new = np.zeros((B, n_t))

        for a in range(B):
            # ---- HJB backward for block a at its local field ----
            V = np.zeros((sc.N_q, sc.N_e))
            u_traj = np.zeros((n_t, sc.N_q, sc.N_e))
            for k in range(n_t - 1, -1, -1):
                V, u_opt = _hamiltonian_min(V, q_grid, e_grid,
                                             float(abar[a, k]),
                                             cfg_b[a], t_grid[k], dt_grid)
                u_traj[k] = u_opt
            policy_tables[a] = u_traj
            # ---- FP forward for block a ----
            m = np.zeros((sc.N_q, sc.N_e)); m[0, -1] = 1.0
            for k in range(n_t):
                alpha_new[a, k] = float((e_grid[None, :] * m).sum())
                m = _fp_forward_step(m, u_traj[k], q_grid, e_grid,
                                      float(abar[a, k]), cfg_b[a],
                                      t_grid[k], dt_grid)

        delta = float(np.max(np.abs(alpha_new - alpha_blocks)))
        alpha_blocks = (1 - sc.kappa) * alpha_blocks + sc.kappa * alpha_new
        history.append(alpha_blocks.copy())
        n_iters = outer + 1
        if verbose and outer % 5 == 0:
            print(f"  iter {outer:3d}: delta={delta:.4f}  "
                  f"alpha_blocks={np.round(alpha_blocks.mean(axis=1), 3)}")
        if delta < sc.tol_outer:
            break

    return {
        "alpha_blocks": alpha_blocks,
        "policy_tables": policy_tables,
        "q_grid": q_grid, "e_grid": e_grid, "t_grid": t_grid,
        "W": W, "block_lambda0": block_lambda0,
        "n_iters": n_iters, "delta_final": delta,
        "history": history,
    }


class GraphonController:
    """Deploys per-block policy tables: cell i in block b(i) uses that
    block's feedback u*_b(t, q_i, e_i)."""

    name = "MFG-Graphon"

    def __init__(self, cfg: SimCfg, N: int, labels: np.ndarray,
                 policy_tables: np.ndarray, q_grid, e_grid, t_grid):
        self.cfg = cfg
        self.N = N
        self.labels = labels
        self.policy = policy_tables       # (B, n_t, N_q, N_e)
        self.q_grid = q_grid
        self.e_grid = e_grid
        self.t_grid = t_grid

    def reset(self, *args, **kwargs):
        pass

    def act(self, q: np.ndarray, e: np.ndarray, alpha: float,
            t_s: float = 0.0) -> np.ndarray:
        ti = int(min(np.searchsorted(self.t_grid, t_s),
                     len(self.t_grid) - 1))
        u_out = np.zeros(self.N)
        for b in range(self.policy.shape[0]):
            idx = np.where(self.labels == b)[0]
            if len(idx) == 0:
                continue
            u_out[idx] = _bilinear(self.policy[b, ti], self.q_grid,
                                    self.e_grid, q[idx], e[idx])
        return np.clip(u_out, -self.cfg.energy.u_bar, +self.cfg.energy.u_bar)
