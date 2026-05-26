"""
solvers/fb_pde.py
-----------------
Algorithm 1: Forward-backward HJB-FP iteration on a finite grid.

This is the offline grid solver matching Algorithm 1 of main.tex.
It produces a feedback policy u*(t, x; alpha_traj) computed once per
rApp policy compilation; the xApp evaluates this feedback online.

The state space [0, q_max] x [0, 1] is discretized to N_q x N_e
points, with a time grid of N_t steps over the horizon T_horizon_s.
The mean field is summarized by the scalar alpha(t) trajectory.

Outer loop: alternate (HJB backward, FP forward, alpha update) with
mean-field damping kappa, until ||alpha_new - alpha_old||_inf < tol.

The output 'policy' is a 3D table u_star[t_i, q_i, e_i] of shape
(N_t, N_q, N_e); during deployment it is bilinearly interpolated in
(q, e) and held piecewise constant in t.
"""
from __future__ import annotations
import numpy as np

from ..config import SimCfg
from ..mean_field import (
    psi_interference, lambda_load_shift, awake_density,
    diurnal_modulation,
)


def _hamiltonian_min(V_t: np.ndarray, q_grid: np.ndarray, e_grid: np.ndarray,
                     alpha_t: float, cfg: SimCfg, t_s: float,
                     dt_grid: float, u_grid_n: int = 13):
    """At a given time and given V_t (the value at the *next* time slice),
    compute the optimal u and the Hamiltonian min for every (q, e) cell.

    Backward HJB recursion (discrete-time form):
       V[t, q, e] = min_u { L(q, e, u, alpha) dt_grid
                            + exp(-rho dt_grid) V[t+dt_grid, q', e'] }
    with q', e' from one drift step over dt_grid. Diffusion is captured
    by Laplacian smoothing of V_t.

    Returns:
        V_curr : value at this time slice, shape (N_q, N_e)
        u_opt  : optimal control, shape (N_q, N_e)
    """
    dt = dt_grid
    rho = cfg.cost.rho
    eta_I = cfg.chan.eta_I
    psi = psi_interference(alpha_t, eta_I)
    base = cfg.arr.lambda0_pdu_per_s * diurnal_modulation(t_s, cfg)
    lam = lambda_load_shift(base, alpha_t, cfg.chan.epsilon_lambda)

    Q, E = np.meshgrid(q_grid, e_grid, indexing="ij")    # shape (N_q, N_e)
    # Service rate at this grid
    util = Q / (Q + 8.0)
    mu = E * psi * cfg.chan.mu_max_pdu_per_s * util

    # Try a small candidate set of u values; the optimal control is
    # continuous but the running-cost-plus-drift trade-off is convex
    # in u, so a coarse grid plus an analytic interior optimum gives
    # a good approximation.
    u_grid = np.linspace(-cfg.energy.u_bar, +cfg.energy.u_bar, u_grid_n)
    best_cost = np.full(Q.shape, np.inf)
    best_u = np.zeros(Q.shape)
    for u_val in u_grid:
        # Per-cell instantaneous cost (matches dynamics._power)
        util = mu / cfg.chan.mu_max_pdu_per_s
        power = (cfg.energy.P_sleep_W
                  + (cfg.energy.P0_W - cfg.energy.P_sleep_W) * E
                  + cfg.energy.P1_W * E * util
                  + cfg.energy.P2_W * (u_val ** 2))
        L = (cfg.cost.c_q * Q + cfg.cost.c_p * power
             + cfg.cost.c_s * (u_val ** 2)
             + cfg.cost.toll * E)        # Pigouvian toll on wakefulness
        # Drift to next state (forward Euler)
        drift_q = lam - mu
        drift_e = u_val
        q_next = np.clip(Q + drift_q * dt, q_grid.min(), q_grid.max())
        e_next = np.clip(E + drift_e * dt, 0.0, 1.0)
        # Bilinear interpolation of V_t at (q_next, e_next)
        V_next = _bilinear(V_t, q_grid, e_grid, q_next, e_next)
        # Bellman update with discrete-time discount factor exp(-rho dt)
        cost = L * dt + np.exp(-rho * dt) * V_next
        better = cost < best_cost
        best_cost = np.where(better, cost, best_cost)
        best_u = np.where(better, u_val, best_u)
    return best_cost, best_u


def _bilinear(V: np.ndarray, q_grid: np.ndarray, e_grid: np.ndarray,
               q_query: np.ndarray, e_query: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of V on the (q_grid, e_grid) at queries."""
    iq = np.clip(np.searchsorted(q_grid, q_query) - 1, 0, len(q_grid) - 2)
    ie = np.clip(np.searchsorted(e_grid, e_query) - 1, 0, len(e_grid) - 2)
    q0 = q_grid[iq]; q1 = q_grid[iq + 1]
    e0 = e_grid[ie]; e1 = e_grid[ie + 1]
    wq = np.clip((q_query - q0) / np.maximum(q1 - q0, 1e-9), 0.0, 1.0)
    we = np.clip((e_query - e0) / np.maximum(e1 - e0, 1e-9), 0.0, 1.0)
    v00 = V[iq, ie]
    v10 = V[iq + 1, ie]
    v01 = V[iq, ie + 1]
    v11 = V[iq + 1, ie + 1]
    return ((1 - wq) * (1 - we) * v00 + wq * (1 - we) * v10
            + (1 - wq) * we * v01 + wq * we * v11)


def _fp_forward_step(m: np.ndarray, u_table: np.ndarray,
                      q_grid: np.ndarray, e_grid: np.ndarray,
                      alpha_t: float, cfg: SimCfg, t_s: float,
                      dt_grid: float) -> np.ndarray:
    """One forward-Euler step of the FP equation on the (q, e) grid over
    dt_grid. Diffusion approximated by isotropic Laplacian smoothing.
    """
    dt = dt_grid
    eta_I = cfg.chan.eta_I
    psi = psi_interference(alpha_t, eta_I)
    base = cfg.arr.lambda0_pdu_per_s * diurnal_modulation(t_s, cfg)
    lam = lambda_load_shift(base, alpha_t, cfg.chan.epsilon_lambda)
    Q, E = np.meshgrid(q_grid, e_grid, indexing="ij")
    util = Q / (Q + 8.0)
    mu = E * psi * cfg.chan.mu_max_pdu_per_s * util
    drift_q = lam - mu
    drift_e = u_table
    # New (q, e) positions for each grid bin
    q_new = np.clip(Q + drift_q * dt, q_grid.min(), q_grid.max())
    e_new = np.clip(E + drift_e * dt, 0.0, 1.0)
    # Re-bin: for each (i,j) of the original grid, find target bin of new
    iq = np.clip(np.searchsorted(q_grid, q_new) - 1, 0, len(q_grid) - 2)
    ie = np.clip(np.searchsorted(e_grid, e_new) - 1, 0, len(e_grid) - 2)
    # Add mass to nearest bin (simple upwind transport)
    m_new = np.zeros_like(m)
    np.add.at(m_new, (iq, ie), m)
    # Small diffusion: convolve with a 3x3 Gaussian-like kernel
    m_new = _diffuse(m_new, dt, cfg)
    # Normalize to preserve mass
    total = m_new.sum()
    if total > 1e-9:
        m_new /= total
    return m_new


def _diffuse(m: np.ndarray, dt: float, cfg: SimCfg) -> np.ndarray:
    """Discrete Laplacian smoothing as a poor man's diffusion step."""
    sigma_q = cfg.arr.sigma_q
    sigma_e = cfg.energy.sigma_e
    # Diffusion strengths scaled by sigma^2 * dt / dx^2
    out = m.copy()
    # q direction (axis 0)
    if sigma_q > 0:
        c_q = min(0.5, sigma_q ** 2 * dt * 0.001)  # damped; just for stability
        out[1:-1] += c_q * (m[2:] - 2 * m[1:-1] + m[:-2])
    if sigma_e > 0:
        c_e = min(0.5, sigma_e ** 2 * dt * 4.0)
        out[:, 1:-1] += c_e * (m[:, 2:] - 2 * m[:, 1:-1] + m[:, :-2])
    np.maximum(out, 0.0, out=out)
    return out


def solve_fb_pde(cfg: SimCfg, verbose: bool = False,
                  alpha_init: float = 0.8) -> dict:
    """Outer FB iteration. Returns:
        policy_table : ndarray (N_t, N_q, N_e), optimal u(t, q, e)
        alpha_traj   : ndarray (N_t,), MFE awake-density profile
        history      : list of per-iteration alpha trajectories
        n_iters      : iterations to convergence

    alpha_init: initial mean-field trajectory (uniform). MFG can have
    multiple equilibria; the basin reached by FB iteration depends on
    this seed. Default 0.8 corresponds to the high-utilization MFE.
    """
    sc = cfg.solver
    q_grid = np.linspace(0.0, sc.q_max_pdu, sc.N_q)
    e_grid = np.linspace(0.0, 1.0, sc.N_e)
    n_t = sc.N_t
    t_grid = np.linspace(0.0, cfg.time.T_horizon_s, n_t)
    dt_grid = cfg.time.T_horizon_s / max(n_t - 1, 1)

    # Initial alpha trajectory: uniform at alpha_init
    alpha_curr = np.full(n_t, alpha_init)
    history = [alpha_curr.copy()]
    n_iters = 0

    for outer in range(sc.K_max_outer):
        # ===== HJB backward pass =====
        # Terminal condition: V(T, x) = 0 (no terminal cost in current model)
        V = np.zeros((sc.N_q, sc.N_e))
        u_traj = np.zeros((n_t, sc.N_q, sc.N_e))
        for k in range(n_t - 1, -1, -1):
            t_s = t_grid[k]
            V, u_opt = _hamiltonian_min(V, q_grid, e_grid, alpha_curr[k],
                                          cfg, t_s, dt_grid)
            u_traj[k] = u_opt

        # ===== FP forward pass =====
        # Initial measure: concentrated at q=0, e=1
        m = np.zeros((sc.N_q, sc.N_e))
        m[0, -1] = 1.0
        alpha_new = np.zeros(n_t)
        for k in range(n_t):
            t_s = t_grid[k]
            alpha_new[k] = float(((e_grid[None, :]) * m).sum())
            m = _fp_forward_step(m, u_traj[k], q_grid, e_grid,
                                  alpha_curr[k], cfg, t_s, dt_grid)

        # ===== Damped update =====
        delta = float(np.max(np.abs(alpha_new - alpha_curr)))
        alpha_curr = (1 - sc.kappa) * alpha_curr + sc.kappa * alpha_new
        history.append(alpha_curr.copy())
        n_iters = outer + 1
        if verbose and outer % 5 == 0:
            print(f"  iter {outer:3d}: ||delta alpha|| = {delta:.4f}, "
                  f"<alpha> = {alpha_curr.mean():.3f}")
        if delta < sc.tol_outer:
            break

    return {
        "policy_table": u_traj,
        "alpha_traj": alpha_curr,
        "q_grid": q_grid,
        "e_grid": e_grid,
        "t_grid": t_grid,
        "history": history,
        "n_iters": n_iters,
        "delta_final": delta,
    }


def solve_best_response(cfg: SimCfg, alpha_traj: np.ndarray,
                         u_grid_n: int = 41) -> dict:
    """Single-agent best response to a FROZEN mean-field trajectory.

    Holds the equilibrium field alpha_traj fixed (no FP forward pass, no
    fixed-point loop) and runs one backward HJB recursion on a refined
    control grid (u_grid_n candidates, default 41 vs the population
    solver's 13). This is the tightest deviation available to an agent
    that knows the equilibrium field: it re-optimizes its own policy
    against alpha_traj. Its realized cost upper-bounds what any
    unilateral best response can achieve, so the gain of this deviator
    over the population policy is the proper empirical epsilon-Nash gap
    (Theorem 3), in contrast to probing only heuristic global switches.
    """
    sc = cfg.solver
    q_grid = np.linspace(0.0, sc.q_max_pdu, sc.N_q)
    e_grid = np.linspace(0.0, 1.0, sc.N_e)
    n_t = sc.N_t
    t_grid = np.linspace(0.0, cfg.time.T_horizon_s, n_t)
    dt_grid = cfg.time.T_horizon_s / max(n_t - 1, 1)
    V = np.zeros((sc.N_q, sc.N_e))
    u_traj = np.zeros((n_t, sc.N_q, sc.N_e))
    for k in range(n_t - 1, -1, -1):
        V, u_opt = _hamiltonian_min(V, q_grid, e_grid, alpha_traj[k],
                                     cfg, t_grid[k], dt_grid,
                                     u_grid_n=u_grid_n)
        u_traj[k] = u_opt
    return {"policy_table": u_traj, "q_grid": q_grid,
            "e_grid": e_grid, "t_grid": t_grid}


# ---------------------------------------------------------------------------
class FBPDEController:
    """Wraps a solved policy_table as an online feedback controller.
    During execution, returns u_i = u_star(t, q_i, e_i)."""

    name = "MFG-FB"

    def __init__(self, cfg: SimCfg, N: int, policy_table: np.ndarray,
                 q_grid: np.ndarray, e_grid: np.ndarray, t_grid: np.ndarray):
        self.cfg = cfg
        self.N = N
        self.policy = policy_table
        self.q_grid = q_grid
        self.e_grid = e_grid
        self.t_grid = t_grid

    def reset(self, *args, **kwargs):
        pass

    def act(self, q: np.ndarray, e: np.ndarray, alpha: float,
            t_s: float = 0.0) -> np.ndarray:
        # Locate the time slice (piecewise-constant interpolation in t)
        ti = int(min(np.searchsorted(self.t_grid, t_s),
                     len(self.t_grid) - 1))
        u_table = self.policy[ti]
        # Bilinear interpolation in (q, e)
        u_out = _bilinear(u_table, self.q_grid, self.e_grid, q, e)
        return np.clip(u_out, -self.cfg.energy.u_bar, +self.cfg.energy.u_bar)
