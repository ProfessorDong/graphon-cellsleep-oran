"""
metrics.py
----------
Episode-level metrics matching §IX's metrics list:
  energy_per_area_W_per_km2
  energy_per_bit_J_per_bit
  mean_queue_pdu
  p95_queue_pdu
  mean_alpha
  toggles_per_min_per_cell
"""
from __future__ import annotations
import numpy as np

from .config import SimCfg


def summarize_episode(energy_W: np.ndarray, q_traj: np.ndarray,
                       alpha_traj: np.ndarray, e_traj: np.ndarray,
                       n_toggles: int, served_pdu: float,
                       area_km2: float, dt_s: float) -> dict:
    """All inputs are time series of shape (n_slots,) except q/e/alpha
    which are (n_slots, N) for per-cell or (n_slots,) scalar."""
    # energy_W: total over the cluster, per slot
    mean_power_W = float(np.mean(energy_W))                  # cluster average power
    energy_J = float(np.sum(energy_W) * dt_s)                # total energy
    energy_per_area = mean_power_W / max(area_km2, 1e-6)     # W/km^2
    # Per-bit energy: energy_J / total served (PDU); PDU stand-in for "bit"
    served = max(served_pdu, 1e-6)
    energy_per_bit = energy_J / served                       # J/PDU
    # Queue stats: average mean across cells then time
    if q_traj.ndim == 2:
        mean_q = float(np.mean(q_traj))
        p95_q = float(np.quantile(q_traj.flatten(), 0.95))
    else:
        mean_q = float(np.mean(q_traj))
        p95_q = float(np.quantile(q_traj, 0.95))
    # Awake density
    mean_alpha = float(np.mean(alpha_traj))
    # Toggles/min/cell
    duration_min = (q_traj.shape[0] * dt_s) / 60.0
    N_cells = q_traj.shape[1] if q_traj.ndim == 2 else 1
    toggles_per_min_per_cell = n_toggles / max(duration_min * N_cells, 1e-6)
    return {
        "energy_per_area_W_per_km2": energy_per_area,
        "energy_per_bit_J_per_pdu": energy_per_bit,
        "mean_power_W": mean_power_W,
        "mean_queue_pdu": mean_q,
        "p95_queue_pdu": p95_q,
        "mean_alpha": mean_alpha,
        "toggles_per_min_per_cell": toggles_per_min_per_cell,
        "total_energy_J": energy_J,
        "total_served_pdu": served,
    }


def aggregate(rows: list[dict]) -> dict:
    """Aggregate per-seed metric rows by mean and 95% bootstrap CI."""
    if not rows:
        return {}
    keys = rows[0].keys()
    out = {}
    for k in keys:
        if not isinstance(rows[0][k], (int, float)):
            continue
        vals = np.array([r[k] for r in rows])
        rng = np.random.default_rng(0)
        boot = np.array([np.mean(rng.choice(vals, size=len(vals), replace=True))
                          for _ in range(1000)])
        out[k] = {
            "mean": float(np.mean(vals)),
            "lo": float(np.quantile(boot, 0.025)),
            "hi": float(np.quantile(boot, 0.975)),
            "_per_seed": vals.tolist(),
        }
    out["_n_seeds"] = len(rows)
    return out


def run_one_episode(cfg: SimCfg, env, controller, n_slots: int) -> dict:
    """Single-episode rollout: returns aggregated metrics."""
    import numpy as np
    env.reset()
    controller.reset() if hasattr(controller, "reset") else None
    energy_W_per_slot = np.zeros(n_slots)
    served_pdu = 0.0
    q_traj = np.zeros((n_slots, env.N))
    e_traj = np.zeros((n_slots, env.N))
    alpha_traj = np.zeros(n_slots)
    for k in range(n_slots):
        alpha = float(np.mean(env.e))
        u = controller.act(env.q, env.e, alpha, t_s=env.t_s)
        info = env.step(u)
        energy_W_per_slot[k] = info["energy_W"].sum()
        served_pdu += float(np.sum(info["mu"])) * cfg.time.dt_s
        q_traj[k] = env.q
        e_traj[k] = env.e
        alpha_traj[k] = info["alpha"]
    return summarize_episode(energy_W_per_slot, q_traj, alpha_traj, e_traj,
                              env.n_toggles, served_pdu,
                              cfg.topo.area_km2, cfg.time.dt_s)
