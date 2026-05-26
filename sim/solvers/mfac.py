"""
solvers/mfac.py
---------------
Algorithm 3: Mean-Field Actor-Critic for xApp deployment.

Each cell maintains a local actor (Gaussian policy with mean linear in
features) and a local critic (linear value function on features). The
features are
    phi(q, e, alpha) = [1, q, e, alpha, q*alpha, q*e].

Three time scales:
  Fast       : critic TD update (lr_critic on slot scale)
  Intermediate: actor policy-gradient (lr_actor on slot scale)
  Slow       : rApp mean-field broadcast update (damping kappa, per round)

Stability of the three-time-scale recursion follows from Borkar
two-time-scale stochastic approximation, with critic-approximation
bias bounded by epsilon_Q.

A MF-RL-NAIVE baseline is obtained by setting alpha_features to zero,
disabling the mean-field information channel.
"""
from __future__ import annotations
import numpy as np

from ..config import SimCfg
from ..dynamics import MFGEnv
from ..mean_field import awake_density


def _phi(q: float | np.ndarray, e: float | np.ndarray, alpha: float,
         use_alpha: bool = True) -> np.ndarray:
    """Feature vector (1, q, e, alpha, q*alpha, q*e) of shape (..., 6).

    When use_alpha=False, alpha-coupled entries are zeroed
    (the MF-RL-NAIVE ablation)."""
    q = np.asarray(q); e = np.asarray(e)
    a = float(alpha) if use_alpha else 0.0
    ones = np.ones_like(q)
    return np.stack([ones, q, e, a * ones, q * a, q * e], axis=-1)


class _MFACAgent:
    """A single cell's actor + critic + per-cell rollout buffer."""

    def __init__(self, cfg: SimCfg, use_mf: bool = True, seed: int = 0):
        self.cfg = cfg
        self.use_mf = use_mf
        self.rng = np.random.default_rng(seed)
        # Actor: u = theta . phi, plus Gaussian exploration noise.
        self.theta = np.zeros(6)
        self.log_sigma = -1.0  # exploration std exp(-1) = 0.37
        # Critic: V_phi(s) = w . phi
        self.w = np.zeros(6)

    def policy(self, q: float, e: float, alpha: float,
               stochastic: bool = True) -> float:
        feat = _phi(q, e, alpha, self.use_mf)
        mean = float(feat @ self.theta)
        if stochastic:
            mean += float(self.rng.standard_normal()) * np.exp(self.log_sigma)
        return float(np.clip(mean, -self.cfg.energy.u_bar,
                              +self.cfg.energy.u_bar))

    def value(self, q: float, e: float, alpha: float) -> float:
        return float(_phi(q, e, alpha, self.use_mf) @ self.w)

    def td_update(self, q: float, e: float, alpha: float, cost: float,
                  q_next: float, e_next: float, alpha_next: float,
                  lr: float, gamma: float):
        target = cost + gamma * self.value(q_next, e_next, alpha_next)
        feat = _phi(q, e, alpha, self.use_mf)
        td_err = target - float(feat @ self.w)
        # Clip TD error to stabilize the linear critic on large rewards
        td_err_c = float(np.clip(td_err, -100.0, 100.0))
        self.w += lr * td_err_c * feat
        # Mild L2 shrinkage to keep weights bounded
        self.w *= (1.0 - 1e-4)
        return td_err_c

    def actor_update(self, q: float, e: float, alpha: float,
                     u_sample: float, advantage: float, lr: float):
        """REINFORCE-style update under Gaussian policy.

        log pi = -0.5 ((u - mean) / sigma)^2 - log sigma + const
        grad_theta log pi = (u - mean) / sigma^2 * phi
        For COST minimization we step against the advantage direction."""
        feat = _phi(q, e, alpha, self.use_mf)
        mean = float(feat @ self.theta)
        sigma2 = float(np.exp(2 * self.log_sigma))
        # Normalize advantage and clip to prevent blow-up
        adv = float(np.clip(advantage, -20.0, 20.0))
        grad = (u_sample - mean) / max(sigma2, 1e-4) * feat
        # Clip per-update magnitude
        step = lr * adv * grad
        step_norm = np.linalg.norm(step)
        if step_norm > 0.01:
            step = step * (0.01 / step_norm)
        self.theta = self.theta - step
        # Mild L2 shrinkage
        self.theta *= (1.0 - 1e-4)


def solve_mfac(cfg: SimCfg, positions: np.ndarray, seed: int = 0,
                use_mf: bool = True, verbose: bool = False) -> dict:
    """Train K rounds of MFAC for all N cells in parallel; return policy.

    If use_mf=False, agents see alpha=0 always (MF-RL-NAIVE ablation)."""
    sc = cfg.solver
    rng = np.random.default_rng(seed)
    env = MFGEnv(cfg, positions, seed=seed)
    N = env.N
    agents = [_MFACAgent(cfg, use_mf=use_mf, seed=int(rng.integers(0, 2**31)))
              for _ in range(N)]
    history = []
    bar_alpha = 0.5  # broadcast statistic

    for n in range(sc.mfac_K_rounds):
        # Collect rollout
        env.reset()
        traj = []
        for k in range(sc.mfac_rollout_slots):
            obs_alpha = bar_alpha
            us = np.array([agents[i].policy(env.q[i], env.e[i], obs_alpha,
                                              stochastic=True)
                            for i in range(N)])
            q_pre = env.q.copy(); e_pre = env.e.copy()
            info = env.step(us)
            # Per-cell instantaneous cost (queue + power + slew)
            slew = us ** 2
            L_i = (cfg.cost.c_q * env.q
                   + cfg.cost.c_p * info["energy_W"]
                   + cfg.cost.c_s * slew)
            traj.append((q_pre, e_pre, obs_alpha, us, L_i,
                          env.q.copy(), env.e.copy(), info["alpha"]))
        # Update each agent's critic + actor on this rollout
        for k, (q_pre, e_pre, obs_a, us, L_i, q_next, e_next, a_next) in enumerate(traj):
            for i in range(N):
                td_err = agents[i].td_update(q_pre[i], e_pre[i], obs_a,
                                              float(L_i[i]),
                                              q_next[i], e_next[i], a_next,
                                              sc.mfac_lr_critic, sc.mfac_gamma)
                # Advantage = td_err (one-step advantage)
                agents[i].actor_update(q_pre[i], e_pre[i], obs_a,
                                        us[i], advantage=td_err,
                                        lr=sc.mfac_lr_actor)
        # rApp damped mean-field update
        new_alpha = float(np.mean([traj_k[7] for traj_k in traj]))
        bar_alpha = (1 - sc.mfac_kappa) * bar_alpha + sc.mfac_kappa * new_alpha
        mean_cost = float(np.mean([np.sum(t[4]) for t in traj]))
        history.append({"round": n, "bar_alpha": bar_alpha,
                         "mean_cost_per_slot": mean_cost})
        if verbose and n % 10 == 0:
            print(f"  MFAC round {n:3d}: bar_alpha={bar_alpha:.3f} "
                  f"cost/slot={mean_cost:.1f}")
    return {"agents": agents, "bar_alpha": bar_alpha, "history": history}


# ---------------------------------------------------------------------------
class MFACController:
    """Online controller using a population of trained MFAC agents."""

    name = "MFG-MFAC"

    def __init__(self, cfg: SimCfg, N: int, agents: list,
                 bar_alpha: float = 0.5, name: str | None = None):
        self.cfg = cfg
        self.N = N
        self.agents = agents
        self.bar_alpha = bar_alpha
        if name is not None:
            self.name = name

    def reset(self, *args, **kwargs):
        pass

    def act(self, q: np.ndarray, e: np.ndarray, alpha: float,
            t_s: float = 0.0) -> np.ndarray:
        # Use the broadcast estimate at deployment time (or current alpha)
        a = alpha
        return np.array([self.agents[i].policy(q[i], e[i], a,
                                                 stochastic=False)
                          for i in range(self.N)])


class SafeMFACController(MFACController):
    """MFAC with a Foster-Lyapunov safety shield.

    The learned actor is free to put cells to sleep, but whenever a
    cell's backlog exceeds the stability threshold q_safe the control
    is overridden to drive the cell awake at the maximum slew rate.
    This enforces the drift condition of Assumption (queue stability):
    outside the compact set {q <= q_safe} the controlled service rate
    is pushed above the arrival rate, so the backlog cannot grow without
    bound. The shield is a control-barrier projection of the
    unconstrained MFAC policy onto the safe set, decoupled from training,
    and is exactly what the deployed xApp would carry to make the
    learned policy safe.
    """

    name = "MFG-MFAC-SAFE"

    def __init__(self, cfg: SimCfg, N: int, agents: list,
                 bar_alpha: float = 0.5, q_safe: float | None = None,
                 name: str | None = None):
        super().__init__(cfg, N, agents, bar_alpha, name)
        # Default threshold ties to the independent-threshold wake level.
        self.q_safe = (cfg.indep_th.q_th_high_pdu if q_safe is None
                       else q_safe)

    def act(self, q: np.ndarray, e: np.ndarray, alpha: float,
            t_s: float = 0.0) -> np.ndarray:
        u = super().act(q, e, alpha, t_s=t_s)
        # Safety override: force wake-up where backlog exceeds q_safe.
        unsafe = q > self.q_safe
        u = np.where(unsafe, self.cfg.energy.u_bar, u)
        return u
