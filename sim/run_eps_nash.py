"""
run_eps_nash.py
---------------
Verify Theorem 3 (epsilon-Nash property):
   ε_N = max_i [J_i^N(tilde u_i, u_{-i}^*) - J_i^N(u^*)]^- <= C * N^{-1/2}.

For each N in args.Ns:
  1. Train MFG-FB once (population strategy u_pop = u*[m*]).
  2. Roll out the N-agent system under u_pop, measure cluster average cost
     J_N(u_pop).
  3. For each of K probe agents, replace its policy with a small set of
     deviation policies (ALWAYS-ON-like and INDEP-TH-like) and measure
     the deviation's per-cell cost reduction Delta_i.
  4. epsilon_emp = max_i Delta_i.

Log-log slope of epsilon_emp vs N should be near -1/2.

Output: sim/results/eps_nash.json
"""
from __future__ import annotations
import argparse
import json
import os
import time
import numpy as np

from .config import default_cfg
from .topology import make_layout_and_arrivals
from .dynamics import MFGEnv
from .solvers.fb_pde import (
    solve_fb_pde, solve_best_response, FBPDEController,
)
from .baselines import AlwaysOnController, IndepThController
from .mean_field import awake_density


def _episode_avg_cluster_cost(cfg, env, controller) -> float:
    env.reset()
    total = 0.0
    for k in range(cfg.n_slots_episode):
        alpha = awake_density(env.e)
        u = controller.act(env.q, env.e, alpha, t_s=env.t_s)
        info = env.step(u)
        slew = (u ** 2).sum()
        L = (cfg.cost.c_q * env.q.sum()
              + cfg.cost.c_p * info["energy_W"].sum()
              + cfg.cost.c_s * slew)
        total += L * cfg.time.dt_s
    return total / env.N           # per-cell time-integrated cost


def _episode_probe_costs(cfg, env, pop_controller, probe_idx: list[int],
                          deviator_controller=None) -> dict:
    """Roll out the all-population system; if deviator_controller is given,
    the probe agents in probe_idx instead play the deviation policy while
    everyone else plays pop. Returns each probe agent's own
    time-integrated cost. Comparing the same agent index across the
    deviator=None and deviator!=None calls (same seed) differences out
    that agent's location heterogeneity, giving the true unilateral
    deviation gain rather than a cross-cell cost difference."""
    env.reset()
    per_probe_cost = {i: 0.0 for i in probe_idx}
    for k in range(cfg.n_slots_episode):
        alpha = awake_density(env.e)
        u = pop_controller.act(env.q, env.e, alpha, t_s=env.t_s)
        if deviator_controller is not None:
            u_dev = deviator_controller.act(env.q, env.e, alpha, t_s=env.t_s)
            u = u.copy()
            for i in probe_idx:
                u[i] = u_dev[i]
        info = env.step(u)
        slew = u ** 2
        L_i = (cfg.cost.c_q * env.q
                + cfg.cost.c_p * info["energy_W"]
                + cfg.cost.c_s * slew)
        for i in probe_idx:
            per_probe_cost[i] += float(L_i[i]) * cfg.time.dt_s
    return per_probe_cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ns",     type=int, nargs="+",
                     default=[50, 100, 200, 400, 800])
    ap.add_argument("--layout", default="milan")
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--n_probes", type=int, default=8)
    ap.add_argument("--out", default="sim/results/eps_nash.json")
    args = ap.parse_args()

    out = {"Ns": args.Ns, "layout": args.layout,
           "n_seeds": args.n_seeds, "n_probes": args.n_probes,
           "by_N": {}}
    t_global = time.time()

    for N in args.Ns:
        print(f"\n=== N = {N} ===")
        cfg = default_cfg()
        cfg.topo.N = N; cfg.topo.layout = args.layout
        pos, fac = make_layout_and_arrivals(cfg)

        # Train MFG-FB once (compile policy)
        t0 = time.time()
        fb = solve_fb_pde(cfg, alpha_init=0.8)
        pop_ctl = FBPDEController(cfg, N, fb["policy_table"],
                                    fb["q_grid"], fb["e_grid"], fb["t_grid"])
        # Single-agent best response to the FROZEN equilibrium field:
        # a refined-grid re-optimization, the tightest unilateral
        # deviation and the proper epsilon-Nash probe (cf. Theorem 3).
        br = solve_best_response(cfg, fb["alpha_traj"], u_grid_n=41)
        br_ctl = FBPDEController(cfg, N, br["policy_table"],
                                  br["q_grid"], br["e_grid"], br["t_grid"])
        # Deviation set: the best response plus two heuristic global
        # switches (which serve as weaker reference deviations).
        deviators = [
            ("BEST-RESP", br_ctl),
            ("ALWAYS-ON", AlwaysOnController(cfg, N)),
            ("INDEP-TH",  IndepThController(cfg, N)),
        ]
        print(f"  trained MFG-FB in {time.time()-t0:.0f}s, "
              f"alpha*={fb['alpha_traj'].mean():.3f}")

        # Measure costs per seed, average eps_emp over seeds
        seed_eps_emps = []        # best-response gap (the proper eps-Nash)
        seed_heur_emps = []       # best gap over heuristic deviations
        seed = 1000 * (N + 7)
        for s in range(args.n_seeds):
            sd = seed + s
            env = MFGEnv(cfg, pos, seed=sd, arrival_factor=fac)
            J_pop = _episode_avg_cluster_cost(cfg, env, pop_ctl)
            # Pick K probe agent indices uniformly
            rng = np.random.default_rng(s)
            probe_idx = list(rng.choice(N, size=min(args.n_probes, N),
                                          replace=False))
            # Baseline: each probe agent's OWN cost when it plays pop.
            base = _episode_probe_costs(
                cfg, MFGEnv(cfg, pos, seed=sd, arrival_factor=fac),
                pop_ctl, probe_idx)
            br_delta = 0.0          # gain of the single-agent best response
            heur_delta = 0.0        # best gain over heuristic deviations
            for dname, dctl in deviators:
                dev = _episode_probe_costs(
                    cfg, MFGEnv(cfg, pos, seed=sd, arrival_factor=fac),
                    pop_ctl, probe_idx, deviator_controller=dctl)
                # Per-agent, location-differenced gain: i's pop cost minus
                # i's deviation cost (same i, same seed).
                gain = max(base[i] - dev[i] for i in probe_idx)
                gain = max(gain, 0.0)  # >0 means the deviation lowers cost
                if dname == "BEST-RESP":
                    br_delta = gain
                else:
                    heur_delta = max(heur_delta, gain)
            # Normalize by the mean per-agent baseline cost over probes.
            base_mean = float(np.mean(list(base.values())))
            seed_eps_emps.append(br_delta)
            seed_heur_emps.append(heur_delta)
            print(f"  seed {s}: base/agent={base_mean:.1f}, "
                  f"best-response gap={br_delta:.3f} "
                  f"({100*br_delta/max(base_mean,1e-9):.2f}%), "
                  f"heuristic gap={heur_delta:.2f}")

        out["by_N"][str(N)] = {
            "alpha_star_mfe": float(fb["alpha_traj"].mean()),
            "n_iters_fb": fb["n_iters"],
            "eps_emp_per_seed": seed_eps_emps,
            "eps_emp_mean": float(np.mean(seed_eps_emps)),
            "eps_emp_max":  float(np.max(seed_eps_emps)),
            "heur_emp_mean": float(np.mean(seed_heur_emps)),
            "heur_emp_max":  float(np.max(seed_heur_emps)),
        }

    out["wall_clock_s"] = time.time() - t_global
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n[run_eps_nash] saved {args.out} ({out['wall_clock_s']:.0f}s)")


if __name__ == "__main__":
    main()
