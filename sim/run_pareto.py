"""
run_pareto.py
-------------
Pareto sweep over the energy/queue trade-off by varying the relative
weight c_p / c_q. We compare the proposed MFG-FB controller against
two reference controllers (CENT-SOC, INDEP-TH) at each price ratio.

Output: sim/results/pareto.json
"""
from __future__ import annotations
import argparse
import json
import os
import time
import numpy as np

from .config import default_cfg, canonical_seeds
from .topology import make_layout_and_arrivals
from .dynamics import MFGEnv
from .baselines import (AlwaysOnController, IndepThController,
                          CentSocController)
from .solvers.fb_pde import solve_fb_pde, FBPDEController
from .metrics import run_one_episode, aggregate


def evaluate(cfg, pos, fac, ctl, seeds):
    rows = []
    for s in seeds:
        env = MFGEnv(cfg, pos, seed=s, arrival_factor=fac)
        rows.append(run_one_episode(cfg, env, ctl, cfg.n_slots_episode))
    return aggregate(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N",      type=int, default=100)
    ap.add_argument("--layout", default="milan")
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--ratios", type=float, nargs="+",
                     default=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--out", default="sim/results/pareto.json")
    args = ap.parse_args()

    seeds = canonical_seeds(args.n_seeds)
    out = {"N": args.N, "layout": args.layout, "n_seeds": args.n_seeds,
           "ratios": args.ratios, "by_ratio": {}}
    t0 = time.time()

    for r in args.ratios:
        cfg = default_cfg()        # T_horizon=60, N_t=60 defaults
        cfg.topo.N = args.N; cfg.topo.layout = args.layout
        cfg.cost.c_p = r           # vary c_p; keep c_q = 2.0 default
        cfg.solver.K_max_outer = 30
        pos, fac = make_layout_and_arrivals(cfg)
        print(f"\n  c_p/c_q = {r/cfg.cost.c_q:.3f} (c_p={r}, c_q={cfg.cost.c_q}) ...")

        results = {}
        # MFG-FB at this ratio
        t1 = time.time()
        fb = solve_fb_pde(cfg, alpha_init=0.8)
        ctl_fb = FBPDEController(cfg, args.N, fb["policy_table"],
                                   fb["q_grid"], fb["e_grid"], fb["t_grid"])
        results["MFG-FB"] = evaluate(cfg, pos, fac, ctl_fb, seeds)
        print(f"    MFG-FB ({time.time()-t1:.0f}s): "
              f"P={results['MFG-FB']['mean_power_W']['mean']:.0f}, "
              f"Q={results['MFG-FB']['mean_queue_pdu']['mean']:.1f}")
        # Baselines
        for name, Ctl in [("CENT-SOC", CentSocController),
                           ("INDEP-TH", IndepThController),
                           ("ALWAYS-ON", AlwaysOnController)]:
            results[name] = evaluate(cfg, pos, fac, Ctl(cfg, args.N), seeds)
        out["by_ratio"][f"{r:.4f}"] = results

    out["wall_clock_s"] = time.time() - t0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n[run_pareto] saved {args.out} ({out['wall_clock_s']:.0f}s)")


if __name__ == "__main__":
    main()
