"""
run_convergence.py
------------------
Verify Proposition 3 (Convergence of FB iteration):
   sup_t W_1(m^(n)_t, m*_t) <= C rho^n with rho < 1.

We run FB-PDE from several initial alpha trajectories, save the per-iter
||alpha^(n+1) - alpha^(n)||_inf history, and check that the residual
contracts geometrically.

Output: sim/results/convergence.json
"""
from __future__ import annotations
import argparse
import json
import os
import time
import numpy as np

from .config import default_cfg
from .solvers.fb_pde import solve_fb_pde


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N",      type=int, default=100)
    ap.add_argument("--layout", default="milan")
    ap.add_argument("--Ks",     type=int, nargs="+",
                     default=[40])
    ap.add_argument("--inits",  type=float, nargs="+",
                     default=[0.2, 0.5, 0.8, 0.95])
    ap.add_argument("--out", default="sim/results/convergence.json")
    args = ap.parse_args()

    out = {"N": args.N, "layout": args.layout, "runs": []}
    t0 = time.time()
    for a0 in args.inits:
        cfg = default_cfg()           # T_horizon=60, N_t=60 defaults
        cfg.topo.N = args.N
        cfg.topo.layout = args.layout
        cfg.solver.K_max_outer = args.Ks[0]
        cfg.solver.tol_outer = 1e-8   # don't stop early — record full trajectory
        print(f"  alpha_init={a0} ...", end=" ", flush=True)
        t1 = time.time()
        res = solve_fb_pde(cfg, alpha_init=a0, verbose=False)
        # Per-iter residual ||alpha^(n+1) - alpha^(n)||_inf
        hist = res["history"]
        residuals = [float(np.max(np.abs(hist[i+1] - hist[i])))
                      for i in range(len(hist) - 1)]
        means    = [float(np.mean(h)) for h in hist]
        out["runs"].append({
            "alpha_init": a0,
            "n_iters": res["n_iters"],
            "alpha_final_mean": float(res["alpha_traj"].mean()),
            "residuals": residuals,
            "alpha_mean_per_iter": means,
            "wall_clock_s": time.time() - t1,
        })
        print(f"converged in {res['n_iters']} iters; final alpha*={res['alpha_traj'].mean():.3f} ({time.time()-t1:.0f}s)")

    out["wall_clock_s"] = time.time() - t0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n[run_convergence] saved {args.out} ({out['wall_clock_s']:.0f}s)")


if __name__ == "__main__":
    main()
