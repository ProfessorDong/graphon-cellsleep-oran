"""
run_comparative.py
------------------
Verify Lemma 2 (Comparative statics):
   d alpha*/d c_p < 0
   d alpha*/d c_q > 0
   d alpha*/d c_s <= 0 (weakly)

For each of (c_p, c_q, c_s), sweep one cost-price while holding the
others at default, run FB-PDE to MFE, and record alpha*.

Output: sim/results/comparative.json
"""
from __future__ import annotations
import argparse
import json
import os
import time
import numpy as np

from .config import default_cfg
from .solvers.fb_pde import solve_fb_pde


def sweep(name: str, values: list[float], layout: str, N: int):
    out = []
    for v in values:
        cfg = default_cfg()      # T_horizon_s=60, N_t=60 by default
        cfg.topo.N = N; cfg.topo.layout = layout
        cfg.solver.K_max_outer = 30
        setattr(cfg.cost, name, v)
        t0 = time.time()
        res = solve_fb_pde(cfg, alpha_init=0.8)
        out.append({
            name: float(v),
            "alpha_star": float(res["alpha_traj"].mean()),
            "n_iters": int(res["n_iters"]),
            "wall_clock_s": time.time() - t0,
        })
        print(f"  {name}={v:6.2f}: alpha*={out[-1]['alpha_star']:.3f}  ({time.time()-t0:.0f}s)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N",      type=int, default=100)
    ap.add_argument("--layout", default="milan")
    ap.add_argument("--out", default="sim/results/comparative.json")
    args = ap.parse_args()

    t0 = time.time()
    out = {"N": args.N, "layout": args.layout, "sweeps": {}}

    print("c_p sweep (Lemma 2: d alpha*/d c_p < 0):")
    out["sweeps"]["c_p"] = sweep("c_p", [0.1, 0.25, 0.5, 1.0, 2.0],
                                    args.layout, args.N)
    print("\nc_q sweep (Lemma 2: d alpha*/d c_q > 0):")
    out["sweeps"]["c_q"] = sweep("c_q", [0.5, 1.0, 2.0, 4.0, 8.0],
                                    args.layout, args.N)
    print("\nc_s sweep (Lemma 2: d alpha*/d c_s <= 0):")
    out["sweeps"]["c_s"] = sweep("c_s", [0.0, 1.0, 5.0, 20.0, 50.0],
                                    args.layout, args.N)

    # Signs check
    for name in ["c_p", "c_q", "c_s"]:
        alphas = [r["alpha_star"] for r in out["sweeps"][name]]
        # Crude monotonicity check
        diffs = np.diff(alphas)
        sgn_pos = (diffs > 0).all()
        sgn_neg = (diffs < 0).all()
        print(f"  {name}: alpha* sequence = {[round(a, 3) for a in alphas]}, monotone {'+' if sgn_pos else ('-' if sgn_neg else 'mixed')}")
        out["sweeps"][name + "_monotone"] = ("+" if sgn_pos else
                                              ("-" if sgn_neg else "mixed"))

    out["wall_clock_s"] = time.time() - t0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n[run_comparative] saved {args.out} ({out['wall_clock_s']:.0f}s)")


if __name__ == "__main__":
    main()
