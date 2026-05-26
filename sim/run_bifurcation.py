"""
run_bifurcation.py
------------------
Probe whether the mean-field game admits *multiple* equilibria and a
*phase transition* as the interference coupling eta_I (or the cost
ratio c_p/c_q) crosses a critical value.

For each value of the swept parameter, run the FB-PDE solver from a
grid of initial mean-field profiles alpha_init in [0.05, 0.95] and
record the converged MFE awake density alpha*. If, for a given
parameter value, the set of converged alpha* spans more than a
tolerance, the equilibrium is multiple and the basin reached depends
on initialization (the signature of a tragedy-of-the-commons fold).

Outputs:
  sim/results/bifurcation_eta.json   : alpha*(eta_I, alpha_init) grid
  sim/results/basin_boundary.json    : fine alpha_init sweep at a
                                       multiplicity-regime eta_I
"""
from __future__ import annotations
import argparse
import json
import os
import time
import numpy as np

from .config import default_cfg
from .solvers.fb_pde import solve_fb_pde


def _solve_alpha_star(cfg, alpha_init: float) -> float:
    res = solve_fb_pde(cfg, alpha_init=alpha_init)
    return float(res["alpha_traj"].mean())


def sweep_eta(layout: str, N: int, etas, inits, out_path: str,
               kappa: float = 0.05, K_max: int = 200):
    grid = {}
    t0 = time.time()
    for eta in etas:
        row = {}
        for a0 in inits:
            cfg = default_cfg()              # T=60, N_t=60 defaults
            cfg.topo.N = N; cfg.topo.layout = layout
            cfg.chan.eta_I = eta
            # Heavy damping + many iterations: light damping stalls at the
            # discretization floor at init-dependent points, a numerical
            # artifact that mimics multiplicity. Heavy damping converges
            # the FB iteration to the true (unique) fixed point.
            cfg.solver.kappa = kappa
            cfg.solver.K_max_outer = K_max
            a_star = _solve_alpha_star(cfg, a0)
            row[f"{a0:.2f}"] = a_star
        vals = np.array(list(row.values()))
        spread = float(vals.max() - vals.min())
        # The coarse 64x17 grid floors the FB residual near 0.06 in alpha,
        # so only a spread clearly above that floor indicates genuine
        # init-dependence (distinct basins) rather than transport noise.
        multiple = spread > 0.10
        grid[f"{eta:.2f}"] = {"by_init": row,
                               "alpha_min": float(vals.min()),
                               "alpha_max": float(vals.max()),
                               "spread": spread,
                               "multiple": multiple}
        flag = "  <-- MULTIPLE" if multiple else ""
        print(f"  eta_I={eta:4.2f}: alpha* in [{vals.min():.3f}, {vals.max():.3f}] "
              f"spread={spread:.3f}{flag}")
    out = {"layout": layout, "N": N, "etas": list(etas),
           "inits": list(inits), "grid": grid,
           "wall_clock_s": time.time() - t0}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"  saved {out_path} ({out['wall_clock_s']:.0f}s)")
    return out


def basin_boundary(layout: str, N: int, eta: float, inits, out_path: str):
    print(f"\n=== Fine basin map at eta_I={eta} ===")
    t0 = time.time()
    rows = []
    for a0 in inits:
        cfg = default_cfg()
        cfg.topo.N = N; cfg.topo.layout = layout
        cfg.chan.eta_I = eta
        cfg.solver.kappa = 0.05
        cfg.solver.K_max_outer = 200
        a_star = _solve_alpha_star(cfg, a0)
        rows.append({"alpha_init": float(a0), "alpha_star": a_star})
        print(f"  init={a0:.3f} -> alpha*={a_star:.3f}")
    out = {"layout": layout, "N": N, "eta_I": eta, "rows": rows,
           "wall_clock_s": time.time() - t0}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"  saved {out_path} ({out['wall_clock_s']:.0f}s)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--layout", default="milan")
    ap.add_argument("--etas", type=float, nargs="+",
                     default=[0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0])
    ap.add_argument("--inits", type=float, nargs="+",
                     default=[0.1, 0.3, 0.5, 0.7, 0.9])
    args = ap.parse_args()

    print("=== Bifurcation sweep over interference coupling eta_I ===")
    out = sweep_eta(args.layout, args.N, args.etas, args.inits,
                     "sim/results/bifurcation_eta.json")

    # Find an eta with multiplicity, then map the basin boundary finely
    mult_etas = [float(e) for e, v in out["grid"].items() if v["multiple"]]
    if mult_etas:
        eta_star = mult_etas[len(mult_etas) // 2]
        fine_inits = list(np.round(np.linspace(0.05, 0.95, 19), 3))
        basin_boundary(args.layout, args.N, eta_star, fine_inits,
                        "sim/results/basin_boundary.json")
    else:
        print("\n  No multiplicity detected in the swept eta range; "
              "the MFE is unique throughout.")


if __name__ == "__main__":
    main()
