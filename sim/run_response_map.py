"""
run_response_map.py
-------------------
Compute the scalar best-response map Psi(alpha): freeze the mean-field
awake density at a constant alpha_in, solve the representative agent's
HJB best response and push the population forward by one Fokker-Planck
pass, and record the induced awake density alpha_out = Psi(alpha_in).

A fixed point Psi(alpha)=alpha is a mean-field equilibrium. The slope
Psi'(alpha) is the contraction modulus of Theorem 2: where |Psi'|<1 the
map crosses the diagonal once (a unique, stable MFE); as the
interference coupling eta_I grows the map steepens, and once |Psi'|>1
it can cross the diagonal three times, the bistable coordination
collapse of the bifurcation experiment. This driver therefore visualizes
the *mechanism* behind Theorem 2 and Remark 3, complementing the
outcome shown by the bifurcation diagram.

Output: sim/results/response_map.json
"""
from __future__ import annotations
import argparse, json, os, time
import warnings
warnings.filterwarnings("ignore")
import numpy as np

from .config import default_cfg
from .solvers.fb_pde import solve_fb_pde


def psi_of_alpha(eta_I: float, alphas, kappa: float = 1.0) -> np.ndarray:
    """Psi(alpha_in) = time-averaged induced awake density from a single
    HJB best response + FP push at the frozen field alpha_in."""
    out = np.zeros(len(alphas))
    for i, a in enumerate(alphas):
        cfg = default_cfg()
        cfg.chan.eta_I = eta_I
        cfg.solver.K_max_outer = 1     # one HJB+FP pass = one best response
        cfg.solver.kappa = kappa       # kappa=1 => alpha_out is the raw FP density
        res = solve_fb_pde(cfg, alpha_init=float(a))
        out[i] = float(res["alpha_traj"].mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--etas", type=float, nargs="+", default=[1.2, 8.0])
    ap.add_argument("--n_alpha", type=int, default=19)
    ap.add_argument("--out", default="sim/results/response_map.json")
    args = ap.parse_args()

    alphas = np.round(np.linspace(0.05, 0.95, args.n_alpha), 4)
    t0 = time.time()
    curves = {}
    for eta in args.etas:
        psi = psi_of_alpha(eta, alphas)
        # numerical slope and fixed points (sign changes of Psi(a)-a)
        diff = psi - alphas
        fps = []
        for k in range(len(alphas) - 1):
            if diff[k] == 0.0:
                fps.append(float(alphas[k]))
            elif diff[k] * diff[k + 1] < 0:
                # linear interpolation of the crossing
                a0, a1 = alphas[k], alphas[k + 1]
                f0, f1 = diff[k], diff[k + 1]
                fps.append(float(a0 - f0 * (a1 - a0) / (f1 - f0)))
        max_slope = float(np.max(np.abs(np.diff(psi) / np.diff(alphas))))
        curves[f"{eta:.2f}"] = {"alpha_in": alphas.tolist(),
                                 "psi": psi.tolist(),
                                 "fixed_points": fps,
                                 "max_slope": max_slope}
        print(f"  eta_I={eta:4.2f}: max|Psi'|={max_slope:.2f}, "
              f"fixed points at {[round(x,3) for x in fps]}")
    out = {"etas": args.etas, "n_alpha": args.n_alpha, "curves": curves,
           "wall_clock_s": time.time() - t0}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[run_response_map] saved {args.out} ({out['wall_clock_s']:.0f}s)")


if __name__ == "__main__":
    main()
