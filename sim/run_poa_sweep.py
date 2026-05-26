"""
run_poa_sweep.py
----------------
Coordination-collapse curve: price of anarchy versus interference
coupling eta_I (the data behind the left panel of the PoA figure).

For each eta_I we compute:
  * the social optimum J_MFC by enforcing a target awake fraction and
    minimizing the realized weighted social cost (mean-field control),
  * the selfish equilibrium J_MFE by solving the forward-backward HJB-FP
    system and evaluating its realized social cost,
and report PoA = J_MFE / J_MFC together with the selfish and social
awake densities. As eta_I grows, selfish cells over-sleep to dodge
interference while the planner keeps capacity awake, so the PoA climbs.

Output: sim/results/poa_sweep.json  (schema consumed by make_figures.fig_poa)
"""
from __future__ import annotations
import argparse, json, os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np

from .config import default_cfg, canonical_seeds
from .topology import make_layout_and_arrivals
from .dynamics import MFGEnv
from .solvers.fb_pde import solve_fb_pde, FBPDEController
from .run_poa import FixedFractionController, _social_cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--layout", default="milan")
    ap.add_argument("--n_seeds", type=int, default=4)
    ap.add_argument("--etas", type=float, nargs="+",
                     default=[1.2, 2.0, 3.0, 4.5, 6.0, 8.0])
    ap.add_argument("--fracs", type=float, nargs="+",
                     default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ap.add_argument("--out", default="sim/results/poa_sweep.json")
    args = ap.parse_args()

    cfg0 = default_cfg(); cfg0.topo.N = args.N; cfg0.topo.layout = args.layout
    pos, fac = make_layout_and_arrivals(cfg0)
    seeds = canonical_seeds(args.n_seeds)
    n = cfg0.n_slots_episode
    t0 = time.time()

    etas, poas, a_mfes, a_mfcs, J_mfes, J_mfcs = [], [], [], [], [], []
    for eta in args.etas:
        cfg = default_cfg(); cfg.topo.N = args.N; cfg.topo.layout = args.layout
        cfg.chan.eta_I = eta; cfg.solver.kappa = 0.1

        # --- Social optimum (MFC): minimize realized J over awake budgets ---
        best_J, best_a = np.inf, 1.0
        for fr in args.fracs:
            ctl = FixedFractionController(cfg, args.N, fr)
            rows = [_social_cost(cfg, MFGEnv(cfg, pos, seed=s, arrival_factor=fac),
                                  ctl, n) for s in seeds]
            J = float(np.mean([r["J"] for r in rows]))
            if J < best_J:
                best_J = J
                best_a = float(np.mean([r["alpha"] for r in rows]))
        J_mfc, a_mfc = best_J, best_a

        # --- Selfish equilibrium (MFE): FB solve, evaluate social cost ---
        res = solve_fb_pde(cfg, alpha_init=0.7)
        ctl = FBPDEController(cfg, args.N, res["policy_table"],
                               res["q_grid"], res["e_grid"], res["t_grid"])
        rows = [_social_cost(cfg, MFGEnv(cfg, pos, seed=s, arrival_factor=fac),
                              ctl, n) for s in seeds]
        J_mfe = float(np.mean([r["J"] for r in rows]))
        a_mfe = float(np.mean([r["alpha"] for r in rows]))

        poa = J_mfe / max(J_mfc, 1e-9)
        etas.append(float(eta)); poas.append(poa)
        a_mfes.append(a_mfe); a_mfcs.append(a_mfc)
        J_mfes.append(J_mfe); J_mfcs.append(J_mfc)
        print(f"  eta={eta:4.1f}: PoA={poa:5.2f}  "
              f"(J_MFE={J_mfe:6.1f}/J_MFC={J_mfc:6.1f})  "
              f"a_MFE={a_mfe:.2f}  a_MFC={a_mfc:.2f}")

    out = {"etas": etas, "poa": poas, "alpha_mfe": a_mfes,
           "alpha_mfc": a_mfcs, "J_mfe": J_mfes, "J_mfc": J_mfcs,
           "layout": args.layout, "N": args.N, "n_seeds": args.n_seeds,
           "wall_clock_s": time.time() - t0}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n[run_poa_sweep] saved {args.out} ({out['wall_clock_s']:.0f}s)")


if __name__ == "__main__":
    main()
