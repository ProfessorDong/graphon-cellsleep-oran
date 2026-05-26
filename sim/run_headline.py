"""
run_headline.py
---------------
Main driver: train each MFG solver once, then evaluate all six controllers
on n_seeds episodes for N in {100, 200, 500}.

Outputs:  sim/results/headline.json  with per-controller per-N aggregates.

Run from the mfg-cell-sleep/ directory:
    python3 -m sim.run_headline
"""
from __future__ import annotations
import argparse
import json
import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np

from .config import default_cfg, canonical_seeds
from .topology import make_layout, make_layout_and_arrivals
from .dynamics import MFGEnv
from .baselines import (
    AlwaysOnController, IndepThController, CentSocController,
)
from .solvers.fb_pde import solve_fb_pde, FBPDEController
from .solvers.pmfpi import solve_pmfpi, PMFPIController
from .solvers.mfac import solve_mfac, MFACController
from .metrics import run_one_episode, aggregate


def evaluate_controller(cfg, positions, controller, seeds: list[int],
                          arrival_factor=None) -> dict:
    rows = []
    for s in seeds:
        env = MFGEnv(cfg, positions, seed=s, arrival_factor=arrival_factor)
        m = run_one_episode(cfg, env, controller, cfg.n_slots_episode)
        rows.append(m)
    return aggregate(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ns",     type=int, nargs="+", default=[100, 200, 500])
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--layout", default="hex",
                     choices=["hex", "ppp", "milan", "shanghai", "netmob"])
    ap.add_argument("--out", default="sim/results/headline.json")
    args = ap.parse_args()

    seeds = canonical_seeds(args.n_seeds)
    out = {"Ns": args.Ns, "n_seeds": args.n_seeds, "layout": args.layout,
           "by_N": {}}
    t_global = time.time()

    for N in args.Ns:
        print(f"\n=== N = {N} ({args.layout}) ===")
        cfg = default_cfg()
        cfg.topo.N = N
        cfg.topo.layout = args.layout
        pos, arrival_factor = make_layout_and_arrivals(cfg)
        if arrival_factor is not None:
            print(f"  ({args.layout} real data: per-cell diurnal profiles loaded)")
        results = {}

        # ----- baselines -----
        for name, Ctl in [
            ("ALWAYS-ON",  AlwaysOnController),
            ("INDEP-TH",   IndepThController),
            ("CENT-SOC",   CentSocController),
        ]:
            ctl = Ctl(cfg, N)
            agg = evaluate_controller(cfg, pos, ctl, seeds, arrival_factor=arrival_factor)
            results[name] = agg
            mp = agg["mean_power_W"]["mean"]; mq = agg["mean_queue_pdu"]["mean"]
            ma = agg["mean_alpha"]["mean"]; mt = agg["toggles_per_min_per_cell"]["mean"]
            print(f"  {name:<14} P={mp:7.0f}W  <Q>={mq:6.1f}  <α>={ma:.3f}  tog={mt:6.2f}/min/cell")

        # ----- MFG solvers (train once, evaluate on the same seeds) -----
        print("  training MFG-FB ...")
        t0 = time.time(); fb_out = solve_fb_pde(cfg, alpha_init=0.8)
        print(f"    [{time.time()-t0:.0f}s] fixed point alpha*={fb_out['alpha_traj'].mean():.3f}")
        ctl_fb = FBPDEController(cfg, N, fb_out["policy_table"],
                                   fb_out["q_grid"], fb_out["e_grid"],
                                   fb_out["t_grid"])
        results["MFG-FB"] = evaluate_controller(cfg, pos, ctl_fb, seeds, arrival_factor=arrival_factor)

        print("  training MFG-PMFPI ...")
        t0 = time.time(); pmfpi_out = solve_pmfpi(cfg, pos, seed=0)
        print(f"    [{time.time()-t0:.0f}s] theta*={pmfpi_out['theta_star'].round(3).tolist()}")
        ctl_pmfpi = PMFPIController(cfg, N, pmfpi_out["theta_star"])
        results["MFG-PMFPI"] = evaluate_controller(cfg, pos, ctl_pmfpi, seeds, arrival_factor=arrival_factor)

        print("  training MFG-MFAC ...")
        t0 = time.time(); mfac_out = solve_mfac(cfg, pos, seed=0, use_mf=True)
        print(f"    [{time.time()-t0:.0f}s] bar_alpha={mfac_out['bar_alpha']:.3f}")
        ctl_mfac = MFACController(cfg, N, mfac_out["agents"], mfac_out["bar_alpha"])
        results["MFG-MFAC"] = evaluate_controller(cfg, pos, ctl_mfac, seeds, arrival_factor=arrival_factor)

        print("  training MF-RL-NAIVE ...")
        t0 = time.time(); mfac_n_out = solve_mfac(cfg, pos, seed=0, use_mf=False)
        ctl_mfn = MFACController(cfg, N, mfac_n_out["agents"], mfac_n_out["bar_alpha"],
                                   name="MF-RL-NAIVE")
        results["MF-RL-NAIVE"] = evaluate_controller(cfg, pos, ctl_mfn, seeds, arrival_factor=arrival_factor)

        # Save per-N
        out["by_N"][str(N)] = results
        for name in ["MFG-FB", "MFG-PMFPI", "MFG-MFAC", "MF-RL-NAIVE"]:
            agg = results[name]
            mp = agg["mean_power_W"]["mean"]; mq = agg["mean_queue_pdu"]["mean"]
            ma = agg["mean_alpha"]["mean"]; mt = agg["toggles_per_min_per_cell"]["mean"]
            print(f"  {name:<14} P={mp:7.0f}W  <Q>={mq:6.1f}  <α>={ma:.3f}  tog={mt:6.2f}/min/cell")

    out["wall_clock_s"] = time.time() - t_global
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n[run_headline] wall-clock {out['wall_clock_s']:.0f}s, saved {args.out}")


if __name__ == "__main__":
    main()
