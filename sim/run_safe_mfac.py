"""
run_safe_mfac.py
----------------
Safety of the mean-field actor-critic (reviewer item k).

The unconstrained MFAC learns an aggressive sleep policy that drives
huge backlogs. We add a Foster-Lyapunov safety shield (SafeMFACController)
that overrides the learned action to wake any cell whose backlog exceeds
the stability threshold q_safe, enforcing Assumption (queue stability).
We train MFAC once and evaluate the unshielded and shielded controllers
on the same trace, reporting per-cell power, mean and 95th-percentile
queue, awake density, toggling, and the weighted objective
J = c_q*Qbar + c_p*Pbar_cell.

Output: sim/results/safe_mfac.json
"""
from __future__ import annotations
import argparse, json, os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np

from .config import default_cfg, canonical_seeds
from .topology import make_layout_and_arrivals
from .dynamics import MFGEnv
from .metrics import run_one_episode, aggregate
from .solvers.mfac import solve_mfac, MFACController, SafeMFACController


def _eval(cfg, pos, ctl, seeds, fac):
    rows = [run_one_episode(cfg, MFGEnv(cfg, pos, seed=s, arrival_factor=fac),
                             ctl, cfg.n_slots_episode) for s in seeds]
    return aggregate(rows)


def _row(cfg, agg):
    P = agg["mean_power_W"]["mean"]; Q = agg["mean_queue_pdu"]["mean"]
    p95 = agg["p95_queue_pdu"]["mean"]; a = agg["mean_alpha"]["mean"]
    tog = agg["toggles_per_min_per_cell"]["mean"]
    Pcell = P / cfg.topo.N
    J = cfg.cost.c_q * Q + cfg.cost.c_p * Pcell
    return {"P_kW": P / 1e3, "P_cell_W": Pcell, "Qbar": Q, "p95_Q": p95,
            "alpha": a, "toggles": tog, "J": J}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--layout", default="milan")
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--q_safe", type=float, default=None)
    ap.add_argument("--out", default="sim/results/safe_mfac.json")
    args = ap.parse_args()

    cfg = default_cfg(); cfg.topo.N = args.N; cfg.topo.layout = args.layout
    pos, fac = make_layout_and_arrivals(cfg)
    seeds = canonical_seeds(args.n_seeds)
    t0 = time.time()

    print("Training MFAC ...")
    mfac = solve_mfac(cfg, pos, seed=0, use_mf=True)
    agents, ba = mfac["agents"], mfac["bar_alpha"]

    ctl_unsafe = MFACController(cfg, args.N, agents, ba)
    ctl_safe = SafeMFACController(cfg, args.N, agents, ba, q_safe=args.q_safe)
    q_safe = ctl_safe.q_safe

    out = {"N": args.N, "layout": args.layout, "n_seeds": args.n_seeds,
           "q_safe": q_safe}
    for tag, ctl in [("MFAC", ctl_unsafe), ("MFAC-SAFE", ctl_safe)]:
        agg = _eval(cfg, pos, ctl, seeds, fac)
        r = _row(cfg, agg)
        out[tag] = r
        print(f"  {tag:10s} P/cell={r['P_cell_W']:6.1f}W  Qbar={r['Qbar']:7.1f}  "
              f"p95Q={r['p95_Q']:7.1f}  alpha={r['alpha']:.2f}  "
              f"tog={r['toggles']:.1f}  J={r['J']:.1f}")

    out["wall_clock_s"] = time.time() - t0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n[run_safe_mfac] saved {args.out} ({out['wall_clock_s']:.0f}s)")


if __name__ == "__main__":
    main()
