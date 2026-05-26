"""
run_graphon.py
--------------
Graphon (multi-population) MFG experiment on real heterogeneous
topology. We:

  1. Load real cell positions + per-cell traffic intensity (Milan).
  2. Partition cells into B traffic-intensity blocks; estimate the
     block-graphon kernel W from spatial proximity.
  3. Solve the graphon MFG -> per-block awake-density profile alpha*_b
     and per-block feedback policies.
  4. Solve the scalar-global MFG (single population, homogeneous
     lambda_0 = cluster average) as the baseline.
  5. Deploy BOTH on the SAME real heterogeneous N-cell environment
     (per-cell arrival = intensity_i * lambda_0) and compare energy,
     per-cell queue, and per-block queue.

The thesis: a scalar-global policy tuned to the average load
under-serves busy cells (queue blowup) and over-serves quiet cells
(wasted energy); the graphon policy tailors wakefulness to local
traffic and Pareto-improves on both axes.

Output: sim/results/graphon.json
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
from .realdata import load_milan, load_shanghai
from .mean_field import block_partition, block_graphon_kernel
from .solvers.graphon import solve_graphon_mfg, GraphonController
from .solvers.fb_pde import solve_fb_pde, FBPDEController
from .dynamics import MFGEnv
from .metrics import summarize_episode


def _het_arrival_factor(arrival_factor, intensity):
    """Fold per-cell intensity into the diurnal factor so the deployed
    environment is genuinely heterogeneous: cell i's mean arrival is
    intensity_i * lambda_0."""
    return arrival_factor * intensity[:, None]


def _episode(cfg, env, controller, labels, B):
    env.reset()
    n = cfg.n_slots_episode
    energy = np.zeros(n); served = 0.0
    q_tr = np.zeros((n, env.N)); a_tr = np.zeros(n)
    for k in range(n):
        alpha = float(np.mean(env.e))
        u = controller.act(env.q, env.e, alpha, t_s=env.t_s)
        info = env.step(u)
        energy[k] = info["energy_W"].sum()
        served += float(np.sum(info["mu"])) * cfg.time.dt_s
        q_tr[k] = env.q; a_tr[k] = info["alpha"]
    m = summarize_episode(energy, q_tr, a_tr, None, env.n_toggles,
                           served, cfg.topo.area_km2, cfg.time.dt_s)
    # Per-block mean queue and per-block deployed awake density
    e_tr = np.zeros((n, env.N))   # recompute awake per block from final-state proxy
    m["queue_by_block"] = [float(q_tr[:, labels == b].mean())
                            for b in range(B)]
    return m


def _episode_with_awake(cfg, env, controller, labels, B):
    """Like _episode but also records per-block deployed awake density."""
    env.reset()
    n = cfg.n_slots_episode
    energy = np.zeros(n); served = 0.0
    q_tr = np.zeros((n, env.N)); a_tr = np.zeros(n); e_tr = np.zeros((n, env.N))
    for k in range(n):
        alpha = float(np.mean(env.e))
        u = controller.act(env.q, env.e, alpha, t_s=env.t_s)
        info = env.step(u)
        energy[k] = info["energy_W"].sum()
        served += float(np.sum(info["mu"])) * cfg.time.dt_s
        q_tr[k] = env.q; a_tr[k] = info["alpha"]; e_tr[k] = env.e
    m = summarize_episode(energy, q_tr, a_tr, None, env.n_toggles,
                           served, cfg.topo.area_km2, cfg.time.dt_s)
    m["queue_by_block"] = [float(q_tr[:, labels == b].mean()) for b in range(B)]
    m["awake_by_block"] = [float(e_tr[:, labels == b].mean()) for b in range(B)]
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--B", type=int, default=3)
    ap.add_argument("--layout", default="milan", choices=["milan", "shanghai"])
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--radius_km", type=float, default=0.15)
    ap.add_argument("--out", default="sim/results/graphon.json")
    args = ap.parse_args()

    cfg = default_cfg()
    cfg.topo.N = args.N; cfg.topo.layout = args.layout
    cfg.solver.kappa = 0.15
    loader = load_milan if args.layout == "milan" else load_shanghai
    d = loader(cfg)
    pos = d["positions"]
    intensity = d.get("cell_intensity", np.ones(args.N))
    arr = d["arrival_factor"]
    het_arr = _het_arrival_factor(arr, intensity)
    seeds = canonical_seeds(args.n_seeds)

    B = args.B
    labels = block_partition(intensity, B)
    block_lambda0 = np.array([cfg.arr.lambda0_pdu_per_s *
                               intensity[labels == b].mean() for b in range(B)])
    W = block_graphon_kernel(pos, labels, B, radius_km=args.radius_km)
    print(f"=== Graphon MFG on {args.layout}, N={args.N}, B={B} ===")
    print(f"  block sizes  = {[int((labels==b).sum()) for b in range(B)]}")
    print(f"  block lambda0= {np.round(block_lambda0,1)}")
    print(f"  graphon W diag = {np.round(np.diag(W),3)} (1/B={1/B:.3f} if well-mixed)")

    t0 = time.time()
    # ---- Graphon MFG ----
    gres = solve_graphon_mfg(cfg, block_lambda0, W, alpha_init=0.7)
    alpha_blocks = gres["alpha_blocks"].mean(axis=1)
    print(f"  graphon alpha*_b = {np.round(alpha_blocks,3)} "
          f"(spread {alpha_blocks.max()-alpha_blocks.min():.3f}), "
          f"{time.time()-t0:.0f}s")
    g_ctl = GraphonController(cfg, args.N, labels, gres["policy_tables"],
                               gres["q_grid"], gres["e_grid"], gres["t_grid"])

    # ---- Scalar-global MFG (homogeneous lambda_0 = average) ----
    cfg_s = default_cfg()
    cfg_s.topo.N = args.N; cfg_s.topo.layout = args.layout
    cfg_s.solver.kappa = 0.15
    cfg_s.arr.lambda0_pdu_per_s = float(cfg.arr.lambda0_pdu_per_s *
                                         intensity.mean())   # = lambda_0 (mean intensity=1)
    sres = solve_fb_pde(cfg_s, alpha_init=0.7)
    s_ctl = FBPDEController(cfg_s, args.N, sres["policy_table"],
                             sres["q_grid"], sres["e_grid"], sres["t_grid"])
    print(f"  scalar alpha* = {sres['alpha_traj'].mean():.3f}")

    # ---- Deploy both on the heterogeneous environment ----
    def eval_ctl(ctl):
        rows = [_episode_with_awake(cfg, MFGEnv(cfg, pos, seed=s,
                                                 arrival_factor=het_arr),
                          ctl, labels, B) for s in seeds]
        agg = {}
        for key in ["mean_power_W", "mean_queue_pdu", "p95_queue_pdu",
                     "mean_alpha", "toggles_per_min_per_cell"]:
            vals = np.array([r[key] for r in rows])
            agg[key] = {"mean": float(vals.mean()),
                         "lo": float(np.quantile(vals, 0.025)),
                         "hi": float(np.quantile(vals, 0.975))}
        qb = np.array([r["queue_by_block"] for r in rows])  # (seeds, B)
        ab = np.array([r["awake_by_block"] for r in rows])
        agg["queue_by_block"] = qb.mean(axis=0).tolist()
        agg["awake_by_block"] = ab.mean(axis=0).tolist()
        return agg

    graphon_eval = eval_ctl(g_ctl)
    scalar_eval = eval_ctl(s_ctl)

    print(f"\n  {'controller':<14} {'P(kW)':>7} {'<Q>':>7} {'p95Q':>7} {'<a>':>6}")
    for name, ev in [("Graphon-MFG", graphon_eval), ("Scalar-MFG", scalar_eval)]:
        print(f"  {name:<14} {ev['mean_power_W']['mean']/1000:>7.2f} "
              f"{ev['mean_queue_pdu']['mean']:>7.1f} "
              f"{ev['p95_queue_pdu']['mean']:>7.1f} "
              f"{ev['mean_alpha']['mean']:>6.3f}")
    print(f"  per-block <Q>: graphon={np.round(graphon_eval['queue_by_block'],1)}  "
          f"scalar={np.round(scalar_eval['queue_by_block'],1)}")

    # ---- Modeling-fidelity: model-predicted vs deployed awake density ----
    scalar_pred = float(sres["alpha_traj"].mean())   # one number for all blocks
    graphon_pred = alpha_blocks                        # per-block prediction
    deployed_g = np.array(graphon_eval["awake_by_block"])
    err_scalar = float(np.mean(np.abs(scalar_pred - deployed_g)))
    err_graphon = float(np.mean(np.abs(graphon_pred - deployed_g)))
    qb = np.array(graphon_eval["queue_by_block"])
    print(f"\n  Modeling fidelity (per-block awake-density prediction error):")
    print(f"    scalar model (uniform alpha*={scalar_pred:.2f}): err={err_scalar:.3f}")
    print(f"    graphon model (profile {np.round(graphon_pred,2)}): err={err_graphon:.3f}")
    print(f"    per-block QoS disparity (queue ratio max/min): "
          f"{qb.max()/max(qb.min(),1e-6):.1f}x")

    out = {
        "fidelity": {"scalar_pred_uniform": scalar_pred,
                      "graphon_pred_profile": graphon_pred.tolist(),
                      "deployed_awake_by_block": deployed_g.tolist(),
                      "scalar_pred_err": err_scalar,
                      "graphon_pred_err": err_graphon,
                      "queue_disparity_ratio": float(qb.max()/max(qb.min(),1e-6))},
        "layout": args.layout, "N": args.N, "B": B,
        "n_seeds": args.n_seeds, "radius_km": args.radius_km,
        "block_sizes": [int((labels == b).sum()) for b in range(B)],
        "block_lambda0": block_lambda0.tolist(),
        "graphon_W": W.tolist(),
        "alpha_blocks_star": alpha_blocks.tolist(),
        "scalar_alpha_star": float(sres["alpha_traj"].mean()),
        "graphon_eval": graphon_eval,
        "scalar_eval": scalar_eval,
        "wall_clock_s": time.time() - t0,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n[run_graphon] saved {args.out} ({out['wall_clock_s']:.0f}s)")


if __name__ == "__main__":
    main()
