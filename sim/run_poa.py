"""
run_poa.py
----------
Price of anarchy and Pigouvian mean-field toll.

The selfish mean-field equilibrium (MFE) need not minimize the
population-average (social) cost: each cell ignores the interference
externality its wakefulness imposes on neighbors. We:

  1. Trace the SOCIAL cost J(alpha) by enforcing a target awake
     fraction alpha (highest-queue cells kept awake) and measuring the
     realized weighted cost J = c_q*Qbar + c_p*Pbar_percell + c_s*Sbar.
     The mean-field-control optimum is alpha_MFC = argmin_alpha J(alpha).
  2. Compute the selfish MFE awake density alpha_MFE from the FB solver.
  3. Add a Pigouvian toll tau on wakefulness (the per-agent running cost
     gains tau*e). Sweeping tau traces alpha_MFE(tau); the toll
     tau* that makes alpha_MFE(tau*) = alpha_MFC decentralizes the
     social optimum.

Outputs: sim/results/poa.json
"""
from __future__ import annotations
import argparse, json, os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np

from .config import default_cfg, canonical_seeds
from .topology import make_layout_and_arrivals
from .dynamics import MFGEnv
from .solvers.fb_pde import solve_fb_pde, FBPDEController
from .metrics import summarize_episode
from .mean_field import awake_density


class FixedFractionController:
    """Keeps a target fraction `frac` of cells awake: the highest-queue
    cells are driven awake, the rest to sleep. Traces the social cost
    J(alpha) for a planner that can dictate the awake budget."""
    name = "FIXED-FRAC"
    def __init__(self, cfg, N, frac):
        self.cfg, self.N, self.frac = cfg, N, frac
    def reset(self, *a, **k): pass
    def act(self, q, e, alpha, t_s=0.0):
        k = int(round(self.frac * self.N))
        target = np.zeros(self.N)
        if k > 0:
            target[np.argsort(-q)[:k]] = 1.0
        return np.sign(target - e) * self.cfg.energy.u_bar


def _social_cost(cfg, env, ctl, n):
    env.reset()
    E = np.zeros(n); q = np.zeros((n, env.N)); a = np.zeros(n); S = 0.0
    for kk in range(n):
        al = awake_density(env.e)
        u = ctl.act(env.q, env.e, al, t_s=env.t_s)
        info = env.step(u)
        E[kk] = info["energy_W"].sum(); q[kk] = env.q; a[kk] = info["alpha"]
        S += float((u ** 2).sum())
    Pbar_per = E.mean() / env.N
    Qbar = q.mean()
    Sbar = S / n / env.N
    J = cfg.cost.c_q * Qbar + cfg.cost.c_p * Pbar_per + cfg.cost.c_s * Sbar
    return {"J": float(J), "Qbar": float(Qbar), "Pbar_per": float(Pbar_per),
            "alpha": float(a.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--layout", default="milan")
    ap.add_argument("--eta_I", type=float, default=1.2)
    ap.add_argument("--n_seeds", type=int, default=4)
    ap.add_argument("--fracs", type=float, nargs="+",
                     default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ap.add_argument("--tolls", type=float, nargs="+",
                     default=[0.0, 10.0, 20.0, 40.0, 80.0, 160.0])
    ap.add_argument("--lambda0", type=float, default=None,
                     help="override baseline arrival (with --eps_lambda, sets load-shift strength)")
    ap.add_argument("--eps_lambda", type=float, default=None,
                     help="override load-shift regularizer; large value => weak load shifting")
    ap.add_argument("--c_q", type=float, default=None, help="override queue price")
    ap.add_argument("--out", default="sim/results/poa.json")
    args = ap.parse_args()

    cfg = default_cfg(); cfg.topo.N = args.N; cfg.topo.layout = args.layout
    cfg.chan.eta_I = args.eta_I
    if args.lambda0 is not None:    cfg.arr.lambda0_pdu_per_s = args.lambda0
    if args.eps_lambda is not None: cfg.chan.epsilon_lambda = args.eps_lambda
    if args.c_q is not None:        cfg.cost.c_q = args.c_q
    cfg.solver.kappa = 0.1
    pos, fac = make_layout_and_arrivals(cfg)
    seeds = canonical_seeds(args.n_seeds)
    n = cfg.n_slots_episode
    t0 = time.time()

    # ---- 1. Social cost curve J(alpha) via enforced awake fraction ----
    print("Social cost J(alpha) by enforced awake fraction:")
    social = []
    for fr in args.fracs:
        ctl = FixedFractionController(cfg, args.N, fr)
        rows = [_social_cost(cfg, MFGEnv(cfg, pos, seed=s, arrival_factor=fac),
                              ctl, n) for s in seeds]
        J = float(np.mean([r["J"] for r in rows]))
        al = float(np.mean([r["alpha"] for r in rows]))
        Q = float(np.mean([r["Qbar"] for r in rows]))
        P = float(np.mean([r["Pbar_per"] for r in rows]))
        social.append({"frac": fr, "alpha": al, "J": J, "Qbar": Q, "Pbar_per": P})
        print(f"  frac={fr:.2f} -> alpha={al:.3f}  J={J:.1f}  (Q={Q:.0f}, P/cell={P:.0f}W)")
    J_arr = np.array([s["J"] for s in social])
    a_arr = np.array([s["alpha"] for s in social])
    i_mfc = int(np.argmin(J_arr))
    alpha_mfc = a_arr[i_mfc]; J_mfc = J_arr[i_mfc]
    print(f"  => social optimum (MFC): alpha_MFC={alpha_mfc:.3f}, J_MFC={J_mfc:.1f}")

    # ---- 2 & 3. Selfish MFE and toll sweep ----
    print("\nSelfish MFE awake density vs Pigouvian toll tau:")
    toll_curve = []
    for tau in args.tolls:
        cfg.cost.toll = tau
        res = solve_fb_pde(cfg, alpha_init=0.7)
        ctl = FBPDEController(cfg, args.N, res["policy_table"],
                               res["q_grid"], res["e_grid"], res["t_grid"])
        # deploy to measure realized alpha and social cost J
        rows = [_social_cost(cfg, MFGEnv(cfg, pos, seed=s, arrival_factor=fac),
                              ctl, n) for s in seeds]
        # NOTE: J here uses the TRUE social cost (toll is a transfer, not a
        # social cost), so we recompute J without the toll term.
        cfg_eval = default_cfg(); cfg_eval.topo.N = args.N
        cfg_eval.topo.layout = args.layout; cfg_eval.chan.eta_I = args.eta_I  # toll=0 for J
        if args.c_q is not None: cfg_eval.cost.c_q = args.c_q  # match social-curve pricing
        Js = [cfg_eval.cost.c_q * r["Qbar"] + cfg_eval.cost.c_p * r["Pbar_per"]
              + cfg_eval.cost.c_s * 0.0 for r in rows]
        a_mfe = float(np.mean([r["alpha"] for r in rows]))
        J_mfe = float(np.mean(Js))
        toll_curve.append({"tau": tau, "alpha_mfe": a_mfe, "J_social": J_mfe,
                            "alpha_star_solver": float(res["alpha_traj"].mean())})
        print(f"  tau={tau:6.1f} -> alpha_MFE={a_mfe:.3f}  J_social={J_mfe:.1f}")
    cfg.cost.toll = 0.0

    alpha_mfe0 = toll_curve[0]["alpha_mfe"]
    J_mfe0 = toll_curve[0]["J_social"]
    # Find toll that drives alpha_MFE closest to alpha_MFC
    diffs = [abs(tc["alpha_mfe"] - alpha_mfc) for tc in toll_curve]
    i_star = int(np.argmin(diffs))
    tau_star = toll_curve[i_star]["tau"]
    poa = J_mfe0 / max(J_mfc, 1e-9)
    poa_tolled = toll_curve[i_star]["J_social"] / max(J_mfc, 1e-9)
    print(f"\n  Price of anarchy J(MFE)/J(MFC) = {J_mfe0:.1f}/{J_mfc:.1f} = {poa:.3f}")
    print(f"  Pigouvian toll tau* = {tau_star:.1f} -> alpha_MFE={toll_curve[i_star]['alpha_mfe']:.3f} "
          f"(target alpha_MFC={alpha_mfc:.3f})")
    print(f"  Residual PoA after toll = {poa_tolled:.3f} "
          f"({(poa-poa_tolled)/(poa-1+1e-9)*100:.0f}% of gap closed)")

    out = {
        "layout": args.layout, "N": args.N, "n_seeds": args.n_seeds,
        "social_curve": social,
        "alpha_mfc": float(alpha_mfc), "J_mfc": float(J_mfc),
        "alpha_mfe": float(alpha_mfe0), "J_mfe": float(J_mfe0),
        "toll_curve": toll_curve,
        "tau_star": float(tau_star),
        "poa": float(poa), "poa_tolled": float(poa_tolled),
        "wall_clock_s": time.time() - t0,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n[run_poa] saved {args.out} ({out['wall_clock_s']:.0f}s)")


if __name__ == "__main__":
    main()
