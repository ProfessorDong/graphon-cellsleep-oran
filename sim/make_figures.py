"""
make_figures.py
---------------
Produce the six publication-quality figures from the result JSONs.

  fig/f1_convergence.pdf    : FB iteration residual ||alpha^(n+1) - alpha^(n)||_inf
                              vs n on log-y, multiple initial alphas
  fig/f2_eps_nash.pdf       : empirical eps-Nash gap vs N on log-log,
                              with reference slope -1/2
  fig/f3_headline.pdf       : grouped bar / scatter of all 7 controllers
                              at N=200 (energy on x, queue on y)
  fig/f4_pareto.pdf         : Pareto frontier in (energy, queue) plane
                              across c_p/c_q ratios
  fig/f5_comparative.pdf    : alpha* vs c_p, c_q, c_s on three subplots
                              illustrating Lemma 2's sign predictions
  fig/f6_alpha_traj.pdf     : steady-state awake-density profile over
                              the 24-hour day for MFG-FB vs CENT-SOC
                              (validates the diurnal MFE behavior)

Usage:
    python3 -m sim.make_figures
"""
from __future__ import annotations
import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# JSON baseline keys map to the display names used in the paper.
DISPLAY = {"CENT-SOC": "CENT-MYOPIC"}


def _disp(name: str) -> str:
    return DISPLAY.get(name, name)

# Publication-quality defaults: Times-like serif (STIX) to match the
# IEEEtran body font, fully embedded TrueType fonts (type 42, required
# by IEEE), high savefig resolution for any rasterized elements, and
# tuned line/marker/grid weights for two-column reproduction.
mpl_rc = {
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 1.9,
    "lines.markersize": 6,
    "axes.linewidth": 0.8,
    "axes.grid": False,
    "grid.linewidth": 0.6,
    "grid.alpha": 0.4,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "0.7",
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
plt.rcParams.update(mpl_rc)


def fig_convergence(in_path: str, out_path: str):
    with open(in_path) as f: d = json.load(f)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for k, run in enumerate(d["runs"]):
        res = np.array(run["residuals"])
        res = np.maximum(res, 1e-9)
        ax.semilogy(np.arange(1, len(res) + 1), res,
                     marker="o", ms=4, color=colors[k % len(colors)],
                     label=fr"$\alpha^{{(0)}}={run['alpha_init']}$")
    # Reference geometric line at rate 0.7
    n_max = max(len(r["residuals"]) for r in d["runs"])
    ref = 0.7 ** np.arange(n_max)
    ax.semilogy(np.arange(1, n_max + 1), ref * res[0] * 1.5, "k--",
                 lw=1.0, alpha=0.6, label="ref $\\rho^n,\\ \\rho=0.7$")
    ax.set_xlabel("FB iteration $n$")
    ax.set_ylabel(r"$\Vert\alpha^{(n+1)}-\alpha^{(n)}\Vert_{\infty}$")
    ax.set_title(f"FB-PDE convergence (N={d['N']}, {d['layout']})")
    ax.grid(True, ls="--", alpha=0.4, which="both")
    ax.legend(loc="lower left", framealpha=0.95, fontsize=12.5)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_eps_nash(in_path: str, out_path: str):
    """Empirical epsilon-Nash gap vs N. The proper probe is the
    single-agent best response to the frozen equilibrium field
    (location-differenced); we also show the best gain over heuristic
    global deviations. The best-response gap is small and bounded,
    resting on the solver's O(1) discretization floor rather than the
    N^{-1/2} propagation-of-chaos rate, which is drawn for reference."""
    with open(in_path) as f: d = json.load(f)
    Ns = np.array(sorted(int(N) for N in d["by_N"].keys()))
    br_mean = np.array([d["by_N"][str(N)]["eps_emp_mean"] for N in Ns])
    br_max = np.array([d["by_N"][str(N)]["eps_emp_max"] for N in Ns])
    heur_mean = np.array([d["by_N"][str(N)].get("heur_emp_mean", 0.0)
                          for N in Ns])

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(Ns, br_mean, "o-", color="#1f77b4", ms=6,
            label="best-response gap (mean)")
    ax.plot(Ns, br_max, "s--", color="#1f77b4", ms=6, alpha=0.55,
            label="best-response gap (max)")
    ax.plot(Ns, heur_mean, "^-", color="#c4762e", ms=6,
            label="best heuristic-deviation gap")
    # N^{-1/2} reference anchored at the first point (propagation of chaos).
    ref = br_mean[0] * np.sqrt(Ns[0]) / np.sqrt(Ns)
    ax.plot(Ns, ref, "k:", lw=1.3,
            label=r"$N^{-1/2}$ reference")
    ax.set_xscale("log")
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(N) for N in Ns])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("$N$ (number of cells)")
    ax.set_ylabel(r"empirical $\varepsilon$-Nash gap (per cell)")
    ax.set_title(rf"Best-response $\varepsilon$-Nash gap on {d['layout']} topology")
    ax.grid(True, ls="--", alpha=0.4, which="both")
    ax.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_headline(in_path: str, out_path: str, N_pick: int | None = None):
    with open(in_path) as f: d = json.load(f)
    if N_pick is None:
        N_pick = max(int(k) for k in d["by_N"].keys())
    res = d["by_N"][str(N_pick)]
    names = ["ALWAYS-ON", "INDEP-TH", "CENT-SOC",
              "MFG-FB", "MFG-PMFPI", "MFG-MFAC", "MF-RL-NAIVE"]
    colors = ["#7f7f7f", "#c4762e", "#8c564b",
               "#1f77b4", "#2ca02c", "#d62728", "#bcbd22"]
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for i, n in enumerate(names):
        if n not in res:
            continue
        P = res[n]["mean_power_W"]["mean"] / 1000.0       # kW
        Q = res[n]["mean_queue_pdu"]["mean"]
        Plo, Phi = (res[n]["mean_power_W"]["lo"] / 1000.0,
                    res[n]["mean_power_W"]["hi"] / 1000.0)
        Qlo, Qhi = res[n]["mean_queue_pdu"]["lo"], res[n]["mean_queue_pdu"]["hi"]
        ax.errorbar(P, Q, xerr=[[P - Plo], [Phi - P]],
                     yerr=[[Q - Qlo], [Qhi - Q]],
                     marker="o", ms=8, capsize=3, color=colors[i],
                     label=_disp(n))
    ax.set_xlabel("Average cluster power (kW)")
    ax.set_ylabel("Time-averaged queue $\\bar Q$ (PDU)")
    ax.set_title(f"Energy/queue operating points at N={N_pick} ({d['layout']})")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(loc="upper left", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_pareto(in_path: str, out_path: str):
    with open(in_path) as f: d = json.load(f)
    rs = sorted(d["by_ratio"].keys(), key=float)
    names = ["MFG-FB", "CENT-SOC", "INDEP-TH", "ALWAYS-ON"]
    colors = ["#1f77b4", "#8c564b", "#c4762e", "#7f7f7f"]
    markers = ["o", "s", "^", "v"]
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for k, name in enumerate(names):
        Ps = []; Qs = []
        for r in rs:
            blob = d["by_ratio"][r].get(name)
            if blob is None:
                continue
            Ps.append(blob["mean_power_W"]["mean"] / 1000.0)
            Qs.append(blob["mean_queue_pdu"]["mean"])
        ax.plot(Ps, Qs, marker=markers[k], color=colors[k], lw=1.5,
                 ms=7, label=_disp(name))
    ax.set_xlabel("Average cluster power (kW)")
    ax.set_ylabel(r"Time-averaged queue $\bar Q$ (PDU)")
    ax.set_title(f"Energy/queue Pareto sweep over $c_p/c_q$ (N={d['N']})")
    ax.set_yscale("log")
    ax.grid(True, ls="--", alpha=0.4, which="both")
    ax.legend(loc="best", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_comparative(in_path: str, out_path: str):
    with open(in_path) as f: d = json.load(f)
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4), sharey=True)
    for ax, name, expected in zip(
            axes, ["c_p", "c_q", "c_s"],
            [r"$\partial\alpha^*/\partial c_p<0$",
              r"$\partial\alpha^*/\partial c_q>0$",
              r"$\partial\alpha^*/\partial c_s\leq 0$"]):
        rows = d["sweeps"][name]
        xs = np.array([r[name] for r in rows])
        ys = np.array([r["alpha_star"] for r in rows])
        ax.plot(xs, ys, "o-", ms=7, lw=1.6, color="#1f77b4")
        ax.set_xscale("log" if name in ("c_p", "c_q", "c_s") else "linear")
        ax.set_xlabel(rf"${name[0]}_{{{name[2]}}}$")
        ax.set_title(expected, fontsize=10)
        ax.grid(True, ls="--", alpha=0.4, which="both")
    axes[0].set_ylabel(r"MFE awake density $\alpha^*$")
    fig.suptitle(f"Comparative statics: signed price sensitivities "
                  f"({d['layout']}, $N$={d['N']})", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_alpha_traj(in_path: str, out_path: str):
    """Optional: diurnal alpha trajectory from the convergence run, if
    history records are present. We pull the LAST iteration's alpha
    profile and plot it over time."""
    with open(in_path) as f: d = json.load(f)
    runs = d["runs"]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for k, run in enumerate(runs):
        means = run["alpha_mean_per_iter"][-1]
        # Means is a scalar (mean over t); we don't have t-resolved trajectory
        # here. Fall back to a horizontal line.
        ax.axhline(run["alpha_final_mean"], lw=1.5,
                    label=fr"$\alpha^{{(0)}}={run['alpha_init']}$: $\bar\alpha^*={run['alpha_final_mean']:.3f}$")
    ax.set_xlabel("normalized time over horizon")
    ax.set_ylabel(r"MFE awake density $\alpha^*(t)$")
    ax.set_title(f"MFE awake-density profile (N={d['N']}, {d['layout']})")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(loc="best", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_graphon(in_path: str, out_path: str):
    """Two panels: (left) spatial MFE awake-density profile alpha*_b that
    the graphon predicts vs the scalar model's single value; (right) the
    realized per-block queue disparity that the scalar model is blind to."""
    with open(in_path) as f: d = json.load(f)
    B = d["B"]
    blocks = np.arange(B)
    alpha_g = np.array(d["alpha_blocks_star"])
    alpha_s = d["scalar_alpha_star"]
    lam = np.array(d["block_lambda0"])
    qb_g = np.array(d["graphon_eval"]["queue_by_block"])
    qb_s = np.array(d["scalar_eval"]["queue_by_block"])

    with plt.rc_context({"font.size": 15, "axes.labelsize": 16,
                          "axes.titlesize": 16, "legend.fontsize": 14,
                          "xtick.labelsize": 14, "ytick.labelsize": 14,
                          "figure.titlesize": 17}):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.8))
        # Left: spatial awake-density profile
        ax1.plot(blocks, alpha_g, "o-", ms=9, lw=2, color="#1f77b4",
                 label="graphon MFE $\\alpha^*_b$")
        ax1.axhline(alpha_s, ls="--", lw=1.8, color="#d62728",
                    label=f"scalar MFE (uniform $\\alpha^*={alpha_s:.2f}$)")
        ax1.set_xticks(blocks)
        ax1.set_xticklabels([f"B{b}\n$\\lambda_0$={lam[b]:.0f}" for b in blocks])
        ax1.set_xlabel("traffic-intensity block (low $\\to$ high)")
        ax1.set_ylabel("MFE awake density $\\alpha^*$")
        ax1.set_title("Spatial awake-density profile")
        ax1.grid(True, ls="--", alpha=0.4)
        ax1.legend(loc="upper left", framealpha=0.95)
        # Right: per-block queue disparity
        w = 0.38
        ax2.bar(blocks - w/2, qb_g, w, color="#1f77b4", label="graphon control")
        ax2.bar(blocks + w/2, qb_s, w, color="#8c564b", label="scalar control")
        ax2.set_xticks(blocks)
        ax2.set_xticklabels([f"B{b}" for b in blocks])
        ax2.set_xlabel("traffic-intensity block")
        ax2.set_ylabel("per-block mean queue $\\bar Q_b$ (PDU)")
        disp = d["fidelity"]["queue_disparity_ratio"]
        ax2.set_title(f"Per-region QoS disparity ({disp:.1f}$\\times$)")
        ax2.grid(True, ls="--", alpha=0.4, axis="y")
        ax2.legend(loc="upper left", framealpha=0.95)
        fig.suptitle(f"Graphon MFG on {d['layout']} topology "
                     f"($N$={d['N']}, $B$={B})", y=1.02)
        fig.tight_layout()
        fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_bifurcation(in_path: str, out_path: str):
    """Bifurcation diagram: the FB iteration's awake-density fixed point
    versus interference coupling eta_I, swept from low and high
    initializations. Below a threshold the two coincide (unique MFE);
    above it they split into a high-awake and a low-awake (coordination-
    collapsed) equilibrium, the gap widening with eta_I. The shaded band
    is the multiplicity region; faint dots are individual initializations."""
    with open(in_path) as f: d = json.load(f)
    etas = np.array(sorted(float(e) for e in d["grid"].keys()))
    cells = [d["grid"][f"{e:.2f}"] for e in etas]
    a_hi = np.array([g["alpha_max"] for g in cells])
    a_lo = np.array([g["alpha_min"] for g in cells])
    inits = d["inits"]

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.fill_between(etas, a_lo, a_hi, color="#d62728", alpha=0.12,
                    label="multiplicity region")
    # Faint per-initialization endpoints.
    for j, a0 in enumerate(inits):
        ys = [d["grid"][f"{e:.2f}"]["by_init"][f"{a0:.2f}"] for e in etas]
        ax.plot(etas, ys, color="0.6", lw=0.7, alpha=0.5, zorder=1)
    ax.plot(etas, a_hi, "o-", color="#1f77b4", ms=6, zorder=3,
            label=r"high-awake equilibrium ($\alpha^{(0)}\geq0.3$)")
    ax.plot(etas, a_lo, "s--", color="#d62728", ms=6, zorder=3,
            label=r"low-awake equilibrium ($\alpha^{(0)}=0.1$)")
    ax.set_xlabel(r"interference coupling $\eta_I$")
    ax.set_ylabel(r"MFE awake density $\alpha^\star$")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Equilibrium bifurcation ({d['layout']}, $N$={d['N']})")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(loc="lower left", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_poa(in_path: str, out_path: str):
    """Two panels: (left) the coordination-collapse curve PoA vs
    interference coupling eta_I, with selfish vs social awake density;
    (right) the Pigouvian subsidy closing the gap at strong coupling."""
    with open(in_path) as f: sweep = json.load(f)
    eta_path = os.path.join(os.path.dirname(in_path), "poa_eta6.json")
    with open(eta_path) as f: tolld = json.load(f)

    etas = np.array(sweep["etas"])
    poa = np.array(sweep["poa"])
    a_mfe = np.array(sweep["alpha_mfe"]); a_mfc = np.array(sweep["alpha_mfc"])

    with plt.rc_context({"font.size": 16, "axes.labelsize": 18,
                          "axes.titlesize": 18, "xtick.labelsize": 14,
                          "ytick.labelsize": 14}):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9))
        # Left: PoA + awake densities vs eta
        c1 = "#d62728"
        ax1.plot(etas, poa, "o-", ms=7, lw=2, color=c1, label="price of anarchy")
        ax1.set_xlabel("interference coupling $\\eta_I$")
        ax1.set_ylabel("price of anarchy $J_{\\mathrm{MFE}}/J_{\\mathrm{MFC}}$",
                        color=c1)
        ax1.tick_params(axis="y", labelcolor=c1)
        ax1.axhline(1.0, ls=":", color="gray", lw=1)
        ax1.grid(True, ls="--", alpha=0.4)
        axb = ax1.twinx()
        axb.plot(etas, a_mfe, "s--", ms=5, color="#1f77b4", label="$\\alpha^*_{\\mathrm{MFE}}$ (selfish)")
        axb.plot(etas, a_mfc, "^--", ms=5, color="#2ca02c", label="$\\alpha^*_{\\mathrm{MFC}}$ (social)")
        axb.set_ylabel("awake density $\\alpha^*$")
        axb.set_ylim(0, 1.05)
        ax1.set_title("Coordination collapse")
        lines = ax1.get_lines()[:1] + axb.get_lines()
        ax1.legend(lines, [l.get_label() for l in lines], loc="center left",
                   fontsize=14, framealpha=0.95)

        # Right: subsidy sweep at eta=6
        tc = tolld["toll_curve"]
        taus = np.array([t["tau"] for t in tc])
        Js = np.array([t["J_social"] for t in tc])
        a_t = np.array([t["alpha_mfe"] for t in tc])
        order = np.argsort(-taus)   # from 0 to most negative
        subsidy = -taus[order]      # subsidy magnitude
        ax2.plot(subsidy, Js[order], "o-", ms=6, lw=2, color="#9467bd",
                 label="social cost $J$ under subsidy")
        ax2.axhline(tolld["J_mfc"], ls="--", color="#2ca02c", lw=1.6,
                    label=f"social optimum $J_{{\\mathrm{{MFC}}}}$={tolld['J_mfc']:.0f}")
        ax2.set_xlabel("Pigouvian wakefulness subsidy $|\\tau|$")
        ax2.set_ylabel("realized social cost $J$")
        ax2.set_yscale("log")
        ax2.set_title(f"Subsidy closes the gap ($\\eta_I$=6, PoA={tolld['poa']:.1f}$\\times$)")
        ax2.grid(True, ls="--", alpha=0.4, which="both")
        ax2.legend(loc="upper right", fontsize=14, framealpha=0.95)
        fig.tight_layout()
        fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_crossval(results_dir: str, out_path: str):
    """Cross-dataset robustness (Milan vs Shanghai): (left) the graphon
    MFE awake-density profile rises with traffic intensity on BOTH real
    topologies, while each scalar model collapses to one value; (right)
    the price of anarchy grows with the interference coupling on both,
    the coordination collapse of Theorem 4 reproducing across datasets."""
    def _load(name):
        p = os.path.join(results_dir, name)
        return json.load(open(p)) if os.path.exists(p) else None
    gm, gs = _load("graphon.json"), _load("graphon_shanghai.json")
    pm, ps = _load("poa_sweep.json"), _load("poa_sweep_shanghai.json")
    if not all([gm, gs, pm, ps]):
        print("  SKIP f10_crossval (missing cross-dataset results)"); return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.7))
    # Left: graphon awake-density profile, both datasets
    for d, c, mk, lab in [(gm, "#1f77b4", "o", "Milan"),
                           (gs, "#d62728", "s", "Shanghai")]:
        ab = np.array(d["alpha_blocks_star"]); B = d["B"]
        ax1.plot(np.arange(B), ab, marker=mk, ms=8, lw=2, color=c,
                 label=f"{lab} graphon $\\alpha^*_b$")
        ax1.axhline(d["scalar_alpha_star"], ls="--", lw=1.3, color=c, alpha=0.6)
    ax1.set_xticks(np.arange(gm["B"]))
    ax1.set_xticklabels([f"B{b}" for b in range(gm["B"])])
    ax1.set_xlabel("traffic-intensity block (low $\\to$ high)")
    ax1.set_ylabel("MFE awake density $\\alpha^*_b$")
    ax1.set_title("Non-flat profile (dashed = scalar MFE)")
    ax1.grid(True, ls="--", alpha=0.4)
    ax1.legend(loc="upper left", fontsize=9)
    # Right: PoA vs eta_I, both datasets
    for d, c, mk, lab in [(pm, "#1f77b4", "o", "Milan"),
                           (ps, "#d62728", "s", "Shanghai")]:
        ax2.plot(d["etas"], d["poa"], marker=mk, ms=7, lw=2, color=c,
                 label=f"{lab}")
    ax2.axhline(1.0, ls=":", color="gray", lw=1)
    ax2.set_xlabel("interference coupling $\\eta_I$")
    ax2.set_ylabel("price of anarchy $J_{\\mathrm{MFE}}/J_{\\mathrm{MFC}}$")
    ax2.set_title("Coordination collapse")
    ax2.grid(True, ls="--", alpha=0.4)
    ax2.legend(loc="upper left", fontsize=9)
    fig.suptitle("Cross-dataset replication on real topologies ($N$=100)",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_energy(in_path: str, out_path: str):
    """NetData power-model calibration: within-cell measured vs
    model-predicted power deviation, supporting the affine-in-awake,
    load-proportional running cost of the HJB. Annotates the in-sample
    and out-of-sample fit quality."""
    with open(in_path) as f: d = json.load(f)
    sc = d.get("scatter")
    if sc is None:
        print("  SKIP f11_energy (no scatter in energy_calib.json)"); return
    x = np.array(sc["Pd_pred"]); y = np.array(sc["Pd_meas"])
    fe = d["fixed_effects_fit"]; oos = d.get("out_of_sample", {})
    with plt.rc_context({"font.size": 15, "axes.labelsize": 17,
                          "axes.titlesize": 16, "xtick.labelsize": 14,
                          "ytick.labelsize": 14}):
        fig, ax = plt.subplots(figsize=(5.6, 4.4))
        ax.scatter(x, y, s=6, alpha=0.25, color="#1f77b4", edgecolors="none")
        lim = np.percentile(np.abs(np.concatenate([x, y])), 99)
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1.5, label="identity")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("model-predicted power deviation (W)")
        ax.set_ylabel("measured power deviation (W)")
        ax.set_title("NetData power-model calibration\n"
                     f"({d['n_records']:,} records, {d['n_cells']:,} cells)")
        txt = (f"within-cell $R^2$={fe['R2_within']:.2f}, MAPE={fe['MAPE_within_pct']:.1f}%\n"
               f"out-of-sample $R^2$={oos.get('R2_oos',0):.2f}, "
               f"MAPE={oos.get('MAPE_oos_pct',0):.1f}%\n"
               f"awake swing {fe['awake_swing_W']:.0f} W, $P_1$={fe['P1_load_slope_W']:.0f} W")
        ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top", ha="left",
                fontsize=13, bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
        ax.grid(True, ls="--", alpha=0.4)
        ax.legend(loc="lower right", fontsize=14, framealpha=0.5)
        fig.tight_layout()
        fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="sim/results")
    ap.add_argument("--fig_dir", default="fig")
    args = ap.parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)

    plans = [
        ("convergence.json", "f1_convergence.pdf", fig_convergence),
        ("eps_nash.json",    "f2_eps_nash.pdf",    fig_eps_nash),
        ("headline.json",    "f3_headline.pdf",    fig_headline),
        ("pareto.json",      "f4_pareto.pdf",      fig_pareto),
        ("comparative.json", "f5_comparative.pdf", fig_comparative),
        ("graphon.json",     "f7_graphon.pdf",     fig_graphon),
        ("poa_sweep.json",   "f8_poa.pdf",         fig_poa),
        ("bifurcation_eta.json", "f9_bifurcation.pdf", fig_bifurcation),
    ]
    for ifn, ofn, fn in plans:
        ipath = os.path.join(args.results_dir, ifn)
        opath = os.path.join(args.fig_dir, ofn)
        if os.path.exists(ipath):
            fn(ipath, opath)
        else:
            print(f"  SKIP {ofn} (missing {ipath})")

    # Cross-dataset robustness figure reads several result files at once.
    fig_crossval(args.results_dir, os.path.join(args.fig_dir, "f10_crossval.pdf"))
    ep = os.path.join(args.results_dir, "energy_calib.json")
    if os.path.exists(ep):
        fig_energy(ep, os.path.join(args.fig_dir, "f11_energy.pdf"))


if __name__ == "__main__":
    main()
