"""
config.py
---------
Single source of truth for all MFG cell-sleep simulator parameters.
Mirrors the symbol table of `mfg-cell-sleep/main.tex`.

Canonical seed sequence for Paper 5: numpy.random.SeedSequence(20260714),
distinct from Paper 3 (20260517) and Paper 4 (20260601).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


# ========== topology ==========
@dataclass
class TopoCfg:
    N: int = 200                       # number of small cells
    layout: str = "hex"                # "hex" | "ppp" | "milan" | "netmob"
    area_km2: float = 1.0              # observation window (1 km^2 default)
    seed_layout: int = 1               # PPP layout reproducibility


# ========== time ==========
@dataclass
class TimeCfg:
    dt_s: float = 0.1                  # SDE Euler step, s (= 100 ms control slot)
    T_horizon_s: float = 60.0          # finite-horizon variant horizon
    T_episode_s: float = 60.0          # evaluation episode length
    bc_period_s: float = 0.1           # mean-field broadcast cadence


# ========== arrivals ==========
@dataclass
class ArrivalsCfg:
    lambda0_pdu_per_s: float = 15.0    # per-cell baseline arrival rate (PDU/s)
    sigma_q: float = 2.0               # queue diffusion (Brownian volatility)
    diurnal_amp: float = 0.40          # +/- amplitude over the day
    busy_hour_frac: float = 0.83       # peak at 20:00 = 0.83 of day
    hotspot_mult: float = 5.0          # hotspot intensity multiplier
    hotspot_frac: float = 0.0          # fraction of area covered by hotspots
    burst_prob: float = 0.04           # Pareto burst event probability per slot
    pareto_shape: float = 2.5
    burst_scale: float = 1.5
    # Real-trace anchors (optional; activated by topology choice)
    milan_csv: str = "milan_cells.csv"            # Milan cell positions
    netmob_csv: str = "netmob2023_cells.csv"      # NetMob 2023 Orange France
    shanghai_csv: str = "shanghai_cells.csv"      # Shanghai Telecom (fallback)


# ========== channel / service ==========
@dataclass
class ChannelCfg:
    mu_max_pdu_per_s: float = 80.0     # peak service rate per cell (PDU/s)
    eta_I: float = 1.2                 # interference sensitivity
    epsilon_lambda: float = 0.05       # small constant to keep arrival rate finite


# ========== energy (EARTH-style) ==========
@dataclass
class EnergyCfg:
    P_sleep_W: float = 8.0             # deep-sleep floor (always on, even at e=0)
    P0_W: float = 130.0                # idle-awake floor (when e=1, no traffic)
    P1_W: float = 100.0                # dynamic load-proportional amplifier
    P2_W: float = 5.0                  # slew penalty (per u^2)
    sigma_e: float = 0.05              # energy-mode jitter
    u_bar: float = 0.5                 # max slew rate per second


# ========== cost prices ==========
@dataclass
class CostCfg:
    c_q: float = 2.0                   # queue (delay) price
    c_p: float = 0.5                   # power price (smaller than c_q so sleeping pays off)
    c_s: float = 5.0                   # switching (slew) price (high to discourage chatter)
    rho: float = 0.1                   # discount rate, /s
    toll: float = 0.0                  # Pigouvian mean-field toll tau on wakefulness
                                       # (adds tau*e to the per-agent running cost)


# ========== algorithm / solver ==========
@dataclass
class SolverCfg:
    # Forward-backward HJB-FP grid solver
    N_q: int = 64                      # queue-grid points
    N_e: int = 17                      # energy-mode-grid points (relaxed [0,1])
    N_t: int = 60                      # time-grid points (with T_horizon_s)
    q_max_pdu: float = 300.0           # queue grid upper bound
    kappa: float = 0.3                 # mean-field damping
    K_max_outer: int = 80              # outer iterations
    tol_outer: float = 1e-4            # convergence tolerance for ||alpha^(n+1)-alpha^(n)||_inf
    # Parametric MFPI
    pmfpi_M_traj: int = 64             # trajectories per gradient estimate
    pmfpi_lr: float = 5e-3
    pmfpi_K: int = 100
    pmfpi_n_params: int = 5            # linear class: [theta_0..theta_4]
    # MFAC actor-critic
    mfac_hidden: int = 32              # MLP hidden width for actor/critic
    mfac_lr_actor: float = 1e-4
    mfac_lr_critic: float = 1e-3
    mfac_gamma: float = 0.99           # discount factor for the cost MDP
    mfac_rollout_slots: int = 128
    mfac_K_rounds: int = 100
    mfac_kappa: float = 0.1


# ========== independent-threshold baseline ==========
@dataclass
class IndepThCfg:
    q_th_low_pdu: float = 8.0          # sleep when q < q_th_low
    q_th_high_pdu: float = 40.0        # wake when q > q_th_high


# ========== top-level bundle ==========
@dataclass
class SimCfg:
    topo: TopoCfg = field(default_factory=TopoCfg)
    time: TimeCfg = field(default_factory=TimeCfg)
    arr: ArrivalsCfg = field(default_factory=ArrivalsCfg)
    chan: ChannelCfg = field(default_factory=ChannelCfg)
    energy: EnergyCfg = field(default_factory=EnergyCfg)
    cost: CostCfg = field(default_factory=CostCfg)
    solver: SolverCfg = field(default_factory=SolverCfg)
    indep_th: IndepThCfg = field(default_factory=IndepThCfg)
    data_dir: str = "sim/data"
    results_dir: str = "sim/results"

    @property
    def n_slots_episode(self) -> int:
        return int(round(self.time.T_episode_s / self.time.dt_s))

    @property
    def n_slots_horizon(self) -> int:
        return int(round(self.time.T_horizon_s / self.time.dt_s))


def default_cfg(**overrides) -> SimCfg:
    cfg = SimCfg()
    for k, v in overrides.items():
        if "." in k:
            head, tail = k.split(".", 1)
            setattr(getattr(cfg, head), tail, v)
        else:
            setattr(cfg, k, v)
    return cfg


# Canonical seed sequence for Paper 5
CANONICAL_SEED = 20260714


def canonical_seeds(n: int = 10):
    rng = np.random.SeedSequence(CANONICAL_SEED)
    return [int(s) for s in rng.generate_state(n)]
