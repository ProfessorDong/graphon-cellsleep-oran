"""
mean_field.py
-------------
Mean-field coupling functions and aggregate statistics matching
eqs. (service_mf), (lambda_mf) in main.tex:

  psi(alpha)         = 1 / (1 + eta_I (1 - alpha))           [interference]
  lambda_i(m)        = lambda_0(z_i) / (alpha + eps_lambda)   [load shifting]
  mu(q, e, u, m)     = e * psi(alpha) * mu_max * sigmoid(q)   [service rate]

The mean field enters the HJB ONLY through the scalar awake density
alpha(t), which makes the projection (Lemma 1) valid.
"""
from __future__ import annotations
import numpy as np

from .config import SimCfg


def psi_interference(alpha: float, eta_I: float) -> float:
    """Service-rate reducing interference coupling psi(alpha) in (0, 1].

    psi(alpha) = 1 / (1 + eta_I * alpha)
    -> psi(0) = 1            (no awake neighbors -> no interference)
    -> psi(1) = 1/(1+eta_I)  (all awake -> heavy interference)

    Decreasing in alpha, as required by the aversive-coupling regime of
    Remark `rem:mono`: more awake neighbors strictly *harms* a
    representative agent's effective service rate.
    """
    return 1.0 / (1.0 + eta_I * alpha)


def lambda_load_shift(lambda0: float, alpha: float, eps: float) -> float:
    """Load-shifted arrival rate at a cell, given baseline lambda0 and
    population awake density alpha.

    lambda_i(m) = lambda_0 / (alpha + eps_lambda)
    -> When most neighbors sleep (alpha small), this cell absorbs their
       load: lambda grows.
    -> When alpha = 1, lambda equals lambda_0 / (1+eps).
    """
    return lambda0 / (alpha + eps)


def service_rate(q: np.ndarray, e: np.ndarray, alpha: float,
                 mu_max: float, eta_I: float,
                 sigmoid_scale: float = 8.0) -> np.ndarray:
    """Per-cell service rate.

    mu_i = e_i * psi(alpha) * mu_max * tanh-shape in q_i

    The q-dependence reflects the fact that a server with empty buffer
    cannot push out more than what arrives, so the effective rate is
    capped near q=0. We use a smooth sigmoid q/(q+s) shape.
    """
    psi = psi_interference(alpha, eta_I)
    util = q / (q + sigmoid_scale)  # smooth on/off as q grows from 0
    return e * psi * mu_max * util


def awake_density(e: np.ndarray) -> float:
    """alpha(t) = mean of e over cells (=integral of e against the
    empirical mean-field measure)."""
    return float(np.mean(e))


def diurnal_modulation(t_s: float, cfg: SimCfg) -> float:
    """Diurnal arrival multiplier in [1-amp, 1+amp]. Peaks at the busy
    hour fraction of a 24-hour day. We map cfg.time.T_episode_s onto
    one diurnal cycle, so a 60s episode captures one full day."""
    T = cfg.time.T_episode_s
    phase = 2.0 * np.pi * (t_s / T - cfg.arr.busy_hour_frac)
    return 1.0 + cfg.arr.diurnal_amp * np.cos(phase)


# ===================================================================
#  Block-graphon (multi-population) coupling
# ===================================================================
#  The symmetric model couples every cell to the same scalar awake
#  density alpha = mean(e). The graphon model partitions cells into B
#  blocks and couples a block-a cell to a LOCAL field
#      abar_a = sum_b W[a,b] * alpha_b
#  where W is a row-stochastic block-interaction kernel and alpha_b is
#  the awake density of block b. B=1 (or a uniform W) recovers the
#  scalar model. A piecewise-constant graphon is exactly this block
#  (stochastic-block-model) structure.
# ===================================================================

def block_partition(intensity: np.ndarray, B: int) -> np.ndarray:
    """Partition N cells into B blocks by an intensity score (e.g.,
    per-cell traffic load or local spatial density). Block 0 is the
    lowest-intensity quantile, block B-1 the highest.

    Returns: integer block label array of shape (N,) in {0,...,B-1}.
    """
    N = len(intensity)
    order = np.argsort(intensity)
    labels = np.zeros(N, dtype=int)
    edges = np.linspace(0, N, B + 1).astype(int)
    for b in range(B):
        labels[order[edges[b]:edges[b + 1]]] = b
    return labels


def local_density(positions: np.ndarray, radius_km: float) -> np.ndarray:
    """Per-cell local density = number of other cells within radius_km."""
    d2 = ((positions[:, None, :] - positions[None, :, :]) ** 2).sum(-1)
    return (np.sqrt(d2) < radius_km).sum(axis=1).astype(float) - 1.0


def block_graphon_kernel(positions: np.ndarray, labels: np.ndarray,
                          B: int, radius_km: float) -> np.ndarray:
    """Row-stochastic block-interaction kernel.

    W[a,b] = average fraction of a block-a cell's interference
    neighbors (cells within radius_km) that belong to block b.

    Each row sums to 1, so abar_a = sum_b W[a,b] alpha_b is a weighted
    average of neighbor awake densities. Spatially segregated blocks
    give a near-diagonal W (a core cell mostly sees core neighbors);
    well-mixed blocks give near-uniform rows (recovering the scalar
    model).
    """
    N = positions.shape[0]
    d = np.sqrt(((positions[:, None, :] - positions[None, :, :]) ** 2).sum(-1))
    within = (d < radius_km)
    np.fill_diagonal(within, False)
    W = np.zeros((B, B))
    for a in range(B):
        cells_a = np.where(labels == a)[0]
        if len(cells_a) == 0:
            W[a, a] = 1.0
            continue
        # For each block-a cell, fraction of its neighbors in each block
        rows = []
        for i in cells_a:
            nbrs = np.where(within[i])[0]
            if len(nbrs) == 0:
                frac = np.zeros(B); frac[a] = 1.0
            else:
                frac = np.bincount(labels[nbrs], minlength=B) / len(nbrs)
            rows.append(frac)
        W[a] = np.mean(rows, axis=0)
        if W[a].sum() <= 1e-9:
            W[a, a] = 1.0
        else:
            W[a] /= W[a].sum()
    return W


def local_field(alpha_blocks: np.ndarray, W: np.ndarray) -> np.ndarray:
    """abar_a = sum_b W[a,b] alpha_b for all blocks a. Shapes:
    alpha_blocks (B,), W (B,B) -> returns (B,)."""
    return W @ alpha_blocks
