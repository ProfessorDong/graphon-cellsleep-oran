"""
topology.py
-----------
Spatial layout generators and Voronoi-style association maps for the
N-cell cluster. Layouts:
  hex   : regular hexagonal grid scaled to fill area_km2
  ppp   : homogeneous Poisson point process at density N/area_km2
  milan : real cell positions from Milan Telecom dataset (Phase 4 loader)
  netmob: real cell positions from NetMob 2023 Orange France (Phase 4 loader)

Every layout returns positions in km coordinates inside a window of
size sqrt(area_km2) x sqrt(area_km2) centered at the origin.
"""
from __future__ import annotations
import math
import os
import numpy as np

from .config import SimCfg


# ---------------------------------------------------------------------------
def hex_grid(N: int, area_km2: float = 1.0) -> np.ndarray:
    """Regular hexagonal grid of N points roughly filling area_km2.

    Returns: array of shape (N, 2) with (x, y) coordinates in km.
    """
    L = math.sqrt(area_km2)
    # Choose row/col counts so we get >= N hex sites
    nrows = max(2, int(math.ceil(math.sqrt(N * 1.155))))
    ncols = max(2, int(math.ceil(N / nrows)))
    # Spacing scaled so the bounding box is L x L
    dx = L / ncols
    dy = dx * math.sqrt(3) / 2.0
    pts = []
    for r in range(nrows):
        for c in range(ncols):
            x = c * dx + (dx / 2.0 if r % 2 else 0.0)
            y = r * dy
            pts.append((x, y))
    pts = np.array(pts, dtype=np.float64)
    # Re-center to origin
    pts -= pts.mean(axis=0)
    # Truncate / pad to exactly N (we asked for >= N)
    if len(pts) >= N:
        # Keep N closest to center
        d2 = (pts ** 2).sum(axis=1)
        idx = np.argsort(d2)[:N]
        return pts[idx]
    return pts


def ppp_layout(N: int, area_km2: float = 1.0, seed: int = 1) -> np.ndarray:
    """Homogeneous PPP layout: N uniform i.i.d. points in [-L/2, L/2]^2."""
    L = math.sqrt(area_km2)
    rng = np.random.default_rng(seed)
    pts = rng.uniform(low=-L / 2, high=L / 2, size=(N, 2))
    return pts


def voronoi_areas_km2(positions: np.ndarray, area_km2: float = 1.0,
                       n_samples: int = 20_000, seed: int = 0) -> np.ndarray:
    """Estimate per-cell Voronoi areas (in km^2) by Monte-Carlo sampling.

    Each sample point is assigned to its nearest cell; the per-cell
    fraction of samples is the estimated area share.
    """
    L = math.sqrt(area_km2)
    rng = np.random.default_rng(seed)
    samples = rng.uniform(low=-L / 2, high=L / 2, size=(n_samples, 2))
    # Nearest-neighbor: argmin over distances
    # For modest N this is fast enough with numpy broadcasting.
    d2 = ((samples[:, None, :] - positions[None, :, :]) ** 2).sum(axis=-1)
    nearest = d2.argmin(axis=1)
    counts = np.bincount(nearest, minlength=positions.shape[0])
    return counts.astype(np.float64) / n_samples * area_km2


def neighbor_set(positions: np.ndarray, k: int = 6) -> np.ndarray:
    """k-nearest-neighbor index array. Returns (N, k) integer matrix
    where row i lists the k closest cells (excluding i itself)."""
    N = positions.shape[0]
    d2 = ((positions[:, None, :] - positions[None, :, :]) ** 2).sum(axis=-1)
    np.fill_diagonal(d2, np.inf)
    return np.argsort(d2, axis=1)[:, :k]


# ---------------------------------------------------------------------------
def load_real_topology(cfg: SimCfg) -> np.ndarray | None:
    """Try to load a real-cell-position CSV from cfg.data_dir.

    Returns positions in km coordinates centered at the origin, scaled
    to cfg.topo.area_km2, or None if the file is missing.

    CSV format: 'lat,lon' or 'x_km,y_km'. The loader auto-detects.
    """
    fname_map = {
        "milan":  cfg.arr.milan_csv,
        "netmob": cfg.arr.netmob_csv,
    }
    layout = cfg.topo.layout
    if layout not in fname_map:
        return None
    path = os.path.join(cfg.data_dir, fname_map[layout])
    if not os.path.exists(path):
        return None
    with open(path) as f:
        header = f.readline().strip().lower()
    if "lat" in header and "lon" in header:
        data = np.genfromtxt(path, delimiter=",", skip_header=1)
        # Convert (lat, lon) deg -> approx km coordinates near center
        lat = data[:, 0]
        lon = data[:, 1]
        lat0, lon0 = lat.mean(), lon.mean()
        x_km = (lon - lon0) * 111.320 * math.cos(math.radians(lat0))
        y_km = (lat - lat0) * 110.574
        pts = np.column_stack([x_km, y_km])
    else:
        pts = np.genfromtxt(path, delimiter=",", skip_header=1)[:, :2]
    pts -= pts.mean(axis=0)
    # Rescale to the requested area_km2 (preserve aspect by isotropic scale)
    bb = np.max(np.abs(pts))
    if bb <= 0:
        return None
    L = math.sqrt(cfg.topo.area_km2)
    pts = pts / (2 * bb) * L
    # Sub-sample / pad to N
    N = cfg.topo.N
    if pts.shape[0] >= N:
        d2 = (pts ** 2).sum(axis=1)
        idx = np.argsort(d2)[:N]
        return pts[idx]
    return pts


def make_layout(cfg: SimCfg) -> np.ndarray:
    """Dispatch by cfg.topo.layout; fall back to PPP if real-data file
    is unavailable. Returns positions in km coordinates."""
    layout = cfg.topo.layout
    if layout == "hex":
        return hex_grid(cfg.topo.N, cfg.topo.area_km2)
    if layout == "ppp":
        return ppp_layout(cfg.topo.N, cfg.topo.area_km2, cfg.topo.seed_layout)
    # Real-data layouts use the realdata.py loaders
    from .realdata import load_layout_and_arrivals
    bundle = load_layout_and_arrivals(cfg)
    if bundle is not None:
        return bundle["positions"]
    # Last-resort fallback to PPP
    return ppp_layout(cfg.topo.N, cfg.topo.area_km2, cfg.topo.seed_layout)


def make_layout_and_arrivals(cfg: SimCfg):
    """Return (positions, arrival_factor) where arrival_factor has shape
    (N, n_slots) and is None for synthetic layouts."""
    layout = cfg.topo.layout
    if layout in ("hex", "ppp"):
        return make_layout(cfg), None
    from .realdata import load_layout_and_arrivals
    bundle = load_layout_and_arrivals(cfg)
    if bundle is not None:
        return bundle["positions"], bundle["arrival_factor"]
    # Fall back to synthetic
    return ppp_layout(cfg.topo.N, cfg.topo.area_km2, cfg.topo.seed_layout), None
