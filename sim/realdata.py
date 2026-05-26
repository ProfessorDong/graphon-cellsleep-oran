"""
realdata.py
-----------
Real-dataset loaders for the spatial topology and diurnal arrival
profile. Three sources are supported:

    Milan Telecom Big Data Challenge (2013):
        Harvard Dataverse DOI 10.7910/DVN/EGZHFV.
        Format: 100x100 grid covering ~10x10 km of Milan; per-cell
        per-10-min activity (SMS in/out, calls in/out, internet).

    Shanghai Telecom six-month session dataset (Yu et al. TMC 2018):
        Real base-station-level session traces with (lat, lon)
        coordinates and per-session timestamps; we use top-N busiest
        base stations in a 1-2 km^2 central window.

    NetMob 2023 Orange France challenge:
        Per-cell traffic over French metropolitan cities. Registration
        required at https://netmob2023challenge.networks.imdea.org/.
        Loader stub provided; downloads documented in README.

If the underlying data file is absent, the loader returns None and the
caller falls back to PPP (synthetic).
"""
from __future__ import annotations
import math
import os
import zipfile
from typing import Optional
import numpy as np

from .config import SimCfg


# Cache so repeated calls do not re-parse the 350 MB Milan file
_CACHE: dict = {}


# ============ Milan Telecom (Harvard Dataverse) ============
MILAN_GRID_SIDE = 100               # 100x100 grid
MILAN_AREA_KM_SIDE = 23.5           # Milan grid covers ~23.5 km on a side
MILAN_BINS_PER_DAY = 144            # 10-minute bins
MILAN_BIN_MS = 600_000


def load_milan(cfg: SimCfg, milan_path: Optional[str] = None) -> Optional[dict]:
    """Load Milan Telecom data and return:

        positions      : (N, 2) array of cell positions in km, centered at 0
        arrival_factor : (N, n_slots_episode) per-cell time-varying multiplier
                          (1.0 = mean activity for that cell over the day)
        cell_ids       : list of selected Milan grid cell IDs

    If milan_path is None, look at default sensing-green-ran data location.
    Returns None if data file is missing.
    """
    key = ("milan", cfg.topo.N, cfg.topo.area_km2,
            int(cfg.time.T_episode_s * 1000))
    if key in _CACHE:
        return _CACHE[key]

    if milan_path is None:
        candidates = [
            os.path.join(cfg.data_dir, "milan_2013-11-04.txt"),
            "../sensing-green-ran/sim/data/milan_2013-11-04.txt",
            "/home/dong/Workspace/WritePaper/ISAC_ORAN/sensing-green-ran/"
            "sim/data/milan_2013-11-04.txt",
        ]
        milan_path = next((p for p in candidates if os.path.exists(p)
                            and os.path.getsize(p) > 1024), None)
    if milan_path is None or not os.path.exists(milan_path):
        return None

    # ----- Pick central N-cell subgrid -----
    N = cfg.topo.N
    side = int(math.ceil(math.sqrt(N)))
    half = side // 2
    cx, cy = MILAN_GRID_SIDE // 2, MILAN_GRID_SIDE // 2
    r0, r1 = cx - half, cx - half + side
    c0, c1 = cy - half, cy - half + side
    target_ids = set()
    id_to_pos = {}                          # cell_id -> (row, col)
    km_per_grid = MILAN_AREA_KM_SIDE / MILAN_GRID_SIDE
    for r in range(r0, r1):
        for c in range(c0, c1):
            if 0 <= r < MILAN_GRID_SIDE and 0 <= c < MILAN_GRID_SIDE:
                cid = r * MILAN_GRID_SIDE + c + 1
                target_ids.add(cid)
                id_to_pos[cid] = (r, c)
    target_ids = list(target_ids)[:N]
    if len(target_ids) < N:
        return None

    # ----- Parse the file, accumulate per-cell hourly bins -----
    cell_to_bin_total = {cid: np.zeros(MILAN_BINS_PER_DAY) for cid in target_ids}
    t0_ms = None
    with open(milan_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            try:
                cid = int(parts[0]); ts = int(parts[1])
            except ValueError:
                continue
            if cid not in cell_to_bin_total:
                continue
            if t0_ms is None:
                t0_ms = ts
            bin_idx = ((ts - t0_ms) // MILAN_BIN_MS) % MILAN_BINS_PER_DAY
            # Internet activity is the last column; SMS+calls are earlier.
            # Sum SMS in/out + Calls in/out + Internet for total activity.
            tot = 0.0
            for v in parts[3:]:
                v = v.strip()
                if v:
                    try:
                        tot += float(v)
                    except ValueError:
                        pass
            cell_to_bin_total[cid][bin_idx] += tot

    # Per-cell time-varying multiplier: normalize so each cell's mean = 1.
    # We also keep each cell's *absolute* daily activity (cell_intensity)
    # so downstream code can build traffic-intensity blocks.
    n_slots = cfg.n_slots_episode
    arrival_factor = np.zeros((N, n_slots))
    positions = np.zeros((N, 2))
    cell_intensity = np.zeros(N)
    for i, cid in enumerate(target_ids):
        r, c = id_to_pos[cid]
        # Position in km, centered at 0
        positions[i, 0] = (c - MILAN_GRID_SIDE / 2) * km_per_grid
        positions[i, 1] = (r - MILAN_GRID_SIDE / 2) * km_per_grid
        # Diurnal profile: interp from 144 10-min bins to n_slots
        bins = cell_to_bin_total[cid]
        cell_intensity[i] = float(bins.sum())
        if bins.sum() <= 1e-9:
            arrival_factor[i] = 1.0
        else:
            x_in = np.linspace(0.0, 1.0, MILAN_BINS_PER_DAY)
            x_out = np.linspace(0.0, 1.0, n_slots)
            interp = np.interp(x_out, x_in, bins)
            arrival_factor[i] = interp / max(np.mean(interp), 1e-9)
    # Relative intensity (mean 1): cell's busyness vs the cluster average
    if cell_intensity.sum() > 0:
        cell_intensity = cell_intensity / max(cell_intensity.mean(), 1e-9)
    else:
        cell_intensity = np.ones(N)

    # Rescale positions to cfg.topo.area_km2
    bb = float(np.max(np.abs(positions)))
    if bb > 0:
        L = math.sqrt(cfg.topo.area_km2)
        positions = positions / (2 * bb) * L

    out = {"positions": positions, "arrival_factor": arrival_factor,
            "cell_intensity": cell_intensity,
            "cell_ids": target_ids, "source": "Milan Telecom 2013-11-04"}
    _CACHE[key] = out
    return out


# ============ Shanghai Telecom (six-month session dataset) ============
def load_shanghai(cfg: SimCfg, shanghai_dir: Optional[str] = None) -> Optional[dict]:
    """Load Shanghai Telecom and return positions + arrival_factor.

    The dataset ships as 12 .xlsx files (one per half-month). We use
    the busiest N base stations from data_6.1~6.15.xlsx. A compact
    cache (top_cells_N.npz) is written so subsequent runs avoid
    re-parsing the 200 MB xlsx.
    """
    if shanghai_dir is None:
        candidates = [
            os.path.join(cfg.data_dir, "shanghai_telecom"),
            "../safe-rl-oran/sim/data/shanghai_telecom",
            "/home/dong/Workspace/WritePaper/ISAC_ORAN/safe-rl-oran/"
            "sim/data/shanghai_telecom",
        ]
        shanghai_dir = next((p for p in candidates if os.path.isdir(p)), None)
    if shanghai_dir is None:
        return None

    N = cfg.topo.N
    n_slots = cfg.n_slots_episode
    cache_npz = os.path.join(shanghai_dir, f"top_cells_N{N}.npz")
    if os.path.exists(cache_npz):
        z = np.load(cache_npz)
        lats, lons, hourly = z["lats"], z["lons"], z["hourly"]
    else:
        xlsx = next((os.path.join(shanghai_dir, f)
                      for f in os.listdir(shanghai_dir)
                      if f.startswith("data_6.1~") and f.endswith(".xlsx")), None)
        if xlsx is None:
            return None
        try:
            import pandas as pd
        except ImportError:
            return None
        df = pd.read_excel(xlsx, engine="openpyxl",
                            usecols=["start time", "latitude", "longitude"])
        # Group by (lat, lon) to find unique base-station cells
        cell_counts = (df.groupby(["latitude", "longitude"]).size()
                          .reset_index(name="n").sort_values("n", ascending=False))
        if len(cell_counts) < N:
            return None
        top = cell_counts.head(N).copy()
        # Hourly diurnal profile per top cell
        df["hour"] = df["start time"].dt.hour
        df_top = df[df.set_index(["latitude", "longitude"]).index.isin(
            top.set_index(["latitude", "longitude"]).index)]
        hourly_df = (df_top.groupby(["latitude", "longitude", "hour"])
                              .size().unstack(fill_value=0))
        # Reindex to match top order, columns 0..23
        hourly_df = hourly_df.reindex(
            list(zip(top["latitude"], top["longitude"])))
        hourly_df = hourly_df.reindex(columns=range(24), fill_value=0)
        lats = top["latitude"].to_numpy()
        lons = top["longitude"].to_numpy()
        hourly = hourly_df.to_numpy(dtype=np.float64)
        np.savez_compressed(cache_npz, lats=lats, lons=lons, hourly=hourly)

    # Convert (lat, lon) to km coordinates centered at the mean
    lat0 = float(np.mean(lats)); lon0 = float(np.mean(lons))
    x_km = (lons - lon0) * 111.320 * math.cos(math.radians(lat0))
    y_km = (lats - lat0) * 110.574
    positions = np.column_stack([x_km, y_km])
    bb = float(np.max(np.abs(positions)))
    if bb > 0:
        L = math.sqrt(cfg.topo.area_km2)
        positions = positions / (2 * bb) * L

    # Build arrival_factor: per-cell 24-hour profile, interpolated to
    # the simulator's n_slots and normalized so each cell's mean = 1
    arrival_factor = np.zeros((N, n_slots))
    x_in = np.linspace(0.0, 1.0, 24)
    x_out = np.linspace(0.0, 1.0, n_slots)
    for i in range(N):
        prof = hourly[i]
        if prof.sum() < 1e-9:
            arrival_factor[i] = 1.0
            continue
        interp = np.interp(x_out, x_in, prof)
        arrival_factor[i] = interp / max(np.mean(interp), 1e-9)

    return {"positions": positions, "arrival_factor": arrival_factor,
             "source": "Shanghai Telecom six-month session dataset",
             "top_lat_lon": np.column_stack([lats, lons])}


# ============ NetMob 2023 Orange France ============
def load_netmob(cfg: SimCfg, netmob_dir: Optional[str] = None) -> Optional[dict]:
    """Load NetMob 2023 Orange France data.

    Dataset distribution requires registration at
    https://netmob2023challenge.networks.imdea.org/. The challenge
    organizers provide ~80 cities with per-cell hourly traffic and
    cell GPS positions for a six-month observation window.

    Expected file layout under cfg.data_dir/netmob2023/:
        <city>/cells.csv           (cell_id, lat, lon, area_km2)
        <city>/traffic_<date>.csv  (cell_id, hour, dl_bytes, ul_bytes)

    If the data is not present, returns None.
    """
    if netmob_dir is None:
        netmob_dir = os.path.join(cfg.data_dir, "netmob2023")
    if not os.path.isdir(netmob_dir):
        return None
    # Pick the first city subdirectory available
    cities = sorted([d for d in os.listdir(netmob_dir)
                      if os.path.isdir(os.path.join(netmob_dir, d))])
    if not cities:
        return None
    city_dir = os.path.join(netmob_dir, cities[0])
    cells_csv = os.path.join(city_dir, "cells.csv")
    if not os.path.exists(cells_csv):
        return None

    # Load cell positions
    cells = np.genfromtxt(cells_csv, delimiter=",", skip_header=1)
    if cells.ndim != 2 or cells.shape[0] < cfg.topo.N:
        return None
    # Sort by some measure of activity if available; otherwise take first N
    cells = cells[:cfg.topo.N]
    lat = cells[:, 1]; lon = cells[:, 2]
    lat0 = float(np.mean(lat)); lon0 = float(np.mean(lon))
    x_km = (lon - lon0) * 111.320 * math.cos(math.radians(lat0))
    y_km = (lat - lat0) * 110.574
    positions = np.column_stack([x_km, y_km])
    bb = float(np.max(np.abs(positions)))
    if bb > 0:
        L = math.sqrt(cfg.topo.area_km2)
        positions = positions / (2 * bb) * L
    n_slots = cfg.n_slots_episode
    arrival_factor = np.ones((cfg.topo.N, n_slots))
    return {"positions": positions, "arrival_factor": arrival_factor,
             "source": f"NetMob 2023 Orange France ({cities[0]})"}


# ============ Unified dispatcher ============
def load_layout_and_arrivals(cfg: SimCfg) -> Optional[dict]:
    """Try the requested layout's real-data loader; return None if the
    data file is absent. Caller falls back to a synthetic layout."""
    layout = cfg.topo.layout
    if layout == "milan":
        return load_milan(cfg)
    if layout == "shanghai":
        return load_shanghai(cfg)
    if layout == "netmob":
        return load_netmob(cfg)
    return None
