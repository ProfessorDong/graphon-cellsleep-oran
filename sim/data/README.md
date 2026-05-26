# Real datasets for the MFG cell-sleep simulator

Three real datasets are supported. The simulator's loader detects which
files are present and dispatches accordingly. If a dataset is missing the
loader returns `None` and the caller falls back to PPP.

## Milan Telecom Big Data Challenge (2013)

- **Source**: Harvard Dataverse, DOI `10.7910/DVN/EGZHFV`.
- **Files used**: `milan_2013-11-04.txt` (Monday) and `milan_2013-11-15.txt`
  (Friday). Each file is ~350 MB and encodes per-cell per-10-minute SMS,
  call, and Internet activity over a 100x100 grid covering ~10x10 km.
- **Default location**: copy or symlink to
  `mfg-cell-sleep/sim/data/milan_2013-11-04.txt`. The loader also
  auto-detects the same file under `../sensing-green-ran/sim/data/`.
- **Loader output**: `(N, 2)` cell positions in km (central N-cell
  subgrid of the 100x100) plus an `(N, n_slots)` per-cell diurnal
  factor (mean 1.0 per cell).

## Shanghai Telecom six-month session dataset

- **Source**: Yu et al., "Provisioning of adaptive scalable bandwidth
  with adaptive request-aware planning", IEEE Trans. Mobile Comput.,
  2018. Available at the authors' Tianchi repository.
- **Files used**: 12 half-month `.xlsx` files; we use
  `data_6.1~6.15.xlsx` for June 1-15 2014 (~220 MB unpacked).
- **Default location**: extract the zip to
  `mfg-cell-sleep/sim/data/shanghai_telecom/`. The loader also
  auto-detects the same directory under `../safe-rl-oran/sim/data/`.
- **Loader output**: top-N busiest base stations as `(N, 2)` positions
  plus `(N, n_slots)` real per-cell hourly diurnal profile. A compressed
  `top_cells_N{N}.npz` cache is written on first load so subsequent
  runs avoid re-parsing the xlsx (~14 s -> <0.1 s).

## NetMob 2023 Orange France challenge

- **Source**: https://netmob2023challenge.networks.imdea.org/.
  Registration required; participants receive per-cell 15-minute
  uplink and downlink traffic over ~80 French cities and 6 months.
- **Default layout (expected)**:
  ```
  mfg-cell-sleep/sim/data/netmob2023/<city>/cells.csv
  mfg-cell-sleep/sim/data/netmob2023/<city>/traffic_<date>.csv
  ```
- **Status**: the loader is implemented in `sim/realdata.py` but
  the dataset is not redistributed; supply your own copy after
  registering. If absent, the loader returns `None` and the simulator
  falls back to PPP.

## Reproducibility

All loaders are deterministic given the input files. The simulator
caches parsed cell positions and hourly profiles. No randomized
processing is applied to the trace data itself; per-seed reproducibility
comes from `numpy.random.SeedSequence(20260714)` in `sim/config.py`.
