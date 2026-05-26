# graphon-cellsleep-oran

Reproducibility code for the paper

> **A Graphon Mean-Field Game for Distributed Cell Sleep in Ultra-Dense O-RAN**
> Liang Dong, Department of Electrical and Computer Engineering, Baylor University.

The repository contains the trace-driven simulator, the equilibrium and
social-planner solvers, the experiment drivers, and the figure-generation
scripts used in the paper. The raw third-party traces (Milan, Shanghai,
NetData) are **not** redistributed here; the loaders point to their original
public sources (see [Datasets](#datasets)).

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Only NumPy and Matplotlib are required for the synthetic and figure pipelines;
`openpyxl` is needed only to parse the Shanghai Telecom `.xlsx` traces.

## Repository layout

```
sim/
  config.py        canonical parameters and SeedSequence
  topology.py      cell layouts (PPP, grid, and real Milan/Shanghai positions)
  dynamics.py      per-cell queue/energy dynamics and the EARTH-style power model
  mean_field.py    mean-field couplings (interference, load-shift, service)
  metrics.py       energy, delay, drop, and welfare metrics
  baselines.py     always-on / threshold / random-sleep baselines
  realdata.py      Milan / Shanghai / NetData loaders (return None if absent)
  solvers/
    fb_pde.py      forward-backward HJB-FP finite-difference equilibrium solver
    graphon.py     block-graphon (multi-population) MFG coupling and solver
    pmfpi.py       projected mean-field policy iteration
    mfac.py        safety-shielded mean-field actor-critic
  run_*.py         one driver per experiment (writes sim/results/*.json)
  make_figures.py  builds fig/*.pdf from the result JSONs
fig/               publication figures (PDF, 600 dpi, type-42 fonts)
sim/results/       cached experiment outputs (JSON)
sim/data/          dataset download/preprocessing instructions (no raw data)
```

## Reproducing the results

Run the drivers as modules from the repository root; each writes a JSON file
under `sim/results/`.

```bash
python -m sim.run_convergence     # FB-PDE convergence            -> f1_convergence
python -m sim.run_eps_nash        # epsilon-Nash gap vs N         -> f2_eps_nash
python -m sim.run_headline        # headline energy reduction     -> f3_headline
python -m sim.run_pareto          # energy-delay Pareto front     -> f4_pareto
python -m sim.run_comparative     # vs baselines (Milan/Shanghai) -> f5_comparative
python -m sim.run_graphon         # spatial awake-density profile -> f7_graphon
python -m sim.run_poa_sweep       # price of anarchy + toll       -> f8_poa
python -m sim.run_bifurcation     # multiplicity / phase change   -> f9_bifurcation
python -m sim.run_energy_calib    # NetData power-model calibration
python -m sim.run_safe_mfac       # safety-shielded MFAC

python -m sim.make_figures        # rebuild all fig/*.pdf from sim/results/*.json
```

The result JSONs shipped in `sim/results/` were produced with the canonical
seed in `sim/config.py`; rerunning a driver overwrites its JSON in place.
Experiments that need a real trace fall back to a Poisson point process when
the corresponding dataset is absent, so the synthetic-topology results are
reproducible without downloading any data.

## Datasets

The energy calibration and the real-topology experiments use three public
datasets. None are redistributed here; download them from the sources below
and place them as described in [`sim/data/README.md`](sim/data/README.md).

- **Milan Telecom Big Data Challenge (2013)** — Harvard Dataverse, DOI
  `10.7910/DVN/EGZHFV` (Barlacchi et al., *Sci. Data*, 2015).
- **Shanghai Telecom six-month session dataset** — Wang et al., *IEEE Trans.
  Mobile Comput.*, 2021 (authors' Tianchi repository).
- **NetData (Tsinghua FIB Lab)** — `https://github.com/tsinghua-fib-lab/NetData`;
  the calibration uses `Performance_5G_Weekday.csv` (see
  [`sim/data/netdata/README.md`](sim/data/netdata/README.md)).

## License

MIT License, Copyright (c) 2026 Liang Dong. See [LICENSE](LICENSE).
