# Toward Equitable Healthcare Access — code & data

Reproducible pipeline for *"Toward Equitable Healthcare Access: Navigating the
Cream-Skimming Dilemma Through a Financial-Ethical Multi-Objective Case-Mix Planning
Approach."* It reproduces the clustering, the optimization outputs, the finance-equity
trade-off analysis, the TOPSIS selection (Table 2), and Figures 2–4.

## Quick start

```bash
pip install -r requirements.txt          # or: conda env create -f environment.yml
cd src
python reproduce.py                       # reproduces clustering, Table 2, trade-off, Figs 2-4
```

`reproduce.py` runs entirely from the shipped, solved outputs — **no optimization solver
is required**. It prints the clustering summary, the trade-off numbers, manuscript Table 2
(with a parity check against the published values), and writes the figures to `figures/`.

## Repository layout

```
.
├── README.md, LICENSE
├── requirements.txt, environment.yml
├── data/
│   ├── inputs/
│   │   ├── weight.xlsx                  DRG master: Revenue, MeanLOS, ClusterId (765 DRGs)
│   │   ├── weight_with_severity.xlsx    DRG severity weights (S5 sensitivity)
│   │   ├── discharge_20.xlsx            per-region per-DRG demand (20 regions)
│   │   ├── dischargebycluster_20.xlsx   per-region per-cluster demand
│   │   └── beds_20.xlsx                 synthetic bed-day capacity
│   └── outputs/
│       ├── base_outputs.csv.gz          MILP allocations, availability 0.8, all equity targets
│       ├── availability_outputs.csv.gz  MILP allocations, availability sweep 0.6–1.0
│       └── solver_log.csv.gz            objective / optimality gap / runtime per scenario
└── src/
    ├── paths.py          path & constant configuration
    ├── dataio.py         data loaders
    ├── clustering.py     Stage 1 — revenue-based DRG clustering
    ├── optimization.py   Stage 2 — financial-ethical case-mix MILP (gurobipy)
    ├── analysis.py       Stage 3 — trade-off analysis + TOPSIS (Table 2)
    ├── figures.py        Figures 2–4
    └── reproduce.py      one-shot reproduction from shipped outputs
```

## Pipeline

1. **Clustering** (`clustering.py`) — k-means (k=6) on revenue per patient partitions the
   765 DRGs into six profitability tiers. The authoritative assignment is stored in
   `weight.xlsx` (`ClusterId`).
2. **Optimization** (`optimization.py`) — a MILP maximizes revenue subject to bed-day
   capacity, per-DRG demand, and inter-/intra-cluster equity (ε-constraint). Solved with
   Gurobi. The intra-cluster equity is written in the equivalent **max-min** form
   (`U ≥ ratioᵢ ≥ L`, `U − L ≤ θ`), reducing the model from ~1.7M to ~20k constraints
   without changing the feasible region.
3. **Analysis** (`analysis.py`) — recomputes per-scenario revenue and patient volume,
   the finance-equity trade-off, and the TOPSIS ranking that yields **Table 2**.
4. **Figures** (`figures.py`) — Fig 2 (clustering), Fig 3 (regional health profiles),
   Fig 4 (treatment-rate ratios across resource availability).

## Regenerating the optimization outputs (optional)

The solved outputs ship with the repo. To regenerate them you need **Gurobi**
(`pip install gurobipy` + a license):

```bash
cd src
python optimization.py --within 0.1 --among 0.1 --availability 0.8   # one scenario
python optimization.py --sweep                                       # full availability sweep
```

A license-free open solver (HiGHS) can be substituted in `optimization.py` for users
without Gurobi.
