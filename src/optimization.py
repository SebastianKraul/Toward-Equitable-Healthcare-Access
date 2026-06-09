"""Stage 2 - financial-ethical case-mix planning MILP (Pyomo-free gurobipy engine).

Maximizes hospital revenue subject to: bed-day capacity, per-DRG demand limits, and the
inter-/intra-cluster equity constraints (epsilon-constraint method). The intra-cluster
(within) equity is expressed in max-min form -- per (region, cluster) auxiliary bounds
U >= ratio_i >= L with U - L <= threshold -- which is mathematically equivalent to the
pairwise treatment-rate-ratio constraints in the manuscript but reduces the model from
~1.7M to ~20k constraints (build once, sweep by updating constraint RHS).

Requires gurobipy + a Gurobi license. Re-running is optional: the repository ships the
solved outputs (data/outputs/*.csv.gz); this module lets you regenerate them.
"""
import os

import gurobipy as gp
import pandas as pd
from gurobipy import GRB

import dataio
import paths

BASE_AVAILABILITY = 0.8


class CaseMixModel:
    def __init__(self, mip_gap=0.005, time_limit=60, threads=0, verbose=False):
        drg = dataio.load_weight()
        demand = dataio.load_discharge()
        dc = dataio.load_discharge_cluster()
        beds = dataio.load_beds()
        revenue = dict(drg[["DRGCode", "Revenue"]].values)
        los = dict(drg[["DRGCode", "MeanLOS"]].values)
        cluster = dict(drg[["DRGCode", "ClusterId"]].values)
        self.base_beds = {r.Region: float(r.BedDays) for r in beds.itertuples(index=False)}
        dcd = {(r.Region, r.ClusterId): r.Discharge for r in dc.itertuples(index=False)}
        pos = demand[demand.Discharge > 0]

        m = gp.Model("casemix")
        m.Params.OutputFlag = 1 if verbose else 0
        m.Params.MIPGap = mip_gap
        m.Params.TimeLimit = time_limit
        if threads:
            m.Params.Threads = threads
        n = {}
        for r in pos.itertuples(index=False):
            n[(r.Region, r.DRGCode)] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=float(r.Discharge),
                                                obj=revenue[r.DRGCode], name=f"n[{r.Region},{r.DRGCode}]")
        m.ModelSense = GRB.MAXIMIZE
        m.update()
        disc = {(r.Region, r.DRGCode): r.Discharge for r in pos.itertuples(index=False)}
        by_rc = {}
        for (s, d) in n:
            by_rc.setdefault((s, cluster[d]), []).append(d)

        # (5) bed-day capacity (RHS scaled by availability)
        self.bed_constr = {s: m.addConstr(gp.quicksum(n[(s, d)] * los[d] for d in [d for (ss, d) in n if ss == s])
                                          <= self.base_beds[s], name=f"bed[{s}]") for s in self.base_beds}
        # (3) intra-cluster equity, max-min reformulation (RHS = threshold_within)
        self.within_constr = []
        for (s, c), ds in by_rc.items():
            U = m.addVar(lb=0, ub=1, name=f"U[{s},{c}]"); L = m.addVar(lb=0, ub=1, name=f"L[{s},{c}]")
            m.update()
            for i in ds:
                ai = 1.0 / disc[(s, i)]
                m.addConstr(ai * n[(s, i)] <= U)
                m.addConstr(ai * n[(s, i)] >= L)
            self.within_constr.append(m.addConstr(U - L <= 1.0))
        # (2) inter-cluster equity (RHS = threshold_among)
        self.among_constr = []
        for s in self.base_beds:
            for u in [str(k) for k in range(1, 7)]:
                for v in [str(k) for k in range(1, 7)]:
                    if u != v and dcd.get((s, u), 0) and dcd.get((s, v), 0):
                        lhs = gp.quicksum(n[(s, d)] for d in by_rc.get((s, u), [])) / dcd[(s, u)]
                        rhs = gp.quicksum(n[(s, d)] for d in by_rc.get((s, v), [])) / dcd[(s, v)]
                        self.among_constr.append(m.addConstr(lhs - rhs <= 1.0))
        m.update()
        self.m, self.n, self.revenue = m, n, revenue

    def solve(self, within, among, availability=BASE_AVAILABILITY):
        for c in self.within_constr:
            c.RHS = within
        for c in self.among_constr:
            c.RHS = among
        for s, c in self.bed_constr.items():
            c.RHS = self.base_beds[s] * availability / BASE_AVAILABILITY
        self.m.optimize()
        return {"within": within, "among": among, "availability": availability,
                "objective": self.m.ObjVal, "gap_pct": self.m.MIPGap * 100,
                "runtime_s": self.m.Runtime, "status": int(self.m.Status)}

    def solution_frame(self, within, among):
        rows = [{"Threshold_InGroup": within, "Threshold_Among": among, "Region": s, "DRG": d,
                 "ProposedVolume": int(round(v.X))} for (s, d), v in self.n.items() if v.X > 0.5]
        return pd.DataFrame(rows)


def sweep(availabilities, thresholds, out_csv, mip_gap=0.005, time_limit=60):
    """Solve every (availability, within, among) and write a consolidated csv.gz + solver log."""
    model = CaseMixModel(mip_gap=mip_gap, time_limit=time_limit)
    frames, log = [], []
    for a in availabilities:
        for w in thresholds:
            for am in thresholds:
                res = model.solve(w, am, a)
                df = model.solution_frame(w, am); df["availability"] = a
                frames.append(df); log.append(res)
                print(f"avail={a} within={w} among={am} obj={res['objective']/1e6:.0f}M "
                      f"gap={res['gap_pct']:.3f}% t={res['runtime_s']:.1f}s", flush=True)
    pd.concat(frames, ignore_index=True).to_csv(out_csv, index=False, compression="gzip")
    pd.DataFrame(log).to_csv(os.path.join(paths.OUTPUTS, "solver_log.csv.gz"), index=False, compression="gzip")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Solve / regenerate the case-mix MILP outputs.")
    ap.add_argument("--within", type=float, default=0.1)
    ap.add_argument("--among", type=float, default=0.1)
    ap.add_argument("--availability", type=float, default=BASE_AVAILABILITY)
    ap.add_argument("--sweep", action="store_true", help="full availability x threshold sweep")
    a = ap.parse_args()
    if a.sweep:
        avs = [round(0.6 + 0.05 * i, 2) for i in range(9)]
        ths = [round(1.0 - 0.1 * k, 1) for k in range(10)]
        sweep(avs, ths, paths.AVAIL_OUTPUTS)
    else:
        model = CaseMixModel(mip_gap=0.001, time_limit=300)
        print(model.solve(a.within, a.among, a.availability))
