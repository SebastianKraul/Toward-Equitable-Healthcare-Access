"""Stage 3 - trade-off analysis and TOPSIS selection (manuscript Table 2).

Reads the consolidated base-case allocations and the DRG master; no recomputation of
the optimization is needed. TOPSIS criterion weights follow the manuscript:
within=0.15, among=0.15, revenue=0.40, patient volume=0.30, with max-normalization.
"""
import numpy as np
import pandas as pd

import dataio

W_WITHIN, W_AMONG, W_REVENUE, W_VOLUME = 0.15, 0.15, 0.40, 0.30


def scenario_metrics(base=None, weight=None):
    """Per (within, among, region): total Revenue (mil USD) and patient volume."""
    base = dataio.load_base_outputs() if base is None else base
    rev = dict((dataio.load_weight() if weight is None else weight)[["DRGCode", "Revenue"]].values)
    b = base.copy()
    b["rev"] = b["DRG"].map(rev).astype(float) * b["ProposedVolume"]
    g = (b.groupby(["Threshold_InGroup", "Threshold_Among", "Region"], as_index=False)
         .agg(Revenue=("rev", "sum"), PatientVolume=("ProposedVolume", "sum")))
    g["Revenue_mil"] = g["Revenue"] / 1e6
    return g.rename(columns={"Threshold_InGroup": "within", "Threshold_Among": "among"})


def _weighted_matrix(scen, w_within, w_among, w_rev, w_vol, norm="max"):
    """Weighted-normalized criteria for arbitrary weights.

    norm="max"    : thresholds raw, revenue/volume max-normalized within region
                    (manuscript scheme, reproduces Table 2).
    norm="vector" : every criterion vector-normalized within region (x/sqrt(sum x^2)).
    """
    df = scen.copy()
    g = df.groupby("Region")
    if norm == "max":
        df["wWithin"] = df["within"] * w_within
        df["wAmong"] = df["among"] * w_among
        df["wRevenue"] = df["Revenue"] / g["Revenue"].transform("max") * w_rev
        df["wVolume"] = df["PatientVolume"] / g["PatientVolume"].transform("max") * w_vol
    elif norm == "vector":
        def vnorm(col):
            d = np.sqrt(df.groupby("Region")[col].transform(lambda s: (s ** 2).sum()))
            return df[col] / d
        df["wWithin"] = vnorm("within") * w_within
        df["wAmong"] = vnorm("among") * w_among
        df["wRevenue"] = vnorm("Revenue") * w_rev
        df["wVolume"] = vnorm("PatientVolume") * w_vol
    else:
        raise ValueError(f"unknown norm: {norm!r}")
    return df


def _rank_by_closeness(df):
    """PIS/NIS separation, relative closeness RC and per-region rank.
    Within/among are cost criteria (min is ideal); revenue/volume are benefits."""
    g = df.groupby("Region")
    s_plus = np.sqrt((df.wWithin - g.wWithin.transform("min")) ** 2 + (df.wAmong - g.wAmong.transform("min")) ** 2
                     + (df.wRevenue - g.wRevenue.transform("max")) ** 2 + (df.wVolume - g.wVolume.transform("max")) ** 2)
    s_minus = np.sqrt((df.wWithin - g.wWithin.transform("max")) ** 2 + (df.wAmong - g.wAmong.transform("max")) ** 2
                      + (df.wRevenue - g.wRevenue.transform("min")) ** 2 + (df.wVolume - g.wVolume.transform("min")) ** 2)
    df["RC"] = s_minus / (s_plus + s_minus)
    df["Rank"] = df.groupby("Region")["RC"].rank(ascending=False, method="first").astype(int)
    return df


def topsis(scen, norm="max"):
    return _rank_by_closeness(_weighted_matrix(scen, W_WITHIN, W_AMONG, W_REVENUE, W_VOLUME, norm=norm))


def _weight_grid():
    """Factorial grid (TOPSIS.Rmd): w_threshold (= within = among) in 0.10..1.0,
    w_revenue and w_volume in 0.20..1.0 step 0.05, keeping 2*w_threshold + w_revenue + w_volume == 1."""
    w_thr = np.round(np.arange(0.10, 1.0 + 1e-9, 0.05), 2)
    w_rev = np.round(np.arange(0.20, 1.0 + 1e-9, 0.05), 2)
    w_vol = np.round(np.arange(0.20, 1.0 + 1e-9, 0.05), 2)
    return [(float(t), float(r), float(v)) for t in w_thr for r in w_rev for v in w_vol
            if np.isclose(2 * t + r + v, 1.0)]


def sensitivity_weights(scen=None, norm="max"):
    """For every weight vector in the factorial grid, record where each region's
    base (manuscript-weight Rank-1) pick lands. Percentile = (n-Rank+1)/n*100 measures
    how robustly that pick survives alternative weightings (100 = still best)."""
    scen = scenario_metrics() if scen is None else scen
    base = topsis(scen)
    base_codes = set(map(tuple, base.loc[base.Rank == 1, ["Region", "within", "among"]].values))
    rows = []
    for w_thr, w_rev, w_vol in _weight_grid():
        df = _rank_by_closeness(_weighted_matrix(scen, w_thr, w_thr, w_rev, w_vol, norm=norm))
        df["n_options"] = df.groupby("Region")["RC"].transform("size")
        for region, within, among, rank, n in df[["Region", "within", "among", "Rank", "n_options"]].itertuples(index=False):
            if (region, within, among) in base_codes:
                rows.append({"Region": region, "w_threshold": w_thr, "w_revenue": w_rev,
                             "w_volume": w_vol, "Rank": int(rank), "n_options": int(n),
                             "Percentile": (n - rank + 1) / n * 100})
    return pd.DataFrame(rows)


def sensitivity_summary(sens=None, norm="max"):
    """Mean percentile quality of the base solution vs each weight dimension."""
    sens = sensitivity_weights(norm=norm) if sens is None else sens
    return {
        "n_weight_vectors": int(sens.groupby(["w_threshold", "w_revenue", "w_volume"]).ngroups),
        "overall_mean_percentile": round(sens.Percentile.mean(), 1),
        "overall_min_percentile": round(sens.Percentile.min(), 1),
        "by_w_revenue": sens.groupby("w_revenue")["Percentile"].mean().round(1).to_dict(),
        "by_w_volume": sens.groupby("w_volume")["Percentile"].mean().round(1).to_dict(),
        "by_w_threshold": sens.groupby("w_threshold")["Percentile"].mean().round(1).to_dict(),
    }


def table2(scen=None):
    scen = scenario_metrics() if scen is None else scen
    best = topsis(scen)
    best = best[best.Rank == 1].sort_values("Region").copy()
    best["Revenue_milUSD"] = best["Revenue_mil"].round(3)
    return best[["Region", "within", "among", "PatientVolume", "Revenue_milUSD", "RC"]].reset_index(drop=True)


def tradeoff_summary(scen=None):
    scen = scenario_metrics() if scen is None else scen
    agg = scen.groupby(["within", "among"], as_index=False).agg(Rev=("Revenue", "sum"), Vol=("PatientVolume", "sum"))
    base = agg[(agg.within == 1.0) & (agg.among == 1.0)].iloc[0]
    eq = agg[(agg.within == 0.1) & (agg.among == 0.1)].iloc[0]
    piv_v = scen.pivot_table(index="Region", columns=["within", "among"], values="PatientVolume")
    incr = (piv_v[(0.1, 0.1)] - piv_v[(1.0, 1.0)]) / piv_v[(1.0, 1.0)] * 100
    return {
        "revenue_range_mil": (round(eq.Rev / 1e6), round(base.Rev / 1e6)),
        "aggregate_revenue_drop_pct": round((base.Rev - eq.Rev) / base.Rev * 100, 2),
        "aggregate_admissions_change_pct": round((eq.Vol - base.Vol) / base.Vol * 100, 2),
        "per_region_admissions_pct": (round(incr.min(), 1), round(incr.max(), 1), round(incr.mean(), 1)),
    }


if __name__ == "__main__":
    pd.set_option("display.max_rows", None, "display.width", 200)
    t2 = table2()
    print(t2.to_string(index=False))
    print("\nTrade-off summary:", tradeoff_summary())
    print("\nWeight sensitivity (percentile quality of base solution):", sensitivity_summary())
