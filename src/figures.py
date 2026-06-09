"""Generate the manuscript data figures (Fig 2-4) from the consolidated data.

Fig 1 (framework diagram) is not produced here - it is a drawing in the manuscript.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import dataio
import paths

CC = paths.CLUSTER_COLORS


def fig2_clusters(path):
    drg = dataio.load_weight(as_int_cluster=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    for c in range(1, 7):
        sub = drg[drg.ClusterId == c]
        ax.scatter(sub.MeanLOS, sub.Revenue, s=14, alpha=0.6, color=CC[c - 1], label=f"Cluster {c}")
    ax.set_xlabel("Length of Stay (days)"); ax.set_ylabel("Revenue per patient (USD)")
    ax.set_title("Revenue-based clustering of 765 DRGs"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def fig3_health_profile(path):
    drg = dataio.load_weight(as_int_cluster=True)
    dis = dataio.load_discharge().merge(drg[["DRGCode", "MeanLOS", "ClusterId"]], on="DRGCode")
    dis["BedDays"] = dis.Discharge * dis.MeanLOS
    prof = dis.groupby(["Region", "ClusterId"])["BedDays"].sum().unstack("ClusterId").fillna(0)
    prof = prof.div(prof.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(prof))
    for c in range(1, 7):
        ax.bar(prof.index, prof[c], bottom=bottom, color=CC[c - 1], label=f"Cluster {c}")
        bottom += prof[c].values
    ax.set_ylabel("Demand in bed-days contribution (%)"); ax.set_title("Regional health demand profiles by DRG cluster")
    ax.set_xticks(range(len(prof.index))); ax.set_xticklabels(prof.index, rotation=90, fontsize=7)
    ax.legend(frameon=False, fontsize=8, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    fig.tight_layout(); fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def fig4_availability(path):
    av = dataio.load_availability_outputs()
    clu = dataio.cluster_map()
    dc = dataio.load_discharge_cluster()
    demand = {(r.Region, r.ClusterId): r.Discharge for r in dc.itertuples(index=False)}
    av = av.rename(columns={"Threshold_InGroup": "within", "Threshold_Among": "among"})
    av = av[(av.within != 0) & (av.among != 0)].copy()
    av["cluster"] = av.DRG.map(clu)
    g = (av.groupby(["availability", "within", "among", "Region", "cluster"], as_index=False)
         .agg(vol=("ProposedVolume", "sum")))
    g["demand"] = [demand.get((r, c), 0) for r, c in zip(g.Region, g.cluster)]
    g = g[g.demand > 0]
    g["ratio"] = g.vol / g.demand
    prof = g.groupby(["availability", "among", "cluster"], as_index=False)["ratio"].mean()

    avails = sorted(av.availability.unique(), reverse=True)
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, a in zip(axes, avails):
        for c in ["1", "2", "3", "4", "5", "6"]:
            s = prof[(prof.availability == a) & (prof.cluster == c)].sort_values("among")
            if not s.empty:
                ax.plot(s["among"], s["ratio"], marker="o", ms=3, color=CC[int(c) - 1], label=f"Cluster {c}")
        ax.set_title(f"availability = {a:g}", fontsize=9); ax.set_ylim(0, 1); ax.invert_xaxis()
    for ax in axes[len(avails):]:
        ax.set_visible(False)
    fig.supxlabel("Target inter-cluster inequity (e1)"); fig.supylabel("Average treatment-rate ratio")
    fig.suptitle("Average treatment-rate ratios per cluster by resource availability")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0.02, 0.04, 1, 0.97)); fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def make_all():
    os.makedirs(paths.FIGURES, exist_ok=True)
    fig2_clusters(os.path.join(paths.FIGURES, "Fig2.png"))
    fig3_health_profile(os.path.join(paths.FIGURES, "Fig3.png"))
    fig4_availability(os.path.join(paths.FIGURES, "Fig4.png"))
    print("figures written to", paths.FIGURES)


if __name__ == "__main__":
    make_all()
