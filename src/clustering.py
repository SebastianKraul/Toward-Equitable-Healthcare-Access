"""Stage 1 - revenue-based DRG clustering.

The authoritative 6-cluster assignment is stored in weight.xlsx (ClusterId); it is a
clean, contiguous partition of DRGs by revenue per patient and is consumed by every
downstream stage. This module documents/reproduces that step: k-means (k=6) on revenue,
ordered by increasing mean revenue, and reports agreement with the stored assignment.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

import dataio
import paths


def revenue_kmeans(seed=42):
    drg = dataio.load_weight(as_int_cluster=True)
    X = drg[["Revenue"]].values
    km = KMeans(n_clusters=paths.N_CLUSTERS, n_init=25, random_state=seed).fit(X)
    order = list(np.argsort([X[km.labels_ == k].mean() for k in range(paths.N_CLUSTERS)]))
    remap = {old: i + 1 for i, old in enumerate(order)}
    drg["kmeans_cluster"] = [remap[l] for l in km.labels_]
    sil = silhouette_score(X, km.labels_)
    ari = adjusted_rand_score(drg["ClusterId"], drg["kmeans_cluster"])
    return drg, {"silhouette": round(float(sil), 3), "ari_vs_stored": round(float(ari), 3)}


def summary():
    drg = dataio.load_weight(as_int_cluster=True)
    s = drg.groupby("ClusterId").agg(
        n=("DRGCode", "size"), mean_revenue=("Revenue", "mean"), mean_LOS=("MeanLOS", "mean")).round(2)
    s["share_pct"] = (s["n"] / s["n"].sum() * 100).round(1)
    return s


if __name__ == "__main__":
    drg, m = revenue_kmeans()
    print("Cluster summary (stored ClusterId):")
    print(summary().to_string())
    print(f"\nclusters 1&2 share: {summary().loc[[1, 2], 'share_pct'].sum():.1f}% (manuscript: 77%)")
    print(f"k-means k=6 on revenue: silhouette={m['silhouette']}, ARI vs stored={m['ari_vs_stored']}")
