"""One-shot reproduction of the paper's results from the shipped data (no solver needed).

Runs clustering verification, the trade-off summary, TOPSIS (Table 2), and regenerates
Fig 2-4. To regenerate the optimization outputs themselves, see optimization.py.
"""
import analysis
import clustering
import figures


def main():
    print("=" * 70)
    print("STAGE 1  Revenue-based clustering")
    print("=" * 70)
    drg, m = clustering.revenue_kmeans()
    print(clustering.summary().to_string())
    share = clustering.summary().loc[[1, 2], "share_pct"].sum()
    print(f"clusters 1&2 share: {share:.1f}%  (manuscript: 77%)")
    print(f"k-means(k=6) silhouette={m['silhouette']}  ARI vs stored ClusterId={m['ari_vs_stored']}")

    print("\n" + "=" * 70)
    print("STAGE 3  Trade-off summary + TOPSIS (Table 2)")
    print("=" * 70)
    scen = analysis.scenario_metrics()
    print("trade-off:", analysis.tradeoff_summary(scen))
    t2 = analysis.table2(scen)
    print(t2.head(6).to_string(index=False))
    print("... (20 regions total)")

    # parity check against published Table 2
    idx = t2.set_index("Region")
    checks = {"Alabama": (94028, 938.093), "California": (483573, 5025.575), "Florida": (457958, 4466.076)}
    print("\nParity vs published Table 2:")
    for region, (vol, rev) in checks.items():
        row = idx.loc[region]
        ok = int(row.PatientVolume) == vol and abs(row.Revenue_milUSD - rev) < 0.5
        print(f"  {region}: {int(row.PatientVolume)} pts / {row.Revenue_milUSD} M  "
              f"e1={row['within']} e2={row['among']}  {'OK' if ok else 'MISMATCH'}")

    print("\n" + "=" * 70)
    print("STAGE 4  Weight-sensitivity robustness (TOPSIS factorial weights)")
    print("=" * 70)
    summ = analysis.sensitivity_summary(analysis.sensitivity_weights(scen))
    print(f"{summ['n_weight_vectors']} weight vectors x 20 regions  "
          f"mean percentile={summ['overall_mean_percentile']}  min={summ['overall_min_percentile']}")
    print("mean percentile vs revenue weight:", summ["by_w_revenue"])
    print("(base solution stays high-percentile until revenue weight exceeds equity weight)")

    print("\n" + "=" * 70)
    print("Figures")
    print("=" * 70)
    figures.make_all()


if __name__ == "__main__":
    main()
