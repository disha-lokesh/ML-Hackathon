"""Exploratory data analysis of the raw dataset (train + student_test partitions).

Outputs
  results/eda_inventory.csv        file availability, sizes, row counts per trip
  results/eda_trip_stats.csv       sampling, gaps, extent, speed, network connectivity, route checks
  results/eda_missing_values.csv   missing / invalid value counts per file type
  results/eda_summary.json         partition-level summary
  figures/eda_*.png                plots
  maps/eda_trip_<id>.html          interactive Folium map (GPS + road network + reference route)
"""
import json

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from _common import FIGURES, MAPS, RESULTS
from pathsnap import load_config
from pathsnap.geo import LocalProjection, haversine
from pathsnap.io import list_trips, partition_dir

cfg = load_config()


def raw(path, dtype=float):
    return np.loadtxt(path, dtype=dtype, ndmin=2)


def two_way_frac(ar):
    pairs = set(map(tuple, ar.tolist()))
    return float(np.mean([(b, a) in pairs for a, b in ar.tolist()]))


inv, stats, miss = [], [], []
for part in ["train", "student_test"]:
    for tid in list_trips(part, cfg):
        d = partition_dir(part, cfg) / tid
        files = {ext: d / f"{tid}.{ext}" for ext in ["track", "nodes", "arcs", "route"]}
        row = {"partition": part, "trip_id": tid}
        for ext, p in files.items():
            row[f"{ext}_exists"] = p.exists()
            row[f"{ext}_kb"] = round(p.stat().st_size / 1024, 1) if p.exists() else 0
        tr = raw(files["track"]); nd = raw(files["nodes"]); ar = raw(files["arcs"], np.int64)
        ro = np.loadtxt(files["route"], dtype=np.int64, ndmin=1)
        row.update(n_gps=len(tr), n_nodes=len(nd), n_arcs=len(ar), n_route=len(ro))
        inv.append(row)

        # ---- missing / invalid values
        miss.append({"partition": part, "trip_id": tid,
                     "track_nan": int(np.isnan(tr).sum()), "nodes_nan": int(np.isnan(nd).sum()),
                     "track_bad_coord": int(((np.abs(tr[:, 0]) > 180) | (np.abs(tr[:, 1]) > 90)).sum()),
                     "track_zero_coord": int(((tr[:, 0] == 0) & (tr[:, 1] == 0)).sum()),
                     "arcs_invalid_node": int(((ar < 0) | (ar >= len(nd))).sum()),
                     "route_invalid_arc": int(((ro < 0) | (ro >= len(ar))).sum()),
                     "dup_timestamps": int((np.diff(tr[:, 2]) == 0).sum()),
                     "non_monotonic_t": int((np.diff(tr[:, 2]) < 0).sum())})

        # ---- trajectory statistics
        dt = np.diff(tr[:, 2])
        step = haversine(tr[:-1, 0], tr[:-1, 1], tr[1:, 0], tr[1:, 1])
        spd = step / np.maximum(dt, 1e-6)
        proj = LocalProjection.from_track(tr)
        xy = proj.to_xy(tr[:, :2])
        # ---- network connectivity
        g = coo_matrix((np.ones(len(ar)), (ar[:, 0], ar[:, 1])), shape=(len(nd), len(nd)))
        n_wcc, lab = connected_components(g, directed=True, connection="weak")
        n_scc, slab = connected_components(g, directed=True, connection="strong")
        deg = np.bincount(ar.ravel(), minlength=len(nd))
        route_cont = float(np.mean(ar[ro[:-1], 1] == ar[ro[1:], 0])) if len(ro) > 1 else 1.0
        stats.append({
            "partition": part, "trip_id": tid, "n_gps": len(tr),
            "duration_min": (tr[-1, 2] - tr[0, 2]) / 60, "frac_dt_1s": float(np.mean(np.isclose(dt, 1.0))),
            "max_gap_s": float(dt.max()), "gaps_over_5s": int((dt > 5).sum()), "gaps_over_30s": int((dt > 30).sum()),
            "length_km": float(step.sum() / 1000), "extent_x_km": float(np.ptp(xy[:, 0]) / 1000),
            "extent_y_km": float(np.ptp(xy[:, 1]) / 1000), "lon_c": float(tr[:, 0].mean()), "lat_c": float(tr[:, 1].mean()),
            "median_speed_ms": float(np.median(spd)), "p99_speed_ms": float(np.percentile(spd, 99)),
            "jumps_over_70ms": int((spd > 70).sum()),
            "n_nodes": len(nd), "n_arcs": len(ar), "isolated_nodes": int((deg == 0).sum()),
            "weak_components": int(n_wcc), "largest_wcc_frac": float(np.bincount(lab).max() / len(nd)),
            "strong_components": int(n_scc), "largest_scc_frac": float(np.bincount(slab).max() / len(nd)),
            "two_way_frac": two_way_frac(ar),
            "n_route": len(ro), "route_unique": len(set(ro.tolist())), "route_continuity": route_cont,
        })
        print(part, tid, len(tr), flush=True)

inv = pd.DataFrame(inv); stats = pd.DataFrame(stats); miss = pd.DataFrame(miss)
inv.to_csv(RESULTS / "eda_inventory.csv", index=False)
stats.to_csv(RESULTS / "eda_trip_stats.csv", index=False)
miss.to_csv(RESULTS / "eda_missing_values.csv", index=False)
summary = {}
for part, g in stats.groupby("partition"):
    summary[part] = {
        "trips": int(len(g)), "gps_points": int(g.n_gps.sum()),
        "gps_per_trip_median": float(g.n_gps.median()), "gps_per_trip_min": int(g.n_gps.min()), "gps_per_trip_max": int(g.n_gps.max()),
        "frac_dt_1s_mean": float(g.frac_dt_1s.mean()), "trips_with_gap_over_30s": int((g.gaps_over_30s > 0).sum()),
        "length_km_total": float(g.length_km.sum()), "nodes_median": float(g.n_nodes.median()), "arcs_median": float(g.n_arcs.median()),
        "trips_with_multiple_wcc": int((g.weak_components > 1).sum()), "isolated_nodes_total": int(g.isolated_nodes.sum()),
        "route_continuity_mean": float(g.route_continuity.mean()), "missing_values_total": int(miss[miss.partition == part].iloc[:, 2:].sum().sum()),
    }
json.dump(summary, open(RESULTS / "eda_summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))

# ------------------------------------------------------------------ plots
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for part, g in stats.groupby("partition"):
    axes[0, 0].hist(g.n_gps, bins=30, alpha=0.6, label=part)
    axes[0, 1].hist(g.length_km, bins=30, alpha=0.6, label=part)
    axes[0, 2].hist(g.median_speed_ms * 3.6, bins=30, alpha=0.6, label=part)
    axes[1, 0].hist(g.max_gap_s, bins=np.logspace(0, 3.5, 30), alpha=0.6, label=part)
    axes[1, 1].hist(g.n_arcs, bins=np.logspace(3, 5.5, 30), alpha=0.6, label=part)
axes[0, 0].set_title("GPS observations per trip"); axes[0, 1].set_title("Trajectory length (km)")
axes[0, 2].set_title("Median speed (km/h)"); axes[1, 0].set_title("Largest time gap (s)"); axes[1, 0].set_xscale("log")
axes[1, 1].set_title("Road-network arcs per trip"); axes[1, 1].set_xscale("log")
axes[1, 2].scatter(stats.lon_c, stats.lat_c, c=(stats.partition == "train"), cmap="coolwarm", s=20)
axes[1, 2].set_title("Trip centroids (lon/lat) - red=train, blue=student_test")
for ax in axes.ravel()[:5]:
    ax.legend()
fig.suptitle("Dataset overview (train + student_test partitions)"); fig.tight_layout()
fig.savefig(FIGURES / "eda_overview.png", dpi=130); plt.close(fig)

# time-step distribution over all points (train)
dts = []
for tid in list_trips("train", cfg):
    tr = raw(partition_dir("train", cfg) / tid / f"{tid}.track")
    dts.append(np.diff(tr[:, 2]))
dts = np.concatenate(dts)
fig, ax = plt.subplots(figsize=(7, 4))
vals, cnts = np.unique(np.clip(np.round(dts), 0, 30), return_counts=True)
ax.bar(vals, cnts); ax.set_yscale("log"); ax.set_xlabel("time step between fixes (s, clipped at 30)"); ax.set_ylabel("count")
ax.set_title(f"Sampling intervals, train partition ({np.mean(np.isclose(dts, 1)):.1%} exactly 1 s)")
fig.tight_layout(); fig.savefig(FIGURES / "eda_sampling_intervals.png", dpi=130); plt.close(fig)

# GPS-to-reference-route distance (needs 02_build_dataset label quality)
qp = RESULTS / "label_quality_train.csv"
if qp.exists():
    q = pd.read_csv(qp, dtype={"trip_id": str})
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(q.trip_id, q.align_dist_median_m, label="median"); ax.bar(q.trip_id, q.align_dist_p95_m, alpha=0.4, label="95th pct")
    ax.set_yscale("log"); ax.set_ylabel("m"); ax.tick_params(axis="x", rotation=90, labelsize=6); ax.legend()
    ax.set_title("Distance GPS fix -> aligned reference arc per trip (train)")
    fig.tight_layout(); fig.savefig(FIGURES / "eda_gps_to_route_distance.png", dpi=130); plt.close(fig)


# trajectory + network plots in metres for 4 trips
def plot_trip(ax, part, tid):
    d = partition_dir(part, cfg) / tid
    tr = raw(d / f"{tid}.track"); nd = raw(d / f"{tid}.nodes"); ar = raw(d / f"{tid}.arcs", np.int64)
    ro = np.loadtxt(d / f"{tid}.route", dtype=np.int64, ndmin=1)
    proj = LocalProjection.from_track(tr); xy = proj.to_xy(tr[:, :2]); nxy = proj.to_xy(nd)
    lo, hi = xy.min(0) - 300, xy.max(0) + 300
    A, B = nxy[ar[:, 0]], nxy[ar[:, 1]]
    sel = ((A >= lo) & (A <= hi)).all(1)
    from matplotlib.collections import LineCollection
    ax.add_collection(LineCollection(np.stack([A[sel], B[sel]], 1), colors="0.8", linewidths=0.5))
    ax.plot(xy[:, 0], xy[:, 1], ".", ms=1.0, color="tab:blue", alpha=0.6, zorder=2)
    ax.add_collection(LineCollection(np.stack([nxy[ar[ro, 0]], nxy[ar[ro, 1]]], 1), colors="green", linewidths=1.2,
                                     alpha=0.9, zorder=3))
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal")
    ax.set_title(f"{part} {tid}: grey=network, green=reference, blue=GPS", fontsize=8)


fig, axes = plt.subplots(2, 2, figsize=(13, 12))
for ax, (p, t) in zip(axes.ravel(), [("train", "00000001"), ("train", "00000002"), ("train", "00000054"), ("student_test", "00000007")]):
    plot_trip(ax, p, t)
fig.tight_layout(); fig.savefig(FIGURES / "eda_trajectories.png", dpi=130); plt.close(fig)


# ------------------------------------------------------------------ folium
def folium_trip(part, tid):
    d = partition_dir(part, cfg) / tid
    tr = raw(d / f"{tid}.track"); nd = raw(d / f"{tid}.nodes"); ar = raw(d / f"{tid}.arcs", np.int64)
    ro = np.loadtxt(d / f"{tid}.route", dtype=np.int64, ndmin=1)
    m = folium.Map(location=[tr[:, 1].mean(), tr[:, 0].mean()], zoom_start=14, tiles="OpenStreetMap")
    lo = tr[:, :2].min(0) - 0.003; hi = tr[:, :2].max(0) + 0.003
    A, B = nd[ar[:, 0]], nd[ar[:, 1]]
    sel = np.where(((A >= lo) & (A <= hi)).all(1))[0][:20000]
    net = folium.FeatureGroup("Road network (arcs)", show=True)
    for a in sel:
        folium.PolyLine([(A[a, 1], A[a, 0]), (B[a, 1], B[a, 0])], color="#999999", weight=1, opacity=0.6).add_to(net)
    net.add_to(m)
    ref = folium.FeatureGroup("Reference route", show=True)
    for a in ro:
        folium.PolyLine([(nd[ar[a, 0], 1], nd[ar[a, 0], 0]), (nd[ar[a, 1], 1], nd[ar[a, 1], 0])], color="green", weight=5, opacity=0.8).add_to(ref)
    ref.add_to(m)
    gps = folium.FeatureGroup("GPS observations", show=True)
    for lon, lat, t in tr[::2]:
        folium.CircleMarker((lat, lon), radius=2, color="blue", fill=True, opacity=0.7, tooltip=f"t={t:.0f}s").add_to(gps)
    gps.add_to(m)
    folium.Marker((tr[0, 1], tr[0, 0]), tooltip="start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker((tr[-1, 1], tr[-1, 0]), tooltip="end", icon=folium.Icon(color="red", icon="stop")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[tr[:, 1].min(), tr[:, 0].min()], [tr[:, 1].max(), tr[:, 0].max()]])
    m.save(str(MAPS / f"eda_{part}_{tid}.html"))


for p, t in [("train", "00000002"), ("student_test", "00000007")]:
    folium_trip(p, t)
print("EDA done")
