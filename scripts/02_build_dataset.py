"""Build the candidate/feature table for the train and student_test partitions.

For each trip: clean GPS -> project to metres -> generate candidates -> features ->
derive per-point reference labels by monotone alignment with the .route file ->
label each candidate row (1 if candidate arc == reference arc of that point).

Output: data/processed/<partition>_candidates.pkl and results/label_quality_<partition>.csv
"""
import argparse
import time

import numpy as np
import pandas as pd

from _common import PROCESSED, RESULTS
from pathsnap import load_config
from pathsnap.features import build_candidate_table
from pathsnap.io import list_trips, load_route, load_trip
from pathsnap.labels import align_points_to_route


def process_partition(partition, cfg):
    frames, quality = [], []
    for tid in list_trips(partition, cfg):
        t0 = time.perf_counter()
        trip = load_trip(partition, tid, cfg)
        df, net, xy = build_candidate_table(trip, cfg)
        route = load_route(partition, tid, cfg)
        labels, path, adist = align_points_to_route(xy, net, route)
        df["label"] = (df["arc"].to_numpy() == labels[df["point"].to_numpy()]).astype(np.int8)
        df["t"] = trip.track[df["point"].to_numpy(), 2]
        df["align_dist"] = adist[df["point"].to_numpy()]
        covered = df.groupby("point")["label"].max().reindex(range(len(xy)), fill_value=0)
        quality.append({
            "trip_id": tid, "n_points": len(xy), "n_route_arcs": len(route),
            "route_arcs_with_points": len(np.unique(path)),
            "align_dist_median_m": float(np.median(adist)), "align_dist_p95_m": float(np.percentile(adist, 95)),
            "align_dist_max_m": float(adist.max()), "points_over_50m": int((adist > 50).sum()),
            "candidate_rows": len(df), "mean_candidates": len(df) / max(len(xy), 1),
            "candidate_miss_rate": float(1 - covered.mean()), "seconds": time.perf_counter() - t0,
        })
        print(f"[{partition}] {tid}: pts={len(xy)} rows={len(df)} miss={quality[-1]['candidate_miss_rate']:.3f} "
              f"({quality[-1]['seconds']:.1f}s)", flush=True)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.to_pickle(PROCESSED / f"{partition}_candidates.pkl")
    q = pd.DataFrame(quality)
    q.to_csv(RESULTS / f"label_quality_{partition}.csv", index=False)
    print(f"[{partition}] rows={len(out)} pos_rate={out['label'].mean():.3f} "
          f"miss_rate(points)={np.average(q['candidate_miss_rate'], weights=q['n_points']):.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--partitions", nargs="+", default=["train", "student_test"])
    args = ap.parse_args()
    cfg = load_config()
    for p in args.partitions:
        process_partition(p, cfg)
