"""HMM / Viterbi route reconstruction: decoder tuning, evaluation and export.

1. Out-of-fold probabilities for EVERY observation of the training trips
   (5-fold GroupKFold; final tuned model refit on the training folds) - so the
   decoder is tuned on honest, unseen-trip probabilities.
2. Grid search of decoder parameters (beta, emission weight) on training trips.
3. Student-test comparison: classifier-only vs classifier+HMM vs distance-only HMM.
4. Export artifacts/inference_config.json and Folium maps of student-test trips.
"""
import argparse
import itertools
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from _common import ARTIFACTS, FIGURES, MAPS, PROCESSED, RESULTS
from pathsnap import load_config
from pathsnap.features import FEATURES, prepare_trip
from pathsnap.hmm import build_lattice, classifier_only, decode_lattice, reconstruct_route, viterbi_decode
from pathsnap.io import load_route, load_trip
from pathsnap.labels import align_points_to_route
from pathsnap.metrics import point_accuracy, route_continuity, route_overlap
from pathsnap.training import MAX_TRIP_MEDIAN_ALIGN
from pathsnap.viz import nearby_arcs, route_map


def oof_full_train(model, cfg):
    full = pd.read_pickle(PROCESSED / "train_candidates.pkl").reset_index(drop=True)
    sub = pd.read_pickle(PROCESSED / "train_subsample.pkl")
    trips = np.array(sorted(full["trip_id"].unique()))
    full["prob"] = np.nan
    for k, (tr, va) in enumerate(GroupKFold(n_splits=cfg["cv"]["n_splits"]).split(trips, groups=trips)):
        fit_rows = sub[sub["trip_id"].isin(trips[tr])]
        m = clone(model).fit(fit_rows[FEATURES].to_numpy(np.float32), fit_rows["label"].to_numpy())
        mask = full["trip_id"].isin(trips[va]).to_numpy()
        full.loc[mask, "prob"] = m.predict_proba(full.loc[mask, FEATURES].to_numpy(np.float32))[:, 1]
        print(f"OOF fold {k} done", flush=True)
    return full


def evaluate_trip(part, tid, cand, cfg, params, nets_cache, prob_col="prob"):
    if tid not in nets_cache:
        trip = load_trip(part, tid, cfg)
        net, xy = prepare_trip(trip, cfg)
        route = load_route(part, tid, cfg)
        labels, _, adist = align_points_to_route(xy, net, route)
        nets_cache[tid] = (trip, net, xy, route, labels)
    trip, net, xy, route, labels = nets_cache[tid]
    g = cand.reset_index(drop=True)
    t0 = time.perf_counter()
    pred, info = viterbi_decode(g, xy, trip.track[:, 2], net, params, prob_col=prob_col)
    dec_s = time.perf_counter() - t0
    clf = classifier_only(g, len(xy), prob_col=prob_col)
    out = {"trip_id": tid, "n_points": len(xy), "decode_s": dec_s, "breaks": info["breaks"], "skipped": info["skipped"]}
    for tag, pp in [("clf", clf), ("hmm", pred)]:
        r, comp, unf = reconstruct_route(pp, net)
        out[f"{tag}_point_acc"] = point_accuracy(pp, labels)
        out[f"{tag}_continuity"] = route_continuity(comp, net)
        out[f"{tag}_n_route_changes"] = len(comp)
        for k, v in route_overlap(r, route, net).items():
            out[f"{tag}_{k}"] = v
        out[f"{tag}_route"] = r
    out["clf_points"] = clf
    return out


def static_route_plot(tid, xy, net, ref, r):
    """Static (PNG) version of the map for the PDF report: full trip + zoom on the worst 600 m window."""
    from matplotlib.collections import LineCollection
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    for ax, zoom in zip(axes, [False, True]):
        ax.plot(xy[:, 0], xy[:, 1], ".", ms=1.5, color="tab:blue", alpha=0.6, label="GPS", zorder=2)
        ax.add_collection(LineCollection(np.stack([net.A[ref], net.B[ref]], 1), colors="#1a9850", linewidths=6, alpha=0.45, label="reference", zorder=1))
        ax.add_collection(LineCollection(np.stack([net.A[r["hmm_route"]], net.B[r["hmm_route"]]], 1), colors="#d73027", linewidths=1.6, label="predicted (HMM)", zorder=3))
        cp = r["clf_points"]; ok = cp >= 0
        ax.plot(*(0.5 * (net.A[cp[ok]] + net.B[cp[ok]])).T, "x", ms=2, color="#fdae61", alpha=0.5, label="classifier-only arc midpoints", zorder=2)
        if zoom:
            c = xy[len(xy) // 2]; ax.set_xlim(c[0] - 300, c[0] + 300); ax.set_ylim(c[1] - 300, c[1] + 300)
            ax.set_title(f"zoom: 600 m window at mid-trip")
        else:
            pad = 200; ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad); ax.set_ylim(xy[:, 1].min() - pad, xy[:, 1].max() + pad)
            ax.set_title(f"student_test {tid} - HMM point acc {r['hmm_point_acc']:.3f}, classifier-only {r['clf_point_acc']:.3f}")
        ax.set_aspect("equal"); ax.set_xlabel("x east (m)"); ax.set_ylabel("y north (m)")
    axes[0].legend(loc="best", fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / f"pred_route_{tid}.png", dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune-trips", type=int, default=25)
    args = ap.parse_args()
    cfg = load_config(); seed = cfg["random_seed"]
    model = joblib.load(ARTIFACTS / "final_model.joblib")

    # ---------- 1. honest OOF probabilities on full training trajectories
    oof_path = PROCESSED / "train_oof_full.pkl"
    if oof_path.exists():
        full = pd.read_pickle(oof_path)
    else:
        full = oof_full_train(model, cfg); full.to_pickle(oof_path)

    # ---------- 2. decoder grid search on training trips
    q = pd.read_csv(RESULTS / "label_quality_train.csv", dtype={"trip_id": str})
    good = q[q.align_dist_median_m <= MAX_TRIP_MEDIAN_ALIGN].trip_id.tolist()
    rng = np.random.default_rng(seed)
    tune_trips = sorted(rng.choice(good, size=min(args.tune_trips, len(good)), replace=False))
    grid = list(itertools.product([0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 100.0], [0.1, 0.25, 0.5, 1.0, 2.0]))
    lattices = {}
    for tid, g in full[full.trip_id.isin(tune_trips)].groupby("trip_id"):
        trip = load_trip("train", tid, cfg); net, xy = prepare_trip(trip, cfg)
        labels, _, _ = align_points_to_route(xy, net, load_route("train", tid, cfg))
        lattices[tid] = (build_lattice(g.reset_index(drop=True), xy, trip.track[:, 2], net, cfg["hmm"]), labels)
        print("lattice", tid, flush=True)
    rows = []
    for beta, we in grid:
        params = dict(cfg["hmm"], beta=beta, emission_weight=we)
        accs, w = [], []
        for tid, (lat, labels) in lattices.items():
            pred, _ = decode_lattice(lat, params)
            accs.append(point_accuracy(pred, labels)); w.append(len(labels))
        rows.append({"beta": beta, "emission_weight": we, "point_acc": float(np.average(accs, weights=w)),
                     "point_acc_trip_mean": float(np.mean(accs))})
        print(rows[-1], flush=True)
    grid_df = pd.DataFrame(rows); grid_df.to_csv(RESULTS / "hmm_grid_search_train.csv", index=False)
    best = grid_df.sort_values("point_acc", ascending=False).iloc[0]
    hmm_params = dict(cfg["hmm"], beta=float(best.beta), emission_weight=float(best.emission_weight))
    print("best decoder params:", hmm_params)
    piv = grid_df.pivot(index="beta", columns="emission_weight", values="point_acc")
    fig, ax = plt.subplots(figsize=(6, 4)); im = ax.imshow(piv.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(piv.columns)), piv.columns); ax.set_yticks(range(len(piv.index)), piv.index)
    ax.set_xlabel("emission weight"); ax.set_ylabel("beta (m)")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.4f}", ha="center", va="center", color="w", fontsize=8)
    ax.set_title(f"HMM grid search: point accuracy ({len(tune_trips)} train trips, OOF probs)")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(FIGURES / "hmm_grid_search.png", dpi=130); plt.close(fig)

    # ---------- 3. student-test evaluation
    test = pd.read_pickle(PROCESSED / "student_test_candidates.pkl").reset_index(drop=True)
    test["prob"] = model.predict_proba(test[FEATURES].to_numpy(np.float32))[:, 1]
    # distance-only baseline emission: Gaussian GPS noise model, sigma = 5 m (no classifier)
    test["prob_dist"] = np.exp(-0.5 * (test["dist"] / 5.0) ** 2)
    cache, res, base = {}, [], []
    for tid, g in test.groupby("trip_id"):
        r = evaluate_trip("student_test", tid, g, cfg, hmm_params, cache); res.append(r)
        b = evaluate_trip("student_test", tid, g, cfg, dict(hmm_params, emission_weight=1.0), cache, prob_col="prob_dist")
        base.append({"trip_id": tid, "distonly_hmm_point_acc": b["hmm_point_acc"], "distonly_hmm_continuity": b["hmm_continuity"],
                     "distonly_hmm_len_f1": b["hmm_len_f1"]})
        trip, net, xy, route, labels = cache[tid]
        m = route_map(trip.track, net, pred_route=r["hmm_route"], ref_route=route, clf_points=r["clf_points"],
                      network_arcs=nearby_arcs(net, xy), title=f"student_test {tid}: HMM acc {r['hmm_point_acc']:.3f}")
        m.save(str(MAPS / f"pred_student_test_{tid}.html"))
        static_route_plot(tid, xy, net, route, r)
        print(tid, {k: round(v, 4) for k, v in r.items() if isinstance(v, float)}, flush=True)
    df = pd.DataFrame(res).drop(columns=["clf_route", "hmm_route", "clf_points"]).merge(pd.DataFrame(base), on="trip_id")
    df.to_csv(RESULTS / "hmm_student_test_per_trip.csv", index=False)
    w = df.n_points
    summ = {}
    for c in [c for c in df.columns if c.startswith(("clf_", "hmm_", "distonly_"))]:
        summ[c] = float(np.average(df[c], weights=w)) if "point_acc" in c else float(df[c].mean())
    summ.update(total_breaks=int(df.breaks.sum()), total_skipped=int(df.skipped.sum()),
                mean_decode_s=float(df.decode_s.mean()), hmm_params=hmm_params, tune_trips=tune_trips)
    json.dump(summ, open(RESULTS / "hmm_student_test_summary.json", "w"), indent=2)
    print(json.dumps(summ, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5)); xs = np.arange(len(df)); wd = 0.27
    axes[0].bar(xs - wd, df.clf_point_acc, wd, label="classifier only")
    axes[0].bar(xs, df.hmm_point_acc, wd, label="classifier + HMM")
    axes[0].bar(xs + wd, df.distonly_hmm_point_acc, wd, label="distance-only HMM")
    axes[0].set_ylim(0.6, 1.0); axes[0].set_title("Point-wise accuracy per student-test trip")
    axes[1].bar(xs - wd / 2, df.clf_continuity, wd, label="classifier only"); axes[1].bar(xs + wd / 2, df.hmm_continuity, wd, label="classifier + HMM")
    axes[1].set_title("Route continuity per student-test trip")
    for ax in axes:
        ax.set_xticks(xs, df.trip_id, rotation=45); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / "hmm_vs_classifier.png", dpi=130); plt.close(fig)

    # ---------- 4. export inference configuration
    best_model = json.load(open(RESULTS / "bayes_opt_best.json"))
    export = {"feature_columns": FEATURES, "candidates": cfg["candidates"], "features": cfg["features"],
              "hmm": hmm_params, "model": best_model["model"], "model_params": best_model["best_params"],
              "random_seed": seed}
    json.dump(export, open(ARTIFACTS / "inference_config.json", "w"), indent=2)
    print("exported artifacts/inference_config.json")


if __name__ == "__main__":
    main()
