"""Train and compare >= 5 candidate-classification models.

* 5-fold GroupKFold on the 80 training trips (groups = trip ids -> no trip leaks across folds)
* each model refit on all training trips, then evaluated on the 10 student-testing trips
* metrics: accuracy, macro precision/recall/F1, log loss, observation-level Top-1,
  candidate miss rate, classifier-only route continuity, training / inference time
Outputs: results/model_comparison.csv, results/cv_folds_all_models.csv, figures/*.png,
         data/processed/oof_<model>.npy (out-of-fold probabilities for HMM tuning)
"""
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, log_loss
from sklearn.model_selection import GroupShuffleSplit

from _common import FIGURES, PROCESSED, RESULTS
from pathsnap import load_config
from pathsnap.features import FEATURES, prepare_trip
from pathsnap.hmm import classifier_only
from pathsnap.io import load_trip
from pathsnap.metrics import route_continuity, row_metrics, top1_accuracy
from pathsnap.training import cross_validate, fit_pipeline, load_training_table, make_models


def student_test_networks(test_df, cfg):
    nets = {}
    for tid in test_df["trip_id"].unique():
        net, xy = prepare_trip(load_trip("student_test", tid, cfg), cfg)
        nets[tid] = (net, len(xy))
    return nets


def classifier_continuity(test_df, prob, nets):
    vals = []
    d = test_df[["trip_id", "point", "arc"]].copy(); d["prob"] = prob
    for tid, g in d.groupby("trip_id"):
        net, n = nets[tid]
        pred = classifier_only(g.reset_index(drop=True), n)
        seq = [a for a in pred if a >= 0]
        comp = [a for k, a in enumerate(seq) if k == 0 or a != seq[k - 1]]
        vals.append(route_continuity(comp, net))
    return float(np.mean(vals))


def mlp_learning_curve(train_df, seed, epochs=40):
    """Epoch-wise train/validation log loss for the MLP (genuine training history)."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr, va = next(gss.split(train_df, groups=train_df["trip_id"]))
    X = train_df[FEATURES].to_numpy(np.float32); y = train_df["label"].to_numpy()
    sc = StandardScaler().fit(X[tr]); Xtr, Xva = sc.transform(X[tr]), sc.transform(X[va])
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=1024, learning_rate_init=1e-3,
                        random_state=seed)
    rng = np.random.default_rng(seed); hist = []
    for ep in range(epochs):
        idx = rng.permutation(len(Xtr))
        for s in range(0, len(idx), 1024):
            b = idx[s:s + 1024]
            mlp.partial_fit(Xtr[b], y[tr][b], classes=[0, 1])
        hist.append({"epoch": ep + 1,
                     "train_log_loss": log_loss(y[tr], mlp.predict_proba(Xtr)[:, 1]),
                     "val_log_loss": log_loss(y[va], mlp.predict_proba(Xva)[:, 1])})
    return pd.DataFrame(hist)


def xgb_learning_curve(train_df, seed):
    from xgboost import XGBClassifier
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr, va = next(gss.split(train_df, groups=train_df["trip_id"]))
    X = train_df[FEATURES].to_numpy(np.float32); y = train_df["label"].to_numpy()
    m = XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                      tree_method="hist", eval_metric="logloss", n_jobs=-1, random_state=seed)
    m.fit(X[tr], y[tr], eval_set=[(X[tr], y[tr]), (X[va], y[va])], verbose=False)
    r = m.evals_result()
    return pd.DataFrame({"round": np.arange(1, len(r["validation_0"]["logloss"]) + 1),
                         "train_log_loss": r["validation_0"]["logloss"], "val_log_loss": r["validation_1"]["logloss"]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--point-fraction", type=float, default=0.35)
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_config(); seed = cfg["random_seed"]; np.random.seed(seed)

    train_df, bad = load_training_table(PROCESSED / "train_candidates.pkl", args.point_fraction, seed,
                                         RESULTS / "label_quality_train.csv")
    test_df = pd.read_pickle(PROCESSED / "student_test_candidates.pkl").reset_index(drop=True)
    print(f"train rows={len(train_df)} (excluded trips {bad}), test rows={len(test_df)}")
    train_df.to_pickle(PROCESSED / "train_subsample.pkl")
    nets = student_test_networks(test_df, cfg)
    Xte = test_df[FEATURES].to_numpy(np.float32); yte = test_df["label"].to_numpy()
    n_test_points = test_df[["trip_id", "point"]].drop_duplicates().shape[0]

    models = make_models(seed)
    if args.models:
        models = {k: v for k, v in models.items() if k in args.models}
    rows, all_folds = [], []
    fig_cm, axes_cm = plt.subplots(2, 4, figsize=(18, 8))
    for ax, (name, (family, est, desc)) in zip(axes_cm.ravel(), models.items()):
        print(f"== {name}", flush=True)
        folds, oof = cross_validate(name, est, train_df, cfg["cv"]["n_splits"], seed)
        folds.insert(0, "model", name); all_folds.append(folds)
        np.save(PROCESSED / f"oof_{name}.npy", oof)
        pipe, fit_s = fit_pipeline(name, est, train_df[FEATURES].to_numpy(np.float32), train_df["label"].to_numpy(), seed)
        t0 = time.perf_counter(); p = pipe.predict_proba(Xte)[:, 1]; inf_s = time.perf_counter() - t0
        np.save(PROCESSED / f"test_prob_{name}.npy", p)
        m = row_metrics(yte, p)
        sub = test_df[["trip_id", "point", "label"]].copy(); sub["prob"] = p
        cont = classifier_continuity(test_df, p, nets)
        rows.append({
            "model": name, "family": family, "config": desc,
            "cv_f1_macro_mean": folds["f1_macro"].mean(), "cv_f1_macro_std": folds["f1_macro"].std(),
            "cv_top1_mean": folds["top1"].mean(), "cv_top1_std": folds["top1"].std(),
            "cv_log_loss_mean": folds["log_loss"].mean(), "cv_train_f1_macro_mean": folds["train_f1_macro"].mean(),
            "test_accuracy": m["accuracy"], "test_precision_macro": m["precision_macro"],
            "test_recall_macro": m["recall_macro"], "test_f1_macro": m["f1_macro"], "test_f1_pos": m["f1_pos"],
            "test_log_loss": m["log_loss"], "test_top1": top1_accuracy(sub),
            "test_route_continuity_clf_only": cont,
            "fit_time_s": fit_s, "test_inference_s": inf_s,
            "inference_ms_per_1k_points": 1e6 * inf_s / n_test_points,
        })
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}), flush=True)
        cm = confusion_matrix(yte, (p >= 0.5).astype(int), normalize="true")
        ConfusionMatrixDisplay(cm, display_labels=["not ref", "ref arc"]).plot(ax=ax, colorbar=False, values_format=".3f")
        ax.set_title(name)
    fig_cm.suptitle("Student-test confusion matrices (row-normalised, threshold 0.5)")
    fig_cm.tight_layout(); fig_cm.savefig(FIGURES / "confusion_matrices.png", dpi=130); plt.close(fig_cm)

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "model_comparison.csv", index=False)
    pd.concat(all_folds).to_csv(RESULTS / "cv_folds_all_models.csv", index=False)

    # ---- comparison plots
    folds = pd.concat(all_folds)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    order = res.sort_values("cv_f1_macro_mean")["model"]
    ax.boxplot([folds.loc[folds.model == m, "f1_macro"] for m in order], tick_labels=order, vert=False)
    ax.set_xlabel("Macro F1 (per fold)"); ax.set_title("5-fold GroupKFold CV Macro F1 (train partition)")
    fig.tight_layout(); fig.savefig(FIGURES / "cv_macro_f1.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.5)); w = 0.38; xs = np.arange(len(res))
    ax.bar(xs - w / 2, res["test_top1"], w, label="Top-1 accuracy")
    ax.bar(xs + w / 2, res["test_f1_macro"], w, label="Macro F1")
    ax.set_xticks(xs, res["model"], rotation=25, ha="right"); ax.set_ylim(0.5, 1.0); ax.legend()
    ax.set_title("Student-test partition: Top-1 accuracy and Macro F1")
    fig.tight_layout(); fig.savefig(FIGURES / "student_test_top1_f1.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(xs - w / 2, res["cv_log_loss_mean"], w, label="CV"); ax.bar(xs + w / 2, res["test_log_loss"], w, label="Student test")
    ax.set_xticks(xs, res["model"], rotation=25, ha="right"); ax.set_yscale("log"); ax.legend()
    ax.set_title("Log loss (lower is better, log scale)")
    fig.tight_layout(); fig.savefig(FIGURES / "log_loss.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(res["inference_ms_per_1k_points"], res["test_top1"])
    for _, r in res.iterrows():
        ax.annotate(r["model"], (r["inference_ms_per_1k_points"], r["test_top1"]), fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("Inference time (ms per 1000 GPS points)"); ax.set_ylabel("Student-test Top-1")
    ax.set_title("Accuracy vs inference cost"); fig.tight_layout(); fig.savefig(FIGURES / "accuracy_vs_speed.png", dpi=130); plt.close(fig)

    # ---- genuine training histories (only for models that have them)
    if not args.models or "MLP" in args.models:
        h = mlp_learning_curve(train_df, seed); h.to_csv(RESULTS / "mlp_learning_curve.csv", index=False)
        fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(h.epoch, h.train_log_loss, label="train"); ax.plot(h.epoch, h.val_log_loss, label="validation (held-out trips)")
        ax.set_xlabel("epoch"); ax.set_ylabel("log loss"); ax.set_title("MLP training history"); ax.legend()
        fig.tight_layout(); fig.savefig(FIGURES / "mlp_loss_curve.png", dpi=130); plt.close(fig)
    if not args.models or "XGBoost" in args.models:
        h = xgb_learning_curve(train_df, seed); h.to_csv(RESULTS / "xgb_learning_curve.csv", index=False)
        fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(h["round"], h.train_log_loss, label="train"); ax.plot(h["round"], h.val_log_loss, label="validation (held-out trips)")
        ax.set_xlabel("boosting round"); ax.set_ylabel("log loss"); ax.set_title("XGBoost training history"); ax.legend()
        fig.tight_layout(); fig.savefig(FIGURES / "xgb_loss_curve.png", dpi=130); plt.close(fig)
    print(res.to_string())


if __name__ == "__main__":
    main()
