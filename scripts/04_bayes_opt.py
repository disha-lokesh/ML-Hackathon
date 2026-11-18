"""Bayesian hyperparameter optimisation (Optuna TPE sampler) of the selected model.

Objective : maximise mean observation-level Top-1 accuracy over a 3-fold GroupKFold
            (groups = trips) on the training partition only. Top-1 is the quantity
            the HMM ultimately relies on (ranking of candidates per GPS point);
            log loss is recorded per trial as a secondary probability-quality check.
The student-test partition is used only AFTER tuning, to compare tuned vs untuned.
Outputs: results/bayes_opt_trials.csv, results/bayes_opt_best.json, results/tuned_vs_untuned.csv,
         figures/bayes_opt_history.png, figures/bayes_opt_params.png,
         artifacts/final_model.joblib, data/processed/oof_tuned.npy, data/processed/test_prob_tuned.npy
"""
import argparse
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

from _common import ARTIFACTS, FIGURES, PROCESSED, RESULTS
from pathsnap import load_config
from pathsnap.features import FEATURES
from pathsnap.metrics import row_metrics, top1_accuracy
from pathsnap.training import cross_validate, fit_pipeline, make_models

SPACES = {
    # learning rate / n_estimators trade-off, tree depth (feature-interaction order),
    # row/column subsampling and L1/L2/min-child regularisation against overfitting to trips.
    "XGBoost": lambda tr, seed: XGBClassifier(
        n_estimators=tr.suggest_int("n_estimators", 100, 800, step=50),
        learning_rate=tr.suggest_float("learning_rate", 0.01, 0.3, log=True),
        max_depth=tr.suggest_int("max_depth", 3, 10),
        min_child_weight=tr.suggest_float("min_child_weight", 1.0, 50.0, log=True),
        subsample=tr.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=tr.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_lambda=tr.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        reg_alpha=tr.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
        gamma=tr.suggest_float("gamma", 1e-4, 5.0, log=True),
        tree_method="hist", eval_metric="logloss", n_jobs=-1, random_state=seed),
    "HistGradientBoosting": lambda tr, seed: HistGradientBoostingClassifier(
        max_iter=tr.suggest_int("max_iter", 100, 800, step=50),
        learning_rate=tr.suggest_float("learning_rate", 0.01, 0.3, log=True),
        max_leaf_nodes=tr.suggest_int("max_leaf_nodes", 15, 127),
        min_samples_leaf=tr.suggest_int("min_samples_leaf", 10, 200, log=True),
        l2_regularization=tr.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
        max_features=tr.suggest_float("max_features", 0.5, 1.0), random_state=seed),
    "RandomForest": lambda tr, seed: RandomForestClassifier(
        n_estimators=tr.suggest_int("n_estimators", 100, 400, step=50),
        max_depth=tr.suggest_int("max_depth", 8, 30),
        min_samples_leaf=tr.suggest_int("min_samples_leaf", 1, 50, log=True),
        max_features=tr.suggest_float("max_features", 0.3, 1.0), n_jobs=-1, random_state=seed),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="XGBoost", choices=list(SPACES))
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--folds", type=int, default=3)
    args = ap.parse_args()
    cfg = load_config(); seed = cfg["random_seed"]

    train_df = pd.read_pickle(PROCESSED / "train_subsample.pkl")
    test_df = pd.read_pickle(PROCESSED / "student_test_candidates.pkl").reset_index(drop=True)
    X = train_df[FEATURES].to_numpy(np.float32); y = train_df["label"].to_numpy(); groups = train_df["trip_id"].to_numpy()
    splits = list(GroupKFold(n_splits=args.folds).split(X, y, groups))

    def objective(trial):
        est = SPACES[args.model](trial, seed)
        top1s, lls = [], []
        t0 = time.perf_counter()
        for tr, va in splits:
            pipe, _ = fit_pipeline(args.model, est, X[tr], y[tr], seed)
            p = pipe.predict_proba(X[va])[:, 1]
            sub = train_df.iloc[va][["trip_id", "point", "label"]].copy(); sub["prob"] = p
            top1s.append(top1_accuracy(sub)); lls.append(row_metrics(y[va], p)["log_loss"])
        trial.set_user_attr("cv_log_loss", float(np.mean(lls)))
        trial.set_user_attr("cv_top1_std", float(np.std(top1s)))
        trial.set_user_attr("seconds", time.perf_counter() - t0)
        print(f"trial {trial.number}: top1={np.mean(top1s):.4f} ll={np.mean(lls):.4f} {trial.params}", flush=True)
        return float(np.mean(top1s))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=8),
                                study_name=f"pathsnap_{args.model}")
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=args.trials)
    opt_seconds = time.perf_counter() - t0
    trials = study.trials_dataframe()
    trials.to_csv(RESULTS / "bayes_opt_trials.csv", index=False)
    best = {"model": args.model, "best_cv_top1": study.best_value, "best_params": study.best_params,
            "n_trials": args.trials, "cv_folds": args.folds, "objective": "mean GroupKFold observation-level Top-1",
            "optimisation_seconds": opt_seconds}
    json.dump(best, open(RESULTS / "bayes_opt_best.json", "w"), indent=2)
    print(json.dumps(best, indent=2))

    # ---- plots
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(trials["number"], trials["value"], "o", label="trial")
    ax.plot(trials["number"], trials["value"].cummax(), "-", label="best so far")
    ax.set_xlabel("trial"); ax.set_ylabel("CV Top-1"); ax.set_title(f"Bayesian optimisation history ({args.model}, TPE)"); ax.legend()
    fig.tight_layout(); fig.savefig(FIGURES / "bayes_opt_history.png", dpi=130); plt.close(fig)
    pcols = [c for c in trials.columns if c.startswith("params_")]
    n = len(pcols); fig, axes = plt.subplots(int(np.ceil(n / 3)), 3, figsize=(13, 3.2 * np.ceil(n / 3)))
    for ax, c in zip(axes.ravel(), pcols):
        ax.scatter(trials[c], trials["value"], c=trials["number"], cmap="viridis", s=18)
        ax.set_title(c.replace("params_", "")); ax.set_ylabel("CV Top-1")
        if c.replace("params_", "") in ("learning_rate", "min_child_weight", "reg_lambda", "reg_alpha", "gamma", "l2_regularization"):
            ax.set_xscale("log")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(FIGURES / "bayes_opt_params.png", dpi=130); plt.close(fig)

    # ---- tuned vs untuned on identical partitions / metrics
    tuned_est = SPACES[args.model](optuna.trial.FixedTrial(study.best_params), seed)
    untuned_est = make_models(seed)[args.model][1]
    Xte = test_df[FEATURES].to_numpy(np.float32); yte = test_df["label"].to_numpy()
    n_pts = test_df[["trip_id", "point"]].drop_duplicates().shape[0]
    rows = []
    for tag, est in [("untuned", untuned_est), ("tuned", tuned_est)]:
        folds, oof = cross_validate(args.model, est, train_df, cfg["cv"]["n_splits"], seed)
        pipe, fit_s = fit_pipeline(args.model, est, X, y, seed)
        t1 = time.perf_counter(); p = pipe.predict_proba(Xte)[:, 1]; inf = time.perf_counter() - t1
        m = row_metrics(yte, p); sub = test_df[["trip_id", "point", "label"]].copy(); sub["prob"] = p
        rows.append({"variant": tag, "cv_f1_macro": folds["f1_macro"].mean(), "cv_top1": folds["top1"].mean(),
                     "cv_log_loss": folds["log_loss"].mean(), "test_top1": top1_accuracy(sub), "test_f1_macro": m["f1_macro"],
                     "test_precision_macro": m["precision_macro"], "test_recall_macro": m["recall_macro"],
                     "test_log_loss": m["log_loss"], "fit_time_s": fit_s, "inference_ms_per_1k_points": 1e6 * inf / n_pts})
        if tag == "tuned":
            np.save(PROCESSED / "oof_tuned.npy", oof); np.save(PROCESSED / "test_prob_tuned.npy", p)
            joblib.dump(pipe, ARTIFACTS / "final_model.joblib")
    res = pd.DataFrame(rows); res.to_csv(RESULTS / "tuned_vs_untuned.csv", index=False)
    print(res.to_string())


if __name__ == "__main__":
    main()
