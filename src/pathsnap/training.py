"""Shared training utilities: data loading/filtering, model zoo, CV loop."""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .features import FEATURES
from .metrics import row_metrics, top1_accuracy

# Points whose GPS fix lies further than this from the aligned reference arc have
# unreliable labels (the reference route and the GPS disagree); they are kept for
# evaluation but excluded from model fitting.
MAX_ALIGN_DIST = 60.0
# Trips where the GPS trace systematically disagrees with the reference route.
MAX_TRIP_MEDIAN_ALIGN = 50.0


def load_training_table(path, point_fraction=1.0, seed=42, quality_csv=None):
    df = pd.read_pickle(path)
    bad_trips = []
    if quality_csv is not None:
        q = pd.read_csv(quality_csv, dtype={"trip_id": str})
        bad_trips = sorted(q.loc[q["align_dist_median_m"] > MAX_TRIP_MEDIAN_ALIGN, "trip_id"])
    df = df[~df["trip_id"].isin(bad_trips) & (df["align_dist"] <= MAX_ALIGN_DIST)]
    if point_fraction < 1.0:
        # subsample whole observations (all their candidates) - 1 Hz fixes are highly
        # autocorrelated, so this loses little information but speeds up CV a lot.
        rng = np.random.default_rng(seed)
        keys = df[["trip_id", "point"]].drop_duplicates()
        keys = keys[rng.random(len(keys)) < point_fraction]
        df = df.merge(keys, on=["trip_id", "point"])
    return df.reset_index(drop=True), bad_trips


def make_models(seed=42):
    """name -> (family, estimator, short config description)."""
    return {
        "LogisticRegression": ("Linear", LogisticRegression(C=1.0, max_iter=1000),
                               "L2, C=1.0, lbfgs"),
        "GaussianNB": ("Probabilistic", GaussianNB(var_smoothing=1e-9),
                       "var_smoothing=1e-9"),
        "KNN": ("Instance-based", KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1),
                "k=25, distance-weighted, train cap 150k rows"),
        "DecisionTree": ("Tree", DecisionTreeClassifier(max_depth=12, min_samples_leaf=50, random_state=seed),
                         "max_depth=12, min_leaf=50, gini"),
        "RandomForest": ("Ensemble (bagging)", RandomForestClassifier(n_estimators=200, max_depth=18, min_samples_leaf=5,
                                                                       n_jobs=-1, random_state=seed),
                         "200 trees, max_depth=18, min_leaf=5"),
        "HistGradientBoosting": ("Boosting", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                                                            max_leaf_nodes=31, random_state=seed),
                                 "300 iters, lr=0.1, 31 leaves"),
        "XGBoost": ("Boosting", XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6, subsample=0.8,
                                              colsample_bytree=0.8, tree_method="hist", eval_metric="logloss",
                                              n_jobs=-1, random_state=seed),
                    "300 trees, lr=0.1, depth=6, subsample=0.8"),
        "MLP": ("Neural network", MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=1024,
                                                learning_rate_init=1e-3, max_iter=30, early_stopping=True,
                                                n_iter_no_change=5, random_state=seed),
                "64-32 ReLU, Adam lr=1e-3, alpha=1e-4, early stopping"),
    }


KNN_TRAIN_CAP = 150_000


def pipeline_for(est):
    return Pipeline([("scaler", StandardScaler()), ("clf", est)])


def fit_pipeline(name, est, X, y, seed=42):
    from sklearn.base import clone
    pipe = pipeline_for(clone(est))
    if name == "KNN" and len(X) > KNN_TRAIN_CAP:
        idx = np.random.default_rng(seed).choice(len(X), KNN_TRAIN_CAP, replace=False)
        X, y = X[idx], y[idx]
    t0 = time.perf_counter()
    pipe.fit(X, y)
    return pipe, time.perf_counter() - t0


def cross_validate(name, est, df, n_splits=5, seed=42):
    """Group-aware (by trip) K-fold CV. Returns per-fold metrics and OOF probabilities."""
    X = df[FEATURES].to_numpy(np.float32); y = df["label"].to_numpy()
    groups = df["trip_id"].to_numpy()
    oof = np.zeros(len(df))
    folds = []
    for k, (tr, va) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups)):
        pipe, fit_s = fit_pipeline(name, est, X[tr], y[tr], seed)
        t0 = time.perf_counter()
        p = pipe.predict_proba(X[va])[:, 1]
        inf_s = time.perf_counter() - t0
        oof[va] = p
        m = row_metrics(y[va], p)
        sub = df.iloc[va][["trip_id", "point", "label"]].copy(); sub["prob"] = p
        m.update(fold=k, top1=top1_accuracy(sub), fit_s=fit_s, infer_s=inf_s, n_val=len(va))
        m["train_f1_macro"] = row_metrics(y[tr][:200000], pipe.predict_proba(X[tr][:200000])[:, 1])["f1_macro"]
        folds.append(m)
    return pd.DataFrame(folds), oof
