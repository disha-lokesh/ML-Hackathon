"""End-to-end inference for one trip (used by the API and by the evaluation scripts)."""
import json
import time
from pathlib import Path

import joblib
import numpy as np

from . import PROJECT_ROOT, load_config
from .features import FEATURES, build_candidate_table
from .hmm import classifier_only, reconstruct_route, viterbi_decode
from .labels import align_points_to_route
from .metrics import point_accuracy, route_continuity, route_overlap

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


class PathSnapModel:
    def __init__(self, artifact_dir=ARTIFACT_DIR):
        artifact_dir = Path(artifact_dir)
        self.model = joblib.load(artifact_dir / "final_model.joblib")   # sklearn Pipeline (scaler + clf)
        with open(artifact_dir / "inference_config.json", "r", encoding="utf-8") as f:
            self.cfg = json.load(f)
        self.features = self.cfg.get("feature_columns", FEATURES)
        self.hmm_params = self.cfg["hmm"]

    def predict(self, trip, use_hmm=True):
        """Pure prediction: never sees the reference route."""
        t0 = time.perf_counter()
        df, net, xy = build_candidate_table(trip, self.cfg)
        t1 = time.perf_counter()
        df["prob"] = self.model.predict_proba(df[self.features].to_numpy())[:, 1] if len(df) else []
        t2 = time.perf_counter()
        clf_pred = classifier_only(df, len(xy))
        if use_hmm:
            pred, info = viterbi_decode(df, xy, trip.track[:, 2], net, self.hmm_params)
        else:
            pred, info = clf_pred, {"breaks": 0, "skipped": int((clf_pred < 0).sum())}
        t3 = time.perf_counter()
        route, comp, unfilled = reconstruct_route(pred, net)
        t4 = time.perf_counter()
        return {
            "df": df, "net": net, "xy": xy,
            "pred_points": pred, "clf_pred_points": clf_pred,
            "route": route, "compressed": comp, "unfilled_gaps": unfilled,
            "decode_info": info,
            "timing_s": {"candidates_features": t1 - t0, "classifier": t2 - t1,
                         "viterbi": t3 - t2, "route_reconstruction": t4 - t3, "total": t4 - t0},
        }


def evaluate_prediction(out, ref_route):
    """Server-side scoring against the reference route (kept separate from predict)."""
    net, xy = out["net"], out["xy"]
    labels, _, _ = align_points_to_route(xy, net, np.asarray(ref_route))
    df = out["df"]
    cand_sets = df.groupby("point")["arc"].apply(set).to_dict() if len(df) else {}
    miss = np.mean([labels[i] not in cand_sets.get(i, set()) for i in range(len(labels))])
    clf_route, clf_comp, _ = reconstruct_route(out["clf_pred_points"], net)
    res = {
        "n_points": int(len(xy)),
        "point_accuracy_hmm": point_accuracy(out["pred_points"], labels),
        "point_accuracy_classifier_only": point_accuracy(out["clf_pred_points"], labels),
        "candidate_miss_rate": float(miss),
        "route_continuity_hmm": route_continuity(out["compressed"], net),
        "route_continuity_classifier_only": route_continuity(clf_comp, net),
    }
    for k, v in route_overlap(out["route"], ref_route, net).items():
        res[f"{k}_hmm"] = v
    for k, v in route_overlap(clf_route, ref_route, net).items():
        res[f"{k}_classifier_only"] = v
    return res
