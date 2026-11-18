"""PathSnap FastAPI backend (runs on Laptop 1).

Endpoints
  GET  /health            liveness + loaded model info
  GET  /trips             the reserved evaluation trips available on the server
  POST /predict           {"trip_id": "..."} -> full pipeline + server-side scoring

The reference routes are read ONLY inside evaluate_prediction(), after prediction,
and are never returned to the client: the response contains the predicted route,
the observed GPS points and the accuracy metrics only.

Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000
Env:  PATHSNAP_EVAL_PARTITION (default eval_test; falls back to student_test if empty)
"""
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap import load_config  # noqa: E402
from pathsnap.io import list_trips, load_route, load_trip  # noqa: E402
from pathsnap.pipeline import PathSnapModel, evaluate_prediction  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout),
                              logging.FileHandler(ROOT / "results" / "api_requests.log", encoding="utf-8")])
log = logging.getLogger("pathsnap.api")

CFG = load_config()
PARTITION = os.environ.get("PATHSNAP_EVAL_PARTITION", "eval_test")
if not list_trips(PARTITION, CFG):
    log.warning("No trips found in partition '%s' - falling back to 'student_test' (development mode)", PARTITION)
    PARTITION = "student_test"

MODEL = PathSnapModel()
log.info("Loaded model %s with HMM params %s; serving partition '%s' (%d trips)",
         MODEL.cfg.get("model"), MODEL.hmm_params, PARTITION, len(list_trips(PARTITION, CFG)))

app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    trip_id: str
    use_hmm: bool = True


def _polylines(net, route):
    """Group consecutive connected arcs into polylines of [lat, lon]."""
    lines, cur = [], []
    for k, a in enumerate(route):
        A, B = net.arcs_lonlat([a])
        if cur and k > 0 and net.connected(route[k - 1], a):
            cur.append([float(B[0, 1]), float(B[0, 0])])
        else:
            if cur:
                lines.append(cur)
            cur = [[float(A[0, 1]), float(A[0, 0])], [float(B[0, 1]), float(B[0, 0])]]
    if cur:
        lines.append(cur)
    return lines


@app.middleware("http")
async def log_requests(request, call_next):
    t0 = time.perf_counter()
    resp = await call_next(request)
    log.info("%s %s from %s -> %d (%.1f ms)", request.method, request.url.path,
             request.client.host if request.client else "?", resp.status_code, 1000 * (time.perf_counter() - t0))
    return resp


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL.cfg.get("model"), "hmm": MODEL.hmm_params, "partition": PARTITION}


@app.get("/trips")
def trips():
    out = []
    for tid in list_trips(PARTITION, CFG):
        trip = load_trip(PARTITION, tid, CFG)
        out.append({"trip_id": tid, "n_points": int(len(trip.track)),
                    "duration_min": round(float(trip.track[-1, 2] - trip.track[0, 2]) / 60, 1)})
    return {"partition": PARTITION, "trips": out}


@app.post("/predict")
def predict(req: PredictRequest):
    if req.trip_id not in list_trips(PARTITION, CFG):
        raise HTTPException(404, f"unknown trip {req.trip_id}")
    log.info("PREDICT trip=%s use_hmm=%s", req.trip_id, req.use_hmm)
    trip = load_trip(PARTITION, req.trip_id, CFG)
    out = MODEL.predict(trip, use_hmm=req.use_hmm)          # prediction never sees the route
    ref = load_route(PARTITION, req.trip_id, CFG)           # server-side only
    metrics = evaluate_prediction(out, ref) if ref is not None else {}
    net = out["net"]
    from pathsnap.hmm import reconstruct_route
    clf_route, _, _ = reconstruct_route(out["clf_pred_points"], net)
    step = max(1, len(trip.track) // 3000)
    resp = {
        "trip_id": req.trip_id,
        "n_points": int(len(trip.track)),
        "gps": [[float(lat), float(lon), float(t)] for lon, lat, t in trip.track[::step]],
        "predicted_route": _polylines(net, out["route"]),
        "classifier_only_route": _polylines(net, clf_route),
        "n_predicted_arcs": len(out["route"]),
        "matched_points": int((out["pred_points"] >= 0).sum()),
        "decode_info": out["decode_info"],
        "timing_s": {k: round(v, 3) for k, v in out["timing_s"].items()},
        "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
    }
    log.info("RESULT trip=%s points=%d hmm_acc=%.4f clf_acc=%.4f continuity=%.3f len_f1=%.4f total=%.2fs",
             req.trip_id, resp["n_points"], metrics.get("point_accuracy_hmm", np.nan),
             metrics.get("point_accuracy_classifier_only", np.nan), metrics.get("route_continuity_hmm", np.nan),
             metrics.get("len_f1_hmm", np.nan), out["timing_s"]["total"])
    return resp
