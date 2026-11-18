"""Candidate generation and per-candidate feature engineering.

Every GPS observation i gets a candidate set C_i = up to K arcs within R metres.
Each (observation, candidate arc) pair becomes one row; the learning task is
binary: "is this candidate the reference arc for this observation?".
"""
import numpy as np
import pandas as pd

from .geo import LocalProjection, angle_diff_deg, bearing_deg, project_to_segments
from .network import RoadNetwork

FEATURES = [
    "dist", "dist_rank", "dist_gap", "dist_ratio", "n_cand",
    "head_diff", "head_cos", "head_valid", "speed",
    "arc_len", "t_proj", "at_end",
    "ctx_dist", "ctx_rank", "ctx_head_cos",
    "out_deg", "in_deg", "two_way", "dt_prev",
]


def prepare_trip(trip, cfg):
    """Project a trip and build its road network + spatial index."""
    proj = LocalProjection.from_track(trip.track)
    net = RoadNetwork(trip.nodes, trip.arcs, proj)
    xy = proj.to_xy(trip.track[:, :2])
    c = cfg["candidates"]
    net.build_index(xy, margin=c["radius_m"] + 50.0, step=c["densify_step_m"])
    return net, xy


def gps_kinematics(xy, t, window, min_disp):
    n = len(xy)
    i0 = np.clip(np.arange(n) - window, 0, n - 1)
    i1 = np.clip(np.arange(n) + window, 0, n - 1)
    d = xy[i1] - xy[i0]
    disp = np.hypot(d[:, 0], d[:, 1])
    dt = np.maximum(t[i1] - t[i0], 1e-6)
    heading = bearing_deg(d[:, 0], d[:, 1])
    valid = disp >= min_disp
    speed = disp / dt
    dt_prev = np.r_[1.0, np.diff(t)]
    return heading, valid, speed, dt_prev


def build_candidate_table(trip, cfg, net=None, xy=None):
    """Return (DataFrame of candidate rows, net, xy). No labels involved."""
    if net is None:
        net, xy = prepare_trip(trip, cfg)
    t = trip.track[:, 2]
    fc, cc = cfg["features"], cfg["candidates"]
    heading, hvalid, speed, dt_prev = gps_kinematics(xy, t, fc["heading_window"], fc["min_heading_displacement_m"])
    cands = net.query_candidates(xy, cc["radius_m"], cc["max_candidates"])

    n_c = np.array([len(c[0]) for c in cands])
    pi = np.repeat(np.arange(len(xy)), n_c)
    if len(pi) == 0:
        return pd.DataFrame(columns=["trip_id", "point", "arc"] + FEATURES), net, xy
    arc = np.concatenate([c[0] for c in cands]).astype(np.int64)
    dist = np.concatenate([c[1] for c in cands])
    tproj = np.concatenate([c[2] for c in cands])
    rank = np.concatenate([np.arange(k) for k in n_c])
    dmin = np.repeat(np.array([c[1][0] if len(c[1]) else np.nan for c in cands]), n_c)

    hd = angle_diff_deg(heading[pi], net.bearing[arc])
    hv = hvalid[pi]
    head_diff = np.where(hv, hd, 90.0)
    head_cos = np.where(hv, np.cos(np.radians(hd)), 0.0)

    # temporal context: mean distance / heading agreement of neighbouring points to the same arc
    W = fc["context_window"]
    n = len(xy)
    ctx_sum = np.zeros(len(pi)); ctx_cnt = np.zeros(len(pi))
    hc_sum = np.zeros(len(pi)); hc_cnt = np.zeros(len(pi))
    for off in range(-W, W + 1):
        if off == 0:
            continue
        j = pi + off
        ok = (j >= 0) & (j < n)
        jj = j[ok]
        d_, _ = project_to_segments(xy[jj], net.A[arc[ok]], net.B[arc[ok]])
        ctx_sum[ok] += d_; ctx_cnt[ok] += 1
        hvj = hvalid[jj]
        cosj = np.cos(np.radians(angle_diff_deg(heading[jj], net.bearing[arc[ok]])))
        idx = np.where(ok)[0][hvj]
        hc_sum[idx] += cosj[hvj]; hc_cnt[idx] += 1
    ctx_dist = (ctx_sum + dist) / (ctx_cnt + 1)
    ctx_head = np.where(hc_cnt > 0, hc_sum / np.maximum(hc_cnt, 1), 0.0)

    df = pd.DataFrame({
        "trip_id": trip.trip_id, "point": pi, "arc": arc,
        "dist": dist, "dist_rank": rank, "dist_gap": dist - dmin,
        "dist_ratio": dist / (dmin + 1.0), "n_cand": n_c[pi],
        "head_diff": head_diff, "head_cos": head_cos, "head_valid": hv.astype(np.int8),
        "speed": speed[pi], "arc_len": net.length[arc], "t_proj": tproj,
        "at_end": ((tproj <= 1e-6) | (tproj >= 1 - 1e-6)).astype(np.int8),
        "ctx_dist": ctx_dist, "ctx_head_cos": ctx_head,
        "out_deg": net.out_deg[net.arcs[arc, 1]], "in_deg": net.in_deg[net.arcs[arc, 0]],
        "two_way": net.two_way[arc], "dt_prev": dt_prev[pi],
    })
    df["ctx_rank"] = df.groupby("point")["ctx_dist"].rank(method="first") - 1
    return df, net, xy
