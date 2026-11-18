"""HMM / Viterbi route reconstruction.

Hidden state at observation i : one candidate road arc c in C_i (with its
projected position t along the arc) - so the candidate->arc correspondence is
kept explicitly in every state.

Emission  log P(z_i | c)      = w_e * log(max(p_clf(c | features_i), floor))
          (the tuned classifier's probability that c is the true arc).

Transition log P(c_i | c_{i-1}) = -| d_route - d_gc | / (beta * sqrt(max(dt,1)))
          d_gc    = straight-line distance between the two GPS fixes
          d_route = network driving distance from the projection of fix i-1 on
                    c_{i-1} to the projection of fix i on c_i (Dijkstra over the
                    directed graph, bounded by a cutoff that grows with dt).
          (Newson & Krumm 2009 style; scaling by sqrt(dt) relaxes the
          constraint across GPS gaps.)
          Unreachable within the cutoff -> -inf (invalid transition).

Viterbi recurrence:
    delta_i(c) = E_i(c) + max_{c'} [ delta_{i-1}(c') + T(c', c) ]

Robustness:
  * observations with an empty candidate set are skipped (predicted -1) and the
    next observation is connected to the last valid one with the accumulated dt.
  * if every transition into step i is -inf (disconnected network / huge jump)
    the chain is "broken": decoding is restarted at i from the emissions only
    and the break is counted.
"""
import numpy as np



def _route_dist(net, a, ta, b, tb, cutoff):
    la, lb = net.length[a], net.length[b]
    pa, pb = ta * la, tb * lb
    if a == b:
        return abs(pb - pa)
    sp = net.dijkstra(int(net.arcs[a, 1]), cutoff)
    hit = sp.get(int(net.arcs[b, 0]))
    if hit is None:
        return None
    return (la - pa) + hit[0] + pb


def build_lattice(cand_df, xy, times, net, params, prob_col="prob"):
    """Pre-compute everything that does not depend on (beta, emission weight):
    candidate states per observation and |d_route - d_gc| for every transition."""
    n = len(xy)
    cb, cg, cs = params["sp_cutoff_base_m"], params["sp_cutoff_gc_factor"], params["sp_cutoff_per_second_m"]
    g = cand_df.sort_values(["point", prob_col], ascending=[True, False])
    pts = g["point"].to_numpy(); arcs = g["arc"].to_numpy(); tp = g["t_proj"].to_numpy()
    prob = g[prob_col].to_numpy()
    starts = np.searchsorted(pts, np.arange(n)); ends = np.searchsorted(pts, np.arange(n), side="right")
    valid_pts = [i for i in range(n) if ends[i] > starts[i]]
    steps = []      # per valid point: (point, slice, cost matrix from previous valid point or None, dt)
    prev = None
    for i in valid_pts:
        s, e = starts[i], ends[i]
        if prev is None:
            steps.append((i, s, e, None, 1.0)); prev = i; continue
        ps, pe = starts[prev], ends[prev]
        gc = float(np.hypot(*(xy[i] - xy[prev])))
        dt = max(float(times[i] - times[prev]), 1.0)
        cutoff = cb + cg * gc + cs * dt
        C = np.full((pe - ps, e - s), np.inf)
        for u in range(pe - ps):
            a, ta = arcs[ps + u], tp[ps + u]
            for v in range(e - s):
                rd = _route_dist(net, a, ta, arcs[s + v], tp[s + v], cutoff)
                if rd is not None:
                    C[u, v] = abs(rd - gc)
        steps.append((i, s, e, C, dt)); prev = i
    return {"n": n, "arcs": arcs, "prob": prob, "steps": steps, "skipped": n - len(valid_pts)}


def decode_lattice(lat, params):
    """Viterbi over a pre-built lattice for given beta / emission weight."""
    n, arcs, steps = lat["n"], lat["arcs"], lat["steps"]
    pred = np.full(n, -1, dtype=np.int64)
    if not steps:
        return pred, {"breaks": 0, "skipped": n}
    beta = params["beta"]; we = params["emission_weight"]
    em_all = we * np.log(np.maximum(lat["prob"], params["prob_floor"]))
    segments, cur, delta, breaks = [], [], None, 0
    for (i, s, e, C, dt) in steps:
        E = em_all[s:e]
        if C is None:
            delta = E.copy(); cur = [(i, s, None)]; continue
        M = delta[:, None] - C / (beta * np.sqrt(dt))
        bp = np.argmax(M, axis=0)
        best = M[bp, np.arange(e - s)]
        if not np.any(np.isfinite(best)):
            breaks += 1
            segments.append((cur, delta))
            delta = E.copy(); cur = [(i, s, None)]
        else:
            delta = best + E
            cur.append((i, s, bp))
    segments.append((cur, delta))
    for seg, d in segments:
        k = int(np.argmax(d))
        for i, s, bp in reversed(seg):
            pred[i] = arcs[s + k]
            if bp is not None:
                k = int(bp[k])
    return pred, {"breaks": breaks, "skipped": lat["skipped"]}


def viterbi_decode(cand_df, xy, times, net, params, prob_col="prob"):
    """cand_df: rows (point, arc, t_proj, prob). Returns (pred_arc per point, info)."""
    if len(cand_df) == 0:
        return np.full(len(xy), -1, dtype=np.int64), {"breaks": 0, "skipped": len(xy)}
    lat = build_lattice(cand_df, xy, times, net, params, prob_col)
    return decode_lattice(lat, params)


def classifier_only(cand_df, n, prob_col="prob"):
    pred = np.full(n, -1, dtype=np.int64)
    if len(cand_df):
        best = cand_df.loc[cand_df.groupby("point")[prob_col].idxmax()]
        pred[best["point"].to_numpy()] = best["arc"].to_numpy()
    return pred


def reconstruct_route(pred_points, net, cutoff=3000.0):
    """Turn per-point arcs into a continuous arc sequence.

    Consecutive duplicates are merged; when two consecutive distinct arcs are not
    directly connected the gap is filled with the network shortest path.
    Returns (route_arcs, compressed_sequence, n_unfilled_gaps)."""
    seq = [int(a) for a in pred_points if a >= 0]
    comp = [a for k, a in enumerate(seq) if k == 0 or a != seq[k - 1]]
    route, unfilled = [], 0
    for k, a in enumerate(comp):
        if k > 0:
            b = comp[k - 1]
            if not net.connected(b, a):
                path = net.path_arcs(int(net.arcs[b, 1]), int(net.arcs[a, 0]), cutoff)
                if path is None:
                    unfilled += 1
                else:
                    route.extend(path)
        route.append(a)
    return route, comp, unfilled
