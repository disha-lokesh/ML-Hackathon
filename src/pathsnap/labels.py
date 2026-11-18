"""Derive per-GPS-point reference labels from the reference route.

The .route file only lists the chronological arcs travelled; it does NOT say
which GPS point belongs to which arc. We therefore align the N GPS points to the
M route arcs with a monotone dynamic programme:

    cost(i, j) = distance(point i, route arc j)
    dp[i, j]   = cost(i, j) + min_{j' <= j} dp[i-1, j']

i.e. the assigned route index can only stay or move forward in time (arcs that
are traversed in < 1 s may be skipped). This respects route order, which plain
nearest-arc snapping does not (e.g. on a road driven twice in opposite directions).
"""
import numpy as np

from .geo import project_to_segments


def align_points_to_route(xy, network, route):
    N, M = len(xy), len(route)
    A, B = network.A[route], network.B[route]
    cost = np.empty((N, M))
    for j in range(M):
        cost[:, j], _ = project_to_segments(xy, np.repeat(A[j][None], N, 0), np.repeat(B[j][None], N, 0))

    back = np.empty((N, M), dtype=np.int32)
    dp = cost[0].copy()
    idx = np.arange(M)
    for i in range(1, N):
        # running prefix minimum and its argmin
        pm = np.minimum.accumulate(dp)
        is_new = np.r_[True, dp[1:] < pm[:-1]]
        arg = np.maximum.accumulate(np.where(is_new, idx, 0))
        back[i] = arg
        dp = cost[i] + pm
    j = int(np.argmin(dp))
    path = np.empty(N, dtype=np.int64)
    for i in range(N - 1, -1, -1):
        path[i] = j
        if i > 0:
            j = back[i, j]
    labels = route[path]
    dist = cost[np.arange(N), path]
    return labels, path, dist
