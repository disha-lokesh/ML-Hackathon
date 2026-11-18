"""Directed road network with a spatial index and bounded Dijkstra."""
import heapq

import numpy as np
from scipy.spatial import cKDTree

from .geo import bearing_deg, project_to_segments


class RoadNetwork:
    def __init__(self, nodes_lonlat, arcs, proj):
        self.proj = proj
        self.arcs = np.asarray(arcs, dtype=np.int64)
        self.n_nodes = len(nodes_lonlat)
        self.node_xy = proj.to_xy(nodes_lonlat)
        self.A = self.node_xy[self.arcs[:, 0]]
        self.B = self.node_xy[self.arcs[:, 1]]
        d = self.B - self.A
        self.length = np.hypot(d[:, 0], d[:, 1])
        self.bearing = bearing_deg(d[:, 0], d[:, 1])

        self.out_deg = np.bincount(self.arcs[:, 0], minlength=self.n_nodes)
        self.in_deg = np.bincount(self.arcs[:, 1], minlength=self.n_nodes)
        pairs = set(map(tuple, self.arcs.tolist()))
        self.two_way = np.array([(v, u) in pairs for u, v in self.arcs.tolist()], dtype=np.int8)

        # adjacency: node -> outgoing arc ids
        order = np.argsort(self.arcs[:, 0], kind="stable")
        self._out_order = order
        self._out_ptr = np.r_[0, np.cumsum(self.out_deg)]
        self._sp_cache = {}
        self._index = None

    # ------------------------------------------------------------------ spatial
    def build_index(self, xy_points, margin, step):
        """KD-tree over densified arcs that intersect the bbox of xy_points (+margin)."""
        lo = xy_points.min(axis=0) - margin
        hi = xy_points.max(axis=0) + margin
        amin = np.minimum(self.A, self.B)
        amax = np.maximum(self.A, self.B)
        sel = np.where((amax[:, 0] >= lo[0]) & (amin[:, 0] <= hi[0]) &
                       (amax[:, 1] >= lo[1]) & (amin[:, 1] <= hi[1]))[0]
        n = np.maximum(np.ceil(self.length[sel] / step).astype(int), 1) + 1
        arc_rep = np.repeat(sel, n)
        frac = np.concatenate([np.linspace(0, 1, k) for k in n]) if len(sel) else np.zeros(0)
        pts = self.A[arc_rep] + frac[:, None] * (self.B[arc_rep] - self.A[arc_rep])
        self._index = (cKDTree(pts) if len(pts) else None, arc_rep, step)
        return len(sel)

    def query_candidates(self, xy, radius, k):
        """For each point return (arc_ids, dist, t) of the <=k nearest arcs within radius."""
        tree, arc_rep, step = self._index
        out = []
        if tree is None:
            return [(np.zeros(0, int), np.zeros(0), np.zeros(0)) for _ in range(len(xy))]
        hits = tree.query_ball_point(xy, r=radius + step / 2.0)
        for p, h in zip(xy, hits):
            if not h:
                out.append((np.zeros(0, int), np.zeros(0), np.zeros(0)))
                continue
            ids = np.unique(arc_rep[h])
            P = np.repeat(p[None, :], len(ids), axis=0)
            d, t = project_to_segments(P, self.A[ids], self.B[ids])
            m = d <= radius
            ids, d, t = ids[m], d[m], t[m]
            o = np.argsort(d, kind="stable")[:k]
            out.append((ids[o], d[o], t[o]))
        return out

    # ------------------------------------------------------------------ graph
    def out_arcs(self, node):
        return self._out_order[self._out_ptr[node]:self._out_ptr[node + 1]]

    def dijkstra(self, src, cutoff):
        """Shortest network distances from node src up to cutoff metres (cached).

        Returns dict node -> (dist, predecessor_arc)."""
        c = self._sp_cache.get(src)
        if c is not None and c[0] >= cutoff:
            return c[1]
        dist = {src: (0.0, -1)}
        heap = [(0.0, src)]
        arcs, length = self.arcs, self.length
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u][0]:
                continue
            for a in self.out_arcs(u):
                v = int(arcs[a, 1])
                nd = d + length[a]
                if nd <= cutoff and (v not in dist or nd < dist[v][0]):
                    dist[v] = (nd, int(a))
                    heapq.heappush(heap, (nd, v))
        self._sp_cache[src] = (cutoff, dist)
        return dist

    def path_arcs(self, src, dst, cutoff):
        """Arc sequence of the shortest path src->dst, or None if not within cutoff."""
        sp = self.dijkstra(src, cutoff)
        if dst not in sp:
            return None
        path = []
        v = dst
        while v != src:
            a = sp[v][1]
            path.append(a)
            v = int(self.arcs[a, 0])
        return path[::-1]

    def connected(self, a, b):
        return self.arcs[a, 1] == self.arcs[b, 0]

    def arcs_lonlat(self, arc_ids):
        ids = np.asarray(arc_ids, dtype=int)
        return self.proj.to_lonlat(self.A[ids]), self.proj.to_lonlat(self.B[ids])
