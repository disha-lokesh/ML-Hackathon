"""Geodesy helpers.

Each trip covers at most a few tens of km, so we use a local equirectangular
(tangent-plane) projection centred on the trip's GPS centroid. Over 20 km the
distortion of this projection is < 0.1 %, far below GPS noise (~5 m), while it
lets us use fast Euclidean geometry in metres (degrees are NOT metres: at
lat 50 deg one degree of longitude is ~71 km but one degree of latitude ~111 km).
"""
import numpy as np

R_EARTH = 6371008.8  # mean Earth radius (m)


class LocalProjection:
    def __init__(self, lon0, lat0):
        self.lon0 = float(lon0)
        self.lat0 = float(lat0)
        self.kx = np.radians(1.0) * R_EARTH * np.cos(np.radians(self.lat0))
        self.ky = np.radians(1.0) * R_EARTH

    @classmethod
    def from_track(cls, track):
        return cls(np.mean(track[:, 0]), np.mean(track[:, 1]))

    def to_xy(self, lonlat):
        lonlat = np.asarray(lonlat, dtype=float)
        return np.column_stack([(lonlat[:, 0] - self.lon0) * self.kx, (lonlat[:, 1] - self.lat0) * self.ky])

    def to_lonlat(self, xy):
        xy = np.asarray(xy, dtype=float)
        return np.column_stack([xy[:, 0] / self.kx + self.lon0, xy[:, 1] / self.ky + self.lat0])


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


def project_to_segments(P, A, B):
    """Element-wise projection of points P onto segments A->B (all (n,2)).

    Returns (distance, t) with t in [0,1] the fractional position along the segment.
    """
    AB = B - A
    L2 = np.einsum("ij,ij->i", AB, AB)
    t = np.einsum("ij,ij->i", P - A, AB) / np.maximum(L2, 1e-12)
    t = np.clip(t, 0.0, 1.0)
    Q = A + t[:, None] * AB
    d = np.sqrt(np.einsum("ij,ij->i", P - Q, P - Q))
    return d, t


def bearing_deg(dx, dy):
    """Compass bearing (0 = north, clockwise) of a vector in the local x(east)/y(north) frame."""
    return np.degrees(np.arctan2(dx, dy)) % 360.0


def angle_diff_deg(a, b):
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)
