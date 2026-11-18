"""Folium map generation."""
import folium
import numpy as np


def _arc_lines(net, arc_ids):
    A, B = net.arcs_lonlat(arc_ids)
    return [[(a[1], a[0]), (b[1], b[0])] for a, b in zip(A, B)]


def route_map(track, net, pred_route=None, ref_route=None, clf_points=None, network_arcs=None, title=""):
    """track: (N,3) lon,lat,t. Routes are arc-id lists. clf_points: per-point classifier-only arcs."""
    m = folium.Map(location=[track[:, 1].mean(), track[:, 0].mean()], zoom_start=14, tiles="OpenStreetMap")
    if network_arcs is not None and len(network_arcs):
        g = folium.FeatureGroup("Road network (nearby arcs)", show=False)
        for line in _arc_lines(net, network_arcs):
            folium.PolyLine(line, color="#777777", weight=1, opacity=0.5).add_to(g)
        g.add_to(m)
    if ref_route is not None:
        g = folium.FeatureGroup("Reference route (ground truth)", show=True)
        for line in _arc_lines(net, ref_route):
            folium.PolyLine(line, color="#1a9850", weight=8, opacity=0.55).add_to(g)
        g.add_to(m)
    if clf_points is not None:
        g = folium.FeatureGroup("Classifier-only matched arcs", show=False)
        arcs = np.unique(clf_points[clf_points >= 0])
        for line in _arc_lines(net, arcs):
            folium.PolyLine(line, color="#fdae61", weight=3, opacity=0.9).add_to(g)
        g.add_to(m)
    if pred_route is not None:
        g = folium.FeatureGroup("Predicted route (classifier + HMM)", show=True)
        for line in _arc_lines(net, pred_route):
            folium.PolyLine(line, color="#d73027", weight=3.5, opacity=0.95).add_to(g)
        g.add_to(m)
    g = folium.FeatureGroup("GPS observations", show=True)
    step = max(1, len(track) // 1500)
    for lon, lat, t in track[::step]:
        folium.CircleMarker((lat, lon), radius=2, color="#2166ac", fill=True, weight=1, opacity=0.8,
                            tooltip=f"t={t:.0f}s").add_to(g)
    g.add_to(m)
    folium.Marker((track[0, 1], track[0, 0]), tooltip="start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker((track[-1, 1], track[-1, 0]), tooltip="end", icon=folium.Icon(color="red", icon="stop")).add_to(m)
    legend = f"""<div style="position:fixed;bottom:20px;left:20px;z-index:9999;background:white;padding:8px 10px;
      border:1px solid #888;border-radius:6px;font-size:12px;line-height:1.5">
      <b>{title}</b><br>
      <span style="color:#2166ac">&#9679;</span> GPS observations<br>
      <span style="color:#1a9850">&#9644;</span> Reference route<br>
      <span style="color:#d73027">&#9644;</span> Predicted route (HMM)<br>
      <span style="color:#fdae61">&#9644;</span> Classifier-only arcs<br>
      <span style="color:#777">&#9644;</span> Road network</div>"""
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[track[:, 1].min(), track[:, 0].min()], [track[:, 1].max(), track[:, 0].max()]])
    return m


def nearby_arcs(net, xy, margin=150.0, limit=15000):
    lo, hi = xy.min(0) - margin, xy.max(0) + margin
    sel = np.where(((net.A >= lo) & (net.A <= hi)).all(1))[0]
    return sel[:limit]
