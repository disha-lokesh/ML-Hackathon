"""PathSnap evaluator UI (runs on Laptop 2).

Run:  streamlit run ui/app.py
Set the API address in the sidebar (or env PATHSNAP_API_URL), e.g. http://192.168.1.20:8000
The UI only talks to the API; it holds no model and no reference data.
"""
import os

import folium
import requests
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(page_title="PathSnap", layout="wide")
st.title("PathSnap - map-matching & route reconstruction")

api = st.sidebar.text_input("API URL", os.environ.get("PATHSNAP_API_URL", "http://localhost:8000")).rstrip("/")
use_hmm = st.sidebar.checkbox("Use HMM / Viterbi decoding", value=True)

try:
    health = requests.get(f"{api}/health", timeout=5).json()
    st.sidebar.success(f"API online - model: {health['model']}")
    trips = requests.get(f"{api}/trips", timeout=30).json()
except Exception as e:  # noqa: BLE001
    st.error(f"Cannot reach API at {api}: {e}")
    st.stop()

st.sidebar.caption(f"Serving partition: {trips['partition']}")
labels = {f"{t['trip_id']}  ({t['n_points']} pts, {t['duration_min']} min)": t["trip_id"] for t in trips["trips"]}
choice = st.selectbox(f"Select one of the {len(labels)} evaluation trips", list(labels))

if st.button("Run map-matching", type="primary"):
    with st.spinner("API is running the full pipeline..."):
        r = requests.post(f"{api}/predict", json={"trip_id": labels[choice], "use_hmm": use_hmm}, timeout=600)
    if r.status_code != 200:
        st.error(f"API error {r.status_code}: {r.text}")
    else:
        st.session_state["result"] = r.json()

res = st.session_state.get("result")
if res:
    m = res["metrics"]
    st.subheader(f"Trip {res['trip_id']}  ({res['n_points']} GPS points, {res['n_predicted_arcs']} predicted arcs)")
    c = st.columns(5)
    c[0].metric("Point acc.", f"{m.get('point_accuracy_hmm', 0):.2%}",
                f"{m.get('point_accuracy_hmm', 0) - m.get('point_accuracy_classifier_only', 0):+.2%} vs clf-only")
    c[1].metric("Continuity", f"{m.get('route_continuity_hmm', 0):.2%}")
    c[2].metric("Route F1 (len)", f"{m.get('len_f1_hmm', 0):.2%}")
    c[3].metric("Cand. miss", f"{m.get('candidate_miss_rate', 0):.2%}")
    c[4].metric("API time", f"{res['timing_s']['total']:.2f} s")

    gps = res["gps"]
    fmap = folium.Map(location=[sum(p[0] for p in gps) / len(gps), sum(p[1] for p in gps) / len(gps)], zoom_start=14)
    g = folium.FeatureGroup("GPS observations")
    for lat, lon, t in gps[::max(1, len(gps) // 800)]:
        folium.CircleMarker((lat, lon), radius=2, color="#2166ac", fill=True, tooltip=f"t={t:.0f}s").add_to(g)
    g.add_to(fmap)
    g = folium.FeatureGroup("Classifier-only route", show=False)
    for line in res["classifier_only_route"]:
        folium.PolyLine(line, color="#fdae61", weight=3).add_to(g)
    g.add_to(fmap)
    g = folium.FeatureGroup("Predicted route")
    for line in res["predicted_route"]:
        folium.PolyLine(line, color="#d73027", weight=4).add_to(g)
    g.add_to(fmap)
    folium.Marker(gps[0][:2], tooltip="start", icon=folium.Icon(color="green")).add_to(fmap)
    folium.Marker(gps[-1][:2], tooltip="end", icon=folium.Icon(color="red")).add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.fit_bounds([[min(p[0] for p in gps), min(p[1] for p in gps)], [max(p[0] for p in gps), max(p[1] for p in gps)]])
    st_folium(fmap, height=600, use_container_width=True, returned_objects=[])

    left, right = st.columns(2)
    left.markdown("**All evaluation metrics (computed server-side)**")
    left.dataframe({"metric": list(m), "value": list(m.values())}, hide_index=True, use_container_width=True)
    right.markdown("**Pipeline timing (s)**")
    right.json(res["timing_s"])
    right.markdown("**Decoder info**")
    right.json(res["decode_info"])
