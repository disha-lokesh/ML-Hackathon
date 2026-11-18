"""PathSnap: map-matching of noisy GPS trajectories onto a road network.

Pipeline: candidate generation -> per-candidate classifier -> HMM/Viterbi decoding.
"""
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path=None):
    path = Path(path) if path else PROJECT_ROOT / "configs" / "config.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
