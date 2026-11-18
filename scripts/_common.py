import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
MAPS = ROOT / "maps"
ARTIFACTS = ROOT / "artifacts"
PROCESSED = ROOT / "data" / "processed"
for d in (RESULTS, FIGURES, MAPS, ARTIFACTS, PROCESSED):
    d.mkdir(parents=True, exist_ok=True)
