"""Integration test: call the running API for every served trip and save the responses' metrics.

python scripts/06_test_api.py --api http://localhost:8000
"""
import argparse

import pandas as pd
import requests

from _common import RESULTS

ap = argparse.ArgumentParser()
ap.add_argument("--api", default="http://localhost:8000")
args = ap.parse_args()

trips = requests.get(f"{args.api}/trips", timeout=60).json()
rows = []
for t in trips["trips"]:
    r = requests.post(f"{args.api}/predict", json={"trip_id": t["trip_id"]}, timeout=600)
    r.raise_for_status()
    j = r.json()
    rows.append({"trip_id": t["trip_id"], **j["metrics"], "api_total_s": j["timing_s"]["total"]})
    print(rows[-1], flush=True)
df = pd.DataFrame(rows)
df.to_csv(RESULTS / f"api_integration_{trips['partition']}.csv", index=False)
print(df.mean(numeric_only=True).round(4).to_string())
