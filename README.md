# PathSnap: Distributed Map-Matching & Route Reconstruction

PathSnap snaps noisy 1 Hz GPS traces onto a directed road network and rebuilds the route the vehicle drove.
It runs in three stages:

1. **Candidate generation.** Every GPS fix gets up to 12 nearby road arcs (within 60 m). These come from a KD-tree built over densified arcs in a local metric projection.
2. **Candidate classifier.** A tuned XGBoost model scores each (fix, candidate arc) pair on 19 engineered features: distance, heading agreement, temporal context, arc geometry and topology.
3. **HMM/Viterbi decoding.** The classifier probabilities serve as emissions. Transitions come from network shortest-path distance compared with GPS displacement. Viterbi decoding produces a connected route.

The system is split across two laptops. Laptop 1 runs the **FastAPI** backend, which holds the model, the road networks and the server-side scoring. Laptop 2 runs the **Streamlit** UI.

## Repository layout

```
configs/config.json          candidate radius, feature windows, CV folds, default HMM params, seed
src/pathsnap/
  io.py                      raw readers (.track/.nodes/.arcs; .route loaded separately)
  geo.py                     local tangent-plane projection, point-segment projection, bearings
  network.py                 directed road graph, KD-tree candidate search, bounded Dijkstra
  labels.py                  monotone DP alignment GPS -> reference route (per-point labels)
  features.py                candidate generation + feature engineering
  training.py                model zoo, group-aware CV loop, training-data filtering
  hmm.py                     Viterbi decoder, classifier-only baseline, route reconstruction
  metrics.py                 row metrics, Top-1, continuity, route overlap
  pipeline.py                inference wrapper used by the API (+ server-side evaluation)
  viz.py                     Folium map generation
scripts/
  01_eda.py                  dataset inventory, quality checks, plots, Folium EDA maps
  02_build_dataset.py        candidate/feature tables + labels for train & student_test
  03_train_models.py         8 models, 5-fold GroupKFold CV, student-test comparison, plots
  04_bayes_opt.py            Optuna TPE Bayesian optimisation, tuned vs untuned, export model
  05_hmm.py                  OOF probs, decoder grid search, classifier vs HMM, maps, export config
  06_test_api.py             calls the running API for every trip (integration test)
api/main.py                  FastAPI backend (Laptop 1)
ui/app.py                    Streamlit evaluator UI (Laptop 2)
artifacts/                   final_model.joblib, inference_config.json
results/                     CSV/JSON experiment results and logs
figures/                     all plots used in the report
maps/                        interactive Folium HTML maps
report/                      technical report draft
data/raw/{train,student_test,eval_test}/<trip_id>/<trip_id>.{track,nodes,arcs,route}
```

## Installation

```bash
python -m venv .venv
```
```bash
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on Linux/macOS)
```
```bash
pip install -r requirements.txt
```

The UI laptop only needs `pip install -r requirements-ui.txt`.

## Reproducing all experiments (Laptop 1)

```bash
python scripts/02_build_dataset.py
```
```bash
python scripts/01_eda.py
```
```bash
python scripts/03_train_models.py --point-fraction 0.35
```
```bash
python scripts/04_bayes_opt.py --model XGBoost --trials 30
```
```bash
python scripts/05_hmm.py
```

Script 02 must run before 01, because the EDA plots the label-quality file that 02 writes. Every script reads `configs/config.json` and uses seed 42.

## Deployment (two laptops on the same network)

**Laptop 1 (API).** Copy the 10 evaluator trips into `data/raw/eval_test/`, then start the server:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Find its IP with `ipconfig` (Windows) or `ip a` (Linux), and allow port 8000 through the firewall.
Every request is logged to the console and to `results/api_requests.log`. These are the logs to show the evaluator.
If `eval_test/` is empty, the API falls back to the `student_test` trips so the setup can be rehearsed.

**Laptop 2 (UI).**

```bash
streamlit run ui/app.py
```

Enter `http://<laptop-1-ip>:8000` as the API URL in the sidebar, pick a trip and click **Run map-matching**.

### API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Status, loaded model, HMM params, served partition |
| GET | `/trips` | Evaluation trips (id, #points, duration) |
| POST | `/predict` | Body `{"trip_id": "...", "use_hmm": true}` returns the predicted route polylines, GPS points, metrics and timings |

The reference routes are read only after prediction, inside `pathsnap.pipeline.evaluate_prediction`. They are never returned to the UI.

## Leakage controls

- Trips are never split across partitions. CV uses `GroupKFold` grouped by trip.
- The scaler and models are fitted on training folds only. The decoder parameters are tuned on out-of-fold probabilities of training trips.
- The student-test trips are used only for development comparison. The evaluator trips are used only by the API at demo time.
# ML-Hackathon
