# PathSnap: How to Run on Another Device

This guide sets up and runs the PathSnap map-matching system on any Windows, Mac or Linux computer. It covers running everything on one laptop and the two-laptop evaluation setup.

## 1. Copy the project folder

Copy the whole `pathsnap` folder to the new computer. A zip file, a USB stick or a GitHub clone all work.

- You can leave out `data/processed/` (about 621 MB). It is only needed to retrain the models.
- The trained model is already in `artifacts/`: `final_model.joblib` and `inference_config.json`.
- Keep `data/raw/` if that laptop will run the API, because the API reads the trips from there.

## 2. Install Python 3.11 to 3.13

Download Python from python.org. Version 3.13 is the safest choice because the model was trained with it.

On Windows, tick **"Add Python to PATH"** in the installer.

## 3. Set up the environment (once per computer)

Open a terminal inside the `pathsnap` folder and run the following commands in order.

Create a virtual environment:

`python -m venv .venv`

Activate it. On Windows:

`.venv\Scripts\activate`

On Mac or Linux:

`source .venv/bin/activate`

Then install the packages. On the API laptop, or when running everything on one computer:

`pip install -r requirements.txt`

On a laptop that only runs the UI:

`pip install -r requirements-ui.txt`

The package versions are pinned on purpose. The saved model loads reliably only with the same scikit-learn and XGBoost versions it was trained with.

## 4a. Run everything on one computer

Open two terminals. In both, go to the `pathsnap` folder and activate the venv.

Terminal 1 starts the API:

`uvicorn api.main:app --host 0.0.0.0 --port 8000`

Terminal 2 starts the UI:

`streamlit run ui/app.py`

The browser opens at `http://localhost:8501`. Choose a trip and click **Run map-matching**.

## 4b. Two-laptop setup (evaluation demo)

### Laptop 1: API (model, road networks, scoring)

1. Copy the 10 evaluator trips into `data/raw/eval_test/`. Use one folder per trip, each holding the `.track`, `.nodes`, `.arcs` and `.route` files. If this folder is empty, the API serves the 10 student-test trips instead, which is useful for rehearsal.
2. Start the API: `uvicorn api.main:app --host 0.0.0.0 --port 8000`
3. Keep this terminal visible. It prints a log line for every request, which the evaluators ask to see. The same lines are saved to `results/api_requests.log`.
4. Find this laptop's IP address. On Windows, run `ipconfig` and read the "IPv4 Address" line, for example `192.168.1.20`.
5. Allow port 8000 through Windows Firewall. Either accept the pop-up the first time the API starts, or run this once in an **administrator** terminal: `netsh advfirewall firewall add rule name="PathSnap API" dir=in action=allow protocol=TCP localport=8000`

### Laptop 2: UI (evaluator interface)

1. Start the UI: `streamlit run ui/app.py`
2. In the left sidebar, set **API URL** to Laptop 1's address, for example `http://192.168.1.20:8000`.
3. A green "API online" box means the two laptops are connected. Pick a trip and click **Run map-matching**.

### Network requirements

- Both laptops must be on the same Wi-Fi network or phone hotspot. College networks often block traffic between devices; if so, use a mobile hotspot.
- To test the connection, open `http://<laptop-1-ip>:8000/health` in Laptop 2's browser. It should show `{"status":"ok", ...}`.
- The UI laptop needs internet access for the OpenStreetMap background tiles. The API laptop does not.

## 5. Optional: retrain everything from scratch

Run these on Laptop 1, in this order. They take about 40 minutes in total.

1. `python scripts/02_build_dataset.py` builds the candidates, features and labels.
2. `python scripts/01_eda.py` produces the data exploration plots and maps.
3. `python scripts/03_train_models.py --point-fraction 0.35` trains and compares the 8 models.
4. `python scripts/04_bayes_opt.py --model XGBoost --trials 30` runs the Bayesian tuning and saves the final model.
5. `python scripts/05_hmm.py` tunes and evaluates the HMM and exports the inference configuration.

To rebuild the PDF reports afterwards: `python report/build_report.py`

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| `python` is not recognised | Try `py` instead, or reinstall Python with "Add to PATH" ticked. |
| `uvicorn` or `streamlit` not found | Activate the venv first (step 3). |
| UI shows "Cannot reach API" | Check that the API is running, the IP address is correct, and port 8000 is open in the firewall. Use the `/health` test. |
| Error while loading the model | Package versions differ. Inside the venv, run `pip install -r requirements.txt` again. |
| Map shows no background tiles | The UI laptop has no internet access. The routes still draw. |
| Port 8000 already in use | Start the API with `--port 8001` and use that port in the UI's API URL. |

Rehearse the full two-laptop demo before evaluation day.
