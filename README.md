# Time Series Anomaly (Mini)

A 50-line demo that creates a synthetic time series, injects anomalies, and flags them with IsolationForest.

## What it shows
- 📈 Synthetic data creation for quick experiments
- 🤖 Unsupervised anomaly detection (scikit-learn)
- 🖼️ Simple, readable plotting

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/detect.py
open outputs/anomalies.png  # or just view in the repo
