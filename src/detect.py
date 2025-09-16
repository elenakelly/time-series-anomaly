import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from pathlib import Path

def make_series(n=500, seed=7):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    base = 0.02 * t + np.sin(t/12) + rng.normal(0, 0.2, n)
    # inject anomalies
    idx = rng.choice(n, size=8, replace=False)
    y = base.copy()
    y[idx] += rng.normal(3.5, 0.5, size=len(idx))  # spikes
    return pd.DataFrame({"t": t, "y": y})

def detect(df):
    model = IsolationForest(n_estimators=100, contamination=0.02, random_state=0)
    df["score_raw"] = model.fit_predict(df[["y"]])  # -1 = anomaly
    df["is_anomaly"] = df["score_raw"] == -1
    return df

def plot(df, out="outputs/anomalies.png"):
    Path("outputs").mkdir(exist_ok=True)
    plt.figure()
    plt.plot(df["t"], df["y"], label="signal")
    a = df[df["is_anomaly"]]
    plt.scatter(a["t"], a["y"], s=50, marker="x", label="anomaly")
    plt.legend()
    plt.title("IsolationForest Anomaly Detection (tiny demo)")
    plt.xlabel("t")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches="tight")

if __name__ == "__main__":
    df = make_series()
    df = detect(df)
    plot(df)
    print("Saved plot to outputs/anomalies.png")
