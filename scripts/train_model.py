"""
train_model.py
--------------
Trains an Isolation Forest anomaly-detection model on the synthetic
transaction dataset and persists both the model and the feature scaler.

Usage:
    python scripts/train_model.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data",   "transactions.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "isolation_forest.pkl")
SCALER_PATH= os.path.join(BASE_DIR, "models", "scaler.pkl")
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

# ── Feature columns used for training ─────────────────────────────────────────
FEATURES = ["amount", "hour_of_day", "daily_tx_count", "is_foreign"]

def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print("⚠  Dataset not found. Generating it first …")
        import subprocess, sys
        subprocess.run([sys.executable,
                        os.path.join(BASE_DIR, "data", "generate_dataset.py")],
                       check=True)
    df = pd.read_csv(path)
    print(f"📂  Loaded {len(df)} rows from {path}")
    return df

def preprocess(df: pd.DataFrame):
    X = df[FEATURES].copy()
    y = df["label"].values          # 0 = normal, 1 = suspicious (for evaluation)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

def train(X_scaled: np.ndarray) -> IsolationForest:
    """
    Isolation Forest is an unsupervised algorithm — it trains only on
    feature distributions and assigns an anomaly score to each record.
    contamination=0.05 tells the model to expect ~5 % outliers.
    """
    model = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination=0.05,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled)
    return model

def evaluate(model: IsolationForest, X_scaled: np.ndarray, y_true: np.ndarray):
    """
    Isolation Forest predict() returns  1 (inlier / normal)
                                        or -1 (outlier / suspicious).
    We map -1 → 1 and 1 → 0 to match our label convention.
    """
    raw_preds = model.predict(X_scaled)
    y_pred    = np.where(raw_preds == -1, 1, 0)   # -1 → suspicious

    print("\n── Evaluation (using ground-truth labels for sanity check) ──")
    print(classification_report(y_true, y_pred,
                                 target_names=["Normal", "Suspicious"]))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(f"  {'':12s} Pred Normal  Pred Suspicious")
    print(f"  {'True Normal':12s}  {cm[0,0]:^11d}  {cm[0,1]:^15d}")
    print(f"  {'True Susp.':12s}  {cm[1,0]:^11d}  {cm[1,1]:^15d}")

    scores = model.decision_function(X_scaled)   # higher = more normal
    print(f"\nAnomaly score range: [{scores.min():.4f}, {scores.max():.4f}]")
    return y_pred

def save_artifacts(model, scaler):
    with open(MODEL_PATH,  "wb") as f: pickle.dump(model,  f)
    with open(SCALER_PATH, "wb") as f: pickle.dump(scaler, f)
    print(f"\n✅  Model  saved → {MODEL_PATH}")
    print(f"✅  Scaler saved → {SCALER_PATH}")

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df               = load_data(DATA_PATH)
    X_scaled, y, sc  = preprocess(df)
    model            = train(X_scaled)
    evaluate(model, X_scaled, y)
    save_artifacts(model, sc)
    print("\n🎯  Training complete. Ready to launch the dashboard.")
