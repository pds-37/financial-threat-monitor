"""
fraud_detection.py
------------------
Core detection engine — loads persisted model/scaler artifacts and
exposes a clean predict() API consumed by the Streamlit dashboard
and any future REST endpoint.
"""

import os
import pickle
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Any

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "isolation_forest.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

FEATURES    = ["amount", "hour_of_day", "daily_tx_count", "is_foreign"]

# ── Data model ────────────────────────────────────────────────────────────────
@dataclass
class Transaction:
    transaction_id: str
    amount:         float
    hour_of_day:    int
    daily_tx_count: int
    location:       str
    category:       str
    is_foreign:     int = field(init=False)

    # List of locations considered "domestic" for the demo
    DOMESTIC = {"New York", "Toronto"}

    def __post_init__(self):
        self.is_foreign = int(self.location not in self.DOMESTIC)

    def to_feature_dict(self) -> Dict[str, Any]:
        return {
            "amount":         self.amount,
            "hour_of_day":    self.hour_of_day,
            "daily_tx_count": self.daily_tx_count,
            "is_foreign":     self.is_foreign,
        }

@dataclass
class PredictionResult:
    transaction_id: str
    is_suspicious:  bool
    anomaly_score:  float       # lower = more anomalous
    risk_level:     str         # LOW / MEDIUM / HIGH / CRITICAL
    reasons:        List[str]   # human-readable flags
    raw_features:   Dict[str, Any]

# ── Detector class ────────────────────────────────────────────────────────────
class FraudDetector:
    """Loads trained artifacts and provides a predict() method."""

    # Anomaly-score thresholds (decision_function output — higher = normal)
    _THRESHOLD_HIGH     = -0.05
    _THRESHOLD_CRITICAL = -0.15

    def __init__(self):
        self._model  = self._load(MODEL_PATH,  "model")
        self._scaler = self._load(SCALER_PATH, "scaler")

    # ── Public API ────────────────────────────────────────────────────────────
    def predict(self, txn: Transaction) -> PredictionResult:
        feat_dict = txn.to_feature_dict()
        X         = pd.DataFrame([feat_dict])[FEATURES]
        X_scaled  = self._scaler.transform(X)

        raw_pred  = self._model.predict(X_scaled)[0]        # 1 or -1
        score     = self._model.decision_function(X_scaled)[0]
        suspicious= raw_pred == -1                          # -1 → outlier

        risk   = self._risk_level(score, suspicious)
        reasons= self._explain(txn, score)

        return PredictionResult(
            transaction_id=txn.transaction_id,
            is_suspicious =suspicious,
            anomaly_score =round(float(score), 4),
            risk_level    =risk,
            reasons       =reasons,
            raw_features  =feat_dict,
        )

    def batch_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Accepts a DataFrame with the four feature columns and appends
        prediction columns. Used by the dashboard for bulk analysis.
        """
        X_scaled = self._scaler.transform(df[FEATURES])
        raw_preds= self._model.predict(X_scaled)
        scores   = self._model.decision_function(X_scaled)

        df = df.copy()
        df["is_suspicious"] = (raw_preds == -1).astype(int)
        df["anomaly_score"] = scores.round(4)
        df["risk_level"]    = [
            self._risk_level(s, p == -1)
            for s, p in zip(scores, raw_preds)
        ]
        return df

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _load(path: str, name: str):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"❌  {name.capitalize()} artifact not found at '{path}'.\n"
                f"    Run:  python scripts/train_model.py"
            )
        with open(path, "rb") as f:
            return pickle.load(f)

    def _risk_level(self, score: float, suspicious: bool) -> str:
        if not suspicious:
            return "LOW"
        if score > self._THRESHOLD_HIGH:
            return "MEDIUM"
        if score > self._THRESHOLD_CRITICAL:
            return "HIGH"
        return "CRITICAL"

    @staticmethod
    def _explain(txn: Transaction, score: float) -> List[str]:
        """Generate human-readable reason flags for the dashboard."""
        reasons = []
        if txn.amount > 5000:
            reasons.append(f"💰 Large amount: ${txn.amount:,.2f}")
        if txn.hour_of_day in range(0, 5):
            reasons.append(f"🌙 Unusual hour: {txn.hour_of_day:02d}:00")
        if txn.daily_tx_count > 20:
            reasons.append(f"🔁 High frequency: {txn.daily_tx_count} txns today")
        if txn.is_foreign:
            reasons.append(f"🌍 Foreign location: {txn.location}")
        if score < -0.10:
            reasons.append(f"📊 Low anomaly score: {score:.4f}")
        if not reasons:
            reasons.append("⚠️  Combination of features flagged by model")
        return reasons


# ── Quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    detector = FraudDetector()

    normal_txn = Transaction(
        transaction_id="TEST001",
        amount=120.50,
        hour_of_day=14,
        daily_tx_count=3,
        location="New York",
        category="Retail",
    )

    suspicious_txn = Transaction(
        transaction_id="TEST002",
        amount=45000.00,
        hour_of_day=3,
        daily_tx_count=35,
        location="Tokyo",
        category="Wire Transfer",
    )

    for txn in [normal_txn, suspicious_txn]:
        result = detector.predict(txn)
        label  = "🚨 SUSPICIOUS" if result.is_suspicious else "✅ NORMAL"
        print(f"\n{label}  [{result.risk_level}]  score={result.anomaly_score}")
        print(f"  Reasons: {result.reasons}")
