"""
generate_dataset.py
-------------------
Generates a synthetic financial transaction dataset for training
the anomaly detection model. Saves to data/transactions.csv.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

# Reproducibility
np.random.seed(42)
random.seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
N_NORMAL     = 950   # legitimate transactions
N_ANOMALOUS  = 50    # fraudulent / suspicious transactions
LOCATIONS    = ["New York", "London", "Tokyo", "Sydney", "Dubai",
                "Singapore", "Frankfurt", "Toronto", "Mumbai", "São Paulo"]
CATEGORIES   = ["Retail", "Online", "ATM", "Wire Transfer",
                 "POS Terminal", "Mobile Payment"]

def random_timestamp(start: datetime, days: int) -> datetime:
    delta = timedelta(
        days=random.randint(0, days),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59),
    )
    return start + delta

def hour_of_day(ts: datetime) -> int:
    return ts.hour

# ── Generate normal transactions ──────────────────────────────────────────────
start_date = datetime(2024, 1, 1)
records = []

for i in range(N_NORMAL):
    ts       = random_timestamp(start_date, 180)
    amount   = np.random.lognormal(mean=4.5, sigma=1.2)   # ~$5 – $5 000
    amount   = round(min(amount, 4999), 2)
    location = random.choice(LOCATIONS)
    category = random.choice(CATEGORIES)
    hour     = hour_of_day(ts)
    freq     = random.randint(1, 8)   # daily transaction count for this card

    records.append({
        "transaction_id":  f"TXN{i:05d}",
        "timestamp":       ts.isoformat(),
        "amount":          amount,
        "hour_of_day":     hour,
        "location":        location,
        "category":        category,
        "daily_tx_count":  freq,
        "is_foreign":      int(location not in ["New York", "Toronto"]),
        "label":           0,          # 0 = normal
    })

# ── Generate anomalous transactions ──────────────────────────────────────────
for i in range(N_ANOMALOUS):
    ts   = random_timestamp(start_date, 180)
    anomaly_type = random.choice(["high_amount", "odd_hour", "high_freq", "foreign_spike"])

    if anomaly_type == "high_amount":
        amount = round(random.uniform(8000, 50000), 2)
        hour   = random.randint(8, 18)
        freq   = random.randint(1, 3)
    elif anomaly_type == "odd_hour":
        amount = round(random.uniform(500, 3000), 2)
        hour   = random.choice([0, 1, 2, 3, 4])    # wee hours
        freq   = random.randint(1, 4)
    elif anomaly_type == "high_freq":
        amount = round(random.uniform(100, 1000), 2)
        hour   = random.randint(8, 22)
        freq   = random.randint(25, 50)             # card used many times in a day
    else:   # foreign_spike
        amount = round(random.uniform(2000, 15000), 2)
        hour   = random.randint(0, 23)
        freq   = random.randint(5, 15)

    location = random.choice(LOCATIONS)
    category = random.choice(CATEGORIES)

    records.append({
        "transaction_id":  f"TXN{N_NORMAL + i:05d}",
        "timestamp":       ts.isoformat(),
        "amount":          amount,
        "hour_of_day":     hour,
        "location":        location,
        "category":        category,
        "daily_tx_count":  freq,
        "is_foreign":      int(location not in ["New York", "Toronto"]),
        "label":           1,          # 1 = suspicious
    })

# ── Shuffle & save ────────────────────────────────────────────────────────────
df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)
out_path = os.path.join(os.path.dirname(__file__), "transactions.csv")
df.to_csv(out_path, index=False)

print(f"✅  Dataset saved → {out_path}")
print(f"    Total rows : {len(df)}")
print(f"    Normal     : {(df.label == 0).sum()}")
print(f"    Suspicious : {(df.label == 1).sum()}")
print(df.head())
