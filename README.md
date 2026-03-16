# 🛡️ AI-Based Financial Threat Monitoring System

A prototype anomaly-detection platform that identifies suspicious financial
transactions using **Isolation Forest** (unsupervised ML) and visualises alerts
on a real-time **Streamlit** security dashboard.

---

## 📁 Project Structure

```
financial-threat-monitor/
│
├── data/
│   ├── generate_dataset.py    # Synthetic transaction generator
│   └── transactions.csv       # Generated after running the script
│
├── models/
│   ├── isolation_forest.pkl   # Persisted trained model
│   └── scaler.pkl             # StandardScaler for feature normalisation
│
├── scripts/
│   ├── train_model.py         # Trains + evaluates + saves the model
│   └── fraud_detection.py     # Core detection engine (Transaction / FraudDetector classes)
│
├── dashboard/
│   └── app.py                 # Streamlit monitoring dashboard
│
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1 — Clone & install dependencies

```bash
# Python 3.10+ recommended
pip install -r requirements.txt
```

### 2 — Generate the dataset

```bash
python data/generate_dataset.py
```

Output: `data/transactions.csv` (950 normal + 50 suspicious transactions)

### 3 — Train the model

```bash
python scripts/train_model.py
```

Output: `models/isolation_forest.pkl` + `models/scaler.pkl`

### 4 — Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Open your browser at **http://localhost:8501**

---

## 🔍 How It Works

### Feature Engineering

| Feature           | Description                                         | Why it matters             |
|-------------------|-----------------------------------------------------|----------------------------|
| `amount`          | Dollar value of the transaction                     | Large outliers signal fraud |
| `hour_of_day`     | 0–23 hour when the transaction occurred             | Wee-hour spikes are risky  |
| `daily_tx_count`  | How many times the card was used that day           | Card-cloning produces bursts|
| `is_foreign`      | 1 if the location is outside "domestic" cities      | Geographic anomaly         |

### Isolation Forest Algorithm

Isolation Forest detects anomalies by randomly partitioning the feature space
into binary trees. **Anomalous points require fewer splits to be isolated**,
giving them lower anomaly scores (negative `decision_function` values).

```
Normal point  →  deep in the tree  →  high score  →  SAFE
Anomaly       →  shallow partition →  low score   →  FLAG
```

`contamination=0.05` tells the model ~5 % of the training set is noise/fraud.

### Risk Classification

| Anomaly Score      | Risk Level |
|--------------------|------------|
| > −0.05 (normal)   | LOW        |
| −0.05 to −0.10     | MEDIUM     |
| −0.10 to −0.15     | HIGH       |
| < −0.15            | CRITICAL   |

---

## 🖥️ Dashboard Features

| Panel                          | Description                                         |
|--------------------------------|-----------------------------------------------------|
| **KPI Cards**                  | Real-time counts of normal / suspicious / critical  |
| **Sidebar Analyser**           | Enter any transaction details → instant prediction  |
| **Scatter Plot**               | Amount vs Anomaly Score, sized by daily frequency   |
| **Risk Donut Chart**           | Distribution across LOW / MEDIUM / HIGH / CRITICAL  |
| **Hourly Heatmap**             | When suspicious activity peaks during the day       |
| **Location Bar Chart**         | Geographic origin of alerts                         |
| **Suspicious Transaction Log** | Sortable table of all flagged transactions           |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   DATA INGESTION LAYER                  │
│  CSV / Kafka stream / DB → generate_dataset.py          │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│                   MODEL TRAINING LAYER                  │
│  train_model.py → IsolationForest → pickle artifacts    │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│                 DETECTION ENGINE LAYER                  │
│  fraud_detection.py                                     │
│  ├── Transaction dataclass (feature builder)            │
│  ├── FraudDetector.predict()  — single transaction      │
│  └── FraudDetector.batch_predict() — bulk analysis      │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│                VISUALISATION LAYER                      │
│  dashboard/app.py (Streamlit + Plotly)                  │
│  ├── Live prediction from sidebar input                 │
│  ├── KPI cards, scatter, donut, hourly bar, geo bar     │
│  └── Sortable suspicious transaction table              │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Production-Grade Improvements

### 1. Real-Time Data Pipeline
- Replace CSV with **Apache Kafka** or **AWS Kinesis** for live transaction streams
- Add a **FastAPI** REST endpoint wrapping `FraudDetector.predict()` so any
  banking service can call it synchronously

### 2. Supervised Hybrid Model
- Isolation Forest is unsupervised — it has no memory of confirmed fraud cases.
  Pair it with **XGBoost / LightGBM** trained on labelled fraud data (e.g., the
  Kaggle Credit Card Fraud dataset) for higher precision.
- Use Isolation Forest score as a *feature* for the supervised model.

### 3. Feature Enrichment
- **Velocity checks**: card spend over 1 h / 24 h / 7 days
- **Graph features**: network relationships between merchant, device, IP
- **Behavioural biometrics**: typing speed, mouse movement during checkout
- **Device fingerprinting**: browser, OS, geolocation vs GPS

### 4. Model Lifecycle Management
- Store model versions in **MLflow** or **AWS SageMaker Model Registry**
- Automate re-training on a weekly cadence as fraud patterns drift
- Monitor **data drift** with tools like Evidently AI or WhyLabs

### 5. Alert Management
- Integrate with **PagerDuty / Splunk SIEM** to route CRITICAL alerts
- Add **human-in-the-loop review**: analysts confirm/reject predictions,
  feeding labels back into the next training cycle

### 6. Security Hardening
- Encrypt the model artifact and transaction data at rest (AES-256)
- Role-based access control (RBAC) on the dashboard
- Audit log every prediction call for compliance (PCI-DSS, SOX)
- Run the service inside a **VPC** — never expose raw transaction data externally

### 7. Explainability (XAI)
- Add **SHAP values** so analysts understand *why* a transaction was flagged —
  critical for regulatory compliance and analyst trust

---

## 🧪 Running Unit Tests (Optional Extension)

```bash
pip install pytest
pytest tests/           # extend with a tests/ folder as the project grows
```

---

## 📄 License

MIT — free to use for educational and interview demonstration purposes.
