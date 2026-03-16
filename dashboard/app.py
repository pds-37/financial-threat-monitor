"""
app.py  ·  AI-Based Financial Threat Monitoring System
--------------------------------------------------------
Run with:  streamlit run dashboard/app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from scripts.fraud_detection import FraudDetector, Transaction

# ═══════════════════════════════════════════════════════════════════════════════
# Page config & global CSS
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Financial Threat Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Dark header bar */
    .header-bar {
        background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2a 60%, #112240 100%);
        border-bottom: 2px solid #e63946;
        padding: 1.1rem 1.6rem;
        border-radius: 8px;
        margin-bottom: 1.4rem;
    }
    .header-title {
        font-size: 1.65rem;
        font-weight: 700;
        color: #e2e8f0;
        letter-spacing: .03em;
        margin: 0;
    }
    .header-sub {
        color: #94a3b8;
        font-size: .82rem;
        margin: 0;
    }

    /* KPI cards */
    .kpi-card {
        background: #0d1b2a;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border-left: 4px solid;
        margin-bottom: .5rem;
    }
    .kpi-value { font-size: 1.9rem; font-weight: 800; }
    .kpi-label { font-size: .78rem; color: #94a3b8; text-transform: uppercase; letter-spacing: .08em; }
    .kpi-normal  { border-color: #22c55e; color: #22c55e; }
    .kpi-alert   { border-color: #e63946; color: #e63946; }
    .kpi-medium  { border-color: #f59e0b; color: #f59e0b; }
    .kpi-score   { border-color: #38bdf8; color: #38bdf8; }

    /* Alert badges */
    .badge-critical { background:#7f1d1d; color:#fca5a5; border-radius:6px; padding:3px 10px; font-size:.75rem; font-weight:600; }
    .badge-high     { background:#431407; color:#fdba74; border-radius:6px; padding:3px 10px; font-size:.75rem; font-weight:600; }
    .badge-medium   { background:#422006; color:#fde68a; border-radius:6px; padding:3px 10px; font-size:.75rem; font-weight:600; }
    .badge-low      { background:#052e16; color:#86efac; border-radius:6px; padding:3px 10px; font-size:.75rem; font-weight:600; }
    .badge-normal   { background:#052e16; color:#86efac; border-radius:6px; padding:3px 10px; font-size:.75rem; font-weight:600; }

    /* Alert box */
    .alert-box {
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
        margin-top: .8rem;
        font-size: .92rem;
    }
    .alert-suspicious {
        background: rgba(230,57,70,.15);
        border: 1.5px solid #e63946;
        color: #fca5a5;
    }
    .alert-normal {
        background: rgba(34,197,94,.12);
        border: 1.5px solid #22c55e;
        color: #86efac;
    }

    /* Divider */
    hr { border-color: #1e293b; }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #071525; }
    [data-testid="stSidebar"] label { color: #94a3b8 !important; }
    [data-testid="stSidebar"] .stSelectbox label { color: #94a3b8; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_detector():
    return FraudDetector()

@st.cache_data
def load_dataset():
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "transactions.csv",
    )
    if not os.path.exists(path):
        st.error("Dataset not found. Run:  python data/generate_dataset.py")
        st.stop()
    return pd.read_csv(path)

def risk_badge(risk: str) -> str:
    cls_map = {"CRITICAL":"critical","HIGH":"high","MEDIUM":"medium","LOW":"low","NORMAL":"normal"}
    cls = cls_map.get(risk.upper(), "normal")
    return f'<span class="badge-{cls}">{risk}</span>'

RISK_COLOR = {"LOW":"#22c55e","MEDIUM":"#f59e0b","HIGH":"#f97316","CRITICAL":"#e63946"}

# ═══════════════════════════════════════════════════════════════════════════════
# Load resources
# ═══════════════════════════════════════════════════════════════════════════════
detector = load_detector()
df_raw   = load_dataset()
df_pred  = detector.batch_predict(df_raw)

# ═══════════════════════════════════════════════════════════════════════════════
# Header
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="header-bar">
  <p class="header-title">🛡️ AI-Based Financial Threat Monitoring System</p>
  <p class="header-sub">Powered by Isolation Forest · Real-time anomaly detection ·
     {datetime.now().strftime("%A, %d %b %Y  %H:%M")}</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar — manual transaction input
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🔍 Inspect a Transaction")
    st.markdown("---")

    txn_id   = st.text_input("Transaction ID", value="TXN99999")
    amount   = st.number_input("Amount ($)", min_value=0.01, max_value=500000.0,
                                value=250.00, step=0.01, format="%.2f")
    hour     = st.slider("Hour of Day (0–23)", 0, 23, 14)
    freq     = st.slider("Daily Transaction Count", 1, 60, 4)
    location = st.selectbox("Location", [
        "New York", "Toronto", "London", "Tokyo",
        "Sydney", "Dubai", "Singapore", "Frankfurt",
        "Mumbai", "São Paulo",
    ])
    category = st.selectbox("Category", [
        "Retail", "Online", "ATM",
        "Wire Transfer", "POS Terminal", "Mobile Payment",
    ])

    st.markdown("---")
    analyze_btn = st.button("⚡ Analyze Transaction", use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# KPI row
# ═══════════════════════════════════════════════════════════════════════════════
total      = len(df_pred)
suspicious = df_pred["is_suspicious"].sum()
normal     = total - suspicious
avg_score  = df_pred["anomaly_score"].mean()
critical   = (df_pred["risk_level"] == "CRITICAL").sum()

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="kpi-card kpi-normal"><div class="kpi-value">{normal}</div><div class="kpi-label">✅ Normal Transactions</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="kpi-card kpi-alert"><div class="kpi-value">{suspicious}</div><div class="kpi-label">🚨 Suspicious Alerts</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="kpi-card kpi-medium"><div class="kpi-value">{critical}</div><div class="kpi-label">💀 Critical Threats</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="kpi-card kpi-score"><div class="kpi-value">{avg_score:.3f}</div><div class="kpi-label">📊 Avg. Anomaly Score</div></div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Live prediction result
# ═══════════════════════════════════════════════════════════════════════════════
if analyze_btn:
    txn = Transaction(
        transaction_id=txn_id,
        amount=amount,
        hour_of_day=hour,
        daily_tx_count=freq,
        location=location,
        category=category,
    )
    result = detector.predict(txn)
    cls    = "suspicious" if result.is_suspicious else "normal"
    icon   = "🚨 SUSPICIOUS TRANSACTION DETECTED" if result.is_suspicious else "✅ TRANSACTION APPEARS NORMAL"

    st.markdown(f"""
    <div class="alert-box alert-{cls}">
      <strong>{icon}</strong><br>
      <strong>ID:</strong> {txn_id} &nbsp;|&nbsp;
      <strong>Risk:</strong> {result.risk_level} &nbsp;|&nbsp;
      <strong>Score:</strong> {result.anomaly_score}<br>
      <strong>Flags:</strong> {" · ".join(result.reasons)}
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Charts row 1: scatter + risk donut
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("#### 📡 Transaction Anomaly Landscape")

col_left, col_right = st.columns([3, 1.5])

with col_left:
    fig_scatter = px.scatter(
        df_pred,
        x="amount",
        y="anomaly_score",
        color="risk_level",
        color_discrete_map=RISK_COLOR,
        size="daily_tx_count",
        size_max=18,
        hover_data=["transaction_id", "location", "category", "hour_of_day"],
        labels={"amount": "Transaction Amount ($)", "anomaly_score": "Anomaly Score (lower = riskier)"},
        title="Amount vs Anomaly Score",
        template="plotly_dark",
    )
    fig_scatter.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0a0f1e",
        font_color="#cbd5e1",
        legend_title_text="Risk",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with col_right:
    risk_counts = df_pred["risk_level"].value_counts().reset_index()
    risk_counts.columns = ["risk_level", "count"]
    fig_donut = go.Figure(go.Pie(
        labels=risk_counts["risk_level"],
        values=risk_counts["count"],
        hole=0.55,
        marker_colors=[RISK_COLOR.get(r, "#94a3b8") for r in risk_counts["risk_level"]],
    ))
    fig_donut.update_layout(
        title="Risk Distribution",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#cbd5e1",
        showlegend=True,
        template="plotly_dark",
        margin=dict(t=50, b=10, l=10, r=10),
    )
    st.plotly_chart(fig_donut, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Charts row 2: hour heatmap + location bar
# ═══════════════════════════════════════════════════════════════════════════════
col_a, col_b = st.columns(2)

with col_a:
    hour_risk = (
        df_pred[df_pred["is_suspicious"] == 1]
        .groupby("hour_of_day")
        .size()
        .reindex(range(24), fill_value=0)
        .reset_index(name="alerts")
    )
    fig_hour = px.bar(
        hour_risk, x="hour_of_day", y="alerts",
        color="alerts",
        color_continuous_scale=["#0d1b2a", "#f97316", "#e63946"],
        labels={"hour_of_day": "Hour of Day", "alerts": "# Alerts"},
        title="🕐 Suspicious Activity by Hour",
        template="plotly_dark",
    )
    fig_hour.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a0f1e",
                            font_color="#cbd5e1", coloraxis_showscale=False)
    st.plotly_chart(fig_hour, use_container_width=True)

with col_b:
    loc_data = (
        df_pred[df_pred["is_suspicious"] == 1]
        .groupby("location")
        .size()
        .sort_values(ascending=False)
        .reset_index(name="alerts")
    )
    fig_loc = px.bar(
        loc_data, x="alerts", y="location",
        orientation="h",
        color="alerts",
        color_continuous_scale=["#1e3a5f", "#e63946"],
        labels={"location": "Location", "alerts": "# Alerts"},
        title="🌍 Alert Origin by Location",
        template="plotly_dark",
    )
    fig_loc.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a0f1e",
                           font_color="#cbd5e1", coloraxis_showscale=False,
                           yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_loc, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Suspicious transaction table
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("#### 🚨 Suspicious Transaction Log")

suspicious_df = (
    df_pred[df_pred["is_suspicious"] == 1]
    [["transaction_id", "amount", "hour_of_day", "daily_tx_count",
      "location", "category", "anomaly_score", "risk_level"]]
    .sort_values("anomaly_score")
    .reset_index(drop=True)
)
suspicious_df.columns = [
    "Transaction ID", "Amount ($)", "Hour", "Daily Count",
    "Location", "Category", "Anomaly Score", "Risk Level",
]
suspicious_df["Amount ($)"] = suspicious_df["Amount ($)"].map("${:,.2f}".format)

st.dataframe(
    suspicious_df.style
        .applymap(lambda v: "color: #e63946; font-weight: 700"
                  if v in ("CRITICAL", "HIGH") else
                  ("color: #f59e0b" if v == "MEDIUM" else ""),
                  subset=["Risk Level"])
        .set_properties(**{"background-color": "#071525", "color": "#e2e8f0"}),
    use_container_width=True,
    height=380,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Footer
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    '<p style="color:#475569;font-size:.78rem;text-align:center;">'
    "AI Financial Threat Monitor · v1.0 · "
    "Model: Isolation Forest (sklearn) · Real-time Anomaly Detection System</p>",
    unsafe_allow_html=True,
)
