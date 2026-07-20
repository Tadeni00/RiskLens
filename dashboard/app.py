"""
FraudPolice Dashboard — Main Entry Point
Streamlit multi-page app serving both data scientists and compliance teams.
Run: streamlit run dashboard/app.py
"""
import os

import streamlit as st

st.set_page_config(
    page_title="FraudPolice · Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background: #0F1117; }
    [data-testid="stSidebar"] * { color: #FAFAFA !important; }
    .metric-card {
        background: #1A1D27; border-radius: 8px; padding: 1rem 1.2rem;
        border: 1px solid #2A2D3A; margin-bottom: 0.5rem;
    }
    .metric-value { font-size: 1.8rem; font-weight: 600; color: #FAFAFA; }
    .metric-label { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 0.05em; }
    .alert-critical { background: #3D1515; border-left: 3px solid #E24B4A; padding: 0.75rem 1rem; border-radius: 4px; }
    .alert-warning  { background: #3D2F0A; border-left: 3px solid #EF9F27; padding: 0.75rem 1rem; border-radius: 4px; }
    .alert-ok       { background: #0F2A1A; border-left: 3px solid #1D9E75; padding: 0.75rem 1rem; border-radius: 4px; }
    .phase-badge {
        display: inline-block; padding: 3px 12px; border-radius: 99px;
        font-size: 0.75rem; font-weight: 600; letter-spacing: 0.04em;
    }
    div[data-testid="stMetricValue"] { font-size: 1.6rem !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 6px 16px; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ FraudPolice")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        options=[
            "📊 Overview",
            "🔬 EDA & Data Quality",
            "🤖 Model Performance",
            "🔍 Explainability",
            "📡 Live Monitoring",
            "⚠️ Drift Detection",
            "🔄 Model Lifecycle",
            "📋 Compliance",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    tenant_id = st.selectbox(
        "Tenant",
        options=["bank_ng_gtb", "bank_ke_equity", "fintech_za_yoco", "all_tenants"],
        index=0,
    )
    st.caption(f"Environment: **{os.getenv('ENVIRONMENT', 'development')}**")

# ── Page routing ──────────────────────────────────────────────────────────────
PAGE_MAP = {
    "📊 Overview":         "overview",
    "🔬 EDA & Data Quality": "eda",
    "🤖 Model Performance": "model_performance",
    "🔍 Explainability":   "explainability",
    "📡 Live Monitoring":  "live_monitoring",
    "⚠️ Drift Detection":  "drift",
    "🔄 Model Lifecycle":  "lifecycle",
    "📋 Compliance":       "compliance",
}

if page in PAGE_MAP:
    module_name = PAGE_MAP[page]
    try:
        mod = __import__(f"dashboard.pages.{module_name}", fromlist=["render"])
        mod.render(tenant_id)
    except Exception as exc:
        st.error(f"Failed to load **{page}** page: {exc}")
        st.exception(exc)
