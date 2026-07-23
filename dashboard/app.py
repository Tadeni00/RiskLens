"""
FraudTrap — Enterprise Fraud Intelligence Platform
Main application entry point with professional sidebar navigation.
"""

import sys
from pathlib import Path
from datetime import datetime

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudTrap — Enterprise Fraud Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "FraudTrap — Enterprise Fraud Intelligence Platform",
        "Report a bug": "https://github.com/fraudtrap/issues",
    },
)

# ── Theme Injection ──────────────────────────────────────────────────────────
from dashboard.theme.css import inject_global_css

inject_global_css()

# ── Navigation ───────────────────────────────────────────────────────────────
from dashboard.components.navigation import (
    sidebar_header,
    sidebar_tenant_selector,
    sidebar_section_label,
)

with st.sidebar:
    sidebar_header()

    # Tenant selector
    sidebar_section_label("Environment")
    tenants = [
        "bank_ng_gtb",
        "bank_ng_access",
        "bank_ng_zenith",
        "fintech_ng_opay",
        "fintech_ng_kuda",
        "bank_za_fnb",
        "fintech_za_yoco",
        "all_tenants",
    ]
    selected_tenant = sidebar_tenant_selector(tenants)

    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)

    # Main navigation
    sidebar_section_label("Navigation")

    PAGES = {
        "📊 Overview": "overview",
        "🔍 Risk Intelligence": "intelligence",
        "🧠 Behavior Profiles": "behavior",
        "🤖 Models": "models",
        "💡 Explainability": "explainability",
        "📈 Drift Monitoring": "drift",
        "📡 Live Monitoring": "monitoring",
        "📋 Compliance": "compliance",
        "🔄 Model Lifecycle": "lifecycle",
    }

    selected_page = st.radio(
        "Navigation",
        list(PAGES.keys()),
        key="nav_radio",
        label_visibility="collapsed",
    )

    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)

    # Sidebar footer
    sidebar_section_label("System")
    st.markdown(
        f"""
<div style="padding:0 12px;font-size:12px;color:#6F7B8F">
    <div style="display:flex;justify-content:space-between;padding:4px 0">
        <span>Environment</span><span style="color:#17A673">Production</span>
    </div>
    <div style="display:flex;justify-content:space-between;padding:4px 0">
        <span>API Status</span><span style="color:#17A673">Connected</span>
    </div>
    <div style="display:flex;justify-content:space-between;padding:4px 0">
        <span>Version</span><span>2.1.0</span>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

# ── Global Header ────────────────────────────────────────────────────────────
from dashboard.components.navigation import global_header

global_header(
    tenant=selected_tenant,
    environment="production",
    champion_model="CatBoost v1.0",
    ml_phase="Phase 3",
    last_refresh=datetime.now(),
)

# ── Page Router ──────────────────────────────────────────────────────────────
page_module = PAGES[selected_page]

try:
    if page_module == "overview":
        from dashboard.pages.overview import render

        render(selected_tenant)
    elif page_module == "intelligence":
        from dashboard.pages.intelligence import render

        render(selected_tenant)
    elif page_module == "behavior":
        from dashboard.pages.behavior import render

        render(selected_tenant)
    elif page_module == "models":
        from dashboard.pages.models import render

        render(selected_tenant)
    elif page_module == "explainability":
        from dashboard.pages.explainability import render

        render(selected_tenant)
    elif page_module == "drift":
        from dashboard.pages.drift import render

        render(selected_tenant)
    elif page_module == "monitoring":
        from dashboard.pages.monitoring import render

        render(selected_tenant)
    elif page_module == "compliance":
        from dashboard.pages.compliance import render

        render(selected_tenant)
    elif page_module == "lifecycle":
        from dashboard.pages.lifecycle import render

        render(selected_tenant)
    else:
        from dashboard.pages.overview import render

        render(selected_tenant)
except Exception as e:
    st.error(f"Error loading page: {e}")
    st.exception(e)
