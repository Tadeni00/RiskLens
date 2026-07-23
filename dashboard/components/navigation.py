"""
FraudTrap Dashboard — Navigation Components
Professional sidebar and header navigation.
"""

import streamlit as st
from datetime import datetime
from dashboard.theme.colors import Colors
from dashboard.theme.typography import Typography
from dashboard.theme.icons import Icons


def sidebar_header():
    """Render the sidebar logo and branding."""
    st.markdown(
        f"""
<div class="sidebar-logo">
    <h2>FraudTrap</h2>
    <p>Enterprise Fraud Intelligence</p>
</div>
""",
        unsafe_allow_html=True,
    )


def sidebar_tenant_selector(tenants: list, key: str = "tenant_select"):
    """Render tenant selector in sidebar."""
    selected = st.selectbox(
        "Tenant",
        tenants,
        index=0,
        key=key,
        label_visibility="collapsed",
    )
    return selected


def sidebar_section_label(label: str):
    """Render a navigation section label."""
    st.markdown(f'<div class="sidebar-section">{label}</div>', unsafe_allow_html=True)


def global_header(
    tenant: str,
    environment: str = "production",
    champion_model: str = "CatBoost v1.0",
    ml_phase: str = "Phase 3",
    last_refresh: datetime = None,
):
    """Render the global header bar."""
    refresh_text = (
        last_refresh.strftime("%H:%M:%S")
        if last_refresh
        else datetime.now().strftime("%H:%M:%S")
    )
    current_time = datetime.now().strftime("%H:%M:%S UTC")

    phase_class = (
        "phase-3" if "3" in ml_phase else ("phase-2" if "2" in ml_phase else "phase-1")
    )

    env_color = (
        Colors.SUCCESS if environment.lower() == "production" else Colors.WARNING
    )

    st.markdown(
        f"""
<div style="display:flex;align-items:center;justify-content:space-between;padding:12px 24px;background:{Colors.BG_SECONDARY};border-bottom:1px solid {Colors.BORDER_SUBTLE};margin:-60px -60px 24px -60px">
    <div style="display:flex;align-items:center;gap:24px">
        <div style="display:flex;align-items:center;gap:8px">
            <span style="color:{Colors.ACCENT};font-weight:{Typography.WEIGHT_BOLD};font-size:{Typography.TEXT_LG};letter-spacing:-0.02em">{Icons.html('SHIELD', 20, Colors.ACCENT)} FraudTrap</span>
        </div>
        <div style="display:flex;align-items:center;gap:8px;padding:6px 12px;background:{Colors.BG_CARD};border-radius:8px;border:1px solid {Colors.BORDER_DEFAULT}">
            <span style="font-size:{Typography.TEXT_SM};color:{Colors.TEXT_MUTED}">Tenant</span>
            <span style="font-size:{Typography.TEXT_SM};color:{Colors.TEXT_PRIMARY};font-weight:{Typography.WEIGHT_MEDIUM}">{tenant}</span>
        </div>
        <div style="display:flex;align-items:center;gap:6px;padding:6px 12px;background:{Colors.rgba(env_color, 0.12)};border-radius:8px;border:1px solid {Colors.rgba(env_color, 0.2)}">
            <span style="width:6px;height:6px;border-radius:50%;background:{env_color}"></span>
            <span style="font-size:{Typography.TEXT_SM};color:{env_color};font-weight:{Typography.WEIGHT_MEDIUM}">{environment}</span>
        </div>
        <span class="phase-badge {phase_class}">{ml_phase}</span>
        <div style="display:flex;align-items:center;gap:6px;padding:6px 12px;background:{Colors.BG_CARD};border-radius:8px;border:1px solid {Colors.BORDER_DEFAULT}">
            <span style="font-size:{Typography.TEXT_SM};color:{Colors.TEXT_MUTED}">Champion</span>
            <span style="font-size:{Typography.TEXT_SM};color:{Colors.TEXT_PRIMARY};font-weight:{Typography.WEIGHT_MEDIUM}">{champion_model}</span>
        </div>
    </div>
    <div style="display:flex;align-items:center;gap:16px">
        <div style="display:flex;align-items:center;gap:6px;color:{Colors.TEXT_MUTED};font-size:{Typography.TEXT_SM}">
            {Icons.html('CLOCK', 14, Colors.TEXT_MUTED)}
            <span>{current_time}</span>
        </div>
        <div style="display:flex;align-items:center;gap:6px;color:{Colors.TEXT_MUTED};font-size:{Typography.TEXT_SM}">
            {Icons.html('REFRESH', 14, Colors.TEXT_MUTED)}
            <span>Last: {refresh_text}</span>
        </div>
        <div style="width:32px;height:32px;border-radius:8px;background:{Colors.ACCENT_BG};display:flex;align-items:center;justify-content:center;cursor:pointer;border:1px solid {Colors.rgba(Colors.ACCENT, 0.2)}">
            {Icons.html('BELL', 16, Colors.ACCENT)}
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def tab_navigation(tabs: list, key: str = "main_tabs"):
    """Render tab navigation."""
    return st.tabs(tabs)


def breadcrumb(items: list):
    """Render a breadcrumb trail."""
    parts = []
    for i, item in enumerate(items):
        if i < len(items) - 1:
            parts.append(
                f'<span style="color:{Colors.TEXT_MUTED};cursor:pointer">{item}</span>'
            )
        else:
            parts.append(
                f'<span style="color:{Colors.TEXT_PRIMARY};font-weight:{Typography.WEIGHT_MEDIUM}">{item}</span>'
            )
    st.markdown(
        f'<div style="padding:8px 0;margin-bottom:16px;font-size:{Typography.TEXT_SM}">{" " .join(parts)}</div>',
        unsafe_allow_html=True,
    )
