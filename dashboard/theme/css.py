"""
RiskLens Design System — Global CSS
Injects the complete enterprise design language into Streamlit.
"""

import streamlit as st
from dashboard.theme.colors import Colors
from dashboard.theme.typography import Typography


def inject_global_css():
    """Inject the complete design system CSS."""
    st.markdown(
        f"""
<style>
/* ═══════════════════════════════════════════════════════════════════════════
   RiskLens Enterprise Design System
   ═══════════════════════════════════════════════════════════════════════════ */

/* ── Root Variables ─────────────────────────────────────────────────────── */
:root {{
    --bg-primary: {Colors.BG_PRIMARY};
    --bg-secondary: {Colors.BG_SECONDARY};
    --bg-card: {Colors.BG_CARD};
    --bg-card-hover: {Colors.BG_CARD_HOVER};
    --bg-elevated: {Colors.BG_ELEVATED};
    --bg-input: {Colors.BG_INPUT};
    --border-default: {Colors.BORDER_DEFAULT};
    --border-subtle: {Colors.BORDER_SUBTLE};
    --border-strong: {Colors.BORDER_STRONG};
    --border-focus: {Colors.BORDER_FOCUS};
    --text-primary: {Colors.TEXT_PRIMARY};
    --text-secondary: {Colors.TEXT_SECONDARY};
    --text-muted: {Colors.TEXT_MUTED};
    --accent: {Colors.ACCENT};
    --accent-light: {Colors.ACCENT_LIGHT};
    --accent-bg: {Colors.ACCENT_BG};
    --success: {Colors.SUCCESS};
    --warning: {Colors.WARNING};
    --critical: {Colors.CRITICAL};
    --font-family: {Typography.FONT_FAMILY};
    --font-mono: {Typography.FONT_MONO};
}}

/* ── Global Reset ───────────────────────────────────────────────────────── */
.stApp {{
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-family) !important;
}}

.stApp > header {{
    background-color: var(--bg-primary) !important;
}}

/* ── Sidebar ────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {{
    background-color: {Colors.BG_SECONDARY} !important;
    border-right: 1px solid {Colors.BORDER_SUBTLE} !important;
    padding: 0 !important;
}}

section[data-testid="stSidebar"] > div {{
    padding-top: 0 !important;
}}

section[data-testid="stSidebar"] [data-testid="stSidebarNav"] {{
    padding-top: 0 !important;
}}

section[data-testid="stSidebar"] .stRadio > div {{
    gap: 2px !important;
}}

section[data-testid="stSidebar"] .stRadio > div > label {{
    padding: 10px 16px !important;
    border-radius: 8px !important;
    margin: 0 8px !important;
    transition: all 0.15s ease !important;
    background: transparent !important;
    color: {Colors.TEXT_SECONDARY} !important;
    font-size: {Typography.TEXT_BASE} !important;
    font-weight: {Typography.WEIGHT_MEDIUM} !important;
    cursor: pointer !important;
}}

section[data-testid="stSidebar"] .stRadio > div > label:hover {{
    background: {Colors.BG_CARD} !important;
    color: {Colors.TEXT_PRIMARY} !important;
}}

section[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"] {{
    background: {Colors.ACCENT_BG} !important;
    color: {Colors.ACCENT_LIGHT} !important;
    font-weight: {Typography.WEIGHT_SEMIBOLD} !important;
}}

section[data-testid="stSidebar"] .stRadio > div > label > div {{
    color: inherit !important;
}}

/* ── Sidebar Logo ───────────────────────────────────────────────────────── */
.sidebar-logo {{
    padding: 24px 20px 16px 20px;
    border-bottom: 1px solid {Colors.BORDER_SUBTLE};
    margin-bottom: 16px;
}}

.sidebar-logo h2 {{
    font-family: var(--font-family);
    font-size: {Typography.TEXT_LG};
    font-weight: {Typography.WEIGHT_BOLD};
    color: {Colors.TEXT_PRIMARY};
    margin: 0;
    letter-spacing: -0.02em;
}}

.sidebar-logo p {{
    font-family: var(--font-family);
    font-size: {Typography.TEXT_XS};
    color: {Colors.TEXT_MUTED};
    margin: 4px 0 0 0;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}}

/* ── Sidebar Section Labels ─────────────────────────────────────────────── */
.sidebar-section {{
    padding: 8px 20px 4px 20px;
    font-family: var(--font-family);
    font-size: {Typography.TEXT_XS};
    font-weight: {Typography.WEIGHT_SEMIBOLD};
    color: {Colors.TEXT_MUTED};
    text-transform: uppercase;
    letter-spacing: {Typography.TRACKING_WIDER};
}}

/* ── Page Header ────────────────────────────────────────────────────────── */
.page-header {{
    padding: 0 0 24px 0;
    border-bottom: 1px solid {Colors.BORDER_SUBTLE};
    margin-bottom: 24px;
}}

.page-header h1 {{
    {Typography.page_title_style()}
    margin-bottom: 4px;
}}

.page-header p {{
    {Typography.body_style()}
    color: {Colors.TEXT_MUTED};
    font-size: {Typography.TEXT_BASE};
}}

/* ── Section Header ─────────────────────────────────────────────────────── */
.section-header {{
    padding: 0 0 16px 0;
    margin-bottom: 16px;
}}

.section-header h2 {{
    {Typography.section_title_style()}
    margin-bottom: 4px;
}}

.section-header p {{
    {Typography.body_style()}
    color: {Colors.TEXT_MUTED};
    font-size: {Typography.TEXT_SM};
}}

/* ── Cards ──────────────────────────────────────────────────────────────── */
.ft-card {{
    background: {Colors.BG_CARD};
    border: 1px solid {Colors.BORDER_DEFAULT};
    border-radius: 12px;
    padding: 20px;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
}}

.ft-card:hover {{
    border-color: {Colors.BORDER_STRONG};
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}}

.ft-card-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 12px;
}}

.ft-card-title {{
    {Typography.card_title_style()}
}}

.ft-card-subtitle {{
    font-size: {Typography.TEXT_SM};
    color: {Colors.TEXT_MUTED};
}}

/* ── KPI Cards ──────────────────────────────────────────────────────────── */
.kpi-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 24px;
}}

.kpi-card {{
    background: {Colors.BG_CARD};
    border: 1px solid {Colors.BORDER_DEFAULT};
    border-radius: 12px;
    padding: 20px;
    position: relative;
    overflow: hidden;
}}

.kpi-card::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: {Colors.ACCENT};
    opacity: 0;
    transition: opacity 0.15s ease;
}}

.kpi-card:hover::before {{
    opacity: 1;
}}

.kpi-label {{
    {Typography.metric_label_style()}
    margin-bottom: 8px;
}}

.kpi-value {{
    {Typography.metric_large_style()}
    margin-bottom: 4px;
}}

.kpi-trend {{
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-size: {Typography.TEXT_SM};
    font-weight: {Typography.WEIGHT_MEDIUM};
    padding: 2px 8px;
    border-radius: 6px;
}}

.kpi-trend.up {{
    color: {Colors.TREND_UP};
    background: {Colors.SUCCESS_BG};
}}

.kpi-trend.down {{
    color: {Colors.TREND_DOWN};
    background: {Colors.CRITICAL_BG};
}}

.kpi-trend.neutral {{
    color: {Colors.TREND_NEUTRAL};
    background: {Colors.rgba(Colors.TEXT_MUTED, 0.12)};
}}

/* ── Status Indicators ──────────────────────────────────────────────────── */
.status-dot {{
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
}}

.status-dot.healthy {{
    background: {Colors.STATUS_HEALTHY};
    box-shadow: 0 0 6px {Colors.rgba(Colors.STATUS_HEALTHY, 0.4)};
}}

.status-dot.warning {{
    background: {Colors.STATUS_WARNING};
    box-shadow: 0 0 6px {Colors.rgba(Colors.STATUS_WARNING, 0.4)};
}}

.status-dot.critical {{
    background: {Colors.STATUS_CRITICAL};
    box-shadow: 0 0 6px {Colors.rgba(Colors.STATUS_CRITICAL, 0.4)};
}}

.status-dot.offline {{
    background: {Colors.STATUS_OFFLINE};
}}

/* ── Status Pulse Animation ─────────────────────────────────────────────── */
@keyframes pulse {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.5; }}
}}

.status-pulse {{
    animation: pulse 2s ease-in-out infinite;
}}

/* ── Badges ─────────────────────────────────────────────────────────────── */
.ft-badge {{
    display: inline-flex;
    align-items: center;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: {Typography.TEXT_SM};
    font-weight: {Typography.WEIGHT_MEDIUM};
    font-family: var(--font-family);
    white-space: nowrap;
}}

.ft-badge.success {{
    background: {Colors.SUCCESS_BG};
    color: {Colors.SUCCESS};
    border: 1px solid {Colors.rgba(Colors.SUCCESS, 0.2)};
}}

.ft-badge.warning {{
    background: {Colors.WARNING_BG};
    color: {Colors.WARNING};
    border: 1px solid {Colors.rgba(Colors.WARNING, 0.2)};
}}

.ft-badge.critical {{
    background: {Colors.CRITICAL_BG};
    color: {Colors.CRITICAL};
    border: 1px solid {Colors.rgba(Colors.CRITICAL, 0.2)};
}}

.ft-badge.info {{
    background: {Colors.INFO_BG};
    color: {Colors.ACCENT_LIGHT};
    border: 1px solid {Colors.rgba(Colors.ACCENT, 0.2)};
}}

.ft-badge.muted {{
    background: {Colors.rgba(Colors.TEXT_MUTED, 0.12)};
    color: {Colors.TEXT_MUTED};
    border: 1px solid {Colors.BORDER_SUBTLE};
}}

/* ── Phase Badges ───────────────────────────────────────────────────────── */
.phase-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    border-radius: 8px;
    font-size: {Typography.TEXT_SM};
    font-weight: {Typography.WEIGHT_SEMIBOLD};
    font-family: var(--font-family);
}}

.phase-badge.phase-1 {{
    background: {Colors.rgba(Colors.PHASE_1, 0.15)};
    color: {Colors.PHASE_1};
    border: 1px solid {Colors.rgba(Colors.PHASE_1, 0.3)};
}}

.phase-badge.phase-2 {{
    background: {Colors.rgba(Colors.PHASE_2, 0.15)};
    color: {Colors.PHASE_2};
    border: 1px solid {Colors.rgba(Colors.PHASE_2, 0.3)};
}}

.phase-badge.phase-3 {{
    background: {Colors.rgba(Colors.PHASE_3, 0.15)};
    color: {Colors.PHASE_3};
    border: 1px solid {Colors.rgba(Colors.PHASE_3, 0.3)};
}}

/* ── Tables ─────────────────────────────────────────────────────────────── */
.ft-table {{
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-family: var(--font-family);
    font-size: {Typography.TEXT_BASE};
}}

.ft-table thead th {{
    background: {Colors.BG_SECONDARY};
    color: {Colors.TEXT_MUTED};
    font-weight: {Typography.WEIGHT_SEMIBOLD};
    font-size: {Typography.TEXT_SM};
    text-transform: uppercase;
    letter-spacing: {Typography.TRACKING_WIDER};
    padding: 12px 16px;
    text-align: left;
    border-bottom: 1px solid {Colors.BORDER_DEFAULT};
    position: sticky;
    top: 0;
}}

.ft-table tbody td {{
    padding: 12px 16px;
    color: {Colors.TEXT_SECONDARY};
    border-bottom: 1px solid {Colors.BORDER_SUBTLE};
    vertical-align: middle;
}}

.ft-table tbody tr:hover td {{
    background: {Colors.BG_CARD_HOVER};
}}

.ft-table tbody tr:last-child td {{
    border-bottom: none;
}}

/* ── Alerts ─────────────────────────────────────────────────────────────── */
.ft-alert {{
    padding: 12px 16px;
    border-radius: 8px;
    border-left: 3px solid;
    font-size: {Typography.TEXT_BASE};
    margin-bottom: 8px;
}}

.ft-alert.critical {{
    background: {Colors.rgba(Colors.CRITICAL, 0.08)};
    border-color: {Colors.CRITICAL};
    color: {Colors.CRITICAL_LIGHT};
}}

.ft-alert.warning {{
    background: {Colors.rgba(Colors.WARNING, 0.08)};
    border-color: {Colors.WARNING};
    color: {Colors.WARNING_LIGHT};
}}

.ft-alert.success {{
    background: {Colors.rgba(Colors.SUCCESS, 0.08)};
    border-color: {Colors.SUCCESS};
    color: {Colors.SUCCESS_LIGHT};
}}

.ft-alert.info {{
    background: {Colors.rgba(Colors.ACCENT, 0.08)};
    border-color: {Colors.ACCENT};
    color: {Colors.ACCENT_LIGHT};
}}

/* ── Pipeline Visualization ─────────────────────────────────────────────── */
.pipeline-container {{
    background: {Colors.BG_CARD};
    border: 1px solid {Colors.BORDER_DEFAULT};
    border-radius: 12px;
    padding: 24px;
    overflow-x: auto;
}}

.pipeline-step {{
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 16px;
    background: {Colors.BG_SECONDARY};
    border: 1px solid {Colors.BORDER_DEFAULT};
    border-radius: 8px;
    font-size: {Typography.TEXT_BASE};
    color: {Colors.TEXT_SECONDARY};
    transition: all 0.2s ease;
    min-width: 200px;
}}

.pipeline-step.active {{
    border-color: {Colors.ACCENT};
    background: {Colors.ACCENT_BG};
    color: {Colors.TEXT_PRIMARY};
}}

.pipeline-step.complete {{
    border-color: {Colors.SUCCESS};
    background: {Colors.SUCCESS_BG};
}}

.pipeline-arrow {{
    color: {Colors.TEXT_MUTED};
    font-size: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    min-width: 24px;
}}

/* ── Progress Bar ───────────────────────────────────────────────────────── */
.ft-progress {{
    height: 6px;
    background: {Colors.BG_SECONDARY};
    border-radius: 3px;
    overflow: hidden;
    margin: 8px 0;
}}

.ft-progress-bar {{
    height: 100%;
    border-radius: 3px;
    transition: width 0.3s ease;
}}

.ft-progress-bar.success {{
    background: {Colors.SUCCESS};
}}

.ft-progress-bar.warning {{
    background: {Colors.WARNING};
}}

.ft-progress-bar.critical {{
    background: {Colors.CRITICAL};
}}

.ft-progress-bar.accent {{
    background: {Colors.ACCENT};
}}

/* ── Metric Inline ──────────────────────────────────────────────────────── */
.metric-inline {{
    display: flex;
    align-items: baseline;
    gap: 8px;
}}

.metric-inline .value {{
    font-size: {Typography.TEXT_2XL};
    font-weight: {Typography.WEIGHT_BOLD};
    color: {Colors.TEXT_PRIMARY};
    font-family: var(--font-family);
}}

.metric-inline .unit {{
    font-size: {Typography.TEXT_SM};
    color: {Colors.TEXT_MUTED};
    font-weight: {Typography.WEIGHT_MEDIUM};
}}

/* ── Divider ────────────────────────────────────────────────────────────── */
.ft-divider {{
    border: none;
    border-top: 1px solid {Colors.BORDER_SUBTLE};
    margin: 24px 0;
}}

/* ── Empty State ────────────────────────────────────────────────────────── */
.ft-empty {{
    text-align: center;
    padding: 48px 24px;
    color: {Colors.TEXT_MUTED};
}}

.ft-empty svg {{
    margin-bottom: 16px;
    opacity: 0.5;
}}

/* ── Streamlit Overrides ────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0;
    background: {Colors.BG_SECONDARY};
    border-radius: 10px;
    padding: 4px;
    border: 1px solid {Colors.BORDER_DEFAULT};
}}

.stTabs [data-baseweb="tab"] {{
    padding: 10px 20px;
    border-radius: 8px;
    font-family: var(--font-family);
    font-size: {Typography.TEXT_BASE};
    font-weight: {Typography.WEIGHT_MEDIUM};
    color: {Colors.TEXT_SECONDARY};
    background: transparent;
    border: none;
    transition: all 0.15s ease;
}}

.stTabs [data-baseweb="tab"]:hover {{
    color: {Colors.TEXT_PRIMARY};
    background: {Colors.BG_CARD};
}}

.stTabs [aria-selected="true"] {{
    color: {Colors.TEXT_PRIMARY} !important;
    background: {Colors.BG_CARD} !important;
    font-weight: {Typography.WEIGHT_SEMIBOLD} !important;
}}

.stTabs [data-baseweb="tab-highlight"] {{
    display: none;
}}

.stTabs [data-baseweb="tab-border"] {{
    display: none;
}}

/* ── Selectbox / Input Overrides ────────────────────────────────────────── */
.stSelectbox > div > div,
.stMultiSelect > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stDateInput > div > div {{
    background-color: {Colors.BG_INPUT} !important;
    border-color: {Colors.BORDER_DEFAULT} !important;
    color: {Colors.TEXT_PRIMARY} !important;
    border-radius: 8px !important;
    font-family: var(--font-family) !important;
}}

.stSelectbox > div > div:focus-within,
.stTextInput > div > div > input:focus-within {{
    border-color: {Colors.BORDER_FOCUS} !important;
    box-shadow: 0 0 0 2px {Colors.rgba(Colors.ACCENT, 0.2)} !important;
}}

/* ── Button Overrides ───────────────────────────────────────────────────── */
.stButton > button {{
    background-color: {Colors.ACCENT} !important;
    color: {Colors.TEXT_PRIMARY} !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: var(--font-family) !important;
    font-weight: {Typography.WEIGHT_MEDIUM} !important;
    padding: 8px 20px !important;
    transition: all 0.15s ease !important;
}}

.stButton > button:hover {{
    background-color: {Colors.ACCENT_LIGHT} !important;
    box-shadow: 0 2px 8px {Colors.rgba(Colors.ACCENT, 0.3)} !important;
}}

.stDownloadButton > button {{
    background-color: {Colors.BG_CARD} !important;
    border: 1px solid {Colors.BORDER_DEFAULT} !important;
    color: {Colors.TEXT_SECONDARY} !important;
    border-radius: 8px !important;
    font-family: var(--font-family) !important;
}}

/* ── Checkbox / Radio Overrides ─────────────────────────────────────────── */
.stCheckbox > label > span:first-child,
.stRadio > label > span:first-child {{
    border-color: {Colors.BORDER_DEFAULT} !important;
    background-color: {Colors.BG_INPUT} !important;
}}

.stCheckbox > label > span:first-child[data-checked="true"],
.stRadio > label > span:first-child[data-checked="true"] {{
    background-color: {Colors.ACCENT} !important;
    border-color: {Colors.ACCENT} !important;
}}

/* ── Slider Overrides ───────────────────────────────────────────────────── */
.stSlider > div > div > div > div {{
    background-color: {Colors.ACCENT} !important;
}}

.stSlider > div > div > div > div > div {{
    background-color: {Colors.TEXT_PRIMARY} !important;
    border-color: {Colors.ACCENT} !important;
}}

/* ── Expander Overrides ─────────────────────────────────────────────────── */
.stExpander {{
    background: {Colors.BG_CARD};
    border: 1px solid {Colors.BORDER_DEFAULT};
    border-radius: 10px;
}}

.stExpander header {{
    font-family: var(--font-family);
    font-weight: {Typography.WEIGHT_MEDIUM};
    color: {Colors.TEXT_PRIMARY};
}}

/* ── Plotly Chart Container ─────────────────────────────────────────────── */
.js-plotly-plot .plotly {{
    border-radius: 8px;
    overflow: hidden;
}}

/* ── Scrollbar ──────────────────────────────────────────────────────────── */
::-webkit-scrollbar {{
    width: 6px;
    height: 6px;
}}

::-webkit-scrollbar-track {{
    background: transparent;
}}

::-webkit-scrollbar-thumb {{
    background: {Colors.BORDER_DEFAULT};
    border-radius: 3px;
}}

::-webkit-scrollbar-thumb:hover {{
    background: {Colors.BORDER_STRONG};
}}

/* ── Responsive Grid Helpers ────────────────────────────────────────────── */
.ft-grid-2 {{
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 16px;
}}

.ft-grid-3 {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
}}

.ft-grid-4 {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
}}

/* ── Animations ─────────────────────────────────────────────────────────── */
@keyframes fadeIn {{
    from {{ opacity: 0; transform: translateY(8px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}

.ft-animate {{
    animation: fadeIn 0.3s ease-out;
}}

@keyframes slideIn {{
    from {{ opacity: 0; transform: translateX(-8px); }}
    to {{ opacity: 1; transform: translateX(0); }}
}}

.ft-slide-in {{
    animation: slideIn 0.2s ease-out;
}}
</style>
""",
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str = "", icon: str = ""):
    """Render a page header."""
    icon_html = ""
    if icon:
        from dashboard.theme.icons import Icons

        icon_html = Icons.html(icon, size=28, color=Colors.ACCENT)
        icon_html += "&nbsp;&nbsp;"
    st.markdown(
        f"""
<div class="page-header">
    <h1>{icon_html}{title}</h1>
    {"<p>" + subtitle + "</p>" if subtitle else ""}
</div>
""",
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str = ""):
    """Render a section header."""
    st.markdown(
        f"""
<div class="section-header">
    <h2>{title}</h2>
    {"<p>" + subtitle + "</p>" if subtitle else ""}
</div>
""",
        unsafe_allow_html=True,
    )
