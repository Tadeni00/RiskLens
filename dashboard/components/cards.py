"""
RiskLens Console — Reusable Card Components
Enterprise-grade card components for KPIs, status, and information display.
"""

import streamlit as st
from dashboard.theme.colors import Colors
from dashboard.theme.typography import Typography
from dashboard.theme.icons import Icons


def kpi_card(
    label: str,
    value: str,
    trend: float = None,
    trend_label: str = "",
    icon: str = None,
    status: str = None,
    sparkline_data: list = None,
    delta_suffix: str = "",
):
    """
    Render a KPI card with value, trend, and optional sparkline.

    Args:
        label: Metric label (e.g., "Transactions Today")
        value: Formatted value (e.g., "1.2M")
        trend: Percentage change (positive = up, negative = down)
        trend_label: Label for the trend (e.g., "vs yesterday")
        icon: Icon name from Icons class
        status: "healthy", "warning", "critical", or None
        sparkline_data: List of numeric values for mini sparkline
        delta_suffix: Suffix for trend display (e.g., "%")
    """
    trend_html = ""
    if trend is not None:
        if trend > 0:
            trend_class = "up"
            arrow = Icons.html("TRENDING_UP", size=12, color=Colors.TREND_UP)
            trend_text = f"+{trend:.1f}{delta_suffix}"
        elif trend < 0:
            trend_class = "down"
            arrow = Icons.html("TRENDING_DOWN", size=12, color=Colors.TREND_DOWN)
            trend_text = f"{trend:.1f}{delta_suffix}"
        else:
            trend_class = "neutral"
            arrow = ""
            trend_text = f"0{delta_suffix}"
        trend_html = (
            f'<span class="kpi-trend {trend_class}">{arrow} {trend_text}</span>'
        )
        if trend_label:
            trend_html += f' <span style="color:{Colors.TEXT_MUTED};font-size:{Typography.TEXT_XS};margin-left:4px">{trend_label}</span>'

    icon_html = ""
    if icon:
        icon_html = Icons.html(icon, size=18, color=Colors.ACCENT)

    status_html = ""
    if status:
        status_html = f'<span class="status-dot {status} status-pulse"></span>'

    st.markdown(
        f"""
<div class="kpi-card">
    <div class="kpi-label">{status_html}{icon_html} {label}</div>
    <div class="kpi-value">{value}</div>
    {trend_html}
</div>
""",
        unsafe_allow_html=True,
    )


def kpi_row(kpis: list):
    """
    Render a row of KPI cards in a responsive grid.

    Args:
        kpis: List of dicts, each with keys matching kpi_card() args
    """
    n = len(kpis)
    cols_html = ""
    for kpi in kpis:
        trend_html = ""
        trend = kpi.get("trend")
        trend_label = kpi.get("trend_label", "")
        delta_suffix = kpi.get("delta_suffix", "")
        if trend is not None:
            if trend > 0:
                trend_class = "up"
                arrow = Icons.html("TRENDING_UP", size=12, color=Colors.TREND_UP)
                trend_text = f"+{trend:.1f}{delta_suffix}"
            elif trend < 0:
                trend_class = "down"
                arrow = Icons.html("TRENDING_DOWN", size=12, color=Colors.TREND_DOWN)
                trend_text = f"{trend:.1f}{delta_suffix}"
            else:
                trend_class = "neutral"
                arrow = ""
                trend_text = f"0{delta_suffix}"
            trend_html = (
                f'<span class="kpi-trend {trend_class}">{arrow} {trend_text}</span>'
            )
            if trend_label:
                trend_html += f' <span style="color:{Colors.TEXT_MUTED};font-size:{Typography.TEXT_XS};margin-left:4px">{trend_label}</span>'

        icon_html = ""
        icon = kpi.get("icon")
        if icon:
            icon_html = Icons.html(icon, size=18, color=Colors.ACCENT)

        status_html = ""
        status = kpi.get("status")
        if status:
            status_html = f'<span class="status-dot {status}"></span>'

        cols_html += f"""
<div class="kpi-card">
    <div class="kpi-label">{status_html}{icon_html} {kpi['label']}</div>
    <div class="kpi-value">{kpi['value']}</div>
    {trend_html}
</div>"""

    grid_class = f"ft-grid-{min(n, 4)}" if n <= 4 else "ft-grid-4"
    st.markdown(f'<div class="{grid_class}">{cols_html}</div>', unsafe_allow_html=True)


def status_card(
    title: str, status: str, details: str = "", icon: str = None, metrics: dict = None
):
    """
    Render a status/health card for infrastructure components.

    Args:
        title: Component name (e.g., "Redis")
        status: "healthy", "warning", "critical", "offline"
        details: Additional status text
        icon: Icon name
        metrics: Dict of metric_name -> value to display
    """
    status_labels = {
        "healthy": "Healthy",
        "warning": "Warning",
        "critical": "Critical",
        "offline": "Offline",
    }

    icon_html = ""
    if icon:
        icon_html = Icons.html(icon, size=16, color=Colors.TEXT_SECONDARY)

    metrics_html = ""
    if metrics:
        for name, val in metrics.items():
            metrics_html += f'<div style="display:flex;justify-content:space-between;padding:4px 0;border-top:1px solid {Colors.BORDER_SUBTLE}"><span style="color:{Colors.TEXT_MUTED};font-size:{Typography.TEXT_SM}">{name}</span><span style="color:{Colors.TEXT_SECONDARY};font-size:{Typography.TEXT_SM};font-weight:{Typography.WEIGHT_MEDIUM}">{val}</span></div>'

    st.markdown(
        f"""
<div class="ft-card" style="padding:16px">
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
        {icon_html}
        <span style="color:{Colors.TEXT_PRIMARY};font-weight:{Typography.WEIGHT_SEMIBOLD};font-size:{Typography.TEXT_MD}">{title}</span>
        <span class="ft-badge {status}" style="margin-left:auto">{status_labels.get(status, status)}</span>
    </div>
    {"<p style='color:" + Colors.TEXT_MUTED + ";font-size:" + Typography.TEXT_SM + ";margin:0'>" + details + "</p>" if details else ""}
    {metrics_html}
</div>
""",
        unsafe_allow_html=True,
    )


def info_card(
    title: str,
    content: str,
    icon: str = None,
    badge: str = None,
    badge_type: str = "info",
):
    """Render an information card with optional icon and badge."""
    icon_html = ""
    if icon:
        icon_html = Icons.html(icon, size=16, color=Colors.ACCENT)

    badge_html = ""
    if badge:
        badge_html = f'<span class="ft-badge {badge_type}">{badge}</span>'

    st.markdown(
        f"""
<div class="ft-card" style="padding:16px">
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
        {icon_html}
        <span style="color:{Colors.TEXT_PRIMARY};font-weight:{Typography.WEIGHT_SEMIBOLD};font-size:{Typography.TEXT_MD}">{title}</span>
        {badge_html}
    </div>
    <div style="color:{Colors.TEXT_SECONDARY};font-size:{Typography.TEXT_BASE};line-height:1.6">{content}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def metric_card_row(metrics: list):
    """
    Render a row of metric cards (compact variant).

    Args:
        metrics: List of dicts with 'label', 'value', optional 'color'
    """
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        color = m.get("color", Colors.TEXT_PRIMARY)
        with col:
            st.markdown(
                f"""
<div class="ft-card" style="padding:16px;text-align:center">
    <div style="font-size:{Typography.TEXT_SM};color:{Colors.TEXT_MUTED};margin-bottom:6px;text-transform:uppercase;letter-spacing:{Typography.TRACKING_WIDER}">{m['label']}</div>
    <div style="font-size:{Typography.TEXT_2XL};font-weight:{Typography.WEIGHT_BOLD};color:{color};font-family:{Typography.FONT_FAMILY}">{m['value']}</div>
</div>
""",
                unsafe_allow_html=True,
            )
