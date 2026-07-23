"""
FraudTrap Dashboard — Alert Components
Professional alert and notification display components.
"""

import streamlit as st
from datetime import datetime
from dashboard.theme.colors import Colors
from dashboard.theme.typography import Typography
from dashboard.theme.icons import Icons


def alert(
    message: str, level: str = "info", icon: str = None, timestamp: datetime = None
):
    """
    Render an alert component.

    Args:
        message: Alert message text
        level: "info", "success", "warning", "critical"
        icon: Optional icon name override
        timestamp: Optional timestamp for the alert
    """
    default_icons = {
        "info": "INFO",
        "success": "CHECK_CIRCLE",
        "warning": "ALERT_TRIANGLE",
        "critical": "X_CIRCLE",
    }
    icon_name = icon or default_icons.get(level, "INFO")

    timestamp_html = ""
    if timestamp:
        timestamp_html = f'<span style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_MUTED};margin-left:auto">{timestamp.strftime("%H:%M:%S")}</span>'

    st.markdown(
        f"""
<div class="ft-alert {level}">
    <div style="display:flex;align-items:center;gap:8px">
        {Icons.html(icon_name, 16)}
        <span>{message}</span>
        {timestamp_html}
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def alert_list(alerts: list):
    """
    Render a list of alerts.

    Args:
        alerts: List of dicts with 'message', 'level', optional 'icon', 'timestamp'
    """
    for a in alerts:
        alert(
            message=a["message"],
            level=a.get("level", "info"),
            icon=a.get("icon"),
            timestamp=a.get("timestamp"),
        )


def notification_badge(count: int, level: str = "info"):
    """Render a notification count badge."""
    if count == 0:
        return ""
    return f"""<span style="display:inline-flex;align-items:center;justify-content:center;min-width:20px;height:20px;padding:0 6px;border-radius:10px;font-size:11px;font-weight:700;background:{Colors.rgba(getattr(Colors, level.upper(), Colors.ACCENT), 0.2)};color:{getattr(Colors, level.upper(), Colors.ACCENT)}">{count}</span>"""


def status_timeline(events: list):
    """
    Render a status event timeline.

    Args:
        events: List of dicts with 'time', 'message', 'level'
    """
    items_html = ""
    for e in events:
        level = e.get("level", "info")
        dot_color = {
            "success": Colors.SUCCESS,
            "warning": Colors.WARNING,
            "critical": Colors.CRITICAL,
            "info": Colors.ACCENT,
        }.get(level, Colors.TEXT_MUTED)

        items_html += f"""
<div style="display:flex;gap:12px;padding:8px 0">
    <div style="display:flex;flex-direction:column;align-items:center;min-width:16px">
        <div style="width:8px;height:8px;border-radius:50%;background:{dot_color};margin-top:4px"></div>
    </div>
    <div style="flex:1">
        <div style="font-size:{Typography.TEXT_SM};color:{Colors.TEXT_PRIMARY}">{e['message']}</div>
        <div style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_MUTED}">{e.get('time', '')}</div>
    </div>
</div>"""

    st.markdown(
        f"""
<div style="background:{Colors.BG_CARD};border:1px solid {Colors.BORDER_DEFAULT};border-radius:10px;padding:16px">
    {items_html}
</div>
""",
        unsafe_allow_html=True,
    )
