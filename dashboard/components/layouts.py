"""
FraudTrap Dashboard — Layout Components
Page layout helpers for consistent structure.
"""
import streamlit as st
from dashboard.theme.colors import Colors
from dashboard.theme.typography import Typography
from dashboard.theme.icons import Icons


def page_container(title: str, subtitle: str = "", icon: str = None):
    """
    Create a page container with header.

    Usage:
        with page_container("Overview", "Real-time fraud monitoring", "HOME"):
            # page content here
            pass
    """
    class PageContext:
        def __enter__(self):
            from dashboard.theme.css import page_header
            page_header(title, subtitle, icon)
            return self

        def __exit__(self, *args):
            pass

    return PageContext()


def card_grid(columns: int = 3, gap: int = 16):
    """Create a grid of columns for cards."""
    return st.columns(columns, gap=f"{gap}px")


def section_divider():
    """Render a section divider."""
    st.markdown('<hr class="ft-divider">', unsafe_allow_html=True)


def two_panel(left_content, right_content, left_ratio: int = 1, right_ratio: int = 1):
    """Create a two-panel layout."""
    col1, col2 = st.columns([left_ratio, right_ratio])
    with col1:
        left_content()
    with col2:
        right_content()


def three_panel(p1, p2, p3):
    """Create a three-panel layout."""
    col1, col2, col3 = st.columns(3)
    with col1:
        p1()
    with col2:
        p2()
    with col3:
        p3()


def metric_row(metrics: list):
    """
    Render a row of metrics.

    Args:
        metrics: List of dicts with 'label', 'value', optional 'color', 'icon'
    """
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            icon_html = ""
            if m.get("icon"):
                icon_html = Icons.html(m["icon"], 14, Colors.ACCENT) + " "

            st.markdown(f"""
<div style="padding:12px 16px;background:{Colors.BG_CARD};border:1px solid {Colors.BORDER_DEFAULT};border-radius:8px">
    <div style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_MUTED};text-transform:uppercase;letter-spacing:{Typography.TRACKING_WIDER};margin-bottom:4px">{icon_html}{m['label']}</div>
    <div style="font-size:{Typography.TEXT_LG};font-weight:{Typography.WEIGHT_BOLD};color:{m.get('color', Colors.TEXT_PRIMARY)}">{m['value']}</div>
</div>
""", unsafe_allow_html=True)


def info_panel(title: str, items: list, icon: str = None):
    """Render an info panel with a list of items."""
    items_html = ""
    for item in items:
        if isinstance(item, dict):
            label = item.get("label", "")
            value = item.get("value", "")
            items_html += f"""
<div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid {Colors.BORDER_SUBTLE}">
    <span style="color:{Colors.TEXT_MUTED};font-size:{Typography.TEXT_SM}">{label}</span>
    <span style="color:{Colors.TEXT_PRIMARY};font-size:{Typography.TEXT_SM};font-weight:{Typography.WEIGHT_MEDIUM}">{value}</span>
</div>"""
        else:
            items_html += f"""
<div style="padding:6px 0;border-bottom:1px solid {Colors.BORDER_SUBTLE}">
    <span style="color:{Colors.TEXT_SECONDARY};font-size:{Typography.TEXT_BASE}">{item}</span>
</div>"""

    icon_html = ""
    if icon:
        icon_html = Icons.html(icon, 16, Colors.ACCENT) + " "

    st.markdown(f"""
<div style="background:{Colors.BG_CARD};border:1px solid {Colors.BORDER_DEFAULT};border-radius:10px;padding:20px">
    <div style="font-size:{Typography.TEXT_MD};font-weight:{Typography.WEIGHT_SEMIBOLD};color:{Colors.TEXT_PRIMARY};margin-bottom:12px">{icon_html}{title}</div>
    {items_html}
</div>
""", unsafe_allow_html=True)
