"""
RiskLens Console — Table Components
Professional data tables with sorting, filtering, and status indicators.
"""

import streamlit as st
import pandas as pd
from dashboard.theme.colors import Colors
from dashboard.theme.typography import Typography
from dashboard.theme.icons import Icons


def data_table(
    df: pd.DataFrame,
    columns: dict = None,
    max_rows: int = 50,
    sortable: bool = True,
    status_col: str = None,
    row_actions: list = None,
    striped: bool = False,
) -> str:
    """
    Render a professional data table.

    Args:
        df: DataFrame to display
        columns: Dict of {col_name: display_name} mapping
        max_rows: Maximum rows to display
        sortable: Whether columns are sortable
        status_col: Column name to use for row status coloring
        row_actions: List of action dicts for each row
        striped: Alternate row backgrounds
    """
    if columns:
        display_df = df[list(columns.keys())].head(max_rows).copy()
        display_df.columns = [columns[k] for k in columns.keys()]
    else:
        display_df = df.head(max_rows).copy()

    headers_html = ""
    for col in display_df.columns:
        headers_html += f'<th style="padding:12px 16px;color:{Colors.TEXT_MUTED};font-weight:{Typography.WEIGHT_SEMIBOLD};font-size:{Typography.TEXT_SM};text-transform:uppercase;letter-spacing:{Typography.TRACKING_WIDER};text-align:left;border-bottom:1px solid {Colors.BORDER_DEFAULT};background:{Colors.BG_SECONDARY};position:sticky;top:0">{col}</th>'

    rows_html = ""
    for i, (_, row) in enumerate(display_df.iterrows()):
        row_style = ""
        if striped and i % 2 == 1:
            row_style = f"background:{Colors.BG_SECONDARY};"

        cells_html = ""
        for col in display_df.columns:
            val = row[col]
            cell_style = f"padding:10px 16px;color:{Colors.TEXT_SECONDARY};border-bottom:1px solid {Colors.BORDER_SUBTLE};font-size:{Typography.TEXT_BASE}"

            if status_col and col == display_df.columns[0]:
                status_val = str(row.get(status_col, "")).lower()
                status_colors = {
                    "healthy": Colors.SUCCESS,
                    "active": Colors.SUCCESS,
                    "warning": Colors.WARNING,
                    "pending": Colors.WARNING,
                    "critical": Colors.CRITICAL,
                    "blocked": Colors.CRITICAL,
                    "offline": Colors.TEXT_MUTED,
                    "archived": Colors.TEXT_MUTED,
                }
                dot_color = status_colors.get(status_val, Colors.TEXT_MUTED)
                val = f'<span style="display:flex;align-items:center;gap:8px"><span style="width:8px;height:8px;border-radius:50%;background:{dot_color}"></span>{val}</span>'

            if isinstance(val, float):
                val = f"{val:.4f}" if val < 1 else f"{val:,.2f}"

            cells_html += f'<td style="{cell_style}">{val}</td>'

        rows_html += f'<tr style="{row_style}">{cells_html}</tr>'

    table_html = f"""
<div style="overflow-x:auto;border:1px solid {Colors.BORDER_DEFAULT};border-radius:10px;background:{Colors.BG_CARD}">
    <table style="width:100%;border-collapse:separate;border-spacing:0;font-family:{Typography.FONT_FAMILY};font-size:{Typography.TEXT_BASE}">
        <thead><tr>{headers_html}</tr></thead>
        <tbody>{rows_html}</tbody>
    </table>
</div>
"""
    st.markdown(table_html, unsafe_allow_html=True)


def metric_table(metrics: list, title: str = "") -> str:
    """
    Render a key-value metric table.

    Args:
        metrics: List of dicts with 'label', 'value', optional 'status', 'trend'
    """
    if title:
        st.markdown(
            f"<div style='margin-bottom:12px;color:{Colors.TEXT_PRIMARY};font-weight:{Typography.WEIGHT_SEMIBOLD};font-size:{Typography.TEXT_MD}'>{title}</div>",
            unsafe_allow_html=True,
        )

    rows_html = ""
    for m in metrics:
        status_html = ""
        if m.get("status"):
            status_html = f'<span class="ft-badge {m["status"]}" style="margin-left:8px">{m["status"].title()}</span>'

        trend_html = ""
        if m.get("trend") is not None:
            trend = m["trend"]
            if trend > 0:
                trend_html = f'<span style="color:{Colors.TREND_UP};font-size:{Typography.TEXT_SM}">+{trend:.1f}%</span>'
            elif trend < 0:
                trend_html = f'<span style="color:{Colors.TREND_DOWN};font-size:{Typography.TEXT_SM}">{trend:.1f}%</span>'

        rows_html += f"""
<tr>
    <td style="padding:10px 16px;color:{Colors.TEXT_SECONDARY};border-bottom:1px solid {Colors.BORDER_SUBTLE};font-size:{Typography.TEXT_BASE}">{m['label']}</td>
    <td style="padding:10px 16px;color:{Colors.TEXT_PRIMARY};font-weight:{Typography.WEIGHT_SEMIBOLD};border-bottom:1px solid {Colors.BORDER_SUBTLE};text-align:right;font-size:{Typography.TEXT_BASE}">{m['value']}{status_html} {trend_html}</td>
</tr>"""

    st.markdown(
        f"""
<div style="border:1px solid {Colors.BORDER_DEFAULT};border-radius:10px;background:{Colors.BG_CARD};overflow:hidden">
    <table style="width:100%;border-collapse:collapse;font-family:{Typography.FONT_FAMILY}">
        <tbody>{rows_html}</tbody>
    </table>
</div>
""",
        unsafe_allow_html=True,
    )


def leader_board(
    entries: list,
    title: str = "",
    rank_col: str = "rank",
    name_col: str = "name",
    value_col: str = "value",
) -> str:
    """
    Render a leaderboard card.

    Args:
        entries: List of dicts with rank, name, value, optional 'trend', 'badge'
    """
    if title:
        st.markdown(
            f"<div style='margin-bottom:12px;color:{Colors.TEXT_PRIMARY};font-weight:{Typography.WEIGHT_SEMIBOLD};font-size:{Typography.TEXT_MD}'>{title}</div>",
            unsafe_allow_html=True,
        )

    items_html = ""
    for i, entry in enumerate(entries[:10]):
        rank = entry.get(rank_col, i + 1)
        name = entry.get(name_col, "")
        value = entry.get(value_col, "")
        trend = entry.get("trend")
        badge = entry.get("badge")

        rank_color = Colors.TEXT_MUTED
        if rank == 1:
            rank_color = Colors.WARNING
        elif rank == 2:
            rank_color = Colors.TEXT_SECONDARY
        elif rank == 3:
            rank_color = Colors.PHASE_1

        trend_html = ""
        if trend is not None:
            if trend > 0:
                trend_html = f'<span style="color:{Colors.TREND_UP};font-size:{Typography.TEXT_SM}">+{trend:.1f}%</span>'
            elif trend < 0:
                trend_html = f'<span style="color:{Colors.TREND_DOWN};font-size:{Typography.TEXT_SM}">{trend:.1f}%</span>'

        badge_html = ""
        if badge:
            badge_type = entry.get("badge_type", "info")
            badge_html = f'<span class="ft-badge {badge_type}" style="margin-left:8px">{badge}</span>'

        items_html += f"""
<div style="display:flex;align-items:center;padding:12px 16px;border-bottom:1px solid {Colors.BORDER_SUBTLE};gap:12px">
    <span style="color:{rank_color};font-weight:{Typography.WEIGHT_BOLD};font-size:{Typography.TEXT_MD};min-width:28px;text-align:center">#{rank}</span>
    <span style="flex:1;color:{Colors.TEXT_PRIMARY};font-weight:{Typography.WEIGHT_MEDIUM}">{name}{badge_html}</span>
    <span style="color:{Colors.TEXT_SECONDARY};font-weight:{Typography.WEIGHT_SEMIBOLD}">{value}</span>
    {trend_html}
</div>"""

    st.markdown(
        f"""
<div style="border:1px solid {Colors.BORDER_DEFAULT};border-radius:10px;background:{Colors.BG_CARD};overflow:hidden">
    {items_html}
</div>
""",
        unsafe_allow_html=True,
    )
