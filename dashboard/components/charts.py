"""
FraudTrap Dashboard — Chart Components
Enterprise-grade Plotly chart wrappers with consistent theming.
"""

import plotly.graph_objects as go
import plotly.express as px
from dashboard.theme.colors import Colors
from dashboard.theme.typography import Typography

# ── Shared Layout Defaults ───────────────────────────────────────────────────
LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(
        family="'Inter', 'IBM Plex Sans', sans-serif",
        color=Colors.TEXT_SECONDARY,
        size=12,
    ),
    margin=dict(l=0, r=0, t=32, b=0),
    xaxis=dict(
        showgrid=False,
        showline=False,
        zeroline=False,
        tickfont=dict(color=Colors.TEXT_MUTED, size=11),
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor=Colors.BORDER_SUBTLE,
        gridwidth=1,
        showline=False,
        zeroline=False,
        tickfont=dict(color=Colors.TEXT_MUTED, size=11),
    ),
    legend=dict(
        font=dict(color=Colors.TEXT_SECONDARY, size=11),
        bgcolor="rgba(0,0,0,0)",
        borderwidth=0,
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
    ),
    hoverlabel=dict(
        bgcolor=Colors.BG_ELEVATED,
        bordercolor=Colors.BORDER_DEFAULT,
        font=dict(color=Colors.TEXT_PRIMARY, size=12, family="'Inter', sans-serif"),
    ),
    hovermode="x unified",
)


def _apply_layout(fig: go.Figure, **kwargs) -> go.Figure:
    """Apply shared layout defaults with overrides."""
    layout = {**LAYOUT_DEFAULTS, **kwargs}
    fig.update_layout(**layout)
    return fig


def _apply_default_style(fig: go.Figure) -> go.Figure:
    """Apply consistent trace styling."""
    fig.update_traces(
        line=dict(width=2),
        marker=dict(size=6),
    )
    return fig


def line_chart(
    x,
    y,
    title: str = "",
    color: str = None,
    fill: bool = False,
    height: int = 300,
    show_grid: bool = True,
    **kwargs,
) -> go.Figure:
    """Create a themed line chart."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(color=color or Colors.CHART_1, width=2, shape="spline"),
            fill="tozeroy" if fill else None,
            fillcolor=Colors.rgba(color or Colors.CHART_1, 0.08) if fill else None,
            **kwargs,
        )
    )
    _apply_layout(
        fig,
        height=height,
        title=dict(text=title, font=dict(size=14, color=Colors.TEXT_PRIMARY)),
    )
    if not show_grid:
        fig.update_yaxes(showgrid=False)
    return fig


def area_chart(
    x, y, title: str = "", color: str = None, height: int = 300, **kwargs
) -> go.Figure:
    """Create a themed area chart."""
    return line_chart(x, y, title, color, fill=True, height=height, **kwargs)


def bar_chart(
    x,
    y,
    title: str = "",
    color: str = None,
    height: int = 300,
    orientation: str = "v",
    show_values: bool = False,
    **kwargs,
) -> go.Figure:
    """Create a themed bar chart."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x,
            y=y,
            orientation=orientation,
            marker=dict(
                color=color or Colors.CHART_1,
                cornerradius=4,
            ),
            text=y if show_values else None,
            textposition="auto",
            textfont=dict(color=Colors.TEXT_PRIMARY, size=11),
            **kwargs,
        )
    )
    _apply_layout(
        fig,
        height=height,
        title=dict(text=title, font=dict(size=14, color=Colors.TEXT_PRIMARY)),
    )
    return fig


def horizontal_bar(
    categories, values, title: str = "", colors: list = None, height: int = 300
) -> go.Figure:
    """Create a horizontal bar chart."""
    fig = go.Figure()
    bar_colors = colors or Colors.chart_palette(len(categories))
    fig.add_trace(
        go.Bar(
            y=categories,
            x=values,
            orientation="h",
            marker=dict(color=bar_colors, cornerradius=4),
            text=values,
            textposition="auto",
            textfont=dict(color=Colors.TEXT_PRIMARY, size=11),
        )
    )
    _apply_layout(
        fig,
        height=height,
        title=dict(text=title, font=dict(size=14, color=Colors.TEXT_PRIMARY)),
    )
    fig.update_layout(barmode="stack")
    return fig


def multi_line_chart(
    data: dict, title: str = "", height: int = 300, x_label: str = "", y_label: str = ""
) -> go.Figure:
    """
    Create a multi-line chart.

    Args:
        data: Dict of {name: (x_values, y_values)}
    """
    fig = go.Figure()
    palette = Colors.chart_palette(len(data))
    for i, (name, (x, y)) in enumerate(data.items()):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=name,
                line=dict(color=palette[i], width=2, shape="spline"),
            )
        )
    layout = {**LAYOUT_DEFAULTS, "height": height}
    if title:
        layout["title"] = dict(
            text=title, font=dict(size=14, color=Colors.TEXT_PRIMARY)
        )
    if x_label:
        layout["xaxis_title"] = x_label
    if y_label:
        layout["yaxis_title"] = y_label
    fig.update_layout(**layout)
    return fig


def gauge_chart(
    value: float,
    title: str = "",
    min_val: float = 0,
    max_val: float = 1,
    thresholds: list = None,
    height: int = 250,
) -> go.Figure:
    """Create a gauge chart."""
    if thresholds is None:
        thresholds = [
            {"range": [0, 0.5], "color": Colors.CRITICAL},
            {"range": [0.5, 0.8], "color": Colors.WARNING},
            {"range": [0.8, 1], "color": Colors.SUCCESS},
        ]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number=dict(font=dict(size=28, color=Colors.TEXT_PRIMARY)),
            gauge=dict(
                axis=dict(range=[min_val, max_val], tickcolor=Colors.TEXT_MUTED),
                bar=dict(color=Colors.ACCENT),
                bgcolor=Colors.BG_SECONDARY,
                borderwidth=0,
                steps=thresholds,
            ),
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=20, r=20, t=40, b=0),
        title=dict(text=title, font=dict(size=14, color=Colors.TEXT_PRIMARY)),
    )
    return fig


def heatmap_chart(
    z, x, y, title: str = "", height: int = 350, colorscale: str = "Blues"
) -> go.Figure:
    """Create a heatmap chart."""
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(
                tickfont=dict(color=Colors.TEXT_MUTED, size=10),
                titlefont=dict(color=Colors.TEXT_MUTED, size=10),
            ),
        )
    )
    _apply_layout(
        fig,
        height=height,
        title=dict(text=title, font=dict(size=14, color=Colors.TEXT_PRIMARY)),
    )
    return fig


def scatter_chart(
    x,
    y,
    title: str = "",
    color: str = None,
    size: list = None,
    hover_text: list = None,
    height: int = 350,
) -> go.Figure:
    """Create a scatter chart."""
    fig = go.Figure(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(
                color=color or Colors.CHART_1,
                size=size or 8,
                opacity=0.7,
                line=dict(width=1, color=Colors.BG_CARD),
            ),
            text=hover_text,
            hoverinfo="text+x+y",
        )
    )
    _apply_layout(
        fig,
        height=height,
        title=dict(text=title, font=dict(size=14, color=Colors.TEXT_PRIMARY)),
    )
    return fig


def waterfall_chart(
    categories,
    values,
    title: str = "",
    height: int = 350,
    color_pos: str = None,
    color_neg: str = None,
) -> go.Figure:
    """Create a waterfall chart for SHAP-like feature contributions."""
    measures = ["relative"] * len(values)
    fig = go.Figure(
        go.Waterfall(
            name="SHAP",
            orientation="v",
            measure=measures,
            x=categories,
            y=values,
            textposition="outside",
            text=[f"{v:+.3f}" for v in values],
            textfont=dict(color=Colors.TEXT_SECONDARY, size=11),
            connector=dict(line=dict(color=Colors.BORDER_DEFAULT, width=1)),
            increasing=dict(marker=dict(color=color_pos or Colors.CRITICAL)),
            decreasing=dict(marker=dict(color=color_neg or Colors.SUCCESS)),
            totals=dict(marker=dict(color=Colors.ACCENT)),
        )
    )
    _apply_layout(
        fig,
        height=height,
        title=dict(text=title, font=dict(size=14, color=Colors.TEXT_PRIMARY)),
    )
    return fig


def treemap_chart(
    labels, parents, values, title: str = "", height: int = 350, colors: list = None
) -> go.Figure:
    """Create a treemap chart."""
    fig = go.Figure(
        go.Treemap(
            labels=labels,
            parents=parents,
            values=values,
            marker=dict(
                colors=colors or Colors.chart_palette(len(labels)),
                line=dict(width=2, color=Colors.BG_CARD),
            ),
            textfont=dict(color=Colors.TEXT_PRIMARY, size=12),
            pathbar=dict(visible=False),
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=0, r=0, t=32, b=0),
        title=dict(text=title, font=dict(size=14, color=Colors.TEXT_PRIMARY)),
    )
    return fig


def histogram_chart(
    x,
    title: str = "",
    color: str = None,
    nbins: int = 30,
    height: int = 300,
    show_legend: bool = False,
) -> go.Figure:
    """Create a histogram chart."""
    fig = go.Figure(
        go.Histogram(
            x=x,
            nbinsx=nbins,
            marker=dict(color=color or Colors.CHART_1, cornerradius=2),
            showlegend=show_legend,
        )
    )
    _apply_layout(
        fig,
        height=height,
        title=dict(text=title, font=dict(size=14, color=Colors.TEXT_PRIMARY)),
    )
    fig.update_layout(bargap=0.05)
    return fig


def dual_axis_chart(
    x, y1, y2, label1: str = "", label2: str = "", title: str = "", height: int = 300
) -> go.Figure:
    """Create a dual-axis chart."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y1,
            name=label1,
            line=dict(color=Colors.CHART_1, width=2),
            yaxis="y",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y2,
            name=label2,
            line=dict(color=Colors.CHART_2, width=2),
            yaxis="y2",
        )
    )
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        height=height,
        title=dict(text=title, font=dict(size=14, color=Colors.TEXT_PRIMARY)),
        yaxis=dict(
            title=label1,
            titlefont=dict(color=Colors.CHART_1),
            tickfont=dict(color=Colors.CHART_1),
        ),
        yaxis2=dict(
            title=label2,
            titlefont=dict(color=Colors.CHART_2),
            tickfont=dict(color=Colors.CHART_2),
            overlaying="y",
            side="right",
        ),
    )
    return fig


def progress_ring(
    value: float,
    max_val: float = 1.0,
    size: int = 120,
    color: str = None,
    label: str = "",
    show_pct: bool = True,
):
    """Render an SVG progress ring directly via st.components."""
    import streamlit as st

    color = color or Colors.ACCENT
    pct = min(value / max_val, 1.0) if max_val > 0 else 0
    radius = 40
    circumference = 2 * 3.14159 * radius
    offset = circumference * (1 - pct)

    text_html = ""
    if show_pct:
        text_html = f"""
        <text x="50%" y="50%" text-anchor="middle" dominant-baseline="central"
              fill="{Colors.TEXT_PRIMARY}" font-size="18" font-weight="700"
              font-family="sans-serif">{pct:.0%}</text>
        """
    if label:
        text_html += f"""
        <text x="50%" y="68%" text-anchor="middle" dominant-baseline="central"
              fill="{Colors.TEXT_MUTED}" font-size="10" font-weight="500"
              font-family="sans-serif">{label}</text>
        """

    svg = f"""<svg width="{size}" height="{size}" viewBox="0 0 100 100">
    <circle cx="50" cy="50" r="{radius}" fill="none"
            stroke="{Colors.BORDER_DEFAULT}" stroke-width="8"/>
    <circle cx="50" cy="50" r="{radius}" fill="none"
            stroke="{color}" stroke-width="8"
            stroke-dasharray="{circumference}" stroke-dashoffset="{offset}"
            stroke-linecap="round" transform="rotate(-90 50 50)"/>
    {text_html}
</svg>"""
    st.components.v1.html(svg, width=size, height=size)
