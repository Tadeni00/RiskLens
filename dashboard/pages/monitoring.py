"""
FraudTrap Dashboard — Live Monitoring Page
Real-time system health and performance metrics.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import time

from dashboard.components import (
    kpi_row,
    line_chart,
    area_chart,
    dual_axis_chart,
    metric_table,
    page_container,
    section_divider,
    latency_display,
    metric_row,
    status_card,
)
from dashboard.components.data_loader import make_live_timeseries
from dashboard.theme.colors import Colors
from dashboard.theme.icons import Icons


def render(tenant_id: str):
    live, is_live = make_live_timeseries(48, tenant_id)

    if live.empty:
        st.warning("No live data available yet.")
        return

    rng = np.random.default_rng()

    with page_container(
        "Live Monitoring", "Real-time system health and performance metrics", "ACTIVITY"
    ):

        # ── Section 1: Real-time KPIs ─────────────────────────────────────
        auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)

        last = live.iloc[-1]
        prev = live.iloc[-2] if len(live) > 1 else last

        txn_per_sec = float(last["txn_per_min"]) / 60.0
        txn_per_sec_prev = float(prev["txn_per_min"]) / 60.0

        avg_latency = float(last["latency_p95"]) * 0.72
        avg_latency_prev = float(prev["latency_p95"]) * 0.72
        p95_latency = float(last["latency_p95"])
        p95_latency_prev = float(prev["latency_p95"])

        error_rate = float(rng.beta(1, 80) * 100)
        error_rate_prev = float(rng.beta(1, 80) * 100)
        queue_depth = int(rng.poisson(340))
        queue_depth_prev = int(rng.poisson(340))
        active_models = int(rng.integers(3, 6))
        active_models_prev = int(rng.integers(3, 6))

        kpi_row(
            [
                {
                    "label": "Transactions/sec",
                    "value": f"{txn_per_sec:,.1f}",
                    "trend": txn_per_sec - txn_per_sec_prev,
                    "trend_label": "vs prev hour",
                    "icon": "ACTIVITY",
                    "status": "healthy",
                },
                {
                    "label": "Avg Latency",
                    "value": f"{avg_latency:.1f}ms",
                    "trend": avg_latency - avg_latency_prev,
                    "delta_suffix": "ms",
                    "status": "healthy" if avg_latency < 60 else "warning",
                    "icon": "ZAP",
                },
                {
                    "label": "P95 Latency",
                    "value": f"{p95_latency:.0f}ms",
                    "trend": p95_latency - p95_latency_prev,
                    "delta_suffix": "ms",
                    "status": "healthy" if p95_latency < 100 else "warning",
                    "icon": "CLOCK",
                },
                {
                    "label": "Error Rate",
                    "value": f"{error_rate:.3f}%",
                    "trend": error_rate - error_rate_prev,
                    "delta_suffix": "%",
                    "status": "healthy" if error_rate < 0.5 else "warning",
                    "icon": "ALERT_TRIANGLE",
                },
                {
                    "label": "Queue Depth",
                    "value": f"{queue_depth:,}",
                    "trend": queue_depth - queue_depth_prev,
                    "status": "healthy" if queue_depth < 500 else "warning",
                    "icon": "LAYERS",
                },
                {
                    "label": "Active Models",
                    "value": str(active_models),
                    "trend": active_models - active_models_prev,
                    "status": "healthy",
                    "icon": "BOX",
                },
            ]
        )

        section_divider()

        # ── Section 2: Latency Monitoring ─────────────────────────────────
        p50 = float(last["latency_p95"]) * 0.58
        p95_val = float(last["latency_p95"])
        p99 = float(last["latency_p95"]) * 1.35

        latency_display(p50=p50, p95=p95_val, p99=p99, target=100.0)

        st.markdown(
            f"<div style='color:{Colors.TEXT_PRIMARY};font-weight:600;font-size:16px;margin:20px 0 8px'>"
            f"{Icons.html('CLOCK', 16, Colors.ACCENT)} P95 Latency — 48h</div>",
            unsafe_allow_html=True,
        )

        fig_lat = go.Figure()
        fig_lat.add_trace(
            go.Scatter(
                x=live["timestamp"],
                y=live["latency_p95"],
                mode="lines",
                fill="tozeroy",
                line=dict(color=Colors.CHART_1, width=2, shape="spline"),
                fillcolor=Colors.rgba(Colors.CHART_1, 0.12),
                name="P95 Latency",
            )
        )
        fig_lat.add_hline(
            y=200,
            line_dash="dash",
            line_color=Colors.CRITICAL,
            annotation_text="Critical (200ms)",
            annotation_font_color=Colors.CRITICAL,
        )
        fig_lat.add_hline(
            y=100,
            line_dash="dash",
            line_color=Colors.WARNING,
            annotation_text="Warning (100ms)",
            annotation_font_color=Colors.WARNING,
        )
        fig_lat.update_layout(
            height=280,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(
                family="'Inter', sans-serif", color=Colors.TEXT_SECONDARY, size=12
            ),
            margin=dict(l=0, r=0, t=20, b=0),
            yaxis=dict(
                title="ms",
                showgrid=True,
                gridcolor=Colors.BORDER_SUBTLE,
                tickfont=dict(color=Colors.TEXT_MUTED, size=11),
            ),
            xaxis=dict(showgrid=False, tickfont=dict(color=Colors.TEXT_MUTED, size=11)),
            hoverlabel=dict(
                bgcolor=Colors.BG_ELEVATED,
                bordercolor=Colors.BORDER_DEFAULT,
                font=dict(color=Colors.TEXT_PRIMARY, size=12),
            ),
            hovermode="x unified",
        )
        st.plotly_chart(fig_lat, use_container_width=True)

        section_divider()

        # ── Section 3: Throughput ─────────────────────────────────────────
        st.markdown(
            f"<div style='color:{Colors.TEXT_PRIMARY};font-weight:600;font-size:16px;margin-bottom:8px'>"
            f"{Icons.html('BAR_CHART', 16, Colors.ACCENT)} Throughput</div>",
            unsafe_allow_html=True,
        )

        tps_values = live["txn_per_min"] / 60.0
        pred_throughput = tps_values * rng.uniform(0.6, 0.9, len(live))

        col_tps, col_pred = st.columns(2)

        with col_tps:
            fig_tps = go.Figure()
            fig_tps.add_trace(
                go.Scatter(
                    x=live["timestamp"],
                    y=tps_values,
                    mode="lines",
                    fill="tozeroy",
                    line=dict(color=Colors.CHART_2, width=2, shape="spline"),
                    fillcolor=Colors.rgba(Colors.CHART_2, 0.12),
                    name="Txn/sec",
                )
            )
            fig_tps.update_layout(
                height=280,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(
                    family="'Inter', sans-serif", color=Colors.TEXT_SECONDARY, size=12
                ),
                title=dict(
                    text="Transactions per Second",
                    font=dict(size=14, color=Colors.TEXT_PRIMARY),
                ),
                margin=dict(l=0, r=0, t=32, b=0),
                yaxis=dict(
                    title="txns/sec",
                    showgrid=True,
                    gridcolor=Colors.BORDER_SUBTLE,
                    tickfont=dict(color=Colors.TEXT_MUTED, size=11),
                ),
                xaxis=dict(
                    showgrid=False, tickfont=dict(color=Colors.TEXT_MUTED, size=11)
                ),
                hoverlabel=dict(
                    bgcolor=Colors.BG_ELEVATED,
                    bordercolor=Colors.BORDER_DEFAULT,
                    font=dict(color=Colors.TEXT_PRIMARY, size=12),
                ),
                hovermode="x unified",
            )
            st.plotly_chart(fig_tps, use_container_width=True)

        with col_pred:
            fig_pred = go.Figure()
            fig_pred.add_trace(
                go.Scatter(
                    x=live["timestamp"],
                    y=pred_throughput,
                    mode="lines",
                    fill="tozeroy",
                    line=dict(color=Colors.CHART_5, width=2, shape="spline"),
                    fillcolor=Colors.rgba(Colors.CHART_5, 0.12),
                    name="Predictions/sec",
                )
            )
            fig_pred.update_layout(
                height=280,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(
                    family="'Inter', sans-serif", color=Colors.TEXT_SECONDARY, size=12
                ),
                title=dict(
                    text="Prediction Throughput",
                    font=dict(size=14, color=Colors.TEXT_PRIMARY),
                ),
                margin=dict(l=0, r=0, t=32, b=0),
                yaxis=dict(
                    title="preds/sec",
                    showgrid=True,
                    gridcolor=Colors.BORDER_SUBTLE,
                    tickfont=dict(color=Colors.TEXT_MUTED, size=11),
                ),
                xaxis=dict(
                    showgrid=False, tickfont=dict(color=Colors.TEXT_MUTED, size=11)
                ),
                hoverlabel=dict(
                    bgcolor=Colors.BG_ELEVATED,
                    bordercolor=Colors.BORDER_DEFAULT,
                    font=dict(color=Colors.TEXT_PRIMARY, size=12),
                ),
                hovermode="x unified",
            )
            st.plotly_chart(fig_pred, use_container_width=True)

        section_divider()

        # ── Section 4: Infrastructure Health ──────────────────────────────
        st.markdown(
            f"<div style='color:{Colors.TEXT_PRIMARY};font-weight:600;font-size:16px;margin-bottom:8px'>"
            f"{Icons.html('SERVER', 16, Colors.ACCENT)} Infrastructure Health</div>",
            unsafe_allow_html=True,
        )

        infra = [
            {
                "title": "CPU Utilization",
                "status": "healthy",
                "details": "16 cores — 3.2GHz",
                "icon": "CPU",
                "metrics": {
                    "Current": f"{rng.integers(42, 68)}%",
                    "Load Avg": f"{rng.uniform(1.2, 3.8):.1f}",
                    "Cores": "16",
                },
            },
            {
                "title": "Memory Usage",
                "status": "healthy",
                "details": "64GB DDR5 — ECC enabled",
                "icon": "SERVER",
                "metrics": {
                    "Used": f"{rng.uniform(38, 52):.1f}GB",
                    "Cached": f"{rng.uniform(4, 8):.1f}GB",
                    "Swap": "0.2GB",
                },
            },
            {
                "title": "Redis Hit Rate",
                "status": "healthy",
                "details": "Primary cache — port 6379",
                "icon": "DATABASE",
                "metrics": {
                    "Hit Rate": f"{rng.uniform(93, 98):.1f}%",
                    "Memory": "2.1GB",
                    "Evictions": "0",
                },
            },
            {
                "title": "Kafka Lag",
                "status": "warning",
                "details": "fraud-scores topic — consumer group",
                "icon": "Zap",
                "metrics": {
                    "Lag": f"{rng.integers(8000, 15000):,}",
                    "Throughput": f"{rng.uniform(7, 12):.1f}K/s",
                    "Partitions": "24",
                },
            },
            {
                "title": "Database Connections",
                "status": "healthy",
                "details": "ClickHouse — port 9000",
                "icon": "DATABASE",
                "metrics": {
                    "Active": f"{rng.integers(18, 32)}",
                    "Idle": f"{rng.integers(5, 12)}",
                    "Max": "100",
                },
            },
            {
                "title": "GPU Utilization",
                "status": "healthy",
                "details": "4x NVIDIA A10G — cluster",
                "icon": "CPU",
                "metrics": {
                    "Util": f"{rng.integers(55, 82)}%",
                    "VRAM": f"{rng.integers(14, 22)}/24GB",
                    "Temp": f"{rng.integers(58, 74)}°C",
                },
            },
        ]

        cols = st.columns(3)
        for i, comp in enumerate(infra):
            with cols[i % 3]:
                status_card(
                    title=comp["title"],
                    status=comp["status"],
                    details=comp["details"],
                    icon=comp["icon"],
                    metrics=comp["metrics"],
                )

        section_divider()

        # ── Section 5: Error Monitoring ───────────────────────────────────
        st.markdown(
            f"<div style='color:{Colors.TEXT_PRIMARY};font-weight:600;font-size:16px;margin-bottom:8px'>"
            f"{Icons.html('ALERT_TRIANGLE', 16, Colors.ACCENT)} Error Monitoring</div>",
            unsafe_allow_html=True,
        )

        error_rates = pd.Series(rng.beta(1, 80, len(live)) * 100, name="error_rate")

        fig_err = go.Figure()
        fig_err.add_trace(
            go.Scatter(
                x=live["timestamp"],
                y=error_rates,
                mode="lines",
                fill="tozeroy",
                line=dict(color=Colors.CHART_4, width=2, shape="spline"),
                fillcolor=Colors.rgba(Colors.CHART_4, 0.12),
                name="Error Rate",
            )
        )
        fig_err.add_hline(
            y=0.5,
            line_dash="dash",
            line_color=Colors.WARNING,
            annotation_text="Threshold (0.5%)",
            annotation_font_color=Colors.WARNING,
        )
        fig_err.update_layout(
            height=280,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(
                family="'Inter', sans-serif", color=Colors.TEXT_SECONDARY, size=12
            ),
            title=dict(
                text="Error Rate Over Time",
                font=dict(size=14, color=Colors.TEXT_PRIMARY),
            ),
            margin=dict(l=0, r=0, t=32, b=0),
            yaxis=dict(
                title="%",
                showgrid=True,
                gridcolor=Colors.BORDER_SUBTLE,
                tickfont=dict(color=Colors.TEXT_MUTED, size=11),
            ),
            xaxis=dict(showgrid=False, tickfont=dict(color=Colors.TEXT_MUTED, size=11)),
            hoverlabel=dict(
                bgcolor=Colors.BG_ELEVATED,
                bordercolor=Colors.BORDER_DEFAULT,
                font=dict(color=Colors.TEXT_PRIMARY, size=12),
            ),
            hovermode="x unified",
        )
        st.plotly_chart(fig_err, use_container_width=True)

        error_types = [
            "Timeout",
            "Connection Refused",
            "Rate Limited",
            "Validation Error",
            "Internal Error",
        ]
        error_counts = rng.poisson([12, 8, 5, 3, 2]).tolist()

        fig_err_dist = go.Figure()
        fig_err_dist.add_trace(
            go.Bar(
                x=error_types,
                y=error_counts,
                marker=dict(
                    color=[
                        Colors.CRITICAL,
                        Colors.WARNING,
                        Colors.ACCENT,
                        Colors.CHART_5,
                        Colors.CHART_4,
                    ],
                    cornerradius=4,
                ),
                text=error_counts,
                textposition="auto",
                textfont=dict(color=Colors.TEXT_PRIMARY, size=11),
            )
        )
        fig_err_dist.update_layout(
            height=280,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(
                family="'Inter', sans-serif", color=Colors.TEXT_SECONDARY, size=12
            ),
            title=dict(
                text="Error Distribution by Type",
                font=dict(size=14, color=Colors.TEXT_PRIMARY),
            ),
            margin=dict(l=0, r=0, t=32, b=0),
            yaxis=dict(
                title="Count",
                showgrid=True,
                gridcolor=Colors.BORDER_SUBTLE,
                tickfont=dict(color=Colors.TEXT_MUTED, size=11),
            ),
            xaxis=dict(showgrid=False, tickfont=dict(color=Colors.TEXT_MUTED, size=11)),
            hoverlabel=dict(
                bgcolor=Colors.BG_ELEVATED,
                bordercolor=Colors.BORDER_DEFAULT,
                font=dict(color=Colors.TEXT_PRIMARY, size=12),
            ),
        )
        st.plotly_chart(fig_err_dist, use_container_width=True)

        section_divider()

        # ── Section 6: Recent Alerts ──────────────────────────────────────
        st.markdown(
            f"<div style='color:{Colors.TEXT_PRIMARY};font-weight:600;font-size:16px;margin-bottom:8px'>"
            f"{Icons.html('BELL', 16, Colors.ACCENT)} Recent Alerts</div>",
            unsafe_allow_html=True,
        )

        now = datetime.now(timezone.utc)

        st.markdown(
            f"""
<div class="ft-alert warning">
    <div style="display:flex;align-items:center;gap:8px">
        {Icons.html("ALERT_TRIANGLE", 16)}
        <span>Kafka consumer lag exceeded threshold on fraud-scores topic — 14,200 messages behind</span>
        <span style="font-size:11px;color:{Colors.TEXT_MUTED};margin-left:auto">{(now - timedelta(minutes=4)).strftime("%H:%M")}</span>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
<div class="ft-alert critical">
    <div style="display:flex;align-items:center;gap:8px">
        {Icons.html("X_CIRCLE", 16)}
        <span>P95 latency breached SLA warning threshold — 112ms (target: 100ms)</span>
        <span style="font-size:11px;color:{Colors.TEXT_MUTED};margin-left:auto">{(now - timedelta(minutes=11)).strftime("%H:%M")}</span>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
<div class="ft-alert info">
    <div style="display:flex;align-items:center;gap:8px">
        {Icons.html("INFO", 16)}
        <span>GPU utilization spike resolved — cluster returned to normal operating range</span>
        <span style="font-size:11px;color:{Colors.TEXT_MUTED};margin-left:auto">{(now - timedelta(minutes=23)).strftime("%H:%M")}</span>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
<div class="ft-alert success">
    <div style="display:flex;align-items:center;gap:8px">
        {Icons.html("CHECK_CIRCLE", 16)}
        <span>Automated failover completed — traffic routed to backup inference node</span>
        <span style="font-size:11px;color:{Colors.TEXT_MUTED};margin-left:auto">{(now - timedelta(minutes=41)).strftime("%H:%M")}</span>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

    # ── Auto-refresh ──────────────────────────────────────────────────────
    if auto_refresh:
        time.sleep(30)
        st.rerun()
