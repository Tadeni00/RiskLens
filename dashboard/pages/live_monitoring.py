"""FraudTrap Dashboard — Live Monitoring Page"""

import time
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from dashboard.components.data_loader import make_live_timeseries


def render(tenant_id: str):
    st.title("📡 Live Monitoring")
    st.caption("Real-time operational metrics. Auto-refresh every 30 seconds.")

    auto_refresh = st.toggle("Auto-refresh (30s)", value=False)

    live, _ = make_live_timeseries(48, tenant_id)

    if live.empty:
        st.warning("No live data available yet.")
        return

    # KPIs — guard against short DataFrames
    last_row = live.iloc[-1] if len(live) > 0 else None
    if last_row is None:
        st.warning("Insufficient data points.")
        return

    head_6 = (
        live["fraud_rate"].head(6).mean()
        if len(live) >= 6
        else live["fraud_rate"].mean()
    )
    tail_6 = (
        live["fraud_rate"].tail(6).mean()
        if len(live) >= 6
        else live["fraud_rate"].mean()
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Txn/min (now)", f"{int(last_row['txn_per_min']):,}")
    c2.metric(
        "Fraud Rate (1h)", f"{tail_6*100:.2f}%", delta=f"{(tail_6 - head_6)*100:+.2f}%"
    )
    c3.metric(
        "P95 Latency",
        f"{last_row['latency_p95']:.0f}ms",
        delta_color="inverse",
        delta=f"{last_row['latency_p95']-78:.0f}ms vs target",
    )
    c4.metric("FP Rate (1h)", f"{live['fp_rate'].tail(6).mean()*100:.2f}%")

    st.markdown("---")

    # Latency heatmap
    st.subheader("P95 Scoring Latency (48h)")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=live["timestamp"],
            y=live["latency_p95"],
            mode="lines",
            fill="tozeroy",
            line=dict(color="#3B82F6", width=2),
            fillcolor="rgba(59,130,246,0.15)",
        )
    )
    fig.add_hline(
        y=100,
        line_dash="dash",
        line_color="#EF4444",
        annotation_text="SLA wall (100ms)",
    )
    fig.add_hline(
        y=90, line_dash="dot", line_color="#F59E0B", annotation_text="Warning (90ms)"
    )
    fig.update_layout(
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title="ms",
        margin=dict(l=0, r=0, t=20, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Fraud rate trend
    st.subheader("Fraud Rate Trend (48h)")
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=live["timestamp"],
            y=live["fraud_rate"] * 100,
            mode="lines+markers",
            line=dict(color="#EF4444", width=2),
            marker=dict(size=4),
        )
    )
    fig2.update_layout(
        height=250,
        yaxis_title="Fraud Rate (%)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Auto-refresh via st.rerun()
    if auto_refresh:
        time.sleep(30)
        st.rerun()
