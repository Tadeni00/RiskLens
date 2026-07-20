"""FraudTrap Dashboard — Overview Page"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from dashboard.components.data_loader import load_data, make_live_timeseries, compute_kpis


def render(tenant_id: str):
    st.title("📊 Overview")
    st.caption(f"Tenant: **{tenant_id}** · Refreshed: just now")

    df, is_live = load_data(tenant_id)
    live, _ = make_live_timeseries(24, tenant_id)
    kpis = compute_kpis(df)

    # ── KPI row ───────────────────────────────────────────────────────────────
    total = kpis["total"]
    n_block  = kpis["n_block"]
    n_review = kpis["n_review"]
    fraud_rate = kpis["fraud_rate"]
    avg_lat  = kpis["avg_lat"]

    # Compute deltas from actual data
    fraud_delta = f"{fraud_rate - 1.3:.2f}%" if total > 0 else "—"
    avg_lat_delta = f"{avg_lat - 78:+.0f}ms vs target" if total > 0 else "—"

    c1, c2, c3, c4, c5 = st.columns(5)
    block_pct = f"{n_block/total*100:.1f}% of volume" if total > 0 else "—"
    c1.metric("Transactions (90d)", f"{total:,}")
    c2.metric("Fraud Rate",         f"{fraud_rate:.2f}%", delta=fraud_delta)
    c3.metric("Blocked",            f"{n_block:,}", delta=block_pct)
    c4.metric("In Review",          f"{n_review:,}")
    c5.metric("Avg Latency",        f"{avg_lat:.0f}ms", delta=avg_lat_delta, delta_color="inverse")

    st.markdown("---")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Transactions & Fraud Rate (24h)")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=live["timestamp"], y=live["txn_per_min"],
            name="Txn/min", marker_color="#3B82F6", opacity=0.7,
            yaxis="y",
        ))
        fig.add_trace(go.Scatter(
            x=live["timestamp"], y=live["fraud_rate"] * 100,
            name="Fraud Rate %", line=dict(color="#EF4444", width=2),
            yaxis="y2",
        ))
        fig.update_layout(
            yaxis=dict(title="Transactions/min"),
            yaxis2=dict(title="Fraud Rate %", overlaying="y", side="right"),
            legend=dict(orientation="h", y=1.1),
            height=300, margin=dict(l=0, r=0, t=30, b=0),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Decision Distribution")
        counts = df["decision"].value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=counts.index, values=counts.values,
            hole=0.55,
            marker=dict(colors=["#22C55E", "#F59E0B", "#EF4444"]),
        ))
        fig_pie.update_layout(
            height=300, margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            legend=dict(orientation="h", y=-0.1),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── Model phase status ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Model Lifecycle Status")
    phase_colors = {
        "UNSUPERVISED":    "#6366F1",
        "SEMI_SUPERVISED": "#F59E0B",
        "SUPERVISED":      "#22C55E",
    }
    phase_counts = df["model_phase"].value_counts()
    for phase, count in phase_counts.items():
        color = phase_colors.get(phase, "#888")
        pct   = count / len(df) * 100
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:6px">'
            f'<span style="background:{color};color:white;padding:2px 10px;'
            f'border-radius:99px;font-size:12px;font-weight:600">{phase}</span>'
            f'<span style="color:#888;font-size:13px">{count:,} transactions ({pct:.1f}%)</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Fraud by channel ──────────────────────────────────────────────────────
    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Fraud Rate by Channel")
        ch = df.groupby("channel")["is_fraud"].mean().reset_index()
        ch.columns = ["channel", "fraud_rate"]
        ch["fraud_rate_pct"] = ch["fraud_rate"] * 100
        fig_ch = px.bar(
            ch.sort_values("fraud_rate_pct", ascending=True),
            x="fraud_rate_pct", y="channel", orientation="h",
            color="fraud_rate_pct",
            color_continuous_scale=["#22C55E", "#F59E0B", "#EF4444"],
            labels={"fraud_rate_pct": "Fraud Rate (%)"},
        )
        fig_ch.update_layout(
            height=250, margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_ch, use_container_width=True)

    with col_b:
        st.subheader("Fraud Rate by Country")
        co = df.groupby("country_code")["is_fraud"].mean().reset_index()
        co.columns = ["country", "fraud_rate"]
        co["fraud_rate_pct"] = co["fraud_rate"] * 100
        fig_co = px.bar(
            co.sort_values("fraud_rate_pct", ascending=False),
            x="country", y="fraud_rate_pct",
            color="fraud_rate_pct",
            color_continuous_scale=["#22C55E", "#F59E0B", "#EF4444"],
            labels={"fraud_rate_pct": "Fraud Rate (%)"},
        )
        fig_co.update_layout(
            height=250, margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_co, use_container_width=True)
