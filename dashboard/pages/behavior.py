"""
RiskLens Console — Behavior Intelligence Page
Entity profiles and behavioral pattern analysis for fraud detection.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from dashboard.components import (
    kpi_row,
    line_chart,
    bar_chart,
    area_chart,
    scatter_chart,
    data_table,
    metric_table,
    page_container,
    section_divider,
    info_panel,
    metric_row,
    confidence_display,
)
from dashboard.components.data_loader import make_transactions
from dashboard.theme.colors import Colors
from dashboard.theme.icons import Icons
from dashboard.theme.typography import Typography

# ── Helpers ──────────────────────────────────────────────────────────────────


def _section_heading(icon: str, title: str, color: str = Colors.ACCENT):
    st.markdown(
        f"<div style='color:{Colors.TEXT_PRIMARY};font-weight:{Typography.WEIGHT_SEMIBOLD};"
        f"font-size:{Typography.TEXT_LG};margin-bottom:12px'>"
        f"<span style='color:{color};margin-right:8px'>&#x{icon};</span>{title}</div>",
        unsafe_allow_html=True,
    )


def _profile_card(
    title: str,
    icon: str,
    metrics: dict,
    trust_score: float,
    bg_color: str = Colors.BG_CARD,
):
    metrics_html = ""
    for label, value in metrics.items():
        metrics_html += f"""
<div style="display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid {Colors.BORDER_SUBTLE}">
    <span style="color:{Colors.TEXT_MUTED};font-size:{Typography.TEXT_SM}">{label}</span>
    <span style="color:{Colors.TEXT_PRIMARY};font-size:{Typography.TEXT_SM};font-weight:{Typography.WEIGHT_MEDIUM}">{value}</span>
</div>"""

    trust_color = (
        Colors.SUCCESS
        if trust_score >= 0.8
        else Colors.WARNING if trust_score >= 0.6 else Colors.CRITICAL
    )

    st.markdown(
        f"""
<div style="background:{bg_color};border:1px solid {Colors.BORDER_DEFAULT};border-radius:10px;padding:20px;height:100%">
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px">
        <div style="width:36px;height:36px;border-radius:8px;background:{Colors.rgba(Colors.ACCENT, 0.12)};display:flex;align-items:center;justify-content:center">
            {Icons.html(icon, 18, Colors.ACCENT)}
        </div>
        <span style="font-size:{Typography.TEXT_MD};font-weight:{Typography.WEIGHT_SEMIBOLD};color:{Colors.TEXT_PRIMARY}">{title}</span>
    </div>
    {metrics_html}
    <div style="margin-top:14px">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
            <span style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_MUTED};text-transform:uppercase;letter-spacing:{Typography.TRACKING_WIDER}">Trust / Risk</span>
            <span style="font-size:{Typography.TEXT_SM};color:{trust_color};font-weight:{Typography.WEIGHT_SEMIBOLD}">{trust_score:.1%}</span>
        </div>
        <div class="ft-progress">
            <div class="ft-progress-bar {'success' if trust_score >= 0.8 else 'warning' if trust_score >= 0.6 else 'critical'}" style="width:{trust_score * 100}%"></div>
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


# ── Main Render ──────────────────────────────────────────────────────────────


def render(tenant_id: str):
    df = make_transactions(300)
    if tenant_id != "all_tenants":
        df = df[df["tenant_id"] == tenant_id].reset_index(drop=True)

    rng = np.random.default_rng(42)
    total_txn = len(df)
    fraud_count = int(df["is_fraud"].sum())
    fraud_rate = df["is_fraud"].mean() * 100 if total_txn > 0 else 0.0
    avg_risk = float(df["risk_score"].mean()) if total_txn > 0 else 0.0

    with page_container(
        "Behavior Intelligence",
        "Entity profiles and behavioral pattern analysis",
        "ACTIVITY",
    ):

        # ── KPI Strip ─────────────────────────────────────────────────────────
        kpi_row(
            [
                {
                    "label": "Transactions",
                    "value": f"{total_txn:,}",
                    "icon": "BAR_CHART",
                },
                {
                    "label": "Fraud Detected",
                    "value": f"{fraud_count:,}",
                    "icon": "ALERT_TRIANGLE",
                    "status": "warning",
                },
                {
                    "label": "Fraud Rate",
                    "value": f"{fraud_rate:.2f}%",
                    "status": "healthy" if fraud_rate < 2.0 else "warning",
                    "icon": "TARGET",
                },
                {
                    "label": "Avg Risk",
                    "value": f"{avg_risk:.4f}",
                    "icon": "SHIELD",
                },
                {
                    "label": "Channels",
                    "value": str(df["channel"].nunique()),
                    "icon": "LAYERS",
                },
            ]
        )

        section_divider()

        # ══════════════════════════════════════════════════════════════════════
        # Section 1: Profile Overview
        # ══════════════════════════════════════════════════════════════════════
        _section_heading("1F464", "Entity Profile Overview")

        trust_customer = float(rng.uniform(0.72, 0.99))
        trust_merchant = float(rng.uniform(0.65, 0.95))
        trust_device = float(rng.uniform(0.58, 0.98))
        trust_beneficiary = float(rng.uniform(0.60, 0.92))
        trust_instrument = float(rng.uniform(0.70, 0.97))

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            _profile_card(
                "Customer Profile",
                "USER",
                {
                    "Trust Score": f"{trust_customer:.1%}",
                    "Velocity (1h)": f"{int(rng.integers(2, 45))}",
                    "Risk Trend": rng.choice(["Stable", "Rising", "Declining"]),
                    "Txn Count": f"{int(rng.integers(50, 200))}",
                },
                trust_customer,
            )

        with col2:
            _profile_card(
                "Merchant Profile",
                "BUILDING",
                {
                    "Fraud Rate": f"{rng.uniform(0.5, 12.0):.1f}%",
                    "Customer Diversity": f"{int(rng.integers(20, 300))}",
                    "Avg Amount": f"₦{rng.uniform(50, 5000):,.0f}",
                    "Chargeback Rate": f"{rng.uniform(0.1, 3.0):.1f}%",
                },
                trust_merchant,
            )

        with col3:
            _profile_card(
                "Device Profile",
                "CPU",
                {
                    "Historical Customers": f"{int(rng.integers(1, 8))}",
                    "Risk Score": f"{trust_device:.4f}",
                    "Session Count": f"{int(rng.integers(5, 60))}",
                    "Fingerprint Age": f"{int(rng.integers(1, 90))}d",
                },
                trust_device,
            )

        with col4:
            _profile_card(
                "Beneficiary Profile",
                "USERS",
                {
                    "Sender Diversity": f"{int(rng.integers(2, 25))}",
                    "Mule Risk": f"{rng.uniform(0.01, 0.45):.1%}",
                    "Avg Inflow": f"₦{rng.uniform(100, 8000):,.0f}",
                    "Account Age": f"{int(rng.integers(30, 730))}d",
                },
                trust_beneficiary,
            )

        with col5:
            _profile_card(
                "Payment Instrument",
                "CREDIT_CARD",
                {
                    "Fraud Count": f"{int(rng.integers(0, 5))}",
                    "Trust": f"{trust_instrument:.1%}",
                    "BIN Country": rng.choice(["NG", "KE", "ZA"]),
                    "Issuer Risk": rng.choice(["Low", "Medium", "High"]),
                },
                trust_instrument,
            )

        section_divider()

        # ══════════════════════════════════════════════════════════════════════
        # Section 2: Velocity Analysis
        # ══════════════════════════════════════════════════════════════════════
        _section_heading("26A1", "Velocity Analysis")

        col_velocity, col_channel = st.columns([3, 2])

        with col_velocity:
            df["date"] = df["timestamp"].dt.date
            daily_velocity = (
                df.groupby("date")
                .agg(
                    txn_count=("transaction_id", "count"),
                    unique_customers=("transaction_id", lambda x: rng.integers(10, 80)),
                    avg_amount=("amount", "mean"),
                )
                .reset_index()
            )
            daily_velocity["date"] = pd.to_datetime(daily_velocity["date"])
            daily_velocity = daily_velocity.sort_values("date")

            fig_velocity = area_chart(
                x=daily_velocity["date"],
                y=daily_velocity["txn_count"],
                title="Transaction Velocity Over Time",
                color=Colors.CHART_1,
                height=360,
            )
            fig_velocity.update_layout(
                xaxis=dict(
                    title=dict(
                        text="Date", font=dict(size=12, color=Colors.TEXT_MUTED)
                    ),
                    tickformat="%b %d",
                ),
                yaxis=dict(
                    title=dict(
                        text="Transaction Count",
                        font=dict(size=12, color=Colors.TEXT_MUTED),
                    ),
                ),
            )
            st.plotly_chart(fig_velocity, use_container_width=True)

        with col_channel:
            channel_velocity = (
                df.groupby("channel")
                .agg(
                    txn_count=("transaction_id", "count"),
                    avg_risk=("risk_score", "mean"),
                )
                .reset_index()
                .sort_values("txn_count", ascending=True)
            )

            fig_channel = bar_chart(
                x=channel_velocity["channel"],
                y=channel_velocity["txn_count"],
                title="Velocity by Channel",
                color=Colors.CHART_2,
                height=360,
            )
            fig_channel.update_layout(
                xaxis=dict(
                    title=dict(
                        text="Channel", font=dict(size=12, color=Colors.TEXT_MUTED)
                    )
                ),
                yaxis=dict(
                    title=dict(
                        text="Transaction Count",
                        font=dict(size=12, color=Colors.TEXT_MUTED),
                    )
                ),
                bargap=0.25,
            )
            st.plotly_chart(fig_channel, use_container_width=True)

        section_divider()

        # ══════════════════════════════════════════════════════════════════════
        # Section 3: Behavioral Features
        # ══════════════════════════════════════════════════════════════════════
        _section_heading("1F9E0", "Behavioral Features")

        col_importance, col_distribution = st.columns([3, 2])

        with col_importance:
            features = [
                "acct_v_1h_count",
                "amount_zscore",
                "is_new_device",
                "impossible_travel",
                "geo_speed_kmh",
                "typing_zscore",
                "acct_v_24h_total_amt",
                "device_account_count",
                "is_new_merchant",
                "cross_country_flag",
            ]
            importance = np.array(
                [0.18, 0.15, 0.12, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04]
            )

            sorted_idx = np.argsort(importance)
            features_sorted = [features[i] for i in sorted_idx]
            importance_sorted = importance[sorted_idx]

            fig_importance = go.Figure()
            fig_importance.add_trace(
                go.Bar(
                    y=features_sorted,
                    x=importance_sorted,
                    orientation="h",
                    marker=dict(
                        color=Colors.CHART_PALETTE[: len(features_sorted)],
                        cornerradius=4,
                    ),
                    text=[f"{v:.3f}" for v in importance_sorted],
                    textposition="outside",
                    textfont=dict(color=Colors.TEXT_SECONDARY, size=11),
                    hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
                )
            )
            fig_importance.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(
                    family="'Inter', sans-serif", color=Colors.TEXT_SECONDARY, size=12
                ),
                margin=dict(l=0, r=40, t=32, b=0),
                height=380,
                title=dict(
                    text="Top 10 Behavioral Features",
                    font=dict(size=14, color=Colors.TEXT_PRIMARY),
                ),
                xaxis=dict(
                    showgrid=True,
                    gridcolor=Colors.BORDER_SUBTLE,
                    gridwidth=1,
                    showline=False,
                    zeroline=False,
                    tickfont=dict(color=Colors.TEXT_MUTED, size=11),
                    title=dict(
                        text="Importance", font=dict(size=12, color=Colors.TEXT_MUTED)
                    ),
                ),
                yaxis=dict(
                    showgrid=False,
                    showline=False,
                    tickfont=dict(color=Colors.TEXT_MUTED, size=11),
                ),
                hoverlabel=dict(
                    bgcolor=Colors.BG_ELEVATED,
                    bordercolor=Colors.BORDER_DEFAULT,
                    font=dict(color=Colors.TEXT_PRIMARY, size=12),
                ),
            )
            st.plotly_chart(fig_importance, use_container_width=True)

        with col_distribution:
            fraud_amounts = df[df["is_fraud"] == 1]["amount"].clip(
                upper=df["amount"].quantile(0.99)
            )
            legit_amounts = df[df["is_fraud"] == 0]["amount"].clip(
                upper=df["amount"].quantile(0.99)
            )

            fig_dist = go.Figure()
            fig_dist.add_trace(
                go.Histogram(
                    x=legit_amounts,
                    name="Legitimate",
                    marker=dict(color=Colors.CHART_2, cornerradius=2),
                    opacity=0.7,
                    nbinsx=30,
                )
            )
            fig_dist.add_trace(
                go.Histogram(
                    x=fraud_amounts,
                    name="Fraud",
                    marker=dict(color=Colors.CHART_4, cornerradius=2),
                    opacity=0.7,
                    nbinsx=30,
                )
            )
            fig_dist.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(
                    family="'Inter', sans-serif", color=Colors.TEXT_SECONDARY, size=12
                ),
                margin=dict(l=0, r=0, t=32, b=0),
                height=380,
                barmode="overlay",
                title=dict(
                    text="Feature Distribution by Class",
                    font=dict(size=14, color=Colors.TEXT_PRIMARY),
                ),
                xaxis=dict(
                    showgrid=True,
                    gridcolor=Colors.BORDER_SUBTLE,
                    gridwidth=1,
                    showline=False,
                    zeroline=False,
                    tickfont=dict(color=Colors.TEXT_MUTED, size=11),
                    title=dict(
                        text="Transaction Amount",
                        font=dict(size=12, color=Colors.TEXT_MUTED),
                    ),
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor=Colors.BORDER_SUBTLE,
                    gridwidth=1,
                    showline=False,
                    zeroline=False,
                    tickfont=dict(color=Colors.TEXT_MUTED, size=11),
                    title=dict(
                        text="Count", font=dict(size=12, color=Colors.TEXT_MUTED)
                    ),
                ),
                legend=dict(
                    font=dict(color=Colors.TEXT_SECONDARY, size=11),
                    bgcolor="rgba(0,0,0,0)",
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                ),
                hoverlabel=dict(
                    bgcolor=Colors.BG_ELEVATED,
                    bordercolor=Colors.BORDER_DEFAULT,
                    font=dict(color=Colors.TEXT_PRIMARY, size=12),
                ),
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        section_divider()

        # ══════════════════════════════════════════════════════════════════════
        # Section 4: Entity Relationships
        # ══════════════════════════════════════════════════════════════════════
        _section_heading("1F517", "Entity Relationships")

        col_network, col_risk_amount = st.columns(2)

        with col_network:
            n_entities = 60
            entity_x = rng.uniform(0, 100, n_entities)
            entity_y = rng.uniform(0, 100, n_entities)
            entity_risk = rng.beta(2, 8, n_entities)
            entity_size = rng.integers(6, 22, n_entities)
            entity_labels = [
                f"{'CUST' if rng.random() > 0.4 else 'MERCH'}-{rng.integers(1000, 9999)}"
                for _ in range(n_entities)
            ]

            fig_network = go.Figure()
            fig_network.add_trace(
                go.Scatter(
                    x=entity_x,
                    y=entity_y,
                    mode="markers",
                    marker=dict(
                        size=entity_size.tolist(),
                        color=entity_risk.tolist(),
                        colorscale=[
                            [0, Colors.SUCCESS],
                            [0.5, Colors.WARNING],
                            [1, Colors.CRITICAL],
                        ],
                        showscale=True,
                        colorbar=dict(
                            title=dict(
                                text="Risk", font=dict(size=10, color=Colors.TEXT_MUTED)
                            ),
                            tickfont=dict(size=9, color=Colors.TEXT_MUTED),
                            thickness=12,
                            len=0.6,
                        ),
                        opacity=0.8,
                        line=dict(width=1, color=Colors.BG_CARD),
                    ),
                    text=entity_labels,
                    hovertemplate="<b>%{text}</b><br>Risk: %{marker.color:.4f}<br>Position: (%{x:.1f}, %{y:.1f})<extra></extra>",
                )
            )

            for i in range(min(20, n_entities)):
                j = rng.integers(0, n_entities)
                if i != j:
                    fig_network.add_trace(
                        go.Scatter(
                            x=[entity_x[i], entity_x[j]],
                            y=[entity_y[i], entity_y[j]],
                            mode="lines",
                            line=dict(
                                color=Colors.rgba(Colors.BORDER_DEFAULT, 0.3), width=1
                            ),
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )

            fig_network.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(
                    family="'Inter', sans-serif", color=Colors.TEXT_SECONDARY, size=12
                ),
                margin=dict(l=0, r=0, t=32, b=0),
                height=380,
                title=dict(
                    text="Customer-Merchant Relationship Network",
                    font=dict(size=14, color=Colors.TEXT_PRIMARY),
                ),
                xaxis=dict(
                    showgrid=False,
                    showline=False,
                    zeroline=False,
                    tickfont=dict(color=Colors.TEXT_MUTED, size=11),
                ),
                yaxis=dict(
                    showgrid=False,
                    showline=False,
                    zeroline=False,
                    tickfont=dict(color=Colors.TEXT_MUTED, size=11),
                ),
                hoverlabel=dict(
                    bgcolor=Colors.BG_ELEVATED,
                    bordercolor=Colors.BORDER_DEFAULT,
                    font=dict(color=Colors.TEXT_PRIMARY, size=12),
                ),
                showlegend=False,
            )
            st.plotly_chart(fig_network, use_container_width=True)

        with col_risk_amount:
            risk_x = df["risk_score"].values
            risk_y = df["amount"].values
            risk_colors = (
                df["is_fraud"].map({0: Colors.CHART_2, 1: Colors.CHART_4}).values
            )
            risk_text = [
                f"Txn: {row['transaction_id']}<br>Channel: {row['channel']}<br>Fraud: {'Yes' if row['is_fraud'] else 'No'}"
                for _, row in df.iterrows()
            ]

            fig_risk = go.Figure()
            fig_risk.add_trace(
                go.Scatter(
                    x=risk_x,
                    y=risk_y,
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=risk_colors.tolist(),
                        opacity=0.6,
                        line=dict(width=1, color=Colors.BG_CARD),
                    ),
                    text=risk_text,
                    hovertemplate="%{text}<br>Risk: %{x:.4f}<br>Amount: %{y:,.2f}<extra></extra>",
                )
            )
            fig_risk.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(
                    family="'Inter', sans-serif", color=Colors.TEXT_SECONDARY, size=12
                ),
                margin=dict(l=0, r=0, t=32, b=0),
                height=380,
                title=dict(
                    text="Risk Score vs Transaction Amount",
                    font=dict(size=14, color=Colors.TEXT_PRIMARY),
                ),
                xaxis=dict(
                    showgrid=True,
                    gridcolor=Colors.BORDER_SUBTLE,
                    gridwidth=1,
                    showline=False,
                    zeroline=False,
                    tickfont=dict(color=Colors.TEXT_MUTED, size=11),
                    title=dict(
                        text="Risk Score", font=dict(size=12, color=Colors.TEXT_MUTED)
                    ),
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor=Colors.BORDER_SUBTLE,
                    gridwidth=1,
                    showline=False,
                    zeroline=False,
                    tickfont=dict(color=Colors.TEXT_MUTED, size=11),
                    title=dict(
                        text="Transaction Amount",
                        font=dict(size=12, color=Colors.TEXT_MUTED),
                    ),
                ),
                hoverlabel=dict(
                    bgcolor=Colors.BG_ELEVATED,
                    bordercolor=Colors.BORDER_DEFAULT,
                    font=dict(color=Colors.TEXT_PRIMARY, size=12),
                ),
            )
            st.plotly_chart(fig_risk, use_container_width=True)

        section_divider()

        # ══════════════════════════════════════════════════════════════════════
        # Section 5: Profile History
        # ══════════════════════════════════════════════════════════════════════
        _section_heading("1F4C5", "Profile History")

        col_trust_history, col_velocity_history = st.columns(2)

        with col_trust_history:
            n_days = 60
            hist_dates = pd.date_range(end=pd.Timestamp.now(), periods=n_days, freq="D")
            base_trust = 0.82
            trust_trend = np.cumsum(rng.normal(0.001, 0.015, n_days))
            trust_values = np.clip(base_trust + trust_trend, 0.5, 1.0)

            fig_trust = line_chart(
                x=hist_dates,
                y=trust_values,
                title="Historical Trust Score Trend",
                color=Colors.CHART_5,
                height=340,
            )
            fig_trust.add_hline(
                y=0.7,
                line=dict(color=Colors.WARNING, width=1, dash="dash"),
                annotation_text="Warning Threshold",
                annotation=dict(font=dict(size=10, color=Colors.WARNING)),
            )
            fig_trust.add_hline(
                y=0.9,
                line=dict(color=Colors.SUCCESS, width=1, dash="dash"),
                annotation_text="High Trust",
                annotation=dict(font=dict(size=10, color=Colors.SUCCESS)),
            )
            fig_trust.update_layout(
                xaxis=dict(
                    title=dict(
                        text="Date", font=dict(size=12, color=Colors.TEXT_MUTED)
                    ),
                    tickformat="%b %d",
                ),
                yaxis=dict(
                    title=dict(
                        text="Trust Score", font=dict(size=12, color=Colors.TEXT_MUTED)
                    ),
                    range=[0.45, 1.05],
                ),
            )
            st.plotly_chart(fig_trust, use_container_width=True)

        with col_velocity_history:
            n_hours = 72
            hist_hours = pd.date_range(
                end=pd.Timestamp.now(), periods=n_hours, freq="h"
            )
            base_velocity = 120
            velocity_wave = 30 * np.sin(np.linspace(0, 4 * np.pi, n_hours))
            velocity_noise = rng.poisson(15, n_hours)
            velocity_values = np.clip(
                base_velocity + velocity_wave + velocity_noise, 0, None
            )

            fig_velocity_hist = line_chart(
                x=hist_hours,
                y=velocity_values,
                title="Historical Velocity Trend",
                color=Colors.CHART_7,
                height=340,
            )
            fig_velocity_hist.update_layout(
                xaxis=dict(
                    title=dict(
                        text="Hour", font=dict(size=12, color=Colors.TEXT_MUTED)
                    ),
                    tickformat="%b %d %H:%M",
                    dtick=12,
                ),
                yaxis=dict(
                    title=dict(
                        text="Transactions / Hour",
                        font=dict(size=12, color=Colors.TEXT_MUTED),
                    ),
                ),
            )
            st.plotly_chart(fig_velocity_hist, use_container_width=True)

        section_divider()

        # ══════════════════════════════════════════════════════════════════════
        # Summary Metrics
        # ══════════════════════════════════════════════════════════════════════
        _section_heading("1F4CA", "Behavior Summary")

        metric_row(
            [
                {
                    "label": "Avg Trust Score",
                    "value": f"{np.mean([trust_customer, trust_merchant, trust_device, trust_beneficiary, trust_instrument]):.1%}",
                    "color": Colors.ACCENT,
                    "icon": "STAR",
                },
                {
                    "label": "High Risk Entities",
                    "value": f"{int(rng.integers(3, 18))}",
                    "color": Colors.CRITICAL,
                    "icon": "ALERT_TRIANGLE",
                },
                {
                    "label": "Velocity Peak",
                    "value": f"{int(velocity_values.max())}/hr",
                    "color": Colors.WARNING,
                    "icon": "TRENDING_UP",
                },
                {
                    "label": "Feature Coverage",
                    "value": f"{len(features)} features",
                    "color": Colors.SUCCESS,
                    "icon": "LAYERS",
                },
            ]
        )

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        summary_df = pd.DataFrame(
            [
                {
                    "Entity": "Customer",
                    "Trust Score": f"{trust_customer:.1%}",
                    "Risk Level": (
                        "Low"
                        if trust_customer > 0.8
                        else "Medium" if trust_customer > 0.6 else "High"
                    ),
                    "Status": "Active",
                },
                {
                    "Entity": "Merchant",
                    "Trust Score": f"{trust_merchant:.1%}",
                    "Risk Level": (
                        "Low"
                        if trust_merchant > 0.8
                        else "Medium" if trust_merchant > 0.6 else "High"
                    ),
                    "Status": "Active",
                },
                {
                    "Entity": "Device",
                    "Trust Score": f"{trust_device:.1%}",
                    "Risk Level": (
                        "Low"
                        if trust_device > 0.8
                        else "Medium" if trust_device > 0.6 else "High"
                    ),
                    "Status": "Active",
                },
                {
                    "Entity": "Beneficiary",
                    "Trust Score": f"{trust_beneficiary:.1%}",
                    "Risk Level": (
                        "Low"
                        if trust_beneficiary > 0.8
                        else "Medium" if trust_beneficiary > 0.6 else "High"
                    ),
                    "Status": "Active",
                },
                {
                    "Entity": "Payment Instrument",
                    "Trust Score": f"{trust_instrument:.1%}",
                    "Risk Level": (
                        "Low"
                        if trust_instrument > 0.8
                        else "Medium" if trust_instrument > 0.6 else "High"
                    ),
                    "Status": "Active",
                },
            ]
        )

        data_table(
            df=summary_df,
            columns={
                "Entity": "Entity Type",
                "Trust Score": "Trust Score",
                "Risk Level": "Risk Level",
                "Status": "Status",
            },
            max_rows=10,
            status_col="Risk Level",
            sortable=True,
            striped=True,
        )
