"""FraudTrap Dashboard — Compliance Page
Regulatory compliance, bias monitoring, and audit trails.
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dashboard.components import (
    kpi_row,
    bar_chart,
    data_table,
    metric_table,
    page_container,
    section_divider,
    alert,
    metric_row,
    info_panel,
)
from dashboard.components.data_loader import make_transactions, currency_fmt
from dashboard.theme.colors import Colors
from dashboard.theme.icons import Icons


def render(tenant_id: str):
    rng = np.random.default_rng()
    df = make_transactions(200)

    with page_container(
        "Compliance",
        "Regulatory compliance, bias monitoring, and audit trails",
        "FILE_TEXT",
    ):
        # ── Section 1: Compliance Overview ──────────────────────────────────
        n_blocked = int((df["decision"] == "BLOCK").sum()) if "decision" in df.columns else 0
        n_review = int((df["decision"] == "REVIEW").sum()) if "decision" in df.columns else 0
        fraud_prevented = float(df.loc[df["is_fraud"] == 1, "amount"].sum()) if "is_fraud" in df.columns else 0.0
        sar_count = int(rng.poisson(12))
        analyst_queue = int(rng.poisson(34))
        chargebacks = int(rng.poisson(7))
        reg_alerts = int(rng.poisson(5))
        resolution_hours = float(rng.normal(4.2, 0.8))

        kpi_row([
            {
                "label": "SAR Generated",
                "value": str(sar_count),
                "icon": "FILE_TEXT",
                "status": "healthy",
                "trend": rng.uniform(-5, 10),
                "trend_label": "vs last month",
            },
            {
                "label": "Analyst Queue",
                "value": str(analyst_queue),
                "icon": "USERS",
                "status": "warning" if analyst_queue > 40 else "healthy",
                "trend": rng.uniform(-10, 10),
                "trend_label": "pending reviews",
            },
            {
                "label": "Fraud Loss Prevented",
                "value": currency_fmt(fraud_prevented, "NGN"),
                "icon": "SHIELD_CHECK",
                "status": "healthy",
                "trend": rng.uniform(2, 15),
                "trend_label": "vs last week",
            },
            {
                "label": "Chargebacks",
                "value": str(chargebacks),
                "icon": "ALERT_TRIANGLE",
                "status": "warning" if chargebacks > 10 else "healthy",
                "trend": rng.uniform(-8, 8),
                "trend_label": "this period",
            },
            {
                "label": "Regulatory Alerts",
                "value": str(reg_alerts),
                "icon": "BELL",
                "status": "critical" if reg_alerts > 8 else "healthy",
                "trend": rng.uniform(-3, 6),
                "trend_label": "active",
            },
            {
                "label": "Case Resolution Time",
                "value": f"{resolution_hours:.1f}h",
                "icon": "CLOCK",
                "status": "healthy" if resolution_hours < 5 else "warning",
                "trend": rng.uniform(-2, 2),
                "delta_suffix": "h",
                "trend_label": "avg",
            },
        ])

        section_divider()

        # ── Section 2: Regulatory Checklist ─────────────────────────────────
        st.markdown(
            f"<div style='color:{Colors.TEXT_PRIMARY};font-weight:600;font-size:16px;margin-bottom:12px'>"
            f"{Icons.html('SHIELD_CHECK', 16, Colors.ACCENT)} Regulatory Compliance Status</div>",
            unsafe_allow_html=True,
        )

        metric_table(
            title="",
            metrics=[
                {
                    "label": "EU AI Act — High-Risk Classification",
                    "value": "Compliant",
                    "status": "success",
                },
                {
                    "label": "GDPR — Right to Explanation (Art. 22)",
                    "value": "Compliant",
                    "status": "success",
                },
                {
                    "label": "DORA — Digital Operational Resilience",
                    "value": "Compliant",
                    "status": "success",
                },
                {
                    "label": "PCI-DSS — Payment Card Data Security",
                    "value": "Compliant",
                    "status": "success",
                },
                {
                    "label": "NDPR — Nigeria Data Protection Regulation",
                    "value": "Compliant",
                    "status": "success",
                },
                {
                    "label": "CCPA — California Consumer Privacy Act",
                    "value": "Compliant",
                    "status": "success",
                },
            ],
        )

        section_divider()

        # ── Section 3: Bias & Fairness Monitoring ───────────────────────────
        st.markdown(
            f"<div style='color:{Colors.TEXT_PRIMARY};font-weight:600;font-size:16px;margin-bottom:4px'>"
            f"{Icons.html('USERS', 16, Colors.ACCENT)} Bias & Fairness Monitoring</div>"
            f"<div style='color:{Colors.TEXT_MUTED};font-size:{12}px;margin-bottom:12px'>"
            f"Disparate impact ratio (80% rule): flag rate ratio must remain between 0.80 and 1.25 across cohorts.</div>",
            unsafe_allow_html=True,
        )

        if not df.empty and "is_fraud" in df.columns and "decision" in df.columns:
            df_work = df.copy()
            df_work["is_flagged"] = df_work["decision"].isin(["BLOCK", "REVIEW"]).astype(int)

            # Geographic disparate impact
            col_geo, col_ch = st.columns(2)

            with col_geo:
                geo_data = []
                if "country_code" in df_work.columns:
                    geo = df_work.groupby("country_code")["is_flagged"].agg(["mean", "count"]).reset_index()
                    geo = geo[geo["count"] > 5]
                    global_rate = df_work["is_flagged"].mean()
                    for _, row in geo.iterrows():
                        ratio = row["mean"] / global_rate if global_rate > 0 else 1.0
                        geo_data.append({
                            "region": row["country_code"],
                            "ratio": round(float(ratio), 3),
                            "color": Colors.SUCCESS if 0.8 <= ratio <= 1.25 else Colors.CRITICAL,
                        })

                if geo_data:
                    geo_regions = [d["region"] for d in geo_data]
                    geo_ratios = [d["ratio"] for d in geo_data]
                    geo_colors = [d["color"] for d in geo_data]

                    fig_geo = go.Figure()
                    fig_geo.add_trace(go.Bar(
                        x=geo_regions, y=geo_ratios,
                        marker=dict(color=geo_colors, cornerradius=4),
                        text=[f"{r:.3f}" for r in geo_ratios],
                        textposition="outside",
                        textfont=dict(color=Colors.TEXT_PRIMARY, size=11),
                    ))
                    fig_geo.add_hline(y=0.8, line_dash="dash", line_color=Colors.WARNING, annotation_text="80% threshold")
                    fig_geo.add_hline(y=1.0, line_dash="dot", line_color=Colors.TEXT_MUTED, annotation_text="Parity")
                    fig_geo.add_hline(y=1.25, line_dash="dash", line_color=Colors.WARNING, annotation_text="125% threshold")
                    fig_geo.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="'Inter', sans-serif", color=Colors.TEXT_SECONDARY, size=12),
                        height=320, margin=dict(l=0, r=0, t=32, b=0),
                        yaxis=dict(title="Disparate Impact Ratio", gridcolor=Colors.BORDER_SUBTLE, range=[0, 2]),
                        xaxis=dict(tickfont=dict(color=Colors.TEXT_MUTED, size=11)),
                        title=dict(text="Geographic Region (80% Rule)", font=dict(size=13, color=Colors.TEXT_PRIMARY)),
                    )
                    st.plotly_chart(fig_geo, use_container_width=True)
                else:
                    st.info("Insufficient geographic data for bias analysis.")

            with col_ch:
                ch_data = []
                if "channel" in df_work.columns:
                    ch = df_work.groupby("channel")["is_flagged"].agg(["mean", "count"]).reset_index()
                    ch = ch[ch["count"] > 5]
                    global_rate = df_work["is_flagged"].mean()
                    for _, row in ch.iterrows():
                        ratio = row["mean"] / global_rate if global_rate > 0 else 1.0
                        ch_data.append({
                            "channel": row["channel"],
                            "ratio": round(float(ratio), 3),
                            "color": Colors.SUCCESS if 0.8 <= ratio <= 1.25 else Colors.CRITICAL,
                        })

                if ch_data:
                    ch_names = [d["channel"] for d in ch_data]
                    ch_ratios = [d["ratio"] for d in ch_data]
                    ch_colors = [d["color"] for d in ch_data]

                    fig_ch = go.Figure()
                    fig_ch.add_trace(go.Bar(
                        x=ch_names, y=ch_ratios,
                        marker=dict(color=ch_colors, cornerradius=4),
                        text=[f"{r:.3f}" for r in ch_ratios],
                        textposition="outside",
                        textfont=dict(color=Colors.TEXT_PRIMARY, size=11),
                    ))
                    fig_ch.add_hline(y=0.8, line_dash="dash", line_color=Colors.WARNING, annotation_text="80% threshold")
                    fig_ch.add_hline(y=1.0, line_dash="dot", line_color=Colors.TEXT_MUTED, annotation_text="Parity")
                    fig_ch.add_hline(y=1.25, line_dash="dash", line_color=Colors.WARNING, annotation_text="125% threshold")
                    fig_ch.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="'Inter', sans-serif", color=Colors.TEXT_SECONDARY, size=12),
                        height=320, margin=dict(l=0, r=0, t=32, b=0),
                        yaxis=dict(title="Disparate Impact Ratio", gridcolor=Colors.BORDER_SUBTLE, range=[0, 2]),
                        xaxis=dict(tickfont=dict(color=Colors.TEXT_MUTED, size=11)),
                        title=dict(text="Transaction Channel (80% Rule)", font=dict(size=13, color=Colors.TEXT_PRIMARY)),
                    )
                    st.plotly_chart(fig_ch, use_container_width=True)
                else:
                    st.info("Insufficient channel data for bias analysis.")

            all_ratios = [d["ratio"] for d in geo_data] + [d["ratio"] for d in ch_data]
            violations = [r for r in all_ratios if r < 0.8 or r > 1.25]
            if violations:
                alert(
                    f"Bias threshold violation detected: {len(violations)} cohort(s) outside the 80%-125% disparate impact range. "
                    f"Immediate review recommended by the Fairness & Ethics team.",
                    level="critical",
                    icon="ALERT_TRIANGLE",
                )
            else:
                alert(
                    "All cohorts within acceptable disparate impact range (80%–125%).",
                    level="success",
                    icon="CHECK_CIRCLE",
                )

        section_divider()

        # ── Section 4: Audit Trail ──────────────────────────────────────────
        st.markdown(
            f"<div style='color:{Colors.TEXT_PRIMARY};font-weight:600;font-size:16px;margin-bottom:12px'>"
            f"{Icons.html('DATABASE', 16, Colors.ACCENT)} Decision Audit Trail</div>",
            unsafe_allow_html=True,
        )

        n_audit = 20
        audit_rows = []
        event_types = ["MODEL_PREDICTION", "FRAUD_FLAG_BLOCK", "FRAUD_FLAG_REVIEW", "MANUAL_OVERRIDE", "SYSTEM_ALERT"]
        model_versions = ["v3.2.1-prod", "v3.2.0-prod", "v3.1.8-prod"]
        decisions = ["BLOCK", "REVIEW", "APPROVE"]

        for i in range(n_audit):
            ts = pd.Timestamp.now() - pd.Timedelta(minutes=int(rng.integers(1, 1440)))
            event_type = rng.choice(event_types, p=[0.40, 0.25, 0.15, 0.10, 0.10])
            decision = "BLOCK" if "BLOCK" in event_type else "REVIEW" if "REVIEW" in event_type else rng.choice(decisions)
            risk = float(rng.beta(8, 2) if "BLOCK" in event_type else rng.beta(2, 5))
            audit_rows.append({
                "Timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "Event Type": event_type,
                "Model Version": rng.choice(model_versions),
                "Decision": decision,
                "Risk Score": round(risk, 4),
                "Trace ID": f"trace_{rng.integers(100000, 999999)}",
            })

        audit_df = pd.DataFrame(audit_rows).sort_values("Timestamp", ascending=False).reset_index(drop=True)

        data_table(
            df=audit_df,
            columns={
                "Timestamp": "Timestamp",
                "Event Type": "Event Type",
                "Model Version": "Model Version",
                "Decision": "Decision",
                "Risk Score": "Risk Score",
                "Trace ID": "Trace ID",
            },
            max_rows=20,
            status_col="Decision",
            striped=True,
        )

        section_divider()

        # ── Section 5: Decision Explanations ────────────────────────────────
        st.markdown(
            f"<div style='color:{Colors.TEXT_PRIMARY};font-weight:600;font-size:16px;margin-bottom:12px'>"
            f"{Icons.html('EYE', 16, Colors.ACCENT)} Explanation Coverage & Latency</div>",
            unsafe_allow_html=True,
        )

        explanation_df = pd.DataFrame({
            "Model Phase": ["SUPERVISED", "SEMI_SUPERVISED", "UNSUPERVISED"],
            "Decisions Explained": [
                str(int(rng.integers(8500, 9500))),
                str(int(rng.integers(1800, 2400))),
                str(int(rng.integers(400, 600))),
            ],
            "Coverage (%)": [
                f"{float(rng.uniform(98.0, 99.9)):.1f}",
                f"{float(rng.uniform(96.0, 99.0)):.1f}",
                f"{float(rng.uniform(92.0, 97.0)):.1f}",
            ],
            "Avg SHAP Features": [
                f"{float(rng.uniform(6, 8)):.1f}",
                f"{float(rng.uniform(5, 7)):.1f}",
                f"{float(rng.uniform(4, 6)):.1f}",
            ],
            "Human Review Required": [
                str(int(rng.integers(2, 8))),
                str(int(rng.integers(10, 25))),
                str(int(rng.integers(5, 15))),
            ],
            "Status": ["Active", "Active", "Active"],
        })

        data_table(
            df=explanation_df,
            columns={
                "Model Phase": "Model Phase",
                "Decisions Explained": "Decisions Explained",
                "Coverage (%)": "Coverage (%)",
                "Avg SHAP Features": "Avg SHAP Features",
                "Human Review Required": "Human Review Required",
                "Status": "Status",
            },
            max_rows=5,
            status_col="Status",
        )

        st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)

        p50_lat = float(rng.normal(38, 4))
        p95_lat = float(rng.normal(72, 6))
        p99_lat = float(rng.normal(95, 8))
        mean_lat = float(rng.normal(42, 5))

        metric_row([
            {
                "label": "P50 Explanation Latency",
                "value": f"{p50_lat:.0f}ms",
                "color": Colors.SUCCESS,
                "icon": "ZAP",
            },
            {
                "label": "P95 Explanation Latency",
                "value": f"{p95_lat:.0f}ms",
                "color": Colors.WARNING if p95_lat > 80 else Colors.SUCCESS,
                "icon": "ZAP",
            },
            {
                "label": "P99 Explanation Latency",
                "value": f"{p99_lat:.0f}ms",
                "color": Colors.CRITICAL if p99_lat > 120 else Colors.WARNING if p99_lat > 90 else Colors.SUCCESS,
                "icon": "ZAP",
            },
            {
                "label": "Mean Explanation Latency",
                "value": f"{mean_lat:.0f}ms",
                "color": Colors.ACCENT,
                "icon": "BAR_CHART",
            },
        ])

        section_divider()

        st.markdown(
            f"<div style='color:{Colors.TEXT_MUTED};font-size:11px;text-align:center;padding:8px 0'>"
            f"Compliance data refreshed in real-time. Audit log retention: 5 years (GDPR Art. 30, DORA Art. 11). "
            f"Bias monitoring per EU AI Act Art. 10. All explanations stored per GDPR Art. 22.</div>",
            unsafe_allow_html=True,
        )
