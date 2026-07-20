"""FraudTrap Dashboard — EDA & Data Quality Page"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from dashboard.components.data_loader import load_data


def render(tenant_id: str):
    st.title("🔬 EDA & Data Quality")
    st.caption("Exploratory analysis of the raw transaction dataset ingested for this tenant.")

    df, _ = load_data(tenant_id)

    # ── Dataset snapshot ──────────────────────────────────────────────────────
    st.subheader("Dataset Snapshot")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records",     f"{len(df):,}")
    c2.metric("Fraud Records",     f"{df['is_fraud'].sum():,}")
    c3.metric("Fraud Rate",        f"{df['is_fraud'].mean()*100:.2f}%")
    c4.metric("Features",          f"{len(df.columns)}")

    with st.expander("Show sample records"):
        st.dataframe(df.head(20), use_container_width=True)

    st.markdown("---")

    # ── Class distribution ────────────────────────────────────────────────────
    st.subheader("Class Distribution")
    st.info(
        "**Class imbalance** is the core challenge in fraud ML. "
        "The model training pipeline uses SMOTEENN resampling + cost-sensitive learning "
        "to compensate. PR-AUC is used as the primary metric (not ROC-AUC) "
        "because it is more informative under high imbalance.",
        icon="ℹ️",
    )
    col_l, col_r = st.columns(2)
    with col_l:
        counts = df["is_fraud"].value_counts().rename({0: "Legitimate", 1: "Fraud"})
        fig = go.Figure(go.Bar(
            x=counts.index, y=counts.values,
            marker_color=["#22C55E", "#EF4444"],
            text=counts.values, textposition="outside",
        ))
        fig.update_layout(
            title="Raw class counts", height=300,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        fig2 = go.Figure(go.Pie(
            labels=["Legitimate", "Fraud"],
            values=[counts.get("Legitimate", 0), counts.get("Fraud", 0)],
            hole=0.6,
            marker_colors=["#22C55E", "#EF4444"],
        ))
        fig2.update_layout(
            title="Class proportion", height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # ── Feature distributions ─────────────────────────────────────────────────
    st.subheader("Feature Distributions by Class")
    st.caption("Select a feature to compare its distribution between legitimate and fraud transactions.")

    numeric_features = [
        "amount", "amount_zscore", "acct_v_1h_count", "geo_speed_kmh",
        "typing_zscore", "risk_score", "latency_ms",
    ]
    selected_feat = st.selectbox("Feature", numeric_features)

    fraud_vals = df[df["is_fraud"] == 1][selected_feat].dropna().tolist()
    legit_vals = df[df["is_fraud"] == 0][selected_feat].dropna().tolist()

    if legit_vals and fraud_vals:
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=legit_vals, name="Legitimate", nbinsx=40,
            marker_color="#22C55E", opacity=0.6, histnorm="probability density",
        ))
        fig_dist.add_trace(go.Histogram(
            x=fraud_vals, name="Fraud", nbinsx=40,
            marker_color="#EF4444", opacity=0.6, histnorm="probability density",
        ))
        fig_dist.update_layout(
            barmode="overlay", height=350,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    else:
        st.info("Not enough data to compare distributions.")

    # Stats comparison
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Legitimate**")
        s = pd.Series(legit_vals).describe().round(3) if legit_vals else pd.Series(dtype=float)
        st.dataframe(s.to_frame("value"), use_container_width=True)
    with col_b:
        st.markdown("**Fraud**")
        s = pd.Series(fraud_vals).describe().round(3) if fraud_vals else pd.Series(dtype=float)
        st.dataframe(s.to_frame("value"), use_container_width=True)

    st.markdown("---")

    # ── Correlation heatmap ───────────────────────────────────────────────────
    st.subheader("Feature Correlation Matrix")
    st.caption("Strong correlations between features may indicate redundancy. Highlighted cells show |r| > 0.5.")

    corr_features = [
        "amount", "amount_zscore", "acct_v_1h_count", "geo_speed_kmh",
        "typing_zscore", "is_new_device", "impossible_travel", "is_fraud",
    ]
    corr_features = [c for c in corr_features if c in df.columns]
    if corr_features:
        corr_df = df[corr_features].corr().round(3)
        fig_heat = px.imshow(
            corr_df,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            text_auto=True,
            aspect="auto",
        )
        fig_heat.update_layout(
            height=450, paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")

    # ── Missing value analysis ────────────────────────────────────────────────
    st.subheader("Missing Value Analysis")

    missing_pcts = df.isna().mean() * 100
    optional_fields_list = [
        "merchant_id", "device_id", "ip_address_hash", "latitude", "longitude",
        "typing_cadence_ms", "session_duration_seconds", "merchant_category_code",
        "user_agent_hash", "counterparty_account_id"
    ]

    miss_data = []
    for field in optional_fields_list:
        if field in df.columns:
            pct = missing_pcts[field]
        else:
            pct = 100.0

        # Inverted: High missingness = High impact (need to improve data collection)
        impact = "High" if pct > 50 else "Medium" if pct > 20 else "Low"
        miss_data.append({
            "field": field,
            "missing_pct": pct,
            "impact": impact,
        })

    miss_df = pd.DataFrame(miss_data).sort_values("missing_pct")

    color_map = {"High": "#EF4444", "Medium": "#F59E0B", "Low": "#22C55E"}
    fig_miss = px.bar(
        miss_df, x="missing_pct", y="field", orientation="h",
        color="impact", color_discrete_map=color_map,
        labels={"missing_pct": "Missing (%)"},
        text="missing_pct",
    )
    fig_miss.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
    fig_miss.update_layout(
        height=380, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(range=[0, 110]),
    )
    st.plotly_chart(fig_miss, use_container_width=True)
    st.info(
        "Fields with **High** missingness are strong candidates for SDK instrumentation in the client integration. "
        "The model handles missing values via zero-imputation with a `has_biometrics` indicator flag.",
        icon="💡",
    )

    st.markdown("---")

    # ── Temporal distribution ─────────────────────────────────────────────────
    st.subheader("Transaction Volume Over Time")
    df_work = df.copy()
    df_work["date"] = pd.to_datetime(df_work["timestamp"]).dt.date
    daily = df_work.groupby(["date", "is_fraud"]).size().reset_index(name="count")
    daily["label"] = daily["is_fraud"].map({0: "Legitimate", 1: "Fraud"})
    fig_time = px.area(
        daily, x="date", y="count", color="label",
        color_discrete_map={"Legitimate": "#3B82F6", "Fraud": "#EF4444"},
    )
    fig_time.update_layout(
        height=300, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_time, use_container_width=True)
