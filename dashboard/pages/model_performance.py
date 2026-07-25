"""RiskLens Console — Model Performance Page (Live-Wired)"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from dashboard.components.data_loader import load_data


def render(tenant_id: str):
    st.title("🤖 Model Performance")
    st.caption(
        "Live metrics evaluated on recent scored transactions from the streaming API."
    )

    df, _ = load_data(tenant_id)
    if df.empty or "risk_score" not in df.columns or "is_fraud" not in df.columns:
        st.warning("Insufficient live data to compute model performance.")
        return

    y_true = df["is_fraud"].values
    y_score = df["risk_score"].values

    total_pos = sum(y_true)
    if total_pos == 0:
        st.warning(
            "No fraud labels in the recent data stream. Waiting for more data to compute curves."
        )
        return

    # Compute PR curve dynamically
    from sklearn.metrics import precision_recall_curve, auc, confusion_matrix
    from sklearn.calibration import calibration_curve

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recalls, precisions)

    # Confusion matrix based on default 0.5 threshold (we'll make it dynamic below)
    y_pred = (y_score >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    precision_val = tp / (tp + fp + 1e-8)
    recall_val = tp / (tp + fn + 1e-8)
    f1_val = 2 * precision_val * recall_val / (precision_val + recall_val + 1e-8)

    # ── Current model info ────────────────────────────────────────────────────
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Live PR-AUC", f"{pr_auc:.3f}")
    col2.metric(
        "F2-Score",
        f"{(5*precision_val*recall_val)/(4*precision_val+recall_val+1e-8):.3f}",
    )
    col3.metric("Precision", f"{precision_val:.3f}")
    col4.metric("Recall", f"{recall_val:.3f}")
    col5.metric("F1-Score", f"{f1_val:.3f}")
    col6.metric("Transactions", f"{len(df):,}")

    st.markdown("---")

    col_l, col_r = st.columns(2)

    # ── PR Curve ─────────────────────────────────────────────────────────────
    with col_l:
        st.subheader("Precision–Recall Curve")
        fraud_rate = df["is_fraud"].mean()
        fig_pr = go.Figure()
        fig_pr.add_trace(
            go.Scatter(
                x=recalls,
                y=precisions,
                mode="lines",
                name=f"Live (PR-AUC={pr_auc:.3f})",
                line=dict(color="#3B82F6", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(59,130,246,0.1)",
            )
        )
        fig_pr.add_hline(
            y=fraud_rate,
            line_dash="dash",
            line_color="#888",
            annotation_text=f"Random baseline ({fraud_rate*100:.1f}%)",
            annotation_position="bottom right",
        )
        fig_pr.update_layout(
            xaxis_title="Recall",
            yaxis_title="Precision",
            height=340,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis=dict(range=[0, 1.05]),
            yaxis=dict(range=[0, 1.05]),
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    with col_r:
        st.subheader("Confusion Matrix")
        thresh = st.slider("Decision Threshold", 0.0, 1.0, 0.50, 0.01)

        y_pred_dyn = (y_score >= thresh).astype(int)
        tn_d, fp_d, fn_d, tp_d = confusion_matrix(y_true, y_pred_dyn).ravel()

        z = [[tn_d, fp_d], [fn_d, tp_d]]
        text = [
            [f"TN\n{tn_d:,}", f"FP\n{fp_d:,}"],
            [f"FN\n{fn_d:,}", f"TP\n{tp_d:,}"],
        ]
        fig_cm = go.Figure(
            go.Heatmap(
                z=z,
                x=["Predicted Legit", "Predicted Fraud"],
                y=["Actual Legit", "Actual Fraud"],
                text=text,
                texttemplate="%{text}",
                colorscale=[[0, "#0F2A1A"], [0.5, "#1A4A2E"], [1, "#22C55E"]],
                showscale=False,
            )
        )
        fig_cm.update_layout(
            height=340,
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("---")

    # ── Score distribution ────────────────────────────────────────────────────
    st.subheader("Risk Score Distribution")

    fraud_scores = df[df["is_fraud"] == 1]["risk_score"]
    legit_scores = df[df["is_fraud"] == 0]["risk_score"]

    fig_dist = go.Figure()
    fig_dist.add_trace(
        go.Histogram(
            x=legit_scores,
            name="Legitimate",
            nbinsx=50,
            marker_color="#22C55E",
            opacity=0.6,
            histnorm="probability density",
        )
    )
    fig_dist.add_trace(
        go.Histogram(
            x=fraud_scores,
            name="Fraud",
            nbinsx=50,
            marker_color="#EF4444",
            opacity=0.7,
            histnorm="probability density",
        )
    )
    fig_dist.update_layout(
        barmode="overlay",
        xaxis_title="Risk Score",
        yaxis_title="Density",
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("---")

    # ── Calibration curve ────────────────────────────────────────────────────
    st.subheader("Calibration Curve")
    st.caption("Computed from live scoring stream.")

    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=10)

    fig_cal = go.Figure()
    fig_cal.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect calibration",
            line=dict(color="#888", dash="dash"),
        )
    )
    fig_cal.add_trace(
        go.Scatter(
            x=prob_pred,
            y=prob_true,
            mode="lines+markers",
            name="Live Model",
            line=dict(color="#3B82F6", width=2.5),
            marker=dict(size=7),
        )
    )
    fig_cal.update_layout(
        xaxis_title="Mean Predicted Score",
        yaxis_title="Fraction of Positives",
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(range=[-0.05, 1.05]),
        yaxis=dict(range=[-0.05, 1.05]),
    )
    st.plotly_chart(fig_cal, use_container_width=True)

    st.markdown("---")

    # ── Proxy Feature Importance ────────────────────────────────────────────────────
    st.subheader("Proxy Feature Importance (Correlation)")
    st.caption(
        "Absolute correlation of features with fraud labels in the recent stream."
    )

    numeric_cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c not in ["is_fraud", "risk_score", "timestamp"]
    ]

    corrs = []
    for c in numeric_cols:
        try:
            corr = df[c].corr(df["is_fraud"])
            if not np.isnan(corr):
                corrs.append({"feature": c, "importance": abs(corr)})
        except (ValueError, TypeError):
            pass

    fi_sorted = pd.DataFrame(corrs).sort_values("importance", ascending=True).tail(15)

    colors = [
        "#EF4444" if imp > 0.3 else "#F59E0B" if imp > 0.15 else "#3B82F6"
        for imp in fi_sorted["importance"]
    ]
    fig_fi = go.Figure(
        go.Bar(
            x=fi_sorted["importance"],
            y=fi_sorted["feature"],
            orientation="h",
            marker_color=colors,
            text=fi_sorted["importance"].round(3),
            textposition="outside",
        )
    )
    fig_fi.update_layout(
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Absolute Correlation with is_fraud",
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_fi, use_container_width=True)
