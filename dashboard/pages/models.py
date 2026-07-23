"""FraudTrap Dashboard — Model Architecture Page"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from dashboard.components import (
    kpi_row,
    bar_chart,
    gauge_chart,
    metric_table,
    data_table,
    page_container,
    section_divider,
    architecture_diagram,
    model_performance_summary,
    metric_row,
    progress_ring,
)
from dashboard.components.data_loader import (
    make_pr_curve,
    make_confusion_matrix,
    make_feature_importance,
)
from dashboard.theme.colors import Colors
from dashboard.theme.icons import Icons

# ── Synthetic Data ─────────────────────────────────────────────────────────────

CHAMPION_METRICS = {
    "PR-AUC": 0.9347,
    "ROC-AUC": 0.9891,
    "F2-Score": 0.8712,
    "F1-Score": 0.8543,
    "Precision": 0.8291,
    "Recall": 0.8806,
    "Latency (ms)": 12.4,
}

MODEL_LEADERBOARD = pd.DataFrame(
    [
        {
            "Rank": 1,
            "Model": "CatBoost",
            "Version": "v3.2.1",
            "PR-AUC": 0.9347,
            "ROC-AUC": 0.9891,
            "F2": 0.8712,
            "Latency (ms)": 12.4,
            "Status": "Champion",
        },
        {
            "Rank": 2,
            "Model": "FT-Transformer",
            "Version": "v1.0.0",
            "PR-AUC": 0.9218,
            "ROC-AUC": 0.9845,
            "F2": 0.8534,
            "Latency (ms)": 48.7,
            "Status": "Specialist",
        },
        {
            "Rank": 3,
            "Model": "LightGBM",
            "Version": "v4.1.0",
            "PR-AUC": 0.9183,
            "ROC-AUC": 0.9812,
            "F2": 0.8401,
            "Latency (ms)": 8.2,
            "Status": "Offline",
        },
        {
            "Rank": 4,
            "Model": "XGBoost",
            "Version": "v2.0.3",
            "PR-AUC": 0.9097,
            "ROC-AUC": 0.9788,
            "F2": 0.8325,
            "Latency (ms)": 9.6,
            "Status": "Offline",
        },
    ]
)


def _generate_training_history():
    rng = np.random.default_rng(42)
    iterations = np.arange(1, 501)
    pr_auc = (
        0.5
        + 0.44 * (1 - np.exp(-iterations / 80))
        + rng.normal(0, 0.008, 500).cumsum() * 0.01
    )
    pr_auc = np.clip(pr_auc, 0.5, 0.94)
    pr_auc[-1] = 0.9347

    loss = (
        0.7 * np.exp(-iterations / 60)
        + 0.15
        + rng.normal(0, 0.005, 500).cumsum() * 0.005
    )
    loss = np.clip(loss, 0.12, 0.85)
    loss[-1] = 0.1482
    return iterations, pr_auc, loss


# ── Main Render ────────────────────────────────────────────────────────────────


def render(tenant_id: str):
    pr_curve = make_pr_curve()
    cm = make_confusion_matrix()
    fi_df = make_feature_importance()
    iterations, train_pr_auc, train_loss = _generate_training_history()

    with page_container(
        "Model Architecture", "Production ML models and performance analysis", "BRAIN"
    ):

        # ── KPI Strip ─────────────────────────────────────────────────────────
        kpi_row(
            [
                {
                    "label": "PR-AUC",
                    "value": f"{CHAMPION_METRICS['PR-AUC']:.4f}",
                    "icon": "TARGET",
                    "status": "healthy",
                },
                {
                    "label": "ROC-AUC",
                    "value": f"{CHAMPION_METRICS['ROC-AUC']:.4f}",
                    "icon": "ACTIVITY",
                    "status": "healthy",
                },
                {
                    "label": "F2-Score",
                    "value": f"{CHAMPION_METRICS['F2-Score']:.4f}",
                    "icon": "TRENDING_UP",
                    "status": "healthy",
                },
                {
                    "label": "Latency",
                    "value": f"{CHAMPION_METRICS['Latency (ms)']:.1f}ms",
                    "icon": "TIMER",
                    "status": "healthy",
                },
                {
                    "label": "Precision",
                    "value": f"{CHAMPION_METRICS['Precision']:.4f}",
                    "icon": "CROSSHAIR",
                },
                {
                    "label": "Recall",
                    "value": f"{CHAMPION_METRICS['Recall']:.4f}",
                    "icon": "SEARCH",
                },
            ]
        )

        section_divider()

        # ══════════════════════════════════════════════════════════════════════
        # Section 1: Architecture Diagram
        # ══════════════════════════════════════════════════════════════════════
        st.markdown(
            f"<div style='color:{Colors.TEXT_PRIMARY};font-weight:600;"
            f"font-size:1.1rem;margin-bottom:16px'>"
            f"{Icons.html('LAYERS', 18, Colors.ACCENT)} "
            f"Production ML Pipeline Architecture</div>",
            unsafe_allow_html=True,
        )
        architecture_diagram()

        section_divider()

        # ══════════════════════════════════════════════════════════════════════
        # Section 2: Champion Model
        # ══════════════════════════════════════════════════════════════════════
        st.markdown(
            f"<div style='color:{Colors.TEXT_PRIMARY};font-weight:600;"
            f"font-size:1.1rem;margin-bottom:16px'>"
            f"{Icons.html('SHIELD_CHECK', 18, Colors.ACCENT)} "
            f"Champion Model — CatBoost v3.2.1</div>",
            unsafe_allow_html=True,
        )

        col_summary, col_gauge = st.columns([2, 1])

        with col_summary:
            model_performance_summary("CatBoost Champion", CHAMPION_METRICS)

        with col_gauge:
            st.plotly_chart(
                gauge_chart(
                    CHAMPION_METRICS["PR-AUC"],
                    title="PR-AUC Score",
                    min_val=0,
                    max_val=1,
                    height=260,
                ),
                use_container_width=True,
            )
            progress_ring(
                CHAMPION_METRICS["ROC-AUC"],
                max_val=1.0,
                size=140,
                color=Colors.SUCCESS,
                label="ROC-AUC",
            )

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Calibration curve
        st.markdown(
            f"<div style='color:{Colors.TEXT_SECONDARY};font-size:0.85rem;"
            f"margin-bottom:8px'>Calibration Curve — CatBoost Champion</div>",
            unsafe_allow_html=True,
        )

        rng_cal = np.random.default_rng(7)
        prob_true = np.array(
            [0.002, 0.008, 0.018, 0.035, 0.052, 0.098, 0.145, 0.230, 0.380, 0.560]
        )
        prob_pred = np.array(
            [0.005, 0.012, 0.025, 0.040, 0.065, 0.105, 0.160, 0.250, 0.400, 0.580]
        )

        fig_cal = go.Figure()
        fig_cal.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Perfect calibration",
                line=dict(color=Colors.TEXT_MUTED, dash="dash", width=1.5),
            )
        )
        fig_cal.add_trace(
            go.Scatter(
                x=prob_pred,
                y=prob_true,
                mode="lines+markers",
                name="CatBoost Champion",
                line=dict(color=Colors.ACCENT, width=2.5),
                marker=dict(
                    size=8,
                    color=Colors.ACCENT,
                    line=dict(width=1.5, color=Colors.BG_CARD),
                ),
            )
        )
        fig_cal.update_layout(
            xaxis_title="Mean Predicted Probability",
            yaxis_title="Fraction of Positives",
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(
                range=[-0.02, 1.02], showgrid=True, gridcolor=Colors.BORDER_SUBTLE
            ),
            yaxis=dict(
                range=[-0.02, 1.02], showgrid=True, gridcolor=Colors.BORDER_SUBTLE
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
        st.plotly_chart(fig_cal, use_container_width=True)

        section_divider()

        # ══════════════════════════════════════════════════════════════════════
        # Section 3: Model Leaderboard
        # ══════════════════════════════════════════════════════════════════════
        st.markdown(
            f"<div style='color:{Colors.TEXT_PRIMARY};font-weight:600;"
            f"font-size:1.1rem;margin-bottom:16px'>"
            f"{Icons.html('BAR_CHART_2', 18, Colors.ACCENT)} "
            f"Model Leaderboard</div>",
            unsafe_allow_html=True,
        )

        status_colored = MODEL_LEADERBOARD.copy()
        status_colored["Status"] = status_colored["Status"].apply(
            lambda s: f'<span style="display:flex;align-items:center;gap:6px">'
            f'<span style="width:8px;height:8px;border-radius:50%;'
            f'background:{Colors.SUCCESS if s == "Champion" else Colors.WARNING if s == "Specialist" else Colors.TEXT_MUTED}"></span>'
            f"{s}</span>"
        )

        data_table(
            df=MODEL_LEADERBOARD,
            columns={
                "Rank": "Rank",
                "Model": "Model",
                "Version": "Version",
                "PR-AUC": "PR-AUC",
                "ROC-AUC": "ROC-AUC",
                "F2": "F2",
                "Latency (ms)": "Latency (ms)",
                "Status": "Status",
            },
            max_rows=10,
            status_col="Status",
            striped=True,
        )

        section_divider()

        # ══════════════════════════════════════════════════════════════════════
        # Section 4: Feature Importance
        # ══════════════════════════════════════════════════════════════════════
        st.markdown(
            f"<div style='color:{Colors.TEXT_PRIMARY};font-weight:600;"
            f"font-size:1.1rem;margin-bottom:16px'>"
            f"{Icons.html('SLIDERS', 18, Colors.ACCENT)} "
            f"Feature Importance — Top 15</div>",
            unsafe_allow_html=True,
        )

        fi_sorted = fi_df.sort_values("importance", ascending=True).tail(15)

        fi_colors = [
            (
                Colors.CRITICAL
                if v > 0.12
                else Colors.WARNING if v > 0.06 else Colors.ACCENT
            )
            for v in fi_sorted["importance"]
        ]

        fig_fi = go.Figure(
            go.Bar(
                y=fi_sorted["feature"],
                x=fi_sorted["importance"],
                orientation="h",
                marker=dict(color=fi_colors, cornerradius=4),
                text=fi_sorted["importance"].round(3),
                textposition="outside",
                textfont=dict(color=Colors.TEXT_SECONDARY, size=11),
                hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
            )
        )
        fig_fi.update_layout(
            height=440,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(
                title=dict(
                    text="Importance", font=dict(size=12, color=Colors.TEXT_MUTED)
                ),
                showgrid=True,
                gridcolor=Colors.BORDER_SUBTLE,
                tickfont=dict(color=Colors.TEXT_MUTED, size=11),
            ),
            yaxis=dict(
                tickfont=dict(color=Colors.TEXT_SECONDARY, size=11),
                showgrid=False,
            ),
            hoverlabel=dict(
                bgcolor=Colors.BG_ELEVATED,
                bordercolor=Colors.BORDER_DEFAULT,
                font=dict(color=Colors.TEXT_PRIMARY, size=12),
            ),
        )
        st.plotly_chart(fig_fi, use_container_width=True)

        section_divider()

        # ══════════════════════════════════════════════════════════════════════
        # Section 5: Confusion Matrix
        # ══════════════════════════════════════════════════════════════════════
        st.markdown(
            f"<div style='color:{Colors.TEXT_PRIMARY};font-weight:600;"
            f"font-size:1.1rem;margin-bottom:16px'>"
            f"{Icons.html('TARGET', 18, Colors.ACCENT)} "
            f"Confusion Matrix</div>",
            unsafe_allow_html=True,
        )

        tp, fp, tn, fn = cm["TP"], cm["FP"], cm["TN"], cm["FN"]
        total = tp + fp + tn + fn
        accuracy = (tp + tn) / total
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        col_cm, col_metrics = st.columns([3, 2])

        with col_cm:
            z = [[tn, fp], [fn, tp]]
            text = [
                [f"<b>TN</b><br>{tn:,}", f"<b>FP</b><br>{fp:,}"],
                [f"<b>FN</b><br>{fn:,}", f"<b>TP</b><br>{tp:,}"],
            ]

            fig_cm = go.Figure(
                go.Heatmap(
                    z=z,
                    x=["Predicted Legit", "Predicted Fraud"],
                    y=["Actual Legit", "Actual Fraud"],
                    text=text,
                    texttemplate="%{text}",
                    textfont=dict(size=14),
                    colorscale=[
                        [0, Colors.rgba(Colors.SUCCESS, 0.15)],
                        [0.5, Colors.rgba(Colors.ACCENT, 0.1)],
                        [1, Colors.rgba(Colors.SUCCESS, 0.6)],
                    ],
                    showscale=False,
                    hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z:,}<extra></extra>",
                )
            )
            fig_cm.update_layout(
                height=340,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis=dict(
                    tickfont=dict(color=Colors.TEXT_SECONDARY, size=12),
                    showgrid=False,
                ),
                yaxis=dict(
                    tickfont=dict(color=Colors.TEXT_SECONDARY, size=12),
                    showgrid=False,
                    autorange="reversed",
                ),
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_metrics:
            metric_row(
                [
                    {
                        "label": "Accuracy",
                        "value": f"{accuracy:.4f}",
                        "icon": "CHECK_CIRCLE",
                    },
                    {
                        "label": "Precision",
                        "value": f"{precision:.4f}",
                        "icon": "CROSSHAIR",
                    },
                ]
            )
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            metric_row(
                [
                    {"label": "Recall", "value": f"{recall:.4f}", "icon": "SEARCH"},
                    {
                        "label": "Total Samples",
                        "value": f"{total:,}",
                        "icon": "BAR_CHART",
                    },
                ]
            )
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

            metric_table(
                metrics=[
                    {
                        "label": "True Positives",
                        "value": f"{tp:,}",
                        "status": "healthy",
                    },
                    {
                        "label": "False Positives",
                        "value": f"{fp:,}",
                        "status": "warning",
                    },
                    {
                        "label": "True Negatives",
                        "value": f"{tn:,}",
                        "status": "healthy",
                    },
                    {
                        "label": "False Negatives",
                        "value": f"{fn:,}",
                        "status": "critical",
                    },
                ],
                title="Detailed Counts",
            )

        section_divider()

        # ══════════════════════════════════════════════════════════════════════
        # Section 6: Training History
        # ══════════════════════════════════════════════════════════════════════
        st.markdown(
            f"<div style='color:{Colors.TEXT_PRIMARY};font-weight:600;"
            f"font-size:1.1rem;margin-bottom:16px'>"
            f"{Icons.html('ACTIVITY', 18, Colors.ACCENT)} "
            f"Training History</div>",
            unsafe_allow_html=True,
        )

        col_pr, col_loss = st.columns(2)

        with col_pr:
            fig_pr_hist = go.Figure()
            fig_pr_hist.add_trace(
                go.Scatter(
                    x=iterations,
                    y=train_pr_auc,
                    mode="lines",
                    name="PR-AUC (train)",
                    line=dict(color=Colors.ACCENT, width=2, shape="spline"),
                    fill="tozeroy",
                    fillcolor=Colors.rgba(Colors.ACCENT, 0.08),
                )
            )
            fig_pr_hist.add_hline(
                y=CHAMPION_METRICS["PR-AUC"],
                line_dash="dash",
                line_color=Colors.SUCCESS,
                line_width=1.5,
                annotation_text=f"Final: {CHAMPION_METRICS['PR-AUC']:.4f}",
                annotation_position="top right",
                annotation_font=dict(color=Colors.SUCCESS, size=11),
            )
            fig_pr_hist.update_layout(
                title=dict(
                    text="PR-AUC Over Training",
                    font=dict(size=14, color=Colors.TEXT_PRIMARY),
                ),
                xaxis_title="Iteration",
                yaxis_title="PR-AUC",
                height=320,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
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
                    range=[0.45, 1.0],
                ),
                hoverlabel=dict(
                    bgcolor=Colors.BG_ELEVATED,
                    bordercolor=Colors.BORDER_DEFAULT,
                    font=dict(color=Colors.TEXT_PRIMARY, size=12),
                ),
                hovermode="x unified",
            )
            st.plotly_chart(fig_pr_hist, use_container_width=True)

        with col_loss:
            fig_loss = go.Figure()
            fig_loss.add_trace(
                go.Scatter(
                    x=iterations,
                    y=train_loss,
                    mode="lines",
                    name="Loss (train)",
                    line=dict(color=Colors.CHART_4, width=2, shape="spline"),
                    fill="tozeroy",
                    fillcolor=Colors.rgba(Colors.CHART_4, 0.08),
                )
            )
            fig_loss.add_hline(
                y=train_loss[-1],
                line_dash="dash",
                line_color=Colors.CHART_4,
                line_width=1.5,
                annotation_text=f"Final: {train_loss[-1]:.4f}",
                annotation_position="top right",
                annotation_font=dict(color=Colors.CHART_4, size=11),
            )
            fig_loss.update_layout(
                title=dict(
                    text="Loss Over Training",
                    font=dict(size=14, color=Colors.TEXT_PRIMARY),
                ),
                xaxis_title="Iteration",
                yaxis_title="Loss",
                height=320,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
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
                hoverlabel=dict(
                    bgcolor=Colors.BG_ELEVATED,
                    bordercolor=Colors.BORDER_DEFAULT,
                    font=dict(color=Colors.TEXT_PRIMARY, size=12),
                ),
                hovermode="x unified",
            )
            st.plotly_chart(fig_loss, use_container_width=True)
