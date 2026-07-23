"""FraudTrap Dashboard — Model Lifecycle Page"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

from dashboard.components import (
    kpi_row,
    bar_chart,
    line_chart,
    gauge_chart,
    data_table,
    metric_table,
    page_container,
    section_divider,
    lifecycle_timeline,
    model_performance_summary,
    metric_row,
    progress_ring,
)
from dashboard.components.data_loader import make_pr_curve
from dashboard.theme.colors import Colors
from dashboard.theme.icons import Icons
from dashboard.theme.typography import Typography

# ── Synthetic Data ────────────────────────────────────────────────────────────


def _generate_synthetic_data():
    """Generate all synthetic data for the lifecycle page."""
    rng = np.random.default_rng(42)
    now = datetime.now()

    # Scoring volume over time
    dates = [now - timedelta(days=i) for i in range(30)][::-1]
    scoring_volume = [
        int(rng.poisson(12000 + 3000 * np.sin(i * np.pi / 15))) for i in range(30)
    ]
    avg_risk_scores = [float(rng.beta(2, 8)) for _ in range(30)]

    # Model registry
    models = [
        {
            "Model": "CatBoost",
            "Version": "3.2.1",
            "Status": "Champion",
            "PR-AUC": 0.8342,
            "Latency": "42ms",
            "Training Date": "2026-07-10",
            "Actions": "View",
        },
        {
            "Model": "FT-Transformer",
            "Version": "2.1.0",
            "Status": "Specialist",
            "PR-AUC": 0.8127,
            "Latency": "68ms",
            "Training Date": "2026-07-08",
            "Actions": "View",
        },
        {
            "Model": "TabPFN",
            "Version": "1.4.3",
            "Status": "Specialist",
            "PR-AUC": 0.7654,
            "Latency": "31ms",
            "Training Date": "2026-07-05",
            "Actions": "View",
        },
        {
            "Model": "Isolation Forest",
            "Version": "2.0.0",
            "Status": "Offline",
            "PR-AUC": 0.6891,
            "Latency": "18ms",
            "Training Date": "2026-06-28",
            "Actions": "View",
        },
        {
            "Model": "VAE Anomaly",
            "Version": "1.1.2",
            "Status": "Offline",
            "PR-AUC": 0.6543,
            "Latency": "55ms",
            "Training Date": "2026-06-20",
            "Actions": "View",
        },
        {
            "Model": "CatBoost v2",
            "Version": "3.1.0",
            "Status": "Archived",
            "PR-AUC": 0.7988,
            "Latency": "44ms",
            "Training Date": "2026-06-15",
            "Actions": "View",
        },
    ]

    return {
        "dates": dates,
        "scoring_volume": scoring_volume,
        "avg_risk_scores": avg_risk_scores,
        "models": models,
        "pr_auc": 0.8342,
        "roc_auc": 0.9123,
        "f2_score": 0.7891,
        "latency_ms": 42.3,
        "training_date": "2026-07-10",
        "fraud_labels_current": 3847,
        "fraud_labels_target": 5000,
        "pr_auc_current": 0.8342,
        "pr_auc_target": 0.78,
        "weeks_active": 6,
        "weeks_target": 8,
        "total_scored": 2_847_293,
        "active_models": 3,
        "version": "3.2.1",
    }


# ── Render ────────────────────────────────────────────────────────────────────


def render(tenant_id: str):
    data = _generate_synthetic_data()

    with page_container(
        "Model Lifecycle",
        "Model registry, promotion workflow, and lifecycle management",
        "LAYERS",
    ):

        # ── KPI Row ───────────────────────────────────────────────────────────
        kpi_row(
            [
                {
                    "label": "Total Scored",
                    "value": f"{data['total_scored']:,}",
                    "icon": "DATABASE",
                },
                {
                    "label": "Active Models",
                    "value": str(data["active_models"]),
                    "icon": "BOX",
                },
                {
                    "label": "Champion PR-AUC",
                    "value": f"{data['pr_auc']:.4f}",
                    "icon": "TARGET",
                },
                {
                    "label": "P95 Latency",
                    "value": f"{data['latency_ms']:.0f}ms",
                    "icon": "TIMER",
                },
            ]
        )

        # ══════════════════════════════════════════════════════════════════════
        # SECTION 1: Lifecycle Timeline
        # ══════════════════════════════════════════════════════════════════════
        section_divider()

        st.markdown(
            f"""
<div style="display:flex;align-items:center;gap:10px;margin-bottom:4px">
    {Icons.html('LAYERS', 18, Colors.ACCENT)}
    <span style="font-size:{Typography.TEXT_LG};font-weight:{Typography.WEIGHT_SEMIBOLD};color:{Colors.TEXT_PRIMARY}">Model Lifecycle Timeline</span>
</div>
<div style="font-size:{Typography.TEXT_SM};color:{Colors.TEXT_MUTED};margin-bottom:16px">Current stage: <span style="color:{Colors.ACCENT};font-weight:{Typography.WEIGHT_SEMIBOLD}">Champion</span> — Model actively serving predictions</div>
""",
            unsafe_allow_html=True,
        )

        lifecycle_timeline(current_stage=4)

        # ══════════════════════════════════════════════════════════════════════
        # SECTION 2: Phase Progression
        # ══════════════════════════════════════════════════════════════════════
        section_divider()

        st.markdown(
            f"""
<div style="display:flex;align-items:center;gap:10px;margin-bottom:16px">
    {Icons.html('TRENDING_UP', 18, Colors.ACCENT)}
    <span style="font-size:{Typography.TEXT_LG};font-weight:{Typography.WEIGHT_SEMIBOLD};color:{Colors.TEXT_PRIMARY}">Phase Progression</span>
</div>
""",
            unsafe_allow_html=True,
        )

        phases = [
            {
                "name": "Phase 1 — Unsupervised",
                "model": "VAE + Isolation Forest",
                "min_labels": 0,
                "current": 0,
                "target": 0,
                "color": Colors.PHASE_1,
                "status": "Complete",
                "status_color": Colors.SUCCESS,
                "transition": "Ready",
            },
            {
                "name": "Phase 2 — Semi-supervised",
                "model": "TabPFN Foundation Model",
                "min_labels": 500,
                "current": 3847,
                "target": 5000,
                "color": Colors.PHASE_2,
                "status": "Complete",
                "status_color": Colors.SUCCESS,
                "transition": "Ready",
            },
            {
                "name": "Phase 3 — Supervised",
                "model": "CatBoost Champion",
                "min_labels": 5000,
                "current": 3847,
                "target": 5000,
                "color": Colors.PHASE_3,
                "status": "Active",
                "status_color": Colors.ACCENT,
                "transition": "In Progress",
            },
        ]

        phase_cols = st.columns(3)
        for i, (col, phase) in enumerate(zip(phase_cols, phases)):
            progress_pct = (
                min(phase["current"] / phase["target"] * 100, 100)
                if phase["target"] > 0
                else 100
            )
            ring_color = Colors.SUCCESS if progress_pct >= 100 else phase["color"]

            with col:
                st.markdown(
                    f"""
<div style="background:{Colors.BG_CARD};border:1px solid {Colors.BORDER_DEFAULT};border-radius:12px;padding:20px;height:100%">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
        <span style="font-size:{Typography.TEXT_BASE};font-weight:{Typography.WEIGHT_SEMIBOLD};color:{phase['color']}">{phase['name']}</span>
        <span style="font-size:{Typography.TEXT_XS};padding:3px 8px;border-radius:4px;background:{Colors.rgba(phase['status_color'], 0.15)};color:{phase['status_color']};font-weight:{Typography.WEIGHT_SEMIBOLD}">{phase['status']}</span>
    </div>
    <div style="font-size:{Typography.TEXT_SM};color:{Colors.TEXT_SECONDARY};margin-bottom:8px">Model: <span style="color:{Colors.TEXT_PRIMARY};font-weight:{Typography.WEIGHT_MEDIUM}">{phase['model']}</span></div>
    <div style="font-size:{Typography.TEXT_SM};color:{Colors.TEXT_SECONDARY};margin-bottom:4px">Min Labels: <span style="color:{Colors.TEXT_PRIMARY};font-weight:{Typography.WEIGHT_MEDIUM}">{phase['min_labels']:,}</span></div>
    <div style="display:flex;align-items:center;gap:12px;margin-top:12px">
        <div style="flex:1">
            <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                <span style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_MUTED}">Progress</span>
                <span style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_PRIMARY};font-weight:{Typography.WEIGHT_SEMIBOLD}">{phase['current']:,}/{phase['target']:,}</span>
            </div>
            <div style="height:6px;background:{Colors.BG_SECONDARY};border-radius:3px;overflow:hidden">
                <div style="height:100%;width:{progress_pct}%;background:{ring_color};border-radius:3px;transition:width 0.3s"></div>
            </div>
        </div>
    </div>
    <div style="margin-top:12px;display:flex;align-items:center;gap:6px">
        {Icons.html('CHEVRON_RIGHT' if i < 2 else 'CHECK_CIRCLE', 14, phase['status_color'])}
        <span style="font-size:{Typography.TEXT_XS};color:{phase['status_color']};font-weight:{Typography.WEIGHT_SEMIBOLD}">{phase['transition']}</span>
    </div>
</div>
""",
                    unsafe_allow_html=True,
                )

        # ══════════════════════════════════════════════════════════════════════
        # SECTION 3: Champion Model
        # ══════════════════════════════════════════════════════════════════════
        section_divider()

        st.markdown(
            f"""
<div style="display:flex;align-items:center;gap:10px;margin-bottom:16px">
    {Icons.html('SHIELD_CHECK', 18, Colors.ACCENT)}
    <span style="font-size:{Typography.TEXT_LG};font-weight:{Typography.WEIGHT_SEMIBOLD};color:{Colors.TEXT_PRIMARY}">Champion Model</span>
</div>
""",
            unsafe_allow_html=True,
        )

        col_left, col_right = st.columns([1, 1])

        with col_left:
            model_performance_summary(
                "CatBoost v3.2.1",
                {
                    "PR-AUC": 0.8342,
                    "ROC-AUC": 0.9123,
                    "F2-Score": 0.7891,
                    "Latency": "42.3ms",
                    "Training Date": "2026-07-10",
                    "Status": "Champion",
                },
            )

        with col_right:
            st.plotly_chart(
                gauge_chart(
                    data["pr_auc"],
                    title="PR-AUC vs Phase 3 Gate (0.78)",
                    min_val=0,
                    max_val=1,
                    thresholds=[
                        {
                            "range": [0, 0.50],
                            "color": Colors.rgba(Colors.CRITICAL, 0.3),
                        },
                        {
                            "range": [0.50, 0.78],
                            "color": Colors.rgba(Colors.WARNING, 0.3),
                        },
                        {
                            "range": [0.78, 1.0],
                            "color": Colors.rgba(Colors.SUCCESS, 0.3),
                        },
                    ],
                    height=280,
                ),
                use_container_width=True,
            )

        # ══════════════════════════════════════════════════════════════════════
        # SECTION 4: Model Registry
        # ══════════════════════════════════════════════════════════════════════
        section_divider()

        st.markdown(
            f"""
<div style="display:flex;align-items:center;gap:10px;margin-bottom:16px">
    {Icons.html('DATABASE', 18, Colors.ACCENT)}
    <span style="font-size:{Typography.TEXT_LG};font-weight:{Typography.WEIGHT_SEMIBOLD};color:{Colors.TEXT_PRIMARY}">Model Registry</span>
</div>
""",
            unsafe_allow_html=True,
        )

        registry_df = pd.DataFrame(data["models"])

        status_badge_map = {
            "Champion": "success",
            "Specialist": "info",
            "Offline": "warning",
            "Archived": "muted",
        }

        registry_html_rows = ""
        for _, row in registry_df.iterrows():
            badge_class = status_badge_map.get(row["Status"], "info")
            badge_bg = (
                Colors.SUCCESS
                if row["Status"] == "Champion"
                else (
                    Colors.INFO
                    if row["Status"] == "Specialist"
                    else (
                        Colors.WARNING
                        if row["Status"] == "Offline"
                        else Colors.TEXT_MUTED
                    )
                )
            )
            registry_html_rows += f"""
<tr>
    <td style="padding:12px 16px;color:{Colors.TEXT_PRIMARY};font-weight:{Typography.WEIGHT_MEDIUM};border-bottom:1px solid {Colors.BORDER_SUBTLE}">{row['Model']}</td>
    <td style="padding:12px 16px;color:{Colors.TEXT_SECONDARY};border-bottom:1px solid {Colors.BORDER_SUBTLE}">{row['Version']}</td>
    <td style="padding:12px 16px;border-bottom:1px solid {Colors.BORDER_SUBTLE}">
        <span style="display:inline-block;padding:3px 10px;border-radius:4px;font-size:{Typography.TEXT_XS};font-weight:{Typography.WEIGHT_SEMIBOLD};background:{Colors.rgba(badge_bg, 0.15)};color:{badge_bg}">{row['Status']}</span>
    </td>
    <td style="padding:12px 16px;color:{Colors.TEXT_PRIMARY};font-weight:{Typography.WEIGHT_SEMIBOLD};border-bottom:1px solid {Colors.BORDER_SUBTLE}">{row['PR-AUC']:.4f}</td>
    <td style="padding:12px 16px;color:{Colors.TEXT_SECONDARY};border-bottom:1px solid {Colors.BORDER_SUBTLE}">{row['Latency']}</td>
    <td style="padding:12px 16px;color:{Colors.TEXT_SECONDARY};border-bottom:1px solid {Colors.BORDER_SUBTLE}">{row['Training Date']}</td>
    <td style="padding:12px 16px;color:{Colors.ACCENT};border-bottom:1px solid {Colors.BORDER_SUBTLE};cursor:pointer">{Icons.html('EYE', 14, Colors.ACCENT)}</td>
</tr>"""

        st.markdown(
            f"""
<div style="overflow-x:auto;border:1px solid {Colors.BORDER_DEFAULT};border-radius:10px;background:{Colors.BG_CARD}">
    <table style="width:100%;border-collapse:separate;border-spacing:0;font-family:{Typography.FONT_FAMILY}">
        <thead><tr>
            <th style="padding:12px 16px;color:{Colors.TEXT_MUTED};font-weight:{Typography.WEIGHT_SEMIBOLD};font-size:{Typography.TEXT_SM};text-transform:uppercase;letter-spacing:{Typography.TRACKING_WIDER};text-align:left;border-bottom:1px solid {Colors.BORDER_DEFAULT};background:{Colors.BG_SECONDARY}">Model</th>
            <th style="padding:12px 16px;color:{Colors.TEXT_MUTED};font-weight:{Typography.WEIGHT_SEMIBOLD};font-size:{Typography.TEXT_SM};text-transform:uppercase;letter-spacing:{Typography.TRACKING_WIDER};text-align:left;border-bottom:1px solid {Colors.BORDER_DEFAULT};background:{Colors.BG_SECONDARY}">Version</th>
            <th style="padding:12px 16px;color:{Colors.TEXT_MUTED};font-weight:{Typography.WEIGHT_SEMIBOLD};font-size:{Typography.TEXT_SM};text-transform:uppercase;letter-spacing:{Typography.TRACKING_WIDER};text-align:left;border-bottom:1px solid {Colors.BORDER_DEFAULT};background:{Colors.BG_SECONDARY}">Status</th>
            <th style="padding:12px 16px;color:{Colors.TEXT_MUTED};font-weight:{Typography.WEIGHT_SEMIBOLD};font-size:{Typography.TEXT_SM};text-transform:uppercase;letter-spacing:{Typography.TRACKING_WIDER};text-align:left;border-bottom:1px solid {Colors.BORDER_DEFAULT};background:{Colors.BG_SECONDARY}">PR-AUC</th>
            <th style="padding:12px 16px;color:{Colors.TEXT_MUTED};font-weight:{Typography.WEIGHT_SEMIBOLD};font-size:{Typography.TEXT_SM};text-transform:uppercase;letter-spacing:{Typography.TRACKING_WIDER};text-align:left;border-bottom:1px solid {Colors.BORDER_DEFAULT};background:{Colors.BG_SECONDARY}">Latency</th>
            <th style="padding:12px 16px;color:{Colors.TEXT_MUTED};font-weight:{Typography.WEIGHT_SEMIBOLD};font-size:{Typography.TEXT_SM};text-transform:uppercase;letter-spacing:{Typography.TRACKING_WIDER};text-align:left;border-bottom:1px solid {Colors.BORDER_DEFAULT};background:{Colors.BG_SECONDARY}">Training Date</th>
            <th style="padding:12px 16px;color:{Colors.TEXT_MUTED};font-weight:{Typography.WEIGHT_SEMIBOLD};font-size:{Typography.TEXT_SM};text-transform:uppercase;letter-spacing:{Typography.TRACKING_WIDER};text-align:left;border-bottom:1px solid {Colors.BORDER_DEFAULT};background:{Colors.BG_SECONDARY}">Actions</th>
        </tr></thead>
        <tbody>{registry_html_rows}</tbody>
    </table>
</div>
""",
            unsafe_allow_html=True,
        )

        # ══════════════════════════════════════════════════════════════════════
        # SECTION 5: Promotion Readiness
        # ══════════════════════════════════════════════════════════════════════
        section_divider()

        st.markdown(
            f"""
<div style="display:flex;align-items:center;gap:10px;margin-bottom:16px">
    {Icons.html('TRENDING_UP', 18, Colors.ACCENT)}
    <span style="font-size:{Typography.TEXT_LG};font-weight:{Typography.WEIGHT_SEMIBOLD};color:{Colors.TEXT_PRIMARY}">Promotion Readiness</span>
    <span style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_MUTED};margin-left:8px">Phase 2 → 3 Transition Criteria</span>
</div>
""",
            unsafe_allow_html=True,
        )

        criteria = [
            {
                "name": "Fraud Labels",
                "current": data["fraud_labels_current"],
                "target": data["fraud_labels_target"],
                "unit": "",
                "color": Colors.PHASE_2,
                "icon": "FINGERPRINT",
            },
            {
                "name": "PR-AUC",
                "current": data["pr_auc_current"],
                "target": data["pr_auc_target"],
                "unit": "",
                "color": Colors.SUCCESS,
                "icon": "TARGET",
            },
            {
                "name": "Weeks Active",
                "current": data["weeks_active"],
                "target": data["weeks_target"],
                "unit": " weeks",
                "color": Colors.WARNING,
                "icon": "CLOCK",
            },
        ]

        criteria_cols = st.columns(3)
        for col, c in zip(criteria_cols, criteria):
            pct = min(c["current"] / c["target"], 1.0) if c["target"] > 0 else 1.0
            is_ready = pct >= 1.0
            bar_color = Colors.SUCCESS if is_ready else c["color"]

            with col:
                st.markdown(
                    f"""
<div style="background:{Colors.BG_CARD};border:1px solid {Colors.BORDER_DEFAULT};border-radius:12px;padding:20px">
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:16px">
        {Icons.html(c['icon'], 16, c['color'])}
        <span style="font-size:{Typography.TEXT_BASE};font-weight:{Typography.WEIGHT_SEMIBOLD};color:{Colors.TEXT_PRIMARY}">{c['name']}</span>
    </div>
    <div style="display:flex;align-items:baseline;gap:4px;margin-bottom:12px">
        <span style="font-size:{Typography.TEXT_2XL};font-weight:{Typography.WEIGHT_BOLD};color:{Colors.TEXT_PRIMARY}">{c['current']:,.0f}</span>
        <span style="font-size:{Typography.TEXT_SM};color:{Colors.TEXT_MUTED}">/ {c['target']:,.0f}{c['unit']}</span>
    </div>
    <div style="height:8px;background:{Colors.BG_SECONDARY};border-radius:4px;overflow:hidden;margin-bottom:8px">
        <div style="height:100%;width:{pct * 100}%;background:{bar_color};border-radius:4px;transition:width 0.3s"></div>
    </div>
    <div style="display:flex;justify-content:space-between;align-items:center">
        <span style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_MUTED}">{pct * 100:.1f}% complete</span>
        <span style="font-size:{Typography.TEXT_XS};padding:2px 8px;border-radius:4px;font-weight:{Typography.WEIGHT_SEMIBOLD};background:{Colors.rgba(Colors.SUCCESS if is_ready else Colors.WARNING, 0.15)};color:{Colors.SUCCESS if is_ready else Colors.WARNING}">
            {'Ready' if is_ready else 'In Progress'}
        </span>
    </div>
</div>
""",
                    unsafe_allow_html=True,
                )

        # ══════════════════════════════════════════════════════════════════════
        # SECTION 6: Scoring Volume
        # ══════════════════════════════════════════════════════════════════════
        section_divider()

        st.markdown(
            f"""
<div style="display:flex;align-items:center;gap:10px;margin-bottom:16px">
    {Icons.html('ACTIVITY', 18, Colors.ACCENT)}
    <span style="font-size:{Typography.TEXT_LG};font-weight:{Typography.WEIGHT_SEMIBOLD};color:{Colors.TEXT_PRIMARY}">Scoring Volume</span>
</div>
""",
            unsafe_allow_html=True,
        )

        vol_col, risk_col = st.columns(2)

        with vol_col:
            fig_vol = go.Figure()
            fig_vol.add_trace(
                go.Bar(
                    x=data["dates"],
                    y=data["scoring_volume"],
                    name="Transactions Scored",
                    marker_color=Colors.CHART_1,
                    opacity=0.85,
                    marker=dict(cornerradius=4),
                )
            )
            fig_vol.update_layout(
                height=320,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=Colors.TEXT_SECONDARY, size=12),
                margin=dict(l=0, r=0, t=40, b=0),
                xaxis=dict(
                    showgrid=False,
                    showline=False,
                    zeroline=False,
                    tickfont=dict(color=Colors.TEXT_MUTED, size=10),
                    title=dict(text="", font=dict(color=Colors.TEXT_MUTED, size=11)),
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor=Colors.BORDER_SUBTLE,
                    gridwidth=1,
                    showline=False,
                    zeroline=False,
                    tickfont=dict(color=Colors.TEXT_MUTED, size=10),
                    title=dict(
                        text="Count", font=dict(color=Colors.TEXT_MUTED, size=11)
                    ),
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
                    font=dict(color=Colors.TEXT_PRIMARY, size=12),
                ),
                hovermode="x unified",
                title=dict(
                    text="Scoring Volume Over Time",
                    font=dict(size=14, color=Colors.TEXT_PRIMARY),
                ),
            )
            st.plotly_chart(fig_vol, use_container_width=True)

        with risk_col:
            fig_risk = go.Figure()
            fig_risk.add_trace(
                go.Scatter(
                    x=data["dates"],
                    y=data["avg_risk_scores"],
                    mode="lines+markers",
                    name="Avg Risk Score",
                    line=dict(color=Colors.CHART_5, width=2.5, shape="spline"),
                    marker=dict(size=7, color=Colors.CHART_5),
                )
            )
            fig_risk.add_hline(
                y=0.40,
                line_dash="dash",
                line_color=Colors.WARNING,
                annotation_text="Review threshold (0.40)",
                annotation_position="top right",
                annotation_font=dict(color=Colors.WARNING, size=10),
            )
            fig_risk.add_hline(
                y=0.85,
                line_dash="dash",
                line_color=Colors.CRITICAL,
                annotation_text="Block threshold (0.85)",
                annotation_position="top right",
                annotation_font=dict(color=Colors.CRITICAL, size=10),
            )
            fig_risk.update_layout(
                height=320,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=Colors.TEXT_SECONDARY, size=12),
                margin=dict(l=0, r=0, t=40, b=0),
                xaxis=dict(
                    showgrid=False,
                    showline=False,
                    zeroline=False,
                    tickfont=dict(color=Colors.TEXT_MUTED, size=10),
                    title=dict(text="", font=dict(color=Colors.TEXT_MUTED, size=11)),
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor=Colors.BORDER_SUBTLE,
                    gridwidth=1,
                    showline=False,
                    zeroline=False,
                    tickfont=dict(color=Colors.TEXT_MUTED, size=10),
                    title=dict(
                        text="Avg Risk Score",
                        font=dict(color=Colors.TEXT_MUTED, size=11),
                    ),
                    range=[0, 1.0],
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
                    font=dict(color=Colors.TEXT_PRIMARY, size=12),
                ),
                hovermode="x unified",
                title=dict(
                    text="Average Risk Score Trend",
                    font=dict(size=14, color=Colors.TEXT_PRIMARY),
                ),
            )
            st.plotly_chart(fig_risk, use_container_width=True)
