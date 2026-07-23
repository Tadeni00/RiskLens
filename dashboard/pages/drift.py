"""
FraudTrap Dashboard — Drift Monitoring Page
Data drift detection, PSI analysis, and model stability.
"""
from datetime import datetime, timedelta, timezone

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from dashboard.components import (
    kpi_row,
    bar_chart,
    line_chart,
    area_chart,
    heatmap_chart,
    metric_table,
    data_table,
    page_container,
    section_divider,
    alert,
    metric_row,
    status_timeline,
)
from dashboard.components.data_loader import make_drift_data
from dashboard.theme.colors import Colors
from dashboard.theme.icons import Icons


# ── Synthetic Concept Drift Data ──────────────────────────────────────────────

def _make_concept_drift_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic concept drift time-series data."""
    rng = np.random.default_rng(42)
    now = datetime.now(timezone.utc)
    dates = [now - timedelta(days=30 - i) for i in range(31)]

    kl_values = np.clip(
        np.cumsum(rng.normal(0.003, 0.015, 31)) + 0.12 + rng.normal(0, 0.02, 31),
        0.02, 0.45,
    )
    pr_auc_values = np.clip(
        0.88 - np.cumsum(rng.normal(0.001, 0.005, 31)) + rng.normal(0, 0.008, 31),
        0.65, 0.95,
    )

    concept_df = pd.DataFrame({
        "date": dates,
        "kl_divergence": kl_values,
    })
    performance_df = pd.DataFrame({
        "date": dates,
        "pr_auc": pr_auc_values,
    })
    return concept_df, performance_df


def _make_drift_history() -> pd.DataFrame:
    """Generate synthetic drift event history."""
    rng = np.random.default_rng(42)
    now = datetime.now(timezone.utc)
    features = ["amount", "acct_v_1h_count", "geo_speed_kmh", "typing_zscore",
                "device_account_count", "amount_zscore"]
    actions = ["Monitor", "Retrain Scheduled", "Retraining Complete",
               "Threshold Adjusted", "Investigating", "Auto-Retrained"]

    rows = []
    for i in range(20):
        days_ago = int(rng.integers(0, 30))
        ts = now - timedelta(days=days_ago, hours=int(rng.integers(0, 24)))
        psi = float(rng.uniform(0.02, 0.38))
        if psi < 0.10:
            status = "Stable"
            action = "Monitor"
        elif psi < 0.20:
            status = "Warning"
            action = rng.choice(["Monitor", "Threshold Adjusted", "Investigating"])
        else:
            status = "Critical"
            action = rng.choice(["Retrain Scheduled", "Retraining Complete", "Auto-Retrained"])

        rows.append({
            "Date": ts.strftime("%Y-%m-%d %H:%M"),
            "Feature": rng.choice(features),
            "PSI": round(psi, 4),
            "Status": status,
            "Action Taken": action,
        })
    return pd.DataFrame(sorted(rows, key=lambda r: r["Date"], reverse=True))


def _make_drift_timeline_events() -> list:
    """Generate recent drift events for the status timeline."""
    rng = np.random.default_rng(42)
    now = datetime.now(timezone.utc)
    event_templates = [
        ("Critical drift detected on {feat} (PSI = {psi:.3f})", "critical"),
        ("Warning: {feat} approaching drift threshold (PSI = {psi:.3f})", "warning"),
        ("Model retraining completed — {feat} drift resolved", "success"),
        ("PSI monitoring baseline updated for {feat}", "info"),
        ("Feature {feat} stabilized after retraining (PSI = {psi:.3f})", "success"),
    ]
    features = ["amount", "acct_v_1h_count", "geo_speed_kmh", "typing_zscore", "amount_zscore"]

    events = []
    for i in range(8):
        minutes_ago = int(rng.integers(5, 360))
        ts = now - timedelta(minutes=minutes_ago)
        template, level = rng.choice(event_templates)
        feat = rng.choice(features)
        psi = float(rng.uniform(0.05, 0.35))
        msg = template.format(feat=feat, psi=psi)
        events.append({
            "time": ts.strftime("%H:%M"),
            "message": msg,
            "level": level,
        })
    return events


# ── Main Render ───────────────────────────────────────────────────────────────

def render(tenant_id: str):
    drift = make_drift_data()
    concept_df, performance_df = _make_concept_drift_data()

    n_features = len(drift)
    critical_features = [k for k, v in drift.items() if v["psi"] > 0.20]
    warning_features = [k for k, v in drift.items() if 0.10 <= v["psi"] <= 0.20]
    stable_features = [k for k, v in drift.items() if v["psi"] < 0.10]

    with page_container("Drift Monitoring", "Data drift detection, PSI analysis, and model stability", "ACTIVITY"):

        # ── Section 1: Drift Overview ──────────────────────────────────────
        kpi_row([
            {
                "label": "Features Monitored",
                "value": str(n_features),
                "icon": "LAYERS",
                "trend": 0.0,
                "trend_label": "active",
            },
            {
                "label": "Critical Drifts",
                "value": str(len(critical_features)),
                "icon": "ALERT_TRIANGLE",
                "status": "critical" if critical_features else "healthy",
                "trend": len(critical_features),
                "trend_label": "features",
            },
            {
                "label": "Warning Drifts",
                "value": str(len(warning_features)),
                "icon": "CLOCK",
                "status": "warning" if warning_features else "healthy",
                "trend": len(warning_features),
                "trend_label": "features",
            },
            {
                "label": "Stable Features",
                "value": str(len(stable_features)),
                "icon": "CHECK_CIRCLE",
                "status": "healthy",
                "trend": len(stable_features),
                "trend_label": "features",
            },
        ])

        st.markdown(
            f"<div style='margin-top:12px;margin-bottom:4px;color:{Colors.TEXT_PRIMARY};"
            f"font-weight:600;font-size:14px'>"
            f"{Icons.html('INFO', 14, Colors.ACCENT)} PSI Threshold Guide</div>",
            unsafe_allow_html=True,
        )
        col_s, col_w, col_c = st.columns(3)
        with col_s:
            st.markdown(
                f'<div style="background:{Colors.rgba(Colors.SUCCESS, 0.10)};'
                f'border:1px solid {Colors.rgba(Colors.SUCCESS, 0.25)};'
                f'border-radius:8px;padding:12px 16px">'
                f'<div style="color:{Colors.SUCCESS};font-weight:700;font-size:14px">Stable</div>'
                f'<div style="color:{Colors.TEXT_MUTED};font-size:12px;margin-top:2px">PSI &lt; 0.10</div>'
                f'<div style="color:{Colors.TEXT_SECONDARY};font-size:12px;margin-top:4px">No action needed</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col_w:
            st.markdown(
                f'<div style="background:{Colors.rgba(Colors.WARNING, 0.10)};'
                f'border:1px solid {Colors.rgba(Colors.WARNING, 0.25)};'
                f'border-radius:8px;padding:12px 16px">'
                f'<div style="color:{Colors.WARNING};font-weight:700;font-size:14px">Warning</div>'
                f'<div style="color:{Colors.TEXT_MUTED};font-size:12px;margin-top:2px">PSI 0.10 - 0.20</div>'
                f'<div style="color:{Colors.TEXT_SECONDARY};font-size:12px;margin-top:4px">Monitor closely</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col_c:
            st.markdown(
                f'<div style="background:{Colors.rgba(Colors.CRITICAL, 0.10)};'
                f'border:1px solid {Colors.rgba(Colors.CRITICAL, 0.25)};'
                f'border-radius:8px;padding:12px 16px">'
                f'<div style="color:{Colors.CRITICAL};font-weight:700;font-size:14px">Critical</div>'
                f'<div style="color:{Colors.TEXT_MUTED};font-size:12px;margin-top:2px">PSI &gt; 0.20</div>'
                f'<div style="color:{Colors.TEXT_SECONDARY};font-size:12px;margin-top:4px">Retrain triggered</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        section_divider()

        # ── Section 2: Feature Drift ──────────────────────────────────────
        st.markdown(
            f"<div style='color:{Colors.TEXT_PRIMARY};font-weight:600;font-size:16px;margin-bottom:8px'>"
            f"{Icons.html('BAR_CHART', 16, Colors.ACCENT)} Feature PSI Scores</div>",
            unsafe_allow_html=True,
        )

        psi_df = pd.DataFrame([
            {"Feature": k, "PSI": v["psi"],
             "Status": "Critical" if v["psi"] > 0.20 else "Warning" if v["psi"] > 0.10 else "Stable"}
            for k, v in drift.items()
        ]).sort_values("PSI", ascending=True)

        color_map = {
            "Critical": Colors.CRITICAL,
            "Warning": Colors.WARNING,
            "Stable": Colors.SUCCESS,
        }
        bar_colors = [color_map[s] for s in psi_df["Status"]]

        fig_psi = go.Figure()
        fig_psi.add_trace(go.Bar(
            x=psi_df["PSI"], y=psi_df["Feature"],
            orientation="h",
            marker=dict(color=bar_colors, cornerradius=4),
            text=psi_df["PSI"].round(4),
            textposition="outside",
            textfont=dict(color=Colors.TEXT_SECONDARY, size=11),
        ))
        fig_psi.add_vline(
            x=0.20, line_dash="dash", line_color=Colors.CRITICAL,
            annotation_text="Critical (0.20)",
            annotation_font=dict(color=Colors.CRITICAL, size=11),
        )
        fig_psi.add_vline(
            x=0.10, line_dash="dot", line_color=Colors.WARNING,
            annotation_text="Warning (0.10)",
            annotation_font=dict(color=Colors.WARNING, size=11),
        )
        fig_psi.update_layout(
            height=320,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="'Inter', sans-serif", color=Colors.TEXT_SECONDARY, size=12),
            xaxis=dict(
                title=dict(text="Population Stability Index (PSI)", font=dict(size=12, color=Colors.TEXT_MUTED)),
                range=[0, max(0.30, psi_df["PSI"].max() * 1.3)],
                showgrid=True, gridcolor=Colors.BORDER_SUBTLE,
                zeroline=False, tickfont=dict(color=Colors.TEXT_MUTED, size=11),
            ),
            yaxis=dict(
                showgrid=False, tickfont=dict(color=Colors.TEXT_SECONDARY, size=12),
            ),
            margin=dict(l=0, r=0, t=10, b=0),
            hoverlabel=dict(
                bgcolor=Colors.BG_ELEVATED, bordercolor=Colors.BORDER_DEFAULT,
                font=dict(color=Colors.TEXT_PRIMARY, size=12),
            ),
        )
        st.plotly_chart(fig_psi, use_container_width=True)

        metric_table(
            metrics=[
                {"label": f["Feature"], "value": f"{f['PSI']:.4f}", "status": f["Status"].lower()}
                for _, f in psi_df.sort_values("PSI", ascending=False).iterrows()
            ],
            title="PSI by Feature",
        )

        st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

        st.markdown(
            f"<div style='color:{Colors.TEXT_PRIMARY};font-weight:600;font-size:14px;margin-bottom:8px'>"
            f"{Icons.html('EYE', 14, Colors.ACCENT)} Distribution Comparison</div>",
            unsafe_allow_html=True,
        )

        feat_sel = st.selectbox(
            "Select feature to compare distributions",
            list(drift.keys()),
            label_visibility="collapsed",
        )
        d = drift[feat_sel]

        rng = np.random.default_rng(42)
        baseline_samples = rng.normal(
            d["baseline_mean"],
            max(0.1, abs(d["baseline_mean"]) * 0.2 + 1.0),
            1000,
        )
        current_samples = rng.normal(
            d["current_mean"],
            max(0.1, abs(d["current_mean"]) * 0.2 + 1.0),
            1000,
        )

        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=baseline_samples, name="Baseline (reference)",
            nbinsx=40, marker_color=Colors.CHART_1, opacity=0.6,
            histnorm="probability density",
        ))
        fig_dist.add_trace(go.Histogram(
            x=current_samples, name="Current (recent)",
            nbinsx=40, marker_color=Colors.CHART_4, opacity=0.6,
            histnorm="probability density",
        ))
        fig_dist.update_layout(
            barmode="overlay", height=300,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="'Inter', sans-serif", color=Colors.TEXT_SECONDARY, size=12),
            xaxis=dict(
                title=dict(text=feat_sel, font=dict(size=12, color=Colors.TEXT_MUTED)),
                showgrid=False, zeroline=False,
                tickfont=dict(color=Colors.TEXT_MUTED, size=11),
            ),
            yaxis=dict(
                title=dict(text="Density", font=dict(size=12, color=Colors.TEXT_MUTED)),
                showgrid=True, gridcolor=Colors.BORDER_SUBTLE, zeroline=False,
                tickfont=dict(color=Colors.TEXT_MUTED, size=11),
            ),
            legend=dict(
                font=dict(color=Colors.TEXT_SECONDARY, size=11),
                bgcolor="rgba(0,0,0,0)", orientation="h",
                yanchor="bottom", y=1.02, xanchor="right", x=1,
            ),
            margin=dict(l=0, r=0, t=10, b=0),
            hoverlabel=dict(
                bgcolor=Colors.BG_ELEVATED, bordercolor=Colors.BORDER_DEFAULT,
                font=dict(color=Colors.TEXT_PRIMARY, size=12),
            ),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        status_color = "critical" if d["psi"] > 0.20 else "warning" if d["psi"] > 0.10 else "healthy"
        st.markdown(
            f'<div style="display:inline-flex;align-items:center;gap:8px;padding:6px 14px;'
            f'background:{Colors.BG_CARD};border:1px solid {Colors.BORDER_DEFAULT};'
            f'border-radius:8px;margin-top:4px">'
            f'<span class="status-dot {status_color}"></span>'
            f'<span style="color:{Colors.TEXT_SECONDARY};font-size:13px">'
            f'PSI for <b style="color:{Colors.TEXT_PRIMARY}">{feat_sel}</b>: '
            f'<b style="color:{Colors.TEXT_PRIMARY}">{d["psi"]:.4f}</b> | '
            f'Baseline mean: <b>{d["baseline_mean"]:.3f}</b> | '
            f'Current mean: <b>{d["current_mean"]:.3f}</b></span></div>',
            unsafe_allow_html=True,
        )

        section_divider()

        # ── Section 3: Concept Drift ──────────────────────────────────────
        st.markdown(
            f"<div style='color:{Colors.TEXT_PRIMARY};font-weight:600;font-size:16px;margin-bottom:8px'>"
            f"{Icons.html('ACTIVITY', 16, Colors.ACCENT)} Concept Drift</div>",
            unsafe_allow_html=True,
        )

        col_kl, col_pr = st.columns(2)

        with col_kl:
            fig_kl = go.Figure()
            fig_kl.add_trace(go.Scatter(
                x=concept_df["date"], y=concept_df["kl_divergence"],
                mode="lines+markers",
                line=dict(color=Colors.CHART_5, width=2, shape="spline"),
                marker=dict(size=5, color=Colors.CHART_5),
                name="KL Divergence",
            ))
            fig_kl.add_hline(
                y=0.25, line_dash="dash", line_color=Colors.CRITICAL,
                annotation_text="Threshold",
                annotation_font=dict(color=Colors.CRITICAL, size=10),
            )
            fig_kl.update_layout(
                height=280,
                title=dict(text="Prediction Drift (KL Divergence)", font=dict(size=14, color=Colors.TEXT_PRIMARY)),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="'Inter', sans-serif", color=Colors.TEXT_SECONDARY, size=12),
                xaxis=dict(
                    showgrid=False, showline=False, zeroline=False,
                    tickformat="%b %d",
                    tickfont=dict(color=Colors.TEXT_MUTED, size=11),
                ),
                yaxis=dict(
                    title=dict(text="KL Divergence", font=dict(size=12, color=Colors.TEXT_MUTED)),
                    showgrid=True, gridcolor=Colors.BORDER_SUBTLE, zeroline=False,
                    tickfont=dict(color=Colors.TEXT_MUTED, size=11),
                ),
                margin=dict(l=0, r=0, t=40, b=0),
                hoverlabel=dict(
                    bgcolor=Colors.BG_ELEVATED, bordercolor=Colors.BORDER_DEFAULT,
                    font=dict(color=Colors.TEXT_PRIMARY, size=12),
                ),
                hovermode="x unified",
            )
            st.plotly_chart(fig_kl, use_container_width=True)

        with col_pr:
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(
                x=performance_df["date"], y=performance_df["pr_auc"],
                mode="lines+markers",
                line=dict(color=Colors.CHART_2, width=2, shape="spline"),
                marker=dict(size=5, color=Colors.CHART_2),
                name="PR-AUC",
                fill="tozeroy",
                fillcolor=Colors.rgba(Colors.CHART_2, 0.08),
            ))
            fig_pr.add_hline(
                y=0.80, line_dash="dash", line_color=Colors.WARNING,
                annotation_text="Min Threshold",
                annotation_font=dict(color=Colors.WARNING, size=10),
            )
            fig_pr.update_layout(
                height=280,
                title=dict(text="Model Performance Drift (PR-AUC)", font=dict(size=14, color=Colors.TEXT_PRIMARY)),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="'Inter', sans-serif", color=Colors.TEXT_SECONDARY, size=12),
                xaxis=dict(
                    showgrid=False, showline=False, zeroline=False,
                    tickformat="%b %d",
                    tickfont=dict(color=Colors.TEXT_MUTED, size=11),
                ),
                yaxis=dict(
                    title=dict(text="PR-AUC", font=dict(size=12, color=Colors.TEXT_MUTED)),
                    showgrid=True, gridcolor=Colors.BORDER_SUBTLE, zeroline=False,
                    range=[0.6, 1.0],
                    tickfont=dict(color=Colors.TEXT_MUTED, size=11),
                ),
                margin=dict(l=0, r=0, t=40, b=0),
                hoverlabel=dict(
                    bgcolor=Colors.BG_ELEVATED, bordercolor=Colors.BORDER_DEFAULT,
                    font=dict(color=Colors.TEXT_PRIMARY, size=12),
                ),
                hovermode="x unified",
            )
            st.plotly_chart(fig_pr, use_container_width=True)

        avg_kl = float(concept_df["kl_divergence"].mean())
        latest_pr_auc = float(performance_df["pr_auc"].iloc[-1])
        metric_row([
            {"label": "Avg KL Divergence", "value": f"{avg_kl:.4f}",
             "color": Colors.WARNING if avg_kl > 0.20 else Colors.SUCCESS,
             "icon": "ACTIVITY"},
            {"label": "Current PR-AUC", "value": f"{latest_pr_auc:.4f}",
             "color": Colors.SUCCESS if latest_pr_auc > 0.80 else Colors.CRITICAL,
             "icon": "TARGET"},
            {"label": "Monitoring Window", "value": "30 days",
             "color": Colors.TEXT_PRIMARY, "icon": "CALENDAR"},
        ])

        section_divider()

        # ── Section 4: Drift Alerts ───────────────────────────────────────
        st.markdown(
            f"<div style='color:{Colors.TEXT_PRIMARY};font-weight:600;font-size:16px;margin-bottom:8px'>"
            f"{Icons.html('BELL', 16, Colors.ACCENT)} Drift Alerts & Timeline</div>",
            unsafe_allow_html=True,
        )

        if critical_features:
            alert(
                message=f"Critical drift detected in {len(critical_features)} feature(s): "
                        f"{', '.join(critical_features)}. Retraining recommended.",
                level="critical",
                timestamp=datetime.now(timezone.utc) - timedelta(minutes=2),
            )

        for feat in warning_features:
            psi_val = drift[feat]["psi"]
            alert(
                message=f"Warning: Feature '{feat}' approaching drift threshold (PSI = {psi_val:.4f}). "
                        f"Monitoring more frequently.",
                level="warning",
                timestamp=datetime.now(timezone.utc) - timedelta(minutes=int(psi_val * 100)),
            )

        if not critical_features and not warning_features:
            alert(
                message="All monitored features are within acceptable drift bounds.",
                level="success",
                timestamp=datetime.now(timezone.utc),
            )

        if avg_kl > 0.25:
            alert(
                message=f"Concept drift detected: KL divergence ({avg_kl:.4f}) exceeds threshold (0.25). "
                        f"Model retraining should be considered.",
                level="critical",
                timestamp=datetime.now(timezone.utc) - timedelta(minutes=10),
            )

        timeline_events = _make_drift_timeline_events()
        st.markdown(
            f"<div style='color:{Colors.TEXT_PRIMARY};font-weight:600;font-size:14px;margin:16px 0 8px'>"
            f"{Icons.html('CLOCK', 14, Colors.ACCENT)} Recent Drift Events</div>",
            unsafe_allow_html=True,
        )
        status_timeline(timeline_events)

        section_divider()

        # ── Section 5: Drift History ──────────────────────────────────────
        st.markdown(
            f"<div style='color:{Colors.TEXT_PRIMARY};font-weight:600;font-size:16px;margin-bottom:8px'>"
            f"{Icons.html('FILE_TEXT', 16, Colors.ACCENT)} Drift History Log</div>",
            unsafe_allow_html=True,
        )

        history_df = _make_drift_history()

        data_table(
            df=history_df,
            columns={
                "Date": "Date",
                "Feature": "Feature",
                "PSI": "PSI",
                "Status": "Status",
                "Action Taken": "Action Taken",
            },
            max_rows=20,
            status_col="Status",
            striped=True,
        )

        total_events = len(history_df)
        n_retrains = len(history_df[history_df["Action Taken"].str.contains("Retrain", case=False)])
        avg_psi_history = float(history_df["PSI"].mean())

        metric_row([
            {"label": "Total Events", "value": str(total_events),
             "color": Colors.TEXT_PRIMARY, "icon": "HASH"},
            {"label": "Retrain Triggered", "value": str(n_retrains),
             "color": Colors.WARNING if n_retrains > 0 else Colors.SUCCESS, "icon": "REFRESH"},
            {"label": "Avg Historical PSI", "value": f"{avg_psi_history:.4f}",
             "color": Colors.WARNING if avg_psi_history > 0.15 else Colors.SUCCESS, "icon": "BAR_CHART"},
        ])
