"""
RiskLens Console — Overview Page
Real-time fraud monitoring command center.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components import (
    kpi_row,
    status_card,
    bar_chart,
    line_chart,
    metric_table,
    data_table,
    page_container,
    section_divider,
    alert,
    metric_row,
)
from dashboard.components.data_loader import (
    make_transactions,
    compute_kpis,
    make_live_timeseries,
    currency_fmt,
    CURRENCY_SYMBOLS,
)
from dashboard.theme.colors import Colors
from dashboard.theme.icons import Icons

# ── Helpers ──────────────────────────────────────────────────────────────────


def _sparkline_html(
    values: list[float], color: str = Colors.ACCENT, width: int = 60, height: int = 24
) -> str:
    """Render an inline SVG sparkline."""
    if not values:
        return ""
    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1
    pts = []
    step = width / max(len(values) - 1, 1)
    for i, v in enumerate(values):
        x = i * step
        y = height - ((v - mn) / rng) * height
        pts.append(f"{x:.1f},{y:.1f}")
    path_d = "M" + " L".join(pts)
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'style="display:block;margin-top:4px">'
        f'<path d="{path_d}" fill="none" stroke="{color}" stroke-width="1.5" '
        f'stroke-linecap="round" stroke-linejoin="round"/>'
        f"</svg>"
    )


def _hourly_fraud_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fraud transactions into hourly buckets for the last 24h."""
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=24)
    recent = df[df["timestamp"] >= cutoff].copy()
    recent["hour"] = recent["timestamp"].dt.floor("h")
    hourly = (
        recent.groupby("hour")
        .agg(fraud_count=("is_fraud", "sum"), total=("transaction_id", "count"))
        .reset_index()
        .sort_values("hour")
    )
    return hourly


def _format_amount(val: float, currency: str = "USD") -> str:
    """Format large amounts with K/M suffix and correct currency symbol."""
    return currency_fmt(val, currency)


# ── Main Render ──────────────────────────────────────────────────────────────


def render(tenant_id: str):
    # ── Data Generation ──────────────────────────────────────────────────────
    df = make_transactions(500)
    if tenant_id != "all_tenants":
        df = df[df["tenant_id"] == tenant_id].reset_index(drop=True)

    kpis = compute_kpis(df)
    live_ts, _ = make_live_timeseries(24, tenant_id)

    total = kpis["total"]
    n_block = kpis["n_block"]
    n_review = kpis["n_review"]
    fraud_rate = kpis["fraud_rate"]
    avg_lat = kpis["avg_lat"]
    fp_rate = rng_val = float(np.random.default_rng(42).beta(1, 50) * 100)

    now = datetime.now(timezone.utc)
    today_mask = df["timestamp"].dt.date == now.date()
    txn_today = int(today_mask.sum()) if today_mask.any() else int(total * 0.04)

    total_revenue = float(df["amount"].sum())
    blocked_amount = float(df.loc[df["decision"] == "BLOCK", "amount"].sum())
    revenue_protected = blocked_amount
    primary_currency = (
        df["currency"].mode().iloc[0] if "currency" in df.columns else "NGN"
    )

    sparkline_vals = (
        live_ts["txn_per_min"].tolist()
        if "txn_per_min" in live_ts.columns
        else [
            float(np.random.poisson(800 + 200 * np.sin(h * np.pi / 12)))
            for h in range(24)
        ]
    )

    with page_container("Overview", "Real-time fraud monitoring dashboard", "HOME"):

        # ── Section 1: KPI Strip ─────────────────────────────────────────────
        kpi_row(
            [
                {
                    "label": "Transactions Today",
                    "value": f"{txn_today:,}",
                    "trend": 12.3,
                    "trend_label": "vs yesterday",
                    "icon": "BAR_CHART",
                },
                {
                    "label": "Fraud Prevented",
                    "value": f"{n_block:,}",
                    "trend": -8.1,
                    "trend_label": "vs 7d avg",
                    "icon": "SHIELD_CHECK",
                },
                {
                    "label": "Fraud Rate",
                    "value": f"{fraud_rate:.2f}%",
                    "trend": fraud_rate - 1.3,
                    "delta_suffix": "%",
                    "status": "healthy" if fraud_rate < 2.0 else "warning",
                    "icon": "TARGET",
                },
                {
                    "label": "Revenue Protected",
                    "value": _format_amount(revenue_protected, primary_currency),
                    "trend": 5.4,
                    "trend_label": "this week",
                    "icon": "CREDIT_CARD",
                },
                {
                    "label": "P95 Latency",
                    "value": f"{avg_lat:.0f}ms",
                    "trend": avg_lat - 78.0,
                    "delta_suffix": "ms",
                    "status": "healthy" if avg_lat < 100 else "warning",
                    "icon": "ZAP",
                },
                {
                    "label": "False Positive Rate",
                    "value": f"{fp_rate:.3f}%",
                    "trend": -0.12,
                    "delta_suffix": "%",
                    "status": "healthy" if fp_rate < 1.0 else "warning",
                    "icon": "ALERT_TRIANGLE",
                },
                {
                    "label": "Champion Model",
                    "value": "CatBoost v1.0",
                    "trend": None,
                    "icon": "BRAIN",
                },
                {
                    "label": "Current Drift Status",
                    "value": "Stable",
                    "trend": None,
                    "status": "healthy",
                    "icon": "ACTIVITY",
                },
            ]
        )

        section_divider()

        # ── Section 2: Operational Health ─────────────────────────────────────
        infra_components = [
            {
                "title": "API Gateway",
                "status": "healthy",
                "details": "Port 8000 — 99.97% uptime (30d)",
                "icon": "SERVER",
                "metrics": {"Latency": "12ms", "RPS": "1,247", "Errors": "0.03%"},
            },
            {
                "title": "Redis Cache",
                "status": "healthy",
                "details": "Port 6379 — Hit rate 94.2%",
                "icon": "DATABASE",
                "metrics": {"Memory": "2.1GB", "Connections": "84", "Evictions": "0"},
            },
            {
                "title": "Kafka Broker",
                "status": "warning",
                "details": "Port 9092 — Consumer lag detected",
                "icon": "Zap",
                "metrics": {
                    "Lag": "12,400",
                    "Partitions": "24",
                    "Throughput": "8.2K/s",
                },
            },
            {
                "title": "ClickHouse",
                "status": "healthy",
                "details": "Port 9000 — 47.2M rows indexed",
                "icon": "DATABASE",
                "metrics": {"Query P95": "45ms", "Partitions": "128", "Disk": "142GB"},
            },
            {
                "title": "Feature Store",
                "status": "healthy",
                "details": "Online + offline serving",
                "icon": "LAYERS",
                "metrics": {"Features": "156", "Freshness": "<1s", "Cache": "99.1%"},
            },
            {
                "title": "Model Registry",
                "status": "healthy",
                "details": "3 champion / 2 challenger models",
                "icon": "BOX",
                "metrics": {"Models": "5", "Versions": "12", "Last Deploy": "2h ago"},
            },
            {
                "title": "Inference Engine",
                "status": "healthy",
                "details": "GPU cluster — 4x A10G",
                "icon": "CPU",
                "metrics": {"GPU Util": "67%", "Batch Size": "256", "Queue": "0"},
            },
        ]

        n_cols = len(infra_components)
        cols = st.columns(min(n_cols, 4))
        for i, comp in enumerate(infra_components):
            with cols[i % 4]:
                status_card(
                    title=comp["title"],
                    status=comp["status"],
                    details=comp["details"],
                    icon=comp["icon"],
                    metrics=comp["metrics"],
                )

        section_divider()

        # ── Section 3: Risk Intelligence ─────────────────────────────────────
        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.markdown(
                f"<div style='color:{Colors.TEXT_PRIMARY};font-weight:600;font-size:16px;margin-bottom:8px'>"
                f"{Icons.html('BAR_CHART', 16, Colors.ACCENT)} 24h Fraud Timeline</div>",
                unsafe_allow_html=True,
            )

            hourly = _hourly_fraud_counts(df)
            if hourly.empty:
                hours = pd.date_range(end=now, periods=24, freq="h")
                hourly = pd.DataFrame(
                    {
                        "hour": hours,
                        "fraud_count": np.random.poisson(3, 24),
                    }
                )

            fig_bar = go.Figure()
            fig_bar.add_trace(
                go.Bar(
                    x=hourly["hour"],
                    y=hourly["fraud_count"],
                    marker=dict(
                        color=Colors.CHART_4,
                        cornerradius=4,
                        line=dict(width=0),
                    ),
                    hovertemplate="<b>%{x|%H:%M}</b><br>Fraud: %{y}<extra></extra>",
                )
            )
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(
                    family="'Inter', sans-serif", color=Colors.TEXT_SECONDARY, size=12
                ),
                margin=dict(l=0, r=0, t=8, b=0),
                height=280,
                xaxis=dict(
                    showgrid=False,
                    showline=False,
                    tickformat="%H:%M",
                    tickfont=dict(color=Colors.TEXT_MUTED, size=11),
                    dtick=3600000 * 4,
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor=Colors.BORDER_SUBTLE,
                    gridwidth=1,
                    showline=False,
                    zeroline=False,
                    tickfont=dict(color=Colors.TEXT_MUTED, size=11),
                    title=dict(
                        text="Fraud Count", font=dict(size=12, color=Colors.TEXT_MUTED)
                    ),
                ),
                bargap=0.15,
                hoverlabel=dict(
                    bgcolor=Colors.BG_ELEVATED,
                    bordercolor=Colors.BORDER_DEFAULT,
                    font=dict(color=Colors.TEXT_PRIMARY, size=12),
                ),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_right:
            st.markdown(
                f"<div style='color:{Colors.TEXT_PRIMARY};font-weight:600;font-size:16px;margin-bottom:8px'>"
                f"{Icons.html('TARGET', 16, Colors.ACCENT)} Decision Distribution</div>",
                unsafe_allow_html=True,
            )

            decision_counts = df["decision"].value_counts()
            decisions = ["APPROVE", "REVIEW", "BLOCK"]
            counts = [int(decision_counts.get(d, 0)) for d in decisions]
            pie_colors = [Colors.SUCCESS, Colors.WARNING, Colors.CRITICAL]

            fig_pie = go.Figure(
                go.Pie(
                    labels=decisions,
                    values=counts,
                    hole=0.55,
                    marker=dict(
                        colors=pie_colors, line=dict(color=Colors.BG_CARD, width=2)
                    ),
                    textfont=dict(color=Colors.TEXT_PRIMARY, size=13),
                    hovertemplate="<b>%{label}</b><br>%{value:,} ({%{percent}})<extra></extra>",
                    textinfo="percent",
                    textposition="inside",
                )
            )
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                height=280,
                margin=dict(l=0, r=0, t=8, b=0),
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.05,
                    xanchor="center",
                    x=0.5,
                    font=dict(color=Colors.TEXT_SECONDARY, size=12),
                    bgcolor="rgba(0,0,0,0)",
                ),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        section_divider()

        # ── Section 4: Recent Activity ───────────────────────────────────────
        st.markdown(
            f"<div style='color:{Colors.TEXT_PRIMARY};font-weight:600;font-size:16px;margin-bottom:8px'>"
            f"{Icons.html('CLOCK', 16, Colors.ACCENT)} Recent Activity</div>",
            unsafe_allow_html=True,
        )

        recent = df.sort_values("timestamp", ascending=False).head(10).copy()
        recent["time_str"] = recent["timestamp"].dt.strftime("%H:%M:%S")
        recent["amount_str"] = recent.apply(
            lambda r: currency_fmt(r["amount"], r.get("currency", "NGN")), axis=1
        )
        recent["score_str"] = recent["risk_score"].apply(lambda x: f"{x:.4f}")
        recent["latency_str"] = recent["latency_ms"].apply(lambda x: f"{x:.0f}ms")

        decision_colors = {
            "APPROVE": Colors.SUCCESS,
            "REVIEW": Colors.WARNING,
            "BLOCK": Colors.CRITICAL,
        }

        table_df = recent[
            [
                "time_str",
                "amount_str",
                "channel",
                "decision",
                "score_str",
                "latency_str",
            ]
        ].copy()
        table_df.columns = ["Time", "Amount", "Channel", "Decision", "Score", "Latency"]

        data_table(
            df=table_df,
            columns={
                "Time": "Time",
                "Amount": "Amount",
                "Channel": "Channel",
                "Decision": "Decision",
                "Score": "Score",
                "Latency": "Latency",
            },
            max_rows=10,
            status_col="Decision",
            striped=True,
        )

        section_divider()

        # ── Section 5: Alerts ────────────────────────────────────────────────
        st.markdown(
            f"<div style='color:{Colors.TEXT_PRIMARY};font-weight:600;font-size:16px;margin-bottom:8px'>"
            f"{Icons.html('BELL', 16, Colors.ACCENT)} Active Alerts</div>",
            unsafe_allow_html=True,
        )

        alert(
            message="Kafka consumer lag exceeded threshold on fraud-scores topic (12,400 messages behind)",
            level="warning",
            timestamp=now - timedelta(minutes=3),
        )
        alert(
            message="Anomalous spike detected: 340% increase in transactions from KE channel in last 15 minutes",
            level="critical",
            timestamp=now - timedelta(minutes=7),
        )
        alert(
            message="Model drift score below threshold (PSI = 0.03) — no action required",
            level="info",
            timestamp=now - timedelta(minutes=22),
        )
        alert(
            message="Scheduled model retraining initiated for CatBoost v1.1 champion candidate",
            level="success",
            timestamp=now - timedelta(minutes=45),
        )
