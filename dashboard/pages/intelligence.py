"""
FraudTrap Dashboard — Risk Intelligence Page
Fraud pattern analysis, geographic risk mapping, and entity-level risk leaderboards.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from dashboard.components import (
    kpi_row,
    bar_chart,
    horizontal_bar,
    heatmap_chart,
    treemap_chart,
    scatter_chart,
    data_table,
    leader_board,
    page_container,
    section_divider,
    metric_row,
)
from dashboard.components.data_loader import make_transactions, currency_fmt
from dashboard.theme.colors import Colors
from dashboard.theme.typography import Typography


# ── Helpers ──────────────────────────────────────────────────────────────────

RISK_BUCKETS = ["Very Low", "Low", "Medium", "High", "Very High"]
RISK_COLORS = [Colors.SUCCESS, Colors.CHART_7, Colors.WARNING, Colors.CHART_8, Colors.CRITICAL]

COUNTRY_COORDS = {
    "NG": {"lat": 9.08, "lon": 8.67, "name": "Nigeria", "currency": "NGN"},
    "KE": {"lat": -0.02, "lon": 37.90, "name": "Kenya", "currency": "KES"},
    "ZA": {"lat": -30.56, "lon": 22.94, "name": "South Africa", "currency": "ZAR"},
    "GB": {"lat": 55.38, "lon": -3.44, "name": "United Kingdom", "currency": "GBP"},
    "US": {"lat": 37.09, "lon": -95.71, "name": "United States", "currency": "USD"},
}


def _bucket_risk(score: float) -> str:
    if score < 0.2:
        return RISK_BUCKETS[0]
    if score < 0.4:
        return RISK_BUCKETS[1]
    if score < 0.6:
        return RISK_BUCKETS[2]
    if score < 0.8:
        return RISK_BUCKETS[3]
    return RISK_BUCKETS[4]


def _section_heading(icon: str, title: str, color: str = Colors.ACCENT):
    st.markdown(
        f"<div style='color:{Colors.TEXT_PRIMARY};font-weight:{Typography.WEIGHT_SEMIBOLD};"
        f"font-size:{Typography.TEXT_LG};margin-bottom:12px'>"
        f"<span style='color:{color};margin-right:8px'>&#x{icon};</span>{title}</div>",
        unsafe_allow_html=True,
    )


# ── Main Render ──────────────────────────────────────────────────────────────

def render(tenant_id: str):
    df = make_transactions(500)
    if tenant_id != "all_tenants":
        df = df[df["tenant_id"] == tenant_id].reset_index(drop=True)

    fraud_df = df[df["is_fraud"] == 1]
    total_txn = len(df)
    fraud_count = len(fraud_df)
    fraud_rate = fraud_df["is_fraud"].mean() * 100 if total_txn > 0 else 0.0
    avg_risk = float(df["risk_score"].mean()) if total_txn > 0 else 0.0
    blocked = int((df["decision"] == "BLOCK").sum())
    reviewed = int((df["decision"] == "REVIEW").sum())

    with page_container("Risk Intelligence", "Fraud pattern analysis and risk assessment", "SHIELD"):

        # ── KPI Strip ─────────────────────────────────────────────────────────
        kpi_row([
            {
                "label": "Total Transactions",
                "value": f"{total_txn:,}",
                "trend": None,
                "icon": "BAR_CHART",
            },
            {
                "label": "Fraud Detected",
                "value": f"{fraud_count:,}",
                "trend": None,
                "icon": "ALERT_TRIANGLE",
                "status": "warning",
            },
            {
                "label": "Fraud Rate",
                "value": f"{fraud_rate:.2f}%",
                "trend": None,
                "status": "healthy" if fraud_rate < 2.0 else "warning",
                "icon": "TARGET",
            },
            {
                "label": "Avg Risk Score",
                "value": f"{avg_risk:.4f}",
                "trend": None,
                "icon": "SHIELD",
            },
            {
                "label": "Blocked",
                "value": f"{blocked:,}",
                "trend": None,
                "icon": "X",
            },
            {
                "label": "Under Review",
                "value": f"{reviewed:,}",
                "trend": None,
                "icon": "EYE",
            },
        ])

        section_divider()

        # ══════════════════════════════════════════════════════════════════════
        # Section 1: Risk Distribution
        # ══════════════════════════════════════════════════════════════════════
        _section_heading("2699", "Risk Distribution")

        col_hist, col_map = st.columns([3, 2])

        with col_hist:
            df["risk_bucket"] = df["risk_score"].apply(_bucket_risk)
            bucket_counts = df["risk_bucket"].value_counts().reindex(RISK_BUCKETS, fill_value=0)

            fig_hist = go.Figure(go.Bar(
                x=bucket_counts.index.tolist(),
                y=bucket_counts.values.tolist(),
                marker=dict(
                    color=RISK_COLORS,
                    cornerradius=4,
                    line=dict(width=0),
                ),
                text=bucket_counts.values.tolist(),
                textposition="outside",
                textfont=dict(color=Colors.TEXT_SECONDARY, size=11),
                hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>",
            ))
            fig_hist.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="'Inter', sans-serif", color=Colors.TEXT_SECONDARY, size=12),
                margin=dict(l=0, r=0, t=32, b=0),
                height=320,
                xaxis=dict(showgrid=False, showline=False, tickfont=dict(color=Colors.TEXT_MUTED, size=11)),
                yaxis=dict(
                    showgrid=True, gridcolor=Colors.BORDER_SUBTLE, gridwidth=1,
                    showline=False, zeroline=False,
                    tickfont=dict(color=Colors.TEXT_MUTED, size=11),
                    title=dict(text="Count", font=dict(size=12, color=Colors.TEXT_MUTED)),
                ),
                title=dict(text="Fraud Risk Distribution", font=dict(size=14, color=Colors.TEXT_PRIMARY)),
                hoverlabel=dict(
                    bgcolor=Colors.BG_ELEVATED, bordercolor=Colors.BORDER_DEFAULT,
                    font=dict(color=Colors.TEXT_PRIMARY, size=12),
                ),
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_map:
            geo_data = []
            for code, info in COUNTRY_COORDS.items():
                country_txns = df[df["country_code"] == code]
                if len(country_txns) == 0:
                    continue
                geo_data.append({
                    "country": info["name"],
                    "lat": info["lat"],
                    "lon": info["lon"],
                    "fraud_count": int(country_txns["is_fraud"].sum()),
                    "total": len(country_txns),
                    "fraud_rate": round(country_txns["is_fraud"].mean() * 100, 2),
                    "total_amount": float(country_txns["amount"].sum()),
                    "currency": info.get("currency", "USD"),
                })

            geo_df = pd.DataFrame(geo_data) if geo_data else pd.DataFrame(columns=["country", "lat", "lon", "fraud_count", "total", "fraud_rate", "total_amount"])

            if not geo_df.empty:
                fig_geo = go.Figure(go.Scattergeo(
                    lon=geo_df["lon"],
                    lat=geo_df["lat"],
                    text=geo_df.apply(
                        lambda r: (
                            f"<b>{r['country']}</b><br>"
                            f"Fraud: {r['fraud_count']:,} / {r['total']:,}<br>"
                            f"Rate: {r['fraud_rate']:.2f}%<br>"
                            f"Volume: {currency_fmt(r['total_amount'], r.get('currency', 'USD'))}"
                        ), axis=1
                    ),
                    hoverinfo="text",
                    marker=dict(
                        size=geo_df["fraud_rate"].clip(lower=5).values.tolist(),
                        color=geo_df["fraud_rate"].values.tolist(),
                        colorscale=[[0, Colors.SUCCESS], [0.5, Colors.WARNING], [1, Colors.CRITICAL]],
                        showscale=True,
                        colorbar=dict(
                            title=dict(text="Fraud %", font=dict(size=10, color=Colors.TEXT_MUTED)),
                            tickfont=dict(size=9, color=Colors.TEXT_MUTED),
                            thickness=12,
                            len=0.6,
                        ),
                        line=dict(width=1, color=Colors.BORDER_DEFAULT),
                        opacity=0.85,
                    ),
                ))
                fig_geo.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="'Inter', sans-serif", color=Colors.TEXT_SECONDARY, size=12),
                    margin=dict(l=0, r=0, t=32, b=0),
                    height=320,
                    title=dict(text="Geographic Fraud Map", font=dict(size=14, color=Colors.TEXT_PRIMARY)),
                    geo=dict(
                        projection_type="natural earth",
                        showland=True,
                        landcolor=Colors.BG_SECONDARY,
                        showocean=True,
                        oceancolor=Colors.BG_PRIMARY,
                        showlakes=False,
                        showcountries=True,
                        countrycolor=Colors.BORDER_DEFAULT,
                        coastlinecolor=Colors.BORDER_DEFAULT,
                        bgcolor="rgba(0,0,0,0)",
                        lonaxis=dict(showgrid=False),
                        lataxis=dict(showgrid=False),
                    ),
                )
                st.plotly_chart(fig_geo, use_container_width=True)
            else:
                st.info("No geographic data available for this tenant.")

        section_divider()

        # ══════════════════════════════════════════════════════════════════════
        # Section 2: Temporal Analysis
        # ══════════════════════════════════════════════════════════════════════
        _section_heading("1F4C5", "Temporal Analysis")

        col_hourly, col_channel, col_country = st.columns([2, 1, 1])

        with col_hourly:
            df["hour"] = df["timestamp"].dt.hour
            hourly_fraud = (
                df[df["is_fraud"] == 1]
                .groupby("hour")
                .size()
                .reindex(range(24), fill_value=0)
                .reset_index()
            )
            hourly_fraud.columns = ["hour", "fraud_count"]

            fig_hourly = bar_chart(
                x=hourly_fraud["hour"],
                y=hourly_fraud["fraud_count"],
                title="Hourly Fraud Timeline",
                color=Colors.CHART_4,
                height=320,
            )
            fig_hourly.update_layout(
                xaxis=dict(
                    title=dict(text="Hour of Day", font=dict(size=12, color=Colors.TEXT_MUTED)),
                    dtick=3,
                ),
                bargap=0.2,
            )
            st.plotly_chart(fig_hourly, use_container_width=True)

        with col_channel:
            channel_fraud = (
                df[df["is_fraud"] == 1]
                .groupby("channel")
                .size()
                .sort_values(ascending=True)
                .reset_index()
            )
            channel_fraud.columns = ["channel", "fraud_count"]

            fig_channel = horizontal_bar(
                categories=channel_fraud["channel"].tolist(),
                values=channel_fraud["fraud_count"].tolist(),
                title="Fraud by Channel",
                colors=[Colors.CHART_5, Colors.CHART_6, Colors.CHART_7, Colors.CHART_8],
                height=320,
            )
            st.plotly_chart(fig_channel, use_container_width=True)

        with col_country:
            country_fraud = (
                df[df["is_fraud"] == 1]
                .groupby("country_code")
                .size()
                .sort_values(ascending=True)
                .reset_index()
            )
            country_fraud.columns = ["country", "fraud_count"]

            fig_country = horizontal_bar(
                categories=country_fraud["country"].tolist(),
                values=country_fraud["fraud_count"].tolist(),
                title="Fraud by Country",
                colors=[Colors.CHART_1, Colors.CHART_2, Colors.CHART_3, Colors.CHART_4, Colors.CHART_5],
                height=320,
            )
            st.plotly_chart(fig_country, use_container_width=True)

        section_divider()

        # ══════════════════════════════════════════════════════════════════════
        # Section 3: Leaderboards
        # ══════════════════════════════════════════════════════════════════════
        _section_heading("1F3C6", "Risk Leaderboards")

        col_merch, col_cust = st.columns(2)

        with col_merch:
            rng = np.random.default_rng(42)
            merchants = [f"MERCH-{i:04d}" for i in rng.integers(1000, 9999, size=20)]
            merchant_stats = []
            for m in merchants:
                m_txns = df[df["transaction_id"].str.startswith(m[:5], na=False)]
                if m_txns.empty:
                    m_txns = df.sample(n=max(1, int(len(df) * rng.uniform(0.01, 0.08))), random_state=42)
                fraud_rate_val = m_txns["is_fraud"].mean() * 100
                avg_amt = m_txns["amount"].mean()
                merchant_stats.append({
                    "rank": 0,
                    "name": m,
                    "value": f"{fraud_rate_val:.1f}% fraud rate",
                    "trend": round(float(rng.uniform(-5, 15)), 1),
                    "badge": "HIGH RISK" if fraud_rate_val > 10 else None,
                    "badge_type": "critical" if fraud_rate_val > 10 else "info",
                })

            merchant_stats.sort(key=lambda x: float(x["value"].split("%")[0]), reverse=True)
            for i, entry in enumerate(merchant_stats[:5]):
                entry["rank"] = i + 1

            leader_board(
                merchant_stats[:5],
                title="Merchant Risk Leaderboard",
                rank_col="rank",
                name_col="name",
                value_col="value",
            )

        with col_cust:
            rng_cust = np.random.default_rng(99)
            customers = [f"CUST-{i:06d}" for i in rng_cust.integers(100000, 999999, size=20)]
            customer_stats = []
            for c in customers:
                c_txns = df.sample(n=max(1, int(len(df) * rng_cust.uniform(0.01, 0.06))), random_state=99)
                avg_risk_val = c_txns["risk_score"].mean()
                txn_count = len(c_txns)
                customer_stats.append({
                    "rank": 0,
                    "name": c,
                    "value": f"{avg_risk_val:.4f} risk score",
                    "trend": round(float(rng_cust.uniform(-3, 10)), 1),
                    "badge": "CRITICAL" if avg_risk_val > 0.6 else None,
                    "badge_type": "critical" if avg_risk_val > 0.6 else "info",
                })

            customer_stats.sort(key=lambda x: float(x["value"].split(" risk")[0]), reverse=True)
            for i, entry in enumerate(customer_stats[:5]):
                entry["rank"] = i + 1

            leader_board(
                customer_stats[:5],
                title="Customer Risk Leaderboard",
                rank_col="rank",
                name_col="name",
                value_col="value",
            )

        section_divider()

        # ══════════════════════════════════════════════════════════════════════
        # Section 4: Fraud Categories
        # ══════════════════════════════════════════════════════════════════════
        _section_heading("1F4C1", "Fraud Categories")

        col_treemap, col_dist = st.columns([3, 2])

        with col_treemap:
            category_data = (
                df[df["is_fraud"] == 1]
                .groupby(["transaction_type", "channel"])
                .agg(fraud_count=("is_fraud", "sum"), total_amount=("amount", "sum"))
                .reset_index()
            )

            labels = ["All Fraud"]
            parents = [""]
            values = [0]
            colors_list = [Colors.BG_CARD]

            for _, row in category_data.iterrows():
                labels.append(f"{row['transaction_type']}")
                parents.append("All Fraud")
                values.append(int(row["fraud_count"]))
                colors_list.append(Colors.CHART_PALETTE[len(labels) % len(Colors.CHART_PALETTE)])

            unique_types = category_data["transaction_type"].unique()
            for tt in unique_types:
                type_data = category_data[category_data["transaction_type"] == tt]
                for _, row in type_data.iterrows():
                    labels.append(f"{row['channel']}")
                    parents.append(f"{row['transaction_type']}")
                    values.append(int(row["fraud_count"]))
                    colors_list.append(Colors.CHART_PALETTE[len(labels) % len(Colors.CHART_PALETTE)])

            fig_treemap = treemap_chart(
                labels=labels,
                parents=parents,
                values=values,
                title="Fraud by Category",
                height=380,
                colors=colors_list,
            )
            st.plotly_chart(fig_treemap, use_container_width=True)

        with col_dist:
            fig_box = go.Figure()

            for i, decision in enumerate(["APPROVE", "REVIEW", "BLOCK"]):
                subset = df[df["decision"] == decision]["risk_score"]
                if subset.empty:
                    continue
                fig_box.add_trace(go.Box(
                    y=subset,
                    name=decision,
                    marker=dict(
                        color=[Colors.SUCCESS, Colors.WARNING, Colors.CRITICAL][i],
                        outliercolor=[Colors.SUCCESS, Colors.WARNING, Colors.CRITICAL][i],
                        size=4,
                    ),
                    line=dict(color=[Colors.SUCCESS, Colors.WARNING, Colors.CRITICAL][i], width=1.5),
                    boxpoints="outliers",
                ))

            fig_box.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="'Inter', sans-serif", color=Colors.TEXT_SECONDARY, size=12),
                margin=dict(l=0, r=0, t=32, b=0),
                height=380,
                title=dict(text="Risk Score by Decision", font=dict(size=14, color=Colors.TEXT_PRIMARY)),
                yaxis=dict(
                    showgrid=True, gridcolor=Colors.BORDER_SUBTLE, gridwidth=1,
                    showline=False, zeroline=False,
                    tickfont=dict(color=Colors.TEXT_MUTED, size=11),
                    title=dict(text="Risk Score", font=dict(size=12, color=Colors.TEXT_MUTED)),
                ),
                xaxis=dict(
                    showgrid=False, showline=False,
                    tickfont=dict(color=Colors.TEXT_MUTED, size=11),
                ),
                legend=dict(
                    font=dict(color=Colors.TEXT_SECONDARY, size=11),
                    bgcolor="rgba(0,0,0,0)",
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                ),
                hoverlabel=dict(
                    bgcolor=Colors.BG_ELEVATED, bordercolor=Colors.BORDER_DEFAULT,
                    font=dict(color=Colors.TEXT_PRIMARY, size=12),
                ),
                boxmode="group",
            )
            st.plotly_chart(fig_box, use_container_width=True)

        section_divider()

        # ══════════════════════════════════════════════════════════════════════
        # Section 5: Transaction Table
        # ══════════════════════════════════════════════════════════════════════
        _section_heading("1F4CB", "Transaction Explorer")

        metric_row([
            {"label": "Total", "value": f"{total_txn:,}", "icon": "BAR_CHART"},
            {"label": "Fraud", "value": f"{fraud_count:,}", "color": Colors.CRITICAL, "icon": "ALERT_TRIANGLE"},
            {"label": "Clean", "value": f"{total_txn - fraud_count:,}", "color": Colors.SUCCESS, "icon": "CHECK"},
        ])

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        table_df = df.sort_values("risk_score", ascending=False).copy()
        table_df["time_str"] = table_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        table_df["amount_str"] = table_df.apply(lambda r: currency_fmt(r["amount"], r.get("currency", "NGN")), axis=1)
        table_df["score_str"] = table_df["risk_score"].apply(lambda x: f"{x:.4f}")
        table_df["fraud_label"] = table_df["is_fraud"].map({0: "Legit", 1: "Fraud"})
        table_df["latency_str"] = table_df["latency_ms"].apply(lambda x: f"{x:.0f}ms")

        display_cols = ["transaction_id", "time_str", "amount_str", "channel", "country_code",
                        "fraud_label", "score_str", "decision", "latency_str"]

        data_table(
            df=table_df[display_cols],
            columns={
                "transaction_id": "TXN ID",
                "time_str": "Timestamp",
                "amount_str": "Amount",
                "channel": "Channel",
                "country_code": "Country",
                "fraud_label": "Status",
                "score_str": "Risk Score",
                "decision": "Decision",
                "latency_str": "Latency",
            },
            max_rows=25,
            status_col="fraud_label",
            sortable=True,
            striped=True,
        )
