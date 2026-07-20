"""FraudTrap Dashboard — Drift Detection Page (Live-Wired)"""
import os
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import httpx
from dashboard.components.data_loader import make_drift_data


def _fetch_drift(tenant_id: str) -> dict | None:
    """Fetch real drift metrics from the API."""
    api_url = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
    try:
        resp = httpx.get(f"{api_url}/v1/drift/{tenant_id}", timeout=3.0)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def render(tenant_id: str):
    st.title("⚠️ Live Drift Detection")
    st.caption(
        "Population Stability Index (PSI) monitors feature distributions over time. "
        "PSI > 0.20 indicates significant drift."
    )

    data = _fetch_drift(tenant_id)
    is_live = data is not None and data.get("status") != "insufficient_data" and data.get("metrics")

    if is_live:
        drift = data["metrics"]
        st.markdown(
            f'<div style="display:inline-block;background:#0F2A1A;border:1px solid #1D9E75;'
            f'border-radius:6px;padding:4px 14px;font-size:12px;color:#22C55E;font-weight:600;'
            f'margin-bottom:1rem">● LIVE from API (Comparing {data["n_baseline"]} baseline vs {data["n_current"]} current txns)</div>',
            unsafe_allow_html=True,
        )
    else:
        st.warning("Could not reach drift API — showing simulated drift data for demonstration.")
        drift = make_drift_data()

    # PSI legend
    col1, col2, col3 = st.columns(3)
    col1.markdown('<div style="background:#0F2A1A;padding:.6rem 1rem;border-radius:6px;'
                  'border-left:3px solid #22C55E"><b style="color:#22C55E">PSI &lt; 0.10</b>'
                  '<br><span style="color:#888;font-size:12px">Stable — no action needed</span></div>',
                  unsafe_allow_html=True)
    col2.markdown('<div style="background:#3D2F0A;padding:.6rem 1rem;border-radius:6px;'
                  'border-left:3px solid #F59E0B"><b style="color:#F59E0B">PSI 0.10–0.20</b>'
                  '<br><span style="color:#888;font-size:12px">Moderate drift — monitor closely</span></div>',
                  unsafe_allow_html=True)
    col3.markdown('<div style="background:#3D1515;padding:.6rem 1rem;border-radius:6px;'
                  'border-left:3px solid #EF4444"><b style="color:#EF4444">PSI &gt; 0.20</b>'
                  '<br><span style="color:#888;font-size:12px">Significant drift — retrain triggered</span></div>',
                  unsafe_allow_html=True)

    st.markdown("---")

    # PSI bar chart
    st.subheader("Live Feature PSI Scores")
    psi_df = pd.DataFrame([
        {"feature": k, "psi": v["psi"],
         "status": "critical" if v["psi"] > 0.20 else "warning" if v["psi"] > 0.10 else "ok"}
        for k, v in drift.items()
    ]).sort_values("psi", ascending=True)

    color_map = {"critical": "#EF4444", "warning": "#F59E0B", "ok": "#22C55E"}
    colors = [color_map[s] for s in psi_df["status"]]

    fig = go.Figure(go.Bar(
        x=psi_df["psi"], y=psi_df["feature"],
        orientation="h", marker_color=colors,
        text=psi_df["psi"].round(3), textposition="outside",
    ))
    fig.add_vline(x=0.20, line_dash="dash", line_color="#EF4444",
                  annotation_text="Retrain threshold")
    fig.add_vline(x=0.10, line_dash="dot", line_color="#F59E0B",
                  annotation_text="Warning threshold")
    fig.update_layout(
        height=320, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="PSI",
        xaxis=dict(range=[0, max(0.25, psi_df["psi"].max() * 1.2)]),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Alert
    critical = [k for k, v in drift.items() if v["psi"] > 0.20]
    if critical:
        st.error(
            f"**Drift Alert:** {len(critical)} feature(s) exceed the PSI threshold: "
            f"{', '.join(critical)}.",
            icon="🚨",
        )
    else:
        st.success("All monitored features within acceptable drift bounds.", icon="✅")

    # Distribution comparison
    st.markdown("---")
    st.subheader("Baseline vs Current Distribution (Simulated Normal Fit)")
    feat_sel = st.selectbox("Feature", list(drift.keys()))
    d = drift[feat_sel]

    rng = np.random.default_rng(42)
    baseline_samples = rng.normal(d["baseline_mean"], max(0.1, abs(d["baseline_mean"]) * 0.2 + 1.0), 1000)
    current_samples  = rng.normal(d["current_mean"],  max(0.1, abs(d["current_mean"]) * 0.2 + 1.0), 1000)

    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(
        x=baseline_samples, name="Baseline (older)", nbinsx=40,
        marker_color="#3B82F6", opacity=0.6, histnorm="probability density",
    ))
    fig2.add_trace(go.Histogram(
        x=current_samples, name="Current (newer)", nbinsx=40,
        marker_color="#EF4444", opacity=0.6, histnorm="probability density",
    ))
    fig2.update_layout(
        barmode="overlay", height=300,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title=feat_sel, margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(f"Live PSI for **{feat_sel}**: `{d['psi']:.4f}` | "
               f"Baseline mean: `{d['baseline_mean']:.3f}` | "
               f"Current mean: `{d['current_mean']:.3f}`")
