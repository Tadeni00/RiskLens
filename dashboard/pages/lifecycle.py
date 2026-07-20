"""FraudTrap Dashboard — Model Lifecycle Page (Live-Wired)"""
import os
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import httpx


# ── API helper ────────────────────────────────────────────────────────────────

def _fetch_lifecycle(tenant_id: str) -> dict | None:
    """Fetch real lifecycle metrics from the API."""
    api_url = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
    try:
        resp = httpx.get(
            f"{api_url}/v1/lifecycle/{tenant_id}",
            timeout=3.0,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


# ── Fallback synthetic data ───────────────────────────────────────────────────

def _synthetic_lifecycle() -> dict:
    """Return static placeholder data when the API is unreachable."""
    return {
        "tenant_id": "demo",
        "current_phase": "SEMI_SUPERVISED",
        "model_version": "synthetic-fallback",
        "total_scored": 0,
        "fraud_labels": 0,
        "legit_labels": 0,
        "decisions": {},
        "phase_counts": {},
        "pr_auc": 0.0,
        "avg_latency_ms": 0.0,
        "p95_latency_ms": 0.0,
        "scoring_history": [],
        "transition_readiness": {
            "fraud_labels": {"current": 0, "target": 5000, "pct": 0},
            "pr_auc": {"current": 0.0, "target": 0.78, "pct": 0},
            "champion_challenger": {"current": "pending", "pct": 0},
        },
        "loaded_models": {
            "cold_start": False,
            "semi_supervised": False,
            "supervised": False,
            "simple_model": False,
        },
        "available_tenants": [],
    }


# ── Phase config ──────────────────────────────────────────────────────────────

PHASE_ORDER = ["UNSUPERVISED", "SEMI_SUPERVISED", "SUPERVISED"]
PHASE_LABELS = {
    "UNSUPERVISED": "Phase 1\nUnsupervised",
    "SEMI_SUPERVISED": "Phase 2\nSemi-supervised",
    "SUPERVISED": "Phase 3\nSupervised",
}
PHASE_COLORS = {
    "UNSUPERVISED": "#6366F1",
    "SEMI_SUPERVISED": "#F59E0B",
    "SUPERVISED": "#22C55E",
}


def _phase_index(phase: str) -> int:
    try:
        return PHASE_ORDER.index(phase)
    except ValueError:
        return 0


# ── Render ────────────────────────────────────────────────────────────────────

def render(tenant_id: str):
    st.title("🔄 Model Lifecycle")
    st.caption("Phase progression, retraining history, and champion/challenger status.")

    # Fetch live data
    data = _fetch_lifecycle(tenant_id)
    is_live = data is not None
    if not is_live:
        data = _synthetic_lifecycle()

    # Live / fallback indicator
    if is_live:
        st.markdown(
            '<div style="display:inline-block;background:#0F2A1A;border:1px solid #1D9E75;'
            'border-radius:6px;padding:4px 14px;font-size:12px;color:#22C55E;font-weight:600;'
            'margin-bottom:1rem">● LIVE from API</div>',
            unsafe_allow_html=True,
        )
    else:
        st.warning("⚠️ Could not reach the API — showing placeholder data.")

    # ── KPI row ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Scored", f"{data['total_scored']:,}")
    c2.metric("Fraud Labels", f"{data['fraud_labels']:,}")
    c3.metric("Live PR-AUC", f"{data['pr_auc']:.4f}")
    c4.metric("P95 Latency", f"{data['p95_latency_ms']:.1f}ms")

    st.markdown("---")

    # ── Phase progression cards ───────────────────────────────────────────────
    st.subheader("Cold-Start Phase Progression")
    current = data["current_phase"]
    current_idx = _phase_index(current)

    cols = st.columns(3)
    for i, (col, phase_key) in enumerate(zip(cols, PHASE_ORDER)):
        label = PHASE_LABELS[phase_key].replace("\n", " · ")
        color = PHASE_COLORS[phase_key]
        is_active = i == current_idx
        is_complete = i < current_idx
        is_pending = i > current_idx

        if is_active:
            status = "Active"
            border = f"2px solid {color}"
            status_color = color
        elif is_complete:
            status = "Complete"
            border = f"1px solid {color}"
            status_color = color
        else:
            status = "Pending"
            border = "1px solid #333"
            status_color = "#555"
            color = "#555"

        # Build detail line from real data
        phase_count = data["phase_counts"].get(phase_key, 0)
        detail = f"<div style='font-size:11px;color:#888;margin-top:4px'>{phase_count:,} transactions scored</div>"

        with col:
            st.markdown(
                f'<div style="border:{border};border-radius:8px;padding:1rem;text-align:center">'
                f'<div style="font-size:12px;font-weight:600;color:{color}">{label}</div>'
                f'<div style="font-size:11px;color:#888;margin-top:4px">Status: <b style="color:{status_color}">{status}</b></div>'
                f'{detail}'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Model info ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Active Model")
    m1, m2, m3 = st.columns(3)
    m1.markdown(f"**Phase:** `{data['current_phase']}`")
    m2.markdown(f"**Model Version:** `{data['model_version']}`")
    loaded = data.get("loaded_models", {})
    active_models = [k for k, v in loaded.items() if v]
    m3.markdown(f"**Loaded:** {', '.join(active_models) if active_models else 'none'}")

    if data.get("available_tenants"):
        st.caption(f"Tenants with trained models: {', '.join(data['available_tenants'])}")

    # ── Phase 2 → 3 Transition Readiness ──────────────────────────────────────
    st.markdown("---")
    st.subheader("Phase 2 → 3 Transition Readiness")

    readiness = data["transition_readiness"]

    # Fraud labels gate
    fl = readiness["fraud_labels"]
    fl_pct = fl["pct"]
    fl_done = fl_pct >= 100
    fl_current = fl["current"]
    fl_target = fl["target"]
    fl_label = "✅ Ready" if fl_done else f"{fl_current:,} collected ({fl_pct:.1f}%)"
    st.markdown(f"**Fraud labels (need {fl_target:,})** — {fl_label}")
    st.progress(min(int(fl_pct), 100))

    # PR-AUC gate
    pa = readiness["pr_auc"]
    pa_pct = pa["pct"]
    pa_done = pa_pct >= 100
    pa_current = pa["current"]
    pa_target = pa["target"]
    pa_label = "✅ Ready" if pa_done else f"{pa_current:.4f} ({pa_pct:.1f}%)"
    st.markdown(f"**PR-AUC (need {pa_target})** — {pa_label}")
    st.progress(min(int(pa_pct), 100))

    # Champion/challenger gate
    cc = readiness["champion_challenger"]
    cc_pct = cc["pct"]
    cc_done = cc_pct >= 100
    cc_current = cc["current"]
    cc_label = "✅ Deployed & validated" if cc_done else f"{cc_current} ({cc_pct:.0f}%)"
    st.markdown(f"**Champion/challenger gate** — {cc_label}")
    st.progress(min(int(cc_pct), 100))

    # ── Decision distribution from live stream ────────────────────────────────
    st.markdown("---")
    st.subheader("Live Decision Distribution")

    decisions = data.get("decisions", {})
    if decisions:
        dec_colors = {"APPROVE": "#22C55E", "REVIEW": "#F59E0B", "BLOCK": "#EF4444"}
        labels = list(decisions.keys())
        values = list(decisions.values())
        colors = [dec_colors.get(l, "#888") for l in labels]

        fig_dec = go.Figure(go.Pie(
            labels=labels,
            values=values,
            hole=0.55,
            marker=dict(colors=colors),
            textinfo="label+percent",
        ))
        fig_dec.update_layout(
            height=280,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig_dec, use_container_width=True)
    else:
        st.info("No decisions recorded yet.")

    # ── Scoring history timeline ──────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Scoring Volume Over Time")

    history = data.get("scoring_history", [])
    if history:
        df_hist = pd.DataFrame(history)
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Bar(
            x=df_hist["date"],
            y=df_hist["transactions"],
            name="Transactions",
            marker_color="#3B82F6",
            opacity=0.8,
        ))
        if "fraud_labels" in df_hist.columns:
            fig_hist.add_trace(go.Bar(
                x=df_hist["date"],
                y=df_hist["fraud_labels"],
                name="Fraud Labels",
                marker_color="#EF4444",
                opacity=0.9,
            ))
        fig_hist.update_layout(
            height=280,
            barmode="overlay",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis_title="Count",
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", y=1.08),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # Average score trend
        if "avg_score" in df_hist.columns:
            st.subheader("Average Risk Score Trend")
            fig_score = go.Figure(go.Scatter(
                x=df_hist["date"],
                y=df_hist["avg_score"],
                mode="lines+markers+text",
                text=df_hist["avg_score"].round(4),
                textposition="top center",
                line=dict(color="#8B5CF6", width=2.5),
                marker=dict(size=8, color="#8B5CF6"),
            ))
            fig_score.add_hline(
                y=0.40, line_dash="dash", line_color="#F59E0B",
                annotation_text="Review threshold (0.40)",
            )
            fig_score.add_hline(
                y=0.85, line_dash="dash", line_color="#EF4444",
                annotation_text="Block threshold (0.85)",
            )
            fig_score.update_layout(
                height=260,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis_title="Avg Risk Score",
                yaxis_range=[0, 1.0],
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig_score, use_container_width=True)
    else:
        st.info("No scoring history available yet — waiting for streamed data.")

    # ── PR-AUC gauge ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Live PR-AUC")

    pr_auc_val = data["pr_auc"]
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pr_auc_val,
        delta={"reference": 0.78, "increasing": {"color": "#22C55E"}, "decreasing": {"color": "#EF4444"}},
        gauge={
            "axis": {"range": [0, 1], "tickwidth": 1},
            "bar": {"color": "#3B82F6"},
            "steps": [
                {"range": [0, 0.50], "color": "#2D1515"},
                {"range": [0.50, 0.78], "color": "#3D2F0A"},
                {"range": [0.78, 1.0], "color": "#0F2A1A"},
            ],
            "threshold": {
                "line": {"color": "#22C55E", "width": 3},
                "thickness": 0.8,
                "value": 0.78,
            },
        },
        title={"text": "PR-AUC vs Phase 3 Gate (0.78)"},
    ))
    fig_gauge.update_layout(
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#FAFAFA"},
        margin=dict(l=40, r=40, t=60, b=20),
    )
    st.plotly_chart(fig_gauge, use_container_width=True)
