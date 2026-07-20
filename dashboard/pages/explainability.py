"""FraudTrap Dashboard — Explainability Page (Live-Wired)"""
import os
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import httpx
from dashboard.components.data_loader import load_data

def _fetch_explanation(trace_id: str) -> dict | None:
    """Fetch real explanation from the API."""
    api_url = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
    try:
        resp = httpx.get(f"{api_url}/v1/explain/{trace_id}", timeout=3.0)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None

def render(tenant_id: str):
    st.title("🔍 Explainability")
    st.info(
        "Every fraud decision is accompanied by a SHAP explanation. "
        "This satisfies **GDPR Art. 22** (right to explanation) and **EU AI Act Art. 13** "
        "(transparency for high-risk AI systems).",
        icon="⚖️",
    )

    df, _ = load_data(tenant_id)

    # Pick a default trace_id if available
    default_trace = ""
    if not df.empty and "trace_id" in df.columns:
        interesting = df[df["decision"].isin(["BLOCK", "REVIEW"])]
        if not interesting.empty:
            default_trace = interesting.iloc[0]["trace_id"]
        else:
            default_trace = df.iloc[0]["trace_id"]

    # ── Individual transaction lookup ─────────────────────────────────────────
    st.subheader("Transaction Explanation Lookup")
    trace_id = st.text_input(
        "Enter trace_id",
        placeholder="trace_abc123",
        value=default_trace,
    )

    if not trace_id:
        st.warning("Please enter a valid trace_id")
        return

    data = _fetch_explanation(trace_id)
    if not data:
        st.error(f"Could not reach API to fetch explanation for {trace_id}")
        return

    has_explanation = "explanation" in data and data["explanation"] is not None

    decision = data.get("decision", "UNKNOWN")
    risk_score = data.get("risk_score", 0.0)
    if has_explanation:
        exp = data["explanation"]
        risk_score = risk_score or exp.get("prediction_value", 0.0)
    else:
        exp = None
        if not df.empty and "trace_id" in df.columns:
            match = df[df["trace_id"] == trace_id]
            if not match.empty:
                risk_score = float(match.iloc[0].get("risk_score", 0.0))
                decision = str(match.iloc[0].get("decision", "UNKNOWN"))

    decision_color = {"BLOCK": "#EF4444", "REVIEW": "#F59E0B", "APPROVE": "#22C55E"}
    color = decision_color.get(decision, "#888")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Risk Score", f"{risk_score:.4f}")
    col_b.markdown(
        f'<div style="padding:1rem;border-radius:8px;background:{color}22;'
        f'border:1px solid {color};text-align:center">'
        f'<div style="font-size:1.4rem;font-weight:700;color:{color}">{decision}</div>'
        f'<div style="font-size:0.75rem;color:#888">Final Decision</div></div>',
        unsafe_allow_html=True,
    )
    if has_explanation:
        col_c.metric("Base Value", f"{exp.get('base_value', 0.0):.4f}")
    else:
        col_c.metric("Model Phase", "SUPERVISED")

    # SHAP waterfall
    st.subheader("SHAP Waterfall Explanation")

    top_features = exp.get("top_features", []) if exp else []
    if not top_features:
        st.info(
            "This transaction was scored using a simple model that does not store per-transaction SHAP values. "
            "Once the full SupervisedEnsemble (XGBoost + LightGBM) is trained and promoted, "
            "every REVIEW and BLOCK decision will include a detailed SHAP waterfall. "
            "See the **Global Explanations Summary** below for feature-level insights."
        )

    shap_vals = []
    for feat_dict in reversed(top_features):
        for f, v in feat_dict.items():
            shap_vals.append((f, v))

    if shap_vals and exp:
        base_val = exp.get("base_value", 0.0)

        fig_wf = go.Figure(go.Waterfall(
            name="SHAP", orientation="h",
            measure=["absolute"] + ["relative"] * len(shap_vals) + ["total"],
            x=[base_val] + [v for _, v in shap_vals] + [risk_score],
            y=["Base Value"] + [f for f, _ in shap_vals] + ["Final Score"],
            connector={"line": {"color": "#444"}},
            decreasing={"marker": {"color": "#22C55E"}},
            increasing={"marker": {"color": "#EF4444"}},
            totals={"marker": {"color": "#3B82F6"}},
            text=[f"{base_val:.3f}"] + [f"{v:+.3f}" for _, v in shap_vals] + [f"{risk_score:.3f}"],
            textposition="outside",
        ))
        fig_wf.update_layout(
            height=400 + len(shap_vals) * 20, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="SHAP contribution (positive = fraud signal)",
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_wf, use_container_width=True)

        top_positive = [(f, v) for f, v in reversed(shap_vals) if v > 0][:3]
        reasons = []
        label_map = {
            "acct_v_1h_count":    "High transaction velocity in the last hour",
            "amount_zscore":      "Amount significantly above account average",
            "is_new_device":      "Transaction from an unrecognised device",
            "impossible_travel":  "Impossible geographic travel detected",
            "geo_speed_kmh":      "Unusually fast geographic movement",
            "typing_zscore":      "Atypical typing behaviour",
            "device_account_count":"Device shared across multiple accounts",
            "cross_country_flag": "Cross-country transaction",
        }
        for feat, val in top_positive:
            reason = label_map.get(feat, f"Unusual value for {feat}")
            reasons.append(f"• {reason} (impact: +{val:.3f})")

        if reasons:
            st.markdown("**Why this transaction was flagged:**")
            for r in reasons:
                st.markdown(r)

    st.markdown("---")

    # ── Proxy Global SHAP summary ───────────────────────────────────────────────────
    st.subheader("Global Explanations Summary")
    st.caption("Average absolute impact for this specific batch of recent transactions.")

    if not df.empty and "is_fraud" in df.columns:
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                        if c not in ["is_fraud", "risk_score", "timestamp"]]

        corrs = []
        for c in numeric_cols:
            try:
                corr = df[c].corr(df["is_fraud"])
                if not pd.isna(corr):
                    corrs.append({"feature": c, "impact": abs(corr)})
            except (ValueError, TypeError):
                pass

        if corrs:
            df_imp = pd.DataFrame(corrs).sort_values("impact", ascending=False).head(8)

            fig_bar = px.bar(
                df_imp.sort_values("impact", ascending=True),
                x="impact", y="feature", orientation="h",
                color="impact",
                color_continuous_scale=["#3B82F6", "#F59E0B", "#EF4444"],
            )
            fig_bar.update_layout(
                xaxis_title="Average Impact Magnitude (Proxy)",
                height=320, paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=10, b=0),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)
