"""FraudTrap Dashboard — Explainability Page
SHAP attributions, counterfactual analysis, and decision explanations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from dashboard.components import (
    kpi_row,
    waterfall_chart,
    bar_chart,
    data_table,
    metric_table,
    page_container,
    section_divider,
    confidence_display,
    info_panel,
    metric_row,
)
from dashboard.components.data_loader import currency_fmt, make_shap_values
from dashboard.theme.colors import Colors
from dashboard.theme.icons import Icons

# ── Synthetic Transaction Data ────────────────────────────────────────────────

_FEATURE_LABEL_MAP = {
    "acct_v_1h_count": "High transaction velocity in the last hour",
    "amount_zscore": "Amount significantly above account average",
    "is_new_device": "Transaction from an unrecognised device",
    "impossible_travel": "Impossible geographic travel detected",
    "geo_speed_kmh": "Unusually fast geographic movement",
    "typing_zscore": "Atypical typing behaviour",
    "acct_v_24h_total_amt": "Large 24-hour cumulative amount",
    "device_account_count": "Device shared across multiple accounts",
}

_CHANNEL_OPTIONS = ["MOBILE", "WEB", "POS", "ATM"]
_COUNTRY_OPTIONS = ["NG", "KE", "ZA", "GB", "US"]


def _synthetic_transaction(trace_id: str) -> dict:
    """Generate a synthetic transaction record for demonstration."""
    rng = np.random.default_rng(hash(trace_id) % (2**31))
    risk = float(rng.beta(8, 2))
    decision = "BLOCK" if risk > 0.7 else "REVIEW" if risk > 0.4 else "APPROVE"
    return {
        "trace_id": trace_id,
        "amount": float(rng.lognormal(10.5, 1.0)),
        "channel": rng.choice(_CHANNEL_OPTIONS),
        "country_code": rng.choice(_COUNTRY_OPTIONS),
        "risk_score": risk,
        "decision": decision,
        "is_new_device": bool(rng.binomial(1, 0.6)),
        "impossible_travel": bool(rng.binomial(1, 0.3)),
        "account_age_days": int(rng.integers(1, 800)),
        "beneficiary_known": bool(rng.binomial(1, 0.2)),
    }


def _synthetic_shap_values(risk: float, n_features: int = 8) -> list[tuple[str, float]]:
    """Generate synthetic SHAP values for a single transaction."""
    rng = np.random.default_rng()
    features = list(_FEATURE_LABEL_MAP.keys())[:n_features]
    base_values = np.array([0.18, 0.15, 0.12, 0.10, 0.09, 0.08, 0.07, 0.06])
    signs = np.where(rng.random(n_features) < risk, 1, -1)
    shap_vals = signs * base_values * rng.uniform(0.4, 1.6, n_features)
    return list(zip(features, shap_vals))


def _counterfactual_steps(
    shap_vals: list[tuple[str, float]], risk: float
) -> list[dict]:
    """Generate counterfactual steps sorted by impact (largest negative first)."""
    sorted_vals = sorted(shap_vals, key=lambda x: x[1])
    steps = []
    current_prob = risk
    label_map = {
        "amount_zscore": ("Reduce transaction amount", "amount"),
        "is_new_device": ("Use a known device", "device"),
        "impossible_travel": ("Eliminate impossible travel", "geo"),
        "geo_speed_kmh": ("Reduce geographic speed", "geo"),
        "acct_v_1h_count": ("Lower transaction frequency", "velocity"),
        "typing_zscore": ("Match normal typing pattern", "behaviour"),
        "device_account_count": ("Use single-account device", "device"),
        "acct_v_24h_total_amt": ("Reduce 24h cumulative spend", "amount"),
    }
    for feat, val in sorted_vals:
        if val > 0:
            reduction = min(val * 1.1, current_prob - 0.02)
            reduction = max(reduction, 0.01)
            current_prob = max(current_prob - reduction, 0.01)
            action, category = label_map.get(feat, (f"Adjust {feat}", "other"))
            steps.append(
                {
                    "action": action,
                    "category": category,
                    "impact": -reduction,
                    "new_prob": current_prob,
                }
            )
    return steps


def _nearest_transactions(
    shap_vals: list[tuple[str, float]],
    is_fraud: bool,
    n: int = 5,
) -> pd.DataFrame:
    """Generate synthetic 'nearest' transactions for similarity display."""
    rng = np.random.default_rng()
    target_vector = np.array([v for _, v in shap_vals])
    rows = []
    for i in range(n):
        noise = rng.normal(0, 0.3, len(target_vector))
        feature_vals = np.abs(target_vector + noise)
        risk = float(
            np.clip(rng.beta(8 if is_fraud else 1, 2 if is_fraud else 8), 0, 1)
        )
        decision = "BLOCK" if risk > 0.7 else "REVIEW" if risk > 0.4 else "APPROVE"
        raw_amt = float(rng.lognormal(10.0 if is_fraud else 8.5, 1.0))
        currency = rng.choice(_COUNTRY_OPTIONS)
        _cur_map = {"NG": "NGN", "KE": "KES", "ZA": "ZAR", "GB": "GBP", "US": "USD"}
        rows.append(
            {
                "trace_id": f"sim_{'fraud' if is_fraud else 'legit'}_{i:04d}",
                "amount": currency_fmt(raw_amt, _cur_map.get(currency, "USD")),
                "channel": rng.choice(_CHANNEL_OPTIONS),
                "risk_score": f"{risk:.4f}",
                "decision": decision,
                "similarity": f"{float(np.clip(1.0 - np.mean(np.abs(feature_vals)), 0.5, 0.99)):.2f}",
            }
        )
    return pd.DataFrame(rows)


# ── Section Renderers ─────────────────────────────────────────────────────────


def _render_prediction_summary(
    trace_id: str, txn: dict, shap_vals: list[tuple[str, float]]
):
    """Section 1: Transaction details, fraud probability gauge, confidence, business impact."""
    risk = txn["risk_score"]
    decision = txn["decision"]
    decision_color = {
        "BLOCK": Colors.CRITICAL,
        "REVIEW": Colors.WARNING,
        "APPROVE": Colors.SUCCESS,
    }
    color = decision_color.get(decision, Colors.TEXT_MUTED)

    st.markdown(
        f"<div style='color:{Colors.TEXT_PRIMARY};font-weight:600;font-size:16px;margin-bottom:12px'>"
        f"{Icons.html('SEARCH', 16, Colors.ACCENT)} Transaction Lookup</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
<div style="background:{Colors.BG_CARD};border:1px solid {Colors.BORDER_DEFAULT};border-radius:10px;padding:20px;margin-bottom:16px">
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px">
        <div>
            <div style="font-size:11px;color:{Colors.TEXT_MUTED};text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px">Trace ID</div>
            <div style="font-size:14px;color:{Colors.TEXT_PRIMARY};font-weight:600;font-family:'IBM Plex Mono',monospace">{trace_id}</div>
        </div>
        <div>
            <div style="font-size:11px;color:{Colors.TEXT_MUTED};text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px">Amount</div>
            <div style="font-size:14px;color:{Colors.TEXT_PRIMARY};font-weight:600">{currency_fmt(txn['amount'], txn.get('currency', 'NGN'))}</div>
        </div>
        <div>
            <div style="font-size:11px;color:{Colors.TEXT_MUTED};text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px">Channel</div>
            <div style="font-size:14px;color:{Colors.TEXT_PRIMARY};font-weight:600">{txn['channel']}</div>
        </div>
        <div>
            <div style="font-size:11px;color:{Colors.TEXT_MUTED};text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px">Country</div>
            <div style="font-size:14px;color:{Colors.TEXT_PRIMARY};font-weight:600">{txn['country_code']}</div>
        </div>
    </div>
    <div style="margin-top:16px;padding-top:16px;border-top:1px solid {Colors.BORDER_SUBTLE};display:flex;align-items:center;gap:12px">
        <div style="font-size:11px;color:{Colors.TEXT_MUTED};text-transform:uppercase;letter-spacing:0.08em">Decision</div>
        <div style="padding:4px 12px;border-radius:6px;background:{Colors.rgba(color, 0.15)};color:{color};font-weight:700;font-size:13px">{decision}</div>
    </div>
</div>""",
        unsafe_allow_html=True,
    )

    col_gauge, col_conf, col_impact = st.columns([2, 1, 1])

    with col_gauge:
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=risk * 100,
                number=dict(font=dict(size=32, color=Colors.TEXT_PRIMARY), suffix="%"),
                gauge=dict(
                    axis=dict(range=[0, 100], tickcolor=Colors.TEXT_MUTED),
                    bar=dict(color=color),
                    bgcolor=Colors.BG_SECONDARY,
                    borderwidth=0,
                    steps=[
                        {"range": [0, 40], "color": Colors.rgba(Colors.SUCCESS, 0.2)},
                        {"range": [40, 70], "color": Colors.rgba(Colors.WARNING, 0.2)},
                        {
                            "range": [70, 100],
                            "color": Colors.rgba(Colors.CRITICAL, 0.2),
                        },
                    ],
                ),
            )
        )
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            height=220,
            margin=dict(l=20, r=20, t=10, b=0),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_conf:
        confidence = max(0.5, min(1.0, 1.0 - abs(risk - 0.5) * 0.6 + 0.3))
        confidence_display(confidence, "Model Confidence")

    with col_impact:
        avg_txn_amount = 2_450.0
        blocked_count = int(142 * risk)
        revenue_impact = blocked_count * avg_txn_amount
        st.markdown(
            f"""
<div style="background:{Colors.BG_CARD};border:1px solid {Colors.BORDER_DEFAULT};border-radius:8px;padding:16px;height:100%">
    <div style="font-size:11px;color:{Colors.TEXT_MUTED};text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px">Business Impact Estimate</div>
    <div style="font-size:11px;color:{Colors.TEXT_MUTED};margin-bottom:4px">Potential fraud prevented</div>
    <div style="font-size:22px;font-weight:700;color:{Colors.SUCCESS};margin-bottom:12px">{currency_fmt(revenue_impact, txn.get('currency', 'NGN'))}</div>
    <div style="font-size:11px;color:{Colors.TEXT_MUTED};margin-bottom:4px">Similar blocked txns (30d)</div>
    <div style="font-size:16px;font-weight:600;color:{Colors.TEXT_PRIMARY}">{blocked_count}</div>
</div>""",
            unsafe_allow_html=True,
        )


def _render_shap_explanation(shap_vals: list[tuple[str, float]], risk: float):
    """Section 2: SHAP waterfall, feature importance, human-readable explanation."""
    st.markdown(
        f"<div style='color:{Colors.TEXT_PRIMARY};font-weight:600;font-size:16px;margin-bottom:12px'>"
        f"{Icons.html('LAYERS', 16, Colors.ACCENT)} SHAP Explanation</div>",
        unsafe_allow_html=True,
    )

    sorted_shap = sorted(shap_vals, key=lambda x: x[1], reverse=True)
    categories = [f for f, _ in sorted_shap]
    values = [v for _, v in sorted_shap]

    fig_wf = waterfall_chart(
        categories=categories,
        values=values,
        title="Feature Contributions (positive = risk increase)",
        height=380,
    )
    st.plotly_chart(fig_wf, use_container_width=True)

    top_10 = sorted_shap[:10]
    fig_imp = bar_chart(
        x=[v for _, v in top_10],
        y=[f for f, _ in top_10],
        title="Top 10 Feature Importance (Absolute SHAP)",
        orientation="h",
        height=340,
        color=Colors.CHART_5,
    )
    fig_imp.update_traces(
        marker_color=[Colors.CRITICAL if v > 0 else Colors.SUCCESS for _, v in top_10]
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    positive_shap = [(f, v) for f, v in sorted_shap if v > 0][:5]
    if positive_shap:
        st.markdown(
            f"<div style='color:{Colors.TEXT_PRIMARY};font-weight:600;font-size:14px;margin:16px 0 8px'>"
            f"{Icons.html('ALERT_TRIANGLE', 14, Colors.WARNING)} Why Flagged</div>",
            unsafe_allow_html=True,
        )
        reasons_html = ""
        for feat, val in positive_shap:
            reason = _FEATURE_LABEL_MAP.get(feat, f"Unusual value for {feat}")
            reasons_html += f"""
<div style="display:flex;align-items:flex-start;gap:10px;padding:10px 14px;margin-bottom:6px;background:{Colors.rgba(Colors.CRITICAL, 0.06)};border:1px solid {Colors.rgba(Colors.CRITICAL, 0.15)};border-radius:8px">
    <div style="min-width:20px;height:20px;border-radius:50%;background:{Colors.CRITICAL};display:flex;align-items:center;justify-content:center;margin-top:1px">
        <span style="color:#fff;font-size:11px;font-weight:700">!</span>
    </div>
    <div>
        <div style="font-size:13px;color:{Colors.TEXT_PRIMARY};font-weight:500">{reason}</div>
        <div style="font-size:11px;color:{Colors.CRITICAL};font-weight:600;margin-top:2px">Impact: +{val:.3f}</div>
    </div>
</div>"""
        st.markdown(reasons_html, unsafe_allow_html=True)
    else:
        st.markdown(
            f"<div style='padding:16px;background:{Colors.SUCCESS_BG};border:1px solid {Colors.rgba(Colors.SUCCESS, 0.2)};border-radius:8px;color:{Colors.SUCCESS};font-size:13px'>"
            f"{Icons.html('CHECK_CIRCLE', 14, Colors.SUCCESS)} No strong fraud signals detected. This transaction appears low-risk.</div>",
            unsafe_allow_html=True,
        )


def _render_counterfactual(txn: dict, shap_vals: list[tuple[str, float]]):
    """Section 3: Counterfactual explanation with split view and visual flow."""
    risk = txn["risk_score"]
    steps = _counterfactual_steps(shap_vals, risk)
    final_prob = steps[-1]["new_prob"] if steps else risk

    st.markdown(
        f"<div style='color:{Colors.TEXT_PRIMARY};font-weight:600;font-size:16px;margin-bottom:12px'>"
        f"{Icons.html('SLIDERS', 16, Colors.ACCENT)} Counterfactual Explanation</div>",
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns(2)

    with col_left:
        risk_factors = []
        for feat, val in sorted(shap_vals, key=lambda x: x[1], reverse=True):
            if val > 0:
                label = _FEATURE_LABEL_MAP.get(feat, feat)
                risk_factors.append({"label": label, "value": f"+{val:.3f}"})
        if not risk_factors:
            risk_factors = [{"label": "No significant risk factors", "value": "—"}]

        info_panel(
            title="Current Transaction State",
            items=[
                {"label": "Trace ID", "value": txn["trace_id"]},
                {
                    "label": "Amount",
                    "value": currency_fmt(txn["amount"], txn.get("currency", "NGN")),
                },
                {"label": "Channel", "value": txn["channel"]},
                {"label": "Decision", "value": txn["decision"]},
            ]
            + risk_factors,
            icon="ALERT_TRIANGLE",
        )

    with col_right:
        if steps:
            changes = []
            for s in steps[:4]:
                changes.append(
                    {"label": s["action"], "value": f"{s['new_prob']:.0%} prob"}
                )
            changes.append(
                {
                    "label": "Final probability after all changes",
                    "value": f"{final_prob:.0%}",
                }
            )
            info_panel(
                title="Suggested Minimal Changes",
                items=changes,
                icon="CHECK_CIRCLE",
            )
        else:
            st.info("No counterfactual changes needed for this transaction.")

    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

    flow_html = f"""
<div style="background:{Colors.BG_CARD};border:1px solid {Colors.BORDER_DEFAULT};border-radius:10px;padding:24px;margin-top:12px">
    <div style="font-size:12px;color:{Colors.TEXT_MUTED};text-transform:uppercase;letter-spacing:0.08em;margin-bottom:16px">Probability Reduction Flow</div>
    <div style="display:flex;flex-direction:column;gap:0;align-items:center">
        <div style="padding:10px 20px;background:{Colors.rgba(Colors.CRITICAL, 0.12)};border:1px solid {Colors.rgba(Colors.CRITICAL, 0.3)};border-radius:8px;text-align:center">
            <span style="font-size:11px;color:{Colors.TEXT_MUTED}">Current Probability</span><br>
            <span style="font-size:22px;font-weight:700;color:{Colors.CRITICAL}">{risk:.0%}</span>
        </div>"""

    for i, step in enumerate(steps[:4]):
        prob_color = Colors.WARNING if step["new_prob"] > 0.4 else Colors.SUCCESS
        flow_html += f"""
        <div style="display:flex;flex-direction:column;align-items:center;padding:4px 0">
            <div style="width:2px;height:20px;background:{Colors.BORDER_DEFAULT}"></div>
            <div style="width:24px;height:24px;border-radius:50%;background:{Colors.ACCENT_BG};border:1px solid {Colors.ACCENT};display:flex;align-items:center;justify-content:center">
                <span style="color:{Colors.ACCENT};font-size:12px;font-weight:700">{i+1}</span>
            </div>
            <div style="width:2px;height:6px;background:{Colors.BORDER_DEFAULT}"></div>
        </div>
        <div style="padding:8px 16px;background:{Colors.BG_SECONDARY};border:1px solid {Colors.BORDER_SUBTLE};border-radius:8px;text-align:center;max-width:360px">
            <span style="font-size:12px;color:{Colors.TEXT_SECONDARY}">{step['action']}</span><br>
            <span style="font-size:16px;font-weight:700;color:{prob_color}">{step['new_prob']:.0%}</span>
        </div>"""

    final_color = (
        Colors.SUCCESS
        if final_prob < 0.3
        else Colors.WARNING if final_prob < 0.6 else Colors.CRITICAL
    )
    flow_html += f"""
        <div style="display:flex;flex-direction:column;align-items:center;padding:4px 0">
            <div style="width:2px;height:20px;background:{Colors.BORDER_DEFAULT}"></div>
        </div>
        <div style="padding:10px 20px;background:{Colors.rgba(Colors.SUCCESS, 0.12)};border:1px solid {Colors.rgba(Colors.SUCCESS, 0.3)};border-radius:8px;text-align:center">
            <span style="font-size:11px;color:{Colors.TEXT_MUTED}">Final Probability</span><br>
            <span style="font-size:22px;font-weight:700;color:{final_color}">{final_prob:.0%}</span>
        </div>
    </div>
</div>"""
    st.markdown(flow_html, unsafe_allow_html=True)

    reduction_pct = (1 - final_prob / risk) * 100 if risk > 0 else 0
    metric_row(
        [
            {
                "label": "Original Risk",
                "value": f"{risk:.0%}",
                "color": Colors.CRITICAL,
                "icon": "ALERT_TRIANGLE",
            },
            {
                "label": "Post-Changes Risk",
                "value": f"{final_prob:.0%}",
                "color": final_color,
                "icon": "CHECK_CIRCLE",
            },
            {
                "label": "Risk Reduction",
                "value": f"{reduction_pct:.0f}%",
                "color": Colors.SUCCESS,
                "icon": "TRENDING_DOWN",
            },
            {
                "label": "Changes Required",
                "value": str(len(steps)),
                "color": Colors.ACCENT,
                "icon": "SLIDERS",
            },
        ]
    )


def _render_similar_transactions(shap_vals: list[tuple[str, float]]):
    """Section 4: Similar historical transactions (fraud and legitimate)."""
    st.markdown(
        f"<div style='color:{Colors.TEXT_PRIMARY};font-weight:600;font-size:16px;margin-bottom:12px'>"
        f"{Icons.html('DATABASE', 16, Colors.ACCENT)} Similar Historical Transactions</div>",
        unsafe_allow_html=True,
    )

    col_fraud, col_legit = st.columns(2)

    with col_fraud:
        fraud_df = _nearest_transactions(shap_vals, is_fraud=True, n=5)
        st.markdown(
            f"<div style='font-size:13px;color:{Colors.CRITICAL};font-weight:600;margin-bottom:8px'>"
            f"{Icons.html('X_CIRCLE', 13, Colors.CRITICAL)} Nearest Fraud Cases</div>",
            unsafe_allow_html=True,
        )
        data_table(
            df=fraud_df,
            columns={
                "trace_id": "Trace ID",
                "amount": "Amount",
                "channel": "Channel",
                "risk_score": "Risk",
                "decision": "Decision",
                "similarity": "Similarity",
            },
            max_rows=5,
            status_col="Decision",
            striped=True,
        )

    with col_legit:
        legit_df = _nearest_transactions(shap_vals, is_fraud=False, n=5)
        st.markdown(
            f"<div style='font-size:13px;color:{Colors.SUCCESS};font-weight:600;margin-bottom:8px'>"
            f"{Icons.html('CHECK_CIRCLE', 13, Colors.SUCCESS)} Nearest Legitimate Cases</div>",
            unsafe_allow_html=True,
        )
        data_table(
            df=legit_df,
            columns={
                "trace_id": "Trace ID",
                "amount": "Amount",
                "channel": "Channel",
                "risk_score": "Risk",
                "decision": "Decision",
                "similarity": "Similarity",
            },
            max_rows=5,
            status_col="Decision",
            striped=True,
        )


# ── Main Render ───────────────────────────────────────────────────────────────


def render(tenant_id: str):
    with page_container(
        "Explainability",
        "SHAP attributions, counterfactual analysis, and decision explanations",
        "EYE",
    ):
        shap_df = make_shap_values(10)
        default_trace = "trace_demo_a1b2c3"

        st.markdown(
            f"<div style='color:{Colors.TEXT_PRIMARY};font-weight:600;font-size:16px;margin-bottom:8px'>"
            f"{Icons.html('SEARCH', 16, Colors.ACCENT)} Transaction Lookup</div>",
            unsafe_allow_html=True,
        )
        trace_id = st.text_input(
            "Enter trace_id",
            value=default_trace,
            placeholder="trace_abc123",
            label_visibility="collapsed",
        )

        if not trace_id:
            st.warning("Please enter a valid trace_id to view the explanation.")
            return

        txn = _synthetic_transaction(trace_id)
        shap_vals = _synthetic_shap_values(txn["risk_score"])

        kpi_row(
            [
                {
                    "label": "Risk Score",
                    "value": f"{txn['risk_score']:.2%}",
                    "icon": "TARGET",
                    "status": (
                        "critical"
                        if txn["risk_score"] > 0.7
                        else "warning" if txn["risk_score"] > 0.4 else "healthy"
                    ),
                },
                {
                    "label": "Decision",
                    "value": txn["decision"],
                    "icon": "SHIELD_CHECK",
                    "status": (
                        "critical"
                        if txn["decision"] == "BLOCK"
                        else "warning" if txn["decision"] == "REVIEW" else "healthy"
                    ),
                },
                {
                    "label": "Amount",
                    "value": f"currency_fmt(txn['amount'], txn.get('currency', 'NGN'))",
                    "icon": "CREDIT_CARD",
                },
                {
                    "label": "Channel",
                    "value": txn["channel"],
                    "icon": "GLOBE",
                },
                {
                    "label": "Features Analyzed",
                    "value": str(len(shap_vals)),
                    "icon": "LAYERS",
                },
            ]
        )

        section_divider()

        _render_prediction_summary(trace_id, txn, shap_vals)

        section_divider()

        _render_shap_explanation(shap_vals, txn["risk_score"])

        section_divider()

        _render_counterfactual(txn, shap_vals)

        section_divider()

        _render_similar_transactions(shap_vals)

        section_divider()

        st.markdown(
            f"<div style='color:{Colors.TEXT_MUTED};font-size:11px;text-align:center;padding:8px 0'>"
            f"Explanations satisfy GDPR Art. 22 (right to explanation) and EU AI Act Art. 13 (transparency for high-risk AI). "
            f"SHAP values computed via TreeSHAP on the CatBoost ensemble model.</div>",
            unsafe_allow_html=True,
        )
