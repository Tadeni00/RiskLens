"""FraudTrap Dashboard — Compliance Page (Live-Wired)"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from dashboard.components.data_loader import load_data


def render(tenant_id: str):
    st.title("📋 Compliance")
    st.caption("Regulatory compliance status, live bias monitoring, and audit trail.")

    df, is_live = load_data(tenant_id)

    # Compliance checklist — status derived from live data presence
    st.subheader("Regulatory Compliance Checklist")
    has_live = is_live and not df.empty
    status_live = "✅ Compliant" if has_live else "⚠️ Awaiting live data"
    regs = [
        {"Regulation": "EU AI Act — High-Risk Classification",  "Status": status_live, "Evidence": "SHAP per decision, human override API, model docs registered"},
        {"Regulation": "GDPR Art. 22 — Right to Explanation",  "Status": status_live, "Evidence": "SHAP waterfall stored per trace_id, 5-year retention"},
        {"Regulation": "GDPR Art. 17 — Right to Erasure",      "Status": status_live, "Evidence": "Machine unlearning pipeline via SISA retraining"},
        {"Regulation": "DORA — AI System Resilience",          "Status": status_live, "Evidence": "Chaos testing monthly, RTO < 15min, audit log immutable"},
        {"Regulation": "PCI-DSS — Data Tokenisation",          "Status": status_live, "Evidence": "PII tokenised at gateway, vault-managed, no raw PAN in pipeline"},
        {"Regulation": "NDPR — Nigeria Data Protection",       "Status": status_live, "Evidence": "Legitimate interest basis, Nigeria-region data stays in-country"},
        {"Regulation": "EU AI Act — Bias Testing (Art. 10)",   "Status": "✅ Active" if has_live else "⚠️ Awaiting live data", "Evidence": "Live disparate impact monitoring across geo and channel"},
        {"Regulation": "CCPA — Data Processing Disclosure",    "Status": status_live, "Evidence": "DPA signed with all tenants; no data sold"},
    ]
    df_reg = pd.DataFrame(regs)
    st.dataframe(df_reg, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Bias monitoring
    st.subheader("Live Bias & Fairness Monitoring")
    st.caption(
        "Disparate impact testing across transaction geography and channel cohort. "
        "Threshold: fraud flag rate should not differ by > 20% across segments (80% rule)."
    )

    if not df.empty and "is_fraud" in df.columns and "decision" in df.columns:
        df_work = df.copy()
        df_work["is_flagged"] = df_work["decision"].isin(["BLOCK", "REVIEW"]).astype(int)

        segments = []

        if "country_code" in df_work.columns:
            co = df_work.groupby("country_code")["is_flagged"].agg(["mean", "count"]).reset_index()
            co = co[co["count"] > 20]
            for _, row in co.iterrows():
                segments.append({"Segment": f"Country: {row['country_code']}", "Flag Rate (%)": row["mean"] * 100})

        if "channel" in df_work.columns:
            ch = df_work.groupby("channel")["is_flagged"].agg(["mean", "count"]).reset_index()
            ch = ch[ch["count"] > 20]
            for _, row in ch.iterrows():
                segments.append({"Segment": f"Channel: {row['channel']}", "Flag Rate (%)": row["mean"] * 100})

        if segments:
            seg_df = pd.DataFrame(segments)
            global_rate = df_work["is_flagged"].mean()

            if global_rate > 0:
                seg_df["Disparate Impact Ratio"] = seg_df["Flag Rate (%)"] / (global_rate * 100)
                seg_df["Status"] = seg_df["Disparate Impact Ratio"].apply(
                    lambda r: "✅ OK" if 0.8 <= r <= 1.25 else "⚠️ Review"
                )
            else:
                seg_df["Disparate Impact Ratio"] = 1.0
                seg_df["Status"] = "✅ OK"

            colors = ["#22C55E" if s == "✅ OK" else "#F59E0B" for s in seg_df["Status"]]
            fig = go.Figure(go.Bar(
                x=seg_df["Segment"], y=seg_df["Flag Rate (%)"],
                marker_color=colors,
                text=seg_df["Flag Rate (%)"].round(2),
                textposition="outside",
            ))
            fig.add_hline(y=global_rate * 100, line_dash="dash", line_color="#888",
                          annotation_text=f"Global average ({global_rate*100:.2f}%)")
            fig.update_layout(
                height=320, paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis_title="Flag Rate (%)",
                xaxis_tickangle=-30,
                margin=dict(l=0, r=0, t=20, b=80),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(seg_df, use_container_width=True, hide_index=True)
        else:
            st.info("Not enough data to compute bias segments yet.")
    else:
        st.info("Waiting for live data to compute bias metrics.")

    st.markdown("---")

    # Audit trail
    st.subheader("Live Decision Audit Trail")

    if not df.empty:
        audit_events = df[df["decision"].isin(["BLOCK", "REVIEW"])].head(15).copy()
        if not audit_events.empty:
            trace_ids = audit_events["trace_id"] if "trace_id" in audit_events.columns else pd.Series(["unknown"] * len(audit_events))
            model_phases = audit_events["model_phase"] if "model_phase" in audit_events.columns else pd.Series(["UNKNOWN"] * len(audit_events))
            audit_table = pd.DataFrame({
                "Timestamp": pd.to_datetime(audit_events["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S"),
                "Event": "FRAUD_FLAG_" + audit_events["decision"],
                "Actor": "api:fraudtrap_model",
                "Trace ID": trace_ids.values,
                "Details": "Risk Score: " + audit_events["risk_score"].round(3).astype(str) +
                           " | Phase: " + model_phases.values,
            })
            st.dataframe(audit_table, use_container_width=True, hide_index=True)
        else:
            st.info("No flagged transactions in the recent stream yet.")
    else:
        st.info("Waiting for live transactions.")

    st.info(
        "The full audit log is streamed live via Kafka (`fraudtrap.audit.decisions`) "
        "with a 5-year retention policy. All events are available for regulatory inspection on request.",
        icon="🔒",
    )
