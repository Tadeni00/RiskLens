"""
RiskLens Console — Metric Display Components
Specialized metric rendering for fraud detection context.
"""

import streamlit as st
from dashboard.theme.colors import Colors
from dashboard.theme.typography import Typography
from dashboard.theme.icons import Icons


def latency_display(p50: float, p95: float, p99: float, target: float = 100.0):
    """Render latency metrics with SLA indicators."""

    def _indicator(val, target):
        if val <= target * 0.5:
            return "success", "Excellent"
        elif val <= target:
            return "success", "On Target"
        elif val <= target * 1.5:
            return "warning", "Elevated"
        else:
            return "critical", "SLA Breach"

    p50_class, p50_label = _indicator(p50, target * 0.5)
    p95_class, p95_label = _indicator(p95, target)
    p99_class, p99_label = _indicator(p99, target * 1.5)

    st.markdown(
        f"""
<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px">
    <div style="background:{Colors.BG_CARD};border:1px solid {Colors.BORDER_DEFAULT};border-radius:10px;padding:16px;text-align:center">
        <div style="font-size:{Typography.TEXT_SM};color:{Colors.TEXT_MUTED};text-transform:uppercase;letter-spacing:{Typography.TRACKING_WIDER};margin-bottom:8px">P50 Latency</div>
        <div style="font-size:{Typography.TEXT_2XL};font-weight:{Typography.WEIGHT_BOLD};color:{Colors.TEXT_PRIMARY};font-family:{Typography.FONT_FAMILY}">{p50:.1f}<span style="font-size:{Typography.TEXT_SM};color:{Colors.TEXT_MUTED};margin-left:4px">ms</span></div>
        <span class="ft-badge {p50_class}" style="margin-top:8px">{p50_label}</span>
    </div>
    <div style="background:{Colors.BG_CARD};border:1px solid {Colors.BORDER_DEFAULT};border-radius:10px;padding:16px;text-align:center">
        <div style="font-size:{Typography.TEXT_SM};color:{Colors.TEXT_MUTED};text-transform:uppercase;letter-spacing:{Typography.TRACKING_WIDER};margin-bottom:8px">P95 Latency</div>
        <div style="font-size:{Typography.TEXT_2XL};font-weight:{Typography.WEIGHT_BOLD};color:{Colors.TEXT_PRIMARY};font-family:{Typography.FONT_FAMILY}">{p95:.1f}<span style="font-size:{Typography.TEXT_SM};color:{Colors.TEXT_MUTED};margin-left:4px">ms</span></div>
        <span class="ft-badge {p95_class}" style="margin-top:8px">{p95_label}</span>
    </div>
    <div style="background:{Colors.BG_CARD};border:1px solid {Colors.BORDER_DEFAULT};border-radius:10px;padding:16px;text-align:center">
        <div style="font-size:{Typography.TEXT_SM};color:{Colors.TEXT_MUTED};text-transform:uppercase;letter-spacing:{Typography.TRACKING_WIDER};margin-bottom:8px">P99 Latency</div>
        <div style="font-size:{Typography.TEXT_2XL};font-weight:{Typography.WEIGHT_BOLD};color:{Colors.TEXT_PRIMARY};font-family:{Typography.FONT_FAMILY}">{p99:.1f}<span style="font-size:{Typography.TEXT_SM};color:{Colors.TEXT_MUTED};margin-left:4px">ms</span></div>
        <span class="ft-badge {p99_class}" style="margin-top:8px">{p99_label}</span>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def fraud_rate_display(
    rate: float, threshold_warning: float = 0.02, threshold_critical: float = 0.05
):
    """Render fraud rate with visual indicator."""
    if rate <= threshold_warning:
        color = Colors.SUCCESS
        label = "Normal"
    elif rate <= threshold_critical:
        color = Colors.WARNING
        label = "Elevated"
    else:
        color = Colors.CRITICAL
        label = "High"

    st.markdown(
        f"""
<div style="display:flex;align-items:center;gap:16px;padding:16px 20px;background:{Colors.BG_CARD};border:1px solid {Colors.BORDER_DEFAULT};border-radius:10px">
    <div style="width:48px;height:48px;border-radius:12px;background:{Colors.rgba(color, 0.15)};display:flex;align-items:center;justify-content:center">
        {Icons.html('SHIELD', 24, color)}
    </div>
    <div>
        <div style="font-size:{Typography.TEXT_SM};color:{Colors.TEXT_MUTED};text-transform:uppercase;letter-spacing:{Typography.TRACKING_WIDER}">Fraud Rate</div>
        <div style="display:flex;align-items:baseline;gap:8px">
            <span style="font-size:{Typography.TEXT_3XL};font-weight:{Typography.WEIGHT_BOLD};color:{color}">{rate:.2%}</span>
            <span class="ft-badge {'success' if rate <= threshold_warning else 'warning' if rate <= threshold_critical else 'critical'}">{label}</span>
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def model_performance_summary(model_name: str, metrics: dict):
    """Render a model performance summary card."""
    metrics_html = ""
    for name, val in metrics.items():
        if isinstance(val, float):
            display_val = f"{val:.4f}"
        else:
            display_val = str(val)
        metrics_html += f"""
<div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid {Colors.BORDER_SUBTLE}">
    <span style="color:{Colors.TEXT_MUTED};font-size:{Typography.TEXT_SM}">{name}</span>
    <span style="color:{Colors.TEXT_PRIMARY};font-weight:{Typography.WEIGHT_SEMIBOLD};font-size:{Typography.TEXT_SM}">{display_val}</span>
</div>"""

    st.markdown(
        f"""
<div style="background:{Colors.BG_CARD};border:1px solid {Colors.BORDER_DEFAULT};border-radius:10px;padding:20px">
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px">
        {Icons.html('BOX', 20, Colors.ACCENT)}
        <span style="font-size:{Typography.TEXT_MD};font-weight:{Typography.WEIGHT_SEMIBOLD};color:{Colors.TEXT_PRIMARY}">{model_name}</span>
        <span class="ft-badge success" style="margin-left:auto">Active</span>
    </div>
    {metrics_html}
</div>
""",
        unsafe_allow_html=True,
    )


def confidence_display(confidence: float, label: str = ""):
    """Render confidence score with visual bar."""
    if confidence >= 0.9:
        color = Colors.SUCCESS
    elif confidence >= 0.7:
        color = Colors.WARNING
    else:
        color = Colors.CRITICAL

    st.markdown(
        f"""
<div style="padding:12px 16px;background:{Colors.BG_CARD};border:1px solid {Colors.BORDER_DEFAULT};border-radius:8px">
    <div style="display:flex;justify-content:space-between;margin-bottom:6px">
        <span style="font-size:{Typography.TEXT_SM};color:{Colors.TEXT_MUTED}">{label or 'Confidence'}</span>
        <span style="font-size:{Typography.TEXT_SM};color:{color};font-weight:{Typography.WEIGHT_SEMIBOLD}">{confidence:.1%}</span>
    </div>
    <div class="ft-progress">
        <div class="ft-progress-bar {'success' if confidence >= 0.9 else 'warning' if confidence >= 0.7 else 'critical'}" style="width:{confidence * 100}%"></div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )
