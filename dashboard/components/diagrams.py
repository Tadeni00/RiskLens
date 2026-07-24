"""
FraudTrap Dashboard — Architecture Diagram Components
SVG-based architecture visualizations for the ML pipeline.
"""

import streamlit as st
from dashboard.theme.colors import Colors
from dashboard.theme.typography import Typography
from dashboard.theme.icons import Icons


def pipeline_diagram(active_step: int = None, completed_steps: list = None):
    """
    Render the fraud detection pipeline as an interactive diagram.

    Args:
        active_step: Index of the currently active step (0-based)
        completed_steps: List of completed step indices
    """
    steps = [
        ("Incoming Transaction", "Zap", "Transaction arrives via API"),
        ("Feature Store", "DATABASE", "Redis-backed feature assembly"),
        ("Behavior Engine", "ACTIVITY", "5 entity profiles updated"),
        ("Cold Start Layer", "SHIELD", "VAE + Isolation Forest + Tail"),
        ("Adaptive Learning", "LAYERS", "TabPFN foundation model"),
        ("Supervised Layer", "BRAIN", "CatBoost Champion + FT-Transformer"),
        ("Confidence Check", "TARGET", "Route to specialist if needed"),
        ("Explainability", "EYE", "SHAP + Counterfactual"),
        ("Decision Engine", "SHIELD_CHECK", "APPROVE / REVIEW / BLOCK"),
    ]

    completed_steps = completed_steps or []
    items_html = ""
    for i, (name, icon, desc) in enumerate(steps):
        step_class = "pipeline-step"
        if i == active_step:
            step_class += " active"
        elif i in completed_steps:
            step_class += " complete"

        icon_color = Colors.TEXT_MUTED
        if i == active_step:
            icon_color = Colors.ACCENT
        elif i in completed_steps:
            icon_color = Colors.SUCCESS

        items_html += f"""
<div style="display:flex;flex-direction:column;align-items:center;gap:8px">
    <div class="{step_class}" style="flex-direction:column;text-align:center;min-width:140px;padding:16px 12px">
        <div style="margin-bottom:8px">{Icons.html(icon, 20, icon_color)}</div>
        <div style="font-size:{Typography.TEXT_SM};font-weight:{Typography.WEIGHT_SEMIBOLD};color:{Colors.TEXT_PRIMARY}">{name}</div>
        <div style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_MUTED}">{desc}</div>
    </div>
</div>"""
        if i < len(steps) - 1:
            items_html += f'<div class="pipeline-arrow">→</div>'

    st.markdown(
        f"""
<div class="pipeline-container">
    <div style="display:flex;align-items:flex-start;overflow-x:auto;padding:8px 0">
        {items_html}
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def architecture_diagram():
    """Render the full system architecture as an SVG diagram."""
    svg = f"""
<svg viewBox="0 0 900 600" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:{Typography.FONT_FAMILY}">
    <!-- Background -->
    <rect width="900" height="600" fill="{Colors.BG_CARD}" rx="12"/>

    <!-- Title -->
    <text x="450" y="36" text-anchor="middle" fill="{Colors.TEXT_PRIMARY}" font-size="18" font-weight="700">FraudTrap Production Architecture</text>

    <!-- Input -->
    <rect x="350" y="56" width="200" height="40" rx="8" fill="{Colors.BG_SECONDARY}" stroke="{Colors.BORDER_DEFAULT}"/>
    <text x="450" y="82" text-anchor="middle" fill="{Colors.TEXT_PRIMARY}" font-size="12" font-weight="600">Incoming Transaction</text>

    <!-- Arrow down -->
    <line x1="450" y1="96" x2="450" y2="120" stroke="{Colors.TEXT_MUTED}" stroke-width="1.5" marker-end="url(#arrowhead)"/>

    <!-- Feature Store -->
    <rect x="300" y="120" width="150" height="40" rx="8" fill="{Colors.rgba(Colors.ACCENT, 0.12)}" stroke="{Colors.ACCENT}" stroke-width="1.5"/>
    <text x="375" y="146" text-anchor="middle" fill="{Colors.ACCENT_LIGHT}" font-size="11" font-weight="600">Feature Store</text>

    <!-- Behavior Engine -->
    <rect x="480" y="120" width="150" height="40" rx="8" fill="{Colors.rgba(Colors.ACCENT, 0.12)}" stroke="{Colors.ACCENT}" stroke-width="1.5"/>
    <text x="555" y="146" text-anchor="middle" fill="{Colors.ACCENT_LIGHT}" font-size="11" font-weight="600">Behavior Engine</text>

    <!-- Arrow down -->
    <line x1="450" y1="160" x2="450" y2="190" stroke="{Colors.TEXT_MUTED}" stroke-width="1.5" marker-end="url(#arrowhead)"/>

    <!-- Rules Engine -->
    <rect x="375" y="190" width="150" height="36" rx="8" fill="{Colors.rgba(Colors.WARNING, 0.12)}" stroke="{Colors.WARNING}" stroke-width="1.5"/>
    <text x="450" y="214" text-anchor="middle" fill="{Colors.WARNING_LIGHT}" font-size="11" font-weight="600">Rules Engine (&lt;1ms)</text>

    <!-- Arrow down -->
    <line x1="450" y1="226" x2="450" y2="256" stroke="{Colors.TEXT_MUTED}" stroke-width="1.5" marker-end="url(#arrowhead)"/>

    <!-- ML Router -->
    <rect x="375" y="256" width="150" height="36" rx="8" fill="{Colors.rgba(Colors.ACCENT, 0.12)}" stroke="{Colors.ACCENT}" stroke-width="1.5"/>
    <text x="450" y="280" text-anchor="middle" fill="{Colors.ACCENT_LIGHT}" font-size="11" font-weight="600">ML Model Router</text>

    <!-- Phase arrows down -->
    <line x1="350" y1="292" x2="200" y2="330" stroke="{Colors.TEXT_MUTED}" stroke-width="1" marker-end="url(#arrowhead)"/>
    <line x1="450" y1="292" x2="450" y2="330" stroke="{Colors.TEXT_MUTED}" stroke-width="1" marker-end="url(#arrowhead)"/>
    <line x1="550" y1="292" x2="700" y2="330" stroke="{Colors.TEXT_MUTED}" stroke-width="1" marker-end="url(#arrowhead)"/>

    <!-- Phase 1: Cold Start -->
    <rect x="100" y="330" width="200" height="60" rx="8" fill="{Colors.rgba(Colors.PHASE_1, 0.12)}" stroke="{Colors.PHASE_1}" stroke-width="1.5"/>
    <text x="200" y="356" text-anchor="middle" fill="{Colors.PHASE_1}" font-size="11" font-weight="700">Phase 1: Cold Start</text>
    <text x="200" y="374" text-anchor="middle" fill="{Colors.TEXT_SECONDARY}" font-size="10">VAE + Isolation Forest + Tail</text>

    <!-- Phase 2: Semi-supervised -->
    <rect x="350" y="330" width="200" height="60" rx="8" fill="{Colors.rgba(Colors.PHASE_2, 0.12)}" stroke="{Colors.PHASE_2}" stroke-width="1.5"/>
    <text x="450" y="356" text-anchor="middle" fill="{Colors.PHASE_2}" font-size="11" font-weight="700">Phase 2: Semi-supervised</text>
    <text x="450" y="374" text-anchor="middle" fill="{Colors.TEXT_SECONDARY}" font-size="10">TabPFN Foundation Model</text>

    <!-- Phase 3: Supervised -->
    <rect x="600" y="330" width="200" height="60" rx="8" fill="{Colors.rgba(Colors.PHASE_3, 0.12)}" stroke="{Colors.PHASE_3}" stroke-width="1.5"/>
    <text x="700" y="356" text-anchor="middle" fill="{Colors.PHASE_3}" font-size="11" font-weight="700">Phase 3: Supervised</text>
    <text x="700" y="374" text-anchor="middle" fill="{Colors.TEXT_SECONDARY}" font-size="10">CatBoost + FT-Transformer</text>

    <!-- Arrow down from Phase 3 -->
    <line x1="700" y1="390" x2="700" y2="420" stroke="{Colors.TEXT_MUTED}" stroke-width="1" marker-end="url(#arrowhead)"/>

    <!-- Confidence Check -->
    <rect x="625" y="420" width="150" height="36" rx="8" fill="{Colors.rgba(Colors.INFO, 0.12)}" stroke="{Colors.INFO}" stroke-width="1.5"/>
    <text x="700" y="444" text-anchor="middle" fill="{Colors.ACCENT_LIGHT}" font-size="11" font-weight="600">Confidence Check</text>

    <!-- Arrow to specialist -->
    <line x1="775" y1="438" x2="820" y2="438" stroke="{Colors.TEXT_MUTED}" stroke-width="1" marker-end="url(#arrowhead)"/>

    <!-- FT-Transformer -->
    <rect x="820" y="420" width="60" height="36" rx="6" fill="{Colors.rgba(Colors.CHART_5, 0.12)}" stroke="{Colors.CHART_5}" stroke-width="1"/>
    <text x="850" y="444" text-anchor="middle" fill="{Colors.CHART_5}" font-size="9" font-weight="600">FT-Trans</text>

    <!-- Arrow down to Decision -->
    <line x1="450" y1="390" x2="450" y2="480" stroke="{Colors.TEXT_MUTED}" stroke-width="1.5" marker-end="url(#arrowhead)"/>

    <!-- Decision Engine -->
    <rect x="350" y="480" width="200" height="40" rx="8" fill="{Colors.rgba(Colors.SUCCESS, 0.12)}" stroke="{Colors.SUCCESS}" stroke-width="1.5"/>
    <text x="450" y="506" text-anchor="middle" fill="{Colors.SUCCESS_LIGHT}" font-size="12" font-weight="700">Decision Engine</text>

    <!-- Output arrows -->
    <line x1="380" y1="520" x2="250" y2="555" stroke="{Colors.SUCCESS}" stroke-width="1" marker-end="url(#arrowhead)"/>
    <line x1="450" y1="520" x2="450" y2="555" stroke="{Colors.WARNING}" stroke-width="1" marker-end="url(#arrowhead)"/>
    <line x1="520" y1="520" x2="650" y2="555" stroke="{Colors.CRITICAL}" stroke-width="1" marker-end="url(#arrowhead)"/>

    <!-- Outputs -->
    <rect x="180" y="555" width="140" height="30" rx="6" fill="{Colors.rgba(Colors.SUCCESS, 0.12)}" stroke="{Colors.SUCCESS}" stroke-width="1"/>
    <text x="250" y="575" text-anchor="middle" fill="{Colors.SUCCESS}" font-size="11" font-weight="600">APPROVE</text>

    <rect x="380" y="555" width="140" height="30" rx="6" fill="{Colors.rgba(Colors.WARNING, 0.12)}" stroke="{Colors.WARNING}" stroke-width="1"/>
    <text x="450" y="575" text-anchor="middle" fill="{Colors.WARNING}" font-size="11" font-weight="600">REVIEW</text>

    <rect x="580" y="555" width="140" height="30" rx="6" fill="{Colors.rgba(Colors.CRITICAL, 0.12)}" stroke="{Colors.CRITICAL}" stroke-width="1"/>
    <text x="650" y="575" text-anchor="middle" fill="{Colors.CRITICAL}" font-size="11" font-weight="600">BLOCK</text>

    <!-- Side panels: Explainability -->
    <rect x="20" y="250" width="120" height="50" rx="8" fill="{Colors.BG_SECONDARY}" stroke="{Colors.BORDER_DEFAULT}"/>
    <text x="80" y="272" text-anchor="middle" fill="{Colors.TEXT_SECONDARY}" font-size="10" font-weight="600">Explainability</text>
    <text x="80" y="288" text-anchor="middle" fill="{Colors.TEXT_MUTED}" font-size="9">SHAP + CF</text>
    <line x1="140" y1="275" x2="300" y2="275" stroke="{Colors.BORDER_DEFAULT}" stroke-width="1" stroke-dasharray="4,4"/>

    <!-- Side panels: Drift Monitoring -->
    <rect x="20" y="320" width="120" height="50" rx="8" fill="{Colors.BG_SECONDARY}" stroke="{Colors.BORDER_DEFAULT}"/>
    <text x="80" y="342" text-anchor="middle" fill="{Colors.TEXT_SECONDARY}" font-size="10" font-weight="600">Drift Monitor</text>
    <text x="80" y="358" text-anchor="middle" fill="{Colors.TEXT_MUTED}" font-size="9">PSI + KL Div</text>
    <line x1="140" y1="345" x2="300" y2="345" stroke="{Colors.BORDER_DEFAULT}" stroke-width="1" stroke-dasharray="4,4"/>

    <!-- Model Registry -->
    <rect x="760" y="250" width="120" height="50" rx="8" fill="{Colors.BG_SECONDARY}" stroke="{Colors.BORDER_DEFAULT}"/>
    <text x="820" y="272" text-anchor="middle" fill="{Colors.TEXT_SECONDARY}" font-size="10" font-weight="600">Model Registry</text>
    <text x="820" y="288" text-anchor="middle" fill="{Colors.TEXT_MUTED}" font-size="9">Version + Rollback</text>
    <line x1="600" y1="275" x2="760" y2="275" stroke="{Colors.BORDER_DEFAULT}" stroke-width="1" stroke-dasharray="4,4"/>

    <!-- Feedback Loop -->
    <rect x="760" y="320" width="120" height="50" rx="8" fill="{Colors.BG_SECONDARY}" stroke="{Colors.BORDER_DEFAULT}"/>
    <text x="820" y="342" text-anchor="middle" fill="{Colors.TEXT_SECONDARY}" font-size="10" font-weight="600">Retraining</text>
    <text x="820" y="358" text-anchor="middle" fill="{Colors.TEXT_MUTED}" font-size="9">Auto-triggered</text>
    <line x1="600" y1="345" x2="760" y2="345" stroke="{Colors.BORDER_DEFAULT}" stroke-width="1" stroke-dasharray="4,4"/>

    <!-- Arrowhead marker -->
    <defs>
        <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
            <polygon points="0 0, 10 3.5, 0 7" fill="{Colors.TEXT_MUTED}"/>
        </marker>
    </defs>
</svg>"""
    st.components.v1.html(svg, width=920, height=600, scrolling=True)


def lifecycle_timeline(current_stage: int = 5):
    """Render model lifecycle timeline."""
    stages = [
        ("Training", "BRAIN", 0),
        ("Validation", "CHECK_CIRCLE", 1),
        ("Calibration", "SLIDERS", 2),
        ("Registration", "FILE_TEXT", 3),
        ("Champion", "SHIELD_CHECK", 4),
        ("Monitoring", "ACTIVITY", 5),
        ("Promotion", "TRENDING_UP", 6),
        ("Archived", "FOLDER", 7),
    ]

    items_html = ""
    for i, (name, icon, idx) in enumerate(stages):
        if idx < current_stage:
            state = "complete"
            icon_color = Colors.SUCCESS
            line_color = Colors.SUCCESS
        elif idx == current_stage:
            state = "active"
            icon_color = Colors.ACCENT
            line_color = Colors.ACCENT
        else:
            state = "pending"
            icon_color = Colors.TEXT_MUTED
            line_color = Colors.BORDER_DEFAULT

        connector = ""
        if i < len(stages) - 1:
            connector = (
                f'<div style="width:40px;height:2px;background:{line_color};margin:0 4px"></div>'
            )

        items_html += f"""
<div style="display:flex;align-items:center">
    <div style="display:flex;flex-direction:column;align-items:center;gap:6px">
        <div style="width:36px;height:36px;border-radius:50%;background:{Colors.rgba(icon_color, 0.15) if state != 'pending' else Colors.BG_SECONDARY};border:2px solid {icon_color};display:flex;align-items:center;justify-content:center">
            {Icons.html(icon, 16, icon_color)}
        </div>
        <span style="font-size:{Typography.TEXT_XS};color:{Colors.TEXT_PRIMARY if state == 'active' else Colors.TEXT_SECONDARY if state == 'complete' else Colors.TEXT_MUTED};font-weight:{Typography.WEIGHT_SEMIBOLD if state == 'active' else Typography.WEIGHT_NORMAL}">{name}</span>
    </div>
    {connector}
</div>"""

    st.markdown(
        f"""
<div style="background:{Colors.BG_CARD};border:1px solid {Colors.BORDER_DEFAULT};border-radius:12px;padding:24px;overflow-x:auto">
    <div style="display:flex;align-items:flex-start;min-width:max-content">
        {items_html}
    </div>
</div>
""",
        unsafe_allow_html=True,
    )
