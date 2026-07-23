"""
FraudTrap Design System — Color Palette
Enterprise-grade dark theme inspired by Stripe, Datadog, and Bloomberg Terminal.
"""


class Colors:
    # ── Backgrounds ──────────────────────────────────────────────────────────
    BG_PRIMARY = "#0B1320"
    BG_SECONDARY = "#141C2F"
    BG_CARD = "#1B2537"
    BG_CARD_HOVER = "#1F2B3D"
    BG_ELEVATED = "#212D42"
    BG_INPUT = "#0E1628"

    # ── Borders ──────────────────────────────────────────────────────────────
    BORDER_DEFAULT = "#2D3A53"
    BORDER_SUBTLE = "#1E2A40"
    BORDER_STRONG = "#3D4F6F"
    BORDER_FOCUS = "#2D6CDF"

    # ── Text ─────────────────────────────────────────────────────────────────
    TEXT_PRIMARY = "#F4F7FA"
    TEXT_SECONDARY = "#A7B3C5"
    TEXT_MUTED = "#6F7B8F"
    TEXT_DISABLED = "#4A5568"
    TEXT_INVERSE = "#0B1320"

    # ── Accent ───────────────────────────────────────────────────────────────
    ACCENT = "#2D6CDF"
    ACCENT_LIGHT = "#4A8AF5"
    ACCENT_DARK = "#1E4FA0"
    ACCENT_BG = "rgba(45, 108, 223, 0.12)"
    ACCENT_BG_HOVER = "rgba(45, 108, 223, 0.20)"

    # ── Semantic ─────────────────────────────────────────────────────────────
    SUCCESS = "#17A673"
    SUCCESS_LIGHT = "#22C990"
    SUCCESS_BG = "rgba(23, 166, 115, 0.12)"

    WARNING = "#D69E2E"
    WARNING_LIGHT = "#E8B84A"
    WARNING_BG = "rgba(214, 158, 46, 0.12)"

    CRITICAL = "#D64545"
    CRITICAL_LIGHT = "#E86363"
    CRITICAL_BG = "rgba(214, 69, 69, 0.12)"

    INFO = "#2D6CDF"
    INFO_BG = "rgba(45, 108, 223, 0.12)"

    # ── Chart Palette ────────────────────────────────────────────────────────
    CHART_1 = "#2D6CDF"
    CHART_2 = "#17A673"
    CHART_3 = "#D69E2E"
    CHART_4 = "#D64545"
    CHART_5 = "#8B5CF6"
    CHART_6 = "#EC4899"
    CHART_7 = "#06B6D4"
    CHART_8 = "#F97316"

    CHART_PALETTE = [CHART_1, CHART_2, CHART_3, CHART_4, CHART_5, CHART_6, CHART_7, CHART_8]

    # ── Status ───────────────────────────────────────────────────────────────
    STATUS_HEALTHY = "#17A673"
    STATUS_WARNING = "#D69E2E"
    STATUS_CRITICAL = "#D64545"
    STATUS_OFFLINE = "#6F7B8F"

    # ── Phase Colors ─────────────────────────────────────────────────────────
    PHASE_1 = "#D69E2E"
    PHASE_2 = "#2D6CDF"
    PHASE_3 = "#17A673"

    # ── Sparkline / Trend ────────────────────────────────────────────────────
    TREND_UP = "#17A673"
    TREND_DOWN = "#D64545"
    TREND_NEUTRAL = "#6F7B8F"

    @classmethod
    def rgba(cls, hex_color: str, alpha: float) -> str:
        """Convert hex color to rgba string."""
        hex_color = hex_color.lstrip("#")
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        return f"rgba({r}, {g}, {b}, {alpha})"

    @classmethod
    def chart_palette(cls, n: int) -> list:
        """Return n colors from the chart palette, cycling if needed."""
        return [cls.CHART_PALETTE[i % len(cls.CHART_PALETTE)] for i in range(n)]
