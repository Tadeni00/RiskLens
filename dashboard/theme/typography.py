"""
FraudTrap Design System — Typography
Professional type scale using Inter / IBM Plex Sans.
"""


class Typography:
    FONT_FAMILY = "'Inter', 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
    FONT_MONO = "'IBM Plex Mono', 'Fira Code', 'Consolas', monospace"

    # ── Size Scale ───────────────────────────────────────────────────────────
    TEXT_XS = "0.6875rem"    # 11px
    TEXT_SM = "0.75rem"      # 12px
    TEXT_BASE = "0.875rem"   # 14px
    TEXT_MD = "1rem"         # 16px
    TEXT_LG = "1.125rem"     # 18px
    TEXT_XL = "1.375rem"     # 22px
    TEXT_2XL = "1.75rem"     # 28px
    TEXT_3XL = "2rem"        # 32px
    TEXT_4XL = "2.5rem"      # 40px

    # ── Weight Scale ─────────────────────────────────────────────────────────
    WEIGHT_NORMAL = "400"
    WEIGHT_MEDIUM = "500"
    WEIGHT_SEMIBOLD = "600"
    WEIGHT_BOLD = "700"

    # ── Line Height ──────────────────────────────────────────────────────────
    LEADING_TIGHT = "1.2"
    LEADING_NORMAL = "1.5"
    LEADING_RELAXED = "1.65"

    # ── Letter Spacing ───────────────────────────────────────────────────────
    TRACKING_TIGHT = "-0.02em"
    TRACKING_NORMAL = "0"
    TRACKING_WIDE = "0.04em"
    TRACKING_WIDER = "0.08em"

    @classmethod
    def page_title_style(cls) -> str:
        return (
            f"font-family: {cls.FONT_FAMILY}; "
            f"font-size: {cls.TEXT_3XL}; "
            f"font-weight: {cls.WEIGHT_BOLD}; "
            f"letter-spacing: {cls.TRACKING_TIGHT}; "
            f"line-height: {cls.LEADING_TIGHT}; "
            f"color: #F4F7FA; "
            f"margin: 0; "
            f"padding: 0;"
        )

    @classmethod
    def section_title_style(cls) -> str:
        return (
            f"font-family: {cls.FONT_FAMILY}; "
            f"font-size: {cls.TEXT_XL}; "
            f"font-weight: {cls.WEIGHT_SEMIBOLD}; "
            f"letter-spacing: {cls.TRACKING_TIGHT}; "
            f"line-height: {cls.LEADING_TIGHT}; "
            f"color: #F4F7FA; "
            f"margin: 0; "
            f"padding: 0;"
        )

    @classmethod
    def card_title_style(cls) -> str:
        return (
            f"font-family: {cls.FONT_FAMILY}; "
            f"font-size: {cls.TEXT_MD}; "
            f"font-weight: {cls.WEIGHT_MEDIUM}; "
            f"color: #A7B3C5; "
            f"margin: 0; "
            f"padding: 0;"
        )

    @classmethod
    def body_style(cls) -> str:
        return (
            f"font-family: {cls.FONT_FAMILY}; "
            f"font-size: {cls.TEXT_BASE}; "
            f"font-weight: {cls.WEIGHT_NORMAL}; "
            f"line-height: {cls.LEADING_RELAXED}; "
            f"color: #A7B3C5; "
            f"margin: 0; "
            f"padding: 0;"
        )

    @classmethod
    def metric_large_style(cls) -> str:
        return (
            f"font-family: {cls.FONT_FAMILY}; "
            f"font-size: {cls.TEXT_3XL}; "
            f"font-weight: {cls.WEIGHT_BOLD}; "
            f"letter-spacing: {cls.TRACKING_TIGHT}; "
            f"line-height: 1; "
            f"color: #F4F7FA; "
            f"margin: 0; "
            f"padding: 0;"
        )

    @classmethod
    def metric_label_style(cls) -> str:
        return (
            f"font-family: {cls.FONT_FAMILY}; "
            f"font-size: {cls.TEXT_SM}; "
            f"font-weight: {cls.WEIGHT_MEDIUM}; "
            f"letter-spacing: {cls.TRACKING_WIDER}; "
            f"text-transform: uppercase; "
            f"color: #6F7B8F; "
            f"margin: 0; "
            f"padding: 0;"
        )

    @classmethod
    def code_style(cls) -> str:
        return (
            f"font-family: {cls.FONT_MONO}; "
            f"font-size: {cls.TEXT_SM}; "
            f"color: #A7B3C5; "
            f"background: #0E1628; "
            f"padding: 2px 6px; "
            f"border-radius: 4px; "
            f"border: 1px solid #2D3A53;"
        )
