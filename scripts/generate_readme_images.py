"""Generate placeholder images for README.md. Replace with real screenshots later."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

OUT = Path("docs/images")
OUT.mkdir(parents=True, exist_ok=True)

# Design system colors
BG = "#0B1320"
CARD = "#1B2537"
ACCENT = "#2D6CDF"
GREEN = "#17A673"
YELLOW = "#D69E2E"
RED = "#E53E3E"
TEXT = "#E2E8F0"
MUTED = "#64748B"


def style_ax(ax, title=""):
    ax.set_facecolor(BG)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.text(5, 5.6, title, ha="center", va="center", fontsize=16, fontweight="bold", color=TEXT, fontfamily="monospace")


def card(ax, x, y, w, h, color=CARD, radius=0.15):
    fancy = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={radius}", facecolor=color, edgecolor="none", linewidth=0)
    ax.add_patch(fancy)


def text(ax, x, y, s, **kw):
    defaults = dict(ha="center", va="center", fontsize=9, color=TEXT, fontfamily="monospace")
    defaults.update(kw)
    ax.text(x, y, s, **defaults)


# ── 1. Dashboard Preview ──────────────────────────────────────────────────────
def gen_dashboard():
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=150)
    style_ax(ax, "FraudTrap — Operations Console")

    # Top KPI cards
    kpis = [("Transactions", "1.2M", GREEN), ("Fraud Rate", "1.47%", YELLOW), ("Blocked", "342", RED), ("P95 Latency", "87ms", GREEN)]
    for i, (label, value, color) in enumerate(kpis):
        x = 0.4 + i * 2.4
        card(ax, x, 4.3, 2.1, 1.0)
        text(ax, x + 1.05, 5.0, value, fontsize=14, fontweight="bold", color=color)
        text(ax, x + 1.05, 4.55, label, fontsize=7, color=MUTED)

    # Chart placeholders
    card(ax, 0.4, 0.3, 4.4, 3.8)
    text(ax, 2.6, 3.8, "Score Distribution", fontsize=10, fontweight="bold")
    # Fake bar chart
    bars = [0.3, 0.5, 0.8, 1.0, 0.7, 0.4, 0.2]
    for i, h in enumerate(bars):
        ax.add_patch(FancyBboxPatch((0.8 + i * 0.55, 0.6), 0.4, h * 2.5, boxstyle="round,pad=0.02", facecolor=ACCENT, alpha=0.7))

    card(ax, 5.2, 0.3, 4.4, 3.8)
    text(ax, 7.4, 3.8, "Decision Timeline", fontsize=10, fontweight="bold")
    # Fake line chart
    xs = np.linspace(5.6, 9.2, 30)
    ys = 1.0 + np.sin(xs * 3) * 0.8 + np.random.randn(30) * 0.15
    ax.plot(xs, ys, color=ACCENT, linewidth=2)
    ax.fill_between(xs, ys, 0.6, alpha=0.15, color=ACCENT)

    fig.savefig(OUT / "dashboard-preview.png", dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  dashboard-preview.png")


# ── 2. Architecture Overview ──────────────────────────────────────────────────
def gen_architecture():
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=150)
    style_ax(ax, "FraudTrap — System Architecture")

    layers = [
        (0.5, 4.8, "Transaction", ACCENT),
        (2.5, 4.8, "Feature Store", ACCENT),
        (4.5, 4.8, "Rules Engine", YELLOW),
        (6.5, 4.8, "ML Router", GREEN),
        (8.5, 4.8, "Decision", GREEN),
    ]
    for x, y, label, color in layers:
        card(ax, x, y, 1.6, 0.7, color=color)
        text(ax, x + 0.8, y + 0.35, label, fontsize=8, fontweight="bold")

    # Arrows between top row
    for i in range(len(layers) - 1):
        ax.annotate("", xy=(layers[i+1][0], 5.15), xytext=(layers[i][0] + 1.6, 5.15),
                     arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.5))

    # Middle row
    mid = [
        (1.0, 3.2, "VAE + IF + Tail", YELLOW),
        (4.0, 3.2, "TabPFN", ACCENT),
        (7.0, 3.2, "CatBoost", GREEN),
    ]
    for x, y, label, color in mid:
        card(ax, x, y, 2.0, 0.7, color=color)
        text(ax, x + 1.0, y + 0.35, label, fontsize=8, fontweight="bold")

    # Labels
    text(ax, 2.0, 4.3, "Phase 1", fontsize=7, color=YELLOW)
    text(ax, 5.0, 4.3, "Phase 2", fontsize=7, color=ACCENT)
    text(ax, 8.0, 4.3, "Phase 3", fontsize=7, color=GREEN)

    # Bottom row - storage
    storage = [
        (1.5, 1.6, "Redis", RED),
        (4.0, 1.6, "Kafka", "#231F20"),
        (6.5, 1.6, "ClickHouse", YELLOW),
        (9.0, 1.6, "PostgreSQL", ACCENT),
    ]
    for x, y, label, color in storage:
        card(ax, x, y, 1.6, 0.7, color=CARD)
        text(ax, x + 0.8, y + 0.35, label, fontsize=8, fontweight="bold", color=color)

    text(ax, 5.0, 0.8, "Storage Layer", fontsize=9, color=MUTED)

    fig.savefig(OUT / "architecture-overview.png", dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  architecture-overview.png")


# ── 3. Dashboard Overview ─────────────────────────────────────────────────────
def gen_dashboard_overview():
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=150)
    style_ax(ax, "FraudTrap — Dashboard Pages")

    pages = [
        ("Overview", "8 KPIs, health, timeline"),
        ("Risk Intelligence", "Fraud map, leaderboards"),
        ("Behaviour Profiles", "5 entity profiles"),
        ("Models", "Champion metrics"),
        ("Model Performance", "Precision-recall curves"),
        ("Explainability", "SHAP, counterfactuals"),
        ("Drift Monitoring", "PSI, KL divergence"),
        ("Live Monitoring", "Real-time metrics"),
        ("Compliance", "Audit trail, bias"),
        ("Lifecycle", "Phase progression"),
        ("Monitoring", "System health"),
        ("EDA", "Feature distributions"),
    ]

    cols = 4
    rows = 3
    for i, (name, desc) in enumerate(pages):
        col = i % cols
        row = rows - 1 - i // cols
        x = 0.3 + col * 2.45
        y = 0.5 + row * 1.6
        card(ax, x, y, 2.2, 1.3)
        # Colored top bar
        colors = [ACCENT, GREEN, YELLOW, RED]
        ax.add_patch(FancyBboxPatch((x + 0.1, y + 1.0), 2.0, 0.15, boxstyle="round,pad=0.02", facecolor=colors[i % 4], alpha=0.8))
        text(ax, x + 1.1, y + 0.75, name, fontsize=9, fontweight="bold")
        text(ax, x + 1.1, y + 0.4, desc, fontsize=6, color=MUTED)

    fig.savefig(OUT / "dashboard-overview.png", dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  dashboard-overview.png")


if __name__ == "__main__":
    print("Generating placeholder images...")
    gen_dashboard()
    gen_architecture()
    gen_dashboard_overview()
    print("Done. Replace these with real screenshots.")
