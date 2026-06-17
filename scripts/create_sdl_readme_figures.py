"""Create public README figures for the SDL/MAP extension."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sdl import load_sdl_bundle, recommend_next_experiments  # noqa: E402


MODEL_PATH = PROJECT_ROOT / "models" / "sioc_app_target_models.joblib"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
N_CANDIDATES = 2800
N_SUGGESTIONS = 10
NOVELTY_METHOD = "hybrid"


def save_candidate_search_figure() -> None:
    """Plot the 2,800-candidate SDL screen and selected top 10 designs."""
    bundle = load_sdl_bundle(MODEL_PATH)
    recommendations, ranked = recommend_next_experiments(
        bundle,
        n_candidates=N_CANDIDATES,
        n_suggestions=N_SUGGESTIONS,
        novelty_method=NOVELTY_METHOD,
        random_state=42,
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), dpi=170)
    fig.patch.set_facecolor("white")

    scatter = axes[0].scatter(
        ranked["predicted_qrev_mah_g"],
        ranked["predicted_ce_pct"],
        c=ranked["acquisition_score_pct"],
        cmap="viridis",
        s=18,
        alpha=0.42,
        linewidths=0,
    )
    axes[0].scatter(
        recommendations["predicted_qrev_mah_g"],
        recommendations["predicted_ce_pct"],
        facecolors="none",
        edgecolors="#c2410c",
        linewidths=1.8,
        s=110,
        label="Top 10 proposed",
    )
    axes[0].set_title("Bayesian-Optimization-Style Candidate Search", weight="bold")
    axes[0].set_xlabel("Predicted first-cycle QRev (mAh/g)")
    axes[0].set_ylabel("Predicted first-cycle CE (%)")
    axes[0].legend(frameon=False, loc="lower right")
    fig.colorbar(scatter, ax=axes[0], label="Acquisition score (%)")

    domain = axes[1].scatter(
        ranked["domain_confidence_pct"],
        ranked["gp_qrev_std_mah_g"],
        c=ranked["novelty_score"],
        cmap="plasma",
        s=18,
        alpha=0.42,
        linewidths=0,
    )
    axes[1].scatter(
        recommendations["domain_confidence_pct"],
        recommendations["gp_qrev_std_mah_g"],
        facecolors="none",
        edgecolors="#0f766e",
        linewidths=1.8,
        s=110,
    )
    axes[1].set_title("Exploration, Novelty, and Domain Confidence", weight="bold")
    axes[1].set_xlabel("Domain confidence (%)")
    axes[1].set_ylabel("GP QRev uncertainty proxy (mAh/g)")
    fig.colorbar(domain, ax=axes[1], label="Hybrid novelty score")

    fig.suptitle(
        f"SDL screen: {len(ranked):,} generated SiOC/SiOCN candidate designs -> "
        f"{len(recommendations)} proposed experiments",
        fontsize=14,
        weight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        -0.02,
        "Ranking blends design-stage performance, GP expected improvement, uncertainty, "
        "novelty, domain confidence, Pareto status, and diversity. All outputs require human review.",
        ha="center",
        fontsize=10,
        color="#475569",
    )
    for ax in axes:
        ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "sdl_bayesian_candidate_search.png", bbox_inches="tight")
    plt.close(fig)


def draw_box(ax, x: float, y: float, label: str, color: str) -> None:
    box = FancyBboxPatch(
        (x, y),
        0.22,
        0.16,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.6,
        edgecolor=color,
        facecolor="#f8fafc",
    )
    ax.add_patch(box)
    ax.text(x + 0.11, y + 0.08, label, ha="center", va="center", fontsize=10.2, weight="bold")


def draw_arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.6,
            color="#334155",
            connectionstyle="arc3,rad=0.0",
        )
    )


def save_closed_loop_workflow_figure() -> None:
    """Draw the human-reviewed closed-loop SDL workflow."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 5.8), dpi=170)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.94,
        "Closed-Loop SDL / MAP Workflow for SiOC/SiOCN Anode Discovery",
        ha="center",
        va="center",
        fontsize=15,
        weight="bold",
        color="#0f172a",
    )
    ax.text(
        0.5,
        0.885,
        "recommend -> synthesize -> characterize -> measure -> update acquisition -> recommend again",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#475569",
    )

    boxes = {
        "candidate": (0.03, 0.58, "Generate\n2,800 candidates", "#0f766e"),
        "gp": (0.28, 0.58, "GP expected\nimprovement", "#0369a1"),
        "rank": (0.53, 0.58, "Rank + select\nTop 10", "#c2410c"),
        "manifest": (0.77, 0.58, "Human-reviewed\nmanifest", "#7c3aed"),
        "synthesis": (0.77, 0.25, "Synthesize +\ncharacterize", "#64748b"),
        "measure": (0.52, 0.25, "Measure QRev,\nCE, Qcycled", "#64748b"),
        "store": (0.27, 0.25, "Store optional\ncharacterization", "#64748b"),
    }
    for x, y, label, color in boxes.values():
        draw_box(ax, x, y, label, color)

    draw_arrow(ax, (0.25, 0.66), (0.28, 0.66))
    draw_arrow(ax, (0.50, 0.66), (0.53, 0.66))
    draw_arrow(ax, (0.75, 0.66), (0.77, 0.66))
    draw_arrow(ax, (0.88, 0.58), (0.88, 0.41))
    draw_arrow(ax, (0.77, 0.33), (0.74, 0.33))
    draw_arrow(ax, (0.52, 0.33), (0.49, 0.33))
    draw_arrow(ax, (0.38, 0.41), (0.39, 0.58))

    ax.text(
        0.5,
        0.08,
        "Measured characterization descriptors are stored only. Current deployed capacity models remain unchanged.",
        ha="center",
        fontsize=10.5,
        color="#334155",
        weight="bold",
    )
    fig.savefig(FIGURE_DIR / "sdl_closed_loop_workflow.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    save_candidate_search_figure()
    save_closed_loop_workflow_figure()
    print(f"Wrote SDL README figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
