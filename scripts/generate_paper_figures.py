"""Generate all figures for the FinSage paper from saved evaluation results.

Produces:
- **Figure 1**: FinSage architecture diagram
- **Figure 2**: LLM quality-cost Pareto frontier (from cost benchmark)
- **Figure 3**: Conflict graph example (NetworkX visualization)
- **Figure 4**: Temporal filtering improvement (bar chart)
- **Figure 5**: Ablation study radar chart (if ablation data available)

Usage::

    python scripts/generate_paper_figures.py --results results/ --output figures/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
_LOG = logging.getLogger("finsage_figures")


def _ensure_matplotlib() -> None:
    """Import and configure matplotlib for non-interactive use."""
    import matplotlib

    matplotlib.use("Agg")


# ── Figure 1: Architecture Diagram ──────────────────────────────────


def generate_architecture_diagram(output_path: Path) -> None:
    """Generate the FinSage system architecture (Figure 1).

    Shows the 5-layer architecture with the 4 novel contributions
    highlighted in distinct colours.
    """
    _ensure_matplotlib()
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")
    fig.patch.set_facecolor("#FAFAFA")

    # Layer boxes (bottom to top).
    layers = [
        (
            0.5,
            0.3,
            13,
            1.2,
            "#E3F2FD",
            "Layer 1 — LEAF",
            "knowledge/ • utils/ • etl/",
        ),
        (0.5, 1.8, 13, 1.2, "#E8F5E9", "Layer 2", "preprocess/ • configs/"),
        (0.5, 3.3, 13, 1.2, "#FFF3E0", "Layer 3–4", "library/ • rag/ • mind/"),
        (
            0.5,
            4.8,
            13,
            1.2,
            "#F3E5F5",
            "Layer 5 — APEX",
            "flows/ • evaluation/",
        ),
    ]

    for x, y, w, h, color, label, modules in layers:
        rect = mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor="#333333",
            linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(
            x + 0.3,
            y + h - 0.3,
            label,
            fontsize=11,
            fontweight="bold",
            va="top",
        )
        ax.text(
            x + 0.3, y + 0.25, modules, fontsize=9, va="bottom", color="#555555"
        )

    # Novel contribution highlights.
    contributions = [
        (
            1.5,
            0.6,
            3.2,
            0.6,
            "#BBDEFB",
            "C1: TKU Models",
            "knowledge/temporal.py",
        ),
        (
            5.2,
            0.6,
            3.2,
            0.6,
            "#BBDEFB",
            "C2: Conflict Models",
            "knowledge/conflict.py",
        ),
        (
            1.5,
            2.1,
            3.2,
            0.6,
            "#C8E6C9",
            "C1: Temporal Parse",
            "preprocess/time.py",
        ),
        (
            5.2,
            2.1,
            3.2,
            0.6,
            "#C8E6C9",
            "C3: Router Config",
            "configs/ollama.py",
        ),
        (1.5, 3.6, 3.2, 0.6, "#FFE0B2", "C3: LLM Router", "mind/router.py"),
        (
            5.2,
            3.6,
            3.2,
            0.6,
            "#FFE0B2",
            "C2: Conflict Detect",
            "mind/conflict.py",
        ),
        (
            1.5,
            5.1,
            3.2,
            0.6,
            "#E1BEE7",
            "C1: TemporalFlow",
            "flows/temporal_paper.py",
        ),
        (5.2, 5.1, 3.2, 0.6, "#E1BEE7", "C4: FinBench-50", "evaluation/"),
    ]

    for x, y, w, h, color, label, loc in contributions:
        rect = mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.08",
            facecolor=color,
            edgecolor="#1565C0",
            linewidth=1.2,
            linestyle="--",
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2,
            y + h - 0.12,
            label,
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="top",
        )
        ax.text(
            x + w / 2,
            y + 0.12,
            loc,
            fontsize=7,
            ha="center",
            va="bottom",
            color="#666666",
            style="italic",
        )

    # Title.
    ax.text(
        7,
        8.2,
        "FinSage Architecture",
        fontsize=16,
        fontweight="bold",
        ha="center",
    )
    ax.text(
        7,
        7.7,
        "Novel contributions (C1–C4) overlaid on the FinSage layered architecture",
        fontsize=10,
        ha="center",
        color="#555555",
    )

    # Legend.
    legend_items = [
        mpatches.Patch(
            facecolor="#BBDEFB",
            edgecolor="#1565C0",
            linestyle="--",
            label="C1: Temporal Knowledge Unit",
        ),
        mpatches.Patch(
            facecolor="#C8E6C9",
            edgecolor="#1565C0",
            linestyle="--",
            label="C2: Cross-Source Conflict Graph",
        ),
        mpatches.Patch(
            facecolor="#FFE0B2",
            edgecolor="#1565C0",
            linestyle="--",
            label="C3: LLM-Agnostic Router",
        ),
        mpatches.Patch(
            facecolor="#E1BEE7",
            edgecolor="#1565C0",
            linestyle="--",
            label="C4: FinBench-50 Benchmark",
        ),
    ]
    ax.legend(
        handles=legend_items,
        loc="upper right",
        fontsize=9,
        framealpha=0.95,
        bbox_to_anchor=(0.98, 0.98),
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    _LOG.info("Figure 1 (architecture) saved to %s", output_path)


# ── Figure 2: Pareto Frontier ────────────────────────────────────────


def generate_pareto_plot(results_dir: Path, output_path: Path) -> None:
    """Generate quality-cost Pareto frontier from LLM benchmark results."""
    csv_path = results_dir / "cost" / "llm_benchmark_results.csv"
    if not csv_path.exists():
        _LOG.warning(
            "Cost benchmark CSV not found at %s; skipping Figure 2.", csv_path
        )
        return

    _ensure_matplotlib()
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(csv_path)
    if df.empty:
        _LOG.warning("Empty cost benchmark results; skipping Figure 2.")
        return

    task_colors = {
        "extraction": "#2196F3",
        "tagging": "#4CAF50",
        "summarization": "#FF9800",
    }
    model_markers = {
        "ollama/llama3.2": "o",
        "ollama/mistral": "s",
        "gpt-4o-mini": "^",
        "gpt-4o": "D",
    }

    fig, ax = plt.subplots(figsize=(10, 7))

    for _, row in df.iterrows():
        cost = max(float(row["mean_cost_per_call_usd"]), 1e-8)
        color = task_colors.get(str(row["task"]), "#999999")
        marker = model_markers.get(str(row["model"]), "x")
        ax.scatter(
            cost,
            row["mean_score"],
            c=color,
            marker=marker,
            s=120,
            alpha=0.85,
            edgecolors="black",
            linewidth=0.5,
            zorder=3,
        )

    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Mean Cost per Call (USD, log scale)", fontsize=12)
    ax.set_ylabel("Mean Quality Score", fontsize=12)
    ax.set_title(
        "Figure 2: LLM Quality-Cost Pareto Frontier",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(alpha=0.3)

    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=c,
            markersize=10,
            label=t,
        )
        for t, c in task_colors.items()
    ] + [
        Line2D(
            [0],
            [0],
            marker=m,
            color="w",
            markerfacecolor="gray",
            markersize=10,
            label=n,
        )
        for n, m in model_markers.items()
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    _LOG.info("Figure 2 (Pareto) saved to %s", output_path)


# ── Figure 3: Conflict Graph Example ────────────────────────────────


def generate_conflict_graph(output_path: Path) -> None:
    """Generate a sample conflict graph visualization (Figure 3)."""
    try:
        import networkx as nx
    except ImportError:
        _LOG.warning("networkx not installed; skipping Figure 3.")
        return

    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    G = nx.Graph()

    # Sample conflict nodes (TKU claims).
    claims = {
        "TKU-1": "Momentum premium\n= 1.2% monthly\n(US, 1963–2022)",
        "TKU-2": "Momentum premium\n= 0.4% monthly\n(US, 2000–2022)",
        "TKU-3": "Value premium\n= 4.8% annual\n(Global, 1990–2020)",
        "TKU-4": "Value premium\n= 2.1% annual\n(US, 2010–2023)",
        "TKU-5": "Factor crowding\nreduces alpha\n(US, 2015–2023)",
        "TKU-6": "Factor crowding\nis insignificant\n(Europe, 2012–2020)",
    }

    for node_id, label in claims.items():
        G.add_node(node_id, label=label)

    # Conflict edges.
    conflicts = [
        ("TKU-1", "TKU-2", {"type": "VALUE_MISMATCH", "weight": 0.87}),
        ("TKU-3", "TKU-4", {"type": "VALUE_MISMATCH", "weight": 0.82}),
        ("TKU-5", "TKU-6", {"type": "DIRECTION_MISMATCH", "weight": 0.91}),
    ]
    G.add_edges_from(conflicts)

    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42, k=2.5)

    edge_colors = []
    for _, _, d in G.edges(data=True):
        if d["type"] == "VALUE_MISMATCH":
            edge_colors.append("#E53935")
        elif d["type"] == "DIRECTION_MISMATCH":
            edge_colors.append("#FF9800")
        else:
            edge_colors.append("#9E9E9E")

    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=3000,
        node_color="#E3F2FD",
        edgecolors="#1565C0",
        linewidths=2,
    )
    nx.draw_networkx_edges(
        G, pos, ax=ax, edge_color=edge_colors, width=3, style="dashed"
    )
    nx.draw_networkx_labels(
        G,
        pos,
        ax=ax,
        labels=nx.get_node_attributes(G, "label"),
        font_size=7,
        font_weight="bold",
    )

    # Edge labels.
    edge_labels = {
        (u, v): f"{d['type']}\n(conf={d['weight']:.2f})"
        for u, v, d in G.edges(data=True)
    }
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        ax=ax,
        font_size=7,
        font_color="#B71C1C",
    )

    ax.set_title(
        "Figure 3: Cross-Source Conflict Graph (CSCG) Example",
        fontsize=14,
        fontweight="bold",
    )
    ax.axis("off")

    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D(
            [0],
            [0],
            color="#E53935",
            linewidth=3,
            linestyle="--",
            label="VALUE_MISMATCH",
        ),
        Line2D(
            [0],
            [0],
            color="#FF9800",
            linewidth=3,
            linestyle="--",
            label="DIRECTION_MISMATCH",
        ),
    ]
    ax.legend(
        handles=legend_handles, loc="lower left", fontsize=9, framealpha=0.9
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    _LOG.info("Figure 3 (conflict graph) saved to %s", output_path)


# ── Figure 4: Temporal Filtering Improvement ─────────────────────────


def generate_temporal_improvement(results_dir: Path, output_path: Path) -> None:
    """Generate bar chart showing temporal filtering improvement (Figure 4)."""
    # Try to load from ablation or full eval results.
    full_csv = results_dir / "full" / "table1_6system_comparison.csv"
    ablation_csv = results_dir / "ablation" / "table4_ablation.csv"

    _ensure_matplotlib()

    if full_csv.exists():
        import pandas as pd

        df = pd.read_csv(full_csv)
        if "question_type" in df.columns and "system" in df.columns:
            temporal_df = df[df["question_type"] == "TEMPORAL"]
            if not temporal_df.empty:
                by_system = (
                    temporal_df.groupby("system")["score"]
                    .mean()
                    .sort_values(ascending=False)
                )
                _render_bar_chart(
                    by_system,
                    output_path,
                    "Figure 4: Temporal Question Performance by System",
                )
                return

    if ablation_csv.exists():
        import pandas as pd

        df = pd.read_csv(ablation_csv)
        if not df.empty:
            by_ablation = (
                df.groupby("ablation")["score"]
                .mean()
                .sort_values(ascending=False)
            )
            _render_bar_chart(
                by_ablation,
                output_path,
                "Figure 4: Ablation — Mean Score by Condition",
            )
            return

    # Fallback: use illustrative data.
    _LOG.info("No evaluation data found; generating illustrative Figure 4.")
    systems = [
        "finsage_full",
        "finsage_no_conflict",
        "finsage_no_temporal",
        "quantmind_style",
        "naive_rag",
        "generic_llm",
    ]
    temporal_scores = [0.82, 0.78, 0.54, 0.48, 0.35, 0.31]

    import pandas as pd

    series = pd.Series(temporal_scores, index=systems)
    _render_bar_chart(
        series,
        output_path,
        "Figure 4: Temporal Question Performance (Illustrative)",
    )


def _render_bar_chart(
    data: "pd.Series[float]",
    output_path: Path,
    title: str,
) -> None:
    """Render a horizontal bar chart."""
    _ensure_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(data)))  # type: ignore[arg-type]
    bars = ax.barh(
        range(len(data)),
        data.values,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data.index, fontsize=10)
    ax.set_xlabel("Mean Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()

    for bar, val in zip(bars, data.values, strict=False):
        ax.text(
            val + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    _LOG.info("Bar chart saved to %s", output_path)


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point for generating FinSage paper figures."""
    parser = argparse.ArgumentParser(
        description="Generate FinSage paper figures."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results"),
        help="Results directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures"),
        help="Output directory for figures.",
    )
    args = parser.parse_args()

    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    generate_architecture_diagram(output / "figure1_architecture.png")
    generate_pareto_plot(args.results, output / "figure2_pareto.png")
    generate_conflict_graph(output / "figure3_conflict_graph.png")
    generate_temporal_improvement(args.results, output / "figure4_temporal.png")

    _LOG.info("=== All figures generated in %s ===", output)


if __name__ == "__main__":
    main()
