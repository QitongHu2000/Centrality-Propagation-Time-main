"""Shared figure style, input readers and comparison records for the manuscript."""

import csv
import json
import os
from collections import defaultdict

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ptc-revision-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_finite_time import METHODS

from paths import (
    PROJECT_ROOT as ROOT,
    RESULTS,
    REPORTS,
    figure_path,
)

OUT = REPORTS
# Shared curve colors; individual models retain their submitted variants.
COLORS = dict(
    zip(
        METHODS,
        [
            "black",
            "green",
            "brown",
            "#3233A6",
            "#A632A2",
            "#D0B0B0",
            "#00CBD8",
            "#DBB900",
            "#F28E2B",
            "#4F8A82",
            "#FF0000",
        ],
    )
)
MUTUALISTIC_PALETTE = ("#8DDFBF", "#328CA0", "#006837")
STYLES = {q: "-" for q in METHODS}
STYLES.update(SGM="--", HSI="-.")
INPUTS = set()
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
    }
)


def read(name):
    path = RESULTS / name
    INPUTS.add(path)
    with path.open() as stream:
        return list(csv.DictReader(stream))


def save(fig, name):
    fig.savefig(figure_path(name))
    fig.savefig(figure_path(name, ".png"), dpi=180)
    plt.close(fig)


def working_point_groups():
    """Read matched protocols shared by the finite-time and steady figures."""
    groups = defaultdict(dict)
    for row in read("mechanism/working_points.csv"):
        key, mode = (row["model"], row["network"]), row["mode"]
        if mode in groups[key]:
            raise ValueError(f"Duplicate working-point record: {key}/{mode}")
        groups[key][mode] = dict(
            row,
            ptc_rank=int(row["ptc_rank"]),
            ptc_energy=float(row["ptc_energy"]),
            best_energy=float(row["best_energy"]),
        )
    return groups


def build_score_patterns(model):
    """Original four network/scatter blocks, using the full comparison pool."""
    import networkx as nx
    import numpy as np
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    from run_finite_time import SOCIAL, ECO, load_graph

    names = SOCIAL if model == "E" else ECO
    scatter_color = "#D89528" if model == "E" else MUTUALISTIC_PALETTE[1]
    fit_color = "#A45946" if model == "E" else MUTUALISTIC_PALETTE[2]
    network_color = "#895551" if model == "E" else MUTUALISTIC_PALETTE[1]
    methods = list(METHODS)
    matrix_size = len(methods) - 1
    network_start = (matrix_size + 1) // 2
    records = read("finite_time/scores.csv")
    manifest = []
    # The submitted figure uses four triangular matrices with network drawings
    # in their unused upper triangles. Keep this arrangement and guide curves.
    fig = plt.figure(figsize=(11.2, 12.6))
    outer = fig.add_gridspec(
        2,
        2,
        left=0.045,
        right=0.99,
        bottom=0.035,
        # Match the card-to-panel gap to the visible gap between the two rows.
        top=0.8576,
        hspace=0.19,
        wspace=0.10,
    )
    for index, name in enumerate(names):
        cell = outer[index // 2, index % 2]
        inner = cell.subgridspec(matrix_size, matrix_size, wspace=1.0, hspace=0.90)
        nodes = sorted(
            (r for r in records if r["network"] == name), key=lambda r: int(r["node"])
        )
        score = {q: np.array([float(r[q]) for r in nodes]) for q in methods}
        assert all(np.isfinite(v).all() for v in score.values())
        pairs = []
        for i in range(1, len(methods)):
            for j in range(i):
                ax = fig.add_subplot(inner[i - 1, j])
                x, y = score[methods[i]], score[methods[j]]
                ax.scatter(
                    x,
                    y,
                    s=3.5,
                    facecolors="none",
                    edgecolors=scatter_color,
                    linewidths=0.4,
                )
                if len(np.unique(x)) >= 3:
                    curve = np.polynomial.Polynomial.fit(x, y, 2)
                    xx = np.linspace(x.min(), x.max(), 100)
                    ax.plot(xx, curve(xx), color=fit_color, lw=0.65)
                ax.set_xlabel(methods[i], fontsize=6, labelpad=0)
                ax.set_ylabel(methods[j], fontsize=6, labelpad=0)
                for axis in [ax.xaxis, ax.yaxis]:
                    axis.set_major_locator(MaxNLocator(nbins=2))
                    # Compact raw values avoid offset text colliding with method
                    # labels when all eleven methods share the original layout.
                    axis.set_major_formatter(
                        FuncFormatter(
                            lambda value, _: f"{value:.2g}".replace("e+0", "e")
                            .replace("e-0", "e-")
                            .replace("e+", "e")
                        )
                    )
                # Spread the two x ticks across the observed score range so
                # small decimal-valued scores remain legible in compact panels.
                ax.set_xticks(np.unique([x.min(), x.max()]))
                ax.tick_params(labelsize=4, length=1.5, width=0.4, pad=1)
                for spine in ax.spines.values():
                    spine.set_linewidth(0.45)
                pairs.append({"x": methods[i], "y": methods[j]})
        assert len(pairs) == len(methods) * (len(methods) - 1) // 2
        manifest.append(dict(network=name, nodes=len(nodes), scatter_pairs=pairs))
        ax = fig.add_subplot(inner[:network_start, network_start:])
        graph, _, _, metadata = load_graph(name)
        INPUTS.add(ROOT / metadata["input"])
        statistics = {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "mean_degree": 2 * graph.number_of_edges() / graph.number_of_nodes(),
            "mean_clustering": nx.average_clustering(graph),
        }
        manifest[-1]["statistics"] = statistics
        pos = nx.circular_layout(graph)
        # Reuse the submitted arc3 rule, drawing each undirected edge once
        # from its higher circular index to its lower index. arrows=True
        # enables curved patches; arrowstyle="-" keeps the graph undirected.
        node_index = {node: i for i, node in enumerate(graph)}
        curved_edges = defaultdict(list)
        for u, v in graph.edges:
            if node_index[u] < node_index[v]:
                u, v = v, u
            distance = node_index[u] - node_index[v]
            radius = -0.1 if distance == 1 else 0.3 if distance < 10 else -0.3
            curved_edges[radius].append((u, v))
        for radius, edges in curved_edges.items():
            nx.draw_networkx_edges(
                graph,
                pos,
                ax=ax,
                edgelist=edges,
                arrows=True,
                arrowstyle="-",
                connectionstyle=f"arc3,rad={radius}",
                node_size=5,
                width=0.25,
                alpha=0.35,
                edge_color=network_color,
            )
        nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=5, node_color=network_color)
        ax.set_aspect("equal")
        network_box = ax.get_position()
        ax.set_position(
            [
                network_box.x0,
                network_box.y0 + 0.01,
                network_box.width,
                network_box.height,
            ]
        )
        ax.text(
            0.5,
            -0.04,
            name,
            transform=ax.transAxes,
            ha="center",
            fontsize=13,
            fontweight="bold",
        )
        ax.text(
            0.5,
            -0.075,
            rf"$N={statistics['nodes']},\quad L={statistics['edges']}$"
            + "\n"
            + rf"$\bar d={statistics['mean_degree']:.3f},\quad \bar c={statistics['mean_clustering']:.3f}$",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
            linespacing=1.15,
        )
        ax.axis("off")
        bbox = cell.get_position(fig)
        fig.text(
            network_box.x0,
            bbox.y1 + 0.008,
            f"({chr(97+2*index)})",
            fontsize=13,
            fontweight="bold",
        )
        fig.text(
            bbox.x0 - 0.035,
            bbox.y1 + 0.008,
            f"({chr(98+2*index)})",
            fontsize=13,
            fontweight="bold",
        )
    save(fig, f"scores_{model}")
    (OUT / "verification" / f"scores_{model}_panels.json").write_text(
        json.dumps(
            {
                "methods": methods,
                "rule": "Every unordered method pair is plotted once per network.",
                "panels": manifest,
            },
            indent=2,
        )
        + "\n"
    )
