from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

THETA_J = -1.0
THETA_H = 0.0
THETA_E = -2.0

ENSEMBLE_SIZE = 300

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize(values: np.ndarray) -> np.ndarray:
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if np.isclose(maximum, minimum):
        return np.ones_like(values)
    return (values - minimum) / (maximum - minimum)


def generate_connected_block(
    size: int,
    p_in: float,
) -> nx.Graph:
    if size <= 0:
        return nx.Graph()
    if size == 1:
        block = nx.Graph()
        block.add_node(0)
        return block

    neighbor_count = int(np.ceil((0.10 + p_in) * (size - 1)))
    neighbor_count = max(4, min(size - 1, neighbor_count))
    if neighbor_count % 2 == 1:
        neighbor_count -= 1
    neighbor_count = max(2, neighbor_count)

    for _ in range(1000):
        seed = int(np.random.randint(0, np.iinfo(np.int32).max))
        try:
            return nx.connected_watts_strogatz_graph(size, neighbor_count, 0.04, tries=200, seed=seed)
        except nx.NetworkXError:
            continue

    raise RuntimeError("Failed to generate a connected high-clustering community block.")


def split_sizes(total: int, count: int) -> list[int]:
    if count <= 0:
        return []

    base = total // count
    sizes = [base] * count
    for index in range(total - base * count):
        sizes[index] += 1
    return sizes


def generate_hub_graph() -> tuple[nx.Graph, dict[str, list[list[int]]]]:
    if HUB_COUNT <= 0 or HUB_COUNT >= TOTAL_NODES:
        raise ValueError("HUB_COUNT must be positive and smaller than TOTAL_NODES.")
    if not 0.0 < CONNECTION_DENSITY <= 1.0:
        raise ValueError("CONNECTION_DENSITY must lie in (0, 1].")

    graph = nx.Graph()
    graph.add_nodes_from(range(TOTAL_NODES))

    for index, source in enumerate(HUB_NODES):
        for target in HUB_NODES[index + 1 :]:
            graph.add_edge(source, target)

    community_count = min(3, NON_HUB_COUNT)
    community_sizes = split_sizes(NON_HUB_COUNT, community_count)
    community_anchor_info: list[list[int]] = []
    if NON_HUB_COUNT > 0:
        offset = HUB_COUNT
        for community_size in community_sizes:
            block = generate_connected_block(community_size, CONNECTION_DENSITY)
            block = nx.relabel_nodes(block, {node: node + offset for node in block.nodes})
            graph.add_edges_from(block.edges())

            community_nodes = np.arange(offset, offset + community_size)
            degrees = np.array([graph.degree(node) for node in community_nodes], dtype=float)
            anchor_count = min(community_size, 4)
            anchors = community_nodes[np.argsort(degrees)[:anchor_count]]

            for hub_node in HUB_NODES:
                for anchor_node in anchors:
                    graph.add_edge(hub_node, int(anchor_node))

            community_anchor_info.append([int(node) for node in anchors])
            offset += community_size

        # Shared anchors create redundant inter-community routes, which weakens BC while keeping hub distances short.
        for left_anchors, right_anchors in zip(community_anchor_info[:-1], community_anchor_info[1:]):
            for left_node in left_anchors:
                for right_node in right_anchors:
                    if np.random.rand() < 0.35:
                        graph.add_edge(left_node, right_node)

    if not nx.is_connected(graph):
        raise RuntimeError("The multi-hub construction unexpectedly produced a disconnected graph.")

    return graph, {
        "community_anchors": community_anchor_info,
        "community_sizes": [[int(size)] for size in community_sizes],
    }


def compute_ptc_scores(
    graph: nx.Graph,
    theta_j: float,
    theta_h: float,
    theta_e: float,
    orientation: str = "out",
) -> np.ndarray:
    node_count = graph.number_of_nodes()
    degrees = np.array([graph.degree(node) for node in range(node_count)], dtype=float)
    degree_j = np.power(degrees, theta_j)
    degree_h = np.power(degrees, theta_h)
    degree_e = np.power(degrees, theta_e)

    distances = np.zeros((node_count, node_count), dtype=float)

    for source in range(node_count):
        shortest_lengths = nx.single_source_shortest_path_length(graph, source)
        order = sorted(shortest_lengths.keys(), key=lambda node: shortest_lengths[node])
        path_weight_sum = np.zeros(node_count, dtype=float)
        weighted_time_sum = np.zeros(node_count, dtype=float)
        path_weight_sum[source] = 1.0

        for target in order[1:]:
            current_distance = shortest_lengths[target]
            predecessors = [
                neighbor
                for neighbor in graph.neighbors(target)
                if shortest_lengths.get(neighbor, -1) == current_distance - 1
            ]
            total_weight = 0.0
            total_time = 0.0
            for predecessor in predecessors:
                edge_weight = degree_h[predecessor] * degree_e[target]
                propagated_weight = path_weight_sum[predecessor] * edge_weight
                total_weight += propagated_weight
                total_time += weighted_time_sum[predecessor] * edge_weight
                total_time += propagated_weight * degree_j[target]

            path_weight_sum[target] = total_weight
            weighted_time_sum[target] = total_time
            distances[source, target] = total_time / total_weight

    if orientation == "in":
        totals = np.sum(distances, axis=0)
    elif orientation == "out":
        totals = np.sum(distances, axis=1)
    else:
        raise ValueError("orientation must be 'in' or 'out'.")
    return (node_count - 1) / totals


def compute_scores(graph: nx.Graph) -> dict[str, np.ndarray]:
    ptc_out = compute_ptc_scores(graph, THETA_J, THETA_H, THETA_E, orientation="out")
    closeness = np.array(
        [value for _, value in sorted(nx.closeness_centrality(graph).items())],
        dtype=float,
    )
    betweenness = np.array(
        [value for _, value in sorted(nx.betweenness_centrality(graph, normalized=True).items())],
        dtype=float,
    )
    return {
        "PTC": ptc_out,
        "CC": closeness,
        "BC": betweenness,
    }


def rank_of_node(scores: np.ndarray, node: int) -> int:
    order = np.argsort(-scores, kind="mergesort")
    return int(np.where(order == node)[0][0] + 1)


def average_rank_of_nodes(scores: np.ndarray, nodes: list[int]) -> float:
    ranks = [rank_of_node(scores, node) for node in nodes]
    return float(np.mean(ranks))


def build_layout(graph: nx.Graph) -> dict[int, np.ndarray]:
    initial_positions: dict[int, np.ndarray] = {}

    hub_angles = np.linspace(0.0, 2.0 * np.pi, HUB_COUNT, endpoint=False)
    for index, node in enumerate(HUB_NODES):
        initial_positions[node] = np.array([
            0.55 * np.cos(hub_angles[index]),
            0.55 * np.sin(hub_angles[index]),
        ])

    community_count = min(3, NON_HUB_COUNT)
    community_sizes = split_sizes(NON_HUB_COUNT, community_count)
    community_angles = np.linspace(np.pi / 2.0, np.pi / 2.0 + 2.0 * np.pi, len(community_sizes), endpoint=False)

    offset = HUB_COUNT
    for community_index, community_size in enumerate(community_sizes):
        center = np.array([
            2.55 * np.cos(community_angles[community_index]),
            2.10 * np.sin(community_angles[community_index]),
        ])
        local_angles = np.linspace(0.0, 2.0 * np.pi, community_size, endpoint=False)
        for local_index, node in enumerate(range(offset, offset + community_size)):
            radius = 0.88 + 0.10 * (local_index % 3)
            initial_positions[node] = center + np.array([
                radius * np.cos(local_angles[local_index]),
                radius * np.sin(local_angles[local_index]),
            ])
        offset += community_size

    return nx.spring_layout(
        graph,
        pos=initial_positions,
        fixed=HUB_NODES,
        k=0.48,
        iterations=300,
    )


def plot_network_panel(
    ax: plt.Axes,
    graph: nx.Graph,
    pos: dict[int, np.ndarray],
    scores: np.ndarray,
    measure_name: str,
    hub_rank: float,
    panel_label: str,
) -> None:
    normalized_scores = normalize(scores)

    nx.draw_networkx_edges(graph, pos, ax=ax, width=0.65, alpha=0.22, edge_color="#6c757d")

    node_sizes = 35.0 + 900.0 * np.power(0.25 + normalized_scores, 1.8)
    nodes = nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_size=node_sizes,
        node_color=normalized_scores,
        cmap="YlOrRd",
        linewidths=0.25,
        edgecolors="#f7f7f7",
    )
    nodes.set_clim(0.0, 1.0)
    
    hub_positions = np.array([pos[node] for node in HUB_NODES], dtype=float)
    hub_x, hub_y = np.mean(hub_positions, axis=0)
    ax.text(
        hub_x,
        hub_y + 0.72,
        "hub nodes",
        ha="center",
        va="bottom",
        fontsize=15,
        fontweight="bold",
        color="#1f1f1f",
    )

    # ax.set_title(f"{measure_name} on clustered graph", fontsize=35)
    ax.text(
        0.02,
        0.05,
        f"hub avg rank = {hub_rank:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=25,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82},
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()


def plot_hist_panel(
    ax: plt.Axes,
    ranks: list[float],
    measure_name: str,
    panel_label: str,
) -> None:
    rank_array = np.array(ranks, dtype=float)

    ax.hist(rank_array, bins=20, color="#DC8910", edgecolor="#2f2f2f", alpha=0.75)
    ax.axvline(np.median(rank_array), color="#83272E", linestyle="--", linewidth=3, label="median")
    ax.axvline(np.mean(rank_array), color="#1f77b4", linestyle=":", linewidth=3, label="mean")
    ax.set_xlabel("Average rank", fontsize=35)
    ax.set_ylabel("Count", fontsize=35)
    ax.legend(frameon=False, fontsize=15, loc="upper right")


def save_individual_panels(
    graph: nx.Graph,
    pos: dict[int, np.ndarray],
    representative_scores: dict[str, np.ndarray],
    representative_ranks: dict[str, float],
    ensemble_ranks: dict[str, list[float]],
) -> None:
    panel_specs = [
        ("PTC", "a", "b"),
        ("CC", "c", "d"),
        ("BC", "e", "f"),
    ]
    for measure_name, network_label, hist_label in panel_specs:
        network_figure, network_ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
        plot_network_panel(
            network_ax,
            graph,
            pos,
            representative_scores[measure_name],
            measure_name,
            representative_ranks[measure_name],
            network_label,
        )
        network_path = OUTPUT_DIR / f"hub_node_{measure_name.lower()}_network.pdf"
        network_figure.savefig(network_path, dpi=300, bbox_inches="tight")
        plt.show()

        hist_figure, hist_ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
        bwith=2.5
        hist_ax.spines['bottom'].set_linewidth(bwith)
        hist_ax.spines['left'].set_linewidth(bwith)
        hist_ax.spines['right'].set_linewidth(bwith)
        hist_ax.spines['top'].set_linewidth(bwith)
        # ax.ticklabel_format(style='sci', scilimits=(-1,1), axis='x')
        hist_ax.ticklabel_format(style='sci', scilimits=(-1,1), axis='y')
        
        hist_ax.tick_params(axis='x',labelsize=25)
        hist_ax.tick_params(axis='y',labelsize=25)
        plot_hist_panel(hist_ax, ensemble_ranks[measure_name], measure_name, hist_label)
        hist_path = OUTPUT_DIR / f"hub_node_{measure_name.lower()}_hist.pdf"
        hist_figure.savefig(hist_path, dpi=300, bbox_inches="tight")
        plt.show()


def summarize_ranks(ranks: list[float]) -> dict[str, float]:
    rank_array = np.array(ranks, dtype=float)
    return {
        "mean": float(np.mean(rank_array)),
        "median": float(np.median(rank_array)),
        "std": float(np.std(rank_array)),
        "min": float(np.min(rank_array)),
        "max": float(np.max(rank_array)),
    }


if __name__ == "__main__":
    HUB_COUNT = np.random.randint(3,7)
    HUB_NODES = list(range(HUB_COUNT))
    CONNECTION_DENSITY = (1+np.random.rand())*0.2
    TOTAL_NODES = np.random.randint(60,150)
    NON_HUB_COUNT = TOTAL_NODES - HUB_COUNT
    representative_graph, representative_structure = generate_hub_graph()
    representative_scores = compute_scores(representative_graph)
    representative_ranks = {
        measure: average_rank_of_nodes(scores, HUB_NODES)
        for measure, scores in representative_scores.items()
    }

    ensemble_ranks = {"PTC": [], "CC": [], "BC": []}
    for _ in range(ENSEMBLE_SIZE):
        HUB_COUNT = np.random.randint(3,7)
        HUB_NODES = list(range(HUB_COUNT))
        CONNECTION_DENSITY = (1+np.random.rand())*0.1
        TOTAL_NODES = np.random.randint(60,150)
        NON_HUB_COUNT = TOTAL_NODES - HUB_COUNT
        graph, _ = generate_hub_graph()
        scores = compute_scores(graph)
        for measure_name, values in scores.items():
            ensemble_ranks[measure_name].append(average_rank_of_nodes(values, HUB_NODES))

    positions = build_layout(representative_graph)
    save_individual_panels(
        representative_graph,
        positions,
        representative_scores,
        representative_ranks,
        ensemble_ranks,
    )