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


def generate_connected_block(size: int, p_in: float) -> nx.Graph:
    for _ in range(10000):
        block = nx.fast_gnp_random_graph(size, p_in)
        if nx.is_connected(block):
            return block
    raise RuntimeError("Failed to generate a connected SBM block.")


def choose_anchor_nodes(
    community_nodes: np.ndarray,
    graph: nx.Graph,
    count: int,
) -> np.ndarray:
    degrees = np.array([graph.degree(node) for node in community_nodes], dtype=float)
    order = np.argsort(degrees)
    candidate_nodes = community_nodes[order[: max(4 * count, count)]]
    if len(candidate_nodes) < count:
        candidate_nodes = community_nodes
    return np.sort(np.random.choice(candidate_nodes, size=count, replace=False))


def generate_bridge_graph() -> tuple[nx.Graph, dict[str, list[int]]]:
    left = generate_connected_block(N_LEFT, P_LEFT,)
    right = nx.relabel_nodes(generate_connected_block(N_RIGHT, P_RIGHT), lambda node: node + N_LEFT)

    graph = nx.Graph()
    graph.add_nodes_from(range(TOTAL_NODES))
    graph.add_edges_from(left.edges())
    graph.add_edges_from(right.edges())

    left_nodes = np.arange(N_LEFT)
    right_nodes = np.arange(N_LEFT, N_LEFT + N_RIGHT)

    left_anchor_info: list[int] = []
    right_anchor_info: list[int] = []
    for bridge_node in BRIDGE_NODES:
        left_anchors = choose_anchor_nodes(left_nodes, graph, BRIDGE_LINKS_LEFT)
        right_anchors = choose_anchor_nodes(right_nodes, graph, BRIDGE_LINKS_RIGHT)

        for node in left_anchors:
            graph.add_edge(bridge_node, int(node))
        for node in right_anchors:
            graph.add_edge(bridge_node, int(node))

        left_anchor_info.extend(int(node) for node in left_anchors)
        right_anchor_info.extend(int(node) for node in right_anchors)

    if not nx.is_connected(graph):
        raise RuntimeError("The bridge construction unexpectedly produced a disconnected graph.")

    anchor_info = {
        "left_anchors": left_anchor_info,
        "right_anchors": right_anchor_info,
    }
    return graph, anchor_info


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

    left_angles = np.linspace(0.0, 2.0 * np.pi, N_LEFT, endpoint=False)
    for index, node in enumerate(range(N_LEFT)):
        initial_positions[node] = np.array([
            -2.2 + 0.95 * np.cos(left_angles[index]),
            1.25 * np.sin(left_angles[index]),
        ])

    right_angles = np.linspace(0.0, 2.0 * np.pi, N_RIGHT, endpoint=False)
    for index, node in enumerate(range(N_LEFT, N_LEFT + N_RIGHT)):
        initial_positions[node] = np.array([
            2.2 + 0.95 * np.cos(right_angles[index]),
            1.25 * np.sin(right_angles[index]),
        ])

    bridge_offsets = np.linspace(-0.35, 0.35, BRIDGE_COUNT)
    for index, node in enumerate(BRIDGE_NODES):
        initial_positions[node] = np.array([0.0, bridge_offsets[index]])

    return nx.spring_layout(
        graph,
        pos=initial_positions,
        fixed=BRIDGE_NODES,
        k=0.42,
        iterations=300,
    )


def plot_network_panel(
    ax: plt.Axes,
    graph: nx.Graph,
    pos: dict[int, np.ndarray],
    scores: np.ndarray,
    measure_name: str,
    bridge_rank: float,
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

    bridge_positions = np.array([pos[node] for node in BRIDGE_NODES], dtype=float)
    bridge_x, bridge_y = np.mean(bridge_positions, axis=0)
    ax.text(
        bridge_x,
        bridge_y + 0.3,
        "Bridges",
        ha="center",
        va="bottom",
        fontsize=15,
        fontweight="bold",
        color="#1f1f1f",
    )

    # ax.set_title(f"{measure_name} on bridge graph", fontsize=35)
    ax.text(
        0.02,
        0.05,
        f"bridge avg rank = {bridge_rank:.2f}",
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
        network_path = OUTPUT_DIR / f"community_bridge_{measure_name.lower()}_network.pdf"
        plt.savefig(network_path, dpi=300, bbox_inches="tight")
        plt.show()

        hist_figure, hist_ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
        plot_hist_panel(hist_ax, ensemble_ranks[measure_name], measure_name, hist_label)
        bwith=2.5
        hist_ax.spines['bottom'].set_linewidth(bwith)
        hist_ax.spines['left'].set_linewidth(bwith)
        hist_ax.spines['right'].set_linewidth(bwith)
        hist_ax.spines['top'].set_linewidth(bwith)
        # ax.ticklabel_format(style='sci', scilimits=(-1,1), axis='x')
        hist_ax.ticklabel_format(style='sci', scilimits=(-1,1), axis='y')
        
        hist_ax.tick_params(axis='x',labelsize=25)
        hist_ax.tick_params(axis='y',labelsize=25)
        hist_path = OUTPUT_DIR / f"community_bridge_{measure_name.lower()}_hist.pdf"
        plt.savefig(hist_path, dpi=300, bbox_inches="tight")
        plt.show()


def summarize_ranks(ranks: list[int]) -> dict[str, float]:
    rank_array = np.array(ranks, dtype=float)
    return {
        "mean": float(np.mean(rank_array)),
        "median": float(np.median(rank_array)),
        "std": float(np.std(rank_array)),
        "min": float(np.min(rank_array)),
        "max": float(np.max(rank_array)),
    }


if __name__ == "__main__":
    BRIDGE_COUNT = int(np.random.randint(2, 4))
    N_LEFT = np.random.randint(50,70)
    N_RIGHT = np.random.randint(100,150) - N_LEFT - BRIDGE_COUNT
    TOTAL_NODES = N_LEFT + N_RIGHT + BRIDGE_COUNT
    BRIDGE_NODES = list(range(N_LEFT + N_RIGHT, TOTAL_NODES))
    
    P_LEFT = (1 + np.random.rand()) * 0.1
    P_RIGHT = (1 + np.random.rand()) * 0.1
    BRIDGE_LINKS_LEFT = np.random.randint(5, 8)
    BRIDGE_LINKS_RIGHT = np.random.randint(5, 8)

    representative_config = {
        "BRIDGE_COUNT": BRIDGE_COUNT,
        "N_LEFT": N_LEFT,
        "N_RIGHT": N_RIGHT,
        "TOTAL_NODES": TOTAL_NODES,
        "BRIDGE_NODES": BRIDGE_NODES.copy(),
        "P_LEFT": P_LEFT,
        "P_RIGHT": P_RIGHT,
        "BRIDGE_LINKS_LEFT": BRIDGE_LINKS_LEFT,
        "BRIDGE_LINKS_RIGHT": BRIDGE_LINKS_RIGHT,
    }
    
    representative_graph, representative_anchors = generate_bridge_graph()
    representative_scores = compute_scores(representative_graph)
    representative_ranks = {
        measure: average_rank_of_nodes(scores, BRIDGE_NODES)
        for measure, scores in representative_scores.items()
    }

    ensemble_ranks = {"PTC": [], "CC": [], "BC": []}
    for _ in range(ENSEMBLE_SIZE):
        BRIDGE_COUNT = int(np.random.randint(2, 4))
        N_LEFT = np.random.randint(20,80)
        N_RIGHT = 100 - N_LEFT - BRIDGE_COUNT
        TOTAL_NODES = N_LEFT + N_RIGHT + BRIDGE_COUNT
        BRIDGE_NODES = list(range(N_LEFT + N_RIGHT, TOTAL_NODES))
        
        P_LEFT = (1 + np.random.rand()) * 0.2
        P_RIGHT = (1 + np.random.rand()) * 0.2
        BRIDGE_LINKS_LEFT = np.random.randint(3, 5)
        BRIDGE_LINKS_RIGHT = np.random.randint(3, 5)
        graph, _ = generate_bridge_graph()
        scores = compute_scores(graph)
        for measure_name, values in scores.items():
            ensemble_ranks[measure_name].append(average_rank_of_nodes(values, BRIDGE_NODES))

    BRIDGE_COUNT = representative_config["BRIDGE_COUNT"]
    N_LEFT = representative_config["N_LEFT"]
    N_RIGHT = representative_config["N_RIGHT"]
    TOTAL_NODES = representative_config["TOTAL_NODES"]
    BRIDGE_NODES = representative_config["BRIDGE_NODES"]
    P_LEFT = representative_config["P_LEFT"]
    P_RIGHT = representative_config["P_RIGHT"]
    BRIDGE_LINKS_LEFT = representative_config["BRIDGE_LINKS_LEFT"]
    BRIDGE_LINKS_RIGHT = representative_config["BRIDGE_LINKS_RIGHT"]

    positions = build_layout(representative_graph)
    save_individual_panels(
        representative_graph,
        positions,
        representative_scores,
        representative_ranks,
        ensemble_ranks,
    )