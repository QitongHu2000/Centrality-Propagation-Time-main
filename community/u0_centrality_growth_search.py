from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


U0 = 0
EPSILON = 1e-12
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


CentralityFunction = Callable[[nx.Graph], dict[int, float]]


def degree_scores(graph: nx.Graph) -> dict[int, float]:
    return nx.degree_centrality(graph)


def betweenness_scores(graph: nx.Graph) -> dict[int, float]:
    return nx.betweenness_centrality(graph, normalized=True)


def closeness_scores(graph: nx.Graph) -> dict[int, float]:
    return nx.closeness_centrality(graph)


CENTRALITY_FUNCTIONS: dict[str, CentralityFunction] = {
    "DC": degree_scores,
    "BC": betweenness_scores,
    "CC": closeness_scores,
}


def centrality_share(scores: dict[int, float], u0: int = U0) -> tuple[float, float, float]:
    total_score = float(sum(scores.values()))
    u0_score = float(scores[u0])
    if abs(total_score) <= EPSILON:
        return 1.0 / len(scores), u0_score, total_score
    return u0_score / total_score, u0_score, total_score


def u0_is_maximum(scores: dict[int, float], u0: int = U0) -> bool:
    max_score = max(scores.values())
    return float(scores[u0]) >= max_score - EPSILON


def sample_random_attachment(graph: nx.Graph, rng: np.random.Generator) -> tuple[int, ...]:
    existing_nodes = np.array(sorted(graph.nodes), dtype=int)
    while True:
        mask = rng.random(len(existing_nodes)) < 0.5
        if np.any(mask):
            return tuple(int(node) for node in existing_nodes[mask])


def add_new_node(graph: nx.Graph, attachment: tuple[int, ...]) -> nx.Graph:
    candidate_graph = graph.copy()
    new_node = candidate_graph.number_of_nodes()
    candidate_graph.add_node(new_node)
    candidate_graph.add_edges_from((new_node, node) for node in attachment)
    return candidate_graph


def random_growth_run(
    metric: str,
    umax: int,
    rng: np.random.Generator,
    max_attempts_per_step: int,
    u0: int = U0,
) -> tuple[nx.Graph, list[int], list[float]]:
    centrality_function = CENTRALITY_FUNCTIONS[metric]
    graph = nx.Graph()
    graph.add_node(u0)

    accepted_attempts: list[int] = []
    accepted_shares: list[float] = []

    for _ in range(umax):
        for attempt in range(1, max_attempts_per_step + 1):
            attachment = sample_random_attachment(graph, rng)
            candidate_graph = add_new_node(graph, attachment)
            scores = centrality_function(candidate_graph)
            share, _, _ = centrality_share(scores, u0=u0)
            if u0_is_maximum(scores, u0=u0):
                graph = candidate_graph
                accepted_attempts.append(attempt)
                accepted_shares.append(share)
                break
        else:
            raise RuntimeError(
                f"Failed to find an accepted attachment within {max_attempts_per_step} attempts "
                f"for metric {metric} at node count {graph.number_of_nodes()}."
            )

    return graph, accepted_attempts, accepted_shares


def average_adjacency_matrix(graphs: list[nx.Graph], total_nodes: int) -> np.ndarray:
    accumulator = np.zeros((total_nodes, total_nodes), dtype=float)
    nodelist = list(range(total_nodes))
    for graph in graphs:
        accumulator += nx.to_numpy_array(graph, nodelist=nodelist, dtype=float)
    return accumulator / len(graphs)


def save_average_matrix_outputs(
    metric: str,
    average_matrix: np.ndarray,
    umax: int,
    run_count: int,
) -> tuple[Path, Path]:
    matrix_path = OUTPUT_DIR / f"u0_{metric.lower()}_average_matrix_umax{umax}_runs{run_count}.txt"
    image_path = OUTPUT_DIR / f"u0_{metric.lower()}_average_matrix_umax{umax}_runs{run_count}.pdf"
    np.savetxt(matrix_path, average_matrix, fmt="%.6f")

    figure, axis = plt.subplots(figsize=(6, 6), constrained_layout=True)
    image = axis.imshow(average_matrix, cmap="YlOrRd", vmin=0.0, vmax=1.0)
    axis.set_title(f"{metric} average adjacency matrix")
    axis.set_xlabel("node index")
    axis.set_ylabel("node index")
    tick_step = max(1, average_matrix.shape[0] // 5)
    ticks = list(range(0, average_matrix.shape[0], tick_step))
    if ticks[-1] != average_matrix.shape[0] - 1:
        ticks.append(average_matrix.shape[0] - 1)
    axis.set_xticks(ticks)
    axis.set_yticks(ticks)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="edge frequency")
    figure.savefig(image_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    return matrix_path, image_path


def print_metric_report(
    metric: str,
    average_matrix: np.ndarray,
    mean_attempts: np.ndarray,
    mean_final_share: float,
    matrix_path: Path,
    image_path: Path,
) -> None:
    print(f"\n=== {metric} ===")
    print(f"mean final share of u0: {mean_final_share:.8f}")
    print("mean accepted attempts per step:")
    print(np.array2string(mean_attempts, precision=2, separator=", "))
    print("average adjacency matrix:")
    print(np.array2string(average_matrix, precision=3, suppress_small=True, max_line_width=200))
    print(f"matrix saved to: {matrix_path}")
    print(f"imshow saved to: {image_path}")


if __name__ == "__main__":
    umax = 20
    run_count = 100
    metrics = ["DC", "BC", "CC"]
    random_seed = 20260523
    max_attempts_per_step = 100000

    base_rng = np.random.default_rng(random_seed)

    for metric in metrics:
        graphs: list[nx.Graph] = []
        attempt_histories: list[list[int]] = []
        final_shares: list[float] = []

        for _ in range(run_count):
            run_seed = int(base_rng.integers(0, np.iinfo(np.int32).max))
            run_rng = np.random.default_rng(run_seed)
            graph, accepted_attempts, accepted_shares = random_growth_run(
                metric=metric,
                umax=umax,
                rng=run_rng,
                max_attempts_per_step=max_attempts_per_step,
                u0=U0,
            )
            graphs.append(graph)
            attempt_histories.append(accepted_attempts)
            final_shares.append(accepted_shares[-1])

        average_matrix = average_adjacency_matrix(graphs, total_nodes=umax + 1)
        mean_attempts = np.mean(np.array(attempt_histories, dtype=float), axis=0)
        mean_final_share = float(np.mean(final_shares))
        matrix_path, image_path = save_average_matrix_outputs(
            metric=metric,
            average_matrix=average_matrix,
            umax=umax,
            run_count=run_count,
        )
        print_metric_report(
            metric=metric,
            average_matrix=average_matrix,
            mean_attempts=mean_attempts,
            mean_final_share=mean_final_share,
            matrix_path=matrix_path,
            image_path=image_path,
        )
        fig=plt.figure(figsize=(6,6))
        plt.imshow(average_matrix)
        plt.show()