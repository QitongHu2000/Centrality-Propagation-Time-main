"""Source-oriented shortest-path PTC; no all-pairs distance matrix.

Each source uses a BFS and two accumulators. The recurrence is exactly the
weighted shortest-path definition in main_final.tex, subject to floating-point
arithmetic. Input labels are retained. Disconnected and directed graphs are
rejected because the manuscript definition assumes a connected simple graph.
"""

from collections import deque
import math

import networkx as nx


def ptc_scores(graph, theta_j=-1.0, theta_h=0.0, theta_e=-2.0):
    if graph.is_directed() or graph.is_multigraph() or nx.number_of_selfloops(graph):
        raise ValueError("PTC requires a simple undirected graph without self-loops")
    nodes = list(graph)
    n = len(nodes)
    if n < 2 or not nx.is_connected(graph):
        raise ValueError("PTC requires a connected graph with at least two nodes")
    if not all(math.isfinite(x) for x in (theta_j, theta_h, theta_e)):
        raise ValueError("Exponents must be finite")
    index = {node: i for i, node in enumerate(nodes)}
    neighbors = [[index[v] for v in graph[u]] for u in nodes]
    degree = [len(row) for row in neighbors]
    cost = [d**theta_j for d in degree]
    head = [d**theta_h for d in degree]
    tail = [d**theta_e for d in degree]
    result = {}
    for source in range(n):
        depth = [-1] * n
        weight = [0.0] * n
        weighted_cost = [0.0] * n
        depth[source] = 0
        weight[source] = 1.0
        queue = deque([source])
        total_distance = 0.0
        while queue:
            u = queue.popleft()
            if weight[u] <= 0 or not math.isfinite(weight[u]):
                raise FloatingPointError(
                    "Path weights underflowed/overflowed; use log-domain accumulation"
                )
            distance = weighted_cost[u] / weight[u]
            if not math.isfinite(distance):
                raise FloatingPointError("Non-finite weighted distance")
            total_distance += distance
            next_depth = depth[u] + 1
            for v in neighbors[u]:
                if depth[v] == -1:
                    depth[v] = next_depth
                    queue.append(v)
                if depth[v] == next_depth:
                    edge_weight = head[u] * tail[v]
                    weight[v] += weight[u] * edge_weight
                    weighted_cost[v] += (
                        weighted_cost[u] + weight[u] * cost[v]
                    ) * edge_weight
        result[nodes[source]] = (n - 1) / total_distance
    return result


def shannon_gravity_scores(graph):
    """SGM: Zhang et al. (2025), doi:10.1038/s41598-025-93387-6,
    equations (15)-(19). Two-hop neighborhood excludes the source. No fitting.
    """
    degree = dict(graph.degree())
    core = nx.core_number(graph)
    clustering = nx.average_clustering(graph, count_zeros=True)
    capacity = {
        u: degree[u] * math.log2(1 + sum(degree[v] for v in graph[u])) for u in graph
    }
    attraction = {u: sum(core[v] for v in graph[u]) ** 2 for u in graph}
    result = {}
    for u in graph:
        distances = nx.single_source_shortest_path_length(graph, u, cutoff=2)
        result[u] = (
            attraction[u]
            * capacity[u]
            * sum(
                (2 * clustering * capacity[v] + 1) / distance**2
                for v, distance in distances.items()
                if v != u
            )
        )
    return result


def local_propagation_scores(graph, radius=3, omega=0.5):
    """LPP recurrence written in the manuscript, using target predecessors.
    A layer with no within-layer edges has eta=0. No label-based tuning.
    """
    degrees = dict(graph.degree())
    gamma = omega * sum(degrees.values()) / sum(d * d for d in degrees.values())
    core = nx.core_number(graph)
    result = {}
    for source in graph:
        distances = nx.single_source_shortest_path_length(graph, source, cutoff=radius)
        probability = {source: 1.0}
        total = 0.0
        for level in range(1, radius + 1):
            layer = [u for u, distance in distances.items() if distance == level]
            layer_set = set(layer)
            within = {u: sum(v in layer_set for v in graph[u]) for u in layer}
            maximum = max(within.values(), default=0)
            for u in layer:
                if level == 1:
                    probability[u] = gamma
                else:
                    probability[u] = 1 - math.prod(
                        1 - gamma * probability[v]
                        for v in graph[u]
                        if distances.get(v) == level - 1
                    )
                eta = within[u] / maximum if maximum else 0.0
                total += probability[u] * (1 + eta) * core[u]
        result[source] = total
    return result


def hsi_scores(graph, multiplier=1.0):
    """Wang et al., Chaos Solitons Fractals 186 (2024) 115227, Eqs. 2-5.

    Use synchronous within-layer updates, all BFS layers, and beta=multiplier*
    <k>/(<k^2>-<k>). beta=beta_c is fixed before seeing response labels.
    This transfers the published SIR surrogate as a structural comparator;
    its probability parameter is not equated with an ODE coupling rate.
    """
    degrees = [graph.degree(u) for u in graph]
    denominator = sum(d * d - d for d in degrees)
    if denominator <= 0:
        raise ValueError("HSI threshold is undefined on this graph")
    beta = multiplier * sum(degrees) / denominator
    if not 0 <= beta <= 1:
        raise ValueError("HSI infection probability must lie in [0,1]")
    result = {}
    for source in graph:
        depths = nx.single_source_shortest_path_length(graph, source)
        layers = {}
        for node, depth in depths.items():
            layers.setdefault(depth, []).append(node)
        probability = {source: 1.0}
        for depth in range(1, max(layers) + 1):
            layer = layers[depth]
            first = {
                v: 1
                - math.prod(
                    1 - beta * probability[q]
                    for q in graph[v]
                    if depths[q] == depth - 1
                )
                for v in layer
            }
            for v in layer:
                within = 1 - math.prod(
                    1 - beta * first[q] for q in graph[v] if depths[q] == depth
                )
                probability[v] = first[v] + (1 - first[v]) * within
        result[source] = sum(probability.values())
    return result
