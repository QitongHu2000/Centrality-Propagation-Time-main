"""Steady-response, ranking-cost and core checks for the maintained E/M manuscript.

Run from any directory with Python, NumPy, SciPy, NetworkX and Matplotlib.
Examples:
  python run_experiments.py check
  python run_experiments.py benchmark
  python run_experiments.py response
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import networkx as nx
import numpy as np
import scipy
from scipy.linalg import solve
from scipy.optimize import root
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
from scipy.stats import spearmanr

from centrality import (
    ptc_scores,
    shannon_gravity_scores,
    local_propagation_scores,
    hsi_scores,
)

from paths import FIGURE_MODEL, NETWORKS, PROJECT_ROOT as ROOT, RESULTS as OUT

SEED = 20260527
METHODS = [
    "DC",
    "EC",
    "CC",
    "BC",
    "PR",
    "GC",
    "SP",
    "LPP",
    "SGM",
    "HSI",
    "PTC",
    "PTC_cost",
]


def save_json(name, data):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def save_csv(name, rows):
    OUT.mkdir(exist_ok=True)
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metadata():
    return {
        "seed": SEED,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "networkx": nx.__version__,
        "platform": platform.platform(),
        "processor_architecture": platform.machine(),
        "cpu_model": "not exposed by the execution sandbox",
        "thread_environment": {
            key: os.environ.get(key)
            for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]
        },
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), Path(__file__).with_name("centrality.py")]
        },
    }


def graph_case(family, n, seed):
    if family == "BA":
        return nx.barabasi_albert_graph(n, 3, seed=seed)
    if family == "WS":
        return nx.connected_watts_strogatz_graph(n, 6, 0.1, tries=100, seed=seed)
    raise ValueError(family)


def canonical_graph(graph):
    """Rebuild in numeric raw-ID order, retaining the original label on each node."""
    ordered = sorted(
        graph,
        key=lambda u: (0, int(u)) if str(u).lstrip("-").isdigit() else (1, str(u)),
    )
    mapping = {u: i for i, u in enumerate(ordered)}
    result = nx.Graph()
    result.add_nodes_from((i, {"original_id": u}) for i, u in enumerate(ordered))
    result.add_edges_from((mapping[u], mapping[v]) for u, v in graph.edges())
    return result


def real_graphs():
    for name in ["dolphins", "football", "karate", "polbooks"]:
        path = NETWORKS / (name + ".gml")
        graph = nx.Graph(nx.read_gml(path, label=None))
        graph.remove_edges_from(nx.selfloop_edges(graph))
        graph = graph.subgraph(max(nx.connected_components(graph), key=len)).copy()
        yield name, canonical_graph(graph), path
    for name in ["everglades", "florida", "mangwet", "stmarks"]:
        path = NETWORKS / ("eco-" + name + ".edges")
        weighted = nx.read_weighted_edgelist(path)
        graph = nx.Graph()
        graph.add_nodes_from(weighted)
        graph.add_edges_from(
            (u, v)
            for u, v, d in weighted.edges(data=True)
            if u != v and d["weight"] > 0.1
        )
        graph = graph.subgraph(max(nx.connected_components(graph), key=len)).copy()
        yield name, canonical_graph(graph), path


def brute_ptc(graph, theta):
    tj, th, te = theta
    degree = dict(graph.degree())
    scores = {}
    for source in graph:
        total = 0.0
        for target in graph:
            if target == source:
                continue
            weights, costs = [], []
            for path in nx.all_shortest_paths(graph, source, target):
                weights.append(
                    math.prod(
                        degree[u] ** th * degree[v] ** te
                        for u, v in zip(path, path[1:])
                    )
                )
                costs.append(sum(degree[v] ** tj for v in path[1:]))
            total += np.dot(weights, costs) / sum(weights)
        scores[source] = (len(graph) - 1) / total
    return scores


def values(mapping, graph):
    return np.array([mapping[u] for u in graph], dtype=float)


def check():
    graphs = [g for g in nx.graph_atlas_g() if 2 <= len(g) <= 6 and nx.is_connected(g)]
    triples = [(-1, 0, -2), (-1, -2, 0), (0, 1.2, -0.3), (-0.5, 0, -0.5)]
    maximum = 0.0
    for graph in graphs:
        for theta in triples:
            actual = values(ptc_scores(graph, *theta), graph)
            expected = values(brute_ptc(graph, theta), graph)
            maximum = max(maximum, float(np.max(np.abs(actual - expected))))
            np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(
            values(ptc_scores(graph, -1, 0, -2), graph),
            values(ptc_scores(graph, -1, -2, 0), graph),
            rtol=2e-12,
        )
        np.testing.assert_allclose(
            values(ptc_scores(graph, 0, 1.2, -0.3), graph),
            values(nx.closeness_centrality(graph), graph),
            rtol=2e-12,
        )
    graph = nx.relabel_nodes(nx.karate_club_graph(), lambda u: "node-" + str(u))
    assert set(ptc_scores(graph)) == set(graph)
    for bad in [nx.empty_graph(2), nx.DiGraph([(0, 1)]), nx.empty_graph(1)]:
        try:
            ptc_scores(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid graph was accepted")
    # A star has no triangles: eta_G=0 gives a simple independent SGM check.
    graph = nx.star_graph(4)
    score = shannon_gravity_scores(graph)
    np.testing.assert_allclose(score[0], 16 * (4 * math.log2(5)) * 4)
    np.testing.assert_allclose(score[1], math.log2(5) * (1 + 3 / 4))
    report = metadata() | {
        "connected_atlas_graphs": len(graphs),
        "exponent_triples": len(triples),
        "maximum_absolute_error": maximum,
        "status": "passed",
        "checks": [
            "enumeration equivalence",
            "E/M exponent equivalence",
            "exact closeness when theta_J=0",
            "arbitrary node labels",
            "invalid input rejection",
            "SGM star formula",
        ],
    }
    save_json("verification.json", report)
    print(json.dumps(report), flush=True)


def centrality(graph, method, theta=(-1, 0, -2)):
    if method == "PTC":
        return ptc_scores(graph, *theta)
    if method == "PTC_cost":
        return ptc_scores(graph, theta[0], 0, 0)
    if method == "DC":
        return nx.degree_centrality(graph)
    if method == "EC":
        return nx.eigenvector_centrality(graph, max_iter=10000, tol=1e-10, weight=None)
    if method == "CC":
        return nx.closeness_centrality(graph)
    if method == "BC":
        return nx.betweenness_centrality(graph, normalized=True, weight=None)
    if method == "PR":
        return nx.pagerank(graph, alpha=0.9, tol=1e-10, max_iter=10000, weight=None)
    if method == "HSI":
        return hsi_scores(graph)
    if method == "SGM":
        return shannon_gravity_scores(graph)
    if method == "LPP":
        return local_propagation_scores(graph)
    if method == "GC":
        degree = dict(graph.degree())
        return {
            u: sum(
                degree[u] * degree[v] / d**2
                for v, d in nx.single_source_shortest_path_length(graph, u).items()
                if v != u
            )
            for u in graph
        }
    if method == "SP":
        mean_degree = 2 * graph.number_of_edges() / len(graph)
        result = {}
        for source in graph:
            depth = nx.single_source_shortest_path_length(graph, source, cutoff=3)
            counts = {source: 1}
            for u in sorted(depth, key=depth.get):
                if u != source:
                    counts[u] = sum(
                        counts[v] for v in graph[u] if depth.get(v) == depth[u] - 1
                    )
            result[source] = sum(
                counts[u] * mean_degree ** (-depth[u]) for u in depth if u != source
            )
        return result
    raise ValueError(method)


def worker(args):
    graph = graph_case(args.family, args.n, args.seed)
    start = time.perf_counter()
    score = centrality(graph, args.method)
    elapsed = time.perf_counter() - start
    assert all(np.isfinite(list(score.values())))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mib = rss / (1024**2 if sys.platform == "darwin" else 1024)
    print(
        json.dumps(
            {
                "family": args.family,
                "n": len(graph),
                "m": graph.number_of_edges(),
                "seed": args.seed,
                "method": args.method,
                "seconds": elapsed,
                "peak_process_rss_mib": rss_mib,
            }
        ),
        flush=True,
    )


def benchmark(sizes):
    rows = []
    for family in ["BA", "WS"]:
        for n in sizes:
            for repeat in range(3):
                for method in ["PTC", "CC", "BC", "SGM"]:
                    command = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "worker",
                        "--family",
                        family,
                        "--n",
                        str(n),
                        "--seed",
                        str(SEED + repeat),
                        "--method",
                        method,
                    ]
                    result = subprocess.run(
                        command, check=True, capture_output=True, text=True, timeout=240
                    )
                    row = json.loads(result.stdout)
                    rows.append(row)
                    save_csv("scaling.csv", rows)
                    print(json.dumps(row), flush=True)
    save_json(
        "scaling_metadata.json",
        metadata()
        | {
            "sizes": sizes,
            "repetitions": 3,
            "methods": ["PTC", "CC", "BC", "SGM"],
            "timing_scope": "ranking, graph generation excluded",
            "memory_scope": "peak process RSS, includes Python interpreter, libraries and graph",
            "per_case_timeout_seconds": 240,
        },
    )


def model_equilibrium(adjacency, model):
    """Positive equilibria with unit interaction strength, as in the manuscript."""
    n = len(adjacency)
    if model == "E":
        theta = (-1, 0, -2)
        x = np.full(n, 0.5)

        def fixed(y):
            coupling = adjacency @ y
            return coupling / (1 + coupling)

        def fun(y):
            return -y + (1 - y) * (adjacency @ y)

        def jac(y):
            return np.diag(-1 - adjacency @ y) + (1 - y)[:, None] * adjacency

        parameters = {"B": 1.0}
    elif model == "M":
        theta = (-1, -2, 0)
        x = np.full(n, 0.1)

        def fixed(y):
            return 0.1 * (1 + 5 * (adjacency @ (y / (1 + y))))

        def fun(y):
            return 0.2 * y * (1 - y / 0.1) + y * (adjacency @ (y / (1 + y)))

        def jac(y):
            return (
                np.diag(0.2 * (1 - 2 * y / 0.1) + adjacency @ (y / (1 + y)))
                + y[:, None] * adjacency / (1 + y[None, :]) ** 2
            )

        parameters = {"B": 0.2, "C": 0.1}
    else:
        raise ValueError(model)
    for iteration in range(20000):
        next_x = (x + fixed(x)) / 2
        if np.max(np.abs(next_x - x)) < 1e-12:
            x = next_x
            break
        x = next_x
    else:
        raise RuntimeError("Equilibrium iteration did not converge")
    residual = float(np.max(np.abs(fun(x))))
    if residual > 1e-8 or np.min(x) <= 0:
        raise RuntimeError(f"Invalid equilibrium: residual={residual}")
    jacobian = jac(x)
    # Positive off-diagonal coupling makes the leading eigenvalue real.
    leading = float(
        np.real(
            eigs(
                csr_matrix(jacobian),
                k=1,
                which="LR",
                return_eigenvectors=False,
                tol=1e-9,
                v0=np.ones(n),
            )[0]
        )
    )
    if leading >= -1e-8:
        raise RuntimeError(f"Unstable equilibrium: eigenvalue={leading}")
    susceptibility = solve(-jacobian, np.eye(n), assume_a="gen")
    response = susceptibility / np.diag(susceptibility)[None, :]
    np.testing.assert_allclose(np.diag(response), 1, rtol=1e-12, atol=1e-12)
    energy = np.sum(response**2, axis=0) - 1
    if np.min(energy) < -1e-10:
        raise RuntimeError("Negative response energy")
    return x, fun, jac, energy, response, theta, parameters, residual, leading


def top_mean(score, energy, k):
    """Fractionally include a tied boundary, avoiding node-ID-dependent selection."""
    score = np.asarray(score)
    threshold = np.sort(score)[-k]
    tied = np.isclose(score, threshold, rtol=1e-10, atol=1e-12)
    above = (score > threshold) & ~tied
    remaining = k - int(above.sum())
    return float((energy[above].sum() + remaining * energy[tied].mean()) / k)


def response_experiment(sizes):
    rows, nodes, equilibrium_rows, finite_checks = [], [], [], []
    cases = [
        (
            name,
            graph,
            str(path.relative_to(ROOT)),
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for name, graph, path in real_graphs()
    ]
    cases += [
        (f"{family}_{n}", graph_case(family, n, SEED), "synthetic", "")
        for family in ["BA", "WS"]
        for n in sizes
    ]
    for name, graph, source_path, source_hash in cases:
        a = nx.to_numpy_array(graph, weight=None)
        base = {
            method: values(centrality(graph, method), graph)
            for method in METHODS
            if not method.startswith("PTC")
        }
        for model in [FIGURE_MODEL]:
            start = time.perf_counter()
            x, fun, jac, energy, linear, theta, parameters, residual, leading = (
                model_equilibrium(a, model)
            )
            validation_seconds = time.perf_counter() - start
            scores = dict(base)
            scores["PTC"] = values(centrality(graph, "PTC", theta), graph)
            scores["PTC_cost"] = values(centrality(graph, "PTC_cost", theta), graph)
            equilibrium_rows.append(
                {
                    "network": name,
                    "model": model,
                    "n": len(graph),
                    "m": graph.number_of_edges(),
                    "parameters": json.dumps(parameters),
                    "max_equilibrium_residual": residual,
                    "leading_eigenvalue": leading,
                    "linear_validation_seconds": validation_seconds,
                    "source_file": source_path,
                    "source_sha256": source_hash,
                }
            )
            for method, score in scores.items():
                correlation = float(spearmanr(score, energy).statistic)
                if not math.isfinite(correlation):
                    raise RuntimeError(f"Undefined correlation: {name}/{method}")
                for selection, k in [
                    ("top10", min(10, len(graph))),
                    ("top5pct", max(1, math.ceil(0.05 * len(graph)))),
                ]:
                    rows.append(
                        {
                            "network": name,
                            "model": model,
                            "n": len(graph),
                            "selection": selection,
                            "k": k,
                            "method": method,
                            "mean_energy": top_mean(score, energy, k),
                            "spearman": correlation,
                        }
                    )
            for i in range(len(graph)):
                nodes.append(
                    {
                        "network": name,
                        "model": model,
                        "node": i,
                        "original_id": str(graph.nodes[i].get("original_id", i)),
                        "equilibrium": float(x[i]),
                        "energy": float(energy[i]),
                        **{method: float(score[i]) for method, score in scores.items()},
                    }
                )
            # Independent finite-perturbation root solves for three prespecified
            # node IDs on the real graphs only; these are validation, not fitting.
            if source_path != "synthetic":
                for source in sorted(set([0, len(graph) // 2, len(graph) - 1])):
                    other = np.array([i for i in range(len(graph)) if i != source])
                    for fraction in [0.01, 0.005]:
                        delta = fraction * (
                            min(x[source], 1 - x[source]) if model == "E" else x[source]
                        )
                        pinned = x.copy()
                        pinned[source] += delta

                        def assemble(y):
                            state = pinned.copy()
                            state[other] = y
                            return state

                        solution = root(
                            lambda y: fun(assemble(y))[other],
                            x[other] + delta * linear[other, source],
                            jac=lambda y: jac(assemble(y))[np.ix_(other, other)],
                            tol=1e-10,
                        )
                        state = assemble(solution.x)
                        root_residual = float(np.max(np.abs(fun(state)[other])))
                        if root_residual > 1e-8 or np.min(state) < 0:
                            raise RuntimeError("Pinned equilibrium check failed")
                        finite = float(np.sum(((state[other] - x[other]) / delta) ** 2))
                        finite_checks.append(
                            {
                                "network": name,
                                "model": model,
                                "source": source,
                                "relative_perturbation": fraction,
                                "linear_energy": float(energy[source]),
                                "finite_energy": finite,
                                "relative_error": abs(finite / energy[source] - 1),
                                "pinned_equilibrium_residual": root_residual,
                            }
                        )
            print(
                f"response {name} {model}: N={len(graph)}, stable={leading:.5g}",
                flush=True,
            )
        save_csv("response_metrics.csv", rows)
        save_csv("response_nodes.csv", nodes)
        save_csv("equilibria.csv", equilibrium_rows)
        save_csv("finite_perturbation_checks.csv", finite_checks)
    save_json(
        "response_metadata.json",
        metadata()
        | {
            "synthetic_sizes": sizes,
            "models": [FIGURE_MODEL],
            "methods": METHODS,
            "target": "linear steady response energy outside the clamped source",
            "parameters_selected_before_results": True,
            "top_k_ties": "fractional selection at the boundary, rtol=1e-10, atol=1e-12",
            "preprocessing": "simple undirected binary, ecological weights >0.1, largest connected component by size, numeric raw-ID order",
            "finite_checks": "three fixed source IDs on each real graph, at 1% and 0.5% perturbations",
            "scope": "single deterministic setting per graph/model; no independent replicate confidence interval",
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["check", "benchmark", "response", "worker"])
    parser.add_argument("--sizes", nargs="+", type=int)
    parser.add_argument("--family")
    parser.add_argument("--n", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--method")
    args = parser.parse_args()
    if args.command == "check":
        check()
    elif args.command == "worker":
        worker(args)
    elif args.command == "benchmark":
        benchmark(args.sizes or [100, 500, 1000, 2000])
    elif args.command == "response":
        response_experiment(args.sizes or [100, 500, 1000])


if __name__ == "__main__":
    main()
