"""Figure-local finite-time simulation with the original equations and protocol.

Only the displayed parameter columns of FIGURE_MODEL are generated. Saved
historical settings are retained when refreshing the visible empirical cases.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import networkx as nx
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix, diags, issparse
from scipy.stats import spearmanr

from centrality import hsi_scores
from run_experiments import SEED, centrality, graph_case, metadata, top_mean

from paths import FIGURE_MODEL, NETWORKS, PROJECT_ROOT as ROOT, RESULTS

OUT = RESULTS / "finite_time"
LARGE_OUT = RESULTS / "finite_time_large"
LARGE_NAMES = ["BA_1000", "WS_1000"]
SOCIAL = ["dolphins", "football", "karate", "polbooks"]
ECO = ["everglades", "florida", "mangwet", "stmarks"]
METHODS = ["DC", "EC", "BC", "CC", "PR", "GC", "SP", "LPP", "SGM", "HSI", "PTC"]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def merge_csv(path, rows, keys):
    """Replace regenerated records, retaining historical settings verbatim."""
    updated = {tuple(str(row[key]) for key in keys) for row in rows}
    old = []
    if path.exists():
        with path.open(newline="") as stream:
            old = [
                row
                for row in csv.DictReader(stream)
                if tuple(row[key] for key in keys) not in updated
            ]
    write_csv(path, old + rows)


def generate_scores():
    """Compute the eleven-method score matrices without running dynamics."""
    nodes, graphs = [], []
    for name in SOCIAL if FIGURE_MODEL == "E" else ECO:
        graph, _, labels, provenance = load_graph(name)
        scores, _ = scores_for(graph)
        write_json(
            OUT / f"{name}_graph.json",
            provenance | {"labels": labels, "edges": list(graph.edges())},
        )
        graphs.append({"network": name, **provenance})
        for label in labels:
            node = label["node"]
            nodes.append(
                {
                    "network": name,
                    **label,
                    "degree": graph.degree(node),
                    **{method: float(score[node]) for method, score in scores.items()},
                }
            )
    write_csv(OUT / "scores.csv", nodes)
    write_json(
        OUT / "score_metadata.json",
        metadata() | {"model": FIGURE_MODEL, "graphs": graphs, "methods": METHODS},
    )
    print(f"{FIGURE_MODEL}: regenerated scores for {len(graphs)} networks")


def load_graph(name):
    if name in SOCIAL:
        path = NETWORKS / (name + ".gml")
        graph = nx.Graph(nx.read_gml(path, label=None))
    else:
        path = NETWORKS / ("eco-" + name + ".edges")
        raw = nx.read_weighted_edgelist(path)
        graph = nx.Graph()
        graph.add_nodes_from(raw)
        graph.add_edges_from(
            (u, v) for u, v, d in raw.edges(data=True) if u != v and d["weight"] > 0.1
        )
    graph.remove_edges_from(nx.selfloop_edges(graph))
    component = max(nx.connected_components(graph), key=len)
    graph = graph.subgraph(component).copy()
    # Numeric input IDs are ordered explicitly; no hash/set iteration defines a row.
    original = sorted(
        graph, key=lambda u: (int(u) if str(u).lstrip("-").isdigit() else str(u))
    )
    graph = nx.relabel_nodes(graph, {u: i for i, u in enumerate(original)}, copy=True)
    a = nx.to_numpy_array(graph, nodelist=range(len(graph)), weight=None)
    g = nx.from_numpy_array(a)
    labels = [{"node": i, "original_id": str(u)} for i, u in enumerate(original)]
    return (
        g,
        a,
        labels,
        {
            "input": str(path.relative_to(ROOT)),
            "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "adjacency_sha256": hashlib.sha256(a.astype("<f8").tobytes()).hexdigest(),
            "weighted": False,
            "n": len(g),
            "m": g.number_of_edges(),
        },
    )


def scores_for(g):
    result, times = {}, {}
    for method in METHODS:
        start = time.perf_counter()
        mapping = hsi_scores(g) if method == "HSI" else centrality(g, method)
        result[method] = np.array([mapping[i] for i in range(len(g))])
        if not np.all(np.isfinite(result[method])):
            raise FloatingPointError(f"Nonfinite score in corrected {method}")
        times[method] = time.perf_counter() - start
    for multiplier in [0.5, 1.5]:
        result[f"HSI_{multiplier:g}"] = np.array(
            list(hsi_scores(g, multiplier).values())
        )
    return result, times


def field(a, model, b, c=0.1):
    if model == "E":
        return lambda t, x: -b * x + (1 - x) * (a @ x)
    return lambda t, x: b * x * (1 - x / c) + x * (a @ (x / (1 + x)))


def sparse_jacobian(a, x, model, b, c):
    """Exact Jacobian of the unchanged nonlinear equations on a sparse graph."""
    if model == "E":
        return diags(-b - a @ x) + diags(1 - x) @ a
    return diags(b * (1 - 2 * x / c) + a @ (x / (1 + x))) + diags(x) @ a @ diags(
        (1 + x) ** -2
    )


def baseline(a, model, b, c, name, rtol=1e-10, atol=1e-12, method="LSODA"):
    f = field(a, model, b, c)
    if model == "E":
        x = np.zeros(a.shape[0])
    else:
        initial = np.full(a.shape[0], 0.1 if name == "everglades" else 1.0)
        options = {}
        if issparse(a) and method in {"BDF", "Radau"}:
            options["jac"] = lambda t, y: sparse_jacobian(a, y, model, b, c)
        sol = solve_ivp(
            f, (0, 1000), initial, method=method, rtol=rtol, atol=atol, **options
        )
        if not sol.success:
            raise RuntimeError(sol.message)
        x = sol.y[:, -1]
    residual = np.max(np.abs(f(0, x)))
    if residual > 1e-8:
        raise RuntimeError(f"Baseline residual {residual}")
    return x, float(residual)


def simulate(
    a,
    model,
    b,
    c,
    name,
    *,
    x=None,
    fraction=None,
    rtol=1e-10,
    atol=1e-12,
    method="LSODA",
    sources=None,
    absolute_delta=None,
):
    n = a.shape[0]
    f = field(a, model, b, c)
    if x is None:
        x, residual = baseline(a, model, b, c, name, rtol, atol)
    else:
        residual = float(np.max(np.abs(f(0, x))))
    tmax = 20.0 if model == "E" else (1.0 if name == "everglades" else 2.0)
    ts = np.linspace(0, tmax, 100)
    sources = list(range(n)) if sources is None else list(sources)
    energy = np.zeros((len(sources), len(ts)))
    final_states = np.zeros((len(sources), n))
    for row, source in enumerate(sources):
        delta = (
            (0.1 if model == "E" else 0.3 * x[source])
            if fraction is None
            else (
                fraction * min(x[source], 1 - x[source])
                if model == "E"
                else fraction * x[source]
            )
        )
        if absolute_delta is not None:
            delta = absolute_delta
        initial = x.copy()
        initial[source] += delta

        def clamped(t, y):
            value = f(t, y)
            value[source] = 0.0
            return value

        options = {}
        if issparse(a) and method in {"BDF", "Radau"}:
            mask = np.ones(n)
            mask[source] = 0
            options["jac"] = lambda t, y: diags(mask) @ sparse_jacobian(
                a, y, model, b, c
            )
        sol = solve_ivp(
            clamped,
            (0, tmax),
            initial,
            t_eval=ts,
            method=method,
            rtol=rtol,
            atol=atol,
            **options,
        )
        if not sol.success:
            raise RuntimeError(sol.message)
        np.testing.assert_allclose(sol.y[source], initial[source], atol=1e-12)
        displacement = sol.y - x[:, None]
        displacement[source] = 0
        energy[row] = np.sum(displacement**2, axis=0) / delta**2
        # Independent archived definition: subtract the first perturbed sample.
        archived_formula = np.sum((sol.y - sol.y[:, :1]) ** 2, axis=0) / delta**2
        np.testing.assert_allclose(
            energy[row], archived_formula, rtol=1e-12, atol=1e-12
        )
        final_states[row] = sol.y[:, -1]
    return ts, energy, x, final_states, residual


def simulate_batched(a, model, b, c, name, x, batch_size=20):
    """Solve independent source clamps in columns; trajectories never interact.

    Sparse matrix products accelerate the original finite-time experiment.
    The equations, displacements, horizon and output times match simulate().
    """
    n = a.shape[0]
    a = csr_matrix(a)
    f = field(a, model, b, c)
    tmax = 20.0 if model == "E" else (1.0 if name == "everglades" else 2.0)
    ts = np.linspace(0, tmax, 100)
    energy = np.empty((n, len(ts)))
    for start in range(0, n, batch_size):
        sources = np.arange(start, min(n, start + batch_size))
        columns = np.arange(len(sources))
        delta = np.full(len(sources), 0.1) if model == "E" else 0.3 * x[sources]
        initial = np.repeat(x[:, None], len(sources), axis=1)
        initial[sources, columns] += delta

        def clamped(t, flat):
            value = f(t, flat.reshape(n, len(sources)))
            value[sources, columns] = 0
            return value.ravel()

        sol = solve_ivp(
            clamped,
            (0, tmax),
            initial.ravel(),
            t_eval=ts,
            method="DOP853",
            rtol=2e-12,
            atol=2e-14,
        )
        if not sol.success:
            raise RuntimeError(sol.message)
        trajectories = sol.y.reshape(n, len(sources), len(ts))
        np.testing.assert_allclose(
            trajectories[sources, columns],
            np.repeat(initial[sources, columns, None], len(ts), axis=1),
            rtol=0,
            atol=1e-12,
        )
        displacement = trajectories - x[:, None, None]
        displacement[sources, columns] = 0
        energy[sources] = np.sum(displacement**2, axis=0) / delta[:, None] ** 2
    if not np.isfinite(energy).all():
        raise FloatingPointError("Nonfinite batched source response")
    return ts, energy


def metric_rows(name, model, parameters, ts, energy, scores, times, protocol="binary"):
    rows = []
    for selection, k in [
        ("top10", min(10, len(energy))),
        ("top5pct", math.ceil(0.05 * len(energy))),
    ]:
        means = {
            method: top_mean(scores[method], energy[:, -1], k) for method in METHODS
        }
        for method in METHODS:
            value = means[method]
            # The same numerical tie tolerance is used for all methods.
            tolerance = max(1e-10, abs(value) * 1e-8)
            rank = 1 + sum(
                v > value + tolerance for q, v in means.items() if q != method
            )
            rows.append(
                {
                    "protocol": protocol,
                    "model": model,
                    "network": name,
                    "parameters": parameters,
                    "n": len(energy),
                    "endpoint_time": ts[-1],
                    "selection": selection,
                    "k": k,
                    "method": method,
                    "energy": value,
                    "method_rank": rank,
                    "spearman": float(
                        spearmanr(scores[method], energy[:, -1]).statistic
                    ),
                    "ranking_seconds": times[method],
                }
            )
    return rows


def parameter_grid(model):
    if model == "E":
        return [
            (suffix, b, 0.1)
            for suffix, b in zip(
                ["40", "55", "70", "85", "100"], np.linspace(0.4, 1, 5)
            )
        ]
    # Preserve numeric values rather than the old int(100*x) filename truncation.
    return [(f"{int(100*b)}_10", float(b), 0.1) for b in np.linspace(0.15, 0.25, 5)] + [
        (f"20_{int(100*c)}", 0.2, float(c))
        for c in np.linspace(0.05, 0.25, 5)
        if not np.isclose(c, 0.1)
    ]


def displayed_parameter_grid(model):
    """Only the parameter columns shown in Figures 4 and 6."""
    selected = (
        ["40", "70", "100"]
        if model == "E"
        else ["15_10", "20_10", "25_10", "20_5", "20_20"]
    )
    values = {suffix: (suffix, b, c) for suffix, b, c in parameter_grid(model)}
    return [values[suffix] for suffix in selected]


def run_large():
    """Extend only the displayed columns to the same BA/WS 1,000-node graphs."""
    LARGE_OUT.mkdir(parents=True, exist_ok=True)
    all_rows, checks, nodes, graph_reports, sensitivities = [], [], [], [], []
    for family in ["BA", "WS"]:
        name = f"{family}_1000"
        graph = graph_case(family, 1000, SEED)
        a = nx.to_scipy_sparse_array(graph, nodelist=range(len(graph)), dtype=float)
        scores, times = scores_for(graph)
        provenance = {
            "network": name,
            "family": family,
            "seed": SEED,
            "n": len(graph),
            "m": graph.number_of_edges(),
            "adjacency_sha256": hashlib.sha256(
                a.toarray().astype("<f8").tobytes()
            ).hexdigest(),
            "weighted": False,
        }
        graph_reports.append(provenance)
        write_json(
            LARGE_OUT / f"{name}_graph.json",
            provenance | {"edges": list(graph.edges())},
        )
        for node in range(len(graph)):
            nodes.append(
                {
                    "network": name,
                    "node": node,
                    "original_id": str(node),
                    "degree": graph.degree(node),
                    **{method: float(score[node]) for method, score in scores.items()},
                }
            )
        for model in [FIGURE_MODEL]:
            for suffix, b, c in displayed_parameter_grid(model):
                start = time.perf_counter()
                path = LARGE_OUT / f"{model}_{name}_{suffix}.npz"
                if path.exists():
                    with np.load(path) as saved:
                        ts, energies, equilibrium = (
                            saved["times"],
                            saved["energy"],
                            saved["equilibrium"],
                        )
                    residual = float(
                        np.max(np.abs(field(a, model, b, c)(0, equilibrium)))
                    )
                else:
                    equilibrium, residual = baseline(a, model, b, c, name, method="BDF")
                    ts, energies = simulate_batched(a, model, b, c, name, equilibrium)
                    np.savez_compressed(
                        path,
                        times=ts,
                        energy=energies,
                        equilibrium=equilibrium,
                        adjacency_sha256=provenance["adjacency_sha256"],
                    )
                source_ids = [0, len(graph) // 2, len(graph) - 1]
                _, tight, _, _, _ = simulate(
                    a,
                    model,
                    b,
                    c,
                    name,
                    x=equilibrium,
                    sources=source_ids,
                    rtol=2e-13,
                    atol=2e-15,
                    method="Radau",
                )
                solver_error = float(
                    np.max(np.abs(tight - energies[source_ids]))
                    / max(np.max(tight), 1e-15)
                )
                if solver_error > 2e-6 or residual > 1e-8:
                    raise RuntimeError(
                        f"Large-network solver check failed: {name}/{model}/{suffix}"
                    )
                rows = metric_rows(
                    name,
                    model,
                    suffix,
                    ts,
                    energies,
                    scores,
                    times,
                    protocol="binary_large",
                )
                all_rows.extend(rows)
                checks.append(
                    {
                        "model": model,
                        "network": name,
                        "parameters": suffix,
                        "baseline_residual": residual,
                        "independent_solver_max_scaled_error": solver_error,
                        "simulation_seconds": time.perf_counter() - start,
                        "sources_checked": ";".join(map(str, source_ids)),
                    }
                )
                for multiplier in [0.5, 1.5]:
                    for k in [10, math.ceil(0.05 * len(graph))]:
                        sensitivities.append(
                            {
                                "model": model,
                                "network": name,
                                "parameters": suffix,
                                "k": k,
                                "beta_multiplier": multiplier,
                                "hsi_energy": top_mean(
                                    scores[f"HSI_{multiplier:g}"], energies[:, -1], k
                                ),
                            }
                        )
                write_csv(LARGE_OUT / "metrics.csv", all_rows)
                write_csv(LARGE_OUT / "solver_checks.csv", checks)
                write_csv(LARGE_OUT / "scores.csv", nodes)
                write_csv(LARGE_OUT / "hsi_sensitivity.csv", sensitivities)
                rank = next(
                    r["method_rank"]
                    for r in rows
                    if r["selection"] == "top10" and r["method"] == "PTC"
                )
                print(json.dumps({**checks[-1], "ptc_rank": rank}), flush=True)
    write_json(
        LARGE_OUT / "metadata.json",
        metadata()
        | {
            "graphs": graph_reports,
            "methods": METHODS,
            "model": FIGURE_MODEL,
            "protocol": "Only this figure's displayed parameter columns; original sustained source clamp, nonsource energy and fractional boundary ties",
            "E": {"initial": "zero", "source_delta": 0.1, "endpoint_time": 20.0},
            "M": {
                "initial": "positive state reached from all-ones at t=1000",
                "source_fraction": 0.3,
                "endpoint_time": 2.0,
            },
            "ode": "Independent columns integrated by sparse DOP853, rtol=2e-12, atol=2e-14; three fixed sources per setting verified independently by Radau at 2e-13/2e-15",
            "baseline_solver": "BDF, rtol=1e-10, atol=1e-12, analytic sparse Jacobian",
            "finite_time_source_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "settings": len(checks),
            "source_trajectories": sum(r["n"] for r in graph_reports)
            * len(displayed_parameter_grid(FIGURE_MODEL)),
        },
    )
    print("LARGE-NETWORK FINITE-TIME EXTENSION COMPLETE", flush=True)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows, checks, graph_reports, nodes, sensitivities = [], [], [], [], []
    for name in SOCIAL if FIGURE_MODEL == "E" else ECO:
        model = "E" if name in SOCIAL else "M"
        graph, a, labels, provenance = load_graph(name)
        scores, times = scores_for(graph)
        write_json(
            OUT / f"{name}_graph.json",
            provenance | {"labels": labels, "edges": list(graph.edges())},
        )
        graph_reports.append({"network": name, **provenance})
        for label in labels:
            node = label["node"]
            nodes.append(
                {
                    "network": name,
                    **label,
                    "degree": graph.degree(node),
                    **{method: float(score[node]) for method, score in scores.items()},
                }
            )
        for suffix, b, c in displayed_parameter_grid(model):
            start = time.perf_counter()
            ts, energies, equilibrium, finals, residual = simulate(a, model, b, c, name)
            seconds = time.perf_counter() - start
            np.savez_compressed(
                OUT / f"{model}_{name}_{suffix}.npz",
                times=ts,
                energy=energies,
                equilibrium=equilibrium,
                final_states=finals,
                adjacency=a,
            )
            all_rows += metric_rows(name, model, suffix, ts, energies, scores, times)
            for multiplier in [0.5, 1.5]:
                for k in [10, math.ceil(0.05 * len(graph))]:
                    sensitivities.append(
                        {
                            "model": model,
                            "network": name,
                            "parameters": suffix,
                            "k": k,
                            "beta_multiplier": multiplier,
                            "hsi_energy": top_mean(
                                scores[f"HSI_{multiplier:g}"], energies[:, -1], k
                            ),
                        }
                    )
            # Tighter tolerance and an independent explicit integrator for fixed IDs.
            source_ids = [0, len(graph) // 2, len(graph) - 1]
            _, tight, _, _, _ = simulate(
                a,
                model,
                b,
                c,
                name,
                x=equilibrium,
                sources=source_ids,
                rtol=2e-12,
                atol=2e-14,
                method="DOP853",
            )
            solver_error = float(
                np.max(np.abs(tight - energies[source_ids])) / max(np.max(tight), 1e-15)
            )
            if solver_error > 2e-6:
                raise RuntimeError(
                    f"Solver check failed {name} {suffix}: {solver_error}"
                )
            checks.append(
                {
                    "model": model,
                    "network": name,
                    "parameters": suffix,
                    "baseline_residual": residual,
                    "independent_solver_max_scaled_error": solver_error,
                    "simulation_seconds": seconds,
                }
            )
            merge_csv(
                OUT / "metrics.csv",
                all_rows,
                ["model", "network", "parameters", "selection", "method"],
            )
            rank = next(
                row["method_rank"]
                for row in all_rows[-2 * len(METHODS) :]
                if row["selection"] == "top10" and row["method"] == "PTC"
            )
            print(
                json.dumps(
                    {
                        "network": name,
                        "parameters": suffix,
                        "ptc_rank": rank,
                        "solver_error": solver_error,
                        "seconds": seconds,
                    }
                ),
                flush=True,
            )
    write_json(
        OUT / "metadata.json",
        metadata()
        | {
            "model": FIGURE_MODEL,
            "generated_parameters": [
                s for s, _, _ in displayed_parameter_grid(FIGURE_MODEL)
            ],
            "retained_history": "Existing non-displayed empirical records are preserved, not regenerated.",
            "graphs": graph_reports,
            "methods": METHODS,
            "hsi_beta": "<k>/(<k^2>-<k>), fixed before looking at labels; sensitivity at 0.5 and 1.5 times this",
            "primary_graph_convention": "undirected binary, loops removed, largest component by node count, sorted raw IDs",
            "ode": "LSODA, rtol=1e-10, atol=1e-12; DOP853 verification at rtol=2e-12, atol=2e-14",
            "selection": "fractional boundary ties, rtol=1e-10 and atol=1e-12",
            "method_rank_ties": "rtol=1e-8 and atol=1e-10 on mean energy",
            "finite_time_source_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        },
    )
    merge_csv(OUT / "solver_checks.csv", checks, ["model", "network", "parameters"])
    write_csv(OUT / "scores.csv", nodes)
    merge_csv(
        OUT / "hsi_sensitivity.csv",
        sensitivities,
        ["model", "network", "parameters", "k", "beta_multiplier"],
    )
    print("FINITE-TIME RE-RUN COMPLETE", flush=True)


def check():
    from run_experiments import real_graphs

    for name, graph, _ in real_graphs():
        _, adjacency, labels, _ = load_graph(name)
        assert list(graph) == list(range(len(graph)))
        np.testing.assert_array_equal(nx.to_numpy_array(graph, weight=None), adjacency)
        assert [str(graph.nodes[i]["original_id"]) for i in graph] == [
            r["original_id"] for r in labels
        ]
    # On a tree there is a unique infection route and no within-layer edges.
    graph = nx.path_graph(5)
    degree = np.array(list(dict(graph.degree()).values()))
    beta = 0.5 * degree.sum() / np.sum(degree**2 - degree)
    expected = {
        u: sum(
            beta**d for d in nx.single_source_shortest_path_length(graph, u).values()
        )
        for u in graph
    }
    np.testing.assert_allclose(
        list(hsi_scores(graph, 0.5).values()), list(expected.values())
    )
    # Triangle: both first-layer nodes have one same-layer neighbor.
    graph = nx.complete_graph(3)
    score = hsi_scores(graph, 0.5)
    beta = 0.5
    np.testing.assert_allclose(
        list(score.values()), 1 + 2 * (beta + (1 - beta) * beta**2)
    )
    graph = nx.relabel_nodes(nx.path_graph(5), {i: f"node{i}" for i in range(5)})
    assert set(hsi_scores(graph, 0.5)) == set(graph)
    # Correct LPP and all comparison scores must be invariant to relabeling.
    graph = nx.karate_club_graph()
    perm = np.random.default_rng(20260527).permutation(len(graph))
    relabeled = nx.relabel_nodes(graph, dict(enumerate(perm)))
    for method in ["LPP", "BC", "PTC"]:
        a = centrality(graph, method)
        b = centrality(relabeled, method)
        np.testing.assert_allclose(
            [a[i] for i in graph], [b[perm[i]] for i in graph], rtol=1e-12, atol=1e-12
        )
    write_json(
        OUT / "implementation_checks.json",
        {
            "status": "passed",
            "checks": [
                "HSI unique-route tree formula",
                "HSI triangle same-layer formula",
                "arbitrary HSI labels",
                "LPP, BC, PTC relabeling invariance",
                "finite and steady protocols have identical adjacency and raw-ID mapping",
            ],
        },
    )
    print("HSI analytical and node-relabeling checks passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "check", "large", "scores"])
    args = parser.parse_args()
    {"check": check, "run": run, "large": run_large, "scores": generate_scores}[
        args.command
    ]()
