"""Seeded reproduction of the submitted bridge construction, with saved graphs.

The old script did not save graph seeds/edge sets. These are new realizations
from its 100-node ensemble recipe, not recovered original instances.
"""

import csv
import gzip
import json

import networkx as nx
import numpy as np
from scipy.stats import rankdata, spearmanr

from centrality import ptc_scores

from paths import RESULTS

OUT = RESULTS / "bridge"
SEED = 20261005


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


def graph_case(seed):
    rng = np.random.default_rng(seed)
    count = int(rng.integers(2, 4))
    left = int(rng.integers(20, 80))
    right = 100 - left - count
    pl, pr = rng.uniform(0.2, 0.4, size=2)
    links_left, links_right = (int(v) for v in rng.integers(3, 5, size=2))

    def block(n, p):
        for attempt in range(10000):
            g = nx.fast_gnp_random_graph(n, p, seed=int(rng.integers(0, 2**32 - 1)))
            if nx.is_connected(g):
                return g
        raise RuntimeError("Connected block generation failed")

    g = nx.disjoint_union(block(left, pl), block(right, pr))
    bridges = list(range(left + right, 100))
    g.add_nodes_from(bridges)
    anchors = set()
    for bridge in bridges:
        for community, links in [
            (list(range(left)), links_left),
            (list(range(left, left + right)), links_right),
        ]:
            ordered = sorted(community, key=lambda v: (g.degree(v), v))
            chosen = rng.choice(ordered[: 4 * links], size=links, replace=False)
            anchors.update(int(v) for v in chosen)
            g.add_edges_from((bridge, int(v)) for v in chosen)
    config = {
        "seed": seed,
        "n_left": left,
        "n_right": right,
        "bridge_count": count,
        "p_left": float(pl),
        "p_right": float(pr),
        "links_left": links_left,
        "links_right": links_right,
        "bridge_nodes": bridges,
        "anchor_nodes": sorted(anchors),
    }
    return g, config


def distances(g, source, weighted=True):
    depth = nx.single_source_shortest_path_length(g, source)
    w = np.zeros(len(g))
    q = w.copy()
    w[source] = 1
    for u in depth:
        for v in g[u]:
            if depth[v] == depth[u] + 1:
                edge = g.degree(v) ** -2 if weighted else 1
                w[v] += edge * w[u]
                q[v] += edge * (q[u] + w[u] / g.degree(v))
    return q / w


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    graphs = []
    for seed in range(SEED, SEED + 300):
        g, c = graph_case(seed)
        methods = {
            "PTC": ptc_scores(g),
            "cost_only": ptc_scores(g, -1, 0, 0),
            "CC": nx.closeness_centrality(g),
            "BC": nx.betweenness_centrality(g),
        }
        row = {k: v for k, v in c.items() if k not in ["bridge_nodes", "anchor_nodes"]}
        row.update(
            {
                "imbalance": abs(c["n_left"] - c["n_right"])
                / (c["n_left"] + c["n_right"]),
                "mean_anchor_degree": float(
                    np.mean([g.degree(v) for v in c["anchor_nodes"]])
                ),
            }
        )
        for method, s in methods.items():
            ranks = rankdata(-np.array([s[i] for i in range(100)]), method="average")
            row[method] = float(np.mean(ranks[c["bridge_nodes"]]))
        rows.append(row)
        graphs.append(c | {"edges": list(g.edges())})
        if (seed - SEED + 1) % 50 == 0:
            print("Bridge graphs completed:", seed - SEED + 1, flush=True)
    write_csv(OUT / "ensemble.csv", rows)
    with gzip.open(OUT / "graphs.json.gz", "wt") as f:
        json.dump(graphs, f)
    ranks = np.array([r["PTC"] for r in rows])
    worst = int(np.argmax(ranks))
    median = int(np.argmin(np.abs(ranks - np.median(ranks))))
    example_rows = []
    examples = []
    for role, index in [("worst", worst), ("median", median)]:
        config = graphs[index]
        g = nx.Graph()
        g.add_nodes_from(range(100))
        g.add_edges_from(config["edges"])
        score = ptc_scores(g)
        leader = max(score, key=score.get)
        sources = list(dict.fromkeys(config["bridge_nodes"] + [leader]))
        for source in sources:
            d = distances(g, source)
            cost = distances(g, source, False)
            cc = nx.closeness_centrality(g)
            bc = nx.betweenness_centrality(g)
            example_rows.append(
                {
                    "case": role,
                    "seed": config["seed"],
                    "source": source,
                    "role": (
                        "bridge" if source in config["bridge_nodes"] else "PTC leader"
                    ),
                    "degree": g.degree(source),
                    "PTC": score[source],
                    "CC": cc[source],
                    "BC": bc[source],
                    "total_effective_distance": float(d.sum()),
                    "cost_only_total": float(cost.sum()),
                    "left_contribution": float(d[: config["n_left"]].sum()),
                    "right_contribution": float(
                        d[config["n_left"] : config["n_left"] + config["n_right"]].sum()
                    ),
                    "bridge_contribution": float(d[config["bridge_nodes"]].sum()),
                }
            )
        examples.append(
            config | {"case": role, "PTC_leader": leader, "summary": rows[index]}
        )
    write_csv(OUT / "case_decomposition.csv", example_rows)
    write_json(OUT / "examples.json", examples)
    summary = {
        "seed_start": SEED,
        "realizations": 300,
        "n": 100,
        "scope": "new seeded cohort; original unsaved graphs cannot be recovered",
        "rank_convention": "average ranks for tied scores",
        "statistics": {
            m: {
                "mean": float(np.mean([r[m] for r in rows])),
                "median": float(np.median([r[m] for r in rows])),
                "p90": float(np.quantile([r[m] for r in rows], 0.9)),
                "maximum": float(np.max([r[m] for r in rows])),
            }
            for m in ["PTC", "cost_only", "CC", "BC"]
        },
        "ptc_spearman_with": {
            key: float(spearmanr(ranks, [r[key] for r in rows]).statistic)
            for key in ["imbalance", "mean_anchor_degree", "CC", "cost_only"]
        },
        "worst_case": rows[worst],
        "median_case": rows[median],
        "ptc_worse_than_cost_only": int(
            sum(r["PTC"] > r["cost_only"] + 1e-10 for r in rows)
        ),
    }
    write_json(OUT / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    run()
