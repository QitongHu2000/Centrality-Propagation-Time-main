"""Matched-topology, working-point and perturbation-amplitude diagnostics."""

import numpy as np
from scipy.stats import spearmanr

from run_finite_time import (
    OUT as FINITE,
    SOCIAL,
    ECO,
    METHODS,
    load_graph,
    parameter_grid,
    scores_for,
    simulate,
    write_csv,
    write_json,
)
from run_experiments import model_equilibrium, top_mean

from paths import FIGURE_MODEL, RESULTS

OUT = RESULTS / "mechanism"


def assess(energy, scores, k=10):
    means = {q: top_mean(scores[q], energy, k) for q in METHODS}
    ptc = means["PTC"]
    best = max(v for q, v in means.items() if q != "PTC")
    tol = max(1e-10, 1e-8 * abs(ptc))
    return {
        "ptc_rank": 1 + sum(v > ptc + tol for q, v in means.items() if q != "PTC"),
        "ptc_energy": ptc,
        "best_energy": best,
        "gain_pct": 100 * (ptc / best - 1),
    }


def run():
    OUT.mkdir(exist_ok=True)
    rows = []
    coupling_rows = []
    node_rows = []
    time_rows = []
    for name in SOCIAL if FIGURE_MODEL == "E" else ECO:
        model = "E" if name in SOCIAL else "M"
        suffix = "100" if model == "E" else "20_10"
        g, a, labels, _ = load_graph(name)
        scores, _ = scores_for(g)
        cached = np.load(FINITE / f"{model}_{name}_{suffix}.npz")
        x, fun, jac, linear, _, theta, params, res, leading = model_equilibrium(
            a, model
        )
        # Local stationary transmission factors, obtained directly from J.
        jacobian = jac(x)
        q = jacobian / (-np.diag(jacobian))[:, None]
        np.fill_diagonal(q, 0)
        analytic = (
            (1 - x)[:, None] ** 2 * a
            if model == "E"
            else 0.5 * a / (1 + x[None, :]) ** 2
        )
        np.testing.assert_allclose(q, analytic, rtol=2e-10, atol=1e-12)
        first_hop = np.sum(q * q, axis=0)
        degree = a.sum(axis=0)
        coupling_rows.append(
            {
                "network": name,
                "model": model,
                "rho_degree_energy": float(spearmanr(degree, linear).statistic),
                "rho_ptc_energy": float(spearmanr(scores["PTC"], linear).statistic),
                "rho_one_hop_energy": float(spearmanr(first_hop, linear).statistic),
                "analytic_q_max_error": float(np.max(np.abs(q - analytic))),
                "positive_leading_eigenvalue": leading,
                "zero_leading_eigenvalue": (
                    float(np.linalg.eigvalsh(a)[-1] - 1) if model == "E" else ""
                ),
            }
        )
        modes = {"original_finite": cached["energy"][:, -1], "linear_steady": linear}
        t, e, _, _, _ = simulate(
            a, model, params["B"], params.get("C", 0.1), name, x=x, fraction=0.01
        )
        modes["positive_finite_1pct"] = e[:, -1]
        np.savez_compressed(
            OUT / f"{model}_{name}_positive_small.npz", times=t, energy=e
        )
        if model == "E":
            _, smallzero, _, _, _ = simulate(
                a, model, 1, 0.1, name, absolute_delta=0.001
            )
            modes["zero_finite_delta_0.001"] = smallzero[:, -1]
            np.savez_compressed(
                OUT / f"E_{name}_zero_small.npz", times=t, energy=smallzero
            )
        for mode, energy in modes.items():
            rows.append(
                {
                    "network": name,
                    "model": model,
                    "mode": mode,
                    **assess(energy, scores),
                    "rho_ptc": float(spearmanr(scores["PTC"], energy).statistic),
                }
            )
        for i, label in enumerate(labels):
            node_rows.append(
                {
                    "network": name,
                    "model": model,
                    **label,
                    "degree": degree[i],
                    "positive_equilibrium": x[i],
                    "PTC": scores["PTC"][i],
                    "BC": scores["BC"][i],
                    "one_hop_energy": first_hop[i],
                    **{mode: values[i] for mode, values in modes.items()},
                }
            )
        # The time horizon is fixed by the original experiment, not chosen for rank.
        for suffix, _, _ in parameter_grid(model):
            path = FINITE / f"{model}_{name}_{suffix}.npz"
            data = np.load(path)
            for index in [5, 25, 50, 99]:
                time_rows.append(
                    {
                        "network": name,
                        "model": model,
                        "parameters": path.stem.split(name + "_")[1],
                        "time_fraction": index / 99,
                        "time": data["times"][index],
                        **assess(data["energy"][:, index], scores),
                    }
                )
        print(name, rows[-len(modes) :], flush=True)
    write_csv(OUT / "working_points.csv", rows)
    write_csv(OUT / "coupling_factors.csv", coupling_rows)
    write_csv(OUT / "nodes.csv", node_rows)
    write_csv(OUT / "time_horizon.csv", time_rows)
    summary = {}
    for model in [FIGURE_MODEL]:
        summary[model] = {
            mode: {
                "cases": len(selected),
                "top_three": sum(r["ptc_rank"] <= 3 for r in selected),
                "ranks": [r["ptc_rank"] for r in selected],
            }
            for mode in sorted({r["mode"] for r in rows if r["model"] == model})
            if (
                selected := [
                    r for r in rows if r["model"] == model and r["mode"] == mode
                ]
            )
        }
    summary["time_horizon"] = [
        {
            "model": model,
            "time_fraction": fraction,
            "top_three": sum(r["ptc_rank"] <= 3 for r in selected),
            "cases": len(selected),
            "median_gain_pct": float(np.median([r["gain_pct"] for r in selected])),
        }
        for model in [FIGURE_MODEL]
        for fraction in [5 / 99, 25 / 99, 50 / 99, 1.0]
        if (
            selected := [
                r
                for r in time_rows
                if r["model"] == model and r["time_fraction"] == fraction
            ]
        )
    ]
    write_json(OUT / "summary.json", summary)


if __name__ == "__main__":
    run()
