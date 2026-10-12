"""Check local source, input ownership, scores and saved endpoint comparisons."""

import ast
import csv
import hashlib
import json

from paths import ASSETS, CODE, FIGURE, FIGURE_MODEL, REPORTS, RESULTS, figure_path


def read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    checks = []
    for path in CODE.glob("*.py"):
        ast.parse(path.read_text(), filename=str(path))
    checks.append("All local Python files parse")
    build = json.loads((CODE / "build.json").read_text())
    for section in ["sources", "inputs", "outputs"]:
        for name, digest in build[section].items():
            path = (CODE / name).resolve()
            path.relative_to(CODE)
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
    checks.append(
        "Build sources, inputs and outputs are local and match recorded hashes"
    )
    if FIGURE_MODEL is None:
        assert figure_path().read_bytes() == (ASSETS / "original.pdf").read_bytes()
        checks.append("Submitted vector artwork preserved byte for byte")
    else:
        import numpy as np
        from run_finite_time import METHODS, ECO, SOCIAL, load_graph, scores_for

        rows = read(RESULTS / "finite_time/scores.csv")
        names = SOCIAL if FIGURE_MODEL == "E" else ECO
        assert {row["network"] for row in rows} == set(names)
        for name in names:
            graph, _, labels, _ = load_graph(name)
            nodes = sorted(
                (row for row in rows if row["network"] == name),
                key=lambda row: int(row["node"]),
            )
            assert [row["original_id"] for row in nodes] == [
                row["original_id"] for row in labels
            ]
            calculated, _ = scores_for(graph)
            for method in METHODS:
                np.testing.assert_allclose(
                    [float(row[method]) for row in nodes],
                    calculated[method],
                    rtol=1e-10,
                    atol=1e-12,
                )
        checks.append(
            "All eleven scores recomputed from local raw networks and node IDs"
        )
        if FIGURE in ["figure_4", "figure_6"]:
            from run_finite_time import metric_rows, displayed_parameter_grid
            from plot_finite_time import scan_results

            count = 0
            for directory in ["finite_time", "finite_time_large"]:
                base = RESULTS / directory
                nodes = read(base / "scores.csv")
                recorded = {
                    (r["network"], r["parameters"], r["selection"], r["method"]): r
                    for r in read(base / "metrics.csv")
                }
                assert all(r["model"] == FIGURE_MODEL for r in recorded.values())
                for path in sorted(base.glob(f"{FIGURE_MODEL}_*.npz")):
                    # Names include BA_1000 / WS_1000; use the score table as the key.
                    name = next(
                        name
                        for name in {r["network"] for r in nodes}
                        if path.stem.startswith(f"{FIGURE_MODEL}_{name}_")
                    )
                    suffix = path.stem.removeprefix(f"{FIGURE_MODEL}_{name}_")
                    selected = sorted(
                        (r for r in nodes if r["network"] == name),
                        key=lambda r: int(r["node"]),
                    )
                    scores = {
                        q: np.array([float(r[q]) for r in selected]) for q in METHODS
                    }
                    with np.load(path) as data:
                        assert data["energy"].shape == (
                            len(selected),
                            len(data["times"]),
                        )
                        assert np.isfinite(data["energy"]).all()
                        computed = metric_rows(
                            name,
                            FIGURE_MODEL,
                            suffix,
                            data["times"],
                            data["energy"],
                            scores,
                            dict.fromkeys(METHODS, 0.0),
                        )
                    for row in computed:
                        saved = recorded[name, suffix, row["selection"], row["method"]]
                        assert row["method_rank"] == int(saved["method_rank"])
                        np.testing.assert_allclose(
                            row["energy"], float(saved["energy"]), rtol=1e-12
                        )
                    count += 1
            _, _, _, _, visible = scan_results(FIGURE_MODEL)
            assert len(visible) == 6 * len(displayed_parameter_grid(FIGURE_MODEL))
            checks.append(
                f"{count} saved settings: eleven methods, top-10/top-5% energies and ranks"
            )
            checks.append(
                f"{len(visible)} displayed settings agree with curve and rank panels"
            )
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "validation.json").write_text(
        json.dumps({"status": "passed", "figure": FIGURE, "checks": checks}, indent=2)
        + "\n"
    )
    print(f"{FIGURE}: {len(checks)} checks passed")


if __name__ == "__main__":
    main()
