"""Finite-time curves and matching endpoint comparisons for Figures 4 and 6."""

from functools import lru_cache
from manuscript_assets import (
    COLORS,
    STYLES,
    INPUTS,
    OUT,
    METHODS,
    MUTUALISTIC_PALETTE,
    read,
    save,
    working_point_groups,
)
from paths import RESULTS
from run_finite_time import ECO, SOCIAL, LARGE_NAMES, displayed_parameter_grid
from run_experiments import top_mean
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
import numpy as np
import json


@lru_cache(maxsize=1)
def finite_data(model):
    scores = read("finite_time/scores.csv")
    large_scores = read("finite_time_large/scores.csv")
    cache = {}
    for names in [SOCIAL if model == "E" else ECO]:
        for name in names + LARGE_NAMES:
            directory = "finite_time_large" if name in LARGE_NAMES else "finite_time"
            nodes = sorted(
                (
                    r
                    for r in (large_scores if name in LARGE_NAMES else scores)
                    if r["network"] == name
                ),
                key=lambda r: int(r["node"]),
            )
            for suffix, b, c in displayed_parameter_grid(model):
                path = RESULTS / directory / f"{model}_{name}_{suffix}.npz"
                INPUTS.add(path)
                with np.load(path) as saved:
                    times, energy = saved["times"], saved["energy"]
                    means = {
                        q: np.array(
                            [
                                top_mean([float(r[q]) for r in nodes], energy[:, i], 10)
                                for i in range(len(times))
                            ]
                        )
                        for q in METHODS
                    }
                cache[(model, name, suffix)] = (times, means)
    return cache


def draw_panel(ax, key, title, inset=False, compact=False, ylabel=True):
    times, means = finite_data(key[0])[key]
    # The submitted M curves use pale yellow for LPP. Preserve the E palette.
    colors = dict(COLORS, LPP="#FEFF99") if key[0] == "M" else COLORS
    for q in METHODS:
        ax.plot(
            times,
            means[q],
            color=colors[q],
            ls=STYLES[q],
            lw=1.1 if q == "PTC" else 0.75,
            label=q,
            zorder=5 if q == "PTC" else 2,
        )
    ax.set_title(title, fontsize=10 if compact else 11, pad=7)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useOffset=False)
    ax.set_xlabel(r"$t$", fontsize=13, labelpad=1)
    if ylabel:
        ax.set_ylabel(
            r"$\overline{\mathcal{E}}^{\mathrm{out}}(t)$", fontsize=12, labelpad=1
        )
    ax.tick_params(labelsize=8.5 if compact else 9, direction="in", length=3)
    if compact:
        ax.set_xticks([0, times[-1] / 2, times[-1]])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.yaxis.get_offset_text().set_fontsize(7)
    for spine in ax.spines.values():
        spine.set_linewidth(1)
    if inset:
        small = ax.inset_axes([0.29, 0.17, 0.66, 0.43])
        last = times >= 0.5 * times[-1]
        for q in METHODS:
            small.plot(
                times[last],
                means[q][last],
                color=colors[q],
                ls=STYLES[q],
                lw=0.75 if q == "PTC" else 0.55,
            )
        small.tick_params(labelsize=5.5, pad=1, length=2)
        small.set_xticks([times[-1] / 2, 0.75 * times[-1], times[-1]])
        small.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useOffset=True)
        small.yaxis.set_major_locator(MaxNLocator(nbins=2))
        small.yaxis.get_offset_text().set_fontsize(5.5)
        ax.indicate_inset_zoom(small, edgecolor="0.5", alpha=0.5)


def scan_results(model):
    """Read every top-10 endpoint and check it against the plotted trajectories."""
    names = (SOCIAL if model == "E" else ECO) + LARGE_NAMES
    settings = displayed_parameter_grid(model)
    recorded = {
        (row["network"], row["parameters"]): row
        for row in read("finite_time/metrics.csv")
        + read("finite_time_large/metrics.csv")
        if row["model"] == model
        and row["selection"] == "top10"
        and row["method"] == "PTC"
        and row["parameters"] in {s[0] for s in settings}
    }
    gains = np.empty((len(names), len(settings)))
    ranks = np.empty_like(gains, dtype=int)
    cells = []
    for i, name in enumerate(names):
        for j, (suffix, b, c) in enumerate(settings):
            times, means = finite_data(model)[model, name, suffix]
            energy = float(means["PTC"][-1])
            reference = max(float(means[q][-1]) for q in METHODS if q != "PTC")
            gain = 100 * (energy / reference - 1)
            rank = 1 + sum(
                float(means[q][-1]) > energy + max(1e-10, 1e-8 * abs(energy))
                for q in METHODS
                if q != "PTC"
            )
            record = recorded[name, suffix]
            np.testing.assert_allclose(energy, float(record["energy"]), rtol=1e-12)
            assert rank == int(record["method_rank"])
            gains[i, j], ranks[i, j] = gain, rank
            cells.append(
                dict(
                    model=model,
                    network=name,
                    parameters=suffix,
                    B=b,
                    C=c,
                    observation_time=float(times[-1]),
                    selection="top10",
                    ptc_rank=rank,
                    gain_pct=gain,
                )
            )
    assert len(cells) == len(recorded)
    return names, settings, gains, ranks, cells


def draw_scan(fig, model, displayed, bounds=None):
    names, settings, gains, ranks, cells = scan_results(model)
    scale_min = min(0, float(gains.min()))
    # Match the upper curves exactly, including their column order.
    columns = {suffix: i for i, (suffix, _, _) in enumerate(settings)}
    selected = [columns[suffix] for suffix in displayed]
    settings = [settings[i] for i in selected]
    gains, ranks = gains[:, selected], ranks[:, selected]
    by_setting = {(cell["network"], cell["parameters"]): cell for cell in cells}
    cells = [by_setting[name, suffix] for name in names for suffix in displayed]
    left, right = bounds if bounds is not None else (0.11, 0.89)
    ax = fig.add_axes([left, 0.045, right - left, 0.16])
    palette = (
        LinearSegmentedColormap.from_list(
            "epidemic_gain", ["white", "#EECB8E", "#D89528", "#A45946"]
        )
        if model == "E"
        else LinearSegmentedColormap.from_list(
            "mutualistic_gain", ["white", *MUTUALISTIC_PALETTE]
        )
    )
    heatmap = ax.imshow(
        gains,
        aspect="auto",
        cmap=palette,
        vmin=scale_min,
        vmax=3 if model == "E" else 3500,
        interpolation="nearest",
    )
    ax.set_yticks(
        range(len(names)), labels=[n.replace("_", " ") for n in names], fontsize=10
    )
    labels = (
        [f"{b:g}" for _, b, _ in settings]
        if model == "E"
        else [f"{b:g}\n{c:g}" for _, b, c in settings]
    )
    ax.set_xticks(range(len(settings)), labels=labels, fontsize=10)
    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")
    ax.tick_params(length=0, pad=4)
    ax.set_xlabel(r"$B$" if model == "E" else r"$(B,C)$", fontsize=11, labelpad=3)
    ax.set_xticks(np.arange(len(settings) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(names) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="0.65", linewidth=0.45)
    ax.tick_params(which="minor", length=0)
    if model == "M":
        ax.axvline(2.5, color="0.3", linewidth=1.1)
    for i in range(len(names)):
        for j in range(len(settings)):
            gain = gains[i, j]
            rgb = heatmap.to_rgba(gain)[:3]
            linear = [
                c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb
            ]
            luminance = sum(c * w for c, w in zip(linear, [0.2126, 0.7152, 0.0722]))
            color = "black" if luminance > 0.179 else "white"
            value = f"{gain:+.5f}" if 0 < abs(gain) < 0.01 else f"{gain:+.2f}"
            if gain == 0:
                value = "0.00"
            ax.text(
                j,
                i - 0.15,
                str(ranks[i, j]),
                ha="center",
                va="center",
                fontsize=10.5,
                fontweight="bold",
                color=color,
            )
            ax.text(
                j, i + 0.21, value, ha="center", va="center", fontsize=8.5, color=color
            )
    label_x = left - 0.05 if model == "E" else 0.025
    fig.text(label_x, 0.23, "(g)", fontsize=15, fontweight="bold")
    fig.text(
        (left + right) / 2,
        0.23,
        "PTC rank / gain (%)",
        ha="center",
        fontsize=13,
        fontweight="bold",
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.45)
    color_left = right + 0.022 if model == "E" else 0.918
    color_ax = fig.add_axes([color_left, 0.045, 0.013, 0.16])
    colorbar = fig.colorbar(
        heatmap,
        cax=color_ax,
        extend="max" if float(gains.max()) > heatmap.norm.vmax else "neither",
    )
    colorbar.set_ticks([0, 1, 2, 3] if model == "E" else [0, 1000, 2000, 3000])
    colorbar.ax.tick_params(labelsize=8.5, length=2, pad=2)
    colorbar.ax.set_title(r"$G$ (%)", fontsize=9, pad=7)
    colorbar.outline.set_linewidth(0.45)
    return cells


def build_model(model):
    names = (SOCIAL if model == "E" else ECO) + LARGE_NAMES
    settings = (
        [("40", r"$B=0.4$"), ("70", r"$B=0.7$"), ("100", r"$B=1$")]
        if model == "E"
        else [
            ("15_10", r"$B=0.15$"),
            ("20_10", r"$B=0.2$ (ref.)"),
            ("25_10", r"$B=0.25$"),
            ("20_5", r"$C=0.05$"),
            ("20_20", r"$C=0.2$"),
        ]
    )
    left, right = 0.075, 0.98
    if model == "E":
        # Keep each panel as wide as in Figure 6; halve the actual blank gaps.
        panel_width = (right - left) / (3 + 2 * 1.60)
        grid_width = panel_width * (3 + 2 * 0.80)
        center = (left + right) / 2
        left, right = center - grid_width / 2, center + grid_width / 2
    center = (left + right) / 2 if model == "E" else 0.53
    label_x = left - 0.05 if model == "E" else 0.025
    fig, axes = plt.subplots(len(names), len(settings), figsize=(10.8, 13.4))
    fig.subplots_adjust(
        left=left,
        right=right,
        bottom=0.315,
        top=0.94,
        wspace=0.80 if model == "E" else 0.30,
        hspace=0.72,
    )
    manifest = []
    reference_records = []
    work = working_point_groups()
    reference_suffix = "100" if model == "E" else "20_10"
    for i, name in enumerate(names):
        for j, (suffix, title) in enumerate(settings):
            draw_panel(
                axes[i, j],
                (model, name, suffix),
                title,
                inset=model == "E",
                compact=True,
                ylabel=model == "E" or j == 0,
            )
            if model == "M" and suffix == "20_10":
                axes[i, j].set_facecolor("0.97")
            if i < len(names) - 1:
                axes[i, j].set_xlabel("")
            if suffix == reference_suffix:
                _, means = finite_data(model)[model, name, suffix]
                case = (
                    work[model, name]["original_finite"]
                    if name not in LARGE_NAMES
                    else {
                        "ptc_energy": float(means["PTC"][-1]),
                        "best_energy": max(
                            float(means[q][-1]) for q in METHODS if q != "PTC"
                        ),
                    }
                )
                np.testing.assert_allclose(
                    [case["ptc_energy"], case["best_energy"]],
                    [
                        means["PTC"][-1],
                        max(means[q][-1] for q in METHODS if q != "PTC"),
                    ],
                    rtol=1e-10,
                    atol=1e-12,
                )
                fmt = ".7g" if model == "E" and name in LARGE_NAMES else ".5f"
                labels = [
                    f"PTC/b: {case['ptc_energy']:{fmt}} / {case['best_energy']:{fmt}}"
                ]
                axes[i, j].text(
                    0.96,
                    0.86 if model == "E" or name != "WS_1000" else 0.48,
                    "\n".join(labels),
                    transform=axes[i, j].transAxes,
                    ha="right",
                    va="top",
                    fontsize=6.5,
                    linespacing=1.4,
                    bbox={
                        "facecolor": axes[i, j].get_facecolor(),
                        "edgecolor": "none",
                        "pad": 0.6,
                    },
                )
                reference_records.append(
                    dict(
                        model=model,
                        network=name,
                        parameters=suffix,
                        ptc_energy=case["ptc_energy"],
                        best_reference_energy=case["best_energy"],
                        labels=labels,
                    )
                )
            manifest.append(dict(model=model, network=name, parameters=suffix))
        top = axes[i, 1].get_position().y1
        title = name if name not in LARGE_NAMES else rf"{name.split('_')[0]} ($N=1000$)"
        fig.text(
            center, top + 0.021, title, ha="center", fontsize=15, fontweight="bold"
        )
        fig.text(label_x, top + 0.021, f"({chr(97+i)})", fontsize=15, fontweight="bold")
    if model == "M":
        # Sweep definitions and the shared reference are given in the caption.
        divider = axes[0, 2].get_position().x1
        c_left = axes[0, 3].get_position().x0
        line_x = (divider + c_left) / 2
        fig.add_artist(
            plt.Line2D(
                [line_x, line_x],
                [0.30, axes[0, 0].get_position().y1 + 0.012],
                transform=fig.transFigure,
                color="0.7",
                lw=0.7,
            )
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(center, 0.269),
        ncol=11,
        fontsize=9.5,
        frameon=True,
        fancybox=False,
        framealpha=1,
        edgecolor="black",
        facecolor="white",
        handlelength=1.8,
        columnspacing=1.0,
    )
    legend.get_frame().set_linewidth(0.7)
    scan = draw_scan(
        fig,
        model,
        [suffix for suffix, _ in settings],
        bounds=(left, right) if model == "E" else None,
    )
    save(fig, f"finite_{model}")
    assert len(manifest) == (18 if model == "E" else 30)
    assert len(scan) == len(manifest)
    assert [(p["network"], p["parameters"]) for p in scan] == [
        (p["network"], p["parameters"]) for p in manifest
    ]
    assert len(finite_data(model)) == len(manifest)
    (OUT / "verification" / f"finite_{model}_panels.json").write_text(
        json.dumps(
            {
                "rule": (
                    "Four empirical and two 1,000-node networks by three B values."
                    if model == "E"
                    else "Four empirical and two 1,000-node networks by five settings: three B values at C=0.1 and "
                    "three C values at B=0.2 share the reference (B,C)=(0.2,0.1)."
                ),
                "panels": manifest,
                "reference_energies": reference_records,
                "scan_panels": scan,
                "scan_encoding": "PTC method rank above gain G (%); columns match the upper curves in the same order.",
                "curve_layout": {
                    "rows": len(names),
                    "columns": len(settings),
                    "panel_width_inches": float(axes[0, 0].get_position().width * 10.8),
                    "row_1_bounds": [list(ax.get_position().bounds) for ax in axes[0]],
                    "column_gap_inches": float(
                        (axes[0, 1].get_position().x0 - axes[0, 0].get_position().x1)
                        * 10.8
                    ),
                    "legend": "One shared framed row below all six curve rows.",
                    "legend_frame": True,
                },
                "scan_palette": (
                    "Figure 3 ochre/brown" if model == "E" else "Figure 5 teal/green"
                ),
                "model": model,
                "displayed_settings": len(manifest),
                "archived_empirical_scan_settings": 20 if model == "E" else 36,
                "new_large_network_settings": 6 if model == "E" else 10,
            },
            indent=2,
        )
        + "\n"
    )
