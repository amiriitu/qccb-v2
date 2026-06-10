"""
Publication-grade replot of quantum-hardware benchmark charts from a saved
`quantum_hardware_runs.json` artifact. No chip submission, no benchmark
re-run — just better plots for the dissertation.

Improvements over the original `src.report_v2` charts (which were
"good enough" auto-generated diagnostics):

  1. fidelity_comparison.png  →  3×3 grid (one panel per experiment) with
     grouped bars by backend + 95% CI error bars + value labels.
     Old: 9×6=54 bars in one row, legend collision, hard to read.
  2. fidelity_timeline.png    →  3×3 grid (one panel per experiment) with
     one line per backend across N=10 runs. Mean ± stdev band shown.
     Old: 6 thin horizontal panels with 9 overlapping lines each.
  3. timing_breakdown.png     →  log-y scale stacked bars. Real (170s) and
     ideal (0.1s) both visible.
     Old: linear y-axis, real bars saturated, others invisible.
  4. python_overhead.png      →  same shape, tighter labels, fewer ticks.
  5. counts_overlay.png       →  same idea, Wilson-95% CIs on empirical bars.
  6. chip_calibration.png     →  same, but worst-T2 qubit highlighted in red.

Usage:
    python tools/replot_charts.py  <path_to_results_dir>
or simply
    python tools/replot_charts.py             # uses latest in results/2_quantum_hardware/
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    with suppress(AttributeError, OSError):
        sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------- style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 100,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

# Colour-blind-safe palette (Wong 2011 — Nature Methods)
COLORS = {
    "ideal":           "#0072B2",   # blue
    "gpu":             "#009E73",   # bluish green
    "emulator":        "#E69F00",   # orange
    "emulator_4q_v2":  "#E69F00",
    "emulator_4q_v1":  "#D55E00",   # vermilion
    "emulator_8q_v1":  "#CC79A7",   # reddish purple
    "real":            "#000000",   # black — emphasises the headline result
}
BACKEND_ORDER = ["ideal", "gpu", "emulator_4q_v2", "emulator_4q_v1",
                  "emulator_8q_v1", "emulator", "real"]
BACKEND_LABEL = {
    "ideal":          "Ideal (Aer)",
    "gpu":            "GPU (CuPy)",
    "emulator_4q_v2": "Emu 4q v2",
    "emulator_4q_v1": "Emu 4q v1",
    "emulator_8q_v1": "Emu 8q v1",
    "emulator":       "Emulator",
    "real":           "Real chip",
}


# ============================================================================
# JSON loader
# ============================================================================

def load_data(json_path: Path) -> dict[str, Any]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return data


def group_by_experiment(data: dict) -> dict[str, dict[str, dict]]:
    """Return {experiment_label: {backend: entry}} preserving config order."""
    groups: dict[str, dict[str, dict]] = {}
    order: list[str] = []
    for e in data["entries"]:
        label = _short_label(e["experiment_title"], e.get("params") or {})
        if label not in groups:
            groups[label] = {}
            order.append(label)
        groups[label][e["backend"]] = e
    return {k: groups[k] for k in order}


def present_backends(data: dict) -> list[str]:
    s = {e["backend"] for e in data["entries"]}
    return [b for b in BACKEND_ORDER if b in s]


def _short_label(title: str, params: dict) -> str:
    """Compact label for axes: 'Bell', 'GHZ-2', 'BV s=010', 'Shor N=15'."""
    if title.startswith("Bell"):
        return "Bell"
    if title.startswith("GHZ"):
        n = params.get("n") or params.get("Number of qubits (N)") or ""
        return f"GHZ-{n}" if n else "GHZ"
    if title.startswith("Bernstein"):
        s = params.get("secret") or params.get("Hidden 3-bit string (s)") or ""
        return f"BV s={s}" if s else "BV"
    if title.startswith("Compiled Shor") or title.startswith("Shor"):
        return "Shor N=15"
    return title.split("[")[0].strip()


# ============================================================================
# 1. fidelity_comparison.png — 3×3 grid, one panel per experiment
# ============================================================================

def plot_fidelity_comparison(data: dict, output_dir: Path) -> Path:
    groups = group_by_experiment(data)
    backends = present_backends(data)
    if not groups or not backends:
        return _empty_plot(output_dir / "fidelity_comparison.png")

    items = list(groups.items())
    n = len(items)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(11, 3.0 * rows + 0.4),
                              squeeze=False, sharey=True)

    for idx in range(rows * cols):
        ax = axes[idx // cols, idx % cols]
        if idx >= n:
            ax.set_visible(False)
            continue
        label, by_backend = items[idx]

        present = [b for b in backends if b in by_backend]
        x = np.arange(len(present))
        means, lo_err, hi_err, colors = [], [], [], []
        for b in present:
            e = by_backend[b]
            m = e["mean"]
            ci_lo, ci_hi = e["ci_95"]
            means.append(m)
            lo_err.append(max(0.0, m - ci_lo))
            hi_err.append(max(0.0, ci_hi - m))
            colors.append(COLORS.get(b, "#888"))

        bars = ax.bar(x, means, yerr=[lo_err, hi_err], capsize=4,
                       color=colors, edgecolor="black", linewidth=0.5,
                       error_kw={"elinewidth": 1.2})

        # Value labels above each bar
        for xi, m in zip(x, means):
            ax.text(xi, min(m + 0.04, 1.04), f"{m:.3f}",
                    ha="center", va="bottom", fontsize=7.5)

        ax.set_xticks(x)
        ax.set_xticklabels([BACKEND_LABEL.get(b, b) for b in present],
                            rotation=20, ha="right", fontsize=8)
        ax.axhline(0.5, color="#888", linestyle=":", linewidth=0.7)
        ax.axhline(1.0, color="#bbb", linestyle="-", linewidth=0.5)
        ax.set_ylim(0, 1.1)
        ax.set_title(label, fontsize=10)
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    for r in range(rows):
        axes[r, 0].set_ylabel("Empirical fidelity / metric")

    fig.suptitle(
        f"Per-experiment fidelity across backends (95% CI bars, N={data['entries'][0]['repeats']} repeats)\n"
        f"{data['chip']['name']}, dotted line = random baseline (0.5)",
        fontsize=11, fontweight="bold", y=1.005,
    )
    fig.tight_layout()
    out = output_dir / "fidelity_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ============================================================================
# 2. fidelity_timeline.png — 3×3 grid, one panel per experiment
# ============================================================================

def plot_fidelity_timeline(data: dict, output_dir: Path) -> Path:
    """
    Per-run fidelity trajectories — 2×3 grid, one panel per BACKEND,
    each panel shows one line per experiment across N=10 repeats.

    Per-backend layout (rather than per-experiment) lets the eye compare
    how a single backend performs across all experiments at once — useful
    to spot which backend is the noisiest.
    """
    groups = group_by_experiment(data)
    backends = present_backends(data)
    if not backends:
        return _empty_plot(output_dir / "fidelity_timeline.png")

    # Experiment colour palette — 9 distinct, colour-blind-tolerant
    exp_palette = [
        "#0072B2",  # Bell — deep blue
        "#56B4E9",  # GHZ-2 — sky blue
        "#009E73",  # GHZ-3 — green
        "#CC79A7",  # GHZ-4 — magenta
        "#E69F00",  # BV s=000 — orange
        "#D55E00",  # BV s=010 — vermilion
        "#A0522D",  # BV s=101 — sienna
        "#F0E442",  # BV s=111 — yellow
        "#000000",  # Shor — black
    ]
    exp_order = list(groups.keys())
    exp_color = {lbl: exp_palette[i % len(exp_palette)]
                  for i, lbl in enumerate(exp_order)}

    # Force a 2×3 layout (2 rows × 3 cols = 6 panels, matches 6 backends).
    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(14.5, 7.5),
                              squeeze=False, sharey=True)

    for idx in range(rows * cols):
        ax = axes[idx // cols, idx % cols]
        if idx >= len(backends):
            ax.set_visible(False)
            continue
        backend = backends[idx]

        # Plot one line per experiment for this backend
        any_data = False
        for lbl in exp_order:
            e = groups[lbl].get(backend)
            if not e:
                continue
            fids = e.get("fidelities") or []
            if not fids:
                continue
            xs = np.arange(1, len(fids) + 1)
            ys = np.array(fids)
            c = exp_color[lbl]
            ax.plot(xs, ys, marker="o", color=c, linewidth=1.2,
                    markersize=4, label=lbl, alpha=0.9)
            any_data = True

        ax.set_title(BACKEND_LABEL.get(backend, backend), fontsize=10)
        ax.set_xlabel("Repeat #")
        ax.set_xlim(0.5, 10.5)
        ax.set_xticks(range(1, 11))
        ax.set_ylim(0, 1.08)
        ax.axhline(0.5, color="#bbb", linestyle=":", linewidth=0.7,
                   label="_nolegend_")
        ax.axhline(1.0, color="#ddd", linestyle="-", linewidth=0.5,
                   label="_nolegend_")
        ax.grid(True, alpha=0.25)
        ax.set_axisbelow(True)

    for r in range(rows):
        axes[r, 0].set_ylabel("Fidelity / metric")

    # One shared legend below, ordered as experiments — 3 columns × 3 rows for 9 entries
    handles, labels = [], []
    for ax_row in axes:
        for ax in ax_row:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l.startswith("_"):
                    continue
                if l not in labels:
                    handles.append(h)
                    labels.append(l)
    if handles:
        # Order legend in exp_order
        order_idx = {l: i for i, l in enumerate(exp_order)}
        paired = sorted(zip(labels, handles), key=lambda p: order_idx.get(p[0], 99))
        labels = [p[0] for p in paired]
        handles = [p[1] for p in paired]
        fig.legend(handles, labels, loc="lower center",
                   ncol=min(len(labels), 5),
                   bbox_to_anchor=(0.5, -0.01), frameon=False, fontsize=9)

    fig.suptitle(
        f"Per-run fidelity trajectories ({data['chip']['name']}) — "
        f"one panel per backend, one line per experiment (N={data['entries'][0]['repeats']} repeats)",
        fontsize=11, fontweight="bold", y=1.005,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.99])
    out = output_dir / "fidelity_timeline.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ============================================================================
# 3. timing_breakdown.png — log-y stacked bars
# ============================================================================

def plot_timing_breakdown(data: dict, output_dir: Path) -> Path:
    """
    Stacked-bar wall-clock breakdown faceted as a 3×3 grid: one panel per
    experiment, x-axis = backends, log-y because real chip (~170 s/run) and
    ideal (~0.1 s/run) differ by 1700×.
    """
    rows_by_exp: dict[str, list[dict]] = {}
    for e in data["entries"]:
        runs = e.get("per_run") or []
        if not runs:
            continue
        def mean(field):
            xs = [r["timing_s"].get(field, 0.0) for r in runs]
            return statistics.mean(xs) if xs else 0.0
        label = _short_label(e["experiment_title"], e.get("params") or {})
        rows_by_exp.setdefault(label, []).append({
            "backend":   e["backend"],
            "py_setup":  mean("python_setup_s"),
            "transpile": mean("transpile_s"),
            "submit":    mean("submit_s"),
            "queue":     mean("queue_s"),
            "execute":   mean("execute_s"),
            "py_post":   mean("python_post_s"),
        })

    label_order = []
    for e in data["entries"]:
        lbl = _short_label(e["experiment_title"], e.get("params") or {})
        if lbl in rows_by_exp and lbl not in label_order:
            label_order.append(lbl)
    if not label_order:
        return _empty_plot(output_dir / "timing_breakdown.png")

    cols = 3
    rows_count = (len(label_order) + cols - 1) // cols
    fig, axes = plt.subplots(rows_count, cols,
                              figsize=(13, 3.3 * rows_count + 0.5),
                              squeeze=False, sharey=True)

    phase_colors = {
        "py_setup":  "#E74C3C",
        "transpile": "#9B59B6",
        "submit":    "#3498DB",
        "queue":     "#7F8C8D",
        "execute":   "#27AE60",
        "py_post":   "#C0392B",
    }
    phase_label = {
        "py_setup":  "Python setup",
        "transpile": "Transpile (Qiskit)",
        "submit":    "Submit (HTTP)",
        "queue":     "Queue (Bauman)",
        "execute":   "Execute (chip / Aer / GPU)",
        "py_post":   "Python post",
    }
    phases = ["py_setup", "transpile", "submit", "queue", "execute", "py_post"]
    backend_idx = {b: i for i, b in enumerate(BACKEND_ORDER)}

    legend_handles = None
    for idx in range(rows_count * cols):
        ax = axes[idx // cols, idx % cols]
        if idx >= len(label_order):
            ax.set_visible(False)
            continue
        lbl = label_order[idx]
        rows = sorted(rows_by_exp[lbl], key=lambda r: backend_idx.get(r["backend"], 99))
        x = np.arange(len(rows))
        x_labels = [BACKEND_LABEL.get(r["backend"], r["backend"]) for r in rows]

        bottoms = np.full(len(rows), 0.0)
        # On log-y axis we cannot start at 0; we treat phases <1ms as invisible
        # but show them via the bar's bottom edge if present.
        EPS = 1e-4   # 0.1 ms floor visually
        bottoms = np.full(len(rows), EPS)
        for ph in phases:
            h = np.array([max(r[ph], EPS / 10) for r in rows])
            bars = ax.bar(x, h, bottom=bottoms, color=phase_colors[ph],
                          label=phase_label[ph], edgecolor="white",
                          linewidth=0.3)
            bottoms = bottoms + h
            if legend_handles is None:
                pass
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

        # Total label on top of each bar
        for xi, r in enumerate(rows):
            total = sum(r[ph] for ph in phases)
            if total <= 0:
                continue
            ax.text(xi, total * 1.3, _fmt_seconds(total),
                    ha="center", va="bottom", fontsize=7.5, color="#222")

        ax.set_yscale("log")
        ax.set_ylim(1e-3, 1e3)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=25, ha="right", fontsize=8)
        ax.set_title(lbl, fontsize=10)
        ax.grid(True, axis="y", alpha=0.25, which="both")
        ax.set_axisbelow(True)

    for r in range(rows_count):
        axes[r, 0].set_ylabel("Time / run (s, log)")

    if legend_handles:
        fig.legend(legend_handles, legend_labels,
                   loc="lower center", ncol=6,
                   bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=8)
    fig.suptitle(
        f"Wall-clock timing breakdown per experiment ({data['chip']['name']})\n"
        f"log-y axis: real-chip (~170 s) and ideal-Aer (~0.1 s) span ~1700× — both visible",
        fontsize=11, fontweight="bold", y=1.005,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.99])
    out = output_dir / "timing_breakdown.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ============================================================================
# 4. python_overhead.png — stacked bars per backend, value-labelled
# ============================================================================

def plot_python_overhead(data: dict, output_dir: Path) -> Path:
    """
    Per-backend stacked-bar view: which phases dominate wall-clock?
    Bar labels show η_py (Python overhead percentage).
    """
    by_backend: dict[str, list[dict]] = {}
    for e in data["entries"]:
        runs = e.get("per_run") or []
        if not runs:
            continue
        def mean(field):
            return statistics.mean([r["timing_s"].get(field, 0.0) for r in runs])
        row = {
            "label":     _short_label(e["experiment_title"], e.get("params") or {}),
            "py_setup":  mean("python_setup_s") * 1000,
            "transpile": mean("transpile_s") * 1000,
            "submit":    mean("submit_s") * 1000,
            "queue":     mean("queue_s") * 1000,
            "execute":   mean("execute_s") * 1000,
            "py_post":   mean("python_post_s") * 1000,
        }
        total = sum(row[k] for k in ("py_setup", "transpile", "submit",
                                       "queue", "execute", "py_post"))
        py_pct = (row["py_setup"] + row["py_post"]) / total * 100 if total else 0.0
        row["total"] = total
        row["py_pct"] = py_pct
        by_backend.setdefault(e["backend"], []).append(row)

    backends = [b for b in BACKEND_ORDER if b in by_backend]
    if not backends:
        return _empty_plot(output_dir / "python_overhead.png")

    n_back = len(backends)
    fig, axes = plt.subplots(n_back, 1, figsize=(11, 1.7 * n_back + 1.4),
                              squeeze=False, sharex=True)
    axes = axes[:, 0]

    phase_colors = {
        "py_setup":  "#E74C3C",
        "transpile": "#9B59B6",
        "submit":    "#3498DB",
        "queue":     "#7F8C8D",
        "execute":   "#27AE60",
        "py_post":   "#C0392B",
    }
    phase_label = {
        "py_setup":  "Python setup",
        "transpile": "Transpile",
        "submit":    "Submit",
        "queue":     "Queue",
        "execute":   "Execute",
        "py_post":   "Python post",
    }
    phases = ["py_setup", "transpile", "submit", "queue", "execute", "py_post"]

    last_labels: list[str] = []
    for ax, backend in zip(axes, backends):
        rows = by_backend[backend]
        n = len(rows)
        x = np.arange(n)
        last_labels = [r["label"] for r in rows]

        bottoms = np.zeros(n)
        # Skip phases that are sub-millisecond on average — they don't render
        # as visible slices on the linear-ms scale and clutter the legend with
        # noise (especially the queue/submit fields on local backends).
        for ph in phases:
            h = np.array([r[ph] if r[ph] >= 0.05 else 0.0 for r in rows])
            if h.sum() == 0:
                continue
            ax.bar(x, h, bottom=bottoms, color=phase_colors[ph],
                   label=phase_label[ph], edgecolor="white", linewidth=0.3)
            bottoms = bottoms + h

        for xi, r in enumerate(rows):
            color = "#C0392B" if r["py_pct"] > 5 else "#27AE60"
            ax.text(xi, r["total"] * 1.04, f"{r['py_pct']:.1f}%",
                    ha="center", va="bottom", fontsize=7, fontweight="bold",
                    color=color)

        ax.set_ylabel(f"{BACKEND_LABEL.get(backend, backend)}\nms/run",
                       fontsize=9)
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.set_ylim(0, max(bottoms.max() * 1.18 if bottoms.size else 1, 1))

    axes[-1].set_xticks(np.arange(len(last_labels)))
    axes[-1].set_xticklabels(last_labels, rotation=20, ha="right", fontsize=8)

    # One legend, on the top axis
    axes[0].legend(loc="upper right", ncol=3, framealpha=0.92, fontsize=8)
    axes[0].set_title(
        "Where each benchmark's wall-clock is spent (η_py % labels)\n"
        "Red = subtractable Python overhead • blue/green = algorithmic / chip",
        fontweight="bold",
    )
    fig.tight_layout()
    out = output_dir / "python_overhead.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ============================================================================
# 5. counts_overlay.png — observed vs expected with Wilson 95% CIs
# ============================================================================

def plot_counts_overlay(data: dict, output_dir: Path) -> Path:
    groups = group_by_experiment(data)
    items = list(groups.items())
    if not items:
        return _empty_plot(output_dir / "counts_overlay.png")

    cols = 2
    rows = (len(items) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 3.4 * rows),
                              squeeze=False)

    for idx in range(rows * cols):
        ax = axes[idx // cols, idx % cols]
        if idx >= len(items):
            ax.set_visible(False)
            continue
        label, by_backend = items[idx]
        real = by_backend.get("real")
        emu  = by_backend.get("emulator_4q_v2") or by_backend.get("emulator")
        ideal = by_backend.get("ideal")

        # Find the bitstring universe (union of keys across present backends)
        keys: list[str] = []
        for src in (ideal, emu, real):
            if not src:
                continue
            for r in src.get("per_run") or []:
                for k in r.get("counts", {}).keys():
                    if k not in keys:
                        keys.append(k)
        if not keys:
            continue
        # Sort by bit-string then length to keep "000…111" order
        keys.sort(key=lambda s: (len(s), s))

        x = np.arange(len(keys))

        # Expected (ideal): use ideal backend's mean P per key if available.
        # Light blue, full width, alpha-tinted — same look as the original.
        ideal_p = {k: 0.0 for k in keys}
        if ideal and ideal.get("per_run"):
            for r in ideal["per_run"]:
                total = sum(r["counts"].values()) or 1
                for k in keys:
                    ideal_p[k] += r["counts"].get(k, 0) / total
            for k in keys:
                ideal_p[k] /= len(ideal["per_run"])
        ax.bar(x, [ideal_p[k] for k in keys], width=0.78,
               color="#9bc4e2", alpha=0.4,
               label="expected (theory)")

        # Real-chip: mean P + Wilson 95% CI per bitstring across all shots.
        # Project-standard green (#0a7a3a) — same hue as in fidelity_comparison.
        if real and real.get("per_run"):
            n_shots_total = 0
            sums = {k: 0 for k in keys}
            for r in real["per_run"]:
                total = sum(r["counts"].values())
                n_shots_total += total
                for k in keys:
                    sums[k] += r["counts"].get(k, 0)
            real_p   = {k: (sums[k] / n_shots_total) if n_shots_total else 0.0 for k in keys}
            err_lo, err_hi = [], []
            for k in keys:
                lo, hi = _wilson_ci(sums[k], n_shots_total)
                err_lo.append(real_p[k] - lo)
                err_hi.append(hi - real_p[k])
            ax.bar(x, [real_p[k] for k in keys], width=0.46,
                   color="#0a7a3a", edgecolor="#063d1d", linewidth=0.5,
                   yerr=[err_lo, err_hi], error_kw={"elinewidth": 0.8,
                                                       "capsize": 2,
                                                       "ecolor": "#222"},
                   label=f"real (mean N={real['repeats']}, ±95% Wilson)")

        # Annotate "envelope-red skipped" case
        if not real or not real.get("per_run"):
            ax.text(0.5, 0.5,
                    "REAL skipped\n(envelope-red:\ndepth/CZ > limit)",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=10, color="#c0392b",
                    bbox=dict(boxstyle="round,pad=0.6", facecolor="#fdecea",
                              edgecolor="#c0392b", linewidth=1.0))

        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=0, fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("Probability")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    fig.suptitle(
        f"Observed vs expected outcome distributions ({data['chip']['name']})",
        fontsize=11, fontweight="bold", y=1.005,
    )
    fig.tight_layout()
    out = output_dir / "counts_overlay.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ============================================================================
# 6. chip_calibration.png — topology + worst-T2 highlight
# ============================================================================

def plot_chip_calibration(data: dict, output_dir: Path) -> Path:
    chip = data["chip"]
    n_q = chip["num_qubits"]
    timestamp = data.get("timestamp", "")

    # For Snowdrop 4q ver2: star topology, q2 hub
    positions = {0: (1, 1), 1: (0, 0), 2: (1, 0), 3: (2, 0)}
    if n_q != 4:
        positions = {q: (q, 0) for q in range(n_q)}

    t2_values = {int(q): float(v) for q, v in chip.get("per_qubit_t2_us", {}).items()}
    worst_t2_q = min(t2_values, key=t2_values.get) if t2_values else None
    worst_t2_v = t2_values.get(worst_t2_q) if worst_t2_q is not None else None

    fig, ax = plt.subplots(figsize=(8, 6))

    # Couplings
    for pair_str, f in chip.get("per_pair_f2q", {}).items():
        a, b = _parse_pair(pair_str)
        if a is None:
            continue
        xa, ya = positions.get(a, (a, 0))
        xb, yb = positions.get(b, (b, 0))
        ax.plot([xa, xb], [ya, yb], color="#444", linewidth=2.5, alpha=0.55,
                zorder=1)
        mx, my = (xa + xb) / 2, (ya + yb) / 2
        ax.text(mx, my + 0.07, f"CZ {f*100:.2f}%", ha="center",
                va="bottom", fontsize=9, color="#222",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                          edgecolor="#aaa", linewidth=0.6))

    # Qubits — label position depends on qubit's location in the topology:
    #   q0 (top of star)  → label to the LEFT of the circle so it doesn't
    #     collide with the CZ-99.25% coupling label sitting below it.
    #   q1, q2, q3 (bottom row) → label ABOVE the circle as usual.
    # Worst-T2 qubit is highlighted with a red ring + small ⚠ tag.
    for q in range(n_q):
        x, y = positions.get(q, (q, 0))
        t1 = float(chip["per_qubit_t1_us"].get(str(q), 0))
        t2 = float(chip["per_qubit_t2_us"].get(str(q), 0))
        f1q = float(chip["per_qubit_f1q"].get(str(q), 0)) * 100
        ro = float(chip["per_qubit_ro"].get(str(q), 0)) * 100
        fq = float(chip["per_qubit_freq_ghz"].get(str(q), 0))
        is_worst = (q == worst_t2_q)
        ring_color = "#c0392b" if is_worst else "black"
        ring_width = 2.5 if is_worst else 1.2
        ax.scatter([x], [y], s=2200, c="#0a7a3a",
                   edgecolors=ring_color, linewidths=ring_width, zorder=3)
        # Label + info placement:
        #   q0 (top of star) → LEFT of the circle, so the CZ-99.25% coupling
        #     label between q0 and q2 has clean vertical space.
        #   q1, q2, q3 (bottom row) → label above, info below.
        info = (f"f = {fq:.2f} GHz\n"
                f"T1 = {t1:.1f} μs   T2 = {t2:.1f} μs\n"
                f"F_1q = {f1q:.2f}%   F_RO = {ro:.1f}%")
        if is_worst:
            info += "\n⚠ worst T2 (envelope driver)"
        color_q = "#c0392b" if is_worst else "#0a4a25"
        color_info = "#c0392b" if is_worst else "#222"
        if q == 0 and n_q == 4:
            # q0 label to the left of the circle, info just below the label
            ax.text(x - 0.22, y + 0.05, f"q{q}",
                    ha="right", va="bottom",
                    fontweight="bold", fontsize=14,
                    color=color_q, zorder=4)
            ax.text(x - 0.22, y - 0.02, info, ha="right", va="top",
                    fontsize=8.5, color=color_info)
        else:
            ax.text(x, y + 0.18, f"q{q}",
                    ha="center", va="bottom",
                    fontweight="bold", fontsize=14,
                    color=color_q, zorder=4)
            ax.text(x, y - 0.20, info, ha="center", va="top",
                    fontsize=8.5, color=color_info)

    # Symmetric x-axis around the topology centroid (x≈1.0). q0's left-aligned
    # info block extends to roughly x=0.15 — well within [-0.8, 2.8].
    ax.set_xlim(-0.8, 2.8)
    ax.set_ylim(-0.9, 1.7)
    ax.set_aspect("equal")
    ax.axis("off")
    title = f"{chip['name']} — live calibration ({timestamp})"
    if worst_t2_q is not None:
        title += f"\nworst-T2 qubit highlighted (q{worst_t2_q}: {worst_t2_v:.1f} μs)"
    ax.set_title(title, fontweight="bold")

    fig.tight_layout()
    out = output_dir / "chip_calibration.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ============================================================================
# helpers
# ============================================================================

def _empty_plot(path: Path) -> Path:
    fig = plt.figure(figsize=(6, 2))
    fig.text(0.5, 0.5, "(no data)", ha="center", va="center", color="#888")
    fig.savefig(path)
    plt.close(fig)
    return path


def _fmt_seconds(s: float) -> str:
    if s < 0.001:
        return f"{s*1e6:.0f}μs"
    if s < 1.0:
        return f"{s*1000:.1f}ms"
    if s < 60.0:
        return f"{s:.1f}s"
    return f"{s/60:.1f}min"


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% CI for a proportion k/n. Returns (lo, hi)."""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _parse_pair(s) -> tuple[int | None, int | None]:
    """
    Parse a qubit-pair key. The JSON serialiser used by `save_run_json`
    encodes tuple keys as 'A-B' (e.g. '0-2'), but we also handle the
    legacy '(0, 2)' / '[0, 2]' / '0,2' formats for backwards compat.
    """
    if isinstance(s, (tuple, list)) and len(s) == 2:
        return int(s[0]), int(s[1])
    s = str(s).strip("()[] ")
    for sep in ("-", ","):
        if sep in s:
            try:
                a, b = s.split(sep, 1)
                return int(a.strip()), int(b.strip())
            except Exception:
                continue
    return None, None


# ============================================================================
# CLI
# ============================================================================

def find_latest_results() -> Path | None:
    root = Path(__file__).resolve().parent.parent
    base = root / "results" / "2_quantum_hardware"
    if not base.exists():
        return None
    dirs = [d for d in base.iterdir() if d.is_dir() and (d / "quantum_hardware_runs.json").exists()]
    if not dirs:
        return None
    return max(dirs, key=lambda d: d.stat().st_mtime)


def main():
    p = argparse.ArgumentParser(
        description="Re-render publication-quality benchmark charts from a saved JSON."
    )
    p.add_argument("results_dir", nargs="?", default=None,
                   help="Path to a 2_quantum_hardware/<timestamp>/ folder. "
                        "Defaults to the most recently modified one.")
    args = p.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else find_latest_results()
    if results_dir is None or not results_dir.exists():
        print("ERROR: no results directory found.")
        sys.exit(1)
    json_path = results_dir / "quantum_hardware_runs.json"
    if not json_path.exists():
        print(f"ERROR: {json_path} not found.")
        sys.exit(1)

    print(f"Loading: {json_path}")
    data = load_data(json_path)
    print(f"  {len(data['entries'])} entries, chip = {data['chip']['name']}")
    print(f"  Timestamp: {data.get('timestamp', 'n/a')}")
    print()
    print("Re-rendering charts in place:")
    for fn in (plot_fidelity_comparison,
                plot_fidelity_timeline,
                plot_timing_breakdown,
                plot_python_overhead,
                plot_counts_overlay,
                plot_chip_calibration):
        out = fn(data, results_dir)
        size_kb = out.stat().st_size / 1024
        print(f"  ✓  {out.name:<28} ({size_kb:.0f} KB)")
    print()
    print(f"Done. Replotted in {results_dir}")


if __name__ == "__main__":
    main()

