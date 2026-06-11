"""
Live CSV chart previews for the QCCB GUI.

Every pipeline (PQC suite, CRM forecast, GHZ-N scaling, full benchmark)
keeps writing its publication PNG charts to results/ exactly as before.
This module adds a second, screen-oriented path: it reads the *CSV*
artifacts directly and re-renders them as native matplotlib figures inside
the GUI — vector-crisp at any window size, with fonts sized for a
projector, updating live while a run is still in flight.

Two public entry points:

    scan_new_csvs(results_root, since)  -> CSVs written after a timestamp
    render_csv(fig, csv_path)           -> draw a preview into a Figure

`render_csv` dispatches on the file name: every CSV family the pipelines
produce has a dedicated renderer; anything unknown falls back to a generic
bar-chart / table view, so a preview is always available.
"""
from __future__ import annotations

import csv
from pathlib import Path

from matplotlib.figure import Figure

from src.ui_theme import CHART_COLORS, PALETTE

# Preview text sizes — tuned for reading the chart from across a room,
# not for print. The saved PNGs keep their own publication sizing.
SZ_TITLE = 15
SZ_LABEL = 13
SZ_TICK = 12
SZ_LEGEND = 12
SZ_VALUE = 11

BACKEND_COLOR = {
    "ideal":           CHART_COLORS["ideal"],
    "gpu":             CHART_COLORS["gpu"],
    "cpu_aer":         CHART_COLORS["ideal"],
    "gpu_cupy":        CHART_COLORS["gpu"],
    "emulator":        CHART_COLORS["emulator"],
    "emulator_4q_v2":  CHART_COLORS["emulator"],
    "emulator_4q_v1":  "#C8742A",
    "emulator_8q_v1":  "#9A5BD2",
    "real":            CHART_COLORS["real"],
}
BACKEND_ORDER = ["ideal", "gpu", "emulator_4q_v2", "emulator_4q_v1",
                 "emulator_8q_v1", "emulator", "real"]


# ---------------------------------------------------------------- scanning

def scan_new_csvs(results_root: Path, since: float) -> list[Path]:
    """CSV files under results/ modified at/after `since`, oldest first.

    Run folders are results/<category>/<timestamp>/, so two directory
    levels cover every pipeline. Old run folders are pruned by mtime
    before their files are even listed, keeping the poll cheap.
    """
    found: dict[Path, float] = {}
    # 60 s slack tolerates fs-timestamp rounding on run-dir creation.
    for run_dir in _run_dirs(results_root, since - 60):
        for f in _safe_listdir(run_dir):
            if f.suffix.lower() != ".csv":
                continue
            mtime = _mtime(f)
            if mtime is not None and mtime >= since:
                found[f] = mtime
    return sorted(found, key=found.get)


def _run_dirs(results_root: Path, cutoff: float) -> list[Path]:
    """results/<category>/ plus its run subfolders touched after `cutoff`."""
    dirs: list[Path] = []
    for cat in _safe_listdir(results_root):
        if not cat.is_dir():
            continue
        dirs.append(cat)
        dirs.extend(d for d in _safe_listdir(cat)
                    if d.is_dir() and (_mtime(d) or 0.0) >= cutoff)
    return dirs


def _safe_listdir(d: Path) -> list[Path]:
    try:
        return list(d.iterdir())
    except OSError:
        return []


def _mtime(p: Path) -> float | None:
    try:
        return p.stat().st_mtime
    except OSError:
        return None


# ------------------------------------------------------------- CSV loading

def _read_rows(path: Path) -> list[dict[str, str]]:
    """DictReader over a UTF-8 CSV, ignoring '#'-comment and blank lines.

    Blank lines must go too: pqc_avx2_comparison.csv separates its '#'
    metadata block from the header with one, and DictReader would
    otherwise mistake that empty line for the header row.
    """
    text = path.read_text(encoding="utf-8-sig")
    lines = [ln for ln in text.splitlines()
             if ln.strip() and not ln.lstrip().startswith("#")]
    rows = list(csv.DictReader(lines))
    if not rows:
        raise ValueError("CSV has a header but no data rows")
    return rows


def _num(value, default=None):
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _backend_sorted(backends) -> list[str]:
    order = {b: i for i, b in enumerate(BACKEND_ORDER)}
    return sorted(backends, key=lambda b: order.get(b, 99))


# ---------------------------------------------------------- shared drawing

def _grouped_bars(ax, group_labels, series, errs=None, log_y=False):
    """series = {name: [value per group]}; errs optional, same shape."""
    import numpy as np
    x = np.arange(len(group_labels))
    n = max(len(series), 1)
    width = 0.84 / n
    offset = -0.42 + width / 2
    for name, vals in series.items():
        err = errs.get(name) if errs else None
        kwargs = {}
        if err is not None:
            kwargs = {"yerr": err, "capsize": 3,
                      "error_kw": {"elinewidth": 1.1}}
        ax.bar(x + offset, vals, width, label=name,
               color=BACKEND_COLOR.get(name), **kwargs)
        offset += width
    ax.set_xticks(x)
    rotation = 25 if max(len(s) for s in group_labels) > 6 else 0
    ax.set_xticklabels(group_labels, rotation=rotation,
                       ha="right" if rotation else "center",
                       fontsize=SZ_TICK)
    if log_y:
        ax.set_yscale("log")
    ax.tick_params(labelsize=SZ_TICK)
    ax.legend(fontsize=SZ_LEGEND, loc="best")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)


def _value_labels(ax, bars, fmt="{:.2f}"):
    for b in bars:
        h = b.get_height()
        if h <= 0:
            continue
        ax.text(b.get_x() + b.get_width() / 2, h, fmt.format(h),
                ha="center", va="bottom", fontsize=SZ_VALUE)


def _render_table(fig: Figure, path: Path, rows: list[dict],
                  max_rows: int = 12, max_cols: int = 7):
    cols = list(rows[0].keys())[:max_cols]
    shown = rows[:max_rows]
    cell_text = [[str(r.get(c, ""))[:36] for c in cols] for r in shown]
    ax = fig.add_subplot(111)
    ax.axis("off")
    table = ax.table(cellText=cell_text, colLabels=cols,
                     cellLoc="left", colLoc="left", loc="upper center")
    table.auto_set_font_size(False)
    table.set_fontsize(SZ_VALUE)
    table.scale(1, 1.7)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor(PALETTE["border"])
        if row == 0:
            cell.set_facecolor(PALETTE["bg"])
            cell.set_text_props(weight="bold")
    title = path.stem
    if len(rows) > max_rows:
        title += f"   (first {max_rows} of {len(rows)} rows)"
    ax.set_title(title, fontsize=SZ_TITLE, fontweight="bold")


# ------------------------------------------------------ specific renderers

def _render_ghz_scaling(fig: Figure, path: Path):
    rows = _read_rows(path)
    ax = fig.add_subplot(111)
    for backend, label in (("cpu_aer", "CPU (Qiskit Aer)"),
                           ("gpu_cupy", "GPU (CuPy)")):
        pts = [(int(_num(r["n_qubits"], 0)), _num(r["wall_s"]))
               for r in rows
               if r["backend"] == backend and r.get("ok") == "True"]
        pts = [(n, w) for n, w in pts if w and w > 0]
        if pts:
            xs, ys = zip(*sorted(pts))
            ax.plot(xs, ys, marker="o", linewidth=2.2, markersize=7,
                    label=label, color=BACKEND_COLOR[backend])
    ax.axvline(4, color=PALETTE["danger"], linestyle="--", linewidth=1.8,
               label="Snowdrop ceiling (N=4)")
    ax.set_yscale("log")
    ax.set_xlabel("Qubits N (GHZ-N)", fontsize=SZ_LABEL)
    ax.set_ylabel("Wall-clock, s (log)", fontsize=SZ_LABEL)
    ax.set_title("GHZ-N scaling — CPU vs GPU", fontsize=SZ_TITLE,
                 fontweight="bold")
    ax.tick_params(labelsize=SZ_TICK)
    ax.legend(fontsize=SZ_LEGEND)
    ax.grid(True, which="both", alpha=0.3)


def _render_crm_table(fig: Figure, path: Path):
    rows = _read_rows(path)
    ax = fig.add_subplot(111)
    seen: dict[tuple, int] = {}  # chips share coords; stagger their labels
    for r in rows:
        year, crm = _num(r["year"]), _num(r["log2_CRM"])
        if year is None or crm is None:
            continue
        nth = seen.get((year, crm), 0)
        seen[(year, crm)] = nth + 1
        measured = str(r.get("measured", "")).lower() in ("yes", "true", "1")
        if measured:
            ax.scatter([year], [crm], s=220, marker="*", zorder=5,
                       color=PALETTE["danger"],
                       label=f"{r['chip']} (measured)")
            ax.annotate(r["chip"], (year, crm), textcoords="offset points",
                        xytext=(10, 8 + 13 * nth), fontsize=SZ_TICK,
                        color=PALETTE["danger"], fontweight="bold")
        else:
            ax.scatter([year], [crm], s=70, color=CHART_COLORS["ideal"],
                       zorder=3)
            ax.annotate(r["chip"], (year, crm), textcoords="offset points",
                        xytext=(6, 5 + 13 * nth), fontsize=SZ_VALUE - 1,
                        color=PALETTE["text_muted"])
    ax.set_xlabel("Year", fontsize=SZ_LABEL)
    ax.set_ylabel("log₂(CRM) — factorable-N capability", fontsize=SZ_LABEL)
    ax.set_title("CRM: cryptanalytic capability per chip",
                 fontsize=SZ_TITLE, fontweight="bold")
    ax.tick_params(labelsize=SZ_TICK)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles[:1], labels[:1], fontsize=SZ_LEGEND,
                  loc="upper left")
    ax.grid(True, alpha=0.3)


def _render_pqc_benchmarks(fig: Figure, path: Path):
    rows = _read_rows(path)
    kem = [r for r in rows if r.get("type") == "KEM"]
    sig = [r for r in rows if str(r.get("type", "")).lower().startswith("sig")]
    panels = [p for p in (("KEM (ms, log)", kem,
                           [("keygen_ms_mean", "KeyGen"),
                            ("encaps_ms_mean", "Encaps"),
                            ("decaps_ms_mean", "Decaps")]),
                          ("Signatures (ms, log)", sig,
                           [("keygen_ms_mean", "KeyGen"),
                            ("sign_ms_mean", "Sign"),
                            ("verify_ms_mean", "Verify")]))
              if p[1]]
    if not panels:
        raise ValueError("no KEM/SIG rows")
    import numpy as np
    for i, (title, subset, fields) in enumerate(panels):
        ax = fig.add_subplot(1, len(panels), i + 1)
        labels = [r["algorithm"] for r in subset]
        x = np.arange(len(labels))
        width = 0.84 / len(fields)
        offset = -0.42 + width / 2
        for j, (field, name) in enumerate(fields):
            vals = [max(_num(r.get(field), 0.0) or 0.0, 1e-3)
                    for r in subset]
            ax.bar(x + offset, vals, width, label=name,
                   color=CHART_COLORS["series"][j])
            offset += width
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right",
                           fontsize=SZ_TICK - 1)
        ax.set_title(title, fontsize=SZ_TITLE - 1, fontweight="bold")
        ax.tick_params(labelsize=SZ_TICK)
        ax.legend(fontsize=SZ_LEGEND - 1)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("PQC benchmark — operation timings", fontsize=SZ_TITLE,
                 fontweight="bold")


def _render_comparative(fig: Figure, path: Path):
    rows = [r for r in _read_rows(path) if _num(r.get("keygen_ms_mean"))]
    if not rows:
        raise ValueError("no keygen timings")
    ax = fig.add_subplot(111)
    labels = [r["algorithm"] for r in rows]
    vals = [_num(r["keygen_ms_mean"]) for r in rows]
    colors = [PALETTE["success"]
              if str(r.get("quantum_safe", "")).upper() in ("YES", "TRUE")
              else PALETTE["danger"] for r in rows]
    bars = ax.bar(range(len(labels)), vals, color=colors)
    _value_labels(ax, bars, fmt="{:.3g}")
    ax.set_yscale("log")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=SZ_TICK)
    ax.set_ylabel("KeyGen, ms (log)", fontsize=SZ_LABEL)
    ax.set_title("Key generation cost  "
                 "(green = quantum-safe, red = Shor-vulnerable)",
                 fontsize=SZ_TITLE - 1, fontweight="bold")
    ax.tick_params(labelsize=SZ_TICK)
    ax.grid(True, axis="y", alpha=0.3)


def _render_hw_summary(fig: Figure, path: Path):
    rows = _read_rows(path)
    exp_col = "experiment" if "experiment" in rows[0] else "Experiment"
    be_col = "backend" if "backend" in rows[0] else "Backend"
    mean_col = "mean" if "mean" in rows[0] else "Mean"
    lo_col = "ci_lo_95" if "ci_lo_95" in rows[0] else "CI95_low"
    hi_col = "ci_hi_95" if "ci_hi_95" in rows[0] else "CI95_high"

    def label(r):
        params = (r.get("params") or "{}").strip()
        return r[exp_col] if params in ("{}", "") else \
            f"{r[exp_col]} {params}"

    groups: list[str] = []
    data: dict[str, dict[str, tuple]] = {}
    for r in rows:
        lbl = label(r)
        if lbl not in groups:
            groups.append(lbl)
        m = _num(r.get(mean_col))
        if m is None:
            continue
        lo = _num(r.get(lo_col), m)
        hi = _num(r.get(hi_col), m)
        data.setdefault(r[be_col], {})[lbl] = (m, max(0.0, m - lo),
                                               max(0.0, hi - m))
    series, errs = {}, {}
    for be in _backend_sorted(data):
        series[be] = [data[be].get(g, (0, 0, 0))[0] for g in groups]
        errs[be] = [[data[be].get(g, (0, 0, 0))[1] for g in groups],
                    [data[be].get(g, (0, 0, 0))[2] for g in groups]]
    ax = fig.add_subplot(111)
    _grouped_bars(ax, groups, series, errs)
    ax.set_ylim(0, 1.12)
    ax.axhline(1.0, color=PALETTE["border"], linewidth=0.8)
    ax.set_ylabel("Fidelity / metric (95% CI)", fontsize=SZ_LABEL)
    ax.set_title("Fidelity by experiment and backend",
                 fontsize=SZ_TITLE, fontweight="bold")


def _render_hw_timing(fig: Figure, path: Path):
    rows = _read_rows(path)
    groups: list[str] = []
    data: dict[str, dict[str, float]] = {}
    for r in rows:
        # Rows are keyed by (experiment, params, backend) — keep the params
        # in the label or the ghz/bv variants overwrite each other.
        params = (r.get("params") or "{}").strip()
        lbl = r["experiment"] if params in ("{}", "") else \
            f"{r['experiment']} {params}"
        if lbl not in groups:
            groups.append(lbl)
        t = _num(r.get("mean_total_s"))
        if t and t > 0:
            data.setdefault(r["backend"], {})[lbl] = t
    series = {be: [max(data[be].get(g, 0.0), 1e-4) for g in groups]
              for be in _backend_sorted(data)}
    ax = fig.add_subplot(111)
    _grouped_bars(ax, groups, series, log_y=True)
    ax.set_ylabel("Mean total, s (log)", fontsize=SZ_LABEL)
    ax.set_title("Wall-clock per run — experiment × backend",
                 fontsize=SZ_TITLE, fontweight="bold")


def _render_timing_per_backend(fig: Figure, path: Path):
    rows = _read_rows(path)
    ax = fig.add_subplot(111)
    labels = [r["Backend"] for r in rows]
    vals = [max(_num(r["Mean_total_s"], 0.0) or 0.0, 1e-4) for r in rows]
    colors = [BACKEND_COLOR.get(b, "#888") for b in labels]
    bars = ax.bar(labels, vals, color=colors)
    _value_labels(ax, bars, fmt="{:.3g}s")
    ax.set_yscale("log")
    ax.set_ylabel("Mean total per run, s (log)", fontsize=SZ_LABEL)
    ax.set_title("Timing per backend", fontsize=SZ_TITLE, fontweight="bold")
    ax.tick_params(labelsize=SZ_TICK)
    ax.grid(True, axis="y", alpha=0.3)


def _render_sci(fig: Figure, path: Path):
    rows = _read_rows(path)
    name_col = "algorithm" if "algorithm" in rows[0] else "Operation"
    sci_col = "sci" if "sci" in rows[0] else "SCI_QCCB"
    ax = fig.add_subplot(111)

    def label(r):
        # sci_qccb_native.csv repeats Operation values ("Key encaps L1"
        # appears for Kyber, McEliece and HQC) — the PQC column is what
        # actually distinguishes the bars.
        alg = (r.get("PQC") or "").strip()
        return f"{alg}\n{r[name_col]}" if alg else r[name_col]

    labels = [label(r) for r in rows]
    vals = [_num(r.get(sci_col), 0.0) or 0.0 for r in rows]
    bars = ax.bar(range(len(labels)), vals, color=CHART_COLORS["series"][0])
    _value_labels(ax, bars)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=SZ_TICK)
    ax.axhline(2.0, color=PALETTE["warning"], linestyle="--", linewidth=1.4,
               label="SCI = 2 (production threshold)")
    ax.set_ylabel("SCI (lower = better)", fontsize=SZ_LABEL)
    ax.set_title("Security-Cost Index", fontsize=SZ_TITLE, fontweight="bold")
    ax.tick_params(labelsize=SZ_TICK)
    ax.legend(fontsize=SZ_LEGEND)
    ax.grid(True, axis="y", alpha=0.3)


def _render_sci_hw(fig: Figure, path: Path):
    rows = _read_rows(path)
    groups: list[str] = []
    data: dict[str, dict[str, float]] = {}
    for r in rows:
        lbl = r["experiment"]
        if lbl not in groups:
            groups.append(lbl)
        v = _num(r.get("SCI_HW"))
        if v is not None:
            data.setdefault(r["backend"], {})[lbl] = v
    series = {be: [data[be].get(g, 0.0) for g in groups]
              for be in _backend_sorted(data)}
    ax = fig.add_subplot(111)
    _grouped_bars(ax, groups, series)
    ax.set_ylabel("SCI_HW (lower = better)", fontsize=SZ_LABEL)
    ax.set_title("Hardware Security-Cost Index", fontsize=SZ_TITLE,
                 fontweight="bold")


def _render_chip_qubits(fig: Figure, path: Path):
    rows = _read_rows(path)
    import numpy as np
    ax = fig.add_subplot(111)
    x = np.arange(len(rows))
    t1 = [_num(r["T1_us"], 0.0) or 0.0 for r in rows]
    t2 = [_num(r["T2_us"], 0.0) or 0.0 for r in rows]
    b1 = ax.bar(x - 0.2, t1, 0.4, label="T1, μs",
                color=CHART_COLORS["series"][0])
    b2 = ax.bar(x + 0.2, t2, 0.4, label="T2, μs",
                color=CHART_COLORS["series"][1])
    _value_labels(ax, b1, fmt="{:.1f}")
    _value_labels(ax, b2, fmt="{:.1f}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"q{r['qubit']}" for r in rows], fontsize=SZ_TICK)
    ax.set_ylabel("Coherence time, μs", fontsize=SZ_LABEL)
    ax.set_title("Per-qubit calibration — T1 / T2", fontsize=SZ_TITLE,
                 fontweight="bold")
    ax.tick_params(labelsize=SZ_TICK)
    ax.legend(fontsize=SZ_LEGEND)
    ax.grid(True, axis="y", alpha=0.3)


def _render_classical_baseline(fig: Figure, path: Path):
    rows = _read_rows(path)
    ax = fig.add_subplot(111)
    labels = [f"{r['Algorithm']}\n{r['Operation']}" for r in rows]
    vals = [max(_num(r["Mean_ms"], 0.0) or 0.0, 1e-4) for r in rows]
    ax.bar(range(len(labels)), vals, color=CHART_COLORS["series"][0])
    ax.set_yscale("log")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=SZ_TICK - 2)
    ax.set_ylabel("Mean, ms (log)", fontsize=SZ_LABEL)
    ax.set_title("Classical baseline timings", fontsize=SZ_TITLE,
                 fontweight="bold")
    ax.tick_params(labelsize=SZ_TICK)
    ax.grid(True, axis="y", alpha=0.3)


# -------------------------------------------------------- generic fallback

def _render_generic(fig: Figure, path: Path):
    rows = _read_rows(path)
    cols = list(rows[0].keys())
    numeric = [c for c in cols
               if sum(_num(r.get(c)) is not None for r in rows) >= len(rows) * 0.8]
    text_cols = [c for c in cols if c not in numeric]
    if not numeric:
        _render_table(fig, path, rows)
        return

    label_col = text_cols[0] if text_cols else None
    shown = rows[:24]
    labels = ([str(r.get(label_col, ""))[:18] for r in shown]
              if label_col else [str(i + 1) for i in range(len(shown))])
    series = {c: [_num(r.get(c), 0.0) or 0.0 for r in shown]
              for c in numeric[:3]}

    positives = [v for vals in series.values() for v in vals if v > 0]
    log_y = (bool(positives) and max(positives) / max(min(positives), 1e-12) > 200
             and all(v >= 0 for vals in series.values() for v in vals))
    if log_y:
        series = {c: [max(v, 1e-4) for v in vals]
                  for c, vals in series.items()}

    import numpy as np
    ax = fig.add_subplot(111)
    x = np.arange(len(labels))
    width = 0.84 / len(series)
    offset = -0.42 + width / 2
    for i, (c, vals) in enumerate(series.items()):
        ax.bar(x + offset, vals, width, label=c,
               color=CHART_COLORS["series"][i % len(CHART_COLORS["series"])])
        offset += width
    if log_y:
        ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=SZ_TICK - 2)
    title = path.stem
    if len(rows) > len(shown):
        title += f"  (first {len(shown)} of {len(rows)} rows)"
    ax.set_title(title, fontsize=SZ_TITLE, fontweight="bold")
    ax.tick_params(labelsize=SZ_TICK)
    ax.legend(fontsize=SZ_LEGEND - 1)
    ax.grid(True, axis="y", alpha=0.3)


_TABLE_FILES = {
    "quantum_threat_analysis.csv",
    "vulnerability_matrix.csv",
    "quantum_hardware_backends.csv",
    "quantum_hardware_chip_2q.csv",
    "statistical_tests.csv",
    "shor_small_n_table.csv",
}

_RENDERERS = {
    "ghz_scaling.csv":                  _render_ghz_scaling,
    "crm_table.csv":                    _render_crm_table,
    "pqc_benchmarks.csv":               _render_pqc_benchmarks,
    "comparative_analysis.csv":         _render_comparative,
    "quantum_hardware_summary.csv":     _render_hw_summary,
    "algorithmic_fidelity_bca.csv":     _render_hw_summary,
    "quantum_hardware_timing.csv":      _render_hw_timing,
    "timing_per_backend.csv":           _render_timing_per_backend,
    "sci_analysis.csv":                 _render_sci,
    "sci_qccb_native.csv":              _render_sci,
    "quantum_sci_hw.csv":               _render_sci_hw,
    "quantum_hardware_chip.csv":        _render_chip_qubits,
    "extended_classical_baseline.csv":  _render_classical_baseline,
}


def _render_known_table(fig: Figure, path: Path) -> None:
    _render_table(fig, path, _read_rows(path))


def render_csv(fig: Figure, path: Path) -> None:
    """Clear `fig` and draw a preview of `path` into it.

    Raises on unreadable/empty files so the caller can fall back to the
    pipeline-generated PNG. The render is validated on a scratch figure
    first: half-written or malformed CSVs (normal while a pipeline is
    mid-flight) leave the caller's figure — and whatever chart it is
    currently showing — completely untouched.
    """
    path = Path(path)
    renderer = (_render_known_table if path.name in _TABLE_FILES
                else _RENDERERS.get(path.name, _render_generic))
    probe = Figure(figsize=tuple(fig.get_size_inches()))
    renderer(probe, path)  # raises here on bad data, fig stays intact
    fig.clear()
    renderer(fig, path)
