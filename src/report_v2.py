"""
Scientific report generator for the quantum-hardware benchmark suite.

Produces (in `results/quantum/`):
  - quantum_hardware_runs.json          — full raw data (every run, every count)
  - quantum_hardware_summary.csv        — per (experiment, backend) statistics
  - quantum_hardware_timing.csv         — per (experiment, backend) wall-clock breakdown
  - quantum_hardware_chip.csv           — per-qubit calibration snapshot
  - quantum_hardware_chip_2q.csv        — per-pair CZ fidelity snapshot
  - quantum_sci_hw.csv                  — hardware-aware SCI per (experiment, backend)
  - fidelity_comparison.png             — main thesis figure (4 exp × 3 backends, error bars)
  - fidelity_timeline.png               — per-run fidelity trajectories
  - timing_breakdown.png                — transpile/submit/queue/execute bars
  - counts_overlay.png                  — observed vs expected probability per experiment
  - chip_calibration.png                — qubit topology with T1/T2/F annotations
  - report.md                           — markdown thesis-ready report
  - benchmark.log                       — timestamped textual log
"""
from __future__ import annotations

import csv
import json
import math
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.experiments import ExperimentDef
from src.experiments.runner import BenchmarkResult, ExperimentResult
from src.quantum_hardware import ChipSpec, CHIP_NAME
from src.sci_hardware import SCI_HW, compute_sci_hw


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "2_quantum_hardware" / time.strftime("%d%m%Y %H %M")


def _t_critical_95(n: int) -> float:
    if n < 2:
        return 0.0
    try:
        from scipy.stats import t as student_t
        return float(student_t.ppf(0.975, n - 1))
    except Exception:
        return 2.0


def _ci95(values: list[float]) -> tuple[float, float, float, float]:
    """Return (mean, stdev, ci_lo, ci_hi) for a list of measurements."""
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    mean = statistics.mean(values)
    if len(values) < 2:
        return mean, 0.0, mean, mean
    sd = statistics.stdev(values)
    margin = _t_critical_95(len(values)) * sd / math.sqrt(len(values))
    return mean, sd, mean - margin, mean + margin


def _lag1_autocorr(x: list[float]) -> float:
    """Lag-1 autocorrelation; |value| << 1 means runs are essentially independent."""
    n = len(x)
    if n < 3:
        return 0.0
    mean = sum(x) / n
    num = sum((x[i] - mean) * (x[i + 1] - mean) for i in range(n - 1))
    den = sum((v - mean) ** 2 for v in x)
    return num / den if den > 0 else 0.0


class ReportV2:
    """
    Builds the full results/quantum/ artifact set from a list of BenchmarkResults.

    Usage:
        rep = ReportV2(output_dir=Path("results/quantum"), chip_spec=spec)
        rep.add(exp, params, bench_result)
        ...
        rep.write_all()
    """

    def __init__(self, output_dir: Path = DEFAULT_OUTPUT,
                 chip_spec: ChipSpec | None = None,
                 timestamp: str | None = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chip_spec = chip_spec
        self.timestamp = timestamp or time.strftime("%Y-%m-%d %H:%M:%S")

        # Each entry: (exp, params, bench_result)
        self.entries: list[tuple[ExperimentDef, dict, BenchmarkResult]] = []
        self.log_lines: list[str] = []

    def add(self, exp: ExperimentDef, params: dict, bench: BenchmarkResult):
        self.entries.append((exp, params, bench))
        self.log(f"+ {exp.title_with_params(params)} on {bench.backend}: "
                 f"N={bench.repeats}, mean={bench.mean:.4f} ± {bench.stdev:.4f}")

    def log(self, msg: str):
        self.log_lines.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def write_all(self) -> dict[str, Path]:
        """Writes every artifact. Returns map of artifact_name → file path."""
        out: dict[str, Path] = {}
        out["raw_json"] = self._write_raw_json()
        out["summary_csv"] = self._write_summary_csv()
        out["timing_csv"] = self._write_timing_csv()
        if self.chip_spec is not None:
            out["chip_csv"] = self._write_chip_csv()
            out["chip_2q_csv"] = self._write_chip_2q_csv()
        out["sci_csv"] = self._write_sci_hw_csv()
        try:
            out["stat_tests_csv"] = self._write_statistical_tests_csv()
        except Exception as e:
            self.log(f"stat-tests skipped: {e!r}")

        # Dissertation §3.15 Table 10 — backends descriptive
        out["backends_csv"] = self._write_backends_table()
        # Dissertation §3.17 Table 13 — algorithmic fidelity + BCa CI
        try:
            out["fidelity_bca_csv"] = self._write_algorithmic_fidelity_bca_csv()
        except Exception as e:
            self.log(f"fidelity-BCa skipped: {e!r}")
        # Dissertation §3.21 Table 17 — per-backend timing aggregated
        out["timing_per_backend_csv"] = self._write_timing_per_backend_csv()

        out["fidelity_chart"] = self._chart_fidelity_comparison()
        out["timeline_chart"] = self._chart_fidelity_timeline()
        out["timing_chart"] = self._chart_timing_breakdown()
        out["python_overhead_chart"] = self._chart_python_overhead()
        out["counts_chart"] = self._chart_counts_overlay()
        if self.chip_spec is not None:
            out["chip_chart"] = self._chart_chip_calibration()
        out["report_md"] = self._write_markdown_report(out)
        out["log"] = self._write_log()
        return out

    # ------------------------------------------------------------------ data

    def _write_raw_json(self) -> Path:
        chip_serialized = None
        if self.chip_spec is not None:
            d = asdict(self.chip_spec)
            d["coupling_map"] = [list(p) for p in d.get("coupling_map", [])]
            d["per_pair_f2q"] = {
                f"{a}-{b}": v for (a, b), v in d.get("per_pair_f2q", {}).items()
            }
            chip_serialized = d
        payload: dict[str, Any] = {
            "timestamp": self.timestamp,
            "chip": chip_serialized,
            "entries": [],
        }
        for exp, params, bench in self.entries:
            lo, hi = bench.ci_95
            payload["entries"].append({
                "experiment_key": exp.key,
                "experiment_title": exp.title,
                "params": params,
                "metric_name": exp.metric_name,
                "backend": bench.backend,
                "shots_per_run": bench.shots_per_run,
                "repeats": bench.repeats,
                "mean": bench.mean,
                "stdev": bench.stdev,
                "ci_95": [lo, hi],
                "lag1_autocorrelation": _lag1_autocorr(bench.fidelities),
                "fidelities": bench.fidelities,
                "per_run": [
                    {
                        "counts": r.counts,
                        "metric_value": r.metric_value,
                        "transpiled_depth": r.transpiled_depth,
                        "transpiled_ops": r.transpiled_ops,
                        "timing_s": r.timing.to_dict(),
                        "job_id": r.job_id,
                    }
                    for r in bench.runs
                ],
            })
        path = self.output_dir / "quantum_hardware_runs.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                         encoding="utf-8")
        return path

    def _write_statistical_tests_csv(self) -> Path:
        """
        For every experiment-config that has runs on ≥ 2 backends, run
        Kruskal-Wallis (global), Mann-Whitney pairwise + Bonferroni, and a
        BCa bootstrap CI for the metric mean per backend. Emits one CSV row
        per test.
        """
        from src.statistical_tests import full_report
        path = self.output_dir / "statistical_tests.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["experiment", "params", "test", "group_a", "group_b",
                        "statistic", "p_raw", "p_adjusted",
                        "ci_lo", "ci_hi", "n", "interpretation"])
            # Group entries by (exp.key, params)
            from collections import defaultdict
            grouped: dict[tuple, dict[str, list[float]]] = defaultdict(dict)
            for exp, params, bench in self.entries:
                key = (exp.key, json.dumps(params, ensure_ascii=False),
                       exp.metric_name)
                grouped[key][bench.backend] = list(bench.fidelities)
            for (exp_key, params_json, metric), backends in grouped.items():
                if len(backends) < 2:
                    continue
                rep = full_report(exp_key, metric, backends)
                if rep.kruskal:
                    w.writerow([exp_key, params_json,
                                "Kruskal-Wallis",
                                "all", "—",
                                f"{rep.kruskal.h_statistic:.4f}",
                                f"{rep.kruskal.p_value:.6f}", "—",
                                "—", "—",
                                rep.kruskal.n_total,
                                rep.kruskal.interpretation])
                for pw in rep.pairwise:
                    w.writerow([exp_key, params_json,
                                "Mann-Whitney+Bonferroni",
                                pw.group_a, pw.group_b,
                                f"{pw.u_statistic:.2f}",
                                f"{pw.p_raw:.6f}",
                                f"{pw.p_bonferroni:.6f}",
                                "—", "—", "—",
                                "significant" if pw.significant_at_05 else "ns"])
                for name, b in rep.bca_per_group.items():
                    w.writerow([exp_key, params_json,
                                "BCa bootstrap (mean)",
                                name, "—",
                                f"{b.point_estimate:.6f}",
                                "—", "—",
                                f"{b.ci_low:.6f}", f"{b.ci_high:.6f}",
                                b.n_bootstrap,
                                f"95% CI; bias={b.bias:.3f}, "
                                f"accel={b.acceleration:.3f}"])
        return path

    def _write_summary_csv(self) -> Path:
        path = self.output_dir / "quantum_hardware_summary.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["experiment", "params", "backend", "metric",
                        "n_repeats", "shots_per_run",
                        "mean", "stdev", "ci_lo_95", "ci_hi_95",
                        "lag1_autocorr"])
            for exp, params, bench in self.entries:
                lo, hi = bench.ci_95
                w.writerow([
                    exp.key,
                    json.dumps(params, ensure_ascii=False),
                    bench.backend,
                    exp.metric_name,
                    bench.repeats,
                    bench.shots_per_run,
                    f"{bench.mean:.6f}",
                    f"{bench.stdev:.6f}",
                    f"{lo:.6f}", f"{hi:.6f}",
                    f"{_lag1_autocorr(bench.fidelities):.4f}",
                ])
        return path

    def _write_timing_csv(self) -> Path:
        path = self.output_dir / "quantum_hardware_timing.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "experiment", "params", "backend", "n_repeats",
                # Phase-by-phase
                "mean_python_setup_s", "mean_transpile_s", "mean_submit_s",
                "mean_queue_s", "mean_execute_s", "mean_python_post_s",
                "mean_total_s", "std_total_s",
                # Aggregates: the two key columns for Python-overhead attribution
                "mean_python_overhead_s", "mean_algorithmic_s",
                "mean_python_overhead_pct",
            ])
            for exp, params, bench in self.entries:
                runs = bench.runs
                ps = [r.timing.python_setup_s for r in runs]
                tr = [r.timing.transpile_s for r in runs]
                sb = [r.timing.submit_s for r in runs]
                qu = [r.timing.queue_s for r in runs]
                ex = [r.timing.execute_s for r in runs]
                pp = [r.timing.python_post_s for r in runs]
                tot = [r.timing.total_s for r in runs]
                ov = [r.timing.python_overhead_s for r in runs]
                al = [r.timing.algorithmic_s for r in runs]
                pct = [r.timing.python_overhead_pct for r in runs]
                _, total_std, _, _ = _ci95(tot)
                w.writerow([
                    exp.key,
                    json.dumps(params, ensure_ascii=False),
                    bench.backend,
                    bench.repeats,
                    f"{statistics.mean(ps):.4f}" if ps else "",
                    f"{statistics.mean(tr):.4f}" if tr else "",
                    f"{statistics.mean(sb):.4f}" if sb else "",
                    f"{statistics.mean(qu):.4f}" if qu else "",
                    f"{statistics.mean(ex):.4f}" if ex else "",
                    f"{statistics.mean(pp):.4f}" if pp else "",
                    f"{statistics.mean(tot):.4f}" if tot else "",
                    f"{total_std:.4f}",
                    f"{statistics.mean(ov):.4f}" if ov else "",
                    f"{statistics.mean(al):.4f}" if al else "",
                    f"{statistics.mean(pct):.2f}" if pct else "",
                ])
        return path

    def _write_chip_csv(self) -> Path:
        spec = self.chip_spec
        path = self.output_dir / "quantum_hardware_chip.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["qubit", "frequency_GHz",
                        "F_1q", "F_RO", "T1_us", "T2_us"])
            for q in range(spec.num_qubits):
                w.writerow([
                    q,
                    f"{spec.per_qubit_freq_ghz.get(q, 0):.4f}",
                    f"{spec.per_qubit_f1q.get(q, 0):.4f}",
                    f"{spec.per_qubit_ro.get(q, 0):.4f}",
                    f"{spec.per_qubit_t1_us.get(q, 0):.2f}",
                    f"{spec.per_qubit_t2_us.get(q, 0):.2f}",
                ])
        return path

    def _write_chip_2q_csv(self) -> Path:
        spec = self.chip_spec
        path = self.output_dir / "quantum_hardware_chip_2q.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["qubit_a", "qubit_b", "F_cz"])
            for (a, b), fid in spec.per_pair_f2q.items():
                w.writerow([a, b, f"{fid:.4f}"])
        return path

    def _compute_sci_hw_table(self) -> list[SCI_HW]:
        """Pair every (experiment, non-ideal backend) with its ideal counterpart."""
        ideal_index: dict[str, BenchmarkResult] = {}
        params_index: dict[str, dict] = {}
        exp_index: dict[str, ExperimentDef] = {}

        for exp, params, bench in self.entries:
            key = f"{exp.key}::{json.dumps(params, sort_keys=True)}"
            if bench.backend == "ideal":
                ideal_index[key] = bench
                params_index[key] = params
                exp_index[key] = exp

        sci_rows: list[SCI_HW] = []
        for exp, params, bench in self.entries:
            if bench.backend == "ideal":
                continue
            key = f"{exp.key}::{json.dumps(params, sort_keys=True)}"
            ideal = ideal_index.get(key)
            if ideal is None or not ideal.runs:
                continue

            ideal_run = ideal.runs[0]
            obs_run = bench.runs[0] if bench.runs else None
            if obs_run is None:
                continue

            sci = compute_sci_hw(
                experiment=exp.title_with_params(params),
                backend=bench.backend,
                metric_observed=bench.mean,
                metric_ideal=ideal.mean,
                time_observed_s=statistics.mean(
                    [r.timing.total_s for r in bench.runs] or [0]
                ),
                time_ideal_s=statistics.mean(
                    [r.timing.total_s for r in ideal.runs] or [0]
                ),
                transpiled_depth=obs_run.transpiled_depth,
                logical_depth=ideal_run.transpiled_depth,
            )
            sci_rows.append(sci)
        return sci_rows

    def _write_sci_hw_csv(self) -> Path:
        path = self.output_dir / "quantum_sci_hw.csv"
        rows = self._compute_sci_hw_table()
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["experiment", "backend", "time_factor",
                        "error_factor", "routing_factor",
                        "SCI_HW", "interpretation"])
            for r in rows:
                w.writerow([
                    r.experiment, r.backend,
                    f"{r.time_factor:.4f}",
                    f"{r.error_factor:.4f}",
                    f"{r.routing_factor:.4f}",
                    f"{r.sci_value:.4f}",
                    r.interpretation,
                ])
        return path

    # ------------------------------------------------------------------
    # Dissertation tables 10, 13, 17
    # ------------------------------------------------------------------

    _BACKEND_DESCRIPTIONS: dict[str, tuple[str, str, str]] = {
        # backend → (Type, Calibration source, Role)
        "ideal":           ("State-vector, CPU",
                            "—",
                            "Reference; noise-free"),
        "gpu":             ("State-vector, CUDA (CuPy)",
                            "—",
                            "Production GPU acceleration of ideal"),
        "emulator_4q_v1":  ("Density-matrix",
                            "Snowdrop 4q v1 saved calibration JSON",
                            "Noisy emulator; first chip snapshot"),
        "emulator_4q_v2":  ("Density-matrix",
                            "Snowdrop 4q v2 saved calibration JSON",
                            "Noisy emulator; current chip snapshot"),
        "emulator_8q_v1":  ("Density-matrix",
                            "Snowdrop 8q v1 projection JSON",
                            "Noisy emulator; extended topology"),
        "real":            ("Superconducting QPU (Bauman Octillion)",
                            "Live Snowdrop 4q v2 (Bauman Octillion API)",
                            "Production real hardware"),
    }

    def _write_backends_table(self) -> Path:
        """Table 10 — backends descriptive."""
        used = sorted({bench.backend for _, _, bench in self.entries})
        path = self.output_dir / "quantum_hardware_backends.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Backend", "Type", "Calibration_Source", "Role"])
            for be in used:
                t, cal, role = self._BACKEND_DESCRIPTIONS.get(
                    be, ("—", "—", "—"))
                w.writerow([be, t, cal, role])
        return path

    def _write_algorithmic_fidelity_bca_csv(self) -> Path:
        """
        Table 13 — algorithmic fidelity with mean + BCa-bootstrap 95% CI
        per (experiment, backend) cell. Format matches dissertation §3.17.
        """
        from src.statistical_tests import bca_bootstrap
        path = self.output_dir / "algorithmic_fidelity_bca.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Experiment", "Backend", "Metric", "Mean",
                        "CI95_low", "CI95_high", "N_repeats",
                        "Shots_per_repeat", "Lag1_autocorr"])
            for exp, params, bench in self.entries:
                samples = list(bench.fidelities)
                if not samples:
                    continue
                if len(samples) >= 2:
                    try:
                        bca = bca_bootstrap(samples, n_bootstrap=10_000)
                        ci_lo, ci_hi = bca.ci_low, bca.ci_high
                    except Exception:
                        ci_lo, ci_hi = bench.ci_95
                else:
                    ci_lo = ci_hi = samples[0]
                label = exp.title_with_params(params)
                w.writerow([
                    label,
                    bench.backend,
                    exp.metric_name,
                    f"{bench.mean:.4f}",
                    f"{ci_lo:.4f}",
                    f"{ci_hi:.4f}",
                    bench.repeats,
                    bench.shots_per_run,
                    f"{_lag1_autocorr(bench.fidelities):.4f}",
                ])
        return path

    def _write_timing_per_backend_csv(self) -> Path:
        """
        Table 17 — wall-clock timing aggregated per backend across all
        algorithmic experiments. Columns: Backend | Mean_total_s |
        Mean_execute_s | Python_overhead_pct | Total_relative_to_ideal.
        """
        # Aggregate per backend
        per_backend: dict[str, list[BenchmarkResult]] = {}
        for _exp, _params, bench in self.entries:
            per_backend.setdefault(bench.backend, []).append(bench)

        # Mean total wall-clock per backend
        backend_total: dict[str, float] = {}
        backend_exec:  dict[str, float] = {}
        backend_pyo:   dict[str, float] = {}
        for be, benches in per_backend.items():
            totals: list[float] = []
            execs: list[float] = []
            pyos: list[float] = []
            for bench in benches:
                for run in bench.runs:
                    t = run.timing
                    total_s = (t.python_setup_s + t.transpile_s + t.submit_s
                                + t.queue_s + t.execute_s + t.python_post_s)
                    if total_s > 0:
                        totals.append(total_s)
                        execs.append(t.execute_s)
                        py_over = total_s - t.execute_s
                        pyos.append(100.0 * py_over / total_s)
            if totals:
                backend_total[be] = sum(totals) / len(totals)
                backend_exec[be]  = sum(execs)  / len(execs)
                backend_pyo[be]   = sum(pyos)   / len(pyos)

        ideal_total = backend_total.get("ideal", 0) or 0.0

        path = self.output_dir / "timing_per_backend.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Backend", "Mean_total_s", "Mean_execute_s",
                        "Python_overhead_pct", "Total_relative_to_ideal",
                        "N_runs"])
            for be in sorted(backend_total, key=lambda x: backend_total[x]):
                rel = (backend_total[be] / ideal_total) if ideal_total else 1.0
                n_runs = sum(len(b.runs) for b in per_backend[be])
                w.writerow([
                    be,
                    f"{backend_total[be]:.4f}",
                    f"{backend_exec[be]:.4f}",
                    f"{backend_pyo[be]:.2f}",
                    f"{rel:.2f}",
                    n_runs,
                ])
        return path

    # ------------------------------------------------------------------ charts

    def _chart_fidelity_comparison(self) -> Path:
        """Main thesis figure: bar chart with 95% CI error bars per (exp, backend)."""
        path = self.output_dir / "fidelity_comparison.png"

        groups: dict[str, dict[str, BenchmarkResult]] = {}
        order: list[tuple[ExperimentDef, dict]] = []
        for exp, params, bench in self.entries:
            key = exp.title_with_params(params)
            if key not in groups:
                groups[key] = {}
                order.append((exp, params))
            groups[key][bench.backend] = bench

        # Only render backends that actually appear in `self.entries` so the
        # chart never has gray empty bars for backends the user didn't pick.
        present_backends = {b for g in groups.values() for b in g}
        canonical_order = ["ideal", "gpu", "emulator_4q_v2", "emulator_4q_v1",
                            "emulator_8q_v1", "emulator", "real"]
        backends = [b for b in canonical_order if b in present_backends]
        colors = {
            "ideal":         "#9bc4e2",
            "gpu":           "#76b900",
            "emulator":      "#f0a847",
            "emulator_4q_v2": "#f0a847",
            "emulator_4q_v1": "#e09a3a",
            "emulator_8q_v1": "#c97f1f",
            "real":          "#0a7a3a",
        }
        labels = list(groups.keys())
        n = len(labels)
        x = np.arange(n)
        # Bar width sized to fit however many backends were used
        width = 0.8 / max(len(backends), 1)

        fig, ax = plt.subplots(figsize=(max(8, n * 1.4), 5))

        for i, b in enumerate(backends):
            heights = []
            errs = []
            for label in labels:
                bench = groups[label].get(b)
                if bench is None:
                    heights.append(0)
                    errs.append(0)
                    continue
                lo, hi = bench.ci_95
                heights.append(bench.mean)
                errs.append((hi - lo) / 2 if hi > lo else 0)
            offset = (i - (len(backends) - 1) / 2) * width
            bars = ax.bar(x + offset, heights, width,
                          yerr=errs, capsize=4,
                          label=b, color=colors.get(b, "#888"),
                          edgecolor="black", linewidth=0.4)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylim(0, 1.1)
        ax.axhline(0.5, color="#888", linestyle=":", linewidth=0.8,
                   label="random baseline")
        ax.set_ylabel("Empirical fidelity / metric")
        ax.set_title(f"Quantum-hardware experiment fidelities "
                     f"(95% CI error bars)\n{CHIP_NAME}, "
                     f"recorded {self.timestamp}")
        ax.legend(loc="lower left")
        ax.grid(True, axis="y", alpha=0.3)

        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return path

    def _chart_fidelity_timeline(self) -> Path:
        path = self.output_dir / "fidelity_timeline.png"
        if not self.entries:
            fig = plt.figure()
            fig.savefig(path)
            plt.close(fig)
            return path

        canonical_order = ["ideal", "gpu",
                            "emulator_4q_v2", "emulator_4q_v1", "emulator_8q_v1",
                            "emulator", "real"]
        present = {b.backend for _e, _p, b in self.entries if b.runs}
        backends_order = [b for b in canonical_order if b in present]
        if not backends_order:
            fig = plt.figure()
            fig.savefig(path)
            plt.close(fig)
            return path

        fig, axes = plt.subplots(1, len(backends_order),
                                  figsize=(5 * len(backends_order), 4.2),
                                  sharey=True, squeeze=False)
        axes = axes[0]
        for ax, backend in zip(axes, backends_order):
            for exp, params, bench in self.entries:
                if bench.backend != backend or not bench.runs:
                    continue
                xs = list(range(1, bench.repeats + 1))
                ax.plot(xs, bench.fidelities, marker="o",
                        label=exp.title_with_params(params), linewidth=1.0)
            ax.set_title(f"backend = {backend}")
            ax.set_xlabel("Run #")
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)
            ax.axhline(0.5, color="#888", linestyle=":", linewidth=0.8)
            if any(p.get_label() and not p.get_label().startswith("_")
                   for p in ax.lines):
                ax.legend(loc="lower right", fontsize=7)
        axes[0].set_ylabel("Metric value")
        fig.suptitle(f"Per-run fidelity trajectories ({CHIP_NAME})",
                     y=1.02, fontweight="bold")
        fig.tight_layout()
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return path

    def _chart_timing_breakdown(self) -> Path:
        """
        Stacked-bar wall-clock breakdown across ALL backends present in the
        run set (not just real). If no real-hw data is available, the chart
        still renders for ideal/gpu/emulator with the correct title.
        """
        path = self.output_dir / "timing_breakdown.png"

        labels: list[str] = []
        transpile_v: list[float] = []
        submit_v: list[float] = []
        queue_v: list[float] = []
        execute_v: list[float] = []

        for exp, params, bench in self.entries:
            if not bench.runs:
                continue
            tr = [r.timing.transpile_s for r in bench.runs]
            sb = [r.timing.submit_s for r in bench.runs]
            qu = [r.timing.queue_s for r in bench.runs]
            ex = [r.timing.execute_s for r in bench.runs]
            labels.append(f"{exp.title_with_params(params)}\n[{bench.backend}]")
            transpile_v.append(statistics.mean(tr) if tr else 0)
            submit_v.append(statistics.mean(sb) if sb else 0)
            queue_v.append(statistics.mean(qu) if qu else 0)
            execute_v.append(statistics.mean(ex) if ex else 0)

        if not labels:
            fig = plt.figure()
            fig.savefig(path)
            plt.close(fig)
            return path

        fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.0), 5.0))
        x = np.arange(len(labels))
        ax.bar(x, transpile_v, label="transpile (Qiskit Rust+Py)", color="#9bc4e2",
               edgecolor="black", linewidth=0.4)
        ax.bar(x, submit_v, bottom=transpile_v,
               label="submit (network)", color="#f0a847",
               edgecolor="black", linewidth=0.4)
        bottom2 = np.array(transpile_v) + np.array(submit_v)
        ax.bar(x, queue_v, bottom=bottom2,
               label="queue (Bauman scheduler)", color="#c0a8e8",
               edgecolor="black", linewidth=0.4)
        bottom3 = bottom2 + np.array(queue_v)
        ax.bar(x, execute_v, bottom=bottom3,
               label="execute (chip / Aer / GPU)", color="#0a7a3a",
               edgecolor="black", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Seconds (mean per run)")
        ax.set_title(f"Wall-clock timing breakdown — all backends ({CHIP_NAME})",
                     fontweight="bold")
        ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=200, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        plt.close(fig)
        return path

    def _chart_python_overhead(self) -> Path:
        """
        Stacked-bar view of where wall-clock time is spent per (exp, backend),
        with Python orchestration phases visually separated from the
        algorithmic / hardware phases. The legend names which colors are
        Python-overhead (subtractable on a Cython/C wrapper) vs
        algorithmic (irreducible cost of the algorithm + hardware).
        """
        path = self.output_dir / "python_overhead.png"

        rows = []
        for exp, params, bench in self.entries:
            label = f"{exp.title_with_params(params)}\n({bench.backend})"
            rows.append({
                "label": label,
                "py_setup":  bench.avg_setup_s * 1000,
                "transpile": bench.avg_transpile_s * 1000,
                "submit":    statistics.mean(
                    [r.timing.submit_s for r in bench.runs]) * 1000 if bench.runs else 0,
                "queue":     bench.avg_queue_s * 1000,
                "execute":   bench.avg_execute_s * 1000,
                "py_post":   bench.avg_post_s * 1000,
                "overhead_pct": bench.avg_python_overhead_pct,
                "total":     bench.avg_total_s * 1000,
                "algo":      bench.avg_algorithmic_s * 1000,
            })

        if not rows:
            fig = plt.figure()
            fig.savefig(path)
            plt.close(fig)
            return path

        # Group rows by backend so we can put each backend in its own subplot
        # (avoids the "75 squashed bars in one row" mess).
        rows_by_backend: dict[str, list[dict]] = {}
        for r, (_e, _p, bench) in zip(rows, self.entries):
            rows_by_backend.setdefault(bench.backend, []).append(r)

        canonical_order = ["ideal", "gpu",
                            "emulator_4q_v2", "emulator_4q_v1",
                            "emulator_8q_v1", "emulator", "real"]
        backends = [b for b in canonical_order if b in rows_by_backend]
        backends += [b for b in rows_by_backend if b not in backends]
        n_back = len(backends)

        fig, axes = plt.subplots(n_back, 1,
                                  figsize=(max(11, len(rows) // n_back * 1.0 + 3),
                                            3.0 * n_back),
                                  squeeze=False)
        axes = axes[:, 0]

        for ax, backend in zip(axes, backends):
            sub = rows_by_backend[backend]
            n = len(sub)
            x = np.arange(n)
            labels = [r["label"].split("\n")[0] for r in sub]

            py_setup  = np.array([r["py_setup"] for r in sub])
            transpile = np.array([r["transpile"] for r in sub])
            submit    = np.array([r["submit"] for r in sub])
            queue     = np.array([r["queue"] for r in sub])
            execute   = np.array([r["execute"] for r in sub])
            py_post   = np.array([r["py_post"] for r in sub])

            bottom = np.zeros_like(py_setup)
            ax.bar(x, py_setup, bottom=bottom, color="#E74C3C",
                   label="Python setup (overhead)",
                   edgecolor="black", linewidth=0.4)
            bottom += py_setup
            ax.bar(x, transpile, bottom=bottom, color="#9B59B6",
                   label="Transpile (Qiskit Rust+Py)",
                   edgecolor="black", linewidth=0.4)
            bottom += transpile
            ax.bar(x, submit, bottom=bottom, color="#3498DB",
                   label="Submit (network)",
                   edgecolor="black", linewidth=0.4)
            bottom += submit
            ax.bar(x, queue, bottom=bottom, color="#7F8C8D",
                   label="Queue (Bauman scheduler)",
                   edgecolor="black", linewidth=0.4)
            bottom += queue
            ax.bar(x, execute, bottom=bottom, color="#27AE60",
                   label="Execute (chip / Aer / GPU)",
                   edgecolor="black", linewidth=0.4)
            bottom += execute
            ax.bar(x, py_post, bottom=bottom, color="#C0392B",
                   label="Python post (overhead)",
                   edgecolor="black", linewidth=0.4)
            bottom += py_post

            for i, r in enumerate(sub):
                top = r["total"]
                pct = r["overhead_pct"]
                ax.annotate(f"{pct:.1f}%",
                              xy=(i, top), xytext=(0, 4),
                              textcoords="offset points",
                              ha="center", va="bottom",
                              fontsize=7, fontweight="bold",
                              color=("#C0392B" if pct > 5 else "#27AE60"))

            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=7)
            ax.set_ylabel(f"{backend}\n(ms / run)", fontsize=9)
            ymax = bottom.max() if len(bottom) else 1
            ax.set_ylim(0, max(ymax * 1.18, 1))
            ax.grid(True, axis="y", alpha=0.3)
            ax.set_axisbelow(True)

        axes[0].set_title(
            "Where the benchmark's wall-clock time is spent (per backend)\n"
            "Python orchestration (red) is subtractable; Network/Compute "
            "(blue/green) is algorithmic.  Bar labels show η_py %.",
            fontweight="bold"
        )
        axes[0].legend(loc="upper right", framealpha=0.9, fontsize=8, ncol=2)

        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight",
                     facecolor="white", edgecolor="none")
        plt.close(fig)
        return path

    def _chart_counts_overlay(self) -> Path:
        path = self.output_dir / "counts_overlay.png"

        seen: dict[str, tuple[ExperimentDef, dict, dict[str, BenchmarkResult]]] = {}
        for exp, params, bench in self.entries:
            key = exp.title_with_params(params)
            if key not in seen:
                seen[key] = (exp, params, {})
            seen[key][2][bench.backend] = bench

        items = list(seen.values())
        n = len(items)
        if n == 0:
            fig = plt.figure()
            fig.savefig(path)
            plt.close(fig)
            return path

        cols = min(2, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 3.6 * rows))
        if rows * cols == 1:
            axes = np.array([[axes]])
        elif rows == 1 or cols == 1:
            axes = np.array(axes).reshape(rows, cols)

        for idx, (exp, params, by_backend) in enumerate(items):
            ax = axes[idx // cols, idx % cols]
            expected = exp.expected(params)
            keys = sorted(set(expected.keys()) |
                          set(k for b in by_backend.values()
                              for k in b.runs[0].counts.keys()
                              if b.runs))
            x = np.arange(len(keys))

            ideal_vals = [expected.get(k, 0.0) for k in keys]
            ax.bar(x, ideal_vals, width=0.7, color="#9bc4e2",
                   alpha=0.4, label="expected (theory)")

            real_bench = by_backend.get("real")
            if real_bench:
                p_avg = {k: 0.0 for k in keys}
                for r in real_bench.runs:
                    total = sum(r.counts.values()) or 1
                    for k in keys:
                        p_avg[k] += r.counts.get(k, 0) / total
                for k in keys:
                    p_avg[k] /= max(len(real_bench.runs), 1)
                ax.bar(x, [p_avg[k] for k in keys], width=0.4,
                       color="#0a7a3a", label=f"real (mean N={real_bench.repeats})")

            ax.set_xticks(x)
            ax.set_xticklabels(keys, rotation=0, fontsize=8)
            ax.set_ylim(0, 1.1)
            ax.set_title(exp.title_with_params(params), fontsize=10)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, axis="y", alpha=0.3)

        for idx in range(n, rows * cols):
            axes[idx // cols, idx % cols].axis("off")

        fig.suptitle(f"Observed vs expected outcome distributions ({CHIP_NAME})",
                     y=1.02, fontweight="bold")
        fig.tight_layout()
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return path

    def _chart_chip_calibration(self) -> Path:
        path = self.output_dir / "chip_calibration.png"
        spec = self.chip_spec
        positions = {
            0: (1, 1), 1: (0, 0), 2: (1, 0), 3: (2, 0),
        }
        if spec.num_qubits != 4:
            positions = {q: (q, 0) for q in range(spec.num_qubits)}

        fig, ax = plt.subplots(figsize=(7, 5))

        for (a, b) in spec.per_pair_f2q:
            xa, ya = positions.get(a, (a, 0))
            xb, yb = positions.get(b, (b, 0))
            f = spec.per_pair_f2q[(a, b)]
            ax.plot([xa, xb], [ya, yb], color="#444", linewidth=2, alpha=0.6)
            mx, my = (xa + xb) / 2, (ya + yb) / 2
            ax.text(mx, my + 0.06, f"F_cz={f*100:.2f}%", ha="center",
                    fontsize=8, color="#222")

        for q in range(spec.num_qubits):
            x, y = positions.get(q, (q, 0))
            t1 = spec.per_qubit_t1_us.get(q, 0)
            t2 = spec.per_qubit_t2_us.get(q, 0)
            f1q = spec.per_qubit_f1q.get(q, 0) * 100
            ro = spec.per_qubit_ro.get(q, 0) * 100
            fq = spec.per_qubit_freq_ghz.get(q, 0)
            ax.scatter([x], [y], s=2200, c="#0a7a3a", edgecolors="black",
                       linewidths=1.5, zorder=3)
            ax.text(x, y, f"q{q}", ha="center", va="center",
                    color="white", fontweight="bold", fontsize=14, zorder=4)
            ax.text(x, y - 0.18,
                    f"f={fq:.2f}GHz\nT1={t1:.1f}us  T2={t2:.1f}us\n"
                    f"F1q={f1q:.2f}%  RO={ro:.1f}%",
                    ha="center", va="top", fontsize=8)

        ax.set_xlim(-0.7, 2.7)
        ax.set_ylim(-0.7, 1.5)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{CHIP_NAME} — live calibration ({self.timestamp})",
                     fontweight="bold")
        fig.tight_layout()
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return path

    # ------------------------------------------------------------------ md

    def _write_markdown_report(self, artifact_paths: dict[str, Path]) -> Path:
        path = self.output_dir / "report.md"
        sci_rows = self._compute_sci_hw_table()

        lines: list[str] = []
        lines.append(f"# Quantum Hardware Benchmark — {CHIP_NAME}\n")
        lines.append(f"Generated: **{self.timestamp}**\n")

        if self.chip_spec is not None:
            spec = self.chip_spec
            lines.append("## Chip calibration snapshot\n")
            lines.append(f"- Qubits: **{spec.num_qubits}**")
            lines.append(f"- Basis gates: `{', '.join(spec.basis_gates)}`")
            lines.append(f"- Coupling map: `{spec.coupling_map}`")
            lines.append(f"- Avg F_1q = **{spec.avg_f1q*100:.2f}%**, "
                         f"F_2q (CZ) = **{spec.avg_f2q*100:.2f}%**, "
                         f"F_RO = **{spec.avg_ro*100:.2f}%**")
            lines.append(f"- Avg T1 = **{spec.avg_t1_us:.2f} us**, "
                         f"T2 = **{spec.avg_t2_us:.2f} us**")
            lines.append(f"\n![Chip calibration](chip_calibration.png)\n")

        lines.append("## Experiment results (mean ± stdev, 95% CI)\n")
        lines.append("| Experiment | Backend | N | mean | stdev | 95% CI | "
                     "lag-1 autocorr |")
        lines.append("|---|---|---:|---:|---:|---|---:|")
        for exp, params, bench in self.entries:
            lo, hi = bench.ci_95
            lag1 = _lag1_autocorr(bench.fidelities)
            lines.append(
                f"| {exp.title_with_params(params)} "
                f"| {bench.backend} "
                f"| {bench.repeats} "
                f"| {bench.mean:.4f} "
                f"| {bench.stdev:.4f} "
                f"| [{lo:.4f}, {hi:.4f}] "
                f"| {lag1:+.3f} |"
            )

        lines.append(f"\n![Fidelity comparison](fidelity_comparison.png)\n")
        lines.append(f"![Per-run trajectories](fidelity_timeline.png)\n")
        lines.append(f"![Outcome distributions](counts_overlay.png)\n")

        if sci_rows:
            lines.append("## SCI_HW — hardware-aware Security Cost Index\n")
            lines.append(
                "Extension of the published SCI (DOI 10.53364/24138614_2026_40_1_11) "
                "with measured hardware quantities:\n\n"
                "**SCI_HW = (T_obs / T_ideal) × |M_obs − M_ideal| × "
                "(D_transpiled / D_logical)**\n\n"
                "where M_ideal is the metric value measured on the ideal simulator "
                "(the achievable baseline — for Shor it is ≈0.5, for Bell it is ≈1.0). "
                "Lower is better; SCI_HW = 0 iff the backend matches the noiseless "
                "reference exactly.\n"
            )
            lines.append("| Experiment | Backend | Time factor | Error factor "
                         "| Routing factor | SCI_HW | Interpretation |")
            lines.append("|---|---|---:|---:|---:|---:|---|")
            for r in sci_rows:
                lines.append(
                    f"| {r.experiment} | {r.backend} "
                    f"| ×{r.time_factor:.2f} "
                    f"| {r.error_factor:.4f} "
                    f"| ×{r.routing_factor:.2f} "
                    f"| **{r.sci_value:.4f}** "
                    f"| {r.interpretation} |"
                )

        lines.append("\n## Wall-clock timing (real hardware)\n")
        lines.append(f"![Timing breakdown](timing_breakdown.png)\n")

        lines.append("\n## Python overhead attribution (methodology)\n")
        lines.append(
            "Each `run_circuit` call is instrumented with phase timers that "
            "split wall-clock time into six categories:\n\n"
            "1. **`python_setup_s`** — Python orchestration: client init, "
            "`get_backend()`, `describe_chip()`, dataclass building. "
            "*Subtractable on a Cython/C wrapper.*\n"
            "2. **`transpile_s`** — `qiskit.transpile()`. Qiskit 2.x has a "
            "Rust core for the heavy passes (sabre routing, gate optimization), "
            "so this is **mostly C-level work** with a thin Python driver. "
            "Counted as algorithmic.\n"
            "3. **`submit_s`** — HTTP submit to Octillion API. Pure network. "
            "Counted as algorithmic.\n"
            "4. **`queue_s`** — Bauman backend queue wait, measured between "
            "`status==QUEUE` and `status==EXECUTING`. Pure environment. "
            "Counted as algorithmic.\n"
            "5. **`execute_s`** — actual chip / Aer / GPU compute time, "
            "measured between `EXECUTING` and `COMPLETE`. Counted as algorithmic.\n"
            "6. **`python_post_s`** — counts normalization, metric "
            "computation, formatting. *Subtractable on a Cython/C wrapper.*\n"
            "\n"
            "**Definition:** `python_overhead = python_setup_s + python_post_s`. "
            "**Definition:** `algorithmic = transpile_s + submit_s + queue_s + "
            "execute_s` (everything outside the two pure-Python phases).\n"
            "\n"
            "**Why this attribution is justified:** the two Python phases "
            "(`setup`, `post`) wrap calls that are themselves implemented "
            "in fast cores; their wall time is dominated by Python interpreter "
            "overhead — attribute lookups, dict/object creation, format strings, "
            "argument marshalling. Replacing the wrapper with Cython or a "
            "compiled language eliminates this cost without changing the "
            "underlying numerical behavior. The four algorithmic phases are "
            "not Python-bound: they are dominated by Rust core (transpile), "
            "TCP latency (submit), Bauman scheduler (queue), or C++ / CUDA "
            "kernels (execute).\n"
            "\n"
            "**See:** `python_overhead.png` for per-experiment stacked bars; "
            "`quantum_hardware_timing.csv` columns `mean_python_overhead_s`, "
            "`mean_algorithmic_s`, `mean_python_overhead_pct`.\n"
        )
        lines.append(f"\n![Python overhead attribution](python_overhead.png)\n")

        lines.append("## Reproducibility notes\n")
        lines.append(
            "- Every individual run's raw counts and timing are preserved in "
            "`quantum_hardware_runs.json` and in `Bauman/runs/<job_id>.json`.\n"
            "- Lag-1 autocorrelations of fidelity time-series are reported above. "
            "Values close to zero confirm the runs are statistically independent, "
            "validating the use of standard 95% CI under iid assumption.\n"
            "- Transpilation uses `layout_method='trivial'` to keep "
            "logical→physical qubit mapping identity (avoids "
            "`qiskit.qasm2.dumps()` losing layout metadata when sent to real chip).\n"
            "- `optimization_level=1` and `routing_method='sabre'` for SWAP insertion.\n"
        )

        lines.append("\n## Data files\n")
        for name, p in artifact_paths.items():
            lines.append(f"- `{p.name}` — {name.replace('_', ' ')}")

        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def _write_log(self) -> Path:
        path = self.output_dir / "benchmark.log"
        path.write_text("\n".join(self.log_lines) + "\n", encoding="utf-8")
        return path

