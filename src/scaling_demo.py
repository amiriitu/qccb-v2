"""
GHZ-N scaling demo: ideal CPU sim vs GPU sim vs Snowdrop chip,
for N from 2 up to GPU's memory limit (~25 qubits on 8 GB VRAM).

Why this is the key thesis figure:
  Snowdrop chip = 4 qubits (hard ceiling).
  Ideal sim: CPU is fine up to ~22-24 qubits then RAM bottlenecks.
  GPU sim:   stays fast and memory-efficient up to ~28 qubits on 8 GB VRAM.

The chart this generates is the answer to: "Where is today's NISQ-vs-classical
boundary on consumer hardware?" — direct evidence for chapters that argue PQC
needs to be deployed BEFORE quantum hardware reaches breaking depth.
"""
from __future__ import annotations

import csv
import json
import sys
from contextlib import suppress
import time
from pathlib import Path
from typing import Callable

if sys.platform == "win32":
    with suppress(AttributeError, OSError):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _build_ghz(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n, n, name=f"ghz_{n}")
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    qc.measure(range(n), range(n))
    return qc


def _bench_cpu(n: int, shots: int) -> dict:
    sim = AerSimulator()
    qc = _build_ghz(n)
    qc_t = sim.run(qc, shots=1).result()  # warmup
    t0 = time.perf_counter()
    res = sim.run(qc, shots=shots).result()
    dt = time.perf_counter() - t0
    counts = res.get_counts()
    fid = (counts.get("0" * n, 0) + counts.get("1" * n, 0)) / shots
    return {"n": n, "backend": "cpu_aer", "wall_s": dt, "fidelity": fid,
            "ok": True, "error": ""}


def _bench_gpu(n: int, shots: int) -> dict:
    from src.gpu_simulator import run_circuit_gpu
    qc = _build_ghz(n)
    try:
        stats = run_circuit_gpu(qc, shots=shots, warmup=True)
        fid = (stats.counts.get("0" * n, 0) + stats.counts.get("1" * n, 0)) / shots
        return {"n": n, "backend": "gpu_cupy", "wall_s": stats.total_s,
                "gate_apply_s": stats.gate_apply_s,
                "vram_mb": stats.vram_used_mb,
                "fidelity": fid, "ok": True, "error": ""}
    except Exception as e:
        return {"n": n, "backend": "gpu_cupy", "wall_s": 0.0,
                "fidelity": 0.0, "ok": False, "error": repr(e)}


def run_scaling(
    n_range: range,
    shots: int = 1024,
    do_cpu: bool = True,
    do_gpu: bool = True,
    output_dir: Path | None = None,
    snowdrop_n_max: int = 4,
    cpu_skip_n_above: int = 22,
    status: Callable[[str], None] | None = None,
) -> dict[str, Path]:
    """Run GHZ-N scaling. Returns dict of artifact paths."""
    status = status or (lambda m: None)
    if output_dir is None:
        stamp = time.strftime("%d%m%Y %H %M")
        output_dir = (PROJECT_ROOT / "results" / "scaling_demo" / stamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for n in n_range:
        if do_cpu and n <= cpu_skip_n_above:
            status(f"[cpu_aer]  GHZ-{n}: starting...")
            r = _bench_cpu(n, shots)
            rows.append(r)
            status(f"[cpu_aer]  GHZ-{n}: {r['wall_s']:.3f}s, F={r['fidelity']:.3f}")
        elif do_cpu:
            status(f"[cpu_aer]  GHZ-{n}: skipped (n > {cpu_skip_n_above})")

        if do_gpu:
            status(f"[gpu_cupy] GHZ-{n}: starting...")
            r = _bench_gpu(n, shots)
            rows.append(r)
            if r["ok"]:
                status(f"[gpu_cupy] GHZ-{n}: {r['wall_s']:.3f}s, "
                       f"VRAM {r['vram_mb']:.0f}MB, F={r['fidelity']:.3f}")
            else:
                status(f"[gpu_cupy] GHZ-{n}: FAILED ({r['error']})")
                if "out of memory" in r["error"].lower():
                    status("[gpu_cupy] VRAM exhausted, stopping GPU loop")
                    break

    csv_path = output_dir / "ghz_scaling.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n_qubits", "backend", "wall_s", "fidelity",
                    "vram_mb", "ok", "error"])
        for r in rows:
            w.writerow([r["n"], r["backend"], f"{r['wall_s']:.4f}",
                        f"{r['fidelity']:.4f}",
                        f"{r.get('vram_mb', 0):.0f}", r["ok"], r["error"]])

    json_path = output_dir / "ghz_scaling.json"
    json_path.write_text(json.dumps({"shots": shots, "rows": rows},
                                       indent=2), encoding="utf-8")

    chart_path = output_dir / "ghz_scaling.png"
    cpu_xs = [r["n"] for r in rows if r["backend"] == "cpu_aer" and r["ok"]]
    cpu_ys = [r["wall_s"] for r in rows if r["backend"] == "cpu_aer" and r["ok"]]
    gpu_xs = [r["n"] for r in rows if r["backend"] == "gpu_cupy" and r["ok"]]
    gpu_ys = [r["wall_s"] for r in rows if r["backend"] == "gpu_cupy" and r["ok"]]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    if cpu_xs:
        ax.plot(cpu_xs, cpu_ys, marker="o", color="#9bc4e2",
                label="CPU (Qiskit Aer)", linewidth=2)
    if gpu_xs:
        ax.plot(gpu_xs, gpu_ys, marker="s", color="#0a7a3a",
                label="GPU (CuPy state-vector, RTX 4060 Laptop)", linewidth=2)
    ax.axvline(snowdrop_n_max, color="#c00", linestyle="--", linewidth=1.5,
               label=f"Snowdrop 4q ver2 ceiling (N={snowdrop_n_max})")
    ax.set_xlabel("Number of qubits N (GHZ-N circuit)")
    ax.set_ylabel("Wall-clock time (seconds)")
    ax.set_yscale("log")
    ax.set_title(f"GHZ-N simulation scaling: where the NISQ ↔ classical "
                 f"boundary lives ({shots} shots/circuit)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left")

    if gpu_xs and rows:
        gpu_max = max([r for r in rows if r["backend"] == "gpu_cupy" and r["ok"]],
                       key=lambda r: r["n"])
        ax.annotate(f"GPU at N={gpu_max['n']}: VRAM {gpu_max.get('vram_mb', 0):.0f} MB",
                     xy=(gpu_max["n"], gpu_max["wall_s"]),
                     xytext=(gpu_max["n"] - 4, gpu_max["wall_s"] * 5),
                     arrowprops=dict(arrowstyle="->", color="#666"),
                     fontsize=9, color="#222")

    fig.tight_layout()
    fig.savefig(chart_path, dpi=200)
    plt.close(fig)

    status(f"\nArtifacts: {csv_path}, {json_path}, {chart_path}")
    return {"csv": csv_path, "json": json_path, "chart": chart_path}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-min", type=int, default=2)
    p.add_argument("--n-max", type=int, default=24)
    p.add_argument("--shots", type=int, default=1024)
    p.add_argument("--cpu-only", action="store_true")
    p.add_argument("--gpu-only", action="store_true")
    args = p.parse_args()

    do_cpu = not args.gpu_only
    do_gpu = not args.cpu_only

    print(f"=== GHZ-N scaling: N={args.n_min}..{args.n_max}, "
          f"{args.shots} shots, cpu={do_cpu}, gpu={do_gpu} ===\n")

    paths = run_scaling(
        n_range=range(args.n_min, args.n_max + 1),
        shots=args.shots,
        do_cpu=do_cpu,
        do_gpu=do_gpu,
        status=lambda m: print(m),
    )
    print("\nDone:")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

