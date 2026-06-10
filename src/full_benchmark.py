"""
Master orchestrator: runs every experiment on every selected backend with N
repeats, then dumps a complete scientific results/quantum/ artifact set.

CLI:
    python -m src.full_benchmark
    python -m src.full_benchmark --repeats 10 --backends ideal emulator real
    python -m src.full_benchmark --no-real      # skip real hardware
    python -m src.full_benchmark --extras       # also sweep GHZ-N and BV-secrets
"""
from __future__ import annotations

import argparse
import sys
from contextlib import suppress
import time
from pathlib import Path
from typing import Any, Callable

if sys.platform == "win32":
    with suppress(AttributeError, OSError):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments import list_experiments, ExperimentDef
from src.experiments.runner import (
    benchmark, BenchmarkResult, is_cancel_requested, CancelledError,
)
from src.quantum_hardware import get_client, get_backend, describe_chip
from src.report_v2 import ReportV2


def build_default_plan(extras: bool = False
                       ) -> list[tuple[ExperimentDef, dict]]:
    """
    The 4 canonical experiments designed to fit on Snowdrop 4q ver2:
      - Bell pair (2 qubits)
      - GHZ-3 entanglement (3 qubits)
      - Bernstein-Vazirani 3-bit (4 qubits with ancilla)
      - Compiled Shor for N=15, a=4 (4 qubits)

    extras=True adds GHZ-N sweep (N=2,3,4) and 4 BV secret variants.
    """
    by_key = {e.key: e for e in list_experiments()}
    plan: list[tuple[ExperimentDef, dict]] = []

    plan.append((by_key["bell"], {}))

    if extras:
        for n in (2, 3, 4):
            plan.append((by_key["ghz"], {"n": n}))
    else:
        plan.append((by_key["ghz"], {"n": 3}))

    if extras:
        for s in ("000", "010", "101", "111"):
            plan.append((by_key["bv"], {"secret": s}))
    else:
        plan.append((by_key["bv"], {"secret": "101"}))

    plan.append((by_key["shor"], {"a": 4}))

    return plan


def run_full_benchmark(
    backends: list[str],
    repeats: int = 5,
    shots_exponent: int = 10,
    extras: bool = False,
    output_dir: Path | None = None,
    status: Callable[[str], None] | None = None,
    on_run_complete: Callable[[int, int, BenchmarkResult], None] | None = None,
    include_pqc: bool = True,
    real_repeats: int | None = None,
    preflight: bool = True,
    preflight_threshold: float = 0.85,
    force_envelope_red: bool = False,
) -> dict[str, Path]:
    """
    Run the full benchmark suite. Returns a dict of artifact paths.

    Args:
        backends:         subset of {"ideal", "emulator", "real"}
        repeats:          per-(experiment, backend) repeat count
        shots_exponent:   shots = 2^shots_exponent
        extras:           if True, sweep GHZ-N and multiple BV secrets
        output_dir:       defaults to results/quantum/
        status:           callback for live status messages
        on_run_complete:  callback fired after each (config, backend) finishes
                          with (config_index, total_configs, BenchmarkResult)
    """
    status = status or (lambda m: None)

    plan = build_default_plan(extras=extras)
    total_configs = len(plan) * len(backends)

    if "gpu" in backends:
        try:
            from src.gpu_simulator import detect_gpu
            gpu_info = detect_gpu()
            status(f"GPU: {gpu_info['name']} | "
                   f"VRAM {gpu_info['vram_free_mb']}/{gpu_info['vram_total_mb']} MB | "
                   f"compute {gpu_info['compute_capability']}")
        except Exception as e:
            status(f"GPU detection failed ({e!r}), removing 'gpu' backend")
            backends = [b for b in backends if b != "gpu"]

    chip_spec = None
    bauman_backends = {"real"}  # only real hardware needs the live API spec
    if any(b in bauman_backends for b in backends):
        status("Querying live chip calibration from Bauman Octillion...")
        client = get_client()
        be = get_backend(client, real_hardware=("real" in backends))
        chip_spec = describe_chip(be)
        status(f"Chip: {chip_spec.num_qubits}q, "
               f"avg F_1q={chip_spec.avg_f1q*100:.2f}%, "
               f"F_2q={chip_spec.avg_f2q*100:.2f}%, "
               f"T1={chip_spec.avg_t1_us:.1f}us, T2={chip_spec.avg_t2_us:.1f}us")

        # Pre-flight: tiny Bell probe to gate the heavy batch. If the chip
        # just exited Сервис and is still drifting, drop `real` from the
        # backend list rather than slamming it with bad-state jobs.
        if preflight:
            from src.experiments.runner import preflight_real_hw_check
            pre = preflight_real_hw_check(
                threshold=preflight_threshold,
                status=lambda m: status(f"  {m}"),
            )
            if not pre.passed:
                status(f"⚠ Preflight FAILED ({pre.reason}) — removing 'real' "
                       f"from this run's backends. Other backends still execute. "
                       f"Re-run after the chip settles (5-15 min typical).")
                backends = [b for b in backends if b != "real"]
                total_configs = len(plan) * len(backends)
            else:
                status(f"✓ Preflight PASS: Bell fidelity {pre.fidelity:.4f} "
                       f"≥ {preflight_threshold:.2f}, chip looks healthy")

    if output_dir is None:
        # Per-run timestamped folder: results/2_quantum_hardware/DDMMYYYY HH MM/
        stamp = time.strftime("%d%m%Y %H %M")
        output_dir = (PROJECT_ROOT / "results" / "2_quantum_hardware" / stamp)
    rep = ReportV2(output_dir=output_dir, chip_spec=chip_spec)
    rep.log(f"Plan: {len(plan)} configs × {len(backends)} backends "
            f"× {repeats} repeats = {total_configs * repeats} runs total")

    config_idx = 0
    failed_configs: list[str] = []
    # Track real-hw configs that gave 0 runs (envelope-red, fidelity guard,
    # or any other pre-submit skip). These are candidates for the force-pass
    # if force_envelope_red is enabled.
    real_skipped: list[tuple[Any, dict]] = []
    cancelled = False
    for exp, params in plan:
        if cancelled:
            break
        for backend in backends:
            if is_cancel_requested():
                status("Stop requested — finishing current artifacts and exiting suite.")
                cancelled = True
                break
            config_idx += 1
            label = exp.title_with_params(params)
            status(f"[{config_idx}/{total_configs}] {label} on {backend}...")

            t0 = time.perf_counter()
            # Real hardware gets fewer repeats by default — sustained back-to-
            # back submissions push the chip into the maintenance/cool-down
            # cycle. Pass --real-repeats to override.
            effective_repeats = (
                real_repeats if (backend == "real" and real_repeats is not None)
                else repeats
            )
            try:
                bench = benchmark(
                    exp,
                    backend_kind=backend,
                    params=params,
                    shots_exponent=shots_exponent,
                    repeats=effective_repeats,
                    status=lambda m: status(f"  {m}"),
                )
            except CancelledError:
                cancelled = True
                status(f"  ✗ {label} / {backend}: cancelled by user")
                break
            except Exception as e:
                failed_configs.append(f"{label} / {backend}: {e}")
                status(f"  ✗ config FAILED ({e}) — continuing")
                continue
            dt = time.perf_counter() - t0

            if not bench.runs:
                failed_configs.append(f"{label} / {backend}: 0 successful runs")
                status(f"  ✗ all repeats failed — skipping config")
                if backend == "real" and force_envelope_red:
                    real_skipped.append((exp, params))
                continue

            rep.add(exp, params, bench)
            status(f"  → {bench.summary()}  (config wall time {dt:.1f}s)")
            if on_run_complete:
                on_run_complete(config_idx, total_configs, bench)

    # =====================================================================
    # Force-pass: re-run real-hw configs that the envelope/fidelity guards
    # skipped, this time with bypass_envelope=True. Runs at the very end so
    # that all "safe" data is already collected; if the chip CANCELs or
    # stalls in this phase the rest of the run is unaffected.
    # =====================================================================
    if force_envelope_red and real_skipped and not cancelled:
        status(f"\n=== FORCE-PASS: re-running {len(real_skipped)} "
               f"envelope-skipped real-hw config(s) with bypass ===")
        force_repeats = (
            real_repeats if real_repeats is not None else repeats
        )
        for fi, (exp, params) in enumerate(real_skipped, start=1):
            if is_cancel_requested() or cancelled:
                status("Stop requested — aborting force-pass.")
                break
            label = exp.title_with_params(params)
            status(f"[force {fi}/{len(real_skipped)}] {label} on real "
                   f"(bypass envelope)...")
            t0 = time.perf_counter()
            try:
                bench = benchmark(
                    exp,
                    backend_kind="real",
                    params=params,
                    shots_exponent=shots_exponent,
                    repeats=force_repeats,
                    status=lambda m: status(f"  {m}"),
                    bypass_envelope=True,
                )
            except CancelledError:
                cancelled = True
                status(f"  ✗ {label} / real (force): cancelled by user")
                break
            except Exception as e:
                failed_configs.append(
                    f"{label} / real (force): {e}"
                )
                status(f"  ✗ force-run FAILED ({e}) — continuing")
                continue
            dt = time.perf_counter() - t0
            if not bench.runs:
                failed_configs.append(
                    f"{label} / real (force): 0 successful runs "
                    f"(chip likely CANCELED the batch)"
                )
                status(f"  ✗ force-run produced 0 runs — chip may have "
                       f"refused; skipping config")
                continue
            rep.add(exp, params, bench)
            status(f"  → {bench.summary()}  (force-run wall time {dt:.1f}s)")
            if on_run_complete:
                on_run_complete(total_configs, total_configs, bench)

    if failed_configs:
        status(f"\nWARNING: {len(failed_configs)} configs failed:")
        for f in failed_configs:
            status(f"  - {f}")

    status(f"\nWriting artifacts to {output_dir}...")
    artifacts = rep.write_all()
    for name, path in artifacts.items():
        status(f"  ✓ {path.name}")

    if chip_spec is not None:
        status("\nComputing CRM (Cryptographically Reachable Modulus)...")
        try:
            from src.crm_forecast import build_artifacts as build_crm
            measured = {
                "n_qubits": chip_spec.num_qubits,
                "avg_f1q": chip_spec.avg_f1q,
                "avg_f2q": chip_spec.avg_f2q,
                "avg_ro":  chip_spec.avg_ro,
            }
            crm_paths = build_crm(measured, output_dir=output_dir,
                                    status=status)
            artifacts.update(crm_paths)
        except Exception as e:
            status(f"  CRM computation skipped: {e!r}")

    if include_pqc:
        status("\nRunning PQC software pipeline (Kyber/Dilithium + classical baselines + threat matrix + roadmap)...")
        try:
            import os
            os.environ.setdefault("QCCB_FORCE_SIMULATOR", "1")
            from src.pqc_pipeline import run_pqc_pipeline
            stamp = time.strftime("%d%m%Y %H %M")
            pqc_dir = (PROJECT_ROOT / "results" / "1_pqc_benchmarks" / stamp)
            pqc_paths = run_pqc_pipeline(
                output_dir=pqc_dir, quick=True, status=status,
            )
            for k, v in pqc_paths.items():
                artifacts[f"pqc_{k}"] = v
        except Exception as e:
            status(f"  PQC pipeline skipped: {e!r}")

    return artifacts


def main():
    p = argparse.ArgumentParser(description="Full quantum-hardware benchmark suite")
    p.add_argument("--repeats", type=int, default=5,
                   help="Per-(experiment, backend) repeat count (default: 5)")
    p.add_argument("--real-repeats", type=int, default=None,
                   help="Override repeats for real hardware (default: same as "
                        "--repeats; use 1-2 to be polite to the chip)")
    p.add_argument("--shots-exp", type=int, default=10,
                   help="Shots = 2^N (default: 10 → 1024)")
    p.add_argument("--backends", nargs="+",
                   default=["ideal", "gpu", "emulator_4q_v2", "real"],
                   choices=["ideal", "gpu",
                            "emulator", "emulator_4q_v2",
                            "emulator_4q_v1", "emulator_8q_v1",
                            "real"])
    p.add_argument("--no-real", action="store_true",
                   help="Skip real hardware")
    p.add_argument("--no-gpu", action="store_true",
                   help="Skip GPU backend")
    p.add_argument("--include-pqc", action="store_true",
                   help="Also run the PQC software pipeline "
                        "(Kyber/Dilithium/RSA/ECC + threat matrix + roadmap)")
    p.add_argument("--extras", action="store_true",
                   help="Sweep GHZ-N (2,3,4) and multiple BV secrets")
    p.add_argument("--no-preflight", action="store_true",
                   help="Skip the Bell-2q chip-health probe before the real-hw "
                        "batch (default: enabled)")
    p.add_argument("--preflight-threshold", type=float, default=0.85,
                   help="Minimum Bell fidelity for the chip to pass pre-flight "
                        "(default: 0.85; healthy Snowdrop 4q ver2 measures ≈ 0.91-0.96)")
    p.add_argument("--force-envelope-red", action="store_true",
                   help="After the main loop, re-run any real-hw configs that "
                        "were skipped by the envelope/fidelity guards, this "
                        "time with the safety check bypassed. Lets you "
                        "collect data on circuits the chip would normally "
                        "refuse (e.g. GHZ-4q). Force-pass runs at the very "
                        "end, so a chip CANCEL there does NOT affect the "
                        "main results.")
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    backends = args.backends
    if args.no_real and "real" in backends:
        backends = [b for b in backends if b != "real"]
    if args.no_gpu and "gpu" in backends:
        backends = [b for b in backends if b != "gpu"]

    output_dir = Path(args.output) if args.output else None

    print(f"=== QCCB Full Benchmark Suite ===")
    print(f"Backends: {backends}")
    print(f"Repeats:  {args.repeats}")
    print(f"Shots:    2^{args.shots_exp} = {2 ** args.shots_exp}")
    print(f"Extras:   {args.extras}")
    print()

    artifacts = run_full_benchmark(
        backends=backends,
        repeats=args.repeats,
        shots_exponent=args.shots_exp,
        extras=args.extras,
        output_dir=output_dir,
        status=lambda m: print(m),
        include_pqc=args.include_pqc,
        real_repeats=args.real_repeats,
        preflight=not args.no_preflight,
        preflight_threshold=args.preflight_threshold,
        force_envelope_red=args.force_envelope_red,
    )
    print(f"\n=== Done. Artifacts in: ===")
    for path in artifacts.values():
        print(f"  {path}")


if __name__ == "__main__":
    main()

