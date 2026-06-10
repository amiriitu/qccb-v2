"""
Phase 1 smoke-test for Bauman Octillion batch submission.

Goal: verify how `backend.run([qc1, qc2, qc3], shots=10, project="batch_test")`
returns its results, before committing to refactor `runner.py` for batching.

What we want to know (questions the test answers):

  Q1. Does `job.counts` come back as a list of 3 dicts (one per circuit)?
  Q2. Is order preserved (circuit-i counts at index i)?
  Q3. Are values probabilities (∈ [0,1], sum ≈ 1.0) or raw integer shots?
  Q4. Does a 3-circuit batch take ~1× execute time or ~3× execute time?
       (i.e. does the chip parallelise or just sequentialise inside the batch)
  Q5. Does the chip-side autonomous calibration protection treat a batch as
       1 job (good — 1 cooldown) or as 3 jobs in flight (bad — 3× thermal load)?

Why Bell pair × 3 specifically:
  - Smallest 2-qubit entangling circuit (depth ≈ 10 after transpile, 4 CZ).
  - 4 CZ ≪ 8 CZ envelope cap → green for `classify_circuit_risk`.
  - Identical circuits → if order is shuffled or counts collide we can spot it
    by ensuring all 3 fidelities are within sampling noise of each other.
  - Wall-clock budget: ~17s if chip serializes, ~5s if it parallelises.

This script does NOT integrate with `runner.py` — it talks directly to
`quantum_hardware.submit` so the test is independent of the larger benchmark
machinery. Pass `--dry-run` to skip the actual chip submission and just verify
the script wires up.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path so `src.*` imports resolve when running
# this script directly (`python tools/test_batch_api.py`).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qiskit import QuantumCircuit, transpile

from src.quantum_hardware import (
    CHIP_NAME,
    get_client,
    get_backend,
    get_chip_queue_info,
    describe_chip,
    normalize_counts,
)


def build_bell_circuit(name: str = "bell") -> QuantumCircuit:
    """Standard Bell pair |Φ+⟩ = (|00⟩+|11⟩)/√2 with measurement."""
    qc = QuantumCircuit(2, 2, name=name)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def bell_fidelity(counts: dict[str, int]) -> float:
    total = sum(counts.values()) or 1
    return (counts.get("00", 0) + counts.get("11", 0)) / total


def describe_counts_payload(payload: Any) -> dict[str, Any]:
    """
    Return a small dict that captures the shape of `job.counts`:
      - top-level type (list / dict / other)
      - list length
      - each item: type + (if dict) keys + value-type + sum
    """
    info: dict[str, Any] = {"top_type": type(payload).__name__}

    if isinstance(payload, list):
        info["len"] = len(payload)
        info["items"] = []
        for i, item in enumerate(payload):
            if isinstance(item, dict):
                vals = list(item.values())
                value_kind = (
                    "float" if vals and all(isinstance(v, float) for v in vals)
                    else "int" if vals and all(isinstance(v, int) for v in vals)
                    else "mixed"
                )
                info["items"].append({
                    "index": i,
                    "type": "dict",
                    "n_keys": len(item),
                    "keys_sample": list(item.keys())[:8],
                    "value_kind": value_kind,
                    "sum": round(sum(float(v) for v in vals), 6),
                    "min": round(min((float(v) for v in vals), default=0.0), 6),
                    "max": round(max((float(v) for v in vals), default=0.0), 6),
                })
            else:
                info["items"].append({
                    "index": i, "type": type(item).__name__, "repr": repr(item)[:120],
                })
    elif isinstance(payload, dict):
        vals = list(payload.values())
        info["n_keys"] = len(payload)
        info["keys_sample"] = list(payload.keys())[:8]
        info["sum"] = round(sum(float(v) for v in vals), 6)
    else:
        info["repr"] = repr(payload)[:200]
    return info


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bauman Octillion batch-submission smoke test."
    )
    parser.add_argument(
        "--shots-exp", type=int, default=10,
        help="shots exponent (10 → 2^10 = 1024 actual shots; same as benchmark default)",
    )
    parser.add_argument(
        "--n-circuits", type=int, default=3,
        help="how many circuits to put in the batch (default 3)",
    )
    parser.add_argument(
        "--real", action="store_true", default=True,
        help="submit to real Snowdrop 4q ver2 (default; use --emulator to override)",
    )
    parser.add_argument(
        "--emulator", action="store_true",
        help="run on local Aer emulator instead of real chip",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="skip chip submission; just build/transpile and print circuit info",
    )
    parser.add_argument(
        "--timeout", type=float, default=180.0,
        help="poll timeout in seconds (default 180s ~ 3× single Bell job)",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "Bauman" / "runs" / "batch_api_smoke.json"),
        help="where to save the raw response for the runner refactor",
    )
    args = parser.parse_args()

    real_hardware = args.real and not args.emulator
    backend_label = "real" if real_hardware else "emulator"
    n = max(1, args.n_circuits)

    print(f"[batch-smoke] {n}× Bell circuit on {backend_label} "
          f"(shots=2^{args.shots_exp}={2 ** args.shots_exp})")

    # === Build circuits ===
    circuits = [build_bell_circuit(f"bell_batch_{i}") for i in range(n)]
    print(f"[batch-smoke] Built {n} identical Bell circuits "
          f"(depth={circuits[0].depth()}, ops={dict(circuits[0].count_ops())})")

    if args.dry_run:
        print("[batch-smoke] --dry-run: skipping chip submission. Wiring OK.")
        return 0

    # === Init client + backend ===
    t_setup = time.perf_counter()
    client = get_client()
    backend = get_backend(client, real_hardware=real_hardware, chip=CHIP_NAME)
    spec = describe_chip(backend)
    setup_s = time.perf_counter() - t_setup
    print(f"[batch-smoke] Setup {setup_s:.2f}s  "
          f"(basis={spec.basis_gates}, coupling={spec.coupling_map})")

    if real_hardware:
        qinfo = get_chip_queue_info(client, CHIP_NAME)
        print(f"[batch-smoke] Chip status={qinfo.get('status')} "
              f"queue={qinfo.get('queue')} maintenance={qinfo.get('is_maintenance')}")
        if not qinfo.get("ready"):
            print("[batch-smoke] Chip is NOT READY — aborting (no need to wait, "
                  "we'll just rerun when it's up)")
            return 2

    # === Transpile each circuit identically ===
    t = time.perf_counter()
    qcs_t = [
        transpile(
            qc,
            basis_gates=spec.basis_gates,
            coupling_map=[list(p) for p in spec.coupling_map],
            layout_method="trivial",
            routing_method="sabre",
            optimization_level=1,
        )
        for qc in circuits
    ]
    transpile_s = time.perf_counter() - t
    print(f"[batch-smoke] Transpile {n} circuits: {transpile_s*1000:.0f}ms  "
          f"(depth={qcs_t[0].depth()}, ops={dict(qcs_t[0].count_ops())})")

    # === Batch submit ===
    # Per `Bauman/api_example.ipynb`, the vendor's canonical batch call is:
    #     job = backend.run([qc_a, qc_b], shots=12, project="batch_name")
    # We mirror that exact shape.
    project = "qccb_batch_smoke"
    print(f"[batch-smoke] Submitting batch of {n} circuits, project={project!r}...")
    t_submit_start = time.perf_counter()
    try:
        job = backend.run(qcs_t, shots=args.shots_exp, project=project)
    except Exception as e:
        print(f"[batch-smoke] FAIL: backend.run([...]) raised: {e!r}")
        print("[batch-smoke] This means the API does not accept a list — we'll need "
              "to fall back to one-call-per-circuit (no batching speedup possible).")
        return 3
    submit_s = time.perf_counter() - t_submit_start
    print(f"[batch-smoke] Submitted in {submit_s*1000:.0f}ms; job.id={job.id}")

    # === Poll until complete ===
    if real_hardware and job.id is not None:
        from octillion import Job as OJob
        deadline = time.time() + args.timeout
        last_status = None
        t_poll_start = time.perf_counter()
        t_exec_start: float | None = None
        queue_s = 0.0
        execute_s = 0.0

        while time.time() < deadline:
            try:
                job_data = client._api.job(job.id)
            except Exception as e:
                print(f"[batch-smoke]   API error: {e!r} (retrying)")
                time.sleep(2.0)
                continue

            st = str(job_data.get("status", "UNKNOWN")).upper()
            if st != last_status:
                print(f"[batch-smoke]   job {job.id} status={st}")
                if st == "EXECUTING" and t_exec_start is None:
                    queue_s = time.perf_counter() - t_poll_start
                    t_exec_start = time.perf_counter()
                last_status = st

            if st in ("COMPLETE", "DONE", "FINISHED", "SUCCESS"):
                if t_exec_start is not None:
                    execute_s = time.perf_counter() - t_exec_start
                else:
                    queue_s = time.perf_counter() - t_poll_start
                job = OJob(
                    id=job_data.get("batch_id", job.id),
                    status=job_data.get("status"),
                    shots=job_data.get("shots", 0) or 0,
                    counts=job_data.get("counts", []) or [],
                )
                break
            if st in ("FAILED", "ERROR", "CANCELLED", "CANCELED"):
                print(f"[batch-smoke] FAIL: job ended with status={st}")
                _save_response(args.output, job, args, st, queue_s, execute_s, n)
                return 4
            time.sleep(2.0)
        else:
            print(f"[batch-smoke] FAIL: timeout after {args.timeout}s "
                  f"(last status={last_status})")
            return 5
    else:
        queue_s = 0.0
        execute_s = time.perf_counter() - t_submit_start  # emulator path

    total_s = setup_s + transpile_s + submit_s + queue_s + execute_s
    print(f"[batch-smoke] DONE.  queue={queue_s:.2f}s, execute={execute_s:.2f}s, "
          f"submit={submit_s*1000:.0f}ms, total≈{total_s:.2f}s")

    # === Inspect the response shape ===
    shape = describe_counts_payload(job.counts)
    print()
    print("=" * 64)
    print(" job.counts shape:")
    print(json.dumps(shape, indent=2, ensure_ascii=False))
    print("=" * 64)

    # === Per-circuit fidelity check (sanity) ===
    actual_shots = 2 ** args.shots_exp
    print()
    print(" Per-circuit fidelity (should all be ≈ same since circuits are identical):")
    if isinstance(job.counts, list):
        for i, raw in enumerate(job.counts):
            if isinstance(raw, dict):
                normalized = normalize_counts(raw, total_shots=actual_shots)
                f = bell_fidelity(normalized)
                print(f"   circuit {i}:  F_bell = {f:.4f}   "
                      f"(00={normalized.get('00', 0)}, "
                      f"01={normalized.get('01', 0)}, "
                      f"10={normalized.get('10', 0)}, "
                      f"11={normalized.get('11', 0)})")
            else:
                print(f"   circuit {i}: counts is NOT a dict — type={type(raw).__name__}")
    else:
        normalized = normalize_counts(job.counts, total_shots=actual_shots)
        f = bell_fidelity(normalized)
        print(f"   (single-result mode)  F_bell = {f:.4f}  counts={normalized}")

    # === Wall-clock answer to Q4 ===
    # Reference: single Bell on real ≈ 15-17s end-to-end per
    # bell_bench_1_real_fbe1d1d7-*.json (queue 0.5s, execute 15.3s).
    if real_hardware:
        ref_single_total = 16.0
        ratio = total_s / ref_single_total
        print()
        print(f" Q4 answer:  3-circuit batch took {total_s:.1f}s vs "
              f"~{ref_single_total:.0f}s for single circuit = {ratio:.2f}× ")
        if ratio < 1.5:
            print("   → chip executes the batch as a single block (BEST CASE)")
        elif ratio < 2.5:
            print("   → chip multiplexes; partial speedup")
        else:
            print("   → chip serializes; small batch overhead saved on queue only")

    _save_response(args.output, job, args, "COMPLETE", queue_s, execute_s, n)
    print(f"\n[batch-smoke] Saved raw response → {args.output}")
    return 0


def _save_response(out_path: str, job: Any, args: argparse.Namespace,
                   final_status: str, queue_s: float, execute_s: float,
                   n: int) -> None:
    """Persist the raw response so the runner refactor can replay against it."""
    payload = {
        "label": "batch_api_smoke",
        "n_circuits": n,
        "shots_exponent": args.shots_exp,
        "shots_actual": 2 ** args.shots_exp,
        "real_hardware": args.real and not args.emulator,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "job_id": str(job.id) if job and job.id else None,
        "final_status": final_status,
        "queue_s": round(queue_s, 3),
        "execute_s": round(execute_s, 3),
        "counts_raw": job.counts if job is not None else None,
        "counts_shape": describe_counts_payload(job.counts if job is not None else None),
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    sys.exit(main())

