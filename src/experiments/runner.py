"""
Generic experiment runner. Builds, transpiles, submits, polls, saves
any QuantumCircuit. Used by all experiments and the GUI worker thread.

Adds:
  - Timing breakdown: submit / queue / execute / total
  - benchmark(): repeats an experiment N times, returns mean ± std + 95% CI
"""
from __future__ import annotations

import logging
import math
import statistics
import sys
from contextlib import suppress
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from qiskit import QuantumCircuit, transpile

if sys.platform == "win32":
    with suppress(AttributeError, OSError):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

from src.quantum_hardware import (
    get_client,
    get_backend,
    describe_chip,
    submit,
    normalize_counts,
    normalize_counts_list,
    save_run,
    ChipSpec,
)


StatusCallback = Callable[[str], None]


# ============================================================================
# Cooperative cancellation
#
# A single benchmark/suite is in-flight at a time, so a module-level cancel
# state is sufficient. The GUI calls request_cancel() when the user clicks
# Stop, which (a) sets the flag so cooperative checks raise CancelledError,
# and (b) sends a server-side cancel to Bauman if a job is currently in queue
# or executing.
# ============================================================================

class CancelledError(RuntimeError):
    """Raised by _check_cancel() when the user has requested a stop."""


_cancel_event: threading.Event = threading.Event()
_current_remote_job: tuple | None = None  # (client, job_id) or None

# Cooldown between consecutive real-hardware runs in benchmark().
# Sustained back-to-back submissions push the chip into NOTREADY (thermal
# protection); a small breather between runs avoids this.
_REAL_HW_COOLDOWN_S: float = 6.0


def reset_cancel() -> None:
    """Call once at the start of any new benchmark/suite/single run."""
    global _current_remote_job
    _cancel_event.clear()
    _current_remote_job = None


def request_cancel() -> bool:
    """
    User-initiated stop. Returns True iff a remote job was canceled.
    Always sets the cancel flag so the next cooperative check raises.
    """
    global _current_remote_job
    _cancel_event.set()
    if _current_remote_job is not None:
        client, job_id = _current_remote_job
        try:
            client.cancel(job_id)
            return True
        except Exception:
            return False
    return False


def is_cancel_requested() -> bool:
    return _cancel_event.is_set()


def _check_cancel() -> None:
    if _cancel_event.is_set():
        raise CancelledError("Stop requested by user")


def _sleep_with_cancel(seconds: float) -> None:
    """Cooperative sleep — checks the cancel flag every second."""
    if seconds <= 0:
        return
    end = time.time() + seconds
    while True:
        _check_cancel()
        remaining = end - time.time()
        if remaining <= 0:
            return
        time.sleep(min(1.0, remaining))


def _wait_for_chip_ready(
    client, chip_name: str, status: "StatusCallback",
    backend_str: str,
    max_wait_s: float = 300.0,
    poll_interval: float = 30.0,
) -> None:
    """
    Block until the chip transitions to READY. Bauman Octillion chips drop
    into NOTREADY (maintenance / thermal-protection) after sustained use;
    instead of slamming the chip with submissions that will all fail with
    "Failed to connect to Octillion server", we wait politely.

    Raises RuntimeError if max_wait_s elapses with the chip still NOTREADY.
    Honors cooperative cancellation throughout.
    """
    from src.quantum_hardware import get_chip_queue_info
    t_start = time.time()
    deadline = t_start + max_wait_s
    announced = False
    last_state: str | None = None
    while time.time() < deadline:
        _check_cancel()
        try:
            qinfo = get_chip_queue_info(client, chip_name)
            ready = bool(qinfo.get("ready"))
            qstate = str(qinfo.get("status", "?"))
        except Exception as e:
            ready = False
            qstate = f"query-failed ({type(e).__name__})"
        if ready:
            if announced:
                elapsed = int(time.time() - t_start)
                status(f"{backend_str}: chip recovered → READY (after {elapsed}s)")
            return
        elapsed = int(time.time() - t_start)
        if not announced:
            status(f"{backend_str}: chip is {qstate} (cool-down/maintenance) — "
                   f"waiting up to {int(max_wait_s)}s, polling every "
                   f"{int(poll_interval)}s...")
            announced = True
            last_state = qstate
        elif qstate != last_state:
            status(f"{backend_str}: chip status: {last_state} → {qstate} "
                   f"(waited {elapsed}s)")
            last_state = qstate
        _sleep_with_cancel(poll_interval)
    raise RuntimeError(
        f"Chip {chip_name!r} did not return to READY within "
        f"{int(max_wait_s)}s — aborting submission. The chip is in an "
        f"extended maintenance cycle; try again in a few minutes."
    )


def _set_current_remote_job(client, job_id: str | None) -> None:
    global _current_remote_job
    if job_id is None:
        _current_remote_job = None
    else:
        _current_remote_job = (client, job_id)


def _try_cancel_remote_job(client, job_id: str | None,
                            status: "StatusCallback", backend_str: str) -> bool:
    """
    Best-effort server-side cancellation of a remote Octillion job.

    Used when a retry loop is about to submit a fresh batch but the previous
    one is still EXECUTING on the chip (e.g. after a poll timeout caused by
    a transient network outage). Without this, the orphan job continues to
    occupy chip queue/execute slot — wasting chip-time and blocking the
    retry batch behind itself.

    Never raises: if the cancel API itself fails (network still down, job
    already completed, etc.) we just log and move on. Returns True iff the
    DELETE was acknowledged by the server.
    """
    if not job_id:
        return False
    try:
        client.cancel(job_id)
        status(f"{backend_str}: cancelled orphan job {job_id}")
        return True
    except Exception as e:
        status(f"{backend_str}: orphan job {job_id} cancel attempt failed "
               f"({type(e).__name__}: {e}) — moving on")
        return False


# ---------------------------------------------------------------------------
# Pre-flight: tiny chip-health probe before launching a heavy real-hw batch
# ---------------------------------------------------------------------------
# Rationale: real chips can be READY at the API level but be in a degraded
# calibration window (fidelities below paper spec). Submitting a long
# benchmark batch in that state both wastes chip time and risks pushing the
# chip into a deeper Сервис cycle. A 1-shot Bell with N=512 (~10-15 s) is
# cheap insurance — if Bell fidelity < threshold we drop `real` from the
# backend list rather than slamming the chip with bad-state jobs.

@dataclass
class PreflightResult:
    """Outcome of a chip-health pre-flight check."""
    passed: bool
    fidelity: float
    threshold: float
    shots: int
    chip_name: str
    elapsed_s: float
    reason: str

    def to_dict(self) -> dict:
        return {
            "passed": self.passed, "fidelity": self.fidelity,
            "threshold": self.threshold, "shots": self.shots,
            "chip": self.chip_name, "elapsed_s": self.elapsed_s,
            "reason": self.reason,
        }


_preflight_cache: dict[str, tuple[float, "PreflightResult"]] = {}
_PREFLIGHT_TTL_S: float = 300.0  # 5-minute cache: avoid double-checking when
                                  # full_benchmark + benchmark() both ask


def preflight_real_hw_check(
    chip_name: str = "Snowdrop 4q ver2",
    threshold: float = 0.85,
    shots: int = 256,
    status: "StatusCallback | None" = None,
    use_cache: bool = True,
) -> PreflightResult:
    """
    Submit a single 2-qubit Bell circuit to the chip and verify Bell fidelity
    ≥ `threshold`. Used to gate heavy benchmark batches.

    Bell circuit: H(q0); CX(q0,q1); measure both. Ideal counts on noiseless
    backend: {00: 50%, 11: 50%}, fidelity = P(00) + P(11) = 1.0.
    A healthy Snowdrop 4q ver2 in normal calibration measures ≈ 0.91-0.96.
    Below 0.85 typically means the chip just exited Сервис, drift hasn't
    settled, or thermal cycling is underway.

    Honors cooperative cancellation throughout. Result is cached for
    `_PREFLIGHT_TTL_S` seconds, keyed by chip_name, to avoid duplicate
    submissions when nested callers each call this.
    """
    status = status or _noop
    now = time.time()
    if use_cache and chip_name in _preflight_cache:
        ts, cached = _preflight_cache[chip_name]
        if now - ts < _PREFLIGHT_TTL_S:
            status(f"preflight[{chip_name}]: cached result "
                   f"(fidelity={cached.fidelity:.4f}, "
                   f"{int(now - ts)}s old, "
                   f"{'PASS' if cached.passed else 'FAIL'})")
            return cached

    status(f"preflight[{chip_name}]: submitting Bell-2q probe "
           f"({shots} shots, threshold={threshold:.2f})...")
    t_start = time.perf_counter()

    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])

    try:
        client = get_client()
        backend = _get_cached_backend(client, real_hardware=True, chip=chip_name)
        spec = _get_cached_spec(backend, f"1::{chip_name}")
        from src.quantum_hardware import get_chip_queue_info
        qinfo = get_chip_queue_info(client, chip_name)
        if not qinfo.get("ready"):
            _wait_for_chip_ready(client, chip_name, status,
                                  f"preflight[{chip_name}]")

        qc_t = transpile(
            qc, basis_gates=spec.basis_gates,
            coupling_map=[list(p) for p in spec.coupling_map],
            layout_method="trivial", routing_method="sabre",
            optimization_level=1,
        )
        # shots_exponent encodes 2^N; round up so we hit at least `shots`
        shots_exp = max(8, int(math.ceil(math.log2(max(shots, 2)))))
        actual_shots = 2 ** shots_exp

        job = submit(backend, qc_t, shots_exponent=shots_exp,
                      project="qccb_preflight")
        _set_current_remote_job(client, job.id)
        try:
            job, _, _ = _poll_with_timing(
                client, job, status, f"preflight[{chip_name}]"
            )
        finally:
            _set_current_remote_job(client, None)

        counts = normalize_counts(job.counts, total_shots=actual_shots)
        p_00 = counts.get("00", 0) / actual_shots
        p_11 = counts.get("11", 0) / actual_shots
        fidelity = p_00 + p_11
        elapsed = time.perf_counter() - t_start
        passed = fidelity >= threshold

        result = PreflightResult(
            passed=passed, fidelity=fidelity, threshold=threshold,
            shots=actual_shots, chip_name=chip_name, elapsed_s=elapsed,
            reason=("OK" if passed
                     else f"Bell fidelity {fidelity:.4f} < threshold "
                          f"{threshold:.4f} — chip likely in post-Сервис drift, "
                          f"recommend waiting a few minutes"),
        )
        verdict = "PASS" if passed else "FAIL"
        status(f"preflight[{chip_name}]: {verdict} — fidelity={fidelity:.4f} "
               f"({actual_shots} shots, {elapsed:.1f}s)")
        _preflight_cache[chip_name] = (now, result)
        return result
    except CancelledError:
        raise
    except Exception as e:
        elapsed = time.perf_counter() - t_start
        result = PreflightResult(
            passed=False, fidelity=0.0, threshold=threshold,
            shots=shots, chip_name=chip_name, elapsed_s=elapsed,
            reason=f"preflight raised {type(e).__name__}: {e}",
        )
        status(f"preflight[{chip_name}]: ERROR — {result.reason}")
        _preflight_cache[chip_name] = (now, result)
        return result


# ---------------------------------------------------------------------------
# Backend cache
# ---------------------------------------------------------------------------
# `client.local("Snowdrop 4q ver2")` performs two HTTP GETs every time it
# is called (chips() + chip(id)) — costing ~1 s of "Python setup" wall-clock
# on each run. The chip image is static within a session, so we cache the
# backend handle keyed by (real_hardware, chip_name) and reuse it.
_backend_cache: dict[tuple[bool, str], Any] = {}
_cached_client: Any = None
_cached_specs: dict[str, Any] = {}


def _get_cached_backend(client, real_hardware: bool, chip: str):
    global _cached_client
    if _cached_client is not client:
        # Client identity changed (rare) — invalidate
        _backend_cache.clear()
        _cached_specs.clear()
        _cached_client = client
    key = (real_hardware, chip)
    if key not in _backend_cache:
        from src.quantum_hardware import get_backend
        _backend_cache[key] = get_backend(client, real_hardware=real_hardware, chip=chip)
    return _backend_cache[key]


def _get_cached_spec(backend, chip_key: str):
    if chip_key not in _cached_specs:
        from src.quantum_hardware import describe_chip
        _cached_specs[chip_key] = describe_chip(backend)
    return _cached_specs[chip_key]


def clear_backend_cache() -> None:
    """Explicitly drop cached backend handles (e.g. after token rotation)."""
    global _cached_client
    _backend_cache.clear()
    _cached_specs.clear()
    _cached_client = None


@dataclass
class TimingBreakdown:
    """
    Phase-by-phase wall-clock timing of one experiment run, in seconds.

    Phases are categorized into three groups for the "Python overhead vs
    algorithmic time" attribution that the GUI and the final report use.

    PYTHON ORCHESTRATION (overhead — would be saved by Cython/C wrapper):
      - python_setup_s   : client init, get_backend, describe_chip, dataclass build
      - python_post_s    : counts normalization, metric computation, formatting

    MIXED (Python driver + C/Rust core):
      - transpile_s      : qiskit.transpile (Qiskit 2.x has Rust core; counted as algorithmic)
      - serialize_s      : circuit → QASM serialization (Python with C calls; counted as algorithmic)

    ALGORITHMIC / HARDWARE (cannot be saved — fundamental cost):
      - submit_s         : HTTP submit to Octillion API (network)
      - queue_s          : Bauman backend queue wait
      - execute_s        : actual chip / Aer / GPU compute time
    """
    python_setup_s: float = 0.0
    python_post_s: float = 0.0
    transpile_s: float = 0.0
    serialize_s: float = 0.0
    submit_s: float = 0.0
    queue_s: float = 0.0
    execute_s: float = 0.0
    total_s: float = 0.0

    @property
    def python_overhead_s(self) -> float:
        """Pure Python orchestration overhead — the time saved by an optimized stack."""
        return self.python_setup_s + self.python_post_s

    @property
    def algorithmic_s(self) -> float:
        """Time on an optimized (C/Cython/native) stack — what the literature reports."""
        return (self.transpile_s + self.serialize_s
                + self.submit_s + self.queue_s + self.execute_s)

    @property
    def python_overhead_pct(self) -> float:
        return 100.0 * self.python_overhead_s / self.total_s if self.total_s > 0 else 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "python_setup_s": self.python_setup_s,
            "python_post_s": self.python_post_s,
            "transpile_s": self.transpile_s,
            "serialize_s": self.serialize_s,
            "submit_s": self.submit_s,
            "queue_s": self.queue_s,
            "execute_s": self.execute_s,
            "total_s": self.total_s,
            "python_overhead_s": self.python_overhead_s,
            "algorithmic_s": self.algorithmic_s,
            "python_overhead_pct": self.python_overhead_pct,
        }


@dataclass
class ExperimentResult:
    """Generic result container for any experiment."""
    label: str
    backend: str  # "ideal" | "emulator" | "real"
    shots: int
    counts: dict[str, int]
    job_id: str | None
    transpiled_depth: int
    transpiled_ops: dict[str, int]
    chip_spec: ChipSpec | None = None
    metric_name: str = ""
    metric_value: float = 0.0
    expected_distribution: dict[str, float] = field(default_factory=dict)
    save_path: Path | None = None
    timing: TimingBreakdown = field(default_factory=TimingBreakdown)

    def probabilities(self) -> dict[str, float]:
        total = sum(self.counts.values()) or 1
        return {k: v / total for k, v in self.counts.items()}


@dataclass
class BenchmarkResult:
    """Aggregated multi-run benchmark."""
    label: str
    backend: str
    shots_per_run: int
    repeats: int
    runs: list[ExperimentResult]

    @property
    def fidelities(self) -> list[float]:
        return [r.metric_value for r in self.runs]

    @property
    def mean(self) -> float:
        v = self.fidelities
        return statistics.mean(v) if v else 0.0

    @property
    def stdev(self) -> float:
        v = self.fidelities
        return statistics.stdev(v) if len(v) > 1 else 0.0

    @property
    def ci_95(self) -> tuple[float, float]:
        v = self.fidelities
        if len(v) < 2:
            return (self.mean, self.mean)
        try:
            from scipy.stats import t as student_t
            crit = float(student_t.ppf(0.975, len(v) - 1))
        except Exception:
            crit = 2.0
        margin = crit * self.stdev / math.sqrt(len(v))
        return (self.mean - margin, self.mean + margin)

    @property
    def avg_total_s(self) -> float:
        v = [r.timing.total_s for r in self.runs if r.timing.total_s > 0]
        return statistics.mean(v) if v else 0.0

    @property
    def avg_queue_s(self) -> float:
        v = [r.timing.queue_s for r in self.runs if r.timing.queue_s > 0]
        return statistics.mean(v) if v else 0.0

    @property
    def avg_execute_s(self) -> float:
        v = [r.timing.execute_s for r in self.runs if r.timing.execute_s > 0]
        return statistics.mean(v) if v else 0.0

    @property
    def avg_python_overhead_s(self) -> float:
        v = [r.timing.python_overhead_s for r in self.runs]
        return statistics.mean(v) if v else 0.0

    @property
    def avg_algorithmic_s(self) -> float:
        v = [r.timing.algorithmic_s for r in self.runs]
        return statistics.mean(v) if v else 0.0

    @property
    def avg_python_overhead_pct(self) -> float:
        v = [r.timing.python_overhead_pct for r in self.runs]
        return statistics.mean(v) if v else 0.0

    @property
    def avg_setup_s(self) -> float:
        v = [r.timing.python_setup_s for r in self.runs]
        return statistics.mean(v) if v else 0.0

    @property
    def avg_post_s(self) -> float:
        v = [r.timing.python_post_s for r in self.runs]
        return statistics.mean(v) if v else 0.0

    @property
    def avg_transpile_s(self) -> float:
        v = [r.timing.transpile_s for r in self.runs]
        return statistics.mean(v) if v else 0.0

    def summary(self) -> str:
        lo, hi = self.ci_95
        return (
            f"N={self.repeats} | mean={self.mean:.4f} ± {self.stdev:.4f} "
            f"| 95% CI=[{lo:.4f}, {hi:.4f}] "
            f"| total={self.avg_total_s*1000:.0f}ms "
            f"(algo={self.avg_algorithmic_s*1000:.0f}ms, "
            f"py-overhead={self.avg_python_overhead_pct:.1f}%)"
        )


def _noop(_msg: str) -> None:
    """Default status sink: route progress lines to the debug log."""
    logging.getLogger("QCCB.runner").debug(_msg)


def _extract_final_layout(qc_transpiled, n_virtual_qubits: int) -> list[int]:
    """
    Extract `final_index_layout` from a transpiled Qiskit circuit, with
    graceful fallbacks for older Qiskit versions or circuits where no
    routing happened.

    Returns a list of length n_virtual_qubits where index v holds the
    physical-qubit position of virtual qubit v at the END of the circuit.
    If extraction fails (e.g. backend has no layout, or Qiskit API
    differs), returns the identity permutation — which is also the
    correct answer for any circuit that needed no SWAPs.
    """
    identity = list(range(n_virtual_qubits))
    try:
        layout = getattr(qc_transpiled, "layout", None)
        if layout is None:
            return identity
        # Qiskit 1.x+ API
        try:
            perm = layout.final_index_layout(filter_ancillas=True)
        except TypeError:
            perm = layout.final_index_layout()
        if perm is None:
            return identity
        perm = list(perm)
        # Pad/truncate to expected length so callers get a stable shape.
        if len(perm) < n_virtual_qubits:
            perm = perm + identity[len(perm):]
        return perm[:n_virtual_qubits]
    except Exception:
        return identity


def run_circuit(
    circuit: QuantumCircuit,
    label: str,
    backend_kind: str = "real",
    shots_exponent: int = 10,
    expected_distribution: dict[str, float] | None = None,
    metric_name: str = "",
    metric_fn: Callable[[dict[str, int]], float] | None = None,
    project: str | None = None,
    status: StatusCallback | None = None,
) -> ExperimentResult:
    """Submit a circuit to one of three backends and return a uniform result."""
    status = status or _noop
    actual_shots = 2 ** shots_exponent
    backend_kind = backend_kind.lower()

    # All Bauman backends now target the only working chip on the platform:
    # Snowdrop 4q ver2 (emulator + real hardware).
    # Local emulators: Aer + chip noise model from saved JSON, NO network calls
    local_emu_chips = {
        "emulator":         "Snowdrop 4q ver2",
        "emulator_4q_v2":   "Snowdrop 4q ver2",
        "emulator_4q_v1":   "Snowdrop 4q ver1",
        "emulator_8q_v1":   "Snowdrop 8q ver1",
    }

    if backend_kind == "ideal":
        return _run_ideal(circuit, label, actual_shots, expected_distribution,
                          metric_name, metric_fn, status)
    elif backend_kind in ("gpu", "cupy_gpu", "cuda"):
        return _run_gpu(circuit, label, actual_shots, expected_distribution,
                        metric_name, metric_fn, status)
    elif backend_kind in local_emu_chips:
        return _run_local_emulator(
            circuit, label, actual_shots, expected_distribution,
            metric_name, metric_fn, status,
            chip_name=local_emu_chips[backend_kind],
            backend_kind=backend_kind,
        )
    elif backend_kind in ("emu", "local", "noisy"):
        # Legacy alias
        return _run_local_emulator(
            circuit, label, actual_shots, expected_distribution,
            metric_name, metric_fn, status,
            chip_name="Snowdrop 4q ver2",
            backend_kind="emulator",
        )
    elif backend_kind in ("real", "hardware", "device"):
        return _run_octillion(circuit, label, shots_exponent, actual_shots,
                              real_hardware=True,
                              chip_name="Snowdrop 4q ver2",
                              expected_distribution=expected_distribution,
                              metric_name=metric_name, metric_fn=metric_fn,
                              project=project, status=status)
    else:
        raise ValueError(f"Unknown backend_kind: {backend_kind}")


def _run_ideal(
    circuit, label, actual_shots, expected_distribution,
    metric_name, metric_fn, status,
) -> ExperimentResult:
    from qiskit_aer import AerSimulator

    timing = TimingBreakdown()
    t_overall = time.perf_counter()

    # === Phase: Python setup ===
    t = time.perf_counter()
    status("ideal: transpiling circuit (no noise model)...")
    sim = AerSimulator()
    timing.python_setup_s = time.perf_counter() - t

    # === Phase: Transpile (Qiskit core — Rust + Python) ===
    t = time.perf_counter()
    qc_t = transpile(circuit, sim, optimization_level=2)
    timing.transpile_s = time.perf_counter() - t

    # === Phase: Execute (Aer C++ core) ===
    status(f"ideal: running {actual_shots} shots...")
    t = time.perf_counter()
    result = sim.run(qc_t, shots=actual_shots).result()
    timing.execute_s = time.perf_counter() - t

    # === Phase: Python post-processing ===
    t = time.perf_counter()
    counts = {k: int(v) for k, v in result.get_counts().items()}
    metric_value = float(metric_fn(counts)) if metric_fn else 0.0
    timing.python_post_s = time.perf_counter() - t

    timing.total_s = time.perf_counter() - t_overall

    res = ExperimentResult(
        label=label, backend="ideal",
        shots=actual_shots, counts=counts, job_id=None,
        transpiled_depth=qc_t.depth(),
        transpiled_ops={k: int(v) for k, v in qc_t.count_ops().items()},
        metric_name=metric_name, metric_value=metric_value,
        expected_distribution=expected_distribution or {},
        timing=timing,
    )
    status(
        f"ideal: done, {metric_name}={metric_value:.4f} "
        f"(total {timing.total_s*1000:.1f}ms; algo {timing.algorithmic_s*1000:.1f}ms; "
        f"py-overhead {timing.python_overhead_pct:.1f}%)"
        if metric_name else
        f"ideal: done ({timing.total_s:.2f}s, py-overhead {timing.python_overhead_pct:.1f}%)"
    )
    return res


def _run_gpu(
    circuit, label, actual_shots, expected_distribution,
    metric_name, metric_fn, status,
) -> ExperimentResult:
    """GPU state-vector simulation via CuPy + nvidia-cublas-cu12."""
    from src.gpu_simulator import run_circuit_gpu

    timing = TimingBreakdown()
    t_overall = time.perf_counter()

    # === Phase: Python setup (import, basis gate setup) ===
    t = time.perf_counter()
    status("gpu: transpiling for CuPy backend...")
    timing.python_setup_s = time.perf_counter() - t

    # === Phase: Transpile ===
    t = time.perf_counter()
    qc_t = transpile(circuit, basis_gates=["h", "x", "y", "z", "s", "t", "sx",
                                            "rx", "ry", "rz", "cx", "cz", "id"],
                     optimization_level=1)
    timing.transpile_s = time.perf_counter() - t

    # === Phase: Execute (CuPy → cuBLAS CUDA core) ===
    status(f"gpu: running {actual_shots} shots on CuPy state-vector...")
    t = time.perf_counter()
    stats = run_circuit_gpu(qc_t, shots=actual_shots, warmup=True)
    timing.execute_s = time.perf_counter() - t

    # === Phase: Python post-processing ===
    t = time.perf_counter()
    counts = stats.counts
    metric_value = float(metric_fn(counts)) if metric_fn else 0.0
    timing.python_post_s = time.perf_counter() - t

    timing.total_s = time.perf_counter() - t_overall

    res = ExperimentResult(
        label=label, backend="gpu",
        shots=actual_shots, counts=counts, job_id=None,
        transpiled_depth=qc_t.depth(),
        transpiled_ops={k: int(v) for k, v in qc_t.count_ops().items()},
        metric_name=metric_name, metric_value=metric_value,
        expected_distribution=expected_distribution or {},
        timing=timing,
    )
    status(
        f"gpu: done, {metric_name}={metric_value:.4f} "
        f"(total {timing.total_s*1000:.1f}ms; algo {timing.algorithmic_s*1000:.1f}ms; "
        f"py-overhead {timing.python_overhead_pct:.1f}%; "
        f"VRAM {stats.vram_used_mb:.0f}MB)" if metric_name else
        f"gpu: done ({timing.total_s:.2f}s, py-overhead {timing.python_overhead_pct:.1f}%)"
    )
    return res


def _run_local_emulator(
    circuit, label, actual_shots, expected_distribution,
    metric_name, metric_fn, status,
    chip_name: str, backend_kind: str,
) -> ExperimentResult:
    """
    Local Aer simulator with chip noise model loaded from a saved
    Bauman calibration JSON. Zero network calls, zero Bauman dependency.
    """
    from src.local_chip_emulator import build_aer_simulator

    timing = TimingBreakdown()
    t_overall = time.perf_counter()
    _check_cancel()

    backend_str = f"local_emu[{chip_name}]"

    # === Phase: Python setup (parse JSON + build noise model, cached) ===
    t = time.perf_counter()
    status(f"{backend_str}: building Aer noise model from saved calibration...")
    sim, spec = _get_local_aer_for_chip(chip_name)
    timing.python_setup_s = time.perf_counter() - t
    _check_cancel()

    # === Phase: Transpile ===
    status(f"{backend_str}: transpiling for chip basis "
           f"{spec.basis_gates} (coupling {spec.coupling_map})...")
    t = time.perf_counter()
    qc_t = transpile(
        circuit,
        basis_gates=spec.basis_gates,
        coupling_map=[list(p) for p in spec.coupling_map],
        layout_method="trivial",
        routing_method="sabre",
        optimization_level=1,
    )
    timing.transpile_s = time.perf_counter() - t

    # === Phase: Execute (Aer C++ kernel, all-local) ===
    status(f"{backend_str}: running on local Aer (no chip submission) "
           f"(depth={qc_t.depth()}, ops={dict(qc_t.count_ops())}, "
           f"shots={actual_shots})...")
    t = time.perf_counter()
    result = sim.run(qc_t, shots=actual_shots).result()
    timing.execute_s = time.perf_counter() - t

    # === Phase: Python post-processing ===
    t = time.perf_counter()
    counts = {k: int(v) for k, v in result.get_counts().items()}
    metric_value = float(metric_fn(counts)) if metric_fn else 0.0
    timing.python_post_s = time.perf_counter() - t

    timing.total_s = time.perf_counter() - t_overall

    res = ExperimentResult(
        label=label, backend=backend_kind,
        shots=actual_shots, counts=counts, job_id=None,
        transpiled_depth=qc_t.depth(),
        transpiled_ops={k: int(v) for k, v in qc_t.count_ops().items()},
        metric_name=metric_name, metric_value=metric_value,
        expected_distribution=expected_distribution or {},
        timing=timing,
    )
    status(
        f"{backend_str}: done, {metric_name}={metric_value:.4f}  "
        f"(total {timing.total_s*1000:.0f}ms; algo {timing.algorithmic_s*1000:.0f}ms; "
        f"py-overhead {timing.python_overhead_pct:.1f}%)"
        if metric_name else
        f"{backend_str}: done ({timing.total_s:.2f}s)"
    )
    return res


# Local-emulator cache: chip_name → (AerSimulator, LocalChipSpec)
_local_aer_cache: dict[str, Any] = {}


def _get_local_aer_for_chip(chip_name: str):
    """Cached factory for the local Aer simulator built from saved JSON."""
    if chip_name not in _local_aer_cache:
        from src.local_chip_emulator import build_aer_simulator
        _local_aer_cache[chip_name] = build_aer_simulator(chip_name)
    return _local_aer_cache[chip_name]


def _run_octillion(
    circuit, label, shots_exponent, actual_shots, real_hardware,
    expected_distribution, metric_name, metric_fn, project, status,
    chip_name: str = "Snowdrop 4q ver2",
) -> ExperimentResult:
    backend_str = ("real" if real_hardware else "emulator") + (
        f"[{chip_name}]" if chip_name != "Snowdrop 4q ver2" else ""
    )
    timing = TimingBreakdown()
    t_overall = time.perf_counter()
    _check_cancel()

    # === Phase: Python setup (client + chip lookup) ===
    # Cached after the first call within a session — saves ~1 s of HTTP GETs
    # (chips() + chip(id)) on every subsequent run.
    t = time.perf_counter()
    if real_hardware:
        status(f"{backend_str}: preparing remote submission to {chip_name}...")
    else:
        status(f"{backend_str}: preparing local Aer simulation with "
               f"{chip_name} noise model...")
    client = get_client()
    cache_key = f"{int(real_hardware)}::{chip_name}"
    backend = _get_cached_backend(client, real_hardware, chip_name)
    spec = _get_cached_spec(backend, cache_key)
    timing.python_setup_s = time.perf_counter() - t
    _check_cancel()

    # On real hardware, peek at status / queue so the user sees what they're
    # about to wait for. If the chip is in NOTREADY/maintenance, wait politely
    # for it to come back instead of slamming submissions that will all fail
    # with "Failed to connect to Octillion server".
    if real_hardware:
        try:
            from src.quantum_hardware import get_chip_queue_info
            qinfo = get_chip_queue_info(client, chip_name)
            qstate = qinfo.get("status", "?")
            qlen = qinfo.get("queue")
            qlen_txt = f"{qlen}" if qlen is not None else "n/a"
            status(f"{backend_str}: chip {qstate}, queue={qlen_txt} jobs ahead")
            if not qinfo.get("ready"):
                _wait_for_chip_ready(client, chip_name, status, backend_str)
                # Re-pull spec/backend after cool-down (chip metadata fresh)
                _check_cancel()
        except CancelledError:
            raise
        except RuntimeError:
            # _wait_for_chip_ready timed out — fatal, propagate
            raise
        except Exception as exc:
            # status query itself failed — proceed and let backend.run() error
            status(f"{backend_str}: chip status query failed ({exc!r}), "
                   f"continuing anyway")

    # === Phase: Transpile (Qiskit Rust+Python core) ===
    # seed_transpiler is pinned so Sabre routing is deterministic across
    # repeats — combined with the `final_index_layout` capture below, this
    # ensures bitstring permutation is reproducible and audit-traceable.
    status(f"{backend_str}: transpiling for chip basis {spec.basis_gates}...")
    t = time.perf_counter()
    qc_t = transpile(
        circuit,
        basis_gates=spec.basis_gates,
        coupling_map=[list(p) for p in spec.coupling_map],
        layout_method="trivial",
        routing_method="sabre",
        optimization_level=1,
        seed_transpiler=42,
    )
    timing.transpile_s = time.perf_counter() - t
    # NOTE: Bauman's chip returns bitstrings in Qiskit-standard c-register
    # MSB-first order (c[N-1]...c[0]), already honoring `measure q[X] -> c[Y]`
    # remap — verified empirically with an X(q0) probe. So no bit permutation
    # is needed; seed_transpiler=42 alone fixes the BV s=010 issue by making
    # Sabre routing deterministic.

    if real_hardware:
        action_verb = "submitting to chip"
    else:
        action_verb = "running on local Aer (no chip submission)"
    status(f"{backend_str}: {action_verb} (depth={qc_t.depth()}, "
           f"ops={dict(qc_t.count_ops())}, shots={actual_shots})...")

    # Retry up to MAX_RETRIES times on transient failures (CANCELED, ERROR,
    # ConnectionError when chip flips to NOTREADY mid-batch). Real-hardware
    # queues are noisy; a single bad outcome should not abort a long benchmark.
    MAX_RETRIES = 2
    last_err: Exception | None = None
    job = None
    queue_s = exec_s = 0.0
    for attempt in range(MAX_RETRIES + 1):
        _check_cancel()
        try:
            t_submit_start = time.perf_counter()
            job = submit(backend, qc_t, shots_exponent=shots_exponent,
                         project=project or f"qccb_{label}")
            timing.submit_s = time.perf_counter() - t_submit_start
            # Register this job so a Stop click can cancel it server-side
            _set_current_remote_job(client, job.id)
            status(f"{backend_str}: waiting for job to complete..."
                   + (f" (retry {attempt}/{MAX_RETRIES})" if attempt else ""))
            job, queue_s, exec_s = _poll_with_timing(
                client, job, status, backend_str
            )
            break
        except CancelledError:
            raise
        except Exception as e:
            last_err = e
            status(f"{backend_str}: attempt {attempt + 1} failed: {e}")
            # CRITICAL: cancel the orphan server-side job before retrying.
            # See `_run_octillion_batched` for full rationale (Bug A fix).
            if job is not None and job.id is not None:
                _try_cancel_remote_job(client, job.id, status, backend_str)
            if attempt < MAX_RETRIES and real_hardware:
                # The chip likely flipped to NOTREADY mid-batch (it cancels
                # in-flight jobs and rejects new submissions while cooling).
                # Wait for it to recover before the next attempt.
                try:
                    _wait_for_chip_ready(client, chip_name, status, backend_str)
                except CancelledError:
                    raise
                except RuntimeError:
                    # Stayed NOTREADY too long — give up
                    raise last_err
                _sleep_with_cancel(2 + attempt)
                continue
            if attempt < MAX_RETRIES:
                _sleep_with_cancel(2 + attempt)
                continue
            raise
        finally:
            _set_current_remote_job(client, None)
    timing.queue_s = queue_s
    timing.execute_s = exec_s

    # === Phase: Python post-processing (counts → bitstring → metric) ===
    t = time.perf_counter()
    counts = normalize_counts(job.counts, total_shots=actual_shots)
    metric_value = float(metric_fn(counts)) if metric_fn else 0.0
    timing.python_post_s = time.perf_counter() - t

    timing.total_s = time.perf_counter() - t_overall

    extra: dict[str, Any] = {
        "transpiled_depth": qc_t.depth(),
        "transpiled_ops": dict(qc_t.count_ops()),
        "metric_name": metric_name,
        "metric_value": metric_value,
        "expected_distribution": expected_distribution or {},
        "timing_s": timing.to_dict(),
    }
    save_path = save_run(job, label=f"{label}_{backend_str}",
                         total_shots=actual_shots, extra=extra)

    res = ExperimentResult(
        label=label, backend=backend_str,
        shots=actual_shots, counts=counts,
        job_id=str(job.id) if job.id else None,
        transpiled_depth=qc_t.depth(),
        transpiled_ops={k: int(v) for k, v in qc_t.count_ops().items()},
        chip_spec=spec,
        metric_name=metric_name, metric_value=metric_value,
        expected_distribution=expected_distribution or {},
        save_path=save_path,
        timing=timing,
    )
    status(
        f"{backend_str}: done, {metric_name}={metric_value:.4f}  "
        f"(total {timing.total_s*1000:.0f}ms; algo {timing.algorithmic_s*1000:.0f}ms; "
        f"py-overhead {timing.python_overhead_pct:.1f}%; "
        f"queue {timing.queue_s*1000:.0f}ms · exec {timing.execute_s*1000:.0f}ms)"
        if metric_name else
        f"{backend_str}: done ({timing.total_s:.2f}s, py-overhead {timing.python_overhead_pct:.1f}%)"
    )
    return res


def _poll_with_timing(client, job, status: StatusCallback, backend_str: str,
                      poll_interval: float = 2.0, timeout: float = 600.0,
                      max_consecutive_api_errors: int = 10,
                      ) -> tuple[Any, float, float]:
    """
    Poll job until COMPLETE. Returns (final_job, queue_seconds, execute_seconds).

    Failure modes handled:
      - Transient API blips (single connection drop): suppressed, log once,
        keep polling — usually clears within a few seconds.
      - Sustained API outage (`max_consecutive_api_errors` blips in a row):
        bail out fast with ConnectionError so the outer retry loop can
        cancel the orphan server-side job and submit a fresh batch.
        Without this short-circuit, a 30-second Bauman network glitch would
        consume the full `timeout` budget (300+ s for batched jobs) before
        the retry mechanism gets control.
      - Job-side failure (CANCELED/FAILED status): raise immediately.
      - Hard timeout (no progress for `timeout` s with API responsive):
        raise TimeoutError.
    """
    if job.id is None:
        return job, 0.0, 0.0

    from octillion import Job as OJob

    t_start = time.perf_counter()
    t_exec_start: float | None = None
    last_status = None
    deadline = time.time() + timeout
    consecutive_errors = 0

    queue_s = 0.0
    execute_s = 0.0

    while time.time() < deadline:
        _check_cancel()
        try:
            job_data = client._api.job(job.id)
            consecutive_errors = 0   # success — reset the counter
        except Exception as e:
            consecutive_errors += 1
            status(f"{backend_str}: API error {e!r} "
                   f"(consecutive {consecutive_errors}/{max_consecutive_api_errors})")
            if consecutive_errors >= max_consecutive_api_errors:
                # Bauman API is genuinely down — bail out so the outer retry
                # loop can cancel the orphan job and re-submit. The escape is
                # ConnectionError (not TimeoutError) so callers can distinguish
                # "network outage" from "chip stuck on a slow job".
                raise ConnectionError(
                    f"Octillion API unreachable: {max_consecutive_api_errors} "
                    f"consecutive errors polling job {job.id} (last: {e!r})"
                ) from e
            time.sleep(poll_interval)
            continue

        st = str(job_data.get("status", "UNKNOWN")).upper()
        if st != last_status:
            status(f"{backend_str}: job {job.id} status={st}")
            if st == "EXECUTING" and t_exec_start is None:
                queue_s = time.perf_counter() - t_start
                t_exec_start = time.perf_counter()
            last_status = st

        if st in ("COMPLETE", "DONE", "FINISHED", "SUCCESS"):
            if t_exec_start is not None:
                execute_s = time.perf_counter() - t_exec_start
            else:
                queue_s = time.perf_counter() - t_start
            return OJob(
                id=job_data.get("batch_id", job.id),
                status=job_data.get("status"),
                shots=job_data.get("shots", 0) or 0,
                counts=job_data.get("counts", []) or [],
            ), queue_s, execute_s

        if st in ("FAILED", "ERROR", "CANCELLED", "CANCELED"):
            raise RuntimeError(f"Job {job.id} ended with status={st}")
        time.sleep(poll_interval)
    raise TimeoutError(f"Job {job.id} timeout (last status={last_status})")


def _run_octillion_batched(
    circuit, base_label: str, shots_exponent: int, actual_shots: int,
    n_repeats: int, real_hardware: bool,
    expected_distribution, metric_name: str, metric_fn,
    project: str | None, status: "StatusCallback",
    chip_name: str = "Snowdrop 4q ver2",
) -> list[ExperimentResult]:
    """
    Batched real-hardware submission.

    Sends N identical copies of one circuit to Octillion as a SINGLE
    `backend.run([qc] * N, shots=2^shots_exponent, project=...)` call. The
    chip executes the circuits serially (verified empirically: 3-Bell batch
    took 3.29× single-Bell wall-clock) but the batch counts as ONE job
    server-side, which is the main win:

      - 1 queue wait instead of N (each ~0.5 s warm / 3 s cold)
      - 1 cooldown after the batch instead of N-1 cooldowns inside (~6 s each)
      - 1 chip-protection trigger surface instead of N (10× fewer thermal
        events per config)
      - 1 server-side job_id → 1 Stop click cancels the whole batch
      - 1 pre-flight envelope check covers all repeats (same circuit)

    Returns a `list[ExperimentResult]` of length ≤ N. Length < N only on
    partial batch responses (chip canceled mid-batch).

    Per-repeat timing attribution:
      - python_setup_s, transpile_s, submit_s, queue_s:  total / N
        (these are shared overheads paid once for the batch)
      - execute_s: total execute / N (chip serialises inside the batch)
      - python_post_s: per-repeat (each result normalises its own counts)
    """
    if not real_hardware:
        raise ValueError(
            "_run_octillion_batched is real-hw only; use the loop path for emulator"
        )
    if n_repeats < 1:
        return []

    backend_str = "real"
    t_overall = time.perf_counter()
    _check_cancel()

    # === Phase: Python setup (cached client + backend) ===
    t = time.perf_counter()
    status(f"{backend_str}: preparing batched submission ({n_repeats}× "
           f"{base_label}) to {chip_name}...")
    client = get_client()
    cache_key = f"{int(real_hardware)}::{chip_name}"
    backend = _get_cached_backend(client, real_hardware, chip_name)
    spec = _get_cached_spec(backend, cache_key)
    setup_s = time.perf_counter() - t
    _check_cancel()

    # Chip readiness gate (same as single-circuit path)
    try:
        from src.quantum_hardware import get_chip_queue_info
        qinfo = get_chip_queue_info(client, chip_name)
        qstate = qinfo.get("status", "?")
        qlen = qinfo.get("queue")
        qlen_txt = f"{qlen}" if qlen is not None else "n/a"
        status(f"{backend_str}: chip {qstate}, queue={qlen_txt} jobs ahead "
               f"(batch size = {n_repeats})")
        if not qinfo.get("ready"):
            _wait_for_chip_ready(client, chip_name, status, backend_str)
            _check_cancel()
    except CancelledError:
        raise
    except RuntimeError:
        raise
    except Exception as exc:
        # Pre-flight readiness probe is best-effort: a failed query should
        # not abort the benchmark — backend.run() will surface real errors.
        status(f"{backend_str}: chip status query failed ({exc!r}), "
               f"continuing anyway")

    # === Phase: Transpile (once — all N copies are identical) ===
    # seed_transpiler is pinned for deterministic routing across the whole
    # benchmark session; without it, Sabre's randomness produces different
    # final_index_layout per call and the bit-permutation fix below would
    # be applied with the wrong permutation.
    status(f"{backend_str}: transpiling for chip basis {spec.basis_gates}...")
    t = time.perf_counter()
    qc_t = transpile(
        circuit,
        basis_gates=spec.basis_gates,
        coupling_map=[list(p) for p in spec.coupling_map],
        layout_method="trivial",
        routing_method="sabre",
        optimization_level=1,
        seed_transpiler=42,
    )
    transpile_s = time.perf_counter() - t
    batch_circuits = [qc_t] * n_repeats
    # NOTE: bitstrings come back from Bauman in logical c[]-order already
    # (verified by X(q0) probe); seed_transpiler=42 above is the sole BV-fix.

    status(f"{backend_str}: submitting batch of {n_repeats} circuits "
           f"(depth={qc_t.depth()}, ops={dict(qc_t.count_ops())}, "
           f"shots={actual_shots} per circuit, project={project or 'qccb'})")

    # === Phase: Submit + poll ===
    MAX_RETRIES = 2
    last_err: Exception | None = None
    job = None
    submit_s = 0.0
    queue_s = 0.0
    exec_s = 0.0
    for attempt in range(MAX_RETRIES + 1):
        _check_cancel()
        try:
            t_submit_start = time.perf_counter()
            job = submit(
                backend, batch_circuits, shots_exponent=shots_exponent,
                project=project or f"qccb_{base_label}_batch",
            )
            submit_s = time.perf_counter() - t_submit_start
            _set_current_remote_job(client, job.id)
            status(f"{backend_str}: batch job_id={job.id}; waiting for "
                   f"COMPLETE..." + (
                       f" (retry {attempt}/{MAX_RETRIES})" if attempt else ""
                   ))
            job, queue_s, exec_s = _poll_with_timing(
                client, job, status, backend_str,
                # Batch execute time scales ~linearly with N; widen the timeout
                # to N × single-circuit budget so we don't false-cancel a healthy
                # large batch.
                timeout=max(600.0, 30.0 * n_repeats),
            )
            break
        except CancelledError:
            raise
        except Exception as e:
            last_err = e
            status(f"{backend_str}: batch attempt {attempt + 1} failed: {e}")
            # CRITICAL: cancel the orphan server-side job before retrying.
            # If we skipped this, the OLD batch would continue EXECUTING on
            # the chip (consuming queue+execute slots) while we submit a NEW
            # batch behind it — doubling chip-time spent and possibly causing
            # cascading queue=1 contention on subsequent configs.
            # See incident with job 63e4d3bc-… (BV s=101, 14 May 2026).
            if job is not None and job.id is not None:
                _try_cancel_remote_job(client, job.id, status, backend_str)
            if attempt < MAX_RETRIES:
                try:
                    _wait_for_chip_ready(client, chip_name, status, backend_str)
                except CancelledError:
                    raise
                except RuntimeError:
                    raise last_err
                _sleep_with_cancel(2 + attempt)
                continue
            raise
        finally:
            _set_current_remote_job(client, None)

    total_s = time.perf_counter() - t_overall

    # === Phase: Post-process — explode batch counts into N results ===
    t_post_start = time.perf_counter()
    counts_per_repeat = normalize_counts_list(
        job.counts, total_shots=actual_shots
    )
    if len(counts_per_repeat) == 0:
        status(f"{backend_str}: batch returned 0 results (chip refused) — "
               f"treating as full-batch failure")
        raise RuntimeError(
            f"Batch job {job.id} returned no counts (likely partial CANCEL)"
        )
    if len(counts_per_repeat) < n_repeats:
        status(f"{backend_str}: batch returned only "
               f"{len(counts_per_repeat)}/{n_repeats} results — chip canceled "
               f"the tail of the batch; keeping the {len(counts_per_repeat)} "
               f"that came back")

    # Save the FULL batch payload once (one JSON per batch is cleaner than
    # N near-identical files; per-repeat ExperimentResults all point to it).
    batch_label = f"{base_label}_batch_x{n_repeats}_{backend_str}"
    save_extra: dict[str, Any] = {
        "transpiled_depth": qc_t.depth(),
        "transpiled_ops": dict(qc_t.count_ops()),
        "metric_name": metric_name,
        "expected_distribution": expected_distribution or {},
        "n_circuits": n_repeats,
        "n_results": len(counts_per_repeat),
        "per_circuit_counts_int": counts_per_repeat,
        "timing_s": {
            "python_setup_s": setup_s,
            "transpile_s": transpile_s,
            "submit_s": submit_s,
            "queue_s": queue_s,
            "execute_s": exec_s,
            "total_s": total_s,
        },
    }
    save_path = save_run(
        job, label=batch_label, total_shots=actual_shots, extra=save_extra
    )

    # Per-repeat timing attribution — share fixed overheads, divide chip-time
    n_returned = len(counts_per_repeat)
    per_setup = setup_s / n_returned
    per_transpile = transpile_s / n_returned
    per_submit = submit_s / n_returned
    per_queue = queue_s / n_returned
    per_execute = exec_s / n_returned

    results: list[ExperimentResult] = []
    for i, counts in enumerate(counts_per_repeat):
        metric_value = float(metric_fn(counts)) if metric_fn else 0.0
        t_post = time.perf_counter() - t_post_start  # cumulative; small
        per_post = t_post / max(n_returned, 1)
        per_total = (per_setup + per_transpile + per_submit + per_queue
                      + per_execute + per_post)

        timing = TimingBreakdown(
            python_setup_s=per_setup,
            python_post_s=per_post,
            transpile_s=per_transpile,
            submit_s=per_submit,
            queue_s=per_queue,
            execute_s=per_execute,
            total_s=per_total,
        )
        res = ExperimentResult(
            label=f"{base_label}_bench_{i+1}",
            backend=backend_str,
            shots=actual_shots,
            counts=counts,
            job_id=f"{job.id}::{i}" if job.id else None,
            transpiled_depth=qc_t.depth(),
            transpiled_ops={k: int(v) for k, v in qc_t.count_ops().items()},
            chip_spec=spec,
            metric_name=metric_name,
            metric_value=metric_value,
            expected_distribution=expected_distribution or {},
            save_path=save_path,
            timing=timing,
        )
        results.append(res)

    status(
        f"{backend_str}: batch done, {n_returned}/{n_repeats} circuits "
        f"returned. wall-clock={total_s:.1f}s "
        f"(queue {queue_s:.1f}s, execute {exec_s:.1f}s, "
        f"per-repeat ≈ {per_total*1000:.0f}ms attributed)"
    )
    return results


def predict_real_hw_fidelity(qc_transpiled, chip_spec) -> float:
    """
    Coarse pre-submit estimate of end-to-end success probability for a circuit
    on the live chip. Returns a number in (0, 1].

    Multiplicative noise model:
        F_predicted = Π F_1q^n_1q · Π F_2q^n_2q
                     · exp(-t_gates / T2_avg)
                     · Π F_RO^n_qubits

    Notes:
      - We use the WORST T2 across qubits. The chip's autonomous calibration
        protection trips on the worst qubit's behaviour, not the average.
        Using avg-T2 would let GHZ-4 (depth=20, 9 CZ) through even though
        empirically Snowdrop 4q ver2 cancels that circuit at high rate.
      - t_gates excludes readout time: measurement collapses superposition,
        so the readout duration doesn't accumulate T2 decay on the
        measured qubits.
      - Readout error is folded in as a product term F_RO^n_qubits.

    Calibrated against Snowdrop 4q ver2:
       Bell  (depth 10, 4 CZ):  empirical ≈ 0.92, F_pred ≈ 0.62
       GHZ-3 (depth 12, 5 CZ):  empirical ≈ 0.86, F_pred ≈ 0.50
       GHZ-4 (depth 20, 9 CZ):  empirical ≈ 0.75, F_pred ≈ 0.30
                                                  ↑ at-risk, auto-skip
    """
    import math as _math
    if chip_spec is None:
        return 1.0
    ops = dict(qc_transpiled.count_ops())
    n_1q = sum(v for k, v in ops.items()
                if k in ("h", "rx", "ry", "rz", "u", "u3", "id", "x", "y", "z", "s", "t"))
    n_2q = sum(v for k, v in ops.items() if k in ("cz", "cx", "swap", "iswap"))
    n_meas = sum(v for k, v in ops.items() if k == "measure")
    n_qubits_used = max(n_meas, 1)

    f_1q = float(getattr(chip_spec, "avg_f1q", 0.999))
    f_2q = float(getattr(chip_spec, "avg_f2q", 0.99))
    f_ro = float(getattr(chip_spec, "avg_ro", 0.95))

    # Gate-application time only — readout collapses the state, no T2 decay
    # after measurement on measured qubits
    gate_t_1q_ns = float(getattr(chip_spec, "gate_length_1q_ns", 30.0))
    gate_t_2q_ns = float(getattr(chip_spec, "gate_length_2q_ns", 100.0))
    t_gates_us = (n_1q * gate_t_1q_ns + n_2q * gate_t_2q_ns) / 1000.0

    # Worst T2 across qubits — the chip's autonomous protection trips on
    # whichever qubit is decohering fastest, not the average.
    per_qubit_t2 = getattr(chip_spec, "per_qubit_t2_us", {}) or {}
    if per_qubit_t2:
        t2_worst = max(0.5, min(float(v) for v in per_qubit_t2.values()))
    else:
        t2_worst = float(getattr(chip_spec, "avg_t2_us", 7.0)) or 7.0

    gate_term = (f_1q ** n_1q) * (f_2q ** n_2q)
    ro_term = f_ro ** n_qubits_used
    decoh_term = _math.exp(-t_gates_us / t2_worst)
    return float(gate_term * ro_term * decoh_term)


# Session-level memory: (exp_key, params_json, backend) → number of times the
# chip CANCELED this exact config in current benchmark session. After the
# budget is exceeded we skip the remaining repeats — pushing the same
# circuit a second/third time only causes the chip to enter a longer
# service window. Budget = 0 means "one CANCEL is enough to give up on
# this config for the rest of the session".
_canceled_config_counts: dict[tuple[str, str, str], int] = {}
_CANCEL_BUDGET: int = 0


def benchmark(
    exp,
    backend_kind: str,
    params: dict,
    shots_exponent: int,
    repeats: int,
    status: StatusCallback | None = None,
    on_run_complete: Callable[[int, ExperimentResult], None] | None = None,
    skip_real_if_predicted_f_below: float = 0.45,
    bypass_envelope: bool = False,
) -> BenchmarkResult:
    """
    Run an experiment N times to gather statistical fidelity + timing data.
    Individual runs may fail on real hardware (canceled jobs, queue timeouts,
    network blips); failed runs are skipped and reported but do not abort
    the full benchmark.

    For real-hardware runs two extra guards:

      (a) Pre-submit fidelity guard. We estimate the circuit's expected
          end-to-end fidelity from chip calibration (gate F + worst-qubit T2
          + readout F). If it falls below `skip_real_if_predicted_f_below`
          we skip the config entirely — pushing a hopeless circuit to the
          chip only triggers its calibration-protection and stalls the run.

      (b) Session-level CANCELED tracker. If the chip itself canceled the
          same (exp, params, backend) earlier in this session, we don't
          re-submit. Two CANCELs in a row mean the circuit is genuinely
          beyond what this chip can do today.
    """
    import json as _json
    status = status or _noop
    runs: list[ExperimentResult] = []
    failures: list[str] = []

    # --- Guard A: predicted fidelity for real-hw configs ----------------
    if backend_kind == "real" and skip_real_if_predicted_f_below > 0:
        try:
            client = get_client()
            chip_pre = "Snowdrop 4q ver2"
            backend_obj = _get_cached_backend(
                client, real_hardware=True, chip=chip_pre,
            )
            spec_pre = _get_cached_spec(backend_obj, f"1::{chip_pre}")
            qc_probe = exp.build(params)
            qc_probe_t = transpile(
                qc_probe, basis_gates=spec_pre.basis_gates,
                coupling_map=[list(p) for p in spec_pre.coupling_map],
                layout_method="trivial", routing_method="sabre",
                optimization_level=1,
            )
            f_pred = predict_real_hw_fidelity(qc_probe_t, spec_pre)
            ops_str = dict(qc_probe_t.count_ops())
            status(f"benchmark: pre-submit fidelity estimate for {exp.key} on "
                   f"real ≈ {f_pred:.3f} (depth={qc_probe_t.depth()}, "
                   f"ops={ops_str})")

            # Additional structural envelope check (Bauman api_example.ipynb
            # + Habr Dec-2025 specs + empirical CANCEL incidents on this chip)
            try:
                from src.snowdrop_constraints import (
                    classify_circuit_risk, OPERATING_ENVELOPE,
                )
                n_1q = sum(v for k, v in ops_str.items()
                            if k in ("rx", "ry", "rz", "h", "u", "u3",
                                      "id", "x", "y", "z"))
                n_2q = sum(v for k, v in ops_str.items()
                            if k in ("cz", "cx", "swap", "iswap"))
                n_m = ops_str.get("measure", 0)
                risk = classify_circuit_risk(
                    transpiled_depth=qc_probe_t.depth(),
                    n_1q_gates=n_1q, n_2q_gates=n_2q, n_measures=n_m,
                    chip_spec=spec_pre,
                )
                status(f"benchmark: envelope check → {risk.risk_level.upper()} "
                       f"({'; '.join(risk.reasons)})")
                if risk.risk_level == "red":
                    if bypass_envelope:
                        status(f"benchmark: ⚠ envelope-red BYPASSED "
                               f"(force-run requested) — chip may CANCEL "
                               f"the batch; data quality not guaranteed")
                    else:
                        status(f"benchmark: ⚠ envelope-red — skipping all "
                               f"{repeats} repeats on real to spare the chip")
                        return BenchmarkResult(
                            label=exp.key, backend=backend_kind,
                            shots_per_run=2 ** shots_exponent, repeats=0,
                            runs=[],
                        )
            except Exception as e:
                status(f"benchmark: envelope check unavailable ({e!r})")

            if f_pred < skip_real_if_predicted_f_below:
                msg = (f"predicted fidelity {f_pred:.3f} below threshold "
                       f"{skip_real_if_predicted_f_below:.2f} — circuit is "
                       f"too deep for this chip's worst-qubit T2")
                if bypass_envelope:
                    status(f"benchmark: ⚠ {msg} — BYPASSED (force-run); "
                           f"expect noisy/degenerate counts")
                else:
                    status(f"benchmark: ⚠ {msg}; skipping "
                           f"all {repeats} repeats on real to spare the chip")
                    return BenchmarkResult(
                        label=exp.key, backend=backend_kind,
                        shots_per_run=2 ** shots_exponent, repeats=0, runs=[],
                    )
        except CancelledError:
            raise
        except Exception as e:
            status(f"benchmark: pre-submit guard skipped ({e!r}) — proceeding")

    config_key = (exp.key, _json.dumps(params, sort_keys=True), backend_kind)

    # =====================================================================
    # BATCHED real-hw path (Phase 2 refactor — Bauman Octillion supports
    # `backend.run([qc1, qc2, qc3], shots=10, project=...)` natively, see
    # `Bauman/api_example.ipynb` and the Phase-1 smoke test in
    # `tools/test_batch_api.py`). For N ≥ 3 repeats on real hardware, this
    # is strictly cheaper:
    #
    #   - 1 queue wait & 1 chip-protection trigger surface vs N
    #   - 0 cool-downs INSIDE the config (cool-down kept BETWEEN configs)
    #   - 1 server-side job_id → 1 Stop click cancels the whole batch
    #
    # For repeats < 3 or non-real backends, the historical loop path
    # is preserved (the batched path's overhead doesn't pay off for tiny N
    # and emulator/ideal/gpu paths are already microseconds-cheap).
    # =====================================================================
    use_batched = (
        backend_kind == "real" and repeats >= 3 and not is_cancel_requested()
    )

    if use_batched:
        status(f"benchmark: real-hw batched submission ({repeats} circuits "
               f"in one Octillion job)")
        qc = exp.build(params)
        try:
            batch_results = _run_octillion_batched(
                circuit=qc,
                base_label=exp.key,
                shots_exponent=shots_exponent,
                actual_shots=2 ** shots_exponent,
                n_repeats=repeats,
                real_hardware=True,
                expected_distribution=exp.expected(params),
                metric_name=exp.metric_name,
                metric_fn=lambda c: exp.metric_fn(c, params),
                project=None,
                status=status,
                chip_name="Snowdrop 4q ver2",
            )
        except CancelledError:
            status(f"benchmark: cancelled during batched submission")
            batch_results = []
        except Exception as e:
            failures.append(f"batch: {type(e).__name__}: {e}")
            status(f"benchmark: batched submission FAILED ({e})")
            if "CANCELED" in str(e).upper() or "CANCELLED" in str(e).upper():
                # Chip canceled the whole batch — record once
                _canceled_config_counts[config_key] = (
                    _canceled_config_counts.get(config_key, 0) + 1
                )
            batch_results = []

        for i, res in enumerate(batch_results):
            runs.append(res)
            status(f"benchmark: batch repeat {i+1} → "
                   f"{exp.metric_name}={res.metric_value:.4f}")
            if on_run_complete:
                on_run_complete(i, res)
    else:
        # ----- Original per-repeat loop (kept for emulator / ideal / gpu
        # and for tiny repeats counts where batching has no payoff) -----
        for i in range(repeats):
            if is_cancel_requested():
                status(f"benchmark: stop requested — aborting after {len(runs)}/{repeats} runs")
                break

            # --- Guard B: skip remaining repeats if chip CANCELED this
            # config earlier in this session. Pushing the same circuit to
            # the chip a 2nd time only causes a longer service window.
            already_canceled = _canceled_config_counts.get(config_key, 0)
            if backend_kind == "real" and already_canceled > _CANCEL_BUDGET:
                status(f"benchmark: chip has already CANCELED this config "
                       f"{already_canceled}× in this session — skipping remaining "
                       f"{repeats - i} repeats to spare the chip")
                break

            status(f"benchmark: run {i+1}/{repeats}...")
            qc = exp.build(params)
            try:
                res = run_circuit(
                    qc,
                    label=f"{exp.key}_bench_{i+1}",
                    backend_kind=backend_kind,
                    shots_exponent=shots_exponent,
                    expected_distribution=exp.expected(params),
                    metric_name=exp.metric_name,
                    metric_fn=lambda c: exp.metric_fn(c, params),
                    status=status,
                )
            except CancelledError:
                status(f"benchmark: cancelled mid-run ({i+1}/{repeats})")
                break
            except Exception as e:
                failures.append(f"run {i+1}: {type(e).__name__}: {e}")
                status(f"benchmark: run {i+1} FAILED ({e}) — skipping and continuing")
                if "CANCELED" in str(e).upper():
                    _canceled_config_counts[config_key] = (
                        _canceled_config_counts.get(config_key, 0) + 1
                    )
                continue

            runs.append(res)
            status(f"benchmark: run {i+1} → {exp.metric_name}={res.metric_value:.4f}")
            if on_run_complete:
                on_run_complete(i, res)

            # Polite cool-down between consecutive real-hardware submissions.
            # In batched mode this is unnecessary (one submission per config).
            if backend_kind == "real" and i < repeats - 1:
                try:
                    _sleep_with_cancel(_REAL_HW_COOLDOWN_S)
                except CancelledError:
                    status("benchmark: stop requested during cool-down — aborting")
                    break

    if failures:
        status(f"benchmark: {len(failures)} failures during run: {failures}")

    return BenchmarkResult(
        label=exp.key,
        backend=backend_kind,
        shots_per_run=2 ** shots_exponent,
        repeats=len(runs),
        runs=runs,
    )

