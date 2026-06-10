"""
Bauman Octillion Snowdrop 4q integration.

Wraps the octillion_client to provide:
- Token loading from token.env (read at runtime, never logged)
- Backend selection (real hardware vs local emulator)
- Calibration JSON loader (for offline noise-model construction)
- Backend introspection: chip parameters, fidelities, T1/T2

Notes on the octillion_client API (verified against installed v1.0.9):
- Client.local(model_name)  → Aer-based simulator with chip noise model (synchronous)
- Client.remote(model_name) → submits to physical chip (asynchronous, must poll)
- Backend properties: basis_gates, coupling_map, num_qubits  (no parens)
- Backend methods: fidelity(gate, qubits), t1(q), t2(q), f_q(q), readout_fidelity(q),
                   gate_length(gate, qubits), readout_length(q)
- Job properties: id, status, counts  (no parens)
- shots parameter is an EXPONENT of 2 (shots=10 → 2^10 = 1024 actual runs)
"""
from __future__ import annotations

import json
import os
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from octillion.client import Client


CHIP_NAME = "Snowdrop 4q ver2"   # default chip used by single-arg APIs
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOKEN_FILE = PROJECT_ROOT / "token.env"
CALIBRATION_FILE = PROJECT_ROOT / "Bauman" / "snowdrop_4q_ver2.json"


# All Bauman chips reachable via the Octillion API (visible on
# https://octillion.bmstu.ru/qpus). Add new ones as the platform expands.
BAUMAN_CHIPS: dict[str, dict] = {
    "Snowdrop 4q ver2":  {"qubits": 4, "alias": "snowdrop_4q_v2"},
    "Snowdrop 4q ver1":  {"qubits": 4, "alias": "snowdrop_4q_v1"},
    "Snowdrop 8q ver1":  {"qubits": 8, "alias": "snowdrop_8q_v1"},
}


def _load_token() -> str:
    """Read API token from env var or token.env file."""
    token = os.getenv("BAUMAN_OCTILLION_TOKEN")
    if token:
        return token.strip()

    if not TOKEN_FILE.exists():
        raise FileNotFoundError(
            f"Token not found. Set BAUMAN_OCTILLION_TOKEN env var "
            f"or place token in {TOKEN_FILE}"
        )

    text = TOKEN_FILE.read_text(encoding="utf-8").strip()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            if key.strip().upper() in (
                "BAUMAN_OCTILLION_TOKEN", "TOKEN", "OCTILLION_TOKEN"
            ):
                return value.strip().strip('"').strip("'")
        else:
            return line
    raise ValueError(f"Could not parse token from {TOKEN_FILE}")


def load_calibration() -> dict[str, Any]:
    """Load chip calibration JSON (for offline analysis / noise model construction)."""
    if not CALIBRATION_FILE.exists():
        raise FileNotFoundError(
            f"Calibration not found at {CALIBRATION_FILE}. "
            f"Download from https://octillion.bmstu.ru/qpus"
        )
    return json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))


_cached_client: Client | None = None


def get_client() -> Client:
    """
    Initialize (or return cached) Octillion client.

    Cached at module level so the per-run hot path doesn't re-build
    Client and invalidate downstream backend caches.
    """
    global _cached_client
    if _cached_client is None:
        _cached_client = Client(_load_token())
    return _cached_client


def reset_client_cache() -> None:
    """Drop the cached client (e.g. after a token change)."""
    global _cached_client
    _cached_client = None


def get_backend(client: Client, real_hardware: bool = False, chip: str = CHIP_NAME):
    """
    Get a backend handle.

    real_hardware=True  → submits jobs to physical chip
    real_hardware=False → local emulator (Aer + chip's noise model from calibration)
    """
    if real_hardware:
        return client.remote(chip)
    return client.local(chip)


def get_chip_queue_info(client: Client, chip: str = CHIP_NAME) -> dict[str, Any]:
    """
    Best-effort status snapshot for a chip without instantiating a backend.

    Octillion API as of v1.0.9 does not expose a queue-length field; this
    helper returns the closest equivalents:
      - status         : READY / NOTREADY / unknown — the only authoritative
                         signal for "can I submit a job right now"
      - is_maintenance : True if Bauman has flagged the chip for a scheduled
                         service window. Informational ONLY — chips in this
                         state still accept jobs (verified empirically: the
                         Bauman web UI shows "Готов" with is_maintenance=True).
      - virtual        : whether this is the emulator entry
      - queue          : numeric queue length if a future API version starts
                         exposing it; None today
      - ready          : True iff status==READY. Mirrors the green "Готов"
                         badge on the Bauman web UI.

    Never raises — on any API failure returns a dict with 'error' set.
    """
    info: dict[str, Any] = {
        "name": chip,
        "status": "unknown",
        "queue": None,
        "is_maintenance": None,
        "virtual": None,
        "ready": False,
        "error": None,
    }
    try:
        chips = client._api.chips(model_name=chip)
        real = next((c for c in chips if not c.get("virtual", False)), None)
        if real is None:
            info["error"] = f"No non-virtual chip named {chip!r}"
            return info
        info["status"] = real.get("chip_status", "unknown")
        info["is_maintenance"] = bool(real.get("is_maintenance", False))
        info["virtual"] = bool(real.get("virtual", False))
        info["ready"] = (info["status"] == "READY")

        # Queue length lives under GET /api/chip/<id> (without /raw).
        # The octillion library's Api.chip() hits /raw, so we re-issue the
        # request to the non-raw endpoint to pick up the live `queue` field.
        chip_id = real.get("id")
        if chip_id:
            # Queue length is a nice-to-have — any HTTP/parse failure just
            # leaves info["queue"] = None and the fallback below kicks in.
            with suppress(Exception):
                import requests
                api = client._api
                r = requests.get(
                    f"{api._api_url}/chip/{chip_id}",
                    headers=api._auth_header, verify=False, timeout=5,
                )
                if r.status_code == 200:
                    body = r.json()
                    if "queue" in body:
                        info["queue"] = body["queue"]
        # Fallback: maybe a future API version exposes it on chips() too
        if info["queue"] is None:
            for k in ("queue", "queue_length", "jobs_in_queue", "queue_size"):
                if k in real:
                    info["queue"] = real[k]
                    break
    except Exception as e:
        info["error"] = repr(e)
    return info


@dataclass
class ChipSpec:
    """Snapshot of chip parameters for reporting."""
    name: str
    num_qubits: int
    basis_gates: list[str]
    coupling_map: list[tuple[int, int]]
    per_qubit_f1q: dict[int, float] = field(default_factory=dict)
    per_qubit_ro: dict[int, float] = field(default_factory=dict)
    per_qubit_t1_us: dict[int, float] = field(default_factory=dict)
    per_qubit_t2_us: dict[int, float] = field(default_factory=dict)
    per_qubit_freq_ghz: dict[int, float] = field(default_factory=dict)
    per_pair_f2q: dict[tuple[int, int], float] = field(default_factory=dict)

    @property
    def avg_f1q(self) -> float:
        v = list(self.per_qubit_f1q.values())
        return sum(v) / len(v) if v else 0.0

    @property
    def avg_f2q(self) -> float:
        v = list(self.per_pair_f2q.values())
        return sum(v) / len(v) if v else 0.0

    @property
    def avg_ro(self) -> float:
        v = list(self.per_qubit_ro.values())
        return sum(v) / len(v) if v else 0.0

    @property
    def avg_t1_us(self) -> float:
        v = list(self.per_qubit_t1_us.values())
        return sum(v) / len(v) if v else 0.0

    @property
    def avg_t2_us(self) -> float:
        v = list(self.per_qubit_t2_us.values())
        return sum(v) / len(v) if v else 0.0


def describe_chip(backend) -> ChipSpec:
    """Pull chip parameters from a Backend handle (works for local & remote)."""
    n = backend.num_qubits

    coupling = list(backend.coupling_map)
    coupling_pairs = [tuple(p) for p in coupling]
    basis = list(backend.basis_gates)

    spec = ChipSpec(
        name=CHIP_NAME,
        num_qubits=n,
        basis_gates=basis,
        coupling_map=coupling_pairs,
    )

    # Each probe is optional: emulator backends expose only a subset of the
    # introspection API, so a missing/failing accessor just leaves the
    # corresponding spec field unset.
    for q in range(n):
        with suppress(Exception):
            t1_s = backend.t1(q)
            spec.per_qubit_t1_us[q] = float(t1_s) * 1e6
        with suppress(Exception):
            t2_s = backend.t2(q)
            spec.per_qubit_t2_us[q] = float(t2_s) * 1e6
        with suppress(Exception):
            spec.per_qubit_ro[q] = float(backend.readout_fidelity(q))
        with suppress(Exception):
            fq_hz = backend.f_q(q)
            spec.per_qubit_freq_ghz[q] = float(fq_hz) / 1e9
        for g1q in ("rx", "ry", "rz"):
            try:
                spec.per_qubit_f1q[q] = float(backend.fidelity(g1q, q))
                break
            except Exception:
                continue

    seen_pairs = set()
    for pair in coupling_pairs:
        canonical = tuple(sorted(pair))
        if canonical in seen_pairs:
            continue
        seen_pairs.add(canonical)
        with suppress(Exception):
            spec.per_pair_f2q[canonical] = float(backend.fidelity("cz", pair))

    return spec


def submit(backend, circuit, shots_exponent: int = 10,
           project: str | None = None, tags: list[str] | None = None):
    """
    Submit a circuit. shots_exponent=10 → 1024 actual shots.

    For local emulator: returns immediately with completed Job.
    For remote hardware: returns Job with id; use poll_until_done() for results.
    """
    kwargs: dict[str, Any] = {"shots": shots_exponent}
    if project:
        kwargs["project"] = project
    if tags:
        kwargs["tags"] = tags
    return backend.run(circuit, **kwargs)


def poll_until_done(client: Client, job, poll_interval: float = 3.0,
                    timeout: float = 600.0):
    """
    Poll a remote-hardware job until it completes. For local jobs (id=None),
    just returns the job (already complete).

    Note: octillion_client v1.0.9 has a bug where Client.job() raises KeyError
    on 'shots' for in-progress jobs. We bypass it by hitting the raw API.
    """
    if job.id is None:
        return job

    from octillion import Job as OJob

    deadline = time.time() + timeout
    last_status = None
    while time.time() < deadline:
        try:
            job_data = client._api.job(job.id)
        except Exception as e:
            print(f"  [job {job.id}] API error: {e!r} (will retry)")
            time.sleep(poll_interval)
            continue

        status = str(job_data.get("status", "UNKNOWN")).upper()
        if status != last_status:
            print(f"  [job {job.id}] status: {status}")
            last_status = status

        if status in ("COMPLETE", "DONE", "FINISHED", "SUCCESS"):
            return OJob(
                id=job_data.get("batch_id", job.id),
                status=job_data.get("status"),
                shots=job_data.get("shots", 0) or 0,
                counts=job_data.get("counts", []) or [],
            )
        if status in ("FAILED", "ERROR", "CANCELLED", "CANCELED"):
            raise RuntimeError(f"Job {job.id} ended with status={status}")
        time.sleep(poll_interval)
    raise TimeoutError(
        f"Job {job.id} did not complete in {timeout}s (last status={last_status})"
    )


def _permute_bitstring(bitstring: str, final_layout: list[int],
                        n_clbits: int) -> str:
    """
    Re-order a bitstring returned by Bauman's chip in PHYSICAL qubit order
    back into LOGICAL classical-bit order.

    The bug this fixes: when Qiskit's transpile inserts SWAP gates for
    routing (e.g. BV oracle CNOT through Snowdrop's star hub q2), the final
    state of virtual qubit v ends up on some other physical qubit
    p = final_layout[v]. Aer-based backends correctly unscramble this from
    the layout metadata; Bauman's REST API returns raw measurement strings
    in physical-qubit order with no remap. Without this fix, BV s='010'
    on real hardware reports P(correct) ≈ 0.024 (vs ~0.74 on emulator)
    because most shots land on a bit-permuted version of '010'.

    Convention assumed (Qiskit-standard):
      Input "b_{N-1} b_{N-2} ... b_0" is MSB-first physical-qubit ordering,
      i.e. bitstring[0] is the measurement of physical qubit N-1, ...,
      bitstring[N-1] is the measurement of physical qubit 0.

    Output is in MSB-first LOGICAL classical-bit ordering — same as Aer's
    `result.get_counts()` — so the existing metric_fn callbacks
    (`counts.get(secret, 0)`) keep working unchanged.

    Parameters
    ----------
    bitstring : str
        The raw bitstring from the chip (e.g. "100"). May contain spaces
        if the chip uses Qiskit's multi-classical-register separator; those
        are stripped.
    final_layout : list[int]
        `final_layout[v] = p` means virtual qubit v is on physical qubit p
        at the end of the circuit. Length must be ≥ n_clbits.
    n_clbits : int
        Number of classical bits in the original (pre-transpile) circuit.
        Only the first n_clbits virtual qubits are considered measured.

    Returns
    -------
    str
        Permuted bitstring in MSB-first LOGICAL classical-bit ordering,
        length = n_clbits.
    """
    bs = bitstring.replace(" ", "")
    n_phys = len(bs)
    out = []  # MSB-first: virtual q[n_clbits-1], ..., q[0]
    for v in range(n_clbits - 1, -1, -1):
        if v >= len(final_layout):
            out.append("0")
            continue
        p = final_layout[v]
        if 0 <= p < n_phys:
            # Physical qubit p is at position (n_phys - 1 - p) in MSB-first str
            out.append(bs[n_phys - 1 - p])
        else:
            out.append("0")
    return "".join(out)


def _normalize_one_payload(payload, total_shots: int,
                            final_layout: list[int] | None = None,
                            n_clbits: int | None = None) -> dict[str, int]:
    """Normalize a SINGLE counts payload (dict[str, int|float]) to int counts.

    If `final_layout` and `n_clbits` are supplied, each bitstring key is
    permuted from PHYSICAL qubit order (chip's raw output) into LOGICAL
    classical-bit order (what metric_fn expects). Collisions after
    permutation (two physical strings mapping to the same logical string)
    are summed.
    """
    if not isinstance(payload, dict):
        return {}
    values = list(payload.values())
    is_probabilities = (
        bool(values)
        and all(isinstance(v, (int, float)) for v in values)
        and abs(sum(float(v) for v in values) - 1.0) < 0.05
        and any(0 < float(v) < 1 for v in values)
    )

    def _to_int(v) -> int:
        return int(round(float(v) * total_shots)) if is_probabilities else int(v)

    apply_perm = (final_layout is not None and n_clbits is not None
                   and n_clbits > 0)

    if not apply_perm:
        return {str(k): _to_int(v) for k, v in payload.items()}

    out: dict[str, int] = {}
    for k, v in payload.items():
        new_key = _permute_bitstring(str(k), final_layout, n_clbits)
        out[new_key] = out.get(new_key, 0) + _to_int(v)
    return out


def normalize_counts(raw_counts, total_shots: int = 1024,
                      final_layout: list[int] | None = None,
                      n_clbits: int | None = None) -> dict[str, int]:
    """
    Octillion returns counts in different shapes:
      - Local emulator: dict[str, int] of raw counts.
      - Real hardware:  list[dict[str, float]] — probabilities (sum≈1.0), one
                        dict per circuit in the batch.
    For backwards compat with single-circuit callers, when given a list we
    take index 0 (the historical behaviour). Use `normalize_counts_list` for
    the batched path.

    If `final_layout` is supplied (length ≥ n_clbits), each bitstring key
    is permuted from PHYSICAL-qubit order (chip's raw output) into LOGICAL
    classical-bit order — required for any real-hw run that involved
    transpiler-inserted SWAPs (BV oracle, Shor, etc.).
    """
    if raw_counts is None:
        return {}

    payload = raw_counts
    if isinstance(payload, list):
        if not payload:
            return {}
        payload = payload[0]
    return _normalize_one_payload(payload, total_shots, final_layout, n_clbits)


def normalize_counts_list(raw_counts, total_shots: int = 1024,
                           final_layout: list[int] | None = None,
                           n_clbits: int | None = None
                           ) -> list[dict[str, int]]:
    """
    Batched variant of `normalize_counts`. Returns one int-counts dict PER
    circuit in the batch, preserving the order in which circuits were
    submitted via `backend.run([qc0, qc1, ...])`.

    Tolerant of single-dict payloads (returns a 1-element list) so emulator
    and real-hardware paths can share a normalize step.

    See `normalize_counts` for the `final_layout` / `n_clbits` semantics —
    same permutation is applied to every dict in the batch (transpilation
    is identical across batched circuits, so the layout is shared).
    """
    if raw_counts is None:
        return []
    if isinstance(raw_counts, dict):
        # Local Aer path — one circuit, one dict (no batch wrapper)
        return [_normalize_one_payload(raw_counts, total_shots,
                                         final_layout, n_clbits)]
    if isinstance(raw_counts, list):
        return [_normalize_one_payload(p, total_shots, final_layout, n_clbits)
                 for p in raw_counts]
    return []


def save_run(job, label: str, total_shots: int = 1024,
             extra: dict | None = None,
             output_dir: Path | None = None) -> Path:
    """Persist raw counts + metadata for reproducibility."""
    output_dir = output_dir or (PROJECT_ROOT / "Bauman" / "runs")
    output_dir.mkdir(parents=True, exist_ok=True)

    job_id = str(job.id) if job.id else f"local_{int(time.time())}"
    counts = normalize_counts(job.counts, total_shots=total_shots)

    payload = {
        "label": label,
        "job_id": job_id,
        "chip": CHIP_NAME,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": str(job.status) if job.status else "unknown",
        "shots": total_shots,
        "counts_int": counts,
        "counts_raw": job.counts,
    }
    if extra:
        payload.update(extra)

    out = output_dir / f"{label}_{job_id}.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out

