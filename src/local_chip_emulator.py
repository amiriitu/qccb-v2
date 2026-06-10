"""
Local chip emulator built from a saved Bauman calibration JSON.

Why this exists
---------------
`octillion.client.local("<chip>")` is technically "local" (it runs Aer with
the chip's noise model on the user's machine), but it still performs HTTP
GET requests to the Bauman API to fetch the chip image. That:

  • requires network access on every cold cache miss
  • adds ~1 s of "Python setup" per run
  • fails when the chip is offline (NOTREADY) on the platform

This module reads the saved calibration JSON directly from disk
(`Bauman/snowdrop_4q_ver2.json`, etc.), constructs a `qiskit_aer.NoiseModel`
from the per-qubit T1/T2/F_1q/F_RO and per-pair F_cz/gate_length values,
and returns an `AerSimulator` configured with that model plus the chip's
coupling map. No network calls. No dependency on Bauman uptime.

The simulator object exposes the same `.run(circuit, shots=N).result()`
interface as `qiskit_aer.AerSimulator`, so it slots into our runner.py
without changes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Map the user-facing chip name → JSON filename in Bauman/
CHIP_FILES: dict[str, str] = {
    "Snowdrop 4q ver2": "snowdrop_4q_ver2.json",
    "Snowdrop 4q ver1": "snowdrop_4q_ver1.json",
    "Snowdrop 8q ver1": "snowdrop_8q_ver1.json",
}


@dataclass
class LocalChipSpec:
    """Parsed calibration values from a Bauman chip JSON."""
    name: str
    n_qubits: int
    basis_gates: list[str]
    coupling_map: list[tuple[int, int]]
    # Per-qubit
    t1_us: dict[int, float] = field(default_factory=dict)
    t2_us: dict[int, float] = field(default_factory=dict)
    fidelity_1q: dict[int, float] = field(default_factory=dict)
    fidelity_ro: dict[int, float] = field(default_factory=dict)
    freq_ghz: dict[int, float] = field(default_factory=dict)
    gate_length_1q_ns: dict[int, float] = field(default_factory=dict)
    readout_length_us: dict[int, float] = field(default_factory=dict)
    # Per-pair (CZ)
    fidelity_cz: dict[tuple[int, int], float] = field(default_factory=dict)
    gate_length_2q_ns: dict[tuple[int, int], float] = field(default_factory=dict)


def _walk_topology(schema: list) -> list[dict]:
    """Yield each cell's (qubit_dict, row, col, neighbour_directions)."""
    out = []
    for r, row in enumerate(schema):
        for c, cell in enumerate(row):
            if cell is None:
                continue
            out.append((cell, r, c))
    return out


def _coupling_from_schema(schema: list) -> list[tuple[int, int]]:
    """Extract (a,b) pairs from the chip's topology schema."""
    edges: list[tuple[int, int]] = []
    for cell, r, c in _walk_topology(schema):
        q = cell.get("qubit")
        if q is None:
            continue
        for direction, (dr, dc) in (("right", (0, 1)), ("bottom", (1, 0))):
            if direction not in cell:
                continue
            r2, c2 = r + dr, c + dc
            if r2 < len(schema) and c2 < len(schema[r2]):
                other = schema[r2][c2]
                if other is not None and "qubit" in other:
                    edges.append((q, other["qubit"]))
    return edges


def _cz_data_from_schema(schema: list,
                           ) -> dict[tuple[int, int], dict]:
    """Per-pair F_cz and gate length from the topology schema."""
    out: dict[tuple[int, int], dict] = {}
    for cell, r, c in _walk_topology(schema):
        q = cell.get("qubit")
        for direction, (dr, dc) in (("right", (0, 1)), ("bottom", (1, 0))):
            if direction not in cell:
                continue
            r2, c2 = r + dr, c + dc
            if r2 < len(schema) and c2 < len(schema[r2]):
                other = schema[r2][c2]
                if other is None:
                    continue
                q2 = other.get("qubit")
                if q is None or q2 is None:
                    continue
                info = cell[direction]
                pair = tuple(sorted((q, q2)))
                out[pair] = {
                    "fidelity_cz": float(info.get("fidelity_cz", 0)) / 100.0,
                    "gate_length_2q_ns": float(info.get("gate_length_2q", 100)),
                }
    return out


def load_chip_spec(chip_name: str) -> LocalChipSpec:
    """Parse a Bauman chip JSON into a clean LocalChipSpec."""
    fname = CHIP_FILES.get(chip_name)
    if not fname:
        raise ValueError(f"Unknown chip: {chip_name!r}. "
                         f"Known: {list(CHIP_FILES.keys())}")
    path = PROJECT_ROOT / "Bauman" / fname
    if not path.exists():
        raise FileNotFoundError(f"Chip calibration file missing: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))

    schema = raw["topology"]["schema"]
    n = int(raw["model"]["count_qubits"])
    basis = ["id", "h", "rx", "ry", "rz", "cz"]

    spec = LocalChipSpec(name=chip_name, n_qubits=n,
                          basis_gates=basis,
                          coupling_map=_coupling_from_schema(schema))

    # Per-qubit values from cell["volt"] dicts
    for cell, _r, _c in _walk_topology(schema):
        q = cell.get("qubit")
        if q is None:
            continue
        v = cell.get("volt", {})
        spec.t1_us[q] = float(v.get("t1", 0))
        spec.t2_us[q] = float(v.get("t2", 0))
        spec.fidelity_1q[q] = float(v.get("fidelity_1q", 0)) / 100.0
        spec.fidelity_ro[q] = float(v.get("fidelity_ro", 0)) / 100.0
        spec.freq_ghz[q] = float(v.get("f_q", 0))
        spec.gate_length_1q_ns[q] = float(v.get("gate_length_1q", 30))
        spec.readout_length_us[q] = float(v.get("readout_length", 3))

    # Per-pair CZ values
    for pair, info in _cz_data_from_schema(schema).items():
        spec.fidelity_cz[pair] = info["fidelity_cz"]
        spec.gate_length_2q_ns[pair] = info["gate_length_2q_ns"]

    return spec


def build_noise_model(spec: LocalChipSpec):
    """
    Build a Qiskit Aer NoiseModel from a parsed LocalChipSpec.

    Model components per gate:
      - 1-qubit gates  : depolarizing(1 - F_1q) ∘ thermal_relaxation(T1, T2, t_gate)
      - 2-qubit CZ     : depolarizing_2q(1 - F_cz) ∘ pair-thermal-relaxation
      - measurement    : bit-flip readout error from F_RO per qubit
    """
    from qiskit_aer.noise import (
        NoiseModel, depolarizing_error, thermal_relaxation_error, ReadoutError,
    )

    nm = NoiseModel(basis_gates=spec.basis_gates)
    one_q_gates = ["h", "rx", "ry", "rz", "id"]

    for q in range(spec.n_qubits):
        t1_s = spec.t1_us.get(q, 50) * 1e-6
        t2_s = min(spec.t2_us.get(q, 50) * 1e-6, 2 * t1_s)  # T2 ≤ 2·T1 physical bound
        gl_s = spec.gate_length_1q_ns.get(q, 30) * 1e-9
        f_1q = spec.fidelity_1q.get(q, 0.999)

        thermal = thermal_relaxation_error(t1_s, t2_s, gl_s)
        depol = depolarizing_error(max(0.0, 1.0 - f_1q), 1)
        err1q = thermal.compose(depol)
        for gate in one_q_gates:
            nm.add_quantum_error(err1q, gate, [q])

        f_ro = spec.fidelity_ro.get(q, 0.95)
        e = max(0.0, 1.0 - f_ro)
        nm.add_readout_error(ReadoutError([[1 - e, e], [e, 1 - e]]), [q])

    for (a, b), f_cz in spec.fidelity_cz.items():
        gl_s = spec.gate_length_2q_ns.get((a, b), 100) * 1e-9
        # Same thermal model on each qubit, then 2-qubit depolarizing
        t1a = spec.t1_us.get(a, 50) * 1e-6
        t2a = min(spec.t2_us.get(a, 50) * 1e-6, 2 * t1a)
        t1b = spec.t1_us.get(b, 50) * 1e-6
        t2b = min(spec.t2_us.get(b, 50) * 1e-6, 2 * t1b)
        thermal_a = thermal_relaxation_error(t1a, t2a, gl_s)
        thermal_b = thermal_relaxation_error(t1b, t2b, gl_s)
        thermal_2q = thermal_a.expand(thermal_b)
        depol_2q = depolarizing_error(max(0.0, 1.0 - f_cz), 2)
        err2q = thermal_2q.compose(depol_2q)
        nm.add_quantum_error(err2q, "cz", [a, b])
        # The same pair backwards is the same gate physically
        nm.add_quantum_error(err2q, "cz", [b, a])

    return nm


def build_aer_simulator(chip_name: str = "Snowdrop 4q ver2"):
    """
    Return a Qiskit AerSimulator pre-configured with a noise model derived from
    the saved chip calibration JSON. No network calls.

    Returns: (simulator, spec) — keep the spec so the caller can pass
    basis_gates and coupling_map to transpile().
    """
    from qiskit_aer import AerSimulator

    spec = load_chip_spec(chip_name)
    nm = build_noise_model(spec)
    sim = AerSimulator(noise_model=nm,
                        basis_gates=spec.basis_gates,
                        coupling_map=[list(p) for p in spec.coupling_map])
    return sim, spec


# A tiny convenience for live status panels — same structure as
# get_chip_queue_info() but offline.
def get_offline_chip_info(chip_name: str) -> dict[str, Any]:
    try:
        spec = load_chip_spec(chip_name)
    except Exception as e:
        return {"name": chip_name, "available": False, "error": repr(e)}
    return {
        "name": chip_name,
        "available": True,
        "n_qubits": spec.n_qubits,
        "basis_gates": spec.basis_gates,
        "coupling_map": spec.coupling_map,
        "avg_f1q": (sum(spec.fidelity_1q.values()) / max(len(spec.fidelity_1q), 1)),
        "avg_f2q": (sum(spec.fidelity_cz.values()) / max(len(spec.fidelity_cz), 1)),
        "avg_ro":  (sum(spec.fidelity_ro.values()) / max(len(spec.fidelity_ro), 1)),
        "avg_t1_us": (sum(spec.t1_us.values()) / max(len(spec.t1_us), 1)),
        "avg_t2_us": (sum(spec.t2_us.values()) / max(len(spec.t2_us), 1)),
    }

