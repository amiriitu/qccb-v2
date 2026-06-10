"""
Minimal GPU state-vector simulator using CuPy.

Why this exists:
  qiskit-aer-gpu, cudaq, and cuquantum-python have NO Windows wheels for
  CUDA 13 + Python 3.13 (verified empirically). CuPy + the bundled
  nvidia-cublas-cu12 pip package work out of the box and give us direct
  GPU acceleration of the same cuBLAS routines that Aer-GPU uses under the hood.

Design:
  - State vector stored as 1D cupy.complex64 array of length 2^n.
  - Gates applied via cupy.einsum-style tensor contraction by reshaping the
    state to shape [2]*n and contracting the gate against the qubit axis.
  - Sampling done on CPU after collapsing |amplitudes|^2.
  - Supports H/X/Y/Z/S/SX/T/RX/RY/RZ (1q) and CX/CZ (2q); other gates fall back
    to qiskit.quantum_info.Operator(op).data which is numerically correct
    but slower (CPU build of the matrix, then upload).

Limits:
  - Memory: 2^n * 8 bytes = 256 MB at 25 qubits, 4 GB at 28 qubits.
    On 8 GB RTX 4060 Laptop, comfortable up to ~28-29 qubits.
  - Single GPU only. No tensor-network mode.
  - Statevector method only (no DM, no MPS).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import cupy as cp
import numpy as np
from qiskit import QuantumCircuit


# -----------------------------------------------------------------
# Gate lookup tables (numpy, will be uploaded to GPU on first use)
# -----------------------------------------------------------------

_C64 = np.complex64

_I = np.eye(2, dtype=_C64)
_X = np.array([[0, 1], [1, 0]], dtype=_C64)
_Y = np.array([[0, -1j], [1j, 0]], dtype=_C64)
_Z = np.array([[1, 0], [0, -1]], dtype=_C64)
_H = np.array([[1, 1], [1, -1]], dtype=_C64) / np.sqrt(2)
_S = np.array([[1, 0], [0, 1j]], dtype=_C64)
_T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=_C64)
_SX = 0.5 * np.array([[1+1j, 1-1j], [1-1j, 1+1j]], dtype=_C64)


def _rx(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=_C64)


def _ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=_C64)


def _rz(theta: float) -> np.ndarray:
    return np.array(
        [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=_C64
    )


_CX = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=_C64
)
_CZ = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=_C64
)


def _gate_matrix_1q(op) -> np.ndarray:
    name = op.name
    if name == "h": return _H
    if name == "x": return _X
    if name == "y": return _Y
    if name == "z": return _Z
    if name == "s": return _S
    if name == "t": return _T
    if name == "sx": return _SX
    if name == "rx": return _rx(float(op.params[0]))
    if name == "ry": return _ry(float(op.params[0]))
    if name == "rz": return _rz(float(op.params[0]))
    if name == "id": return _I
    from qiskit.quantum_info import Operator
    return Operator(op).data.astype(_C64)


def _gate_matrix_2q(op) -> np.ndarray:
    name = op.name
    if name == "cx": return _CX
    if name == "cz": return _CZ
    from qiskit.quantum_info import Operator
    return Operator(op).data.astype(_C64)


# -----------------------------------------------------------------
# Core state-vector ops on GPU
# -----------------------------------------------------------------

def _apply_1q(state: cp.ndarray, U: cp.ndarray, q: int, n: int) -> cp.ndarray:
    """
    Apply 2x2 unitary U to qubit q in a length-2^n state.
    Qubit indexing convention: qubit 0 is the most significant bit (Qiskit big-endian).
    """
    state = state.reshape([2] * n)
    state = cp.moveaxis(state, q, -1)
    state = state @ U.T
    state = cp.moveaxis(state, -1, q)
    return state.reshape(2 ** n)


def _apply_2q(state: cp.ndarray, U4: cp.ndarray,
              q0: int, q1: int, n: int) -> cp.ndarray:
    """
    Apply 4x4 unitary U4 (basis order |q0 q1⟩ = 00, 01, 10, 11) to qubits q0, q1.
    """
    state = state.reshape([2] * n)
    state = cp.moveaxis(state, [q0, q1], [-2, -1])
    pre_shape = state.shape
    state = state.reshape(*pre_shape[:-2], 4)
    state = state @ U4.T
    state = state.reshape(*pre_shape)
    state = cp.moveaxis(state, [-2, -1], [q0, q1])
    return state.reshape(2 ** n)


# -----------------------------------------------------------------
# Public API
# -----------------------------------------------------------------

@dataclass
class GpuRunStats:
    n_qubits: int
    shots: int
    counts: dict[str, int]
    state_alloc_s: float = 0.0
    gate_apply_s: float = 0.0
    sample_s: float = 0.0
    total_s: float = 0.0
    vram_used_mb: float = 0.0
    gate_count: int = 0


def detect_gpu() -> dict[str, Any]:
    """Return basic GPU info; raises if no CUDA device is present."""
    props = cp.cuda.runtime.getDeviceProperties(0)
    free, total = cp.cuda.runtime.memGetInfo()
    return {
        "name": props["name"].decode(),
        "vram_total_mb": total // (1024 ** 2),
        "vram_free_mb": free // (1024 ** 2),
        "compute_capability": f"{props['major']}.{props['minor']}",
        "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
    }


def run_circuit_gpu(qc: QuantumCircuit, shots: int = 1024,
                     warmup: bool = True) -> GpuRunStats:
    """
    Simulate `qc` on GPU (CuPy state-vector) and return shot-sampled counts.
    """
    n = qc.num_qubits

    if warmup:
        _w = cp.eye(2, dtype=cp.complex64)
        _ = (_w @ _w).sum()
        cp.cuda.runtime.deviceSynchronize()

    t0 = time.perf_counter()

    state = cp.zeros(2 ** n, dtype=cp.complex64)
    state[0] = 1.0
    cp.cuda.runtime.deviceSynchronize()
    t_alloc = time.perf_counter() - t0

    measurements: list[tuple[int, int]] = []
    gate_count = 0
    t_apply_start = time.perf_counter()

    for instr in qc.data:
        op = instr.operation
        qubits = [qc.find_bit(q).index for q in instr.qubits]

        if op.name == "measure":
            cl = qc.find_bit(instr.clbits[0]).index
            measurements.append((qubits[0], cl))
            continue
        if op.name in ("barrier", "reset"):
            continue

        if len(qubits) == 1:
            U = cp.asarray(_gate_matrix_1q(op))
            state = _apply_1q(state, U, qubits[0], n)
        elif len(qubits) == 2:
            U = cp.asarray(_gate_matrix_2q(op))
            state = _apply_2q(state, U, qubits[0], qubits[1], n)
        else:
            raise NotImplementedError(
                f"{len(qubits)}-qubit gate '{op.name}' not supported "
                f"by minimal GPU simulator"
            )
        gate_count += 1

    cp.cuda.runtime.deviceSynchronize()
    t_apply = time.perf_counter() - t_apply_start

    t_sample_start = time.perf_counter()
    probs = cp.abs(state) ** 2
    probs = probs / probs.sum()
    probs_cpu = cp.asnumpy(probs).astype(np.float64)
    if probs_cpu.sum() != 1.0:
        probs_cpu = probs_cpu / probs_cpu.sum()

    rng = np.random.default_rng()
    samples = rng.choice(2 ** n, size=shots, p=probs_cpu)

    n_clbits = qc.num_clbits if qc.num_clbits > 0 else n
    counts: dict[str, int] = {}

    if measurements:
        meas_map = {q: c for q, c in measurements}
        for s in samples:
            bits = ["0"] * n_clbits
            for q in range(n):
                if q in meas_map:
                    qubit_bit = (int(s) >> (n - 1 - q)) & 1
                    bits[meas_map[q]] = str(qubit_bit)
            bitstring = "".join(reversed(bits))
            counts[bitstring] = counts.get(bitstring, 0) + 1
    else:
        for s in samples:
            bitstring = format(int(s), f"0{n}b")
            counts[bitstring] = counts.get(bitstring, 0) + 1
    t_sample = time.perf_counter() - t_sample_start

    free, _total = cp.cuda.runtime.memGetInfo()
    used_mb = (cp.cuda.runtime.getDeviceProperties(0)["totalGlobalMem"] - free) / (1024 ** 2)

    return GpuRunStats(
        n_qubits=n,
        shots=shots,
        counts=counts,
        state_alloc_s=t_alloc,
        gate_apply_s=t_apply,
        sample_s=t_sample,
        total_s=time.perf_counter() - t0,
        vram_used_mb=used_mb,
        gate_count=gate_count,
    )

