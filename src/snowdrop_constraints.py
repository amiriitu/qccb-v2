"""
Snowdrop 4q ver2 — chip constraints and operating envelope.

Codifies what the chip can do and where it breaks, distilled from:

  1. Bauman Habr article (10 Dec 2025) — fleet-wide averages, vendor-published.
  2. Live Bauman Octillion API calibration JSON (`snowdrop_4q_ver2.json`) —
     per-qubit / per-pair numbers at session time.
  3. Bauman `api_example.ipynb` — canonical QASM templates and the actual
     native gate set used in vendor-supplied examples.
  4. Empirical behaviour observed during QCCB v2 benchmark runs — job-cancel
     incidents, NOTREADY transitions, the chip's autonomous calibration
     window.

This module is the SINGLE SOURCE OF TRUTH for "what is realistic on
Snowdrop 4q ver2". GUI and runner both query it instead of hard-coding
numbers; the thesis appendix is generated from it (see
`docs/snowdrop_limitations.md`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ============================================================================
# Vendor-published averages (Bauman Habr news, 10 Dec 2025)
# ============================================================================

CHIP_NAME: str = "Snowdrop 4q ver2"
N_QUBITS: int = 4
COMPUTATIONAL_DIM: int = 2 ** N_QUBITS  # 16 states

# Averages reported by Bauman in their public news, December 2025
AVG_F_1Q_VENDOR: float = 0.99892
AVG_F_2Q_VENDOR: float = 0.9906
AVG_F_RO_VENDOR: float = 0.9617


# ============================================================================
# Topology
# ============================================================================

# Confirmed by `api_example.ipynb`: the only CZ pairs that are physically
# coupled. q2 is the central node (star topology, line through q0-q2-q3
# with q1 also connected to q2).
COUPLING_MAP: list[tuple[int, int]] = [(0, 2), (1, 2), (2, 3)]
CENTER_QUBIT: int = 2  # q2 is the routing hub; every 2q op goes through it


# ============================================================================
# Native gate set
# ============================================================================

# What `backend.basis_gates` reports through the Octillion client wrapper.
# Includes `h` as a convenience alias even though the chip's microwave
# control compiles it to ry(π/2)·rz(π) at the pulse level.
BASIS_GATES: list[str] = ["id", "h", "rx", "ry", "rz", "cz"]

# What the vendor-published QASM example actually uses (api_example.ipynb).
# CX is decomposed manually to "ry(-π/2) target ; cz ; ry(π/2) target".
PREFERRED_NATIVE_GATES: list[str] = ["rx", "ry", "rz", "cz", "barrier", "measure"]


# ============================================================================
# Per-qubit limits (worst case across all qubits)
# ============================================================================

@dataclass(frozen=True)
class QubitLimits:
    qubit: int
    f_1q: float       # 1-qubit gate fidelity
    f_ro: float       # readout fidelity
    t1_us: float      # energy relaxation
    t2_us: float      # dephasing


# Snapshot from `Bauman/snowdrop_4q_ver2.json` shipped with QCCB v2.
# Used by emulator_4q_v2; the live `quantum_hardware.get_chip_queue_info()`
# returns the same shape from the API at session time.
PER_QUBIT_NOMINAL: list[QubitLimits] = [
    QubitLimits(qubit=0, f_1q=0.9991, f_ro=0.9519, t1_us=26.10, t2_us=12.97),
    QubitLimits(qubit=1, f_1q=0.9987, f_ro=0.9334, t1_us=20.50, t2_us=7.35),
    QubitLimits(qubit=2, f_1q=0.9988, f_ro=0.9463, t1_us=26.90, t2_us=4.36),
    QubitLimits(qubit=3, f_1q=0.9991, f_ro=0.9626, t1_us=34.48, t2_us=4.39),
]

# CZ fidelity by pair (saved JSON):
PER_PAIR_F_CZ: dict[tuple[int, int], float] = {
    (0, 2): 0.9925,
    (1, 2): 0.9880,   # ← worst CZ pair
    (2, 3): 0.9915,
}


# ============================================================================
# Gate-time parameters (from chip JSON / vendor doc)
# ============================================================================

GATE_LENGTH_1Q_NS: float = 30.0      # all single-qubit gates ≈ 30 ns
GATE_LENGTH_2Q_NS: float = 100.0     # CZ ≈ 100 ns
READOUT_LENGTH_US: float = 3.0       # 3 μs per qubit readout


# ============================================================================
# Operating envelope — empirically-derived "do not cross" limits
# ============================================================================

@dataclass(frozen=True)
class OperatingEnvelope:
    """
    Empirical envelope of (depth, n_cz) combinations the chip handles
    without entering autonomous calibration protection (the "Сервис" state
    we have repeatedly observed for GHZ-4 depth-20 with 9 CZ).

    These thresholds are conservative; they reflect what survived the
    Snowdrop's chip-side monitor during QCCB benchmark sessions in May 2026
    after multiple cancel-and-recover cycles.
    """
    max_transpiled_depth_real: int = 18
    """Above this depth the chip frequently aborts mid-execution."""

    max_cz_count_real: int = 8
    """≥ 9 CZ gates have been empirically associated with CANCELED jobs."""

    max_circuit_time_us: float = 1.5
    """Total gate time (1q + 2q ns, no readout) above which worst-T2 decay
    exceeds 30%. Worst T2 on this chip is 4.36 μs; exp(-1.5/4.36) ≈ 0.71."""

    min_predicted_fidelity: float = 0.45
    """`runner.predict_real_hw_fidelity()` cutoff — circuits with F < 0.45
    are skipped on real-hw to spare the chip."""

    submission_cooldown_s: float = 6.0
    """Minimum sleep between consecutive real-hw submissions to avoid
    pushing the chip into thermal NOTREADY state."""

    max_repeats_per_session_real: int = 10
    """Per dissertation §2.6: N=10 is the realistic limit for real-hw cells
    under QPU-queue and calibration-drift constraints. N=30 is the golden
    standard but achievable only on density-matrix emulators."""


OPERATING_ENVELOPE = OperatingEnvelope()


# ============================================================================
# Risk classification for a candidate circuit
# ============================================================================

@dataclass(frozen=True)
class CircuitRiskReport:
    transpiled_depth: int
    n_1q_gates: int
    n_2q_gates: int
    n_measures: int
    estimated_time_us: float
    predicted_fidelity: float
    risk_level: Literal["green", "yellow", "red"]
    reasons: list[str] = field(default_factory=list)


def classify_circuit_risk(
    transpiled_depth: int,
    n_1q_gates: int,
    n_2q_gates: int,
    n_measures: int,
    chip_spec=None,
) -> CircuitRiskReport:
    """
    Classify a transpiled circuit against the Snowdrop operating envelope.

    Returns a `CircuitRiskReport`:
      - green  : safe, expected to complete with usable fidelity
      - yellow : borderline, may complete but chip likely degrades during
                 sustained repeats — recommend N ≤ 3 instead of 10
      - red    : circuit pushes the chip outside its empirical envelope;
                 submission is likely to be CANCELED by chip-side autonomous
                 calibration protection. Recommend running on emulator only.
    """
    env = OPERATING_ENVELOPE
    reasons: list[str] = []
    risk = "green"

    t_us = (n_1q_gates * GATE_LENGTH_1Q_NS
             + n_2q_gates * GATE_LENGTH_2Q_NS) / 1000.0

    if transpiled_depth > env.max_transpiled_depth_real:
        risk = "red"
        reasons.append(
            f"depth={transpiled_depth} > {env.max_transpiled_depth_real} "
            f"(envelope max)"
        )
    elif transpiled_depth > env.max_transpiled_depth_real - 4:
        risk = "yellow" if risk == "green" else risk
        reasons.append(f"depth={transpiled_depth} approaches envelope cap")

    if n_2q_gates > env.max_cz_count_real:
        risk = "red"
        reasons.append(
            f"n_cz={n_2q_gates} > {env.max_cz_count_real} "
            f"(chip cancels at this CZ count empirically)"
        )
    elif n_2q_gates > env.max_cz_count_real - 2:
        if risk == "green":
            risk = "yellow"
        reasons.append(f"n_cz={n_2q_gates} approaches CZ envelope cap")

    if t_us > env.max_circuit_time_us:
        if risk == "green":
            risk = "yellow"
        reasons.append(
            f"gate-time {t_us:.2f} μs exceeds worst-T2 budget"
        )

    # Predicted fidelity from chip_spec (if provided) — folds in 1q/2q/RO
    # fidelities and worst-T2 dephasing per `runner.predict_real_hw_fidelity`
    f_pred = 1.0
    if chip_spec is not None:
        try:
            from src.experiments.runner import predict_real_hw_fidelity
            class _FakeQc:
                def count_ops(self):
                    return {
                        "rx": n_1q_gates // 2, "ry": n_1q_gates // 2,
                        "cz": n_2q_gates, "measure": n_measures,
                    }
            f_pred = predict_real_hw_fidelity(_FakeQc(), chip_spec)
        except Exception:
            f_pred = 1.0
        if f_pred < env.min_predicted_fidelity:
            risk = "red"
            reasons.append(
                f"predicted F={f_pred:.3f} < {env.min_predicted_fidelity}"
            )

    if not reasons:
        reasons.append("within operating envelope")

    return CircuitRiskReport(
        transpiled_depth=transpiled_depth,
        n_1q_gates=n_1q_gates,
        n_2q_gates=n_2q_gates,
        n_measures=n_measures,
        estimated_time_us=round(t_us, 3),
        predicted_fidelity=round(f_pred, 3),
        risk_level=risk,
        reasons=reasons,
    )


# ============================================================================
# Vendor-supplied QASM templates (from api_example.ipynb verbatim)
# ============================================================================

GHZ_4_QASM_NATIVE: str = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
ry(-pi/2) q[0];
ry(-pi/2) q[1];
ry(pi/2) q[2];
ry(-pi/2) q[3];
rz(pi) q[0];
rz(pi) q[1];
rx(pi) q[2];
rz(pi) q[3];
barrier q[0],q[1],q[2],q[3];
cz q[0],q[2];
ry(-pi/2) q[0];
cz q[1],q[2];
rz(pi) q[0];
ry(-pi/2) q[1];
cz q[2],q[3];
rz(pi) q[1];
ry(-pi/2) q[3];
rz(pi) q[3];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
"""
"""GHZ-4 state preparation in native Snowdrop gates, verbatim from
Bauman's `api_example.ipynb`. Uses ONLY ry/rz/rx/cz/barrier/measure with
no decomposition overhead from transpile."""


# ============================================================================
# Sanity check on submission parameters
# ============================================================================

def validate_submission(circuit_qubits: int, shots_exponent: int,
                          backend_kind: str) -> list[str]:
    """
    Pre-flight validation for a submission. Returns a list of WARNINGS
    (empty if the submission is well-formed). Caller decides whether to
    proceed or abort.
    """
    warnings: list[str] = []
    if circuit_qubits > N_QUBITS:
        warnings.append(
            f"circuit needs {circuit_qubits} qubits but chip has only "
            f"{N_QUBITS}"
        )
    if shots_exponent < 6 or shots_exponent > 14:
        warnings.append(
            f"shots_exponent={shots_exponent} unusual — typical range is "
            f"6 (2^6=64) to 14 (2^14=16384); 10 (1024) is QCCB default"
        )
    if backend_kind == "real" and shots_exponent > 12:
        warnings.append(
            f"shots=2^{shots_exponent} on real-hw is expensive in queue "
            f"time; consider sticking to 2^10=1024 unless a low-yield "
            f"measurement makes it necessary"
        )
    return warnings

