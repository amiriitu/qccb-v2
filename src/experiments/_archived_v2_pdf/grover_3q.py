"""
Experiment 3: Grover's algorithm for the marked state |101⟩ — 3 work qubits.

Architecture per the SnowDrop 8Q PDF: 3 working qubits + 2 ancilla qubits
(spare qubits used for v-chain decomposition of multi-controlled gates).
This trades width (qubit count) for depth (gate-error budget).

For a database of N=8 elements, the optimal number of Grover iterations is
ceil(π/4 · √N) ≈ 2.

Goal: amplitude amplification of |101⟩. Metric = P(measured == 101).
"""
from __future__ import annotations

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

from src.experiments import ExperimentDef, ParameterSpec


TARGET_STATES = ["000", "001", "010", "011", "100", "101", "110", "111"]


PARAMETERS: list[ParameterSpec] = [
    ParameterSpec(
        name="target", label="Marked state",
        kind="choice", default="101", choices=TARGET_STATES,
        help="3-bit string the oracle marks (phase-flips).",
    ),
    ParameterSpec(
        name="iterations", label="Grover iterations",
        kind="choice", default=2, choices=[1, 2, 3],
        help="Optimal for N=8 is 2 (≈ π/4 · √8).",
    ),
]


def _oracle(qc: QuantumCircuit, qr, anc, target: str) -> None:
    """Phase-flip the |target⟩ component."""
    n = len(qr)
    for i, bit in enumerate(reversed(target)):
        if bit == "0":
            qc.x(qr[i])

    qc.h(qr[n - 1])
    if n >= 3:
        qc.ccx(qr[0], qr[1], anc[0])
        qc.cx(anc[0], qr[n - 1])
        qc.ccx(qr[0], qr[1], anc[0])
    else:
        qc.cz(qr[0], qr[1])
    qc.h(qr[n - 1])

    for i, bit in enumerate(reversed(target)):
        if bit == "0":
            qc.x(qr[i])


def _diffuser(qc: QuantumCircuit, qr, anc) -> None:
    """Standard diffuser H X (CCZ) X H about the all-ones state."""
    n = len(qr)
    qc.h(qr)
    qc.x(qr)

    qc.h(qr[n - 1])
    if n >= 3:
        qc.ccx(qr[0], qr[1], anc[0])
        qc.cx(anc[0], qr[n - 1])
        qc.ccx(qr[0], qr[1], anc[0])
    else:
        qc.cz(qr[0], qr[1])
    qc.h(qr[n - 1])

    qc.x(qr)
    qc.h(qr)


def build(params: dict) -> QuantumCircuit:
    target = str(params.get("target", "101"))
    iterations = int(params.get("iterations", 2))

    qr = QuantumRegister(3, "work")
    anc = QuantumRegister(2, "ancilla")
    cr = ClassicalRegister(3, "meas")
    qc = QuantumCircuit(qr, anc, cr, name=f"grover_{target}_iter{iterations}")

    qc.h(qr)

    for _ in range(iterations):
        _oracle(qc, qr, anc, target)
        qc.barrier()
        _diffuser(qc, qr, anc)
        qc.barrier()

    qc.measure(qr, cr)
    return qc


def expected(params: dict) -> dict[str, float]:
    """Optimal 2-iteration Grover concentrates ≈ 95% of mass on the target."""
    target = str(params.get("target", "101"))
    iterations = int(params.get("iterations", 2))
    p_target = {1: 0.78, 2: 0.945, 3: 0.42}.get(iterations, 0.5)
    p_other = (1.0 - p_target) / 7
    return {s: (p_target if s == target else p_other) for s in TARGET_STATES}


def metric_fn(counts: dict[str, int], params: dict) -> float:
    target = str(params.get("target", "101"))
    total = sum(counts.values()) or 1
    return counts.get(target, 0) / total


EXPERIMENT = ExperimentDef(
    key="grover",
    title="Grover (search |101⟩)",
    description=(
        "Grover's algorithm on 3 working qubits (+ 2 ancilla for v-chain).\n"
        "Database of 8 elements; optimal iterations ≈ 2.\n"
        "Oracle marks |101⟩ via X-CCZ-X. Diffuser is the standard inversion-\n"
        "about-mean. Metric = P(measured == target)."
    ),
    qubits_used=5,
    parameters=PARAMETERS,
    build=build,
    expected=expected,
    metric_fn=metric_fn,
    metric_name="P(target found)",
    interpretation_hint=(
        "Ideal P ≈ 0.94 with 2 iterations. Real-chip drop reflects depth × CCX cost."
    ),
)


if __name__ == "__main__":
    import argparse
    from src.experiments.runner import run_circuit

    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="emulator_8q")
    p.add_argument("--shots-exp", type=int, default=10)
    p.add_argument("--target", default="101", choices=TARGET_STATES)
    p.add_argument("--iterations", type=int, default=2, choices=[1, 2, 3])
    args = p.parse_args()

    params = {"target": args.target, "iterations": args.iterations}
    qc = EXPERIMENT.build(params)
    print(qc.draw(output="text"))
    res = run_circuit(
        qc, label=f"grover_{args.target}_i{args.iterations}",
        backend_kind=args.backend, shots_exponent=args.shots_exp,
        expected_distribution=EXPERIMENT.expected(params),
        metric_name=EXPERIMENT.metric_name,
        metric_fn=lambda c: EXPERIMENT.metric_fn(c, params),
        status=lambda m: print(f"[{m}]"),
    )
    print(f"\nCounts: {res.counts}")
    print(f"{res.metric_name}: {res.metric_value:.4f}")

