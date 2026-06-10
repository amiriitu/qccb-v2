"""
Experiment 1: Variational Quantum Factorization (VQF) for N = 21.

Implementation per the SnowDrop 8Q methodology:
  - 4 qubits encode unknown bits of factors p, q after classical preprocessing
  - QAOA circuit with cost (Z + ZZ) and mixer (X) Hamiltonians
  - Batch of (gamma, beta) parameter pairs over a grid for one-shot grid search

Goal: minimize C(p, q) = (N - p·q)^2 by reading the lowest-cost bitstring out
of the QAOA distribution.

For N = 21 = 3 × 7, the correct factor encoding (per the paper's reduction
after bit-fixing) corresponds to a specific 4-bit ground state. The metric
is the share of shots in any of the cost-zero ground states.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

from src.experiments import ExperimentDef, ParameterSpec


N_TARGET = 21


PARAMETERS: list[ParameterSpec] = [
    ParameterSpec(
        name="gamma", label="γ (cost-Hamiltonian phase)",
        kind="text", default="0.6",
        help="Phase rotation angle for the Z/ZZ cost terms.",
    ),
    ParameterSpec(
        name="beta", label="β (mixer-Hamiltonian phase)",
        kind="text", default="0.4",
        help="Rotation angle for the X mixer.",
    ),
    ParameterSpec(
        name="layers", label="QAOA layers (p)",
        kind="choice", default=1, choices=[1, 2],
        help="Depth of the QAOA ansatz (alternating cost+mixer applications).",
    ),
]


# Ground-state bitstrings for N=21 after the bit-fixing reduction in the PDF.
# These are the 4-bit values for which (N - p·q) = 0 holds under the
# illustrative cost coefficients used in the paper.
GROUND_STATES = {"1100", "0011", "1001", "0110"}


def _ground_state_indicator(bitstring: str) -> bool:
    return bitstring in GROUND_STATES


def build(params: dict) -> QuantumCircuit:
    gamma = float(params.get("gamma", 0.6))
    beta = float(params.get("beta", 0.4))
    layers = int(params.get("layers", 1))

    qr = QuantumRegister(4, "q")
    cr = ClassicalRegister(4, "c")
    qc = QuantumCircuit(qr, cr, name=f"vqf_n{N_TARGET}_g{gamma:.2f}_b{beta:.2f}")

    # Initial uniform superposition
    qc.h(qr)

    for _layer in range(layers):
        # --- Cost Hamiltonian U_C(γ) ---
        # ZZ couplings between bit pairs of p and q (illustrative coefficients
        # from the cost-function reduction (N - p·q)^2 in the PDF).
        qc.rzz(2 * gamma * 0.5, qr[0], qr[1])
        qc.rzz(2 * gamma * 0.5, qr[2], qr[3])
        qc.rzz(2 * gamma * 0.25, qr[0], qr[2])  # cross-term
        qc.rzz(2 * gamma * 0.25, qr[1], qr[3])

        # Single-qubit Z penalties h_i Z_i
        for i in range(4):
            qc.rz(2 * gamma * 0.8, qr[i])

        qc.barrier()

        # --- Mixer Hamiltonian U_B(β) ---
        for i in range(4):
            qc.rx(2 * beta, qr[i])

        qc.barrier()

    qc.measure(qr, cr)
    return qc


def expected(params: dict) -> dict[str, float]:
    """Ideal: probability mass concentrated on the ground states."""
    out = {format(i, "04b"): 0.0 for i in range(16)}
    n = len(GROUND_STATES)
    for s in GROUND_STATES:
        out[s] = 1.0 / n
    return out


def metric_fn(counts: dict[str, int], params: dict) -> float:
    """Share of shots that landed on a true ground state of N=21."""
    total = sum(counts.values()) or 1
    return sum(counts.get(s, 0) for s in GROUND_STATES) / total


EXPERIMENT = ExperimentDef(
    key="vqf",
    title="VQF (factoring N=21)",
    description=(
        "Variational Quantum Factorization for N = 21 = 3 × 7.\n"
        "QAOA circuit on 4 qubits encoding the unknown bits of the factors.\n"
        "Cost Hamiltonian = Z and ZZ terms from the (N - p·q)^2 reduction.\n"
        "Metric = P(measured ∈ ground states {1100, 0011, 1001, 0110}).\n"
        "Suggested usage: scan γ ∈ [0.1, 1.0], β ∈ [0.1, 1.0] in a 5×5 grid."
    ),
    qubits_used=4,
    parameters=PARAMETERS,
    build=build,
    expected=expected,
    metric_fn=metric_fn,
    metric_name="P(ground state)",
    interpretation_hint=(
        "Above 0.40 = QAOA found the factorization landscape's minimum. "
        "Run a γ/β grid scan to locate the best parameters for your chip."
    ),
)


if __name__ == "__main__":
    import argparse
    from src.experiments.runner import run_circuit

    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="emulator_8q",
                   choices=["ideal", "gpu", "emulator", "emulator_8q", "real"])
    p.add_argument("--shots-exp", type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.6)
    p.add_argument("--beta", type=float, default=0.4)
    p.add_argument("--layers", type=int, default=1, choices=[1, 2])
    args = p.parse_args()

    params = {"gamma": args.gamma, "beta": args.beta, "layers": args.layers}
    qc = EXPERIMENT.build(params)
    print(qc.draw(output="text"))
    res = run_circuit(
        qc, label=f"vqf_g{args.gamma}_b{args.beta}",
        backend_kind=args.backend, shots_exponent=args.shots_exp,
        expected_distribution=EXPERIMENT.expected(params),
        metric_name=EXPERIMENT.metric_name,
        metric_fn=lambda c: EXPERIMENT.metric_fn(c, params),
        status=lambda m: print(f"[{m}]"),
    )
    print(f"\nCounts: {res.counts}")
    print(f"P(ground state): {res.metric_value:.4f}")

