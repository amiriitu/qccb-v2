"""
Bernstein-Vazirani algorithm — finds a hidden bit-string s in 1 query.
For 3-bit s, classical needs 3 queries; quantum needs only 1.

Layout (4 qubits):
  q0..q2: input register (will encode s after H-oracle-H sandwich)
  q3:     output qubit (initialized to |1⟩ for the phase-kickback trick)
"""
from __future__ import annotations

from qiskit import QuantumCircuit

from src.experiments import ExperimentDef, ParameterSpec


SECRET_CHOICES = [format(i, "03b") for i in range(8)]


PARAMETERS: list[ParameterSpec] = [
    ParameterSpec(
        name="secret", label="Hidden 3-bit string (s)",
        kind="choice", default="101", choices=SECRET_CHOICES,
        help="Algorithm finds s in 1 oracle query (vs 3 classically).",
    ),
]


def build(params: dict) -> QuantumCircuit:
    secret = str(params.get("secret", "101"))
    n = len(secret)
    qc = QuantumCircuit(n + 1, n, name=f"bv_{secret}")

    qc.x(n)
    qc.h(n)

    for i in range(n):
        qc.h(i)

    qc.barrier()
    for i, bit in enumerate(secret):
        qubit_index = n - 1 - i
        if bit == "1":
            qc.cx(qubit_index, n)
    qc.barrier()

    for i in range(n):
        qc.h(i)
    for i in range(n):
        qc.measure(i, i)

    return qc


def expected(params: dict) -> dict[str, float]:
    secret = str(params.get("secret", "101"))
    return {s: (1.0 if s == secret else 0.0) for s in SECRET_CHOICES}


def metric_fn(counts: dict[str, int], params: dict) -> float:
    secret = str(params.get("secret", "101"))
    total = sum(counts.values()) or 1
    return counts.get(secret, 0) / total


EXPERIMENT = ExperimentDef(
    key="bv",
    title="Bernstein-Vazirani (3-bit)",
    description=(
        "Finds hidden 3-bit string s in a SINGLE query.\n"
        "Classical: 3 queries. Quantum: 1.\n"
        "Ideal: 100% probability on the chosen s.\n"
        "Metric: P(measured == s)."
    ),
    qubits_used=4,
    parameters=PARAMETERS,
    build=build,
    expected=expected,
    metric_fn=metric_fn,
    metric_name="P(correct)",
    interpretation_hint=(
        "Drop from 1.0 reveals 6 H + oracle CNOTs gate-error budget. "
        "Trivial layout adds SWAPs through central q2."
    ),
)


if __name__ == "__main__":
    import argparse
    from src.experiments.runner import run_circuit

    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["ideal", "emulator", "real"], default="real")
    p.add_argument("--shots-exp", type=int, default=10)
    p.add_argument("--secret", default="101", choices=SECRET_CHOICES)
    args = p.parse_args()

    params = {"secret": args.secret}
    qc = EXPERIMENT.build(params)
    print(qc.draw(output="text"))
    res = run_circuit(
        qc, label=f"{EXPERIMENT.key}_{args.secret}", backend_kind=args.backend,
        shots_exponent=args.shots_exp,
        expected_distribution=EXPERIMENT.expected(params),
        metric_name=EXPERIMENT.metric_name,
        metric_fn=lambda c: EXPERIMENT.metric_fn(c, params),
        status=lambda m: print(f"[{m}]"),
    )
    print(f"\nCounts: {res.counts}")
    print(f"{res.metric_name}: {res.metric_value:.4f}")

