"""
GHZ-N state: |GHZ_N⟩ = (|0...0⟩ + |1...1⟩)/√2.
Multi-qubit entanglement, parameterized by N ∈ {2, 3, 4}.

Ideal: P('0...0') = P('1...1') = 0.5, all 2^N - 2 other outcomes = 0.
Tests chain of (N-1) CZ gates routed through chip topology.
"""
from __future__ import annotations

from qiskit import QuantumCircuit

from src.experiments import ExperimentDef, ParameterSpec


PARAMETERS: list[ParameterSpec] = [
    ParameterSpec(
        name="n", label="Number of qubits (N)",
        kind="choice", default=3, choices=[2, 3, 4],
        help="2 = Bell, 3 = standard GHZ, 4 = full-chip GHZ.",
    ),
]


def build(params: dict) -> QuantumCircuit:
    n = int(params.get("n", 3))
    qc = QuantumCircuit(n, n, name=f"ghz_{n}")
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    qc.measure(range(n), range(n))
    return qc


def expected(params: dict) -> dict[str, float]:
    n = int(params.get("n", 3))
    out = {format(i, f"0{n}b"): 0.0 for i in range(2 ** n)}
    out["0" * n] = 0.5
    out["1" * n] = 0.5
    return out


def metric_fn(counts: dict[str, int], params: dict) -> float:
    n = int(params.get("n", 3))
    total = sum(counts.values()) or 1
    return (counts.get("0" * n, 0) + counts.get("1" * n, 0)) / total


EXPERIMENT = ExperimentDef(
    key="ghz",
    title="GHZ-N entanglement",
    description=(
        "N-qubit GHZ state (|0...0⟩+|1...1⟩)/√2.\n"
        "Stresses (N-1) sequential CZ gates on the chip's T-topology.\n"
        "Ideal: P('0...0') = P('1...1') = 0.5.\n"
        "Metric: F_GHZ = P(all-0) + P(all-1)."
    ),
    qubits_used=4,
    parameters=PARAMETERS,
    build=build,
    expected=expected,
    metric_fn=metric_fn,
    metric_name="GHZ fidelity",
    interpretation_hint="Drop vs Bell shows the cost of each extra CZ + routing SWAP.",
)


if __name__ == "__main__":
    import argparse
    from src.experiments.runner import run_circuit

    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["ideal", "emulator", "real"], default="real")
    p.add_argument("--shots-exp", type=int, default=10)
    p.add_argument("-n", type=int, default=3, choices=[2, 3, 4])
    args = p.parse_args()

    params = {"n": args.n}
    qc = EXPERIMENT.build(params)
    print(qc.draw(output="text"))
    res = run_circuit(
        qc, label=f"{EXPERIMENT.key}_{args.n}", backend_kind=args.backend,
        shots_exponent=args.shots_exp,
        expected_distribution=EXPERIMENT.expected(params),
        metric_name=EXPERIMENT.metric_name,
        metric_fn=lambda c: EXPERIMENT.metric_fn(c, params),
        status=lambda m: print(f"[{m}]"),
    )
    print(f"\nCounts: {res.counts}")
    print(f"{res.metric_name}: {res.metric_value:.4f}")

