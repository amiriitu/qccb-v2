"""
Bell pair |Φ+⟩ = (|00⟩ + |11⟩)/√2 — 2-qubit entanglement test.
Ideal: 50% '00' + 50% '11'. Deviation = chip's 2q + readout error budget.
"""
from __future__ import annotations

from qiskit import QuantumCircuit

from src.experiments import ExperimentDef, ParameterSpec


PARAMETERS: list[ParameterSpec] = []


def build(params: dict) -> QuantumCircuit:
    qc = QuantumCircuit(2, 2, name="bell_pair")
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def expected(params: dict) -> dict[str, float]:
    return {"00": 0.5, "01": 0.0, "10": 0.0, "11": 0.5}


def metric_fn(counts: dict[str, int], params: dict) -> float:
    total = sum(counts.values()) or 1
    return (counts.get("00", 0) + counts.get("11", 0)) / total


EXPERIMENT = ExperimentDef(
    key="bell",
    title="Bell pair (2q)",
    description=(
        "Two-qubit entanglement |Φ+⟩ = (|00⟩+|11⟩)/√2.\n"
        "Tests basic 1q+CZ+readout pipeline.\n"
        "Ideal: P(00)=P(11)=0.5; P(01)=P(10)=0.\n"
        "Metric: empirical fidelity = P(00)+P(11)."
    ),
    qubits_used=2,
    parameters=PARAMETERS,
    build=build,
    expected=expected,
    metric_fn=metric_fn,
    metric_name="Bell fidelity",
    interpretation_hint="Above 0.90 = good, below 0.80 = chip degraded.",
)


if __name__ == "__main__":
    import argparse
    from src.experiments.runner import run_circuit

    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["ideal", "emulator", "real"], default="real")
    p.add_argument("--shots-exp", type=int, default=10)
    args = p.parse_args()

    params = EXPERIMENT.default_params()
    qc = EXPERIMENT.build(params)
    print(qc.draw(output="text"))
    res = run_circuit(
        qc, label=EXPERIMENT.key, backend_kind=args.backend,
        shots_exponent=args.shots_exp,
        expected_distribution=EXPERIMENT.expected(params),
        metric_name=EXPERIMENT.metric_name,
        metric_fn=lambda c: EXPERIMENT.metric_fn(c, params),
        status=lambda m: print(f"[{m}]"),
    )
    print(f"\nCounts: {res.counts}")
    print(f"{res.metric_name}: {res.metric_value:.4f}")

