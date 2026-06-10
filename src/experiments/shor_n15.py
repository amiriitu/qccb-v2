"""
Compiled Shor's algorithm for N=15 — 4 qubits.

For N=15, only base a=4 fits in 3 work qubits (period r=2; cycle is {1, 4} ⊂ {0..7}).
Other coprime bases (2, 7, 8, 11, 13, 14) need 4-bit work register, which would
exceed our 4-qubit chip (we'd need 5 qubits total: 1 counting + 4 work).

We expose `a` as a parameter for educational completeness; non-4 values build
a circuit that does NOT realize Shor correctly on this chip and should be
treated as "what happens if you push beyond hardware limits" demonstrations.
"""
from __future__ import annotations

from math import gcd

from qiskit import QuantumCircuit

from src.experiments import ExperimentDef, ParameterSpec


N = 15


PARAMETERS: list[ParameterSpec] = [
    ParameterSpec(
        name="a", label="Base (a, coprime with 15)",
        kind="choice", default=4, choices=[4],
        help="Only a=4 fits in 4 qubits on Snowdrop. Other coprimes need 5+ qubits.",
    ),
]


def _controlled_modmult_a4(qc: QuantumCircuit, ctrl: int,
                            w0: int, w1: int, w2: int) -> None:
    """
    Controlled multiplication by 4 mod 15 on the cycle {|001⟩, |100⟩}.
    Implementation: XOR work register with |101⟩ if ctrl = 1.
    """
    qc.cx(ctrl, w0)
    qc.cx(ctrl, w2)


def build(params: dict) -> QuantumCircuit:
    a = int(params.get("a", 4))
    qc = QuantumCircuit(4, 1, name=f"shor_n{N}_a{a}")

    qc.x(1)
    qc.h(0)

    if a == 4:
        _controlled_modmult_a4(qc, ctrl=0, w0=1, w1=2, w2=3)
    else:
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)

    qc.h(0)
    qc.measure(0, 0)
    return qc


def expected(params: dict) -> dict[str, float]:
    return {"0": 0.5, "1": 0.5}


def metric_fn(counts: dict[str, int], params: dict) -> float:
    total = sum(counts.values()) or 1
    return counts.get("1", 0) / total


def interpret(counts: dict[str, int], params: dict | None = None) -> dict:
    params = params or {"a": 4}
    a = int(params.get("a", 4))
    total = sum(counts.values()) or 1
    p1 = counts.get("1", 0) / total
    p0 = counts.get("0", 0) / total

    factors = None
    if p1 >= 0.30:
        r = 2
        f1 = gcd(a ** (r // 2) - 1, N)
        f2 = gcd(a ** (r // 2) + 1, N)
        factors = sorted({f1, f2} - {1, N})

    return {
        "P(0_uninformative)": p0,
        "P(1_period_found)": p1,
        "inferred_r": 2 if p1 >= 0.30 else None,
        "factors_of_15": factors,
        "verdict": (
            f"Shor succeeded: {N} = {factors[0]} × {factors[1]}"
            if factors and len(factors) == 2 else
            "Inconclusive — need more shots / retry"
        ),
    }


EXPERIMENT = ExperimentDef(
    key="shor",
    title=f"Compiled Shor (N={N})",
    description=(
        f"Finds period r of f(x) = a^x mod {N} via 1-qubit QPE.\n"
        f"For a=4, order r=2 ⇒ factors of {N} = gcd(4±1, {N}) = 3, 5.\n"
        "Ideal: P(measure 1) = 0.5 — that bit is the period bit.\n"
        "Metric: P('1') — share of shots that revealed the period."
    ),
    qubits_used=4,
    parameters=PARAMETERS,
    build=build,
    expected=expected,
    metric_fn=metric_fn,
    metric_name="P(period found)",
    interpretation_hint=(
        "P('1') ≈ 0.5 = success. Significantly less = noise washed out the "
        "period bit; chip 2q fidelity is the bottleneck."
    ),
)


if __name__ == "__main__":
    import argparse
    from src.experiments.runner import run_circuit

    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["ideal", "emulator", "real"], default="real")
    p.add_argument("--shots-exp", type=int, default=10)
    p.add_argument("-a", type=int, default=4)
    args = p.parse_args()

    params = {"a": args.a}
    qc = EXPERIMENT.build(params)
    print(qc.draw(output="text"))
    res = run_circuit(
        qc, label=f"{EXPERIMENT.key}_a{args.a}", backend_kind=args.backend,
        shots_exponent=args.shots_exp,
        expected_distribution=EXPERIMENT.expected(params),
        metric_name=EXPERIMENT.metric_name,
        metric_fn=lambda c: EXPERIMENT.metric_fn(c, params),
        status=lambda m: print(f"[{m}]"),
    )
    print(f"\nCounts: {res.counts}")
    print(f"{res.metric_name}: {res.metric_value:.4f}")
    print("\nInterpretation:")
    for k, v in interpret(res.counts, params).items():
        print(f"  {k}: {v}")

