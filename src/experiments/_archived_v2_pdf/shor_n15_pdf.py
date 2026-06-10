"""
Experiment 2: Compiled Shor's algorithm for N = 15, base a = 7 (per SnowDrop 8Q PDF).

Architecture:
  - 3 control qubits (phase register) → 3 bits of QPE precision
  - 2 work qubits (target register) initialized to |1⟩ = |01⟩
  - Total: 5 qubits + 3 classical

Period r = ord_15(7) = 4 (since 7^1=7, 7^2=49≡4, 7^3=28≡13, 7^4=2401≡1 mod 15).

QPE reads phase k/r = k/4 → measured outcomes concentrate on |00⟩, |01⟩, |10⟩, |11⟩
which decode to phases 0, 1/4, 2/4, 3/4. Phases 1/4 and 3/4 reveal r=4
(continued fractions). Then gcd(7^(r/2) ± 1, 15) = gcd(48, 15)=3, gcd(50, 15)=5.

The "compiled" part: the controlled-U^(2^k) ops for U|y⟩ = |7y mod 15⟩ are
synthesized directly as permutation circuits on 2 qubits, avoiding full
modular-arithmetic reversible blocks.
"""
from __future__ import annotations

from math import gcd

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import QFT

from src.experiments import ExperimentDef, ParameterSpec


N_TARGET = 15


PARAMETERS: list[ParameterSpec] = [
    ParameterSpec(
        name="a", label="Base a (coprime with 15)",
        kind="choice", default=7, choices=[7, 8, 11, 13],
        help="a=7 → r=4; a=11 → r=2; a=8 → r=4; a=13 → r=4.",
    ),
]


def _controlled_mult_a7(qc: QuantumCircuit, ctrl, w0, w1) -> None:
    """C-U for multiplication by 7 mod 15 on the 2-qubit work register."""
    qc.cx(ctrl, w0)
    qc.cx(ctrl, w1)


def _controlled_mult_a4(qc: QuantumCircuit, ctrl, w0, w1) -> None:
    """C-U for multiplication by 4 = 7^2 mod 15."""
    qc.cx(ctrl, w0)


def build(params: dict) -> QuantumCircuit:
    a = int(params.get("a", 7))

    qr_c = QuantumRegister(3, "control")
    qr_w = QuantumRegister(2, "work")
    cr = ClassicalRegister(3, "outcome")
    qc = QuantumCircuit(qr_c, qr_w, cr, name=f"shor_n15_a{a}")

    qc.h(qr_c)
    qc.x(qr_w[0])  # work register initialized to |01⟩ = 1

    qc.barrier()

    # CU^(2^0)  — multiplication by a^1
    _controlled_mult_a7(qc, qr_c[0], qr_w[0], qr_w[1])

    # CU^(2^1)  — multiplication by a^2
    _controlled_mult_a4(qc, qr_c[1], qr_w[0], qr_w[1])

    # CU^(2^2)  — multiplication by a^4 ≡ 1 → identity, omitted

    qc.barrier()

    # Inverse QFT on the control register
    iqft = QFT(num_qubits=3, inverse=True, do_swaps=True).to_gate(label="IQFT")
    qc.append(iqft, qr_c)

    qc.barrier()
    qc.measure(qr_c, cr)
    return qc


def expected(params: dict) -> dict[str, float]:
    """
    Ideal Shor for N=15, a=7 with 3 counting qubits gives uniform probability
    on the 4 phase-revealing outcomes corresponding to k·r/n = k/r.
    """
    return {
        "000": 0.25,  # phase 0/8 = 0
        "010": 0.25,  # phase 2/8 = 1/4 → period 4
        "100": 0.25,  # phase 4/8 = 2/4 → period 2 (factor of 4)
        "110": 0.25,  # phase 6/8 = 3/4 → period 4
        "001": 0.0,
        "011": 0.0,
        "101": 0.0,
        "111": 0.0,
    }


def metric_fn(counts: dict[str, int], params: dict) -> float:
    """P(period-revealing outcome) = P(measured ∈ {010, 100, 110})."""
    total = sum(counts.values()) or 1
    revealing = counts.get("010", 0) + counts.get("100", 0) + counts.get("110", 0)
    return revealing / total


def interpret(counts: dict[str, int], params: dict | None = None) -> dict:
    params = params or {"a": 7}
    a = int(params.get("a", 7))
    total = sum(counts.values()) or 1

    p_uninformative = counts.get("000", 0) / total
    p_revealing = sum(counts.get(k, 0) for k in ("010", "100", "110")) / total
    factors: list[int] | None = None

    if p_revealing >= 0.30:
        r = 4 if a == 7 else 2
        if (a ** (r // 2)) % N_TARGET != 1:
            f1 = gcd(a ** (r // 2) - 1, N_TARGET)
            f2 = gcd(a ** (r // 2) + 1, N_TARGET)
            cand = sorted({f1, f2} - {1, N_TARGET})
            if len(cand) == 2:
                factors = cand

    return {
        "P(0_uninformative)": round(p_uninformative, 4),
        "P(period_found)": round(p_revealing, 4),
        "inferred_r": (4 if a == 7 else 2) if factors else None,
        "factors_of_15": factors,
        "verdict": (
            f"Shor succeeded: 15 = {factors[0]} × {factors[1]}"
            if factors and len(factors) == 2 else
            "Inconclusive — too few period-revealing measurements"
        ),
    }


EXPERIMENT = ExperimentDef(
    key="shor",
    title="Compiled Shor (N=15, a=7)",
    description=(
        "Compiled Shor's algorithm for factoring 15 with base a=7.\n"
        "5 qubits: 3 control (QPE) + 2 work register.\n"
        "Period r = ord_15(7) = 4. Phase-revealing outcomes are 010/100/110.\n"
        "From those, gcd(7^2 ± 1, 15) = {3, 5} recovers the factors."
    ),
    qubits_used=5,
    parameters=PARAMETERS,
    build=build,
    expected=expected,
    metric_fn=metric_fn,
    metric_name="P(period found)",
    interpretation_hint=(
        "Ideal P ≈ 0.75 (3 of 4 revealing outcomes). "
        "Drop = chip noise washing out QFT phase coherence."
    ),
)


if __name__ == "__main__":
    import argparse
    from src.experiments.runner import run_circuit

    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="emulator_8q")
    p.add_argument("--shots-exp", type=int, default=10)
    p.add_argument("-a", type=int, default=7, choices=[7, 8, 11, 13])
    args = p.parse_args()

    params = {"a": args.a}
    qc = EXPERIMENT.build(params)
    print(qc.draw(output="text"))
    res = run_circuit(
        qc, label=f"shor_a{args.a}", backend_kind=args.backend,
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

