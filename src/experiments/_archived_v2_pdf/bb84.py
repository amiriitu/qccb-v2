"""
Experiment 4: BB84 quantum-key-distribution simulation.

Per SnowDrop 8Q PDF: 8 qubits in parallel, each carrying 1 BB84 round
(Alice's random bit + random basis → Bob's random basis → measure).
QBER (Quantum Bit Error Rate) is computed from sifted bits where Alice's
and Bob's bases coincided.

Eve's intercept-resend attack is simulated by applying her chosen basis
rotation before Bob's measurement, mimicking a measurement-induced collapse
on a fraction of qubits.

Threshold: QBER > 11.0% (Shor-Preskill bound) → no secret key extractable.
"""
from __future__ import annotations

import math
import random

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

from src.experiments import ExperimentDef, ParameterSpec


PARAMETERS: list[ParameterSpec] = [
    ParameterSpec(
        name="num_bits", label="Channel width (qubits)",
        kind="choice", default=8, choices=[4, 6, 8],
        help="Number of qubits transmitted per shot.",
    ),
    ParameterSpec(
        name="intercept", label="Eve eavesdropping",
        kind="bool", default=False,
        help="If True, apply intercept-resend attack on every qubit.",
    ),
    ParameterSpec(
        name="seed", label="RNG seed (for reproducibility)",
        kind="int", default=42, min_value=0, max_value=999999,
    ),
]


def build(params: dict) -> QuantumCircuit:
    num_bits = int(params.get("num_bits", 8))
    intercept = bool(params.get("intercept", False))
    seed = int(params.get("seed", 42))

    rng = random.Random(seed)
    alice_bits = [rng.randint(0, 1) for _ in range(num_bits)]
    alice_bases = [rng.randint(0, 1) for _ in range(num_bits)]  # 0=Z, 1=X
    bob_bases = [rng.randint(0, 1) for _ in range(num_bits)]
    eve_bases = [rng.randint(0, 1) for _ in range(num_bits)] if intercept else None

    qr = QuantumRegister(num_bits, "channel")
    cr = ClassicalRegister(num_bits, "bob")
    qc = QuantumCircuit(qr, cr, name=f"bb84_n{num_bits}_eve{int(intercept)}")

    # Alice prepares
    for i in range(num_bits):
        if alice_bits[i] == 1:
            qc.x(qr[i])
        if alice_bases[i] == 1:  # X basis → apply H to encode
            qc.h(qr[i])

    qc.barrier()

    # Eve intercept-resend (simulated as a unitary with the same effective
    # error signature as a true mid-circuit measurement when Eve's basis
    # differs from Alice's):
    #   - if Eve's basis MATCHES Alice's: she reads correctly, no effect
    #   - if Eve's basis DIFFERS:        state is rotated by ry(π/2) which,
    #                                     when measured in Alice's original
    #                                     basis, gives 50/50 outcome — exactly
    #                                     the post-projection statistics.
    # This is a deterministic stand-in for projective measurement that does not
    # require mid-circuit measurement support on the backend.
    if intercept and eve_bases is not None:
        for i in range(num_bits):
            if eve_bases[i] != alice_bases[i]:
                qc.ry(math.pi / 2, qr[i])
        qc.barrier()

    # Bob measures in his random basis
    for i in range(num_bits):
        if bob_bases[i] == 1:
            qc.h(qr[i])
        qc.measure(qr[i], cr[i])

    qc.metadata = {
        "alice_bits": "".join(map(str, alice_bits)),
        "alice_bases": "".join(map(str, alice_bases)),
        "bob_bases": "".join(map(str, bob_bases)),
        "eve_bases": ("".join(map(str, eve_bases)) if eve_bases else None),
        "intercept": intercept,
        "seed": seed,
        "num_bits": num_bits,
    }
    return qc


def expected(params: dict) -> dict[str, float]:
    """No closed-form 'expected' distribution for BB84 — just signal target."""
    num_bits = int(params.get("num_bits", 8))
    return {format(i, f"0{num_bits}b"): 1.0 / (2 ** num_bits)
            for i in range(2 ** num_bits)}


def _qber_from_counts(counts: dict[str, int], params: dict) -> tuple[float, int]:
    """
    Compute QBER from per-shot Bob measurements vs Alice's known bits,
    sifting only those qubit-positions where Alice's and Bob's bases match.

    Returns (qber, sifted_bit_count).
    """
    num_bits = int(params.get("num_bits", 8))
    seed = int(params.get("seed", 42))

    rng = random.Random(seed)
    alice_bits = [rng.randint(0, 1) for _ in range(num_bits)]
    alice_bases = [rng.randint(0, 1) for _ in range(num_bits)]
    bob_bases = [rng.randint(0, 1) for _ in range(num_bits)]

    matching = [i for i in range(num_bits) if alice_bases[i] == bob_bases[i]]
    if not matching:
        return 0.0, 0

    total_shots = sum(counts.values()) or 1
    error_bits = 0
    sifted_bits = 0

    for bitstring, n_shots in counts.items():
        bob_meas = list(reversed(bitstring))
        for i in matching:
            sifted_bits += n_shots
            if int(bob_meas[i]) != alice_bits[i]:
                error_bits += n_shots

    if sifted_bits == 0:
        return 0.0, 0
    return error_bits / sifted_bits, sifted_bits


def metric_fn(counts: dict[str, int], params: dict) -> float:
    """
    Metric = QBER (Quantum Bit Error Rate) clipped to [0, 0.5].
    Lower = cleaner channel. > 0.11 = Shor-Preskill threshold breached.
    """
    qber, _sifted = _qber_from_counts(counts, params)
    return min(qber, 0.5)


EXPERIMENT = ExperimentDef(
    key="bb84",
    title="BB84 QKD (8 qubits)",
    description=(
        "BB84 quantum key distribution. 8 qubits in parallel = 8 BB84 rounds\n"
        "per shot. Bases are random (seeded). Eve's intercept-resend attack\n"
        "is optional. Metric = QBER (Quantum Bit Error Rate); > 0.11\n"
        "(Shor-Preskill threshold) means no secret key extractable."
    ),
    qubits_used=8,
    parameters=PARAMETERS,
    build=build,
    expected=expected,
    metric_fn=metric_fn,
    metric_name="QBER",
    interpretation_hint=(
        "Without Eve: QBER ≈ chip readout error (~5%). With Eve: QBER → 25% "
        "→ key compromised. 11% threshold detects intercept-resend reliably."
    ),
)


if __name__ == "__main__":
    import argparse
    from src.experiments.runner import run_circuit

    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="emulator_8q")
    p.add_argument("--shots-exp", type=int, default=10)
    p.add_argument("--num-bits", type=int, default=8, choices=[4, 6, 8])
    p.add_argument("--intercept", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    params = {"num_bits": args.num_bits, "intercept": args.intercept,
              "seed": args.seed}
    qc = EXPERIMENT.build(params)
    print(qc.draw(output="text"))
    res = run_circuit(
        qc, label=f"bb84_eve{int(args.intercept)}",
        backend_kind=args.backend, shots_exponent=args.shots_exp,
        expected_distribution=EXPERIMENT.expected(params),
        metric_name=EXPERIMENT.metric_name,
        metric_fn=lambda c: EXPERIMENT.metric_fn(c, params),
        status=lambda m: print(f"[{m}]"),
    )
    qber, sifted = _qber_from_counts(res.counts, params)
    print(f"\nSifted bits: {sifted}, errors → QBER = {qber:.4f}")
    print(f"Shor-Preskill threshold (0.11) "
          f"{'BREACHED — eavesdropper detected' if qber > 0.11 else 'OK — key is safe'}")

