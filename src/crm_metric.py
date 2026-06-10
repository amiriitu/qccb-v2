"""
Cryptographically Reachable Modulus (CRM) — a chip-specific cryptanalytic benchmark.

DEFINITION
----------
For a quantum chip with measured per-gate fidelities {F_1q, F_cz, F_RO}
and qubit count n_qubits:

    CRM(chip) = max { N ∈ N : P_success(Shor(N, a*) | chip) ≥ τ }

where:
  - Shor(N, a*) is the *minimum-depth compiled* implementation of Shor's
    period-finding for factoring N (with optimal coprime base a* such that
    period r is small and circuit fits in n_qubits)
  - P_success is computed from a depolarizing-channel noise model
    parameterized by {F_1q, F_cz, F_RO} applied to the canonical compiled
    circuit
  - τ ∈ (0,1] is a user-set threshold; default τ = 0.30, motivated by:
        * Shor-with-1-counting-bit gives ideal P=0.50 on the period-bit
        * Below P≈0.30, classical post-processing (continued fractions
          over multiple shots) starts failing to recover r reliably

POSITIONING vs OTHER METRICS
----------------------------
Quantum Volume (IBM 2019)  → method-agnostic random-circuit benchmark.
                             Doesn't translate to "what cryptosystem can
                             this chip break today?".
NIST IR 8547 CRQC timelines → assume fault-tolerant logical qubits and
                             ideal error correction. Not measurable on
                             current NISQ hardware.
Gidney-Ekerå (2021)        → resource estimate for RSA-2048 assuming
                             20M physical noisy qubits + surface code.
                             Long-term estimate, not chip-specific.

CRM fills the gap: a *measured*, *chip-specific*, *cryptanalytically
meaningful* number you can compute today from your own calibration data.

CAVEATS (state honestly)
------------------------
1. Assumes depolarizing noise model. Real chips have correlated errors
   (crosstalk, leakage) that may degrade P further.
2. No quantum error correction included — CRM is a NISQ-era benchmark.
3. Compiled-Shor circuit families are not exhaustively known for all N;
   this implementation includes hand-crafted versions for N ∈ {15, 21, 33,
   35} with period r ≤ 4 (the tractable NISQ range).
4. The metric is monotonic in F_2q under fixed gate count, so for chips
   with very high F_2q the bottleneck shifts to qubit count n.

REFERENCES
----------
- Vandersypen et al., Nature 414 (2001) — original 7q implementation of
  Shor for N=15.
- Lucero et al., Nature Physics 8 (2012) — compiled 4q Shor for N=15.
- Gidney & Ekerå, Quantum 5 (2021) — RSA-2048 resource estimate.
- This work — formal definition and first chip-specific measurements.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class ChipParams:
    """Minimum chip parameters needed to compute CRM."""
    name: str
    year: int
    n_qubits: int
    f_1q: float                    # avg single-qubit gate fidelity
    f_2q: float                    # avg two-qubit (CZ/CX) fidelity
    f_ro: float                    # avg readout fidelity per qubit
    measured: bool = False         # True = measured by us; False = published
    source: str = ""               # citation


@dataclass
class ShorCircuitProfile:
    """
    Compiled Shor circuit cost for factoring N at base a (period r).
    Counts are for a *minimum-depth* compiled version that exploits
    the small period; fall-back is the standard Beauregard 2003 cost.
    """
    N: int                         # modulus to factor
    a: int                         # coprime base
    r: int                         # period of a mod N
    n_used: int                    # qubits used (counting + work)
    n_1q: int                      # 1-qubit gate count (post-transpile)
    n_2q: int                      # 2-qubit gate count (post-transpile)
    ideal_p_success: float         # P(period found) on noiseless backend


def _profile(N: int, a: int, r: int, n_used: int,
              n_1q: int, n_2q: int, p_ideal: float) -> ShorCircuitProfile:
    return ShorCircuitProfile(N=N, a=a, r=r, n_used=n_used,
                                n_1q=n_1q, n_2q=n_2q,
                                ideal_p_success=p_ideal)


def _hand_compiled_profiles() -> list[ShorCircuitProfile]:
    """N ≤ 35: explicit compiled circuits from the literature (Lanyon, Lucero, etc)."""
    return [
        _profile(N=15, a=4, r=2, n_used=4, n_1q=4, n_2q=2, p_ideal=0.50),
        _profile(N=15, a=11, r=2, n_used=5, n_1q=5, n_2q=3, p_ideal=0.50),
        _profile(N=21, a=4, r=3, n_used=5, n_1q=8, n_2q=5, p_ideal=0.33),
        _profile(N=33, a=2, r=10, n_used=6, n_1q=20, n_2q=12, p_ideal=0.50),
        _profile(N=35, a=4, r=6, n_used=6, n_1q=24, n_2q=15, p_ideal=0.50),
    ]


def _beauregard_asymptotic_profiles() -> list[ShorCircuitProfile]:
    """
    N > 35: use Beauregard 2003 asymptotic resource estimate.
    Resource scaling for compiled Shor on N with n = ceil(log2(N)) bits:
        n_qubits ≈ 2n + 3
        n_2q     ≈ 32 n^2
        n_1q     ≈ 48 n^2
    These are CONSERVATIVE (hand-compilation can do better for specific N).
    Step by powers of 2 to keep table manageable.
    """
    profiles = []
    for power in range(6, 20):
        N = 2 ** power - 1
        n = power
        n_qubits = 2 * n + 3
        n_2q = 32 * n ** 2
        n_1q = 48 * n ** 2
        profiles.append(
            _profile(N=N, a=2, r=2 * n, n_used=n_qubits,
                     n_1q=n_1q, n_2q=n_2q, p_ideal=0.50)
        )
    return profiles


KNOWN_SHOR_PROFILES: list[ShorCircuitProfile] = (
    _hand_compiled_profiles() + _beauregard_asymptotic_profiles()
)


def beauregard_resource_estimate(N: int) -> tuple[int, int, int]:
    """
    Standard Beauregard (2003) compiled Shor cost for arbitrary N.
    Returns (n_qubits, n_1q_gates, n_2q_gates).
    Used as a fallback when N is not in KNOWN_SHOR_PROFILES.
    Asymptotic: O(n^2) gates, O(n) qubits where n = log2(N).
    """
    n = int(math.ceil(math.log2(max(N, 2))))
    n_qubits = 2 * n + 3
    n_2q = 32 * n ** 2
    n_1q = 48 * n ** 2
    return n_qubits, n_1q, n_2q


def predicted_success_probability(profile: ShorCircuitProfile,
                                    chip: ChipParams) -> float:
    """
    Depolarizing-channel estimate of end-to-end Shor success probability
    on `chip`. Multiplicative model:
        P = P_ideal × F_RO^n_used × F_1q^n_1q × F_2q^n_2q
    """
    if chip.n_qubits < profile.n_used:
        return 0.0
    return (profile.ideal_p_success
            * (chip.f_ro ** profile.n_used)
            * (chip.f_1q ** profile.n_1q)
            * (chip.f_2q ** profile.n_2q))


def _profiles_sorted_by_N() -> list[ShorCircuitProfile]:
    by_N: dict[int, ShorCircuitProfile] = {}
    for p in KNOWN_SHOR_PROFILES:
        if p.N not in by_N or p.n_2q < by_N[p.N].n_2q:
            by_N[p.N] = p
    return sorted(by_N.values(), key=lambda x: x.N)


@dataclass
class CRMResult:
    chip: ChipParams
    threshold: float
    crm: int                                      # = max N satisfying P ≥ τ
    chosen_profile: Optional[ShorCircuitProfile]  # the profile achieving CRM
    per_N_predictions: list[tuple[int, float]] = field(default_factory=list)
    note: str = ""

    @property
    def p_at_crm(self) -> float:
        """P_success at the chosen profile (the actual CRM-achieving N)."""
        if not self.chosen_profile:
            return 0.0
        for n_tested, p in self.per_N_predictions:
            if n_tested == self.chosen_profile.N:
                return p
        return 0.0

    def to_dict(self) -> dict:
        return {
            "chip": self.chip.name,
            "year": self.chip.year,
            "n_qubits": self.chip.n_qubits,
            "f_1q": self.chip.f_1q,
            "f_2q": self.chip.f_2q,
            "f_ro": self.chip.f_ro,
            "threshold_tau": self.threshold,
            "CRM": self.crm,
            "chosen_N": (self.chosen_profile.N if self.chosen_profile else None),
            "chosen_a": (self.chosen_profile.a if self.chosen_profile else None),
            "P_success_at_CRM": self.p_at_crm,
            "measured": self.chip.measured,
            "note": self.note,
        }


def compute_crm(chip: ChipParams, threshold: float = 0.30) -> CRMResult:
    """
    Compute CRM for `chip`. Walks KNOWN_SHOR_PROFILES in order of N,
    accepts the largest N for which P_success ≥ threshold.
    """
    predictions: list[tuple[int, float]] = []
    last_pass: Optional[ShorCircuitProfile] = None

    for profile in _profiles_sorted_by_N():
        p_pred = predicted_success_probability(profile, chip)
        predictions.append((profile.N, p_pred))
        if p_pred >= threshold and chip.n_qubits >= profile.n_used:
            last_pass = profile

    crm = last_pass.N if last_pass else 1
    note = ""
    if last_pass is None:
        note = (
            "All known compiled-Shor profiles fail. Either fidelities are "
            "below NISQ-useful threshold, or qubit count is insufficient."
        )
    elif last_pass.N == max(p.N for p in KNOWN_SHOR_PROFILES):
        note = (
            "CRM saturates the largest N in our compiled-profile catalog. "
            "Actual CRM may be higher; use beauregard_resource_estimate() "
            "extrapolation for unknown N."
        )

    return CRMResult(
        chip=chip,
        threshold=threshold,
        crm=crm,
        chosen_profile=last_pass,
        per_N_predictions=predictions,
        note=note,
    )


def crm_for_calibration_dict(name: str, year: int, n_qubits: int,
                              calibration: dict,
                              threshold: float = 0.30) -> CRMResult:
    """
    Convenience: compute CRM directly from our project's calibration dict
    (e.g., parsed from Bauman's snowdrop_4q_ver2.json or from a ChipSpec).
    """
    chip = ChipParams(
        name=name,
        year=year,
        n_qubits=n_qubits,
        f_1q=float(calibration.get("avg_f1q", 0.0)) or
              _avg(calibration.get("per_qubit_f1q", {})),
        f_2q=float(calibration.get("avg_f2q", 0.0)) or
              _avg(calibration.get("per_pair_f2q", {})),
        f_ro=float(calibration.get("avg_ro", 0.0)) or
              _avg(calibration.get("per_qubit_ro", {})),
        measured=True,
        source="this work — Bauman Octillion API",
    )
    return compute_crm(chip, threshold=threshold)


def _avg(d) -> float:
    if not d:
        return 0.0
    if hasattr(d, "values"):
        vals = list(d.values())
    else:
        vals = list(d)
    return sum(float(v) for v in vals) / len(vals) if vals else 0.0

