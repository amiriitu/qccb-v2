"""
SCI_HW — hardware-aware extension of the Security Cost Index from the paper.

Paper's static SCI:
    SCI = Overhead × Size × Complexity   (DOI 10.53364/24138614_2026_40_1_11)

The paper introduces dynamic SCI(h⃗, n⃗, p⃗) but only postulates the form.
Here we operationalize the hardware vector h⃗ with quantities measured directly
on Bauman Octillion Snowdrop 4q ver2:

    SCI_HW(exp, backend) = Time × Error × Routing
                         = (T_obs / T_ideal) × |M_obs − M_ideal| × (D_t / D_l)

where
    T_obs, T_ideal  — measured wall-clock time on `backend` and on ideal sim
    M_obs           — empirical metric value (e.g. fidelity) on `backend`
    M_ideal         — empirical metric value on ideal sim (the achievable baseline,
                      not necessarily 1; for Shor the ideal is ≈0.5)
    D_t             — transpiled circuit depth on the chip
    D_l             — transpiled depth on the ideal backend (no coupling/SWAPs)

Properties:
- Dimensionless. Lower = better.
- For an ideal noiseless simulator: T-factor=1, error-factor=0 → SCI_HW = 0.
- Captures the three dominant cost vectors of NISQ hardware:
    1. Latency (queue + execute on real device)
    2. Deviation from achievable baseline (gate + readout noise)
    3. Topology (extra SWAPs from coupling-map routing)
- Distance from the *ideal-simulator measurement* (not from 1.0) means the
  formula works for any metric: Bell (ideal≈1), Shor (ideal≈0.5), TVD, etc.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SCI_HW:
    """One hardware-aware SCI evaluation for (experiment, backend)."""
    experiment: str
    backend: str
    time_factor: float       # T_real / T_ideal   (≥ 1 typically)
    error_factor: float      # 1 - F_observed     (∈ [0, 1])
    routing_factor: float    # D_transpiled / D_logical  (≥ 1 typically)
    sci_value: float         # product of the above
    interpretation: str

    def to_dict(self) -> dict:
        return {
            "experiment": self.experiment,
            "backend": self.backend,
            "time_factor": self.time_factor,
            "error_factor": self.error_factor,
            "routing_factor": self.routing_factor,
            "sci_hw": self.sci_value,
            "interpretation": self.interpretation,
        }


def _interpret(sci: float) -> str:
    """
    SCI_HW interpretation. Thresholds calibrated against the time_factor ranges
    observed for the Octillion network round-trip (~10-15× for emulator,
    ~50-100× for real hardware via queue), so the labels reflect the algorithm's
    information yield, not the latency.
    """
    if sci < 0.5:
        return "★★★★★ Production-ready"
    if sci < 2.0:
        return "★★★★ Strong — minor noise"
    if sci < 5.0:
        return "★★★ Workable — algorithm succeeds with visible noise"
    if sci < 15.0:
        return "★★ Marginal — useful only as demonstration"
    return "★ Hardware-limited — algorithm output dominated by noise"


def compute_sci_hw(
    experiment: str,
    backend: str,
    metric_observed: float,
    metric_ideal: float,
    time_observed_s: float,
    time_ideal_s: float,
    transpiled_depth: int,
    logical_depth: int,
) -> SCI_HW:
    """Compute SCI_HW for one (experiment, backend) pair from measured quantities.

    `metric_ideal` is the metric value on the IDEAL simulator (the achievable
    baseline). The error factor is |observed − ideal|, which equals 0 when the
    backend matches the noiseless reference exactly.
    """
    time_factor = max(1.0, time_observed_s / max(time_ideal_s, 1e-6))
    error_factor = abs(metric_observed - metric_ideal)
    routing_factor = max(1.0, transpiled_depth / max(logical_depth, 1))

    sci_value = time_factor * error_factor * routing_factor

    return SCI_HW(
        experiment=experiment,
        backend=backend,
        time_factor=time_factor,
        error_factor=error_factor,
        routing_factor=routing_factor,
        sci_value=sci_value,
        interpretation=_interpret(sci_value),
    )

