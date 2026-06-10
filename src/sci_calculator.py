"""
Interactive SCI Calculator — implements both formulas:

1. Original SCI (Zhailin et al., Scientific journal 'Bulletin of the CAA' №1(40), 2026):
       SCI = Overhead × Size × Complexity
   with logarithmic normalization (Eq. 1) for cross-algorithm comparison.

2. Hardware-aware SCI_HW extension (this work):
       SCI_HW = (T_obs / T_ideal) × |M_obs − M_ideal| × (D_t / D_l)
   with empirical hardware-factor from a measured calibration.

Use cases:
  - CLI:  python -m src.sci_calculator
  - GUI:  see src.sci_calculator_gui (Tk dialog with both formulas live)
  - API:  from src.sci_calculator import SCIFormula, SCIInputs, compute_sci
"""
from __future__ import annotations

import argparse
import math
import sys
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Optional

if sys.platform == "win32":
    with suppress(AttributeError, OSError):
        sys.stdout.reconfigure(encoding="utf-8")


# ============================================================================
# Original 3-factor SCI (Scientific journal 'Bulletin of the CAA' 2026, Eq. base + Eq. 1 normalization)
# ============================================================================

@dataclass
class SCIInputs:
    """Inputs for the 3-factor SCI of Zhailin et al. (2026), Eq. 1."""
    # Times in same units (typically ms)
    t_pqc: float = 0.0
    t_classical: float = 0.0
    # Sizes in same units (typically bytes)
    size_pqc: float = 0.0
    size_classical: float = 0.0
    # Complexity score: 1.0 (simple lattice) … 2.5 (complex code-based).
    # Calibrated values: Kyber=1.0, Dilithium=1.07, McEliece=2.5, etc.
    complexity: float = 1.0
    # NIST security level (1, 2, 3, or 5) — paper's NIST-scaling factor.
    nist_level: int = 3
    # Optional cohort context for paper's Eq. (1) log-normalization across
    # multiple algorithms. If supplied, sci_normalized is filled in too.
    cohort_t: list[float] = field(default_factory=list)
    cohort_size: list[float] = field(default_factory=list)


@dataclass
class SCIResult:
    overhead_factor: float          # 1 + log10(t_pqc/t_classical) / 5
    size_penalty: float              # 1 + log10(size_pqc/size_classical) / 5
    complexity_score: float          # 1.0 … 2.5
    nist_factor: float               # 3 / NIST_level
    sci_raw: float                   # product of all four factors
    sci_normalized: Optional[float] = None
    interpretation: str = ""

    def to_dict(self) -> dict:
        return {
            "overhead_factor": round(self.overhead_factor, 4),
            "size_penalty": round(self.size_penalty, 4),
            "complexity_score": round(self.complexity_score, 4),
            "nist_factor": round(self.nist_factor, 4),
            "SCI": round(self.sci_raw, 4),
            "SCI_normalized": (round(self.sci_normalized, 4)
                                if self.sci_normalized is not None else None),
            "interpretation": self.interpretation,
        }


def _log_normalize(x: float, cohort: list[float]) -> float:
    """
    Paper Eq. (1):  r_ij = 1 − (log x − log min) / (log max − log min)
    Cost-criterion (lower is preferable); maps cohort to [0, 1], 1 = best.
    """
    if not cohort:
        return x
    pos = [v for v in cohort + [x] if v > 0]
    if not pos:
        return x
    lo = min(pos)
    hi = max(pos)
    if hi == lo:
        return 1.0
    return 1.0 - (math.log(max(x, 1e-12)) - math.log(lo)) / (math.log(hi) - math.log(lo))


def _interpret_paper_sci(sci: float) -> str:
    if sci < 1.0:
        return "WIN — faster and safer than classical baseline"
    if sci < 2.0:
        return "★★★★★ Production-ready (recommended for hybrid deployments)"
    if sci < 5.0:
        return "★★★ Workable (visible overhead, fine for non-real-time)"
    if sci < 10.0:
        return "★★ Marginal (real-time apps will feel it)"
    return "★ Specialized use only (huge key/signature sizes)"


def compute_sci(inp: SCIInputs) -> SCIResult:
    """
    SCI per Zhailin et al. (2026), Bulletin of the CAA №1(40), pp 124-139,
    DOI 10.53364/24138614_2026_40_1_11.

    Formula:
        SCI = (1 + log10(t_pqc/t_classical)/5)        ← Overhead Factor
            × (1 + log10(size_pqc/size_classical)/5)  ← Size Penalty
            × Complexity                              ← 1.0 (lattice) … 2.5 (code)
            × (3 / NIST_level)                         ← Security scaling

    Reproduces published reference values (paper Tables 3-5):
      Kyber-768 (NIST 3, complexity 1.0):       SCI = 1.42 ✓
      Dilithium-3 (NIST 3, complexity 1.07):    SCI = 1.67 ✓
      Classic McEliece (NIST 5, complexity 2.5): SCI ≈ 3.2 ✓
    """
    if inp.t_classical <= 0:
        raise ValueError("t_classical must be > 0")
    if inp.size_classical <= 0:
        raise ValueError("size_classical must be > 0")
    if inp.nist_level <= 0:
        raise ValueError("nist_level must be > 0")

    overhead_ratio = max(inp.t_pqc / inp.t_classical, 1.0)
    size_ratio = max(inp.size_pqc / inp.size_classical, 1.0)
    complexity = max(1.0, inp.complexity)

    overhead_factor = 1.0 + math.log10(overhead_ratio) / 5.0
    size_factor = 1.0 + math.log10(size_ratio) / 5.0
    nist_factor = 3.0 / inp.nist_level

    sci_raw = overhead_factor * size_factor * complexity * nist_factor

    sci_normalized: Optional[float] = None
    if inp.cohort_t and inp.cohort_size:
        n_t = _log_normalize(inp.t_pqc, inp.cohort_t)
        n_s = _log_normalize(inp.size_pqc, inp.cohort_size)
        sci_normalized = (1.0 - n_t) * (1.0 - n_s) * complexity * nist_factor

    return SCIResult(
        overhead_factor=overhead_factor,
        size_penalty=size_factor,
        complexity_score=complexity,
        nist_factor=nist_factor,
        sci_raw=sci_raw,
        sci_normalized=sci_normalized,
        interpretation=_interpret_paper_sci(sci_raw),
    )


# ============================================================================
# Hardware-aware SCI_HW extension (this work)
# ============================================================================

@dataclass
class SCIHWInputs:
    """Inputs for SCI_HW — measured against an ideal-baseline run."""
    t_obs_s: float = 0.0
    t_ideal_s: float = 0.0
    metric_obs: float = 0.0
    metric_ideal: float = 0.0
    depth_transpiled: int = 0
    depth_logical: int = 0


@dataclass
class SCIHWResult:
    time_factor: float
    error_factor: float
    routing_factor: float
    sci_hw: float
    interpretation: str

    def to_dict(self) -> dict:
        return {
            "time_factor": round(self.time_factor, 4),
            "error_factor": round(self.error_factor, 4),
            "routing_factor": round(self.routing_factor, 4),
            "SCI_HW": round(self.sci_hw, 4),
            "interpretation": self.interpretation,
        }


def _interpret_hw(sci_hw: float) -> str:
    if sci_hw < 0.5:
        return "★★★★★ Production-ready"
    if sci_hw < 2.0:
        return "★★★★ Strong — minor noise"
    if sci_hw < 5.0:
        return "★★★ Workable — noise visible but algorithm succeeds"
    if sci_hw < 15.0:
        return "★★ Marginal — useful only as demonstration"
    return "★ Hardware-limited — algorithm output dominated by noise"


def compute_sci_hw(inp: SCIHWInputs) -> SCIHWResult:
    """Hardware-aware SCI_HW (this work, extends Scientific journal 'Bulletin of the CAA' 2026)."""
    if inp.t_ideal_s <= 0:
        raise ValueError("t_ideal_s must be > 0")
    if inp.depth_logical <= 0:
        raise ValueError("depth_logical must be > 0")

    time_factor = max(1.0, inp.t_obs_s / inp.t_ideal_s)
    error_factor = abs(inp.metric_obs - inp.metric_ideal)
    routing_factor = max(1.0, inp.depth_transpiled / inp.depth_logical)

    sci_hw = time_factor * error_factor * routing_factor

    return SCIHWResult(
        time_factor=time_factor,
        error_factor=error_factor,
        routing_factor=routing_factor,
        sci_hw=sci_hw,
        interpretation=_interpret_hw(sci_hw),
    )


# ============================================================================
# Pre-computed examples from project measurements
# ============================================================================

EXAMPLES: dict[str, dict] = {
    "Kyber768 vs RSA-2048 (paper Table 3 + Table 5)": {
        "formula": "paper_sci",
        "inputs": dict(
            t_pqc=568.16, t_classical=29.65,
            size_pqc=1184, size_classical=256,
            complexity=1.0,         # paper's Kyber complexity (simple lattice)
            nist_level=3,
        ),
        "expected_sci_value": 1.42,
        "note": (
            "Published SCI for Kyber-768 from Zhailin et al. 2026, page 134. "
            "Note that t_pqc here reflects the Python-wrapper application-level "
            "latency (FFI overhead), not the algorithmic lower bound."
        ),
    },
    "Dilithium3 vs RSA-2048 (paper Table 4 + 5)": {
        "formula": "paper_sci",
        "inputs": dict(
            t_pqc=684.8, t_classical=29.65,
            size_pqc=3309, size_classical=256,
            complexity=1.07,        # paper's Dilithium complexity
            nist_level=3,
        ),
        "expected_sci_value": 1.67,
    },
    "Classic McEliece-6688128 vs RSA-2048 (paper §3.13)": {
        "formula": "paper_sci",
        "inputs": dict(
            t_pqc=500.0, t_classical=29.65,
            size_pqc=1357824, size_classical=256,
            complexity=2.5,         # code-based, highest complexity
            nist_level=5,
        ),
        "expected_sci_value": 3.2,
        "note": "McEliece NIST-5 variant — huge public keys dominate SCI",
    },
    "Snowdrop Bell on real chip (SCI_HW)": {
        "formula": "sci_hw",
        "inputs": dict(
            t_obs_s=1.10, t_ideal_s=0.09,
            metric_obs=0.962, metric_ideal=1.0,
            depth_transpiled=10, depth_logical=3,
        ),
    },
    "Snowdrop Shor N=15 on real chip (SCI_HW)": {
        "formula": "sci_hw",
        "inputs": dict(
            t_obs_s=1.07, t_ideal_s=0.09,
            metric_obs=0.4961, metric_ideal=0.50,
            depth_transpiled=11, depth_logical=5,
        ),
    },
}


# ============================================================================
# CLI
# ============================================================================

def _print_paper_sci(res: SCIResult, header: str = "") -> None:
    if header:
        print(f"\n=== {header} ===")
    print(f"  Overhead factor (1 + log10(T_pqc/T_classical)/5):    {res.overhead_factor:.4f}")
    print(f"  Size penalty   (1 + log10(Size_pqc/Size_classical)/5): {res.size_penalty:.4f}")
    print(f"  Complexity score (1.0 simple … 2.5 complex):           {res.complexity_score:.4f}")
    print(f"  NIST factor (3 / NIST_level):                          {res.nist_factor:.4f}")
    print(f"  ──────────────────────────────────────────")
    print(f"  SCI:                                                   {res.sci_raw:.4f}")
    if res.sci_normalized is not None:
        print(f"  SCI (cohort-normalized, Eq. 1):                        {res.sci_normalized:.4f}")
    print(f"  Interpretation:                                        {res.interpretation}")


def _print_sci_hw(res: SCIHWResult, header: str = "") -> None:
    if header:
        print(f"\n=== {header} ===")
    print(f"  Time factor   (T_obs / T_ideal):          ×{res.time_factor:.2f}")
    print(f"  Error factor  |M_obs − M_ideal|:           {res.error_factor:.4f}")
    print(f"  Routing factor (D_transpiled / D_logical): ×{res.routing_factor:.2f}")
    print(f"  ──────────────────────────────────────────")
    print(f"  SCI_HW:                                    {res.sci_hw:.4f}")
    print(f"  Interpretation:                            {res.interpretation}")


def _interactive_paper_sci() -> None:
    print("\n--- Published 3-factor SCI (Zhailin et al. 2026) ---")
    print("Enter time in milliseconds, sizes in bytes.\n")
    t_pqc = float(input("  T_pqc       [ms]     = "))
    t_classical = float(input("  T_classical [ms]     = "))
    size_pqc = float(input("  Size_pqc       [B]   = "))
    size_classical = float(input("  Size_classical [B]   = "))
    complexity = float(input("  Complexity (1.0=lattice … 2.5=code) = "))
    nist_level = int(input("  NIST_level (1/2/3/5) = "))
    res = compute_sci(SCIInputs(
        t_pqc=t_pqc, t_classical=t_classical,
        size_pqc=size_pqc, size_classical=size_classical,
        complexity=complexity, nist_level=nist_level,
    ))
    _print_paper_sci(res, "Result")


def _interactive_sci_hw() -> None:
    print("\n--- SCI_HW: hardware-aware extension (this work) ---")
    print("Enter timing in seconds, fidelity/metric as fraction.\n")
    t_obs = float(input("  T_obs   [s]               = "))
    t_ideal = float(input("  T_ideal [s]               = "))
    m_obs = float(input("  M_obs   (observed metric) = "))
    m_ideal = float(input("  M_ideal (ideal-sim metric)= "))
    d_t = int(input("  D_transpiled (gate depth) = "))
    d_l = int(input("  D_logical    (gate depth) = "))
    res = compute_sci_hw(SCIHWInputs(
        t_obs_s=t_obs, t_ideal_s=t_ideal,
        metric_obs=m_obs, metric_ideal=m_ideal,
        depth_transpiled=d_t, depth_logical=d_l,
    ))
    _print_sci_hw(res, "Result")


def _show_examples() -> None:
    for name, ex in EXAMPLES.items():
        if ex["formula"] == "paper_sci":
            res = compute_sci(SCIInputs(**ex["inputs"]))
            _print_paper_sci(res, name)
        else:
            res = compute_sci_hw(SCIHWInputs(**ex["inputs"]))
            _print_sci_hw(res, name)
        if ex.get("note"):
            print(f"  Note: {ex['note']}")


def main():
    p = argparse.ArgumentParser(
        description="Interactive SCI calculator (paper formula + SCI_HW extension)"
    )
    p.add_argument("--examples", action="store_true",
                   help="Show pre-computed examples and exit")
    p.add_argument("--formula", choices=["paper", "hw", "both"], default="both",
                   help="Which SCI formula to use interactively")
    args = p.parse_args()

    if args.examples:
        _show_examples()
        return

    print("=" * 60)
    print("SCI Calculator")
    print("=" * 60)
    print("References:")
    print("  - Original SCI: Zhailin, Bekarystankyzy, Aktanova,")
    print("    Scientific journal 'Bulletin of the CAA' №1(40), 2026, DOI 10.53364/24138614_2026_40_1_11")
    print("  - SCI_HW extension: this work (Bauman Octillion + measured chip)")
    print()

    if args.formula in ("paper", "both"):
        _interactive_paper_sci()
    if args.formula in ("hw", "both"):
        _interactive_sci_hw()


if __name__ == "__main__":
    main()

