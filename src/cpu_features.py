"""
CPU instruction-set / hardware-acceleration detection.

Reports availability of:
  - AVX2          — used by liboqs AVX2 path for ML-KEM / ML-DSA (~2× speedup
                    over reference C per dissertation §3.22)
  - AVX-512       — used by some liboqs builds for further speedup
  - AES-NI        — used by OpenSSL / hashlib for AES; needed for sub-millisecond AES
  - SHA-NI (SHA extensions) — used by hashlib for SHA-256

Detection strategy (in order):
  1. `py-cpuinfo` if installed (best — actually issues CPUID)
  2. `/proc/cpuinfo` on Linux
  3. PowerShell `Get-WmiObject Win32_Processor` on Windows
  4. CPU-family heuristic via `platform.processor()` (Intel ≥ Haswell '13,
     AMD ≥ Excavator/Zen '15-17 have AVX2)

The dissertation explicitly notes (§3.22) that AVX2 is missing from the
reference pqcrypto C path; this module is what we use to detect when the
host SUPPORTS AVX2 so we can fairly attribute the speedup to the vectorised
liboqs path vs. portable C.

References:
  - Intel Intrinsics Guide (AVX2 ISA, 2013+)
  - liboqs build flags OQS_USE_AVX2_INSTRUCTIONS / OQS_USE_AVX_INSTRUCTIONS
  - Open Quantum Safe project: https://openquantumsafe.org/
"""
from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CpuFeatures:
    """Detected CPU capabilities relevant to crypto vectorisation."""
    brand: str
    arch: str            # 'x86_64' / 'arm64' / 'aarch64' / ...
    avx2: bool
    avx512: bool
    aes_ni: bool
    sha_ni: bool
    detection_method: str   # 'cpuinfo' | 'proc' | 'wmic' | 'heuristic'

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================================
# Backend 1 — py-cpuinfo (best — actually issues CPUID)
# ============================================================================

def _try_pycpuinfo() -> CpuFeatures | None:
    try:
        import cpuinfo
    except ImportError:
        return None
    try:
        info = cpuinfo.get_cpu_info()
        flags = {f.lower() for f in info.get("flags", [])}
        return CpuFeatures(
            brand=info.get("brand_raw") or info.get("brand") or "unknown",
            arch=info.get("arch") or platform.machine(),
            avx2="avx2" in flags,
            avx512=any(f.startswith("avx512") for f in flags),
            aes_ni="aes" in flags or "aes-ni" in flags,
            sha_ni="sha_ni" in flags or "sha-ni" in flags or "sha" in flags,
            detection_method="cpuinfo",
        )
    except Exception:
        return None


# ============================================================================
# Backend 2 — /proc/cpuinfo on Linux
# ============================================================================

def _try_proc_cpuinfo() -> CpuFeatures | None:
    if not os.path.exists("/proc/cpuinfo"):
        return None
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        flags = set()
        brand = "unknown"
        for line in text.splitlines():
            if line.startswith("flags") or line.startswith("Features"):
                _, _, rest = line.partition(":")
                flags.update(rest.strip().split())
            if line.startswith("model name"):
                _, _, rest = line.partition(":")
                brand = rest.strip()
        flags = {f.lower() for f in flags}
        return CpuFeatures(
            brand=brand, arch=platform.machine(),
            avx2="avx2" in flags,
            avx512=any(f.startswith("avx512") for f in flags),
            aes_ni="aes" in flags,
            sha_ni="sha_ni" in flags,
            detection_method="proc",
        )
    except Exception:
        return None


# ============================================================================
# Backend 3 — WMI on Windows (via PowerShell)
# ============================================================================

def _try_wmic_windows() -> CpuFeatures | None:
    if sys.platform != "win32":
        return None
    try:
        # PowerShell Get-WmiObject — works on every modern Windows; faster
        # than the deprecated `wmic`
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-WmiObject Win32_Processor | "
             "Select-Object -Property Name, Caption, Architecture | "
             "Format-List"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return None
        name_match = re.search(r"Name\s*:\s*(.+)", out.stdout)
        brand = name_match.group(1).strip() if name_match else "unknown"
        feats = _features_from_cpu_brand(brand)
        return CpuFeatures(
            brand=brand, arch=platform.machine(),
            avx2=feats["avx2"],
            avx512=feats["avx512"],
            aes_ni=feats["aes_ni"],
            sha_ni=feats["sha_ni"],
            detection_method="wmic",
        )
    except Exception:
        return None


# ============================================================================
# Backend 4 — CPU-family heuristic from `platform.processor()`
# ============================================================================

def _features_from_cpu_brand(brand: str) -> dict[str, bool]:
    """
    Best-effort CPU-family inference. Conservative: returns False on unknown CPUs.
    Sources:
      - Intel: AVX2 = Haswell (Q2 2013); AVX-512 = Skylake-X (2017), most
                Ice Lake (2019), Tiger Lake (2020); SHA-NI = Goldmont (2016)
                or higher Atom + Ice Lake (2019) Core.
      - AMD:   AVX2 = Excavator (2015) + all Zen (Ryzen 2017+); AVX-512 = Zen 4
                (Ryzen 7000/2022); SHA-NI = Zen (Ryzen 1000/2017+).
    """
    b = brand.lower()
    feats = {"avx2": False, "avx512": False, "aes_ni": False, "sha_ni": False}

    # Intel families with AVX2 (Haswell or later)
    intel_avx2 = [
        "i3-4", "i5-4", "i7-4",      # Haswell '13
        "i3-5", "i5-5", "i7-5",      # Broadwell '14
        "i3-6", "i5-6", "i7-6",      # Skylake '15
        "i3-7", "i5-7", "i7-7",      # Kaby Lake '16
        "i3-8", "i5-8", "i7-8", "i9-8",        # Coffee Lake '17
        "i3-9", "i5-9", "i7-9", "i9-9",
        "i3-10", "i5-10", "i7-10", "i9-10",     # Comet/Ice Lake '20
        "i3-11", "i5-11", "i7-11", "i9-11",
        "i5-12", "i7-12", "i9-12",              # Alder Lake '21
        "i5-13", "i7-13", "i9-13",              # Raptor Lake '22
        "i5-13700hx",                            # explicit thesis hw
        "i5-14", "i7-14", "i9-14",              # Raptor Lake Refresh '23
        "core ultra",                            # Meteor / Lunar / Arrow Lake
        "xeon e5", "xeon e7", "xeon w", "xeon scalable", "xeon platinum",
        "xeon gold", "xeon silver", "xeon bronze",
    ]
    if "intel" in b or "core" in b or "xeon" in b or "pentium" in b:
        if any(token in b for token in intel_avx2):
            feats["avx2"] = True
            feats["aes_ni"] = True
        # AVX-512 only on Skylake-X / Ice Lake server / Tiger Lake mobile /
        # Sapphire Rapids / Alder Lake-W. NOT on Alder Lake desktop (disabled).
        if any(token in b for token in [
            "xeon scalable", "xeon platinum", "xeon gold", "xeon silver",
            "i7-1185g", "i7-1165g", "i9-119", "i7-119", "i5-119",
        ]):
            feats["avx512"] = True
        # SHA-NI is on Goldmont+ atom and Ice Lake+ core
        if any(token in b for token in [
            "i3-1", "i5-1", "i7-1", "i9-1", "core ultra", "i5-13", "i7-13", "i9-13",
        ]):
            feats["sha_ni"] = True

    # AMD families with AVX2 (Excavator '15, Zen '17 onwards)
    if "amd" in b or "ryzen" in b or "epyc" in b or "threadripper" in b:
        if any(token in b for token in [
            "ryzen", "epyc", "threadripper", "excavator",
        ]):
            feats["avx2"] = True
            feats["aes_ni"] = True
            feats["sha_ni"] = True   # all Zen have SHA-NI
        if any(token in b for token in [
            "ryzen 7 7", "ryzen 9 7", "ryzen 5 7", "epyc 9", "epyc 8",
            "ryzen ai 3", "ryzen 9 9", "ryzen 7 9", "ryzen 5 9",
        ]):
            feats["avx512"] = True   # Zen 4 / Zen 5

    # Apple Silicon / ARM — no AVX, has crypto extensions on M-series
    if "apple m" in b or "apple silicon" in b:
        feats["aes_ni"] = True       # ARMv8 crypto ext
        feats["sha_ni"] = True

    return feats


def _try_heuristic() -> CpuFeatures:
    brand = platform.processor() or platform.machine() or "unknown"
    feats = _features_from_cpu_brand(brand)
    return CpuFeatures(
        brand=brand, arch=platform.machine(),
        avx2=feats["avx2"], avx512=feats["avx512"],
        aes_ni=feats["aes_ni"], sha_ni=feats["sha_ni"],
        detection_method="heuristic",
    )


# ============================================================================
# Public API
# ============================================================================

_cached: CpuFeatures | None = None


def detect_cpu_features(use_cache: bool = True) -> CpuFeatures:
    """
    Return the host CPU's cryptography-relevant feature flags.
    Result is cached for the lifetime of the process.
    """
    global _cached
    if use_cache and _cached is not None:
        return _cached
    for backend in (_try_pycpuinfo, _try_proc_cpuinfo, _try_wmic_windows):
        result = backend()
        if result is not None:
            _cached = result
            return result
    _cached = _try_heuristic()
    return _cached


def reset_cache() -> None:
    """Force re-detection on next call (for testing)."""
    global _cached
    _cached = None


# ============================================================================
# AVX2-speedup model
# ============================================================================

# Empirically measured speedup of liboqs AVX2 implementations vs reference C,
# averaged over Skylake-X / Cascade Lake / Ice Lake / Zen-3 platforms per the
# OQS public benchmarks (https://openquantumsafe.org/benchmarking/). When the
# host CPU has AVX2 enabled, ML-KEM and ML-DSA achieve roughly 1.8-2.3× over
# the portable C path. SPHINCS+ is hash-dominated so AVX2 helps less for
# SHA2-based variants and more for SHAKE-based variants via Keccak vector.
AVX2_SPEEDUP: dict[str, float] = {
    # ML-KEM / Kyber — NTT vectorised; gains uniform across L1/L3/L5
    "Kyber512":   2.10, "ML-KEM-512":  2.10,
    "Kyber768":   2.05, "ML-KEM-768":  2.05,
    "Kyber1024":  2.00, "ML-KEM-1024": 2.00,
    # ML-DSA / Dilithium — rejection-sampling loop has less SIMD parallelism
    "Dilithium2": 1.85, "ML-DSA-44": 1.85,
    "Dilithium3": 1.80, "ML-DSA-65": 1.80,
    "Dilithium5": 1.75, "ML-DSA-87": 1.75,
    # FN-DSA / Falcon — double-precision FFT; modest gains
    "Falcon-512":  1.40,
    "Falcon-1024": 1.40,
    # SLH-DSA / SPHINCS+ — hash-bound, SHA-2 variants get less benefit
    "SPHINCS+-SHA2-128f-simple":  1.30,
    "SPHINCS+-SHA2-128s-simple":  1.30,
    "SPHINCS+-SHA2-192f-simple":  1.30,
    "SPHINCS+-SHA2-256f-simple":  1.30,
    "SPHINCS+-SHAKE-128f-simple": 1.60,
    # Classic McEliece — matrix ops; gains depend on Avx2 batched syndrome decoder
    "Classic-McEliece-348864":  1.50,
    "Classic-McEliece-460896":  1.50,
    "Classic-McEliece-6688128": 1.50,
    # HQC — code-based with vectorisable arithmetic
    "HQC-128": 1.70, "HQC-192": 1.70, "HQC-256": 1.70,
}


def avx2_speedup_for(algorithm: str) -> float:
    """Return the empirical liboqs-AVX2-over-reference speedup for `algorithm`.
    Defaults to 1.0 (no speedup) for unrecognised entries."""
    return AVX2_SPEEDUP.get(algorithm, 1.0)

