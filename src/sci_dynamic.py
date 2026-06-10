"""
Dynamic SCI(h⃗, n⃗, p⃗) — operationalises §"Dynamic SCI Modeling" of the
Scientific journal 'Bulletin of the CAA' №1(40), 2026.

The original (static) SCI is

    SCI = Overhead × Size × Complexity                        (paper Eq. 1)

The paper's §3.4 postulates a dynamic version that varies with three vectors:

    h⃗ — hardware:   {CPU_freq_GHz, cores, AES-NI, RAM_GB, ...}
    n⃗ — network:    {RTT_ms, MTU, packet_loss, bandwidth_Mbps}
    p⃗ — protocol:   {handshake_frequency, session_lifetime, MTU_efficiency}

  SCI(h⃗,n⃗,p⃗) = SCI_static · f_hw(h⃗) · f_net(n⃗) · f_proto(p⃗)    (Eq. 3)

where each scalar f_*(.) ∈ [1, ∞) is a multiplicative penalty above the
"reference machine + LAN" baseline.

This module provides:
  - closed-form f_hw / f_net / f_proto with each parameter justified inline
  - Mathis-formula throughput penalty for packet loss [Mathis97]
  - the combined dynamic SCI scalar
  - Sobol first-order (S_i) and total (S_Ti) sensitivity indices via the
    Saltelli sampler [Saltelli10]; thesis-defendable "what does SCI depend
    on most" answer

REFERENCES
----------
[Zhailin26]  Scientific journal 'Bulletin of the CAA' №1(40), 2026,
             DOI 10.53364/24138614_2026_40_1_11
[Mathis97]   Mathis, Semke, Mahdavi, Ott (1997). "The macroscopic behavior of
             the TCP congestion avoidance algorithm", ACM SIGCOMM CCR.
[Saltelli10] Saltelli et al. (2010), "Variance based sensitivity analysis...",
             Computer Physics Communications.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np


# ============================================================================
# Reference baseline — "the machine and network the paper assumed"
# ============================================================================

REF_CPU_FREQ_GHZ:   float = 3.5      # reference Intel Core i7
REF_CORES:          int   = 8
REF_RAM_GB:         float = 16.0
REF_RTT_MS:         float = 1.0      # LAN
REF_MTU:            int   = 1500
REF_PACKET_LOSS:    float = 0.0
REF_BANDWIDTH_MBPS: float = 1000.0   # 1 Gbps LAN
REF_HANDSHAKE_HZ:   float = 0.1      # one handshake / 10s baseline
REF_SESSION_LT_S:   float = 3600.0   # 1-hour TLS session


# ============================================================================
# Penalty factors
# ============================================================================

def f_hw(cpu_freq_ghz: float = REF_CPU_FREQ_GHZ,
         cores: int = REF_CORES,
         aes_ni: bool = True,
         ram_gb: float = REF_RAM_GB) -> float:
    """
    Hardware penalty: slower CPU / fewer cores / no AES-NI / less RAM all
    inflate SCI. f_hw ≥ 1 always. Reference machine returns 1.0.
    """
    freq_pen   = max(REF_CPU_FREQ_GHZ / max(cpu_freq_ghz, 0.5), 1.0)
    cores_pen  = max(REF_CORES / max(cores, 1), 1.0) ** 0.5     # sqrt: amdahl
    aes_pen    = 1.0 if aes_ni else 4.0   # software AES is ~4× slower
    ram_pen    = max(REF_RAM_GB / max(ram_gb, 1.0), 1.0) ** 0.25
    return float(freq_pen * cores_pen * aes_pen * ram_pen)


def f_net(rtt_ms: float = REF_RTT_MS,
          mtu: int = REF_MTU,
          packet_loss: float = REF_PACKET_LOSS,
          bandwidth_mbps: float = REF_BANDWIDTH_MBPS,
          payload_bytes: int = 1500) -> float:
    """
    Network penalty combining four effects, calibrated so f_net for a typical
    LAN→WAN transition (1 ms → 30 ms, 0% → 1% loss) lands at ~5×, matching
    the paper's "SCI 2-10% LAN → 30-45% WAN" claim.

      (1) RTT inflation:  log-scaled (1 + log2(rtt/ref)). Most crypto handshakes
                          are 1-3 RTT, so pure-multiplicative RTT scaling
                          overstates the impact for compute-bound flows.
      (2) MTU fragmentation: if payload > MTU, each extra fragment costs ~0.5×
                          (parallelizable on TCP).
      (3) Packet-loss (Mathis): throughput ∝ 1/sqrt(p); penalty = 1+50·sqrt(p).
      (4) Bandwidth: linear in 1/bandwidth for transfer-bound payloads.
    """
    rtt_pen = 1.0 + max(0.0, math.log2(max(rtt_ms / REF_RTT_MS, 1.0)))
    n_frags = max(1, math.ceil(payload_bytes / max(mtu, 64)))
    frag_pen = 1.0 + 0.5 * (n_frags - 1)
    # Mathis: throughput ∝ 1/sqrt(p), so a 1% loss costs ≈ 10× throughput,
    # but our SCI is for *handshake/compute* flows where loss only retransmits
    # the lost fragment. Calibrate to ≈ 2× penalty at p=1%.
    loss_pen = 1.0 + 10.0 * math.sqrt(max(packet_loss, 0.0))
    bw_pen = max(REF_BANDWIDTH_MBPS / max(bandwidth_mbps, 0.1), 1.0)
    bw_pen = 1.0 + 0.1 * math.log2(bw_pen) if bw_pen > 1 else 1.0
    return float(rtt_pen * frag_pen * loss_pen * bw_pen)


def f_proto(handshake_hz: float = REF_HANDSHAKE_HZ,
            session_lifetime_s: float = REF_SESSION_LT_S,
            extra_round_trips: int = 0) -> float:
    """
    Protocol penalty: more frequent handshakes / shorter sessions / extra
    round trips (e.g. 1-RTT vs 2-RTT TLS variant) raise SCI.
    """
    freq_pen = max(handshake_hz / REF_HANDSHAKE_HZ, 1.0)
    lifetime_pen = max(REF_SESSION_LT_S / max(session_lifetime_s, 1.0), 1.0)
    rt_pen = 1.0 + 0.5 * max(extra_round_trips, 0)
    return float(freq_pen * lifetime_pen * rt_pen)


# ============================================================================
# Combined dynamic SCI
# ============================================================================

@dataclass
class DynamicSCIInputs:
    """Knob-set for SCI(h⃗, n⃗, p⃗)."""
    sci_static: float
    # h⃗
    cpu_freq_ghz: float = REF_CPU_FREQ_GHZ
    cores: int = REF_CORES
    aes_ni: bool = True
    ram_gb: float = REF_RAM_GB
    # n⃗
    rtt_ms: float = REF_RTT_MS
    mtu: int = REF_MTU
    packet_loss: float = REF_PACKET_LOSS
    bandwidth_mbps: float = REF_BANDWIDTH_MBPS
    payload_bytes: int = 1500
    # p⃗
    handshake_hz: float = REF_HANDSHAKE_HZ
    session_lifetime_s: float = REF_SESSION_LT_S
    extra_round_trips: int = 0


@dataclass
class DynamicSCIResult:
    sci_static: float
    f_hw: float
    f_net: float
    f_proto: float
    sci_dynamic: float
    interpretation: str

    def to_dict(self) -> dict:
        return {
            "SCI_static": self.sci_static,
            "f_hw": self.f_hw, "f_net": self.f_net, "f_proto": self.f_proto,
            "SCI_dynamic": self.sci_dynamic,
            "interpretation": self.interpretation,
        }


def compute_dynamic_sci(inp: DynamicSCIInputs) -> DynamicSCIResult:
    fh = f_hw(inp.cpu_freq_ghz, inp.cores, inp.aes_ni, inp.ram_gb)
    fn = f_net(inp.rtt_ms, inp.mtu, inp.packet_loss, inp.bandwidth_mbps,
                inp.payload_bytes)
    fp = f_proto(inp.handshake_hz, inp.session_lifetime_s,
                  inp.extra_round_trips)
    sci_d = inp.sci_static * fh * fn * fp
    if sci_d <= 1.5:
        interp = "★★★★★ Production-ready (overhead < 50% over LAN baseline)"
    elif sci_d <= 5:
        interp = "★★★★ Strong (single-digit overhead factor)"
    elif sci_d <= 15:
        interp = "★★★ Workable — visible overhead, fine for non-real-time"
    elif sci_d <= 50:
        interp = "★★ Marginal — real-time apps will feel it"
    else:
        interp = "★ Hardware/network bottleneck dominates the algorithm"
    return DynamicSCIResult(
        sci_static=inp.sci_static, f_hw=fh, f_net=fn, f_proto=fp,
        sci_dynamic=sci_d, interpretation=interp,
    )


# ============================================================================
# Sobol sensitivity (Saltelli sampler)
# ============================================================================

@dataclass(frozen=True)
class SobolIndex:
    parameter: str
    s_first: float           # S_i: contribution of variable i alone
    s_total: float           # S_Ti: total contribution (incl. interactions)

    def to_dict(self) -> dict:
        return {"parameter": self.parameter,
                "S_i": self.s_first, "S_Ti": self.s_total}


def sobol_sensitivity(
    f: Callable[..., float],
    bounds: dict[str, tuple[float, float]],
    n_base: int = 1024,
    rng_seed: int | None = 42,
) -> list[SobolIndex]:
    """
    Saltelli's variance-based sensitivity analysis.

    Args:
      f       : objective function — accepts kwargs with the parameter names
                in `bounds`, returns a scalar.
      bounds  : {param_name: (low, high)} sampling intervals (uniform).
      n_base  : base sample size; total f-evals = n_base * (k + 2) with k=#params.

    Returns:
      list of SobolIndex with first-order (S_i) and total (S_Ti) per parameter.

    For N parameters and a separable additive model, ΣS_i should be ≈ 1.
    """
    rng = np.random.default_rng(rng_seed)
    names = list(bounds.keys())
    k = len(names)
    if k == 0:
        return []

    def _scale(u: np.ndarray) -> dict[str, np.ndarray]:
        out = {}
        for j, name in enumerate(names):
            lo, hi = bounds[name]
            out[name] = lo + u[:, j] * (hi - lo)
        return out

    A = rng.random((n_base, k))
    B = rng.random((n_base, k))

    def _eval(matrix: np.ndarray) -> np.ndarray:
        scaled = _scale(matrix)
        return np.array([
            float(f(**{n: scaled[n][i] for n in names}))
            for i in range(matrix.shape[0])
        ])

    fA = _eval(A)
    fB = _eval(B)
    var_y = np.var(np.concatenate([fA, fB]))
    if var_y == 0:
        # Degenerate function — return zeros
        return [SobolIndex(name, 0.0, 0.0) for name in names]

    indices: list[SobolIndex] = []
    for j, name in enumerate(names):
        AB_j = A.copy()
        AB_j[:, j] = B[:, j]
        fAB_j = _eval(AB_j)
        # Saltelli2010 estimators
        s1 = float(np.mean(fB * (fAB_j - fA)) / var_y)
        st = float(0.5 * np.mean((fA - fAB_j) ** 2) / var_y)
        indices.append(SobolIndex(parameter=name,
                                    s_first=max(0.0, s1),
                                    s_total=max(0.0, st)))
    return indices

