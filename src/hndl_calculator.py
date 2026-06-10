"""
HNDL — "Harvest Now, Decrypt Later" residual-risk calculator.

The central thesis argument for early PQC migration: an attacker who captures
RSA/ECDH-protected ciphertext today can decrypt it once a CRQC (cryptographically
relevant quantum computer) arrives, retroactively breaking confidentiality of
data whose protection lifetime extends past the CRQC arrival year.

This module turns that argument into a tool. Inputs:

  - capture_year       — the year the ciphertext is intercepted (now/past)
  - confidentiality_yr — how many years the data must remain secret
  - data_class         — preset confidentiality lifetime by data type
                          (medical 30y, intel 50y, financial 7y, …)
  - crqc_arrival       — point estimate or distribution for CRQC arrival year
  - crypto_in_use      — RSA-2048 / ECC-P256 / hybrid / PQC

Output:

  - residual_risk      — probability that an attacker recovers the plaintext
                          before its confidentiality lifetime expires
  - safe_until_year    — the latest capture year at which residual risk < 5%

The risk model uses a log-normal CRQC arrival distribution (default:
NIST IR 8547 medians 2030/2035/2045 for early/median/late) per Mosca's heuristic
"y + z > x" rule (z = data lifetime, y = migration time, x = CRQC arrival).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats


# ============================================================================
# Data class presets (years of confidentiality required)
# ============================================================================

DATA_CLASS_LIFETIME = {
    "social-media-posts":     1,
    "session-token":          0,
    "ephemeral-chat":         1,
    "credit-card-pan":        7,
    "tax-records":            10,
    "trade-secret":           15,
    "medical-records":        30,
    "national-id":            30,
    "diplomatic-cable":       50,
    "intelligence-archive":   75,
    "state-secret":          100,
    "human-genome":          100,
}


# ============================================================================
# Crypto vulnerability classes
# ============================================================================

# probability that *given a CRQC has arrived*, the cryptosystem is broken
P_BROKEN_GIVEN_CRQC = {
    # Classical asymmetric — fully broken by Shor
    "RSA-2048":     1.00,
    "RSA-3072":     1.00,
    "RSA-4096":     1.00,
    "ECC-P256":     1.00,
    "ECC-P384":     1.00,
    "DH-2048":      1.00,
    # Symmetric — Grover halves the effective key length
    "AES-128":      0.50,    # Grover brings it to ~2^64; weakened, not broken
    "AES-192":      0.05,    # 2^96 still infeasible
    "AES-256":      0.001,   # 2^128 effective — safe
    "SHA-256":      0.05,
    "SHA3-256":     0.05,
    # PQC standardised algorithms — assumed safe vs CRQC
    "Kyber768":     0.01,
    "Kyber1024":    0.001,
    "Dilithium3":   0.01,
    "Falcon-512":   0.01,
    "SPHINCS+":     0.001,
    # Hybrid combinations — broken iff PQC component is broken (defense-in-depth)
    "X25519+Kyber768":   0.01,
    "RSA-2048+Kyber768": 0.01,
    "ECDSA+Dilithium3":  0.01,
}


# ============================================================================
# CRQC arrival distribution
# ============================================================================

@dataclass(frozen=True)
class CRQCDistribution:
    """
    Log-normal distribution over CRQC arrival year. Defaults match the NIST IR
    8547 (Aug 2024) consensus survey: median ~2034, P95 = 2050.
    """
    median_year: int = 2034
    p95_year: int = 2050
    epoch: int = 2024     # year the distribution was calibrated

    def cdf(self, year: int) -> float:
        """P(CRQC arrives by `year`)."""
        if year <= self.epoch:
            return 0.0
        # Fit log-normal so cdf(median)=0.5 and cdf(p95)=0.95
        years_to_median = self.median_year - self.epoch
        years_to_p95    = self.p95_year - self.epoch
        if years_to_median <= 0:
            return 1.0
        # mu = ln(median); sigma from p95 quantile of standard normal
        mu = math.log(years_to_median)
        z95 = stats.norm.ppf(0.95)
        sigma = (math.log(years_to_p95) - mu) / z95
        z = (math.log(max(year - self.epoch, 1e-9)) - mu) / max(sigma, 1e-9)
        return float(stats.norm.cdf(z))


# ============================================================================
# Computation
# ============================================================================

@dataclass
class HNDLInputs:
    capture_year:           int
    confidentiality_years:  int
    crypto:                 str
    data_class:             str | None = None  # if set, overrides confidentiality_years
    crqc_dist:              CRQCDistribution = field(default_factory=CRQCDistribution)


@dataclass
class HNDLResult:
    inputs:               HNDLInputs
    p_crqc_in_lifetime:   float
    p_broken_if_crqc:     float
    residual_risk:        float
    expiry_year:          int
    safe_until_year:      int          # latest capture year with residual < 5%
    interpretation:       str

    def to_dict(self) -> dict:
        return {
            "capture_year":         self.inputs.capture_year,
            "expiry_year":          self.expiry_year,
            "crypto":               self.inputs.crypto,
            "data_class":           self.inputs.data_class or "custom",
            "p_crqc_in_lifetime":   self.p_crqc_in_lifetime,
            "p_broken_if_crqc":     self.p_broken_if_crqc,
            "residual_risk":        self.residual_risk,
            "safe_until_year":      self.safe_until_year,
            "crqc_median":          self.inputs.crqc_dist.median_year,
            "crqc_p95":             self.inputs.crqc_dist.p95_year,
            "interpretation":       self.interpretation,
        }


def compute_hndl(inp: HNDLInputs) -> HNDLResult:
    if inp.data_class and inp.data_class in DATA_CLASS_LIFETIME:
        lifetime = DATA_CLASS_LIFETIME[inp.data_class]
    else:
        lifetime = inp.confidentiality_years
    expiry = inp.capture_year + lifetime

    p_crqc = inp.crqc_dist.cdf(expiry)
    p_break = P_BROKEN_GIVEN_CRQC.get(inp.crypto, 1.0)
    risk = p_crqc * p_break

    # Find safe_until_year = max capture_year with risk < 0.05
    safe_until = inp.capture_year
    for yr in range(inp.capture_year, inp.capture_year + 200):
        r = inp.crqc_dist.cdf(yr + lifetime) * p_break
        if r >= 0.05:
            break
        safe_until = yr

    if risk < 0.01:
        interp = "★★★★★ Safe — residual risk < 1%"
    elif risk < 0.05:
        interp = "★★★★ Acceptable — residual risk < 5%"
    elif risk < 0.20:
        interp = "★★★ Elevated — migrate within 5 years"
    elif risk < 0.50:
        interp = "★★ High — migrate now"
    else:
        interp = "★ Critical — assume already compromised under HNDL"

    return HNDLResult(
        inputs=inp,
        p_crqc_in_lifetime=p_crqc,
        p_broken_if_crqc=p_break,
        residual_risk=risk,
        expiry_year=expiry,
        safe_until_year=safe_until,
        interpretation=interp,
    )


# ============================================================================
# Convenience: bulk risk table for the thesis
# ============================================================================

def standard_thesis_table(capture_year: int = 2026) -> list[HNDLResult]:
    """
    Compute HNDL residual risk for the cross-product of (data_class, crypto)
    at capture_year. Goes into the thesis appendix as a one-page table.
    """
    cryptos = ["RSA-2048", "ECC-P256", "AES-128", "AES-256",
                "Kyber768", "X25519+Kyber768"]
    classes = ["credit-card-pan", "trade-secret", "medical-records",
                "national-id", "intelligence-archive", "state-secret"]
    out: list[HNDLResult] = []
    dist = CRQCDistribution()
    for dc in classes:
        for cr in cryptos:
            out.append(compute_hndl(HNDLInputs(
                capture_year=capture_year,
                confidentiality_years=DATA_CLASS_LIFETIME[dc],
                crypto=cr,
                data_class=dc,
                crqc_dist=dist,
            )))
    return out

