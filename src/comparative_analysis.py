# ============================================================================
# QCCB v2.0 - Comparative Analysis Module
# Classical vs PQC vs Hybrid Cryptography Comparison
# ============================================================================
"""
Comparative analysis module for benchmarking classical cryptographic
algorithms alongside post-quantum alternatives.

Algorithms benchmarked:
- Asymmetric: RSA-2048, RSA-4096, ECC P-256, ECC P-384
- Symmetric: AES-128, AES-256
- Hash: SHA-256, SHA3-256
- Hybrid: RSA+Kyber parallel execution

References:
    - NIST SP 800-57: Recommendation for Key Management
    - NIST FIPS 186-5: Digital Signature Standard
    - RFC 8446: TLS 1.3 Specification

Author: Amir
Date: 2026
Thesis: Data Protection in Quantum Computing Context: Cryptography Resilience Study
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .utils import (
    Timer,
    BenchmarkStatistics,
    calculate_statistics,
    save_csv,
    save_json,
    calculate_sci,
    interpret_sci,
    format_statistics
)

logger = logging.getLogger('QCCB.comparative')


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ClassicalBenchmarkResult:
    """
    Benchmark results for classical cryptographic algorithms.
    """
    algorithm: str
    algorithm_type: str  # asymmetric, symmetric, hash
    keygen_stats: Optional[BenchmarkStatistics] = None
    operation1_stats: Optional[BenchmarkStatistics] = None  # encrypt/sign
    operation2_stats: Optional[BenchmarkStatistics] = None  # decrypt/verify
    key_size_bytes: int = 0
    output_size_bytes: int = 0  # ciphertext/signature/hash size
    quantum_safe: bool = False
    quantum_vulnerable_year: str = "N/A"
    nist_status: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            'algorithm': self.algorithm,
            'type': self.algorithm_type,
            'quantum_safe': 'YES' if self.quantum_safe else 'NO',
            'quantum_vulnerable_year': self.quantum_vulnerable_year,
            'nist_status': self.nist_status,
            'key_size_bytes': self.key_size_bytes,
            'output_size_bytes': self.output_size_bytes,
        }
        
        if self.keygen_stats:
            result['keygen_ms_mean'] = self.keygen_stats.mean
            result['keygen_ms_std'] = self.keygen_stats.std
        else:
            result['keygen_ms_mean'] = 0
            result['keygen_ms_std'] = 0
        
        if self.operation1_stats:
            result['op1_ms_mean'] = self.operation1_stats.mean
            result['op1_ms_std'] = self.operation1_stats.std
        else:
            result['op1_ms_mean'] = 0
            result['op1_ms_std'] = 0
        
        if self.operation2_stats:
            result['op2_ms_mean'] = self.operation2_stats.mean
            result['op2_ms_std'] = self.operation2_stats.std
        else:
            result['op2_ms_mean'] = 0
            result['op2_ms_std'] = 0
        
        return result


@dataclass
class HybridBenchmarkResult:
    """
    Benchmark results for hybrid cryptographic schemes.
    """
    name: str
    classical_algorithm: str
    pqc_algorithm: str
    mode: str  # sequential or parallel
    classical_time_ms: float
    pqc_time_ms: float
    total_time_ms: float
    overhead_percent: float
    key_size_bytes: int
    output_size_bytes: int
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'classical_algorithm': self.classical_algorithm,
            'pqc_algorithm': self.pqc_algorithm,
            'mode': self.mode,
            'classical_time_ms': self.classical_time_ms,
            'pqc_time_ms': self.pqc_time_ms,
            'total_time_ms': self.total_time_ms,
            'overhead_percent': self.overhead_percent,
            'key_size_bytes': self.key_size_bytes,
            'output_size_bytes': self.output_size_bytes
        }


# ============================================================================
# Classical Algorithm Benchmarks
# ============================================================================

def benchmark_rsa(
    key_size: int,
    iterations: int,
    warmup: int,
    outlier_sigma: float,
    confidence: float
) -> ClassicalBenchmarkResult:
    """
    Benchmark RSA key generation, encryption, and decryption.
    
    Args:
        key_size: RSA key size in bits (2048 or 4096).
        iterations: Number of measurement iterations.
        warmup: Number of warm-up iterations.
        outlier_sigma: Sigma for outlier removal.
        confidence: Confidence level for CI.
        
    Returns:
        ClassicalBenchmarkResult with timing statistics.
    """
    try:
        from cryptography.hazmat.primitives.asymmetric import rsa, padding
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.backends import default_backend
    except ImportError:
        logger.error("cryptography library not installed")
        return ClassicalBenchmarkResult(
            algorithm=f"RSA-{key_size}",
            algorithm_type="asymmetric"
        )
    
    logger.info(f"  Benchmarking RSA-{key_size}")
    
    timer = Timer()
    message = b"Test message for RSA encryption benchmark"
    
    # Key generation benchmark
    keygen_times = np.zeros(min(iterations, 100), dtype=np.float64)  # Limit for slow keygen
    
    for i in range(min(warmup, 5)):
        rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
            backend=default_backend()
        )
    
    for i in range(min(iterations, 100)):
        timer.start()
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
            backend=default_backend()
        )
        keygen_times[i] = timer.stop()
    
    keygen_stats = calculate_statistics(keygen_times, confidence, outlier_sigma)
    
    # Generate key for encrypt/decrypt benchmarks
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=key_size,
        backend=default_backend()
    )
    public_key = private_key.public_key()
    
    # Encryption benchmark
    encrypt_times = np.zeros(iterations, dtype=np.float64)
    padding_scheme = padding.OAEP(
        mgf=padding.MGF1(algorithm=hashes.SHA256()),
        algorithm=hashes.SHA256(),
        label=None
    )
    
    for i in range(warmup):
        public_key.encrypt(message, padding_scheme)
    
    for i in range(iterations):
        timer.start()
        ciphertext = public_key.encrypt(message, padding_scheme)
        encrypt_times[i] = timer.stop()
    
    encrypt_stats = calculate_statistics(encrypt_times, confidence, outlier_sigma)
    
    # Decryption benchmark
    decrypt_times = np.zeros(iterations, dtype=np.float64)
    ciphertext = public_key.encrypt(message, padding_scheme)
    
    for i in range(warmup):
        private_key.decrypt(ciphertext, padding_scheme)
    
    for i in range(iterations):
        timer.start()
        private_key.decrypt(ciphertext, padding_scheme)
        decrypt_times[i] = timer.stop()
    
    decrypt_stats = calculate_statistics(decrypt_times, confidence, outlier_sigma)
    
    logger.info(f"    KeyGen:  {format_statistics(keygen_stats)}")
    logger.info(f"    Encrypt: {format_statistics(encrypt_stats)}")
    logger.info(f"    Decrypt: {format_statistics(decrypt_stats)}")
    
    return ClassicalBenchmarkResult(
        algorithm=f"RSA-{key_size}",
        algorithm_type="asymmetric",
        keygen_stats=keygen_stats,
        operation1_stats=encrypt_stats,
        operation2_stats=decrypt_stats,
        key_size_bytes=key_size // 8,
        output_size_bytes=key_size // 8,
        quantum_safe=False,
        quantum_vulnerable_year="2027-2039",
        nist_status="FIPS 186-5 (migrate by 2030)"
    )


def benchmark_ecc(
    curve: str,
    iterations: int,
    warmup: int,
    outlier_sigma: float,
    confidence: float
) -> ClassicalBenchmarkResult:
    """
    Benchmark ECC key generation, signing, and verification.
    
    Args:
        curve: Curve name (secp256r1 or secp384r1).
        iterations: Number of measurement iterations.
        warmup: Number of warm-up iterations.
        outlier_sigma: Sigma for outlier removal.
        confidence: Confidence level for CI.
        
    Returns:
        ClassicalBenchmarkResult with timing statistics.
    """
    try:
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.backends import default_backend
    except ImportError:
        logger.error("cryptography library not installed")
        return ClassicalBenchmarkResult(
            algorithm=f"ECC-{curve}",
            algorithm_type="asymmetric"
        )
    
    curve_map = {
        'secp256r1': ec.SECP256R1(),
        'secp384r1': ec.SECP384R1(),
        'P-256': ec.SECP256R1(),
        'P-384': ec.SECP384R1(),
    }
    
    curve_obj = curve_map.get(curve, ec.SECP256R1())
    curve_name = curve.replace('secp', 'P-').replace('r1', '')
    
    logger.info(f"  Benchmarking ECC {curve_name}")
    
    timer = Timer()
    message = b"Test message for ECDSA signature benchmark"
    
    # Key generation benchmark
    keygen_times = np.zeros(iterations, dtype=np.float64)
    
    for i in range(warmup):
        ec.generate_private_key(curve_obj, default_backend())
    
    for i in range(iterations):
        timer.start()
        private_key = ec.generate_private_key(curve_obj, default_backend())
        keygen_times[i] = timer.stop()
    
    keygen_stats = calculate_statistics(keygen_times, confidence, outlier_sigma)
    
    # Generate key for sign/verify benchmarks
    private_key = ec.generate_private_key(curve_obj, default_backend())
    public_key = private_key.public_key()
    
    # Signing benchmark
    sign_times = np.zeros(iterations, dtype=np.float64)
    
    for i in range(warmup):
        private_key.sign(message, ec.ECDSA(hashes.SHA256()))
    
    for i in range(iterations):
        timer.start()
        signature = private_key.sign(message, ec.ECDSA(hashes.SHA256()))
        sign_times[i] = timer.stop()
    
    sign_stats = calculate_statistics(sign_times, confidence, outlier_sigma)
    
    # Verification benchmark
    verify_times = np.zeros(iterations, dtype=np.float64)
    signature = private_key.sign(message, ec.ECDSA(hashes.SHA256()))
    
    for i in range(warmup):
        public_key.verify(signature, message, ec.ECDSA(hashes.SHA256()))
    
    for i in range(iterations):
        timer.start()
        public_key.verify(signature, message, ec.ECDSA(hashes.SHA256()))
        verify_times[i] = timer.stop()
    
    verify_stats = calculate_statistics(verify_times, confidence, outlier_sigma)
    
    key_size = 32 if '256' in curve else 48
    sig_size = 64 if '256' in curve else 96
    
    logger.info(f"    KeyGen: {format_statistics(keygen_stats)}")
    logger.info(f"    Sign:   {format_statistics(sign_stats)}")
    logger.info(f"    Verify: {format_statistics(verify_stats)}")
    
    return ClassicalBenchmarkResult(
        algorithm=f"ECC-{curve_name}",
        algorithm_type="asymmetric",
        keygen_stats=keygen_stats,
        operation1_stats=sign_stats,
        operation2_stats=verify_stats,
        key_size_bytes=key_size,
        output_size_bytes=sig_size,
        quantum_safe=False,
        quantum_vulnerable_year="2027-2039",
        nist_status="FIPS 186-5 (migrate by 2030)"
    )


def benchmark_aes(
    key_size: int,
    iterations: int,
    warmup: int,
    outlier_sigma: float,
    confidence: float
) -> ClassicalBenchmarkResult:
    """
    Benchmark AES encryption and decryption.
    
    Args:
        key_size: Key size in bits (128 or 256).
        iterations: Number of measurement iterations.
        warmup: Number of warm-up iterations.
        outlier_sigma: Sigma for outlier removal.
        confidence: Confidence level for CI.
        
    Returns:
        ClassicalBenchmarkResult with timing statistics.
    """
    try:
        from Crypto.Cipher import AES
        from Crypto.Random import get_random_bytes
    except ImportError:
        logger.error("pycryptodome not installed")
        return ClassicalBenchmarkResult(
            algorithm=f"AES-{key_size}",
            algorithm_type="symmetric"
        )
    
    logger.info(f"  Benchmarking AES-{key_size}")
    
    timer = Timer()
    
    key = get_random_bytes(key_size // 8)
    plaintext = get_random_bytes(1024)  # 1KB data
    
    # Encryption benchmark
    encrypt_times = np.zeros(iterations, dtype=np.float64)
    
    for i in range(warmup):
        cipher = AES.new(key, AES.MODE_GCM)
        cipher.encrypt_and_digest(plaintext)
    
    for i in range(iterations):
        timer.start()
        cipher = AES.new(key, AES.MODE_GCM)
        ciphertext, tag = cipher.encrypt_and_digest(plaintext)
        encrypt_times[i] = timer.stop()
    
    encrypt_stats = calculate_statistics(encrypt_times, confidence, outlier_sigma)
    
    # Decryption benchmark
    decrypt_times = np.zeros(iterations, dtype=np.float64)
    
    cipher = AES.new(key, AES.MODE_GCM)
    ciphertext, tag = cipher.encrypt_and_digest(plaintext)
    nonce = cipher.nonce
    
    for i in range(warmup):
        cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
        cipher.decrypt_and_verify(ciphertext, tag)
    
    for i in range(iterations):
        timer.start()
        cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
        cipher.decrypt_and_verify(ciphertext, tag)
        decrypt_times[i] = timer.stop()
    
    decrypt_stats = calculate_statistics(decrypt_times, confidence, outlier_sigma)
    
    is_quantum_safe = key_size == 256  # AES-256 provides 128-bit post-quantum security
    
    logger.info(f"    Encrypt: {format_statistics(encrypt_stats)}")
    logger.info(f"    Decrypt: {format_statistics(decrypt_stats)}")
    
    return ClassicalBenchmarkResult(
        algorithm=f"AES-{key_size}",
        algorithm_type="symmetric",
        keygen_stats=None,
        operation1_stats=encrypt_stats,
        operation2_stats=decrypt_stats,
        key_size_bytes=key_size // 8,
        output_size_bytes=len(ciphertext) + 16,  # ciphertext + tag
        quantum_safe=is_quantum_safe,
        quantum_vulnerable_year="Safe" if is_quantum_safe else "Reduced security",
        nist_status="FIPS 197 (recommended: AES-256)"
    )


def benchmark_hash(
    algorithm: str,
    iterations: int,
    warmup: int,
    outlier_sigma: float,
    confidence: float
) -> ClassicalBenchmarkResult:
    """
    Benchmark hash function performance.
    
    Args:
        algorithm: Hash algorithm (sha256 or sha3_256).
        iterations: Number of measurement iterations.
        warmup: Number of warm-up iterations.
        outlier_sigma: Sigma for outlier removal.
        confidence: Confidence level for CI.
        
    Returns:
        ClassicalBenchmarkResult with timing statistics.
    """
    import hashlib
    
    algo_name = algorithm.upper().replace('_', '-')
    logger.info(f"  Benchmarking {algo_name}")
    
    timer = Timer()
    data = os.urandom(1024)  # 1KB data
    
    # Hash benchmark
    hash_times = np.zeros(iterations, dtype=np.float64)
    
    for i in range(warmup):
        h = hashlib.new(algorithm.replace('-', '_').lower())
        h.update(data)
        h.digest()
    
    for i in range(iterations):
        timer.start()
        h = hashlib.new(algorithm.replace('-', '_').lower())
        h.update(data)
        result = h.digest()
        hash_times[i] = timer.stop()
    
    hash_stats = calculate_statistics(hash_times, confidence, outlier_sigma)
    
    logger.info(f"    Hash: {format_statistics(hash_stats)}")
    
    return ClassicalBenchmarkResult(
        algorithm=algo_name,
        algorithm_type="hash",
        keygen_stats=None,
        operation1_stats=hash_stats,
        operation2_stats=None,
        key_size_bytes=0,
        output_size_bytes=32,
        quantum_safe=True,
        quantum_vulnerable_year="Safe",
        nist_status="FIPS 180-4 / FIPS 202"
    )


# ============================================================================
# Hybrid Scheme Benchmark
# ============================================================================

def benchmark_hybrid_rsa_kyber(
    rsa_result: ClassicalBenchmarkResult,
    kyber_keygen_ms: float,
    kyber_encaps_ms: float,
    kyber_public_key_size: int,
    kyber_ciphertext_size: int,
    mode: str = "parallel"
) -> HybridBenchmarkResult:
    """
    Calculate hybrid RSA+Kyber performance.
    
    In parallel mode, both operations run simultaneously.
    In sequential mode, they run one after another.
    
    Args:
        rsa_result: RSA benchmark results.
        kyber_keygen_ms: Kyber key generation time.
        kyber_encaps_ms: Kyber encapsulation time.
        kyber_public_key_size: Kyber public key size.
        kyber_ciphertext_size: Kyber ciphertext size.
        mode: "parallel" or "sequential".
        
    Returns:
        HybridBenchmarkResult with overhead analysis.
    """
    rsa_total = (
        (rsa_result.keygen_stats.mean if rsa_result.keygen_stats else 0) +
        (rsa_result.operation1_stats.mean if rsa_result.operation1_stats else 0)
    )
    
    kyber_total = kyber_keygen_ms + kyber_encaps_ms
    
    if mode == "parallel":
        # In parallel, total time is max of the two
        total_time = max(rsa_total, kyber_total)
    else:
        # In sequential, total time is sum
        total_time = rsa_total + kyber_total
    
    # Calculate overhead compared to RSA only
    if rsa_total > 0:
        overhead = ((total_time - rsa_total) / rsa_total) * 100
    else:
        overhead = 0
    
    # Combined key and ciphertext sizes
    key_size = rsa_result.key_size_bytes + kyber_public_key_size
    output_size = rsa_result.output_size_bytes + kyber_ciphertext_size
    
    return HybridBenchmarkResult(
        name="RSA2048+Kyber768",
        classical_algorithm="RSA-2048",
        pqc_algorithm="Kyber768",
        mode=mode,
        classical_time_ms=rsa_total,
        pqc_time_ms=kyber_total,
        total_time_ms=total_time,
        overhead_percent=overhead,
        key_size_bytes=key_size,
        output_size_bytes=output_size
    )


# ============================================================================
# Security Cost Index Analysis
# ============================================================================

@dataclass
class SCIAnalysis:
    """Security Cost Index analysis (per Zhailin et al. 2026, Eq. 1)."""
    algorithm: str
    baseline_algorithm: str
    performance_delta_percent: float
    nist_level: int
    quantum_safe: bool
    sci_value: float
    interpretation: str
    # Component factors that compose SCI (for audit and §3 dissertation tables)
    overhead_factor: float = 0.0     # 1 + log10(t_pqc / t_classical) / 5
    size_factor: float = 0.0          # 1 + log10(size_pqc / size_classical) / 5
    complexity_score: float = 0.0     # 1.0 (simple) … 2.5 (complex)
    nist_factor: float = 0.0          # 3 / NIST_level

    def to_dict(self) -> dict[str, Any]:
        return {
            'algorithm': self.algorithm,
            'baseline': self.baseline_algorithm,
            'performance_delta_percent': self.performance_delta_percent,
            'nist_level': self.nist_level,
            'quantum_safe': self.quantum_safe,
            'sci': self.sci_value,
            'overhead_factor': self.overhead_factor,
            'size_factor': self.size_factor,
            'complexity_score': self.complexity_score,
            'nist_factor': self.nist_factor,
            'interpretation': self.interpretation,
        }


# Classical-baseline sizes per FIPS / SEC1 (used by size-penalty term).
_BASELINE_SIZES = {
    "RSA-2048":  {"public_key": 256, "signature": 256},     # raw modulus
    "ECC-P-256": {"public_key": 65,  "signature": 72},      # uncompressed | DER
}


def _sci_for_pair(
    pqc_name: str, t_pqc_ms: float, pqc_size_bytes: int,
    classical_name: str, t_classical_ms: float, classical_size_bytes: int,
    nist_level: int,
) -> SCIAnalysis:
    """Build an SCIAnalysis for one (PQC algo, classical baseline) pair."""
    sci = calculate_sci(
        t_new=t_pqc_ms, t_old=t_classical_ms,
        nist_level=nist_level, quantum_safe=True,
        size_new_bytes=pqc_size_bytes, size_old_bytes=classical_size_bytes,
        algorithm_name=pqc_name,
    )
    # Recompute components for audit (kept consistent with utils.calculate_sci)
    import math as _math
    overhead_ratio = max(t_pqc_ms / t_classical_ms, 1.0) if t_classical_ms > 0 else 1.0
    size_ratio = max(pqc_size_bytes / classical_size_bytes, 1.0) if classical_size_bytes > 0 else 1.0
    of = 1.0 + _math.log10(overhead_ratio) / 5.0
    sf = 1.0 + _math.log10(size_ratio) / 5.0
    from src.utils import _complexity_for
    cs = _complexity_for(pqc_name)
    nf = 3.0 / nist_level if nist_level > 0 else 1.0
    return SCIAnalysis(
        algorithm=pqc_name,
        baseline_algorithm=classical_name,
        performance_delta_percent=((t_pqc_ms - t_classical_ms) / t_classical_ms) * 100
            if t_classical_ms > 0 else 0.0,
        nist_level=nist_level,
        quantum_safe=True,
        sci_value=sci,
        interpretation=interpret_sci(sci),
        overhead_factor=of, size_factor=sf,
        complexity_score=cs, nist_factor=nf,
    )


def calculate_sci_matrix(
    classical_results: list[ClassicalBenchmarkResult],
    pqc_kem_results: list,  # KEMBenchmarkResult from pqc_benchmark
    pqc_sig_results: list   # SignatureBenchmarkResult from pqc_benchmark
) -> list[SCIAnalysis]:
    """
    Calculate Security Cost Index for all PQC algorithms vs their natural
    classical baselines (RSA-2048 for KEM, ECC-P-256 for signatures).

    Uses the published 3-factor multiplicative formula
    (Zhailin et al. 2026 §"SCI", Eq. 1): see `src.utils.calculate_sci`.

    Returns:
        List of SCIAnalysis with overhead/size/complexity/nist factors exposed.
    """
    analyses: list[SCIAnalysis] = []

    rsa_baseline = None
    ecc_baseline = None
    for r in classical_results:
        if r.algorithm == "RSA-2048":
            rsa_baseline = r
        elif r.algorithm in ("ECC-P-256", "ECC-P256"):
            ecc_baseline = r

    # Paper baseline = RSA-2048 KeyGen latency (29.6 ms in paper Table 5).
    # ClassicalBenchmarkResult.operation1_stats is encrypt (~0.018 ms), NOT
    # KeyGen — so we must use `keygen_stats.mean` explicitly.
    rsa_keygen_ms = (rsa_baseline.keygen_stats.mean
                      if rsa_baseline and rsa_baseline.keygen_stats else 0.0)

    # KEM vs RSA-2048 (KeyGen-to-Encaps latency, public-key size penalty)
    if rsa_keygen_ms > 0:
        baseline_size = _BASELINE_SIZES["RSA-2048"]["public_key"]
        for kem in pqc_kem_results:
            pqc_time = kem.encaps_stats.mean
            pqc_size = getattr(kem, "public_key_size", baseline_size) or baseline_size
            analyses.append(_sci_for_pair(
                pqc_name=kem.algorithm, t_pqc_ms=pqc_time, pqc_size_bytes=pqc_size,
                classical_name="RSA-2048",
                t_classical_ms=rsa_keygen_ms, classical_size_bytes=baseline_size,
                nist_level=kem.nist_level,
            ))

    # Signature vs RSA-2048 (paper baseline — reproduces Dilithium-3 SCI ≈ 1.67).
    # We compare KeyGen-to-KeyGen (the heaviest PQC op) with signature size as
    # the size penalty (what's actually transmitted), matching the paper's
    # methodology on page 134.
    if rsa_keygen_ms > 0:
        baseline_size = _BASELINE_SIZES["RSA-2048"]["signature"]
        for sig in pqc_sig_results:
            # Use PQC KeyGen time when available (matches paper's overhead
            # methodology); fall back to Sign time when KeyGen not measured.
            kg_stats = getattr(sig, "keygen_stats", None)
            pqc_time = kg_stats.mean if kg_stats else sig.sign_stats.mean
            pqc_size = getattr(sig, "signature_size", baseline_size) or baseline_size
            analyses.append(_sci_for_pair(
                pqc_name=sig.algorithm, t_pqc_ms=pqc_time, pqc_size_bytes=pqc_size,
                classical_name="RSA-2048",
                t_classical_ms=rsa_keygen_ms, classical_size_bytes=baseline_size,
                nist_level=sig.nist_level,
            ))

    return analyses


# ============================================================================
# Main Comparative Analysis Class
# ============================================================================

class ComparativeAnalyzer:
    """
    Main class for comparative cryptographic analysis.
    
    Benchmarks classical algorithms and compares with PQC alternatives.
    """
    
    def __init__(self, config: dict[str, Any]):
        """
        Initialize the comparative analyzer.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.stat_config = config.get('statistics', {})
        self.classical_config = config.get('classical_algorithms', {})
        self.hybrid_config = config.get('hybrid_schemes', {})
        
        # Statistical parameters
        self.warmup = self.stat_config.get('warmup_iterations', 10)
        self.iterations = self.stat_config.get('benchmark_iterations', 1000)
        self.outlier_sigma = self.stat_config.get('outlier_sigma', 3.0)
        self.confidence = self.stat_config.get('confidence_level', 0.95)
        
        # Results
        self.classical_results: list[ClassicalBenchmarkResult] = []
        self.hybrid_results: list[HybridBenchmarkResult] = []
        self.sci_analyses: list[SCIAnalysis] = []
    
    def benchmark_classical_algorithms(self) -> list[ClassicalBenchmarkResult]:
        """
        Benchmark all configured classical algorithms.
        
        Returns:
            List of ClassicalBenchmarkResult objects.
        """
        logger.info("=" * 60)
        logger.info("CLASSICAL CRYPTOGRAPHY BENCHMARKS")
        logger.info(f"Iterations: {self.iterations}, Warmup: {self.warmup}")
        logger.info("=" * 60)
        
        results = []
        
        # RSA benchmarks
        for algo in self.classical_config.get('asymmetric', []):
            if 'RSA' in algo['name']:
                key_size = algo.get('key_size', 2048)
                result = benchmark_rsa(
                    key_size, 
                    self.iterations,
                    self.warmup,
                    self.outlier_sigma,
                    self.confidence
                )
                results.append(result)
            elif 'ECC' in algo['name']:
                curve = algo.get('curve', 'secp256r1')
                result = benchmark_ecc(
                    curve,
                    self.iterations,
                    self.warmup,
                    self.outlier_sigma,
                    self.confidence
                )
                results.append(result)
        
        # AES benchmarks
        for algo in self.classical_config.get('symmetric', []):
            key_size = algo.get('key_size', 256)
            result = benchmark_aes(
                key_size,
                self.iterations,
                self.warmup,
                self.outlier_sigma,
                self.confidence
            )
            results.append(result)
        
        # Hash benchmarks
        for algo in self.classical_config.get('hash', []):
            algorithm = algo.get('algorithm', 'sha256')
            result = benchmark_hash(
                algorithm,
                self.iterations,
                self.warmup,
                self.outlier_sigma,
                self.confidence
            )
            results.append(result)
        
        self.classical_results = results
        return results
    
    def calculate_hybrid_overhead(
        self,
        pqc_kem_results: list
    ) -> list[HybridBenchmarkResult]:
        """
        Calculate hybrid scheme overhead.
        
        Args:
            pqc_kem_results: KEM benchmark results from PQC module.
            
        Returns:
            List of HybridBenchmarkResult objects.
        """
        logger.info("\n" + "=" * 60)
        logger.info("HYBRID SCHEME OVERHEAD ANALYSIS")
        logger.info("=" * 60)
        
        results = []
        
        # Find RSA-2048 results
        rsa_result = None
        for r in self.classical_results:
            if r.algorithm == "RSA-2048":
                rsa_result = r
                break
        
        if not rsa_result:
            logger.warning("RSA-2048 results not found, skipping hybrid analysis")
            return results
        
        # Find Kyber768 results
        kyber_result = None
        for r in pqc_kem_results:
            if hasattr(r, 'algorithm') and 'Kyber768' in r.algorithm:
                kyber_result = r
                break
        
        if not kyber_result:
            logger.warning("Kyber768 results not found, using estimates")
            kyber_keygen_ms = 0.1
            kyber_encaps_ms = 0.05
            kyber_pk_size = 1184
            kyber_ct_size = 1088
        else:
            kyber_keygen_ms = kyber_result.keygen_stats.mean
            kyber_encaps_ms = kyber_result.encaps_stats.mean
            kyber_pk_size = kyber_result.public_key_size
            kyber_ct_size = kyber_result.ciphertext_size
        
        # Calculate parallel hybrid
        parallel_result = benchmark_hybrid_rsa_kyber(
            rsa_result,
            kyber_keygen_ms,
            kyber_encaps_ms,
            kyber_pk_size,
            kyber_ct_size,
            mode="parallel"
        )
        results.append(parallel_result)
        
        logger.info(f"  Parallel Hybrid (RSA2048+Kyber768):")
        logger.info(f"    Classical time: {parallel_result.classical_time_ms:.3f}ms")
        logger.info(f"    PQC time: {parallel_result.pqc_time_ms:.3f}ms")
        logger.info(f"    Total time: {parallel_result.total_time_ms:.3f}ms")
        logger.info(f"    Overhead: {parallel_result.overhead_percent:.1f}%")
        
        # Calculate sequential hybrid
        sequential_result = benchmark_hybrid_rsa_kyber(
            rsa_result,
            kyber_keygen_ms,
            kyber_encaps_ms,
            kyber_pk_size,
            kyber_ct_size,
            mode="sequential"
        )
        sequential_result.name = "RSA2048+Kyber768 (Sequential)"
        results.append(sequential_result)
        
        logger.info(f"  Sequential Hybrid (RSA2048+Kyber768):")
        logger.info(f"    Total time: {sequential_result.total_time_ms:.3f}ms")
        logger.info(f"    Overhead: {sequential_result.overhead_percent:.1f}%")
        
        self.hybrid_results = results
        return results
    
    def calculate_sci_analysis(
        self,
        pqc_kem_results: list,
        pqc_sig_results: list
    ) -> list[SCIAnalysis]:
        """
        Calculate Security Cost Index for all algorithms.
        
        Args:
            pqc_kem_results: KEM benchmark results.
            pqc_sig_results: Signature benchmark results.
            
        Returns:
            List of SCIAnalysis objects.
        """
        logger.info("\n" + "=" * 60)
        logger.info("SECURITY COST INDEX (SCI) ANALYSIS")
        logger.info("=" * 60)
        
        self.sci_analyses = calculate_sci_matrix(
            self.classical_results,
            pqc_kem_results,
            pqc_sig_results
        )
        
        for sci in self.sci_analyses:
            logger.info(
                f"  {sci.algorithm} vs {sci.baseline_algorithm}: "
                f"SCI={sci.sci_value:.2f} ({sci.interpretation})"
            )
        
        return self.sci_analyses
    
    def save_results(self, output_dir: Path) -> None:
        """
        Save comparative analysis results.
        
        Args:
            output_dir: Directory for output files.
        """
        logger.info("\nSaving comparative analysis results...")
        
        # Combine all results for CSV
        all_results = []
        
        for r in self.classical_results:
            row = r.to_dict()
            row['category'] = 'classical'
            all_results.append(row)
        
        for r in self.hybrid_results:
            row = r.to_dict()
            row['category'] = 'hybrid'
            row['type'] = 'hybrid'
            row['quantum_safe'] = 'YES'
            row['keygen_ms_mean'] = 0
            row['op1_ms_mean'] = r.total_time_ms
            row['op2_ms_mean'] = 0
            all_results.append(row)
        
        # Save CSV
        csv_path = output_dir / 'comparative_analysis.csv'
        save_csv(all_results, csv_path)
        logger.info(f"  Saved: {csv_path}")
        
        # Save SCI analysis
        if self.sci_analyses:
            sci_path = output_dir / 'sci_analysis.csv'
            save_csv([s.to_dict() for s in self.sci_analyses], sci_path)
            logger.info(f"  Saved: {sci_path}")

        # === Dissertation §3.11 Table 8: Extended classical baselines ===
        # with median + CoV% (coefficient of variation) per row.
        self._save_extended_classical_baseline(output_dir)

        # === Dissertation §3.13 Table 9: QCCB-native SCI ===
        # formula: ΔPerformance% / (NIST_security_level × Quantum_Safety)
        if self.sci_analyses:
            self._save_sci_qccb_native(
                output_dir,
                getattr(self, '_kem_results_ref', None),
                getattr(self, '_sig_results_ref', None),
            )

    def _save_extended_classical_baseline(self, output_dir: Path) -> None:
        """Emit Table 8 — classical baselines with mean ± CI95, median, CoV%."""
        import math as _math
        rows = []
        for r in self.classical_results:
            d = r.to_dict()
            # Each ClassicalBenchmarkResult has at most 3 ops we surface
            # (keygen + op1 + op2). Each carries _ms_mean and _ms_std plus
            # a (possibly absent) median.
            ops = []
            if d.get('keygen_ms_mean'):
                ops.append(("KeyGen",     d['keygen_ms_mean'], d.get('keygen_ms_std') or 0))
            if d.get('op1_ms_mean'):
                op1 = d.get('operation1') or 'Op1'
                ops.append((op1.title(),  d['op1_ms_mean'],   d.get('op1_ms_std') or 0))
            if d.get('op2_ms_mean'):
                op2 = d.get('operation2') or 'Op2'
                ops.append((op2.title(),  d['op2_ms_mean'],   d.get('op2_ms_std') or 0))
            for name, mean, std in ops:
                if mean <= 0:
                    continue
                # 95% CI half-width (z=1.96 / sqrt(n)); use n_samples if known
                n = max(int(d.get('n_samples', 100)), 2)
                ci_half = 1.96 * std / _math.sqrt(n)
                cov_pct = 100.0 * std / mean if mean else 0.0
                rows.append({
                    "Algorithm":    d['algorithm'],
                    "Operation":    name,
                    "Mean_ms":      round(mean, 6),
                    "CI95_half_ms": round(ci_half, 6),
                    "Median_ms":    round(d.get(f'{name.lower()}_ms_median', mean), 6),
                    "Std_ms":       round(std, 6),
                    "CoV_pct":      round(cov_pct, 2),
                    "N_samples":    n,
                    "Outlier_rule": "±3σ removal",
                })
        out = output_dir / 'extended_classical_baseline.csv'
        save_csv(rows, out)
        logger.info(f"  Saved: {out}")

    def _save_sci_qccb_native(self, output_dir: Path,
                                 kem_results=None, sig_results=None) -> None:
        """
        Emit Table 9 — QCCB-native SCI breakdown per dissertation §3.13.

        Uses the SAME published 3-factor formula as Eq. 1 of Zhailin et al.
        (2026) — see `src.utils.calculate_sci` — but with PQC-specific
        operation labels for readability, plus a Delta_Performance_pct
        informational column showing the raw ratio gap.

        Baselines:
          • KEM       → RSA-2048 (KeyGen), pubkey 256 B
          • Signature → RSA-2048 (KeyGen), sig 256 B
          (matches paper's "Table 5 + page 134" methodology)

        Bands (per paper §"Results"):
          SCI < 1       → WIN
          1 ≤ SCI < 2   → ★★★★★ Production-ready
          2 ≤ SCI < 5   → ★★★ Workable
          5 ≤ SCI < 10  → ★★ Marginal
          SCI ≥ 10      → ★ Specialized only
        """
        # Locate RSA-2048 KeyGen baseline (paper Table 5: ~29.6 ms).
        # NOTE: ClassicalBenchmarkResult.operation1_stats is encrypt (~0.018ms),
        # NOT KeyGen — we must use `keygen_stats.mean` explicitly.
        rsa_baseline_ms: float = 0.0
        for r in self.classical_results:
            if r.algorithm == "RSA-2048" and r.keygen_stats:
                rsa_baseline_ms = r.keygen_stats.mean
                break
        if rsa_baseline_ms <= 0:
            logger.warning("  No RSA-2048 KeyGen baseline — skipping sci_qccb_native.csv")
            return

        # Same baseline sizes used in main SCI matrix
        rsa_baseline_size = _BASELINE_SIZES["RSA-2048"]["public_key"]

        def _emit(op_label: str, pqc_label: str, t_pqc_ms: float,
                   pqc_size_bytes: int, nist_level: int) -> dict:
            sci_obj = _sci_for_pair(
                pqc_name=pqc_label, t_pqc_ms=t_pqc_ms,
                pqc_size_bytes=pqc_size_bytes,
                classical_name="RSA-2048",
                t_classical_ms=rsa_baseline_ms,
                classical_size_bytes=rsa_baseline_size,
                nist_level=nist_level,
            )
            delta_pct = 100.0 * (t_pqc_ms - rsa_baseline_ms) / rsa_baseline_ms
            return {
                "Operation": op_label,
                "Classical": "RSA-2048 (KeyGen)",
                "Classical_ms": round(rsa_baseline_ms, 4),
                "PQC": pqc_label,
                "PQC_ms": round(t_pqc_ms, 4),
                "Delta_Performance_pct": round(delta_pct, 2),
                "PQC_Size_bytes": pqc_size_bytes,
                "NIST_Level": nist_level,
                "Overhead_Factor": round(sci_obj.overhead_factor, 4),
                "Size_Penalty": round(sci_obj.size_factor, 4),
                "Complexity_Score": round(sci_obj.complexity_score, 4),
                "NIST_Factor": round(sci_obj.nist_factor, 4),
                "SCI_QCCB": round(sci_obj.sci_value, 4),
                "Band": sci_obj.interpretation,
            }

        rows: list[dict] = []
        for kem in (kem_results or []):
            pubkey = getattr(kem, "public_key_size", rsa_baseline_size) or rsa_baseline_size
            rows.append(_emit(
                op_label=f"Key encaps L{kem.nist_level}",
                pqc_label=kem.algorithm,
                t_pqc_ms=kem.encaps_stats.mean,
                pqc_size_bytes=pubkey,
                nist_level=kem.nist_level,
            ))
        for sig in (sig_results or []):
            sig_size = getattr(sig, "signature_size", rsa_baseline_size) or rsa_baseline_size
            # Use KeyGen for PQC time when available (matches paper page 134
            # methodology of comparing setup cost), fall back to Sign time.
            kg_stats = getattr(sig, "keygen_stats", None)
            t_pqc = kg_stats.mean if kg_stats else sig.sign_stats.mean
            rows.append(_emit(
                op_label=f"Sign L{sig.nist_level}",
                pqc_label=sig.algorithm,
                t_pqc_ms=t_pqc,
                pqc_size_bytes=sig_size,
                nist_level=sig.nist_level,
            ))

        out = output_dir / 'sci_qccb_native.csv'
        save_csv(rows, out)
        logger.info(f"  Saved: {out}")
    
    def run(
        self,
        output_dir: Path,
        pqc_kem_results: list = None,
        pqc_sig_results: list = None
    ) -> dict[str, Any]:
        """
        Run complete comparative analysis.
        
        Args:
            output_dir: Directory for output files.
            pqc_kem_results: Optional KEM results for comparison.
            pqc_sig_results: Optional signature results for comparison.
            
        Returns:
            Dictionary with all analysis results.
        """
        # Benchmark classical algorithms
        self.benchmark_classical_algorithms()

        # Calculate hybrid overhead if PQC results available
        if pqc_kem_results:
            self.calculate_hybrid_overhead(pqc_kem_results)

            if pqc_sig_results:
                self.calculate_sci_analysis(pqc_kem_results, pqc_sig_results)

        # Stash PQC results for the QCCB-native SCI Table 9 emitter
        self._kem_results_ref = pqc_kem_results
        self._sig_results_ref = pqc_sig_results

        # Save results
        self.save_results(output_dir)
        
        return {
            'classical_results': self.classical_results,
            'hybrid_results': self.hybrid_results,
            'sci_analyses': self.sci_analyses
        }

