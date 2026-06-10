# ============================================================================
# QCCB v2.0 - Post-Quantum Cryptography Benchmark Module
# ML-KEM (Kyber), ML-DSA (Dilithium), SPHINCS+ Benchmarking
# ============================================================================
"""
Post-quantum cryptography benchmark module using liboqs-python.

Benchmarks NIST standardized post-quantum algorithms:
- ML-KEM (Kyber): FIPS 203 - Key Encapsulation Mechanism
- ML-DSA (Dilithium): FIPS 204 - Digital Signature Algorithm
- SLH-DSA (SPHINCS+): FIPS 205 - Hash-based Signatures

Statistical Protocol:
1. Warm-up: 10 iterations (excluded)
2. Measurement: 1000 iterations
3. Outlier removal: ±3σ rule
4. Report: mean, std, 95% CI

References:
    - NIST FIPS 203: Module-Lattice-Based Key-Encapsulation Mechanism Standard
    - NIST FIPS 204: Module-Lattice-Based Digital Signature Standard
    - NIST FIPS 205: Stateless Hash-Based Digital Signature Standard
    - liboqs: Open Quantum Safe library

Author: Amir
Date: 2026
Thesis: Data Protection in Quantum Computing Context: Cryptography Resilience Study
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import sys

import numpy as np

# Handle both relative and absolute imports
try:
    from .utils import (
        Timer,
        BenchmarkStatistics,
        calculate_statistics,
        benchmark_function,
        save_csv,
        save_json,
        create_progress_bar,
        format_statistics
    )
except ImportError:
    from utils import (
        Timer,
        BenchmarkStatistics,
        calculate_statistics,
        benchmark_function,
        save_csv,
        save_json,
        create_progress_bar,
        format_statistics
    )

logger = logging.getLogger('QCCB.pqc_benchmark')

# Try to import oqs, fall back to simulator if not available.
# Set QCCB_FORCE_SIMULATOR=1 to skip the liboqs probe entirely
# (recommended on Windows where oqs-python tries to git-clone liboqs at import).
import os as _os

if _os.environ.get("QCCB_FORCE_SIMULATOR", "").lower() in ("1", "true", "yes"):
    HAS_LIBOQS = False
    logger.info("QCCB_FORCE_SIMULATOR set — using PQC simulator")
else:
    try:
        import oqs
        try:
            _ = oqs.KeyEncapsulation('Kyber512')
            HAS_LIBOQS = True
            logger.debug("liboqs-python loaded successfully")
        except (RuntimeError, Exception) as e:
            logger.warning(f"liboqs-python found but not working: {e}")
            HAS_LIBOQS = False
    except (ImportError, RuntimeError, Exception) as e:
        HAS_LIBOQS = False
        logger.warning(f"liboqs-python not available: {e}")

if not HAS_LIBOQS:
    # Import simulator
    try:
        try:
            from .pqc_simulator import KEM, Signature
        except ImportError:
            from pqc_simulator import KEM, Signature
        logger.info("PQC Simulator loaded as fallback")
    except ImportError:
        logger.error("PQC Simulator not available either")


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class KEMBenchmarkResult:
    """
    Benchmark results for Key Encapsulation Mechanism algorithms.
    
    Attributes:
        algorithm: Algorithm name (e.g., "Kyber768").
        nist_level: NIST security level (1, 3, or 5).
        family: Algorithm family (e.g., "lattice").
        keygen_stats: Statistics for key generation.
        encaps_stats: Statistics for encapsulation.
        decaps_stats: Statistics for decapsulation.
        public_key_size: Size of public key in bytes.
        secret_key_size: Size of secret key in bytes.
        ciphertext_size: Size of ciphertext in bytes.
        shared_secret_size: Size of shared secret in bytes.
    """
    algorithm: str
    nist_level: int
    family: str
    keygen_stats: BenchmarkStatistics
    encaps_stats: BenchmarkStatistics
    decaps_stats: BenchmarkStatistics
    public_key_size: int
    secret_key_size: int
    ciphertext_size: int
    shared_secret_size: int
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'algorithm': self.algorithm,
            'nist_level': self.nist_level,
            'family': self.family,
            'keygen_ms_mean': self.keygen_stats.mean,
            'keygen_ms_std': self.keygen_stats.std,
            'keygen_ms_ci_lower': self.keygen_stats.ci_lower,
            'keygen_ms_ci_upper': self.keygen_stats.ci_upper,
            'encaps_ms_mean': self.encaps_stats.mean,
            'encaps_ms_std': self.encaps_stats.std,
            'encaps_ms_ci_lower': self.encaps_stats.ci_lower,
            'encaps_ms_ci_upper': self.encaps_stats.ci_upper,
            'decaps_ms_mean': self.decaps_stats.mean,
            'decaps_ms_std': self.decaps_stats.std,
            'decaps_ms_ci_lower': self.decaps_stats.ci_lower,
            'decaps_ms_ci_upper': self.decaps_stats.ci_upper,
            'public_key_bytes': self.public_key_size,
            'secret_key_bytes': self.secret_key_size,
            'ciphertext_bytes': self.ciphertext_size,
            'shared_secret_bytes': self.shared_secret_size,
            'n_samples': self.keygen_stats.n_samples,
            'n_outliers': self.keygen_stats.n_outliers
        }
    
    def to_csv_row(self) -> dict[str, Any]:
        """Convert to flat dictionary for CSV export."""
        return self.to_dict()


@dataclass
class SignatureBenchmarkResult:
    """
    Benchmark results for Digital Signature algorithms.
    
    Attributes:
        algorithm: Algorithm name (e.g., "Dilithium3").
        nist_level: NIST security level (2, 3, or 5).
        family: Algorithm family (e.g., "lattice", "hash-based").
        keygen_stats: Statistics for key generation.
        sign_stats: Statistics for signing.
        verify_stats: Statistics for verification.
        public_key_size: Size of public key in bytes.
        secret_key_size: Size of secret key in bytes.
        signature_size: Size of signature in bytes.
    """
    algorithm: str
    nist_level: int
    family: str
    keygen_stats: BenchmarkStatistics
    sign_stats: BenchmarkStatistics
    verify_stats: BenchmarkStatistics
    public_key_size: int
    secret_key_size: int
    signature_size: int
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'algorithm': self.algorithm,
            'nist_level': self.nist_level,
            'family': self.family,
            'keygen_ms_mean': self.keygen_stats.mean,
            'keygen_ms_std': self.keygen_stats.std,
            'keygen_ms_ci_lower': self.keygen_stats.ci_lower,
            'keygen_ms_ci_upper': self.keygen_stats.ci_upper,
            'sign_ms_mean': self.sign_stats.mean,
            'sign_ms_std': self.sign_stats.std,
            'sign_ms_ci_lower': self.sign_stats.ci_lower,
            'sign_ms_ci_upper': self.sign_stats.ci_upper,
            'verify_ms_mean': self.verify_stats.mean,
            'verify_ms_std': self.verify_stats.std,
            'verify_ms_ci_lower': self.verify_stats.ci_lower,
            'verify_ms_ci_upper': self.verify_stats.ci_upper,
            'public_key_bytes': self.public_key_size,
            'secret_key_bytes': self.secret_key_size,
            'signature_bytes': self.signature_size,
            'n_samples': self.keygen_stats.n_samples,
            'n_outliers': self.keygen_stats.n_outliers
        }
    
    def to_csv_row(self) -> dict[str, Any]:
        """Convert to flat dictionary for CSV export."""
        return self.to_dict()


# ============================================================================
# Algorithm Name Mapping
# ============================================================================

# Map config names to liboqs algorithm names
KEM_ALGORITHM_MAP = {
    # Lattice-based (NIST FIPS 203 — ML-KEM, formerly Kyber)
    'Kyber512': 'Kyber512',
    'Kyber768': 'Kyber768',
    'Kyber1024': 'Kyber1024',
    'ML-KEM-512': 'Kyber512',
    'ML-KEM-768': 'Kyber768',
    'ML-KEM-1024': 'Kyber1024',
    # Code-based (NIST round 4 KEM finalist — paper [Aragon23] cites SCI=3.2)
    'Classic-McEliece-348864':  'Classic-McEliece-348864',   # NIST L1
    'Classic-McEliece-460896':  'Classic-McEliece-460896',   # NIST L3
    'Classic-McEliece-6688128': 'Classic-McEliece-6688128',  # NIST L5
    # Code-based (NIST round 4 standard candidate)
    'HQC-128': 'HQC-128',
    'HQC-192': 'HQC-192',
    'HQC-256': 'HQC-256',
}

SIGNATURE_ALGORITHM_MAP = {
    # Lattice-based (NIST FIPS 204 — ML-DSA, formerly Dilithium)
    'Dilithium2': 'Dilithium2',
    'Dilithium3': 'Dilithium3',
    'Dilithium5': 'Dilithium5',
    'ML-DSA-44': 'Dilithium2',
    'ML-DSA-65': 'Dilithium3',
    'ML-DSA-87': 'Dilithium5',
    # Lattice-based / NTRU (NIST FIPS 206-draft — FN-DSA, formerly Falcon)
    'Falcon-512':  'Falcon-512',   # NIST L1, signature ≈ 666 B
    'Falcon-1024': 'Falcon-1024',  # NIST L5, signature ≈ 1280 B
    # Hash-based (NIST FIPS 205 — SLH-DSA, formerly SPHINCS+)
    'SPHINCS+-SHA2-128f-simple':  'SPHINCS+-SHA2-128f-simple',
    'SPHINCS+-SHA2-128s-simple':  'SPHINCS+-SHA2-128s-simple',
    'SPHINCS+-SHA2-192f-simple':  'SPHINCS+-SHA2-192f-simple',
    'SPHINCS+-SHA2-256f-simple':  'SPHINCS+-SHA2-256f-simple',
    'SPHINCS+-SHAKE-128f-simple': 'SPHINCS+-SHAKE-128f-simple',
}


# ============================================================================
# liboqs Wrapper Functions
# ============================================================================

def get_available_kems() -> list[str]:
    """
    Get list of available KEM algorithms from liboqs.
    
    Returns:
        List of algorithm names.
    """
    try:
        import oqs
        return oqs.get_enabled_kem_mechanisms()
    except ImportError:
        logger.warning("liboqs-python not installed")
        return []


def get_available_signatures() -> list[str]:
    """
    Get list of available signature algorithms from liboqs.
    
    Returns:
        List of algorithm names.
    """
    try:
        import oqs
        return oqs.get_enabled_sig_mechanisms()
    except ImportError:
        logger.warning("liboqs-python not installed")
        return []


def benchmark_kem(
    algorithm: str,
    nist_level: int,
    family: str,
    warmup: int,
    iterations: int,
    outlier_sigma: float,
    confidence: float
) -> Optional[KEMBenchmarkResult]:
    """
    Benchmark a KEM algorithm using liboqs or simulator.
    
    Args:
        algorithm: Algorithm name.
        nist_level: NIST security level.
        family: Algorithm family.
        warmup: Number of warm-up iterations.
        iterations: Number of measurement iterations.
        outlier_sigma: Sigma for outlier removal.
        confidence: Confidence level for CI.
        
    Returns:
        KEMBenchmarkResult or None if algorithm unavailable.
    """
    logger.info(f"  Benchmarking KEM: {algorithm} (NIST L{nist_level}, {family})")
    
    # Try real liboqs first
    if HAS_LIBOQS:
        try:
            import oqs
            
            # Map algorithm name
            oqs_name = KEM_ALGORITHM_MAP.get(algorithm, algorithm)
            
            # Check if algorithm is available
            available = oqs.get_enabled_kem_mechanisms()
            if oqs_name not in available:
                logger.warning(f"KEM algorithm {oqs_name} not available in liboqs, using simulator")
            else:
                # Real liboqs benchmark
                kem = oqs.KeyEncapsulation(oqs_name)
                public_key_size = kem.details['length_public_key']
                secret_key_size = kem.details['length_secret_key']
                ciphertext_size = kem.details['length_ciphertext']
                shared_secret_size = kem.details['length_shared_secret']
                
                # Benchmark key generation
                keygen_times = np.zeros(iterations + warmup, dtype=np.float64)
                timer = Timer()
                
                for i in range(warmup + iterations):
                    kem_instance = oqs.KeyEncapsulation(oqs_name)
                    timer.start()
                    public_key = kem_instance.generate_keypair()
                    keygen_times[i] = timer.stop()
                
                keygen_stats = calculate_statistics(
                    keygen_times[warmup:],
                    confidence=confidence,
                    outlier_sigma=outlier_sigma
                )
                
                # Generate a key pair for encapsulation/decapsulation benchmarks
                kem_instance = oqs.KeyEncapsulation(oqs_name)
                public_key = kem_instance.generate_keypair()
                
                # Benchmark encapsulation
                encaps_times = np.zeros(iterations + warmup, dtype=np.float64)
                
                for i in range(warmup + iterations):
                    timer.start()
                    ciphertext, shared_secret_enc = kem_instance.encap_secret(public_key)
                    encaps_times[i] = timer.stop()
                
                encaps_stats = calculate_statistics(
                    encaps_times[warmup:],
                    confidence=confidence,
                    outlier_sigma=outlier_sigma
                )
                
                # Benchmark decapsulation
                decaps_times = np.zeros(iterations + warmup, dtype=np.float64)
                ciphertext, _ = kem_instance.encap_secret(public_key)
                
                for i in range(warmup + iterations):
                    timer.start()
                    shared_secret_dec = kem_instance.decap_secret(ciphertext)
                    decaps_times[i] = timer.stop()
                
                decaps_stats = calculate_statistics(
                    decaps_times[warmup:],
                    confidence=confidence,
                    outlier_sigma=outlier_sigma
                )
                
                logger.info(f"    KeyGen (real): {format_statistics(keygen_stats)}")
                logger.info(f"    Encaps (real): {format_statistics(encaps_stats)}")
                logger.info(f"    Decaps (real): {format_statistics(decaps_stats)}")
                logger.info(f"    Sizes: PK={public_key_size}B, CT={ciphertext_size}B")
                
                return KEMBenchmarkResult(
                    algorithm=algorithm,
                    nist_level=nist_level,
                    family=family,
                    keygen_stats=keygen_stats,
                    encaps_stats=encaps_stats,
                    decaps_stats=decaps_stats,
                    public_key_size=public_key_size,
                    secret_key_size=secret_key_size,
                    ciphertext_size=ciphertext_size,
                    shared_secret_size=shared_secret_size
                )
        except Exception as e:
            logger.warning(f"liboqs benchmark failed: {e}, falling back to simulator")
    
    # Use simulator fallback
    try:
        try:
            from .pqc_simulator import KEM as KEM_SIM, KEM_PARAMS
        except ImportError:
            from pqc_simulator import KEM as KEM_SIM, KEM_PARAMS
        
        # Map to simulator names
        sim_names = {
            'Kyber512': 'ML-KEM-512',
            'Kyber768': 'ML-KEM-768',
            'Kyber1024': 'ML-KEM-1024',
            'ML-KEM-512': 'ML-KEM-512',
            'ML-KEM-768': 'ML-KEM-768',
            'ML-KEM-1024': 'ML-KEM-1024',
        }
        
        sim_name = sim_names.get(algorithm, algorithm)
        if sim_name not in KEM_PARAMS:
            logger.warning(f"Algorithm {algorithm} not in simulator either")
            return None
        
        kem_sim = KEM_SIM(sim_name)
        
        # Benchmark with simulator
        keygen_times = np.zeros(iterations + warmup, dtype=np.float64)
        timer = Timer()
        
        for i in range(warmup + iterations):
            timer.start()
            kem_sim.keygen()
            keygen_times[i] = timer.stop()
        
        keygen_stats = calculate_statistics(
            keygen_times[warmup:] * 1000,  # Convert to ms
            confidence=confidence,
            outlier_sigma=outlier_sigma
        )
        
        # Benchmark encapsulation
        pk, sk = kem_sim.keygen()
        encaps_times = np.zeros(iterations + warmup, dtype=np.float64)
        
        for i in range(warmup + iterations):
            timer.start()
            kem_sim.encaps(pk)
            encaps_times[i] = timer.stop()
        
        encaps_stats = calculate_statistics(
            encaps_times[warmup:] * 1000,  # Convert to ms
            confidence=confidence,
            outlier_sigma=outlier_sigma
        )
        
        # Benchmark decapsulation
        ct, ss = kem_sim.encaps(pk)
        decaps_times = np.zeros(iterations + warmup, dtype=np.float64)
        
        for i in range(warmup + iterations):
            timer.start()
            kem_sim.decaps(sk, ct)
            decaps_times[i] = timer.stop()
        
        decaps_stats = calculate_statistics(
            decaps_times[warmup:] * 1000,  # Convert to ms
            confidence=confidence,
            outlier_sigma=outlier_sigma
        )
        
        logger.info(f"    KeyGen (sim): {format_statistics(keygen_stats)}")
        logger.info(f"    Encaps (sim): {format_statistics(encaps_stats)}")
        logger.info(f"    Decaps (sim): {format_statistics(decaps_stats)}")
        logger.info(f"    Sizes: PK={kem_sim.alg.public_key_size}B, CT={kem_sim.alg.ciphertext_size}B")
        
        return KEMBenchmarkResult(
            algorithm=algorithm,
            nist_level=nist_level,
            family=family,
            keygen_stats=keygen_stats,
            encaps_stats=encaps_stats,
            decaps_stats=decaps_stats,
            public_key_size=kem_sim.alg.public_key_size,
            secret_key_size=kem_sim.alg.secret_key_size,
            ciphertext_size=kem_sim.alg.ciphertext_size,
            shared_secret_size=kem_sim.alg.shared_secret_size
        )
    except Exception as e:
        logger.error(f"Both liboqs and simulator failed for {algorithm}: {e}")
        return None


def benchmark_signature(
    algorithm: str,
    nist_level: int,
    family: str,
    warmup: int,
    iterations: int,
    outlier_sigma: float,
    confidence: float,
    message_size: int = 64
) -> Optional[SignatureBenchmarkResult]:
    """
    Benchmark a signature algorithm using liboqs or simulator.
    
    Args:
        algorithm: Algorithm name.
        nist_level: NIST security level.
        family: Algorithm family.
        warmup: Number of warm-up iterations.
        iterations: Number of measurement iterations.
        outlier_sigma: Sigma for outlier removal.
        confidence: Confidence level for CI.
        message_size: Size of message to sign in bytes.
        
    Returns:
        SignatureBenchmarkResult or None if algorithm unavailable.
    """
    logger.info(f"  Benchmarking Signature: {algorithm} (NIST L{nist_level}, {family})")
    
    # Try real liboqs first
    if HAS_LIBOQS:
        try:
            import oqs
            
            # Map algorithm name
            oqs_name = SIGNATURE_ALGORITHM_MAP.get(algorithm, algorithm)
            
            # Check if algorithm is available
            available = oqs.get_enabled_sig_mechanisms()
            if oqs_name not in available:
                logger.warning(f"Signature algorithm {oqs_name} not available in liboqs, using simulator")
            else:
                # Real liboqs benchmark
                sig = oqs.Signature(oqs_name)
                public_key_size = sig.details['length_public_key']
                secret_key_size = sig.details['length_secret_key']
                signature_size = sig.details['length_signature']
                message = bytes(range(message_size))
                
                # Benchmark key generation
                keygen_times = np.zeros(iterations + warmup, dtype=np.float64)
                timer = Timer()
                
                for i in range(warmup + iterations):
                    sig_instance = oqs.Signature(oqs_name)
                    timer.start()
                    public_key = sig_instance.generate_keypair()
                    keygen_times[i] = timer.stop()
                
                keygen_stats = calculate_statistics(
                    keygen_times[warmup:],
                    confidence=confidence,
                    outlier_sigma=outlier_sigma
                )
                
                # Generate a key pair for sign/verify benchmarks
                sig_instance = oqs.Signature(oqs_name)
                public_key = sig_instance.generate_keypair()
                
                # Benchmark signing
                sign_times = np.zeros(iterations + warmup, dtype=np.float64)
                
                for i in range(warmup + iterations):
                    timer.start()
                    signature = sig_instance.sign(message)
                    sign_times[i] = timer.stop()
                
                sign_stats = calculate_statistics(
                    sign_times[warmup:],
                    confidence=confidence,
                    outlier_sigma=outlier_sigma
                )
                
                # Generate signature for verification
                signature = sig_instance.sign(message)
                
                # Benchmark verification
                verify_times = np.zeros(iterations + warmup, dtype=np.float64)
                
                for i in range(warmup + iterations):
                    timer.start()
                    valid = sig_instance.verify(message, signature, public_key)
                    verify_times[i] = timer.stop()
                
                verify_stats = calculate_statistics(
                    verify_times[warmup:],
                    confidence=confidence,
                    outlier_sigma=outlier_sigma
                )
                
                logger.info(f"    KeyGen (real): {format_statistics(keygen_stats)}")
                logger.info(f"    Sign (real):   {format_statistics(sign_stats)}")
                logger.info(f"    Verify (real): {format_statistics(verify_stats)}")
                logger.info(f"    Sizes: PK={public_key_size}B, Sig={signature_size}B")
                
                return SignatureBenchmarkResult(
                    algorithm=algorithm,
                    nist_level=nist_level,
                    family=family,
                    keygen_stats=keygen_stats,
                    sign_stats=sign_stats,
                    verify_stats=verify_stats,
                    public_key_size=public_key_size,
                    secret_key_size=secret_key_size,
                    signature_size=signature_size
                )
        except Exception as e:
            logger.warning(f"liboqs benchmark failed: {e}, falling back to simulator")
    
    # Use simulator fallback
    try:
        try:
            from .pqc_simulator import Signature as SIG_SIM, SIG_PARAMS, SPHINCS_PARAMS
        except ImportError:
            from pqc_simulator import Signature as SIG_SIM, SIG_PARAMS, SPHINCS_PARAMS
        
        # Map to simulator names
        sim_names = {
            'Dilithium2': 'ML-DSA-44',
            'Dilithium3': 'ML-DSA-65',
            'Dilithium5': 'ML-DSA-87',
            'ML-DSA-44': 'ML-DSA-44',
            'ML-DSA-65': 'ML-DSA-65',
            'ML-DSA-87': 'ML-DSA-87',
            'SPHINCS+-SHA2-128s': 'SLH-DSA-SHA2-128s',
            'SPHINCS+-SHA2-128f': 'SLH-DSA-SHA2-128f',
            'SPHINCS+-SHA2-192s': 'SLH-DSA-SHA2-192s',
            'SPHINCS+-SHA2-192f': 'SLH-DSA-SHA2-192f',
            'SPHINCS+-SHA2-256s': 'SLH-DSA-SHA2-256s',
            'SPHINCS+-SHA2-256f': 'SLH-DSA-SHA2-256f',
        }
        
        sim_name = sim_names.get(algorithm, algorithm)
        
        # Check if in SIG_PARAMS or SPHINCS_PARAMS
        if sim_name in SIG_PARAMS:
            params = SIG_PARAMS[sim_name]
        elif sim_name in SPHINCS_PARAMS:
            params = SPHINCS_PARAMS[sim_name]
        else:
            logger.warning(f"Algorithm {algorithm} not in simulator either")
            return None
        
        sig_sim = SIG_SIM(sim_name)
        message = bytes(range(message_size))
        
        # Benchmark with simulator
        keygen_times = np.zeros(iterations + warmup, dtype=np.float64)
        timer = Timer()
        
        for i in range(warmup + iterations):
            timer.start()
            sig_sim.keygen()
            keygen_times[i] = timer.stop()
        
        keygen_stats = calculate_statistics(
            keygen_times[warmup:] * 1000,  # Convert to ms
            confidence=confidence,
            outlier_sigma=outlier_sigma
        )
        
        # Benchmark signing
        pk, sk = sig_sim.keygen()
        sign_times = np.zeros(iterations + warmup, dtype=np.float64)
        
        for i in range(warmup + iterations):
            timer.start()
            sig_sim.sign(sk, message)
            sign_times[i] = timer.stop()
        
        sign_stats = calculate_statistics(
            sign_times[warmup:] * 1000,  # Convert to ms
            confidence=confidence,
            outlier_sigma=outlier_sigma
        )
        
        # Benchmark verification
        signature = sig_sim.sign(sk, message)
        verify_times = np.zeros(iterations + warmup, dtype=np.float64)
        
        for i in range(warmup + iterations):
            timer.start()
            sig_sim.verify(pk, message, signature)
            verify_times[i] = timer.stop()
        
        verify_stats = calculate_statistics(
            verify_times[warmup:] * 1000,  # Convert to ms
            confidence=confidence,
            outlier_sigma=outlier_sigma
        )
        
        logger.info(f"    KeyGen (sim): {format_statistics(keygen_stats)}")
        logger.info(f"    Sign (sim):   {format_statistics(sign_stats)}")
        logger.info(f"    Verify (sim): {format_statistics(verify_stats)}")
        logger.info(f"    Sizes: PK={sig_sim.alg.public_key_size}B, Sig={sig_sim.alg.signature_size}B")
        
        return SignatureBenchmarkResult(
            algorithm=algorithm,
            nist_level=nist_level,
            family=family,
            keygen_stats=keygen_stats,
            sign_stats=sign_stats,
            verify_stats=verify_stats,
            public_key_size=sig_sim.alg.public_key_size,
            secret_key_size=sig_sim.alg.secret_key_size,
            signature_size=sig_sim.alg.signature_size
        )
    except Exception as e:
        logger.error(f"Both liboqs and simulator failed for {algorithm}: {e}")
        return None


# ============================================================================
# Main PQC Benchmark Class
# ============================================================================

class PQCBenchmarker:
    """
    Main class for post-quantum cryptography benchmarking.
    
    Benchmarks NIST standardized PQC algorithms using liboqs-python
    with statistical rigor.
    """
    
    def __init__(self, config: dict[str, Any]):
        """
        Initialize the PQC benchmarker.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.stat_config = config.get('statistics', {})
        self.pqc_config = config.get('pqc_algorithms', {})
        
        # Statistical parameters
        self.warmup = self.stat_config.get('warmup_iterations', 10)
        self.iterations = self.stat_config.get('benchmark_iterations', 1000)
        self.outlier_sigma = self.stat_config.get('outlier_sigma', 3.0)
        self.confidence = self.stat_config.get('confidence_level', 0.95)
        
        # Results
        self.kem_results: list[KEMBenchmarkResult] = []
        self.sig_results: list[SignatureBenchmarkResult] = []
    
    def check_liboqs(self) -> bool:
        """
        Check if liboqs-python is available or if simulator can be used.

        Returns:
            True if liboqs is available or simulator is available, False otherwise.
        """
        # Honour the QCCB_FORCE_SIMULATOR env-flag — if set we MUST NOT touch
        # `import oqs`, since the oqs-python package's __init__ tries to
        # git-clone/build liboqs at import time on Windows (currently broken
        # against upstream 0.15.0 because the pin is 0.14.1).
        if not HAS_LIBOQS:
            try:
                try:
                    from .pqc_simulator import KEM, Signature
                except ImportError:
                    from pqc_simulator import KEM, Signature
                logger.info("liboqs-python disabled (env or unavailable) — "
                             "using PQC Simulator")
                return True
            except ImportError:
                logger.error(
                    "PQC Simulator not available. Please install with: "
                    "pip install liboqs-python"
                )
                return False
        try:
            import oqs
            logger.info(f"liboqs version: {oqs.oqs_version()}")
            return True
        except ImportError:
            # Check if simulator is available
            try:
                try:
                    from .pqc_simulator import KEM, Signature
                except ImportError:
                    from pqc_simulator import KEM, Signature
                logger.info("liboqs-python not installed, using PQC Simulator")
                return True
            except ImportError:
                logger.error(
                    "liboqs-python not installed and PQC Simulator not available. "
                    "Please install with: pip install liboqs-python"
                )
                return False
    
    def benchmark_kem_algorithms(self) -> list[KEMBenchmarkResult]:
        """
        Benchmark all configured KEM algorithms.
        
        Returns:
            List of KEMBenchmarkResult objects.
        """
        logger.info("=" * 60)
        logger.info("PQC KEY ENCAPSULATION MECHANISM (KEM) BENCHMARKS")
        logger.info(f"Iterations: {self.iterations}, Warmup: {self.warmup}")
        logger.info("=" * 60)
        
        results = []
        kem_algos = self.pqc_config.get('kem', [])
        
        for algo_config in kem_algos:
            result = benchmark_kem(
                algorithm=algo_config['name'],
                nist_level=algo_config.get('nist_level', 3),
                family=algo_config.get('family', 'lattice'),
                warmup=self.warmup,
                iterations=self.iterations,
                outlier_sigma=self.outlier_sigma,
                confidence=self.confidence
            )
            
            if result:
                results.append(result)
        
        self.kem_results = results
        return results
    
    def benchmark_signature_algorithms(self) -> list[SignatureBenchmarkResult]:
        """
        Benchmark all configured signature algorithms.
        
        Returns:
            List of SignatureBenchmarkResult objects.
        """
        logger.info("\n" + "=" * 60)
        logger.info("PQC DIGITAL SIGNATURE (DSA) BENCHMARKS")
        logger.info(f"Iterations: {self.iterations}, Warmup: {self.warmup}")
        logger.info("=" * 60)
        
        results = []
        sig_algos = self.pqc_config.get('signature', [])
        
        for algo_config in sig_algos:
            result = benchmark_signature(
                algorithm=algo_config['name'],
                nist_level=algo_config.get('nist_level', 3),
                family=algo_config.get('family', 'lattice'),
                warmup=self.warmup,
                iterations=self.iterations,
                outlier_sigma=self.outlier_sigma,
                confidence=self.confidence
            )
            
            if result:
                results.append(result)
        
        self.sig_results = results
        return results
    
    def save_results(self, output_dir: Path) -> None:
        """
        Save benchmark results to files.
        
        Args:
            output_dir: Directory for output files.
        """
        logger.info("\nSaving PQC benchmark results...")
        
        # Combine all results for CSV
        all_results = []
        
        for r in self.kem_results:
            row = r.to_csv_row()
            row['type'] = 'KEM'
            row['operation1'] = 'encaps'
            row['operation2'] = 'decaps'
            row['op1_ms_mean'] = row['encaps_ms_mean']
            row['op2_ms_mean'] = row['decaps_ms_mean']
            all_results.append(row)
        
        for r in self.sig_results:
            row = r.to_csv_row()
            row['type'] = 'Signature'
            row['operation1'] = 'sign'
            row['operation2'] = 'verify'
            row['op1_ms_mean'] = row['sign_ms_mean']
            row['op2_ms_mean'] = row['verify_ms_mean']
            all_results.append(row)
        
        # Save CSV
        csv_path = output_dir / 'pqc_benchmarks.csv'
        save_csv(all_results, csv_path)
        logger.info(f"  Saved: {csv_path}")
        
        # Save detailed JSON
        json_data = {
            'metadata': {
                'warmup_iterations': self.warmup,
                'benchmark_iterations': self.iterations,
                'outlier_sigma': self.outlier_sigma,
                'confidence_level': self.confidence
            },
            'kem_algorithms': [r.to_dict() for r in self.kem_results],
            'signature_algorithms': [r.to_dict() for r in self.sig_results]
        }
        
        json_path = output_dir / 'pqc_benchmarks.json'
        save_json(json_data, json_path)
        logger.info(f"  Saved: {json_path}")
    
    def run(self, output_dir: Path) -> dict[str, Any]:
        """
        Run complete PQC benchmark suite.
        
        Args:
            output_dir: Directory for output files.
            
        Returns:
            Dictionary with all benchmark results.
        """
        if not self.check_liboqs():
            logger.warning("Skipping PQC benchmarks - liboqs not available")
            return {'kem_results': [], 'sig_results': []}
        
        # Run KEM benchmarks
        self.benchmark_kem_algorithms()
        
        # Run signature benchmarks
        self.benchmark_signature_algorithms()
        
        # Save results
        self.save_results(output_dir)
        
        return {
            'kem_results': self.kem_results,
            'sig_results': self.sig_results
        }
    
    def get_summary(self) -> str:
        """
        Get a text summary of benchmark results.
        
        Returns:
            Formatted summary string.
        """
        lines = [
            "\n" + "=" * 60,
            "PQC BENCHMARK SUMMARY",
            "=" * 60,
            "\nKEM Algorithms:"
        ]
        
        for r in self.kem_results:
            lines.append(
                f"  {r.algorithm} (L{r.nist_level}): "
                f"KeyGen={r.keygen_stats.mean:.3f}ms, "
                f"Encaps={r.encaps_stats.mean:.3f}ms, "
                f"Decaps={r.decaps_stats.mean:.3f}ms"
            )
        
        lines.append("\nSignature Algorithms:")
        
        for r in self.sig_results:
            lines.append(
                f"  {r.algorithm} (L{r.nist_level}): "
                f"KeyGen={r.keygen_stats.mean:.3f}ms, "
                f"Sign={r.sign_stats.mean:.3f}ms, "
                f"Verify={r.verify_stats.mean:.3f}ms"
            )
        
        lines.append("=" * 60)
        
        return "\n".join(lines)

