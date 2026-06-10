# ============================================================================
# QCCB v2.0 - Post-Quantum Cryptography Simulator
# Simulates liboqs-python for benchmarking PQC algorithms
# ============================================================================
"""
PQC algorithm simulator module.
Provides simulated benchmarks for NIST-standardized post-quantum algorithms
when liboqs-python is not available.

References:
    - NIST FIPS 203: Module-Lattice-Based KEM
    - NIST FIPS 204: Module-Lattice-Based DSA
    - NIST FIPS 205: Hash-Based DSA
    - OpenQuantumSafe liboqs documentation

Author: Amir
Date: 2026
"""

import hashlib
import time
from typing import Tuple
import numpy as np


class KEMParameter:
    """KEM algorithm parameters."""
    def __init__(self, name: str, public_key_size: int, secret_key_size: int,
                 ciphertext_size: int, shared_secret_size: int,
                 nist_level: int, classical_keygen_ms: float):
        self.name = name
        self.public_key_size = public_key_size
        self.secret_key_size = secret_key_size
        self.ciphertext_size = ciphertext_size
        self.shared_secret_size = shared_secret_size
        self.nist_level = nist_level
        self.classical_keygen_ms = classical_keygen_ms


class SignatureParameter:
    """Signature algorithm parameters."""
    def __init__(self, name: str, public_key_size: int, secret_key_size: int,
                 signature_size: int, nist_level: int, classical_sign_ms: float):
        self.name = name
        self.public_key_size = public_key_size
        self.secret_key_size = secret_key_size
        self.signature_size = signature_size
        self.nist_level = nist_level
        self.classical_sign_ms = classical_sign_ms


# NIST FIPS 203 - ML-KEM (Kyber) Parameters
# Public/secret key/ciphertext sizes are spec-exact; classical_keygen_ms
# values are reference timings on a modern x86 core (used for the simulator
# pathway when liboqs-python is unavailable).
KEM_PARAMS = {
    'ML-KEM-512':  KEMParameter('ML-KEM-512',  800,  1632, 768,  32, 1, 0.045),
    'ML-KEM-768':  KEMParameter('ML-KEM-768',  1184, 2400, 1088, 32, 3, 0.068),
    'ML-KEM-1024': KEMParameter('ML-KEM-1024', 1568, 3168, 1568, 32, 5, 0.095),
    # Legacy Kyber names (same parameters)
    'Kyber512':  KEMParameter('Kyber512',  800,  1632, 768,  32, 1, 0.045),
    'Kyber768':  KEMParameter('Kyber768',  1184, 2400, 1088, 32, 3, 0.068),
    'Kyber1024': KEMParameter('Kyber1024', 1568, 3168, 1568, 32, 5, 0.095),
    # Code-based KEM — Classic McEliece (NIST round-4 finalist).
    # Sizes per the Classic-McEliece-rd2 spec; key sizes are intentionally
    # very large (huge public key) but ciphertext is tiny.
    'Classic-McEliece-348864':  KEMParameter('Classic-McEliece-348864',
                                              261120,   6492,    96, 32, 1, 145.0),
    'Classic-McEliece-460896':  KEMParameter('Classic-McEliece-460896',
                                              524160,   13608,   156, 32, 3, 280.0),
    'Classic-McEliece-6688128': KEMParameter('Classic-McEliece-6688128',
                                              1044992,  13932,   208, 32, 5, 410.0),
    # Code-based KEM — HQC (NIST round-4 standard candidate).
    'HQC-128': KEMParameter('HQC-128', 2249,  2305,  4433,  64, 1, 0.95),
    'HQC-192': KEMParameter('HQC-192', 4522,  4586,  8978,  64, 3, 2.30),
    'HQC-256': KEMParameter('HQC-256', 7245,  7317,  14421, 64, 5, 4.10),
}

# NIST FIPS 204 - ML-DSA (Dilithium) Parameters
SIG_PARAMS = {
    'ML-DSA-44': SignatureParameter('ML-DSA-44', 1312, 2544, 2420, 2, 0.087),
    'ML-DSA-65': SignatureParameter('ML-DSA-65', 1952, 4000, 3309, 3, 0.195),
    'ML-DSA-87': SignatureParameter('ML-DSA-87', 2592, 5216, 4627, 5, 0.362),
    # Legacy Dilithium names
    'Dilithium2': SignatureParameter('Dilithium2', 1312, 2544, 2420, 2, 0.087),
    'Dilithium3': SignatureParameter('Dilithium3', 1952, 4000, 3309, 3, 0.195),
    'Dilithium5': SignatureParameter('Dilithium5', 2592, 5216, 4627, 5, 0.362),
    # NIST FIPS 206 (draft) - FN-DSA / Falcon - lattice/NTRU, compact signatures
    'Falcon-512':  SignatureParameter('Falcon-512',  897,  1281, 666,  1, 0.520),
    'Falcon-1024': SignatureParameter('Falcon-1024', 1793, 2305, 1280, 5, 1.040),
}

# NIST FIPS 205 - SLH-DSA (SPHINCS+) Parameters
# Both `SLH-DSA-*` (FIPS 205 official) and `SPHINCS+-*-simple` (liboqs naming)
# point to the same parameter sets — registered under both keys so the
# simulator works regardless of which convention the caller uses.
SPHINCS_PARAMS = {
    'SLH-DSA-SHA2-128s': SignatureParameter('SLH-DSA-SHA2-128s', 32, 64, 4179, 1, 0.5),
    'SLH-DSA-SHA2-128f': SignatureParameter('SLH-DSA-SHA2-128f', 32, 64, 17088, 1, 8.5),
    'SLH-DSA-SHA2-192s': SignatureParameter('SLH-DSA-SHA2-192s', 48, 96, 6144, 3, 1.2),
    'SLH-DSA-SHA2-192f': SignatureParameter('SLH-DSA-SHA2-192f', 48, 96, 35664, 3, 18.3),
    'SLH-DSA-SHA2-256s': SignatureParameter('SLH-DSA-SHA2-256s', 64, 128, 7856, 5, 1.8),
    'SLH-DSA-SHA2-256f': SignatureParameter('SLH-DSA-SHA2-256f', 64, 128, 49856, 5, 28.5),
    # liboqs naming aliases used by pqc_benchmark.SIGNATURE_ALGORITHM_MAP
    'SPHINCS+-SHA2-128s-simple': SignatureParameter('SPHINCS+-SHA2-128s-simple', 32, 64, 4179, 1, 0.5),
    'SPHINCS+-SHA2-128f-simple': SignatureParameter('SPHINCS+-SHA2-128f-simple', 32, 64, 17088, 1, 8.5),
    'SPHINCS+-SHA2-192f-simple': SignatureParameter('SPHINCS+-SHA2-192f-simple', 48, 96, 35664, 3, 18.3),
    'SPHINCS+-SHA2-256f-simple': SignatureParameter('SPHINCS+-SHA2-256f-simple', 64, 128, 49856, 5, 28.5),
    'SPHINCS+-SHAKE-128f-simple': SignatureParameter('SPHINCS+-SHAKE-128f-simple', 32, 64, 17088, 1, 7.8),
}


def _simulate_timing(nominal_ms: float, iterations: int = 1) -> Tuple[bytes, float]:
    """
    Simulate cryptographic operation timing with realistic variation.
    
    Simulates:
    - Base computational time
    - Random jitter (±15%)
    - Cache effects
    
    Args:
        nominal_ms: Nominal operation time in milliseconds
        iterations: Number of iterations
        
    Returns:
        Tuple of (result_bytes, elapsed_time_seconds)
    """
    # Add realistic timing variation
    # - Normal distribution with ±15% std dev
    # - Occasional cache misses
    jitter = np.random.normal(1.0, 0.15)
    
    # Occasional slower operations (cache miss simulation)
    if np.random.random() < 0.05:  # 5% cache miss probability
        jitter *= np.random.uniform(1.5, 2.5)
    
    actual_ms = nominal_ms * jitter * iterations
    actual_seconds = actual_ms / 1000.0
    
    # Simulate operation
    time.sleep(actual_seconds)
    
    # Generate dummy output
    result = hashlib.sha256(f"sim_{time.time()}".encode()).digest()
    
    return result, actual_seconds


# ============================================================================
# AVX2-vectorisation timing model
# ============================================================================
# Per dissertation §3.22: portable C reference implementations (pqcrypto)
# are ~2× slower than the liboqs AVX2 build for ML-KEM and ML-DSA. We
# implement an `avx2=True` flag on KEM/Signature that scales the simulated
# wall-clock by the empirical liboqs AVX2 speedup table.

try:
    from .cpu_features import avx2_speedup_for, detect_cpu_features
except ImportError:
    from cpu_features import avx2_speedup_for, detect_cpu_features


class KEM:
    """KEM (Key Encapsulation Mechanism) simulator.

    `avx2=True` activates the liboqs-style AVX2 vectorised timing path:
    nominal latencies are divided by the per-algorithm speedup factor from
    `cpu_features.AVX2_SPEEDUP` (Kyber ≈ 2.0×, Dilithium ≈ 1.8×, etc.).

    `avx2='auto'` enables AVX2 if and only if the host CPU exposes the AVX2
    flag. Default is `False` — explicit reference path so that side-by-side
    AVX2-vs-reference comparison tables are produced from two distinct runs.
    """

    def __init__(self, alg_name: str, avx2: bool | str = False):
        if alg_name not in KEM_PARAMS:
            raise ValueError(f"Unknown KEM algorithm: {alg_name}")
        self.alg = KEM_PARAMS[alg_name]
        self.alg_name = alg_name
        if avx2 == "auto":
            avx2 = detect_cpu_features().avx2
        self.avx2 = bool(avx2)
        self._speedup = avx2_speedup_for(alg_name) if self.avx2 else 1.0

    def _scaled(self, ms: float) -> float:
        return ms / self._speedup

    def keygen(self) -> Tuple[bytes, bytes]:
        _, _elapsed = _simulate_timing(self._scaled(self.alg.classical_keygen_ms))
        pk = b'PK_' + bytes(self.alg.public_key_size)
        sk = b'SK_' + bytes(self.alg.secret_key_size)
        return pk, sk

    def encaps(self, pk: bytes) -> Tuple[bytes, bytes]:
        _, _elapsed = _simulate_timing(self._scaled(self.alg.classical_keygen_ms * 0.8))
        ct = b'CT_' + bytes(self.alg.ciphertext_size)
        ss = b'SS_' + bytes(self.alg.shared_secret_size)
        return ct, ss

    def decaps(self, sk: bytes, ct: bytes) -> bytes:
        _, _elapsed = _simulate_timing(self._scaled(self.alg.classical_keygen_ms * 0.6))
        ss = b'SS_' + bytes(self.alg.shared_secret_size)
        return ss


class Signature:
    """Signature algorithm simulator with optional AVX2 vectorisation path
    (see `KEM.__init__` for `avx2` semantics)."""

    def __init__(self, alg_name: str, avx2: bool | str = False):
        if alg_name in SIG_PARAMS:
            self.alg = SIG_PARAMS[alg_name]
        elif alg_name in SPHINCS_PARAMS:
            self.alg = SPHINCS_PARAMS[alg_name]
        else:
            raise ValueError(f"Unknown Signature algorithm: {alg_name}")
        self.alg_name = alg_name
        if avx2 == "auto":
            avx2 = detect_cpu_features().avx2
        self.avx2 = bool(avx2)
        self._speedup = avx2_speedup_for(alg_name) if self.avx2 else 1.0

    def _scaled(self, ms: float) -> float:
        return ms / self._speedup

    def keygen(self) -> Tuple[bytes, bytes]:
        _, _elapsed = _simulate_timing(self._scaled(self.alg.classical_sign_ms * 2.0))
        pk = b'PK_' + bytes(self.alg.public_key_size)
        sk = b'SK_' + bytes(self.alg.secret_key_size)
        return pk, sk

    def sign(self, sk: bytes, message: bytes) -> bytes:
        _, _elapsed = _simulate_timing(self._scaled(self.alg.classical_sign_ms))
        sig = b'SIG_' + bytes(self.alg.signature_size)
        return sig

    def verify(self, pk: bytes, message: bytes, sig: bytes) -> bool:
        _, _elapsed = _simulate_timing(self._scaled(self.alg.classical_sign_ms * 0.5))
        return True

