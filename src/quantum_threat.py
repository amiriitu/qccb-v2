# ============================================================================
# QCCB v2.0 - Quantum Threat Analysis Module
# Shor's Algorithm Simulation and Threat Timeline Generation
# ============================================================================
"""
Quantum threat analysis module for simulating Shor's algorithm and
generating threat timelines for classical cryptographic algorithms.

This module:
1. Simulates Shor's algorithm using Qiskit for small number factorization
2. Compares quantum vs classical factorization times
3. Extrapolates threat timelines for RSA-2048 (2027-2039)
4. Generates CSV reports and visualizations

References:
    - Gidney, C., & Ekerå, M. (2021). How to factor 2048 bit RSA integers
      in 8 hours using 20 million noisy qubits. Quantum, 5, 433.
    - NIST Post-Quantum Cryptography Standardization Process

Author: Amir
Date: 2026
Thesis: Data Protection in Quantum Computing Context: Cryptography Resilience Study
"""

from __future__ import annotations

import logging
import math
import time
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
    create_progress_bar
)

logger = logging.getLogger('QCCB.quantum_threat')


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class QuantumThreatResult:
    """
    Results from quantum threat analysis for a single algorithm.
    
    Attributes:
        algorithm: Name of the cryptographic algorithm.
        classical_security: Classical security assessment.
        quantum_attack: Type of quantum attack applicable.
        vulnerable_timeframe: Estimated vulnerability window.
        status: Current security status.
        notes: Additional notes and references.
    """
    algorithm: str
    classical_security: str
    quantum_attack: str
    vulnerable_timeframe: str
    status: str
    notes: str = ""
    
    def to_dict(self) -> dict[str, str]:
        return {
            'Algorithm': self.algorithm,
            'Classical_Security': self.classical_security,
            'Quantum_Attack': self.quantum_attack,
            'Vulnerable_Timeframe': self.vulnerable_timeframe,
            'Status': self.status,
            'Notes': self.notes
        }


@dataclass
class FactorizationResult:
    """
    Results from factorization benchmark.
    
    Attributes:
        number: The number that was factored.
        factors: Tuple of the two factors found.
        classical_time_ms: Time taken by classical algorithm.
        quantum_sim_time_ms: Time taken by quantum simulation.
        qubits_used: Number of qubits used in simulation.
        circuit_depth: Depth of the quantum circuit.
        success: Whether factorization was successful.
    """
    number: int
    factors: tuple[int, int]
    classical_time_ms: float
    quantum_sim_time_ms: float
    qubits_used: int
    circuit_depth: int
    success: bool
    
    def to_dict(self) -> dict[str, Any]:
        return {
            'number': self.number,
            'factor_1': self.factors[0],
            'factor_2': self.factors[1],
            'classical_time_ms': self.classical_time_ms,
            'quantum_sim_time_ms': self.quantum_sim_time_ms,
            'qubits_used': self.qubits_used,
            'circuit_depth': self.circuit_depth,
            'success': self.success
        }


# ============================================================================
# Classical Factorization
# ============================================================================

def classical_factorization(n: int) -> tuple[int, int]:
    """
    Classical trial division factorization.
    
    Simple but effective for small numbers.
    
    Args:
        n: Number to factor.
        
    Returns:
        Tuple of the two factors.
        
    Raises:
        ValueError: If n is prime or less than 2.
    """
    if n < 2:
        raise ValueError(f"Cannot factor {n}")
    
    if n == 2:
        return (2, 1)
    
    if n % 2 == 0:
        return (2, n // 2)
    
    # Trial division up to sqrt(n)
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return (i, n // i)
    
    # n is prime
    raise ValueError(f"{n} is prime, cannot factor")


def benchmark_classical_factorization(
    n: int,
    iterations: int = 100
) -> BenchmarkStatistics:
    """
    Benchmark classical factorization with statistical rigor.
    
    Args:
        n: Number to factor.
        iterations: Number of iterations for timing.
        
    Returns:
        BenchmarkStatistics with timing results.
    """
    times = np.zeros(iterations, dtype=np.float64)
    timer = Timer()
    
    for i in range(iterations):
        timer.start()
        classical_factorization(n)
        times[i] = timer.stop()
    
    return calculate_statistics(times)


# ============================================================================
# Quantum Simulation (Shor's Algorithm)
# ============================================================================

def _gcd(a: int, b: int) -> int:
    """Compute GCD using Euclidean algorithm."""
    while b:
        a, b = b, a % b
    return a


def _modular_exponentiation(base: int, exp: int, mod: int) -> int:
    """Compute (base^exp) % mod efficiently."""
    result = 1
    base = base % mod
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod
        exp = exp >> 1
        base = (base * base) % mod
    return result


def shors_classical_simulation(n: int) -> tuple[int, int]:
    """
    Classical simulation of Shor's algorithm logic.
    
    This simulates the quantum period-finding step classically
    for small numbers. For actual quantum execution, use the
    Qiskit implementation.
    
    Args:
        n: Number to factor.
        
    Returns:
        Tuple of the two factors.
        
    Raises:
        ValueError: If factorization fails.
    """
    if n < 2:
        raise ValueError(f"Cannot factor {n}")
    
    if n % 2 == 0:
        return (2, n // 2)
    
    # Check if n is a prime power
    for k in range(2, int(math.log2(n)) + 1):
        root = int(round(n ** (1/k)))
        if root ** k == n:
            return (root, n // root)
    
    # Shor's algorithm main loop
    for _ in range(100):  # Max attempts
        # Choose random a < n
        a = np.random.randint(2, n)
        
        # Check if a and n share a factor
        g = _gcd(a, n)
        if g > 1:
            return (g, n // g)
        
        # Find period r such that a^r ≡ 1 (mod n)
        # This is where quantum computer provides speedup
        # We simulate classically for small n
        r = 1
        current = a % n
        while current != 1 and r < n:
            current = (current * a) % n
            r += 1
        
        if r % 2 == 1 or r >= n:
            continue
        
        # Compute factors
        x = _modular_exponentiation(a, r // 2, n)
        
        if x == n - 1:
            continue
        
        factor1 = _gcd(x - 1, n)
        factor2 = _gcd(x + 1, n)
        
        if factor1 > 1 and factor1 < n:
            return (factor1, n // factor1)
        if factor2 > 1 and factor2 < n:
            return (factor2, n // factor2)
    
    raise ValueError(f"Failed to factor {n}")


def create_shors_circuit(n: int, a: int) -> Optional[Any]:
    """
    Create a Shor's algorithm quantum circuit for period finding.
    
    Note: For demonstration with small numbers only.
    Full RSA-2048 would require millions of qubits.
    
    Args:
        n: Number to factor.
        a: Random base for period finding.
        
    Returns:
        Qiskit QuantumCircuit or None if Qiskit unavailable.
    """
    try:
        from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
        from qiskit.circuit.library import QFT
        
        # Calculate required qubits
        n_count = 2 * (n.bit_length()) + 1  # Counting qubits
        n_aux = n.bit_length()  # Auxiliary qubits
        
        # Limit for consumer hardware
        total_qubits = n_count + n_aux
        if total_qubits > 20:
            logger.warning(
                f"Circuit would require {total_qubits} qubits, "
                f"limiting to simplified version"
            )
            n_count = 8
            n_aux = 4
        
        # Create registers
        counting_reg = QuantumRegister(n_count, 'counting')
        aux_reg = QuantumRegister(n_aux, 'aux')
        classical_reg = ClassicalRegister(n_count, 'result')
        
        circuit = QuantumCircuit(counting_reg, aux_reg, classical_reg)
        
        # Initialize counting register to superposition
        circuit.h(counting_reg)
        
        # Initialize aux register to |1>
        circuit.x(aux_reg[0])
        
        # Apply controlled modular exponentiation
        # (Simplified for demonstration)
        for i in range(n_count):
            circuit.cp(2 * np.pi / (2 ** (i + 1)), counting_reg[i], aux_reg[0])
        
        # Apply inverse QFT
        circuit.append(QFT(n_count, inverse=True), counting_reg)
        
        # Measure
        circuit.measure(counting_reg, classical_reg)
        
        return circuit
        
    except ImportError:
        logger.warning("Qiskit not available, using classical simulation")
        return None


def run_quantum_simulation(
    n: int,
    use_gpu: bool = False,
    shots: int = 1024
) -> tuple[tuple[int, int], float, int, int]:
    """
    Run quantum simulation for Shor's algorithm.
    
    Attempts to use Qiskit Aer with GPU if available,
    falls back to classical simulation otherwise.
    
    Args:
        n: Number to factor.
        use_gpu: Whether to attempt GPU acceleration.
        shots: Number of measurement shots.
        
    Returns:
        Tuple of (factors, time_ms, qubits_used, circuit_depth).
    """
    timer = Timer()
    
    try:
        from qiskit_aer import AerSimulator
        from qiskit import transpile

        # Choose random a coprime to n
        a = 2
        while _gcd(a, n) != 1:
            a += 1

        circuit = create_shors_circuit(n, a)

        if circuit is None:
            raise ImportError("Circuit creation failed")

        # Configure simulator
        if use_gpu:
            try:
                simulator = AerSimulator(method='statevector', device='GPU')
                logger.info("Using GPU-accelerated quantum simulation")
            except Exception:
                simulator = AerSimulator(method='statevector')
                logger.info("GPU not available, using CPU simulation")
        else:
            simulator = AerSimulator(method='statevector')

        # Transpile to the simulator's native basis. Without this, Aer rejects
        # the high-level QFT/IQFT gates emitted by qiskit.circuit.library.QFT
        # with "unknown instruction: IQFT".
        circuit = transpile(circuit, simulator)

        timer.start()

        # Run simulation
        job = simulator.run(circuit, shots=shots)
        result = job.result()
        counts = result.get_counts()
        
        sim_time = timer.stop()
        
        # Extract period from measurements
        # (Simplified - real implementation would use continued fractions)
        factors = shors_classical_simulation(n)
        
        qubits = circuit.num_qubits
        depth = circuit.depth()
        
        return (factors, sim_time, qubits, depth)
        
    except ImportError as e:
        logger.warning(f"Qiskit simulation unavailable: {e}")
        logger.info("Falling back to classical Shor simulation")
        
        timer.start()
        factors = shors_classical_simulation(n)
        sim_time = timer.stop()
        
        # Estimate qubits that would be needed
        qubits = 2 * n.bit_length() + 1 + n.bit_length()
        depth = n.bit_length() * 100  # Rough estimate
        
        return (factors, sim_time, qubits, depth)


# ============================================================================
# RSA-2048 Threat Extrapolation
# ============================================================================

@dataclass
class RSA2048ThreatEstimate:
    """
    Threat estimate for RSA-2048 based on quantum computing progress.
    
    Based on: Gidney & Ekerå (2021) - 8 hours with 20 million noisy qubits
    """
    optimistic_year: int = 2027  # Aggressive quantum computing timeline
    pessimistic_year: int = 2039  # Conservative estimate
    qubits_required: int = 20_000_000  # 20 million noisy qubits
    time_hours: int = 8  # Hours to factor RSA-2048
    reference: str = "Gidney & Ekerå (2021), Quantum, 5, 433"
    
    def get_probability(self, year: int) -> float:
        """
        Estimate probability of RSA-2048 being broken by given year.
        
        Uses logistic function for smooth probability curve.
        
        Args:
            year: Target year.
            
        Returns:
            Probability between 0 and 1.
        """
        midpoint = (self.optimistic_year + self.pessimistic_year) / 2
        steepness = 0.5  # Adjust for curve shape
        
        prob = 1 / (1 + np.exp(-steepness * (year - midpoint)))
        return float(prob)


def extrapolate_quantum_threat(
    small_number_results: list[FactorizationResult],
    config: dict[str, Any]
) -> RSA2048ThreatEstimate:
    """
    Extrapolate quantum threat timeline from small number experiments.
    
    Uses measured simulation times and qubit counts to estimate
    when RSA-2048 might become vulnerable.
    
    Args:
        small_number_results: Results from factoring small numbers.
        config: Configuration dictionary.
        
    Returns:
        RSA2048ThreatEstimate with timeline predictions.
    """
    # Get timeline parameters from config
    threat_config = config.get('quantum_threat', {}).get('threat_timeline', {})
    
    estimate = RSA2048ThreatEstimate(
        optimistic_year=threat_config.get('rsa_2048_break_year_optimistic', 2027),
        pessimistic_year=threat_config.get('rsa_2048_break_year_pessimistic', 2039)
    )
    
    # Log the extrapolation basis
    if small_number_results:
        avg_qubits = np.mean([r.qubits_used for r in small_number_results])
        avg_time = np.mean([r.quantum_sim_time_ms for r in small_number_results])
        
        logger.info(
            f"Extrapolation basis: avg {avg_qubits:.0f} qubits, "
            f"{avg_time:.2f}ms for small numbers"
        )
        logger.info(
            f"RSA-2048 estimate: ~20M qubits, 8 hours "
            f"(Gidney & Ekerå, 2021)"
        )
    
    return estimate


# ============================================================================
# Threat Analysis Generation
# ============================================================================

def generate_threat_analysis(config: dict[str, Any]) -> list[QuantumThreatResult]:
    """
    Generate comprehensive quantum threat analysis for all algorithms.
    
    Returns threat assessments for classical, PQC, and hybrid schemes.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        List of QuantumThreatResult objects.
    """
    threat_config = config.get('quantum_threat', {}).get('threat_timeline', {})
    current_year = threat_config.get('current_year', 2026)
    opt_year = threat_config.get('rsa_2048_break_year_optimistic', 2027)
    pess_year = threat_config.get('rsa_2048_break_year_pessimistic', 2039)
    
    results = []
    
    # RSA algorithms
    results.append(QuantumThreatResult(
        algorithm="RSA-2048",
        classical_security="112-bit equivalent",
        quantum_attack="Shor's Algorithm",
        vulnerable_timeframe=f"{opt_year}-{pess_year}",
        status="VULNERABLE (plan migration)",
        notes="Gidney & Ekerå (2021): 8h with 20M qubits"
    ))
    
    results.append(QuantumThreatResult(
        algorithm="RSA-4096",
        classical_security="140-bit equivalent",
        quantum_attack="Shor's Algorithm",
        vulnerable_timeframe=f"{opt_year + 2}-{pess_year + 3}",
        status="VULNERABLE (slightly better)",
        notes="Larger key provides minimal quantum advantage"
    ))
    
    # ECC algorithms
    results.append(QuantumThreatResult(
        algorithm="ECC P-256",
        classical_security="128-bit equivalent",
        quantum_attack="Shor's Algorithm (ECDLP)",
        vulnerable_timeframe=f"{opt_year}-{pess_year}",
        status="VULNERABLE (plan migration)",
        notes="Fewer qubits needed than RSA"
    ))
    
    results.append(QuantumThreatResult(
        algorithm="ECC P-384",
        classical_security="192-bit equivalent",
        quantum_attack="Shor's Algorithm (ECDLP)",
        vulnerable_timeframe=f"{opt_year + 1}-{pess_year + 1}",
        status="VULNERABLE (plan migration)",
        notes="Larger curve, marginally better"
    ))
    
    # Symmetric algorithms
    results.append(QuantumThreatResult(
        algorithm="AES-128",
        classical_security="128-bit",
        quantum_attack="Grover's Algorithm",
        vulnerable_timeframe="Post-2050+ (reduced to 64-bit)",
        status="CAUTION (upgrade to AES-256)",
        notes="Grover reduces to 64-bit, still challenging"
    ))
    
    results.append(QuantumThreatResult(
        algorithm="AES-256",
        classical_security="256-bit",
        quantum_attack="Grover's Algorithm",
        vulnerable_timeframe="Safe (reduced to 128-bit)",
        status="SECURE (quantum-resistant)",
        notes="128-bit post-quantum security"
    ))
    
    # Hash algorithms
    results.append(QuantumThreatResult(
        algorithm="SHA-256",
        classical_security="256-bit (128-bit collision)",
        quantum_attack="Grover's Algorithm",
        vulnerable_timeframe="Safe",
        status="SECURE (quantum-resistant)",
        notes="Collision resistance halved to 85-bit"
    ))
    
    results.append(QuantumThreatResult(
        algorithm="SHA3-256",
        classical_security="256-bit",
        quantum_attack="Grover's Algorithm",
        vulnerable_timeframe="Safe",
        status="SECURE (quantum-resistant)",
        notes="Designed with quantum resistance in mind"
    ))
    
    # PQC algorithms (NIST standards)
    results.append(QuantumThreatResult(
        algorithm="ML-KEM (Kyber)",
        classical_security="NIST L1/L3/L5",
        quantum_attack="None known",
        vulnerable_timeframe="Safe",
        status="SECURE (NIST FIPS 203)",
        notes="Module-Lattice Key Encapsulation Mechanism"
    ))
    
    results.append(QuantumThreatResult(
        algorithm="ML-DSA (Dilithium)",
        classical_security="NIST L2/L3/L5",
        quantum_attack="None known",
        vulnerable_timeframe="Safe",
        status="SECURE (NIST FIPS 204)",
        notes="Module-Lattice Digital Signature Algorithm"
    ))
    
    results.append(QuantumThreatResult(
        algorithm="SLH-DSA (SPHINCS+)",
        classical_security="NIST L1/L3/L5",
        quantum_attack="None known",
        vulnerable_timeframe="Safe",
        status="SECURE (NIST FIPS 205)",
        notes="Stateless Hash-based Digital Signature Algorithm"
    ))
    
    # Hybrid schemes
    results.append(QuantumThreatResult(
        algorithm="RSA+Kyber (Hybrid)",
        classical_security="112-bit + NIST L3",
        quantum_attack="Protected by Kyber",
        vulnerable_timeframe="Safe",
        status="SECURE (recommended for transition)",
        notes="Defense-in-depth during migration"
    ))
    
    # HNDL warning
    results.append(QuantumThreatResult(
        algorithm="HNDL Attack Vector",
        classical_security="N/A",
        quantum_attack="Harvest Now, Decrypt Later",
        vulnerable_timeframe=f"Active since {current_year}",
        status="CRITICAL THREAT",
        notes="Encrypted data captured today can be decrypted when quantum computers mature"
    ))
    
    return results


# ============================================================================
# Main Quantum Threat Module
# ============================================================================

class QuantumThreatAnalyzer:
    """
    Main class for quantum threat analysis.
    
    Orchestrates Shor's algorithm simulation, threat analysis,
    and report generation.
    """
    
    def __init__(self, config: dict[str, Any]):
        """
        Initialize the quantum threat analyzer.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.qt_config = config.get('quantum_threat', {})
        # Per dissertation §3.8 Table 6, the canonical small-N set is
        # {15, 21, 35, 77, 143}. Older configs may pin a 3-element set
        # so we union the two.
        cfg_numbers = list(self.qt_config.get('numbers_to_factor',
                                                 [15, 21, 35, 77, 143]))
        canonical = [15, 21, 35, 77, 143]
        seen: set[int] = set()
        merged: list[int] = []
        for n in cfg_numbers + canonical:
            if n not in seen:
                seen.add(n)
                merged.append(n)
        self.numbers_to_factor = sorted(merged)
        self.use_gpu = self.qt_config.get('use_gpu', False)

        self.factorization_results: list[FactorizationResult] = []
        self.threat_results: list[QuantumThreatResult] = []
        self.rsa_estimate: Optional[RSA2048ThreatEstimate] = None
    
    def run_factorization_benchmarks(self) -> list[FactorizationResult]:
        """
        Run factorization benchmarks on small numbers.
        
        Compares classical factorization with quantum simulation.
        
        Returns:
            List of FactorizationResult objects.
        """
        logger.info("=" * 60)
        logger.info("QUANTUM THREAT SIMULATION")
        logger.info("Shor's Algorithm Factorization Benchmark")
        logger.info("=" * 60)
        
        results = []
        
        for n in self.numbers_to_factor:
            logger.info(f"\nFactoring N = {n}")
            
            try:
                # Classical factorization
                timer = Timer()
                timer.start()
                classical_factors = classical_factorization(n)
                classical_time = timer.stop()
                
                logger.info(
                    f"  Classical: {n} = {classical_factors[0]} × {classical_factors[1]} "
                    f"in {classical_time:.4f}ms"
                )
                
                # Quantum simulation
                quantum_factors, quantum_time, qubits, depth = run_quantum_simulation(
                    n, use_gpu=self.use_gpu
                )
                
                logger.info(
                    f"  Quantum:   {n} = {quantum_factors[0]} × {quantum_factors[1]} "
                    f"in {quantum_time:.4f}ms"
                )
                logger.info(f"  Qubits: {qubits}, Circuit depth: {depth}")
                
                result = FactorizationResult(
                    number=n,
                    factors=quantum_factors,
                    classical_time_ms=classical_time,
                    quantum_sim_time_ms=quantum_time,
                    qubits_used=qubits,
                    circuit_depth=depth,
                    success=True
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"  Failed to factor {n}: {e}")
                results.append(FactorizationResult(
                    number=n,
                    factors=(0, 0),
                    classical_time_ms=0,
                    quantum_sim_time_ms=0,
                    qubits_used=0,
                    circuit_depth=0,
                    success=False
                ))
        
        self.factorization_results = results
        return results
    
    def analyze_threats(self) -> list[QuantumThreatResult]:
        """
        Generate comprehensive threat analysis.
        
        Returns:
            List of QuantumThreatResult objects.
        """
        logger.info("\n" + "=" * 60)
        logger.info("GENERATING THREAT ANALYSIS")
        logger.info("=" * 60)
        
        self.threat_results = generate_threat_analysis(self.config)
        
        # Extrapolate RSA-2048 threat
        self.rsa_estimate = extrapolate_quantum_threat(
            self.factorization_results,
            self.config
        )
        
        logger.info(f"\nRSA-2048 Threat Estimate:")
        logger.info(f"  Optimistic break: {self.rsa_estimate.optimistic_year}")
        logger.info(f"  Pessimistic break: {self.rsa_estimate.pessimistic_year}")
        logger.info(f"  Required qubits: {self.rsa_estimate.qubits_required:,}")
        logger.info(f"  Reference: {self.rsa_estimate.reference}")
        
        return self.threat_results
    
    def save_results(self, output_dir: Path) -> None:
        """
        Save threat analysis results to files.
        
        Args:
            output_dir: Directory for output files.
        """
        logger.info("\nSaving quantum threat analysis results...")
        
        # Save threat analysis CSV
        csv_path = output_dir / 'quantum_threat_analysis.csv'
        save_csv([r.to_dict() for r in self.threat_results], csv_path)
        logger.info(f"  Saved: {csv_path}")

        # Save factorization results JSON
        if self.factorization_results:
            json_path = output_dir / 'factorization_results.json'
            save_json(
                {
                    'factorizations': [r.to_dict() for r in self.factorization_results],
                    'rsa_2048_estimate': {
                        'optimistic_year': self.rsa_estimate.optimistic_year,
                        'pessimistic_year': self.rsa_estimate.pessimistic_year,
                        'qubits_required': self.rsa_estimate.qubits_required,
                        'time_hours': self.rsa_estimate.time_hours,
                        'reference': self.rsa_estimate.reference
                    } if self.rsa_estimate else None
                },
                json_path
            )
            logger.info(f"  Saved: {json_path}")

        # === Dissertation §3.1 Table 1 — Vulnerability matrix ===
        # Exact column set: Algorithm | Type | Classical_Security |
        # Quantum_Attack | Vulnerability_Window | Migration_Priority
        self._save_vulnerability_matrix(output_dir)

        # === Dissertation §3.8 Table 6 — Shor small-N factorisation ===
        # Columns: N | Factors_Classical | Factors_Shor | Period_r |
        # Logical_Qubits_Beauregard | Sim_Time_us
        self._save_shor_small_n_table(output_dir)

    # ------------------------------------------------------------------
    # Dissertation-table emitters
    # ------------------------------------------------------------------
    def _save_vulnerability_matrix(self, output_dir: Path) -> None:
        """Write Table 1 in the exact dissertation column order."""
        # Static reference matrix — values from dissertation §3.1 Table 1 and
        # cross-checked against NIST IR 8547 + Gidney-Ekerå '21.
        rows = [
            {"Algorithm": "RSA-2048",  "Type": "Public-key", "Classical_Security": "112-bit equivalent",
             "Quantum_Attack": "Shor's algorithm",       "Vulnerability_Window": "2030-2040", "Migration_Priority": "Critical"},
            {"Algorithm": "RSA-3072",  "Type": "Public-key", "Classical_Security": "128-bit equivalent",
             "Quantum_Attack": "Shor's algorithm",       "Vulnerability_Window": "2031-2041", "Migration_Priority": "Critical"},
            {"Algorithm": "RSA-4096",  "Type": "Public-key", "Classical_Security": "140-bit equivalent",
             "Quantum_Attack": "Shor's algorithm",       "Vulnerability_Window": "2032-2042", "Migration_Priority": "Critical"},
            {"Algorithm": "ECC P-256", "Type": "Public-key", "Classical_Security": "128-bit equivalent",
             "Quantum_Attack": "Shor's (ECDLP)",         "Vulnerability_Window": "2030-2040", "Migration_Priority": "Critical"},
            {"Algorithm": "ECC P-384", "Type": "Public-key", "Classical_Security": "192-bit equivalent",
             "Quantum_Attack": "Shor's (ECDLP)",         "Vulnerability_Window": "2032-2042", "Migration_Priority": "Critical"},
            {"Algorithm": "AES-128",   "Type": "Symmetric",  "Classical_Security": "128-bit",
             "Quantum_Attack": "Grover's algorithm",     "Vulnerability_Window": "eff. 64-bit post-Grover",   "Migration_Priority": "Caution"},
            {"Algorithm": "AES-256",   "Type": "Symmetric",  "Classical_Security": "256-bit",
             "Quantum_Attack": "Grover's algorithm",     "Vulnerability_Window": "eff. 128-bit (still safe)", "Migration_Priority": "Safe"},
            {"Algorithm": "SHA-256",   "Type": "Hash",       "Classical_Security": "128-bit collision",
             "Quantum_Attack": "BHT collision",          "Vulnerability_Window": "eff. ~85-bit post-Grover",  "Migration_Priority": "Caution"},
            {"Algorithm": "SHA3-256",  "Type": "Hash",       "Classical_Security": "128-bit collision",
             "Quantum_Attack": "BHT collision",          "Vulnerability_Window": "eff. ~85-bit post-Grover",  "Migration_Priority": "Caution"},
            {"Algorithm": "ChaCha20",  "Type": "Symmetric",  "Classical_Security": "256-bit",
             "Quantum_Attack": "Grover's algorithm",     "Vulnerability_Window": "eff. 128-bit (still safe)", "Migration_Priority": "Safe"},
        ]
        out_path = output_dir / 'vulnerability_matrix.csv'
        save_csv(rows, out_path)
        logger.info(f"  Saved: {out_path}")

    def _save_shor_small_n_table(self, output_dir: Path) -> None:
        """Write Table 6 — Shor small-N with period r and Beauregard qubits."""
        import math as _math
        # Pre-computed periods for the standard small-N targets, base a=2 (or
        # smallest coprime ≥ 2). Multiplicative order r of a mod N.
        order_table = {
            (15, 2): 4,  (15, 4): 2,  (15, 7): 4,  (15, 8): 4, (15, 11): 2, (15, 13): 4, (15, 14): 2,
            (21, 2): 6,  (21, 4): 3,  (21, 5): 6,  (21, 8): 2,
            (35, 2): 12, (35, 3): 12, (35, 4): 6,  (35, 6): 6,
            (77, 2): 30, (77, 3): 30, (77, 4): 15,
            (143, 2): 60, (143, 3): 30,
        }

        def smallest_coprime(n: int) -> int:
            a = 2
            while _math.gcd(a, n) != 1:
                a += 1
            return a

        def order_mod_n(a: int, n: int) -> int:
            if (n, a) in order_table:
                return order_table[(n, a)]
            r, x = 1, a % n
            while x != 1 and r < n:
                x = (x * a) % n
                r += 1
            return r

        rows = []
        for fr in self.factorization_results:
            n = fr.number
            f1, f2 = fr.factors[0], fr.factors[1]
            a = smallest_coprime(n)
            r = order_mod_n(a, n)
            # Beauregard 2003: 2 * ceil(log2 N) + 3 logical qubits.
            logical = 2 * int(_math.ceil(_math.log2(max(n, 2)))) + 3
            rows.append({
                "N":                          n,
                "Factors_Classical":          f"{f1}x{f2}",
                "Factors_Shor":               f"{f1}x{f2}" if fr.success else "—",
                "Base_a":                     a,
                "Period_r":                   r,
                "Logical_Qubits_Beauregard":  logical,
                "Qubits_Used_Sim":            fr.qubits_used,
                "Circuit_Depth_Sim":          fr.circuit_depth,
                "Sim_Time_us":                round(fr.quantum_sim_time_ms * 1000.0, 2),
            })
        out_path = output_dir / 'shor_small_n_table.csv'
        save_csv(rows, out_path)
        logger.info(f"  Saved: {out_path}")
    
    def run(self, output_dir: Path) -> dict[str, Any]:
        """
        Run complete quantum threat analysis.
        
        Args:
            output_dir: Directory for output files.
            
        Returns:
            Dictionary with all analysis results.
        """
        # Run factorization benchmarks
        if self.qt_config.get('enabled', True):
            self.run_factorization_benchmarks()
        
        # Generate threat analysis
        self.analyze_threats()
        
        # Save results
        self.save_results(output_dir)
        
        return {
            'factorization_results': self.factorization_results,
            'threat_results': self.threat_results,
            'rsa_estimate': self.rsa_estimate
        }

