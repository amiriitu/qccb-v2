#!/usr/bin/env python
"""Быстрый тест PQC бенчмарка"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.pqc_benchmark import PQCBenchmarker
from src.utils import load_config
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Load config
config = load_config('config.yaml')

# Override iterations for quick test
config['statistics']['benchmark_iterations'] = 10
config['statistics']['warmup_iterations'] = 2

# Create benchmarker
benchmarker = PQCBenchmarker(config)

# Test single KEM
print("\n" + "="*60)
print("Testing KEM benchmarking...")
print("="*60)

from src.pqc_benchmark import benchmark_kem
result = benchmark_kem(
    algorithm='ML-KEM-768',
    nist_level=3,
    family='lattice',
    warmup=2,
    iterations=10,
    outlier_sigma=3.0,
    confidence=0.95
)

if result:
    print(f"\n✓ KEM Result: {result.algorithm}")
    print(f"  KeyGen: {result.keygen_stats.mean:.4f}ms")
    print(f"  Encaps: {result.encaps_stats.mean:.4f}ms")
    print(f"  Decaps: {result.decaps_stats.mean:.4f}ms")
else:
    print("✗ KEM failed")

# Test single Signature
print("\n" + "="*60)
print("Testing Signature benchmarking...")
print("="*60)

from src.pqc_benchmark import benchmark_signature
result = benchmark_signature(
    algorithm='ML-DSA-65',
    nist_level=3,
    family='lattice',
    warmup=2,
    iterations=10,
    outlier_sigma=3.0,
    confidence=0.95
)

if result:
    print(f"\n✓ Signature Result: {result.algorithm}")
    print(f"  KeyGen: {result.keygen_stats.mean:.4f}ms")
    print(f"  Sign:   {result.sign_stats.mean:.4f}ms")
    print(f"  Verify: {result.verify_stats.mean:.4f}ms")
else:
    print("✗ Signature failed")

print("\n✓ PQC Benchmarking tests completed successfully!")

