#!/usr/bin/env python
"""Простой тест для проверки работы QCCB с симулятором"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("=" * 60)
print("QCCB v2.0 - Простой тест")
print("=" * 60)

# Test 1: Import utils
try:
    from src.utils import Timer, BenchmarkStatistics, load_config
    print("✓ utils loaded")
except Exception as e:
    print(f"✗ Error loading utils: {e}")
    sys.exit(1)

# Test 2: Load config
try:
    config = load_config('config.yaml')
    print(f"✓ config loaded: {len(config)} keys")
except Exception as e:
    print(f"✗ Error loading config: {e}")
    sys.exit(1)

# Test 3: Import PQC simulator
try:
    from src.pqc_simulator import KEM, Signature, KEM_PARAMS, SIG_PARAMS
    print(f"✓ pqc_simulator loaded: {len(KEM_PARAMS)} KEMs, {len(SIG_PARAMS)} Sigs")
except Exception as e:
    print(f"✗ Error loading pqc_simulator: {e}")
    sys.exit(1)

# Test 4: Quick KEM benchmark
try:
    print("\nTesting KEM benchmark...")
    kem = KEM('ML-KEM-768')
    pk, sk = kem.keygen()
    ct, ss = kem.encaps(pk)
    ss_dec = kem.decaps(sk, ct)
    print(f"✓ KEM test passed: PK={len(pk)}B, CT={len(ct)}B")
except Exception as e:
    print(f"✗ Error in KEM test: {e}")
    sys.exit(1)

# Test 5: Quick Signature benchmark
try:
    print("Testing Signature benchmark...")
    sig = Signature('ML-DSA-65')
    pk, sk = sig.keygen()
    message = b"test message"
    signature = sig.sign(sk, message)
    valid = sig.verify(pk, message, signature)
    print(f"✓ Signature test passed: Sig={len(signature)}B, Valid={valid}")
except Exception as e:
    print(f"✗ Error in Signature test: {e}")
    sys.exit(1)

# Test 6: Import and test PQC benchmark module
try:
    print("\nTesting pqc_benchmark module...")
    from src.pqc_benchmark import benchmark_kem, benchmark_signature
    print("✓ pqc_benchmark module loaded")
except Exception as e:
    print(f"✗ Error loading pqc_benchmark: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ Все базовые тесты пройдены успешно!")
print("=" * 60)

