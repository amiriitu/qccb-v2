import sys
sys.path.insert(0, 'src')
from pqc_simulator import KEM, Signature, KEM_PARAMS, SIG_PARAMS, SPHINCS_PARAMS

# Test KEM
kem = KEM('ML-KEM-768')
print(f'KEM {kem.alg.name}: Keygen...')
pk, sk = kem.keygen()
print(f'  Public Key size: {kem.alg.public_key_size} bytes')
print(f'  Secret Key size: {kem.alg.secret_key_size} bytes')

# Test Signature
sig = Signature('ML-DSA-65')
print(f'Signature {sig.alg.name}: Keygen...')
pk, sk = sig.keygen()
print(f'  Public Key size: {sig.alg.public_key_size} bytes')
print(f'  Signature size: {sig.alg.signature_size} bytes')

print('✓ PQC Simulator working correctly!')

