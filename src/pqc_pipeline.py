"""
PQC benchmark pipeline — orchestrator for the software-side artifacts:

  - PQC algorithms (Kyber, Dilithium, SPHINCS+) per NIST FIPS 203/204/205
  - Classical baselines (RSA, ECC, AES, SHA)
  - Hybrid scheme analysis (RSA + Kyber)
  - Quantum threat matrix (Shor / Grover vulnerabilities)
  - 5-phase migration roadmap (2024-2035)
  - Original SCI metric per Scientific journal 'Bulletin of the CAA' 2026

Outputs everything to results/1_pqc_benchmarks/ in the new project layout.
Falls back to the included PQC simulator when liboqs is not installed
(the standard situation on Windows).
"""
from __future__ import annotations

import argparse
import logging
import sys
from contextlib import suppress
import time
import uuid
from pathlib import Path
from typing import Callable

if sys.platform == "win32":
    with suppress(AttributeError, OSError):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

import os
os.environ.setdefault("QCCB_FORCE_SIMULATOR", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, setup_logging, detect_hardware
from src.pqc_benchmark import PQCBenchmarker
from src.comparative_analysis import ComparativeAnalyzer
from src.quantum_threat import QuantumThreatAnalyzer
from src.visualization import Visualizer
from src.report_generator import ReportGenerator


def run_pqc_pipeline(
    output_dir: Path | None = None,
    quick: bool = True,
    config_path: str = "config.yaml",
    status: Callable[[str], None] | None = None,
) -> dict[str, Path]:
    """
    Run the full PQC + classical + threat + migration pipeline.
    Returns a dict with at least 'out_dir' pointing to the artifacts folder.
    """
    status = status or (lambda m: None)

    if output_dir is None:
        stamp = time.strftime("%d%m%Y %H %M")
        output_dir = (PROJECT_ROOT / "results" / "1_pqc_benchmarks" / stamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    config["output"]["directory"] = str(output_dir)

    if quick:
        config["statistics"]["benchmark_iterations"] = 100
        config["statistics"]["warmup_iterations"] = 5

    logger = setup_logging(config)
    logger.info("=" * 60)
    logger.info("PQC pipeline starting")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Quick mode: {quick}")
    logger.info("=" * 60)

    hw = detect_hardware()
    logger.info(str(hw))

    status("[1/4] Quantum threat analysis (Shor/Grover vulnerability matrix)...")
    threat = QuantumThreatAnalyzer(config)
    threat_res = threat.run(output_dir)
    threat_results = threat_res.get("threat_results", [])
    factorization_results = threat_res.get("factorization_results", [])

    status("[2/4] PQC benchmarks (Kyber/Dilithium/SPHINCS+)...")
    pqc = PQCBenchmarker(config)
    pqc_res = pqc.run(output_dir)
    kem_results = pqc_res.get("kem_results", [])
    sig_results = pqc_res.get("sig_results", [])

    status("[3/4] Classical baselines + hybrid schemes + SCI...")
    comp = ComparativeAnalyzer(config)
    comp_res = comp.run(
        output_dir,
        pqc_kem_results=kem_results,
        pqc_sig_results=sig_results,
    )

    status("[4/4] Visualizations + migration roadmap + scientific report...")
    viz = Visualizer(config)
    viz.generate_all(
        output_dir,
        threat_results=threat_results,
        kem_results=kem_results,
        sig_results=sig_results,
    )

    rep = ReportGenerator(config)
    rep.generate_report(
        output_dir,
        hw,
        threat_results=threat_results,
        factorization_results=factorization_results,
        kem_results=kem_results,
        sig_results=sig_results,
        classical_results=comp_res.get("classical_results", []),
        hybrid_results=comp_res.get("hybrid_results", []),
        sci_analyses=comp_res.get("sci_analyses", []),
    )

    # === AVX2 vs reference PQC comparison (dissertation §3.22) ===
    # The thesis explicitly notes ~2× speedup from liboqs AVX2 path over
    # portable C. We measure it side-by-side here using the per-algorithm
    # liboqs AVX2 speedup table from `cpu_features.AVX2_SPEEDUP`.
    status("[+] AVX2 vs reference PQC benchmark (dissertation §3.22)...")
    try:
        from src.pqc_avx2_comparison import run_avx2_comparison
        iters = 30 if quick else 100
        run_avx2_comparison(output_dir, iterations=iters)
    except Exception as e:
        logger.warning(f"AVX2 comparison skipped: {e!r}")

    artifacts: dict[str, Path] = {"out_dir": output_dir}
    for f in output_dir.iterdir():
        if f.is_file():
            artifacts[f.name] = f
    status(f"PQC pipeline complete. {len(artifacts)-1} files in {output_dir}")
    return artifacts


def main():
    p = argparse.ArgumentParser(description="PQC benchmark pipeline")
    p.add_argument("--output", type=str, default=None,
                   help="Output directory (default: results/1_pqc_benchmarks/)")
    p.add_argument("--full", action="store_true",
                   help="Full mode: 1000 iterations (default: quick mode 100)")
    p.add_argument("--config", type=str, default="config.yaml")
    args = p.parse_args()

    out_dir = Path(args.output) if args.output else None
    artifacts = run_pqc_pipeline(
        output_dir=out_dir,
        quick=not args.full,
        config_path=args.config,
        status=lambda m: print(m),
    )
    print(f"\nDone. Artifacts: {artifacts['out_dir']}")
    for k, v in artifacts.items():
        if k != "out_dir":
            print(f"  ✓ {v.name}")


if __name__ == "__main__":
    main()

