#!/usr/bin/env python3
# ============================================================================
# QCCB v2.0 - Quantum Computing Cryptography Benchmark (Scientific Edition)
# Main Entry Point
# ============================================================================
"""
QCCB v2.0 - A comprehensive benchmark suite for evaluating post-quantum
cryptographic algorithms on consumer hardware.

This tool is designed for Master's thesis research on:
"Data Protection in Quantum Computing Context: Cryptography Resilience Study"

Features:
- Quantum threat simulation (Shor's algorithm)
- PQC benchmarking (ML-KEM, ML-DSA, SLH-DSA)
- Classical cryptography comparison (RSA, ECC, AES, SHA)
- Hybrid scheme analysis (RSA+Kyber)
- Statistical rigor (1000 iterations, 95% CI)
- Publication-quality visualizations
- Comprehensive scientific report

Usage:
    python main.py                    # Run full benchmark suite
    python main.py --config custom.yaml   # Use custom config
    python main.py --quick            # Quick mode (100 iterations)
    python main.py --pqc-only         # Only PQC benchmarks
    python main.py --help             # Show help

Output:
    results/
    ├── quantum_threat_analysis.csv
    ├── pqc_benchmarks.csv
    ├── pqc_benchmarks.json
    ├── comparative_analysis.csv
    ├── migration_strategy.txt
    ├── report.txt
    ├── quantum_threat_timeline.png
    ├── pqc_performance_comparison.png
    ├── migration_roadmap.png
    └── benchmark.log

Author: Amir
Date: 2026
Version: 2.0.0
License: MIT

References:
    - NIST FIPS 203/204/205
    - Gidney & Ekerå (2021)
    - Open Quantum Safe (liboqs)
"""

from __future__ import annotations

import argparse
import random
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import (
    load_config,
    get_output_dir,
    setup_logging,
    detect_hardware,
    get_library_versions,
    HardwareInfo
)
from src.quantum_threat import QuantumThreatAnalyzer
from src.pqc_benchmark import PQCBenchmarker
from src.comparative_analysis import ComparativeAnalyzer
from src.visualization import Visualizer
from src.report_generator import ReportGenerator


# ============================================================================
# Banner and Version Info
# ============================================================================

BANNER = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ██████╗  ██████╗ ██████╗██████╗     ██╗   ██╗██████╗    ██████╗            ║
║  ██╔═══██╗██╔════╝██╔════╝██╔══██╗    ██║   ██║╚════██╗  ██╔═████╗           ║
║  ██║   ██║██║     ██║     ██████╔╝    ██║   ██║ █████╔╝  ██║██╔██║           ║
║  ██║▄▄ ██║██║     ██║     ██╔══██╗    ╚██╗ ██╔╝██╔═══╝   ████╔╝██║           ║
║  ╚██████╔╝╚██████╗╚██████╗██████╔╝     ╚████╔╝ ███████╗  ╚██████╔╝           ║
║   ╚══▀▀═╝  ╚═════╝ ╚═════╝╚═════╝       ╚═══╝  ╚══════╝   ╚═════╝            ║
║                                                                              ║
║  Quantum Computing Cryptography Benchmark - Scientific Edition               ║
║  Version 2.0.0                                                               ║
║                                                                              ║
║  Master's Thesis: Data Protection in Quantum Computing Context              ║
║  Focus: Cryptography Resilience Study                                        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


def print_banner() -> None:
    """Print the QCCB banner."""
    print(BANNER)


# ============================================================================
# Command Line Arguments
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="QCCB v2.0 - Quantum Computing Cryptography Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                      Run full benchmark suite
  python main.py --quick              Quick mode (100 iterations)
  python main.py --pqc-only           Only run PQC benchmarks
  python main.py --no-quantum         Skip quantum simulation
  python main.py --config custom.yaml Use custom configuration

Output files are saved to the 'results' directory by default.

For Master's thesis research on Post-Quantum Cryptography.
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Quick mode: 100 iterations instead of 1000'
    )
    
    parser.add_argument(
        '--pqc-only',
        action='store_true',
        help='Only run PQC benchmarks (skip classical and quantum)'
    )
    
    parser.add_argument(
        '--classical-only',
        action='store_true',
        help='Only run classical benchmarks'
    )
    
    parser.add_argument(
        '--no-quantum',
        action='store_true',
        help='Skip quantum threat simulation'
    )
    
    parser.add_argument(
        '--no-visualize',
        action='store_true',
        help='Skip visualization generation'
    )
    
    parser.add_argument(
        '--no-report',
        action='store_true',
        help='Skip report generation'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory (default: from config)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='QCCB v2.0.0'
    )
    
    return parser.parse_args()


# ============================================================================
# Main Orchestrator
# ============================================================================

class QCCBenchmark:
    """
    Main orchestrator for the QCCB benchmark suite.
    
    Coordinates all benchmark modules and generates comprehensive output.
    """
    
    def __init__(self, config: dict[str, Any], args: argparse.Namespace):
        """
        Initialize the benchmark orchestrator.
        
        Args:
            config: Configuration dictionary.
            args: Command line arguments.
        """
        self.config = config
        self.args = args
        self.run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        self.seed = self.config.get('statistics', {}).get('random_seed')
        
        # Apply quick mode
        if args.quick:
            self.config['statistics']['benchmark_iterations'] = 100
            self.config['statistics']['warmup_iterations'] = 5
        
        # Set output directory
        if args.output:
            self.config['output']['directory'] = args.output
        
        self.output_dir = get_output_dir(config)
        
        # Setup logging
        self.logger = setup_logging(config)

        # Fix random seeds for reproducibility when provided
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        
        # Detect hardware
        self.hardware_info = detect_hardware()
        
        # Results storage
        self.results = {
            'quantum_threat': None,
            'factorization': None,
            'pqc_kem': None,
            'pqc_sig': None,
            'classical': None,
            'hybrid': None,
            'sci': None
        }
        
        # Timing
        self.start_time = None
        self.end_time = None
    
    def log_header(self) -> None:
        """Log benchmark header information."""
        self.logger.info("=" * 70)
        self.logger.info("QCCB v2.0 - Quantum Computing Cryptography Benchmark")
        self.logger.info("Scientific Edition")
        self.logger.info("=" * 70)
        self.logger.info("")
        self.logger.info(f"Thesis: {self.config['project'].get('thesis_title', 'N/A')}")
        self.logger.info(f"Author: {self.config['project'].get('author', 'N/A')}")
        self.logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("")
        self.logger.info(f"Run ID: {self.run_id}")
        if self.seed is not None:
            self.logger.info(f"Random Seed: {self.seed}")
        self.logger.info(str(self.hardware_info))
        self.logger.info("")
        
        # Log library versions
        self.logger.info("Library Versions:")
        versions = get_library_versions()
        for lib, ver in versions.items():
            self.logger.info(f"  {lib}: {ver}")
        self.logger.info("")
        
        # Log configuration
        stats = self.config.get('statistics', {})
        self.logger.info("Benchmark Configuration:")
        self.logger.info(f"  Iterations: {stats.get('benchmark_iterations', 1000)}")
        self.logger.info(f"  Warmup: {stats.get('warmup_iterations', 10)}")
        self.logger.info(f"  Outlier Sigma: {stats.get('outlier_sigma', 3.0)}")
        self.logger.info(f"  Confidence Level: {stats.get('confidence_level', 0.95)}")
        self.logger.info(f"  Output Directory: {self.output_dir}")
        self.logger.info("")
    
    def run_quantum_threat(self) -> None:
        """Run quantum threat analysis."""
        if self.args.no_quantum or self.args.pqc_only or self.args.classical_only:
            self.logger.info("Skipping quantum threat analysis")
            return
        
        analyzer = QuantumThreatAnalyzer(self.config)
        results = analyzer.run(self.output_dir)
        
        self.results['quantum_threat'] = results.get('threat_results', [])
        self.results['factorization'] = results.get('factorization_results', [])
    
    def run_pqc_benchmarks(self) -> None:
        """Run PQC benchmarks."""
        if self.args.classical_only:
            self.logger.info("Skipping PQC benchmarks")
            return
        
        benchmarker = PQCBenchmarker(self.config)
        results = benchmarker.run(self.output_dir)
        
        self.results['pqc_kem'] = results.get('kem_results', [])
        self.results['pqc_sig'] = results.get('sig_results', [])
        
        # Print summary
        self.logger.info(benchmarker.get_summary())
    
    def run_classical_benchmarks(self) -> None:
        """Run classical cryptography benchmarks."""
        if self.args.pqc_only:
            self.logger.info("Skipping classical benchmarks")
            return
        
        analyzer = ComparativeAnalyzer(self.config)
        results = analyzer.run(
            self.output_dir,
            pqc_kem_results=self.results['pqc_kem'],
            pqc_sig_results=self.results['pqc_sig']
        )
        
        self.results['classical'] = results.get('classical_results', [])
        self.results['hybrid'] = results.get('hybrid_results', [])
        self.results['sci'] = results.get('sci_analyses', [])
    
    def generate_visualizations(self) -> None:
        """Generate all visualizations."""
        if self.args.no_visualize:
            self.logger.info("Skipping visualization generation")
            return
        
        visualizer = Visualizer(self.config)
        visualizer.generate_all(
            self.output_dir,
            threat_results=self.results['quantum_threat'],
            kem_results=self.results['pqc_kem'],
            sig_results=self.results['pqc_sig']
        )
    
    def generate_reports(self) -> None:
        """Generate scientific reports."""
        if self.args.no_report:
            self.logger.info("Skipping report generation")
            return
        
        generator = ReportGenerator(self.config)
        generator.generate_report(
            self.output_dir,
            self.hardware_info,
            threat_results=self.results['quantum_threat'],
            factorization_results=self.results['factorization'],
            kem_results=self.results['pqc_kem'],
            sig_results=self.results['pqc_sig'],
            classical_results=self.results['classical'],
            hybrid_results=self.results['hybrid'],
            sci_analyses=self.results['sci']
        )
    
    def log_summary(self) -> None:
        """Log final summary."""
        elapsed = self.end_time - self.start_time
        
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("BENCHMARK COMPLETE")
        self.logger.info("=" * 70)
        self.logger.info(f"Run ID: {self.run_id}")
        if self.seed is not None:
            self.logger.info(f"Random Seed: {self.seed}")
        self.logger.info(f"Total execution time: {elapsed:.2f} seconds")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info("")
        self.logger.info("Generated files:")
        
        for f in self.output_dir.iterdir():
            size = f.stat().st_size
            if size > 1024:
                size_str = f"{size/1024:.1f} KB"
            else:
                size_str = f"{size} B"
            self.logger.info(f"  {f.name}: {size_str}")
        
        self.logger.info("")
        self.logger.info("For thesis usage:")
        self.logger.info("  • Include quantum_threat_timeline.png in Chapter 2")
        self.logger.info("  • Include pqc_benchmarks.csv as Table in Chapter 3")
        self.logger.info("  • Include pqc_performance_comparison.png in Chapter 3")
        self.logger.info("  • Include migration_roadmap.png in Chapter 4")
        self.logger.info("  • Quote findings from report.txt")
        self.logger.info("  • Include benchmark.log as Appendix")
        self.logger.info("")
        self.logger.info("Cite as: 'Original research conducted using QCCB v2.0'")
        self.logger.info("=" * 70)
    
    def run(self) -> int:
        """
        Run the complete benchmark suite.
        
        Returns:
            Exit code (0 for success, 1 for failure).
        """
        self.start_time = time.time()
        
        try:
            # Log header
            self.log_header()
            
            # Run benchmarks
            self.run_quantum_threat()
            self.run_pqc_benchmarks()
            self.run_classical_benchmarks()
            
            # Generate outputs
            self.generate_visualizations()
            self.generate_reports()
            
            self.end_time = time.time()
            self.log_summary()
            
            return 0
            
        except KeyboardInterrupt:
            self.logger.warning("\nBenchmark interrupted by user")
            return 1
            
        except Exception as e:
            self.logger.exception(f"Benchmark failed with error: {e}")
            return 1


# ============================================================================
# Main Entry Point
# ============================================================================

def main() -> int:
    """
    Main entry point for QCCB v2.0.
    
    Returns:
        Exit code.
    """
    # Print banner
    print_banner()
    
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {args.config}")
        print("Please ensure config.yaml exists in the current directory.")
        return 1
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1
    
    # Create and run benchmark
    benchmark = QCCBenchmark(config, args)
    return benchmark.run()


if __name__ == '__main__':
    sys.exit(main())


