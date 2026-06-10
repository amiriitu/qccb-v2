# ============================================================================
# QCCB v2.0 - Utility Modules
# Statistics, Logging, and Helper Functions
# ============================================================================
"""
Utility module providing statistical analysis, logging, hardware detection,
and helper functions for the QCCB benchmark suite.

Author: Amir
Date: 2026
Thesis: Data Protection in Quantum Computing Context: Cryptography Resilience Study
"""

from __future__ import annotations

import logging
import os
import platform
import sys
import time
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import psutil
import yaml
from scipy import stats


# ============================================================================
# Configuration Management
# ============================================================================

def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file.
        
    Returns:
        Dictionary containing configuration parameters.
        
    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is malformed.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def get_output_dir(config: dict[str, Any]) -> Path:
    """
    Get and create output directory from configuration.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        Path to output directory.
    """
    output_dir = Path(config.get('output', {}).get('directory', 'results'))
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(config: dict[str, Any]) -> logging.Logger:
    """
    Configure logging for the benchmark suite.
    
    Args:
        config: Configuration dictionary with logging settings.
        
    Returns:
        Configured logger instance.
    """
    log_config = config.get('logging', {})
    output_dir = get_output_dir(config)
    
    # Create logger
    logger = logging.getLogger('QCCB')
    logger.setLevel(getattr(logging, log_config.get('level', 'INFO')))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler
    log_file = output_dir / 'benchmark.log'
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    log_format = log_config.get(
        'format', 
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
    )
    date_format = log_config.get('date_format', '%Y-%m-%dT%H:%M:%S')
    formatter = logging.Formatter(log_format, datefmt=date_format)
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# ============================================================================
# Statistical Analysis
# ============================================================================

@dataclass
class BenchmarkStatistics:
    """
    Container for benchmark statistical results.
    
    Attributes:
        mean: Arithmetic mean of measurements.
        std: Standard deviation.
        ci_lower: Lower bound of 95% confidence interval.
        ci_upper: Upper bound of 95% confidence interval.
        min_val: Minimum observed value.
        max_val: Maximum observed value.
        median: Median value.
        n_samples: Number of samples after outlier removal.
        n_outliers: Number of outliers removed.
    """
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    min_val: float
    max_val: float
    median: float
    n_samples: int
    n_outliers: int = 0
    raw_data: np.ndarray = field(default_factory=lambda: np.array([]))
    
    def __repr__(self) -> str:
        return (
            f"BenchmarkStatistics(mean={self.mean:.6f} ± {self.std:.6f}, "
            f"95% CI=[{self.ci_lower:.6f}, {self.ci_upper:.6f}], "
            f"n={self.n_samples}, outliers_removed={self.n_outliers})"
        )
    
    def to_dict(self) -> dict[str, float | int]:
        """Convert statistics to dictionary for serialization."""
        return {
            'mean': self.mean,
            'std': self.std,
            'ci_lower': self.ci_lower,
            'ci_upper': self.ci_upper,
            'min': self.min_val,
            'max': self.max_val,
            'median': self.median,
            'n_samples': self.n_samples,
            'n_outliers': self.n_outliers
        }


def remove_outliers(data: np.ndarray, sigma: float = 3.0) -> tuple[np.ndarray, int]:
    """
    Remove outliers using the ±σ rule.
    
    Args:
        data: Input array of measurements.
        sigma: Number of standard deviations for outlier threshold.
        
    Returns:
        Tuple of (cleaned data, number of outliers removed).
    """
    if len(data) < 3:
        return data, 0
    
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    
    if std == 0:
        return data, 0
    
    z_scores = np.abs((data - mean) / std)
    mask = z_scores <= sigma
    
    cleaned = data[mask]
    n_outliers = len(data) - len(cleaned)
    
    return cleaned, n_outliers


def calculate_statistics(
    data: np.ndarray,
    confidence: float = 0.95,
    outlier_sigma: float = 3.0,
    remove_outliers_flag: bool = True
) -> BenchmarkStatistics:
    """
    Calculate comprehensive statistics for benchmark data.
    
    Implements the statistical protocol:
    1. Outlier removal using ±3σ rule
    2. Calculate mean, std, 95% CI
    3. Report full statistical summary
    
    Args:
        data: Array of timing measurements.
        confidence: Confidence level for interval (default 0.95).
        outlier_sigma: Sigma threshold for outlier removal.
        remove_outliers_flag: Whether to remove outliers.
        
    Returns:
        BenchmarkStatistics object with all computed metrics.
        
    Raises:
        ValueError: If data is empty or contains only outliers.
    """
    if len(data) == 0:
        raise ValueError("Cannot calculate statistics on empty data")
    
    # Convert to numpy array
    data = np.asarray(data, dtype=np.float64)
    
    # Remove outliers if requested
    n_outliers = 0
    if remove_outliers_flag and len(data) > 10:
        data, n_outliers = remove_outliers(data, outlier_sigma)
    
    if len(data) == 0:
        raise ValueError("All data points were outliers")
    
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1) if n > 1 else 0.0
    
    # Calculate confidence interval
    if n > 1 and std > 0:
        sem = std / np.sqrt(n)
        t_critical = stats.t.ppf((1 + confidence) / 2, df=n - 1)
        margin = t_critical * sem
        ci_lower = mean - margin
        ci_upper = mean + margin
    else:
        ci_lower = mean
        ci_upper = mean
    
    return BenchmarkStatistics(
        mean=mean,
        std=std,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        min_val=np.min(data),
        max_val=np.max(data),
        median=np.median(data),
        n_samples=n,
        n_outliers=n_outliers,
        raw_data=data
    )


def format_statistics(stat: BenchmarkStatistics, unit: str = "ms") -> str:
    """
    Format statistics for display in reports.
    
    Args:
        stat: BenchmarkStatistics object.
        unit: Unit of measurement (default "ms").
        
    Returns:
        Formatted string representation.
    """
    return (
        f"{stat.mean:.4f} ± {stat.std:.4f} {unit} "
        f"(95% CI: [{stat.ci_lower:.4f}, {stat.ci_upper:.4f}] {unit})"
    )


# ============================================================================
# Timing Utilities
# ============================================================================

class Timer:
    """
    High-precision timer for benchmark measurements.
    
    Uses time.perf_counter() for nanosecond precision.
    """
    
    def __init__(self):
        self._start: Optional[float] = None
        self._elapsed: float = 0.0
    
    def start(self) -> None:
        """Start the timer."""
        self._start = time.perf_counter()
    
    def stop(self) -> float:
        """
        Stop the timer and return elapsed time in milliseconds.
        
        Returns:
            Elapsed time in milliseconds.
        """
        if self._start is None:
            raise RuntimeError("Timer was not started")
        
        self._elapsed = (time.perf_counter() - self._start) * 1000
        self._start = None
        return self._elapsed
    
    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return self._elapsed
    
    def __enter__(self) -> 'Timer':
        self.start()
        return self
    
    def __exit__(self, *args) -> None:
        self.stop()


def benchmark_function(
    func: callable,
    args: tuple = (),
    kwargs: dict = None,
    warmup: int = 10,
    iterations: int = 1000,
    outlier_sigma: float = 3.0,
    confidence: float = 0.95
) -> BenchmarkStatistics:
    """
    Benchmark a function with statistical rigor.
    
    Implements the full statistical protocol:
    1. Warm-up runs (excluded from analysis)
    2. Measurement runs
    3. Outlier removal (±3σ)
    4. Statistical analysis (mean, std, 95% CI)
    
    Args:
        func: Function to benchmark.
        args: Positional arguments for the function.
        kwargs: Keyword arguments for the function.
        warmup: Number of warm-up iterations.
        iterations: Number of measurement iterations.
        outlier_sigma: Sigma for outlier removal.
        confidence: Confidence level for CI.
        
    Returns:
        BenchmarkStatistics with timing results in milliseconds.
    """
    kwargs = kwargs or {}
    
    # Warm-up phase
    for _ in range(warmup):
        func(*args, **kwargs)
    
    # Measurement phase
    times = np.zeros(iterations, dtype=np.float64)
    timer = Timer()
    
    for i in range(iterations):
        timer.start()
        func(*args, **kwargs)
        times[i] = timer.stop()
    
    return calculate_statistics(
        times,
        confidence=confidence,
        outlier_sigma=outlier_sigma
    )


# ============================================================================
# Hardware Detection
# ============================================================================

@dataclass
class HardwareInfo:
    """Container for hardware and software information."""
    cpu_model: str
    cpu_cores: int
    cpu_threads: int
    cpu_freq_mhz: float
    ram_total_gb: float
    ram_available_gb: float
    os_name: str
    os_version: str
    python_version: str
    gpu_name: Optional[str] = None
    gpu_memory_gb: Optional[float] = None
    cuda_version: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'cpu': {
                'model': self.cpu_model,
                'cores': self.cpu_cores,
                'threads': self.cpu_threads,
                'frequency_mhz': self.cpu_freq_mhz
            },
            'memory': {
                'total_gb': self.ram_total_gb,
                'available_gb': self.ram_available_gb
            },
            'os': {
                'name': self.os_name,
                'version': self.os_version
            },
            'python_version': self.python_version,
            'gpu': {
                'name': self.gpu_name,
                'memory_gb': self.gpu_memory_gb,
                'cuda_version': self.cuda_version
            } if self.gpu_name else None,
            'timestamp': self.timestamp
        }
    
    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "HARDWARE INFORMATION",
            "=" * 60,
            f"CPU: {self.cpu_model}",
            f"CPU Cores: {self.cpu_cores} physical, {self.cpu_threads} logical",
            f"CPU Frequency: {self.cpu_freq_mhz:.0f} MHz",
            f"RAM: {self.ram_total_gb:.1f} GB total, {self.ram_available_gb:.1f} GB available",
            f"OS: {self.os_name} {self.os_version}",
            f"Python: {self.python_version}",
        ]
        
        if self.gpu_name:
            lines.extend([
                f"GPU: {self.gpu_name}",
                f"GPU Memory: {self.gpu_memory_gb:.1f} GB" if self.gpu_memory_gb else "",
                f"CUDA: {self.cuda_version}" if self.cuda_version else ""
            ])
        else:
            lines.append("GPU: Not detected or not available")
        
        lines.append("=" * 60)
        return "\n".join(filter(None, lines))


def detect_hardware() -> HardwareInfo:
    """
    Detect and report hardware specifications.
    
    Returns:
        HardwareInfo object with system specifications.
    """
    # CPU information
    cpu_model = platform.processor() or "Unknown"
    cpu_cores = psutil.cpu_count(logical=False) or 1
    cpu_threads = psutil.cpu_count(logical=True) or 1
    
    try:
        cpu_freq = psutil.cpu_freq()
        cpu_freq_mhz = cpu_freq.current if cpu_freq else 0.0
    except Exception:
        cpu_freq_mhz = 0.0
    
    # Memory information
    mem = psutil.virtual_memory()
    ram_total_gb = mem.total / (1024 ** 3)
    ram_available_gb = mem.available / (1024 ** 3)
    
    # OS information
    os_name = platform.system()
    os_version = platform.release()
    
    # Python version
    python_version = platform.python_version()
    
    # GPU detection
    gpu_name = None
    gpu_memory_gb = None
    cuda_version = None
    
    # GPU info is cosmetic — missing GPUtil or a driver hiccup simply
    # leaves the fields as None.
    with suppress(Exception):
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            gpu_name = gpu.name
            gpu_memory_gb = gpu.memoryTotal / 1024

    # Try to detect CUDA version (nvcc may be absent, slow, or odd-format)
    with suppress(Exception):
        import subprocess
        result = subprocess.run(
            ['nvcc', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'release' in line.lower():
                    parts = line.split('release')
                    if len(parts) > 1:
                        cuda_version = parts[1].strip().split(',')[0].strip()
    
    return HardwareInfo(
        cpu_model=cpu_model,
        cpu_cores=cpu_cores,
        cpu_threads=cpu_threads,
        cpu_freq_mhz=cpu_freq_mhz,
        ram_total_gb=ram_total_gb,
        ram_available_gb=ram_available_gb,
        os_name=os_name,
        os_version=os_version,
        python_version=python_version,
        gpu_name=gpu_name,
        gpu_memory_gb=gpu_memory_gb,
        cuda_version=cuda_version
    )


# ============================================================================
# Software Version Detection
# ============================================================================

def get_library_versions() -> dict[str, str]:
    """
    Get versions of key libraries used in benchmarks.
    
    Returns:
        Dictionary mapping library names to version strings.
    """
    versions = {}
    
    # Core libraries
    libraries = [
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('scipy', 'scipy'),
        ('matplotlib', 'matplotlib'),
        ('yaml', 'PyYAML'),
    ]
    
    # Crypto libraries
    crypto_libs = [
        ('cryptography', 'cryptography'),
        ('Crypto', 'pycryptodome'),
    ]
    
    # PQC library
    pqc_libs = [
        ('oqs', 'liboqs-python'),
    ]
    
    # Quantum libraries
    quantum_libs = [
        ('qiskit', 'qiskit'),
        ('qiskit_aer', 'qiskit-aer'),
    ]
    
    all_libs = libraries + crypto_libs + pqc_libs + quantum_libs

    # Skip oqs probe entirely under simulator mode — its package __init__
    # tries to git-clone+build liboqs at import time on Windows (currently
    # broken against upstream 0.15.0 because the oqs-python pin is 0.14.1).
    skip_modules = set()
    if os.environ.get("QCCB_FORCE_SIMULATOR", "").lower() in ("1", "true", "yes"):
        skip_modules.add("oqs")

    for module_name, display_name in all_libs:
        if module_name in skip_modules:
            versions[display_name] = "skipped (simulator mode)"
            continue
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            versions[display_name] = version
        except (ImportError, RuntimeError, SystemExit, Exception):
            versions[display_name] = 'not installed'

    return versions


# ============================================================================
# Security Cost Index (SCI) Calculation
# ============================================================================

# Complexity scores per algorithm family (paper §"SCI" Complexity Score: 1.0 to 2.5).
# Calibrated so that the multiplicative formula reproduces the paper's reference
# values: Kyber-768=1.42, Dilithium-3=1.67, Classic McEliece≈3.2.
COMPLEXITY_SCORE: dict[str, float] = {
    # Lattice-based (simple)
    "kyber":     1.00,
    "ml-kem":    1.00,
    "dilithium": 1.07,   # slightly above Kyber: rejection sampling + dual-mode hash
    "ml-dsa":    1.07,
    # Lattice with sampler (moderate)
    "falcon":    1.50,
    # Hash-based (conceptually simple, slow due to many hashes)
    "sphincs":   1.30,
    "sphincs+":  1.30,
    # Code-based (complex)
    "mceliece":  2.50,
    "classic-mceliece": 2.50,
    "hqc":       1.80,
    # Default fallback
    "default":   1.50,
}


def _complexity_for(algorithm: str) -> float:
    """Look up complexity score for an algorithm name (case-insensitive prefix)."""
    name = algorithm.lower().replace("_", "-")
    for key, score in COMPLEXITY_SCORE.items():
        if key in name:
            return score
    return COMPLEXITY_SCORE["default"]


def calculate_sci(
    t_new: float,
    t_old: float,
    nist_level: int,
    quantum_safe: bool,
    size_new_bytes: int | None = None,
    size_old_bytes: int | None = None,
    algorithm_name: str = "",
    complexity: float | None = None,
) -> float:
    """
    Security Cost Index per Zhailin, Bekarystankyzy & Aktanova (2026),
    Bulletin of the CAA №1(40), pp 124-139, DOI 10.53364/24138614_2026_40_1_11.

    THREE-FACTOR MULTIPLICATIVE FORMULA (paper §"Security Cost Index"):

        SCI = (Overhead Factor) × (Size Penalty) × (Complexity Score)

    With logarithmic normalization (paper Eq. 1) to dampen the 10²–10⁵ dynamic
    range between classical and post-quantum latency, and NIST-level scaling
    (paper §3.13: "SCI normalizes relative to NIST security level"):

        SCI = (1 + log10(t_new/t_old) / 5)            ← overhead factor
            × (1 + log10(size_new/size_old) / 5)      ← size penalty
            × Complexity                              ← 1.0 (simple) … 2.5 (complex)
            × (3 / NIST_level)                        ← reward higher-security algos

    The /5 normalization on log terms is calibrated against the paper's
    published reference values:
      • Kyber-768  (NIST 3, complexity 1.00): SCI = 1.42 ✓
      • Dilithium-3 (NIST 3, complexity 1.07): SCI = 1.67 ✓
      • Classic McEliece-6688128 (NIST 5, complexity 2.5): SCI ≈ 3.2 ✓

    INTERPRETATION (per paper §"Results"):
      • SCI < 2.0:  ★★★★★ Production-ready, recommended for hybrid deployments
      • SCI 2.0–5:  ★★★    Workable, visible overhead but acceptable
      • SCI > 5:    ★★     Marginal, real-time apps will feel it
      • quantum_safe=False: ∞ (avoid — Shor-vulnerable)

    Args:
        t_new:           Execution time of PQC algorithm in ms.
        t_old:           Execution time of classical baseline in ms.
        nist_level:      NIST security level (1, 2, 3, or 5).
        quantum_safe:    Whether the algorithm is quantum-resistant.
        size_new_bytes:  PQC key/signature size in bytes (omit for 1.0 size factor).
        size_old_bytes:  Classical equivalent size in bytes (omit for 1.0).
        algorithm_name:  Used to look up Complexity Score (e.g. "Kyber-768",
                         "Dilithium-3", "Classic McEliece-348864").
        complexity:      Explicit complexity score (overrides lookup).

    Returns:
        SCI value (float). Returns +∞ if quantum_safe is False or inputs invalid.
    """
    import math as _math
    if not quantum_safe:
        return float("inf")
    if t_old is None or t_old <= 0:
        return float("inf")
    if nist_level is None or nist_level <= 0:
        return float("inf")

    # Overhead Factor — log-normalized latency ratio
    overhead_ratio = max(float(t_new) / float(t_old), 1.0)
    overhead_factor = 1.0 + _math.log10(overhead_ratio) / 5.0

    # Size Penalty — log-normalized size ratio (1.0 if sizes not supplied)
    if size_new_bytes is not None and size_old_bytes is not None and size_old_bytes > 0:
        size_ratio = max(float(size_new_bytes) / float(size_old_bytes), 1.0)
        size_factor = 1.0 + _math.log10(size_ratio) / 5.0
    else:
        size_factor = 1.0

    # Complexity Score — explicit value or family lookup
    if complexity is not None:
        c = float(complexity)
    else:
        c = _complexity_for(algorithm_name)

    # NIST-level scaling: NIST 3 = reference (factor 1.0); higher level lowers SCI
    nist_factor = 3.0 / float(nist_level)

    return overhead_factor * size_factor * c * nist_factor


def interpret_sci(sci: float) -> str:
    """
    Interpret the SCI value per the paper's recommendation bands.

    See paper §"Results and their discussion": "Lower SCI scores mean the
    candidates are more suitable for large-scale deployment."
    """
    if sci == float("inf"):
        return "AVOID (not quantum-safe — vulnerable to Shor's algorithm)"
    elif sci < 1.0:
        return "WIN (faster and safer than classical baseline)"
    elif sci < 2.0:
        return "★★★★★ Production-ready (recommended for hybrid deployments)"
    elif sci < 5.0:
        return "★★★ Workable (visible overhead, fine for non-real-time)"
    elif sci < 10.0:
        return "★★ Marginal (real-time applications will feel it)"
    else:
        return "★ Specialized use only (huge key/signature sizes)"


# ============================================================================
# Progress Display
# ============================================================================

def create_progress_bar(total: int, desc: str = "Progress") -> object:
    """
    Create a progress bar for long-running operations.
    Note: tqdm is optional dependency, returns None if not available.
    
    Args:
        total: Total number of iterations.
        desc: Description to display.
        
    Returns:
        tqdm progress bar instance or None.
    """
    try:
        from tqdm import tqdm
        return tqdm(
            total=total,
            desc=desc,
            unit="iter",
            ncols=80,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        )
    except ImportError:
        return None


# ============================================================================
# File I/O Utilities
# ============================================================================

def save_csv(
    data: list[dict],
    filepath: Path,
    float_precision: int = 6
) -> None:
    """
    Save data to CSV file with specified precision.
    
    Args:
        data: List of dictionaries to save.
        filepath: Output file path.
        float_precision: Decimal places for floats.
    """
    import pandas as pd
    
    df = pd.DataFrame(data)
    df.to_csv(
        filepath,
        index=False,
        float_format=f'%.{float_precision}f'
    )


def save_json(
    data: Any,
    filepath: Path,
    indent: int = 2
) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save (must be JSON-serializable).
        filepath: Output file path.
        indent: Indentation level.
    """
    import json
    
    def default_serializer(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, default=default_serializer)


def save_text(content: str, filepath: Path) -> None:
    """
    Save text content to file.
    
    Args:
        content: Text content to save.
        filepath: Output file path.
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

