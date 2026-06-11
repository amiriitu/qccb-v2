# QCCB v2.0 — Quantum Computing Cryptography Benchmark

**Master's thesis tool: Data Protection in Quantum Computing Context — Cryptography Resilience Study**

A reproducible measurement framework that evaluates post-quantum cryptography
performance, runs canonical quantum algorithms on a real NISQ chip (Bauman
Octillion Snowdrop 4q ver2), and quantifies the boundary between current
quantum hardware and the classical-simulation ceiling.

## Project layout

```
QCCB_v2/
├── README.md                   # this file
├── run_gui.py                  # entry point for the Tk GUI
├── main.py                     # legacy entry point (deprecated, kept for reference)
├── config.yaml                 # statistical/algorithm parameters
├── token.env                   # Bauman Octillion API token (gitignored)
├── requirements.txt
├── assets/
│   ├── QBBC logo.svg           # project logo (vector source)
│   ├── qccb_icon.ico           # window/taskbar icon (16–256 px)
│   └── qccb_icon_256.png       # icon raster used by Tk windows
├── Bauman/
│   ├── snowdrop_4q_ver2.json   # live chip calibration snapshot
│   └── runs/                   # per-job raw counts archive (gitignored)
├── src/
│   ├── quantum_hardware.py     # Bauman Octillion API client
│   ├── gpu_simulator.py        # CuPy state-vector simulator
│   ├── experiments/            # 4 quantum experiments
│   │   ├── runner.py           # generic runner with timing + benchmark()
│   │   ├── bell_state.py
│   │   ├── ghz_state.py
│   │   ├── bernstein_vazirani.py
│   │   └── shor_n15.py
│   ├── pqc_benchmark.py        # PQC algorithms (Kyber/Dilithium/Falcon/SPHINCS+/HQC/McEliece)
│   ├── pqc_simulator.py        # PQC simulator (used when liboqs unavailable)
│   ├── pqc_avx2_comparison.py  # reference-C vs AVX2 timing comparison
│   ├── comparative_analysis.py # classical baselines + hybrid + classical SCI
│   ├── tls_hybrid_handshake.py # hybrid RSA+ML-KEM TLS 1.3 handshake model
│   ├── statistical_tests.py    # Kruskal-Wallis, Mann-Whitney+Bonferroni, BCa bootstrap
│   ├── quantum_threat.py       # Shor/Grover threat matrix
│   ├── quantum_resource_estimator.py # published Shor/Grover resource estimates
│   ├── hndl_calculator.py      # Harvest-Now-Decrypt-Later risk model
│   ├── local_chip_emulator.py  # offline Snowdrop noise emulators (4q v1/v2, 8q v1)
│   ├── snowdrop_constraints.py # chip topology + operating envelope
│   ├── cpu_features.py         # AVX2/AES-NI/SHA-NI detection
│   ├── visualization.py        # PQC + threat charts
│   ├── report_generator.py     # PQC scientific report (.txt)
│   ├── report_v2.py            # quantum-hardware scientific report (.md)
│   ├── full_benchmark.py       # quantum-hardware orchestrator
│   ├── pqc_pipeline.py         # PQC orchestrator
│   ├── scaling_demo.py         # GHZ-N scaling, CPU vs GPU
│   ├── sci_hardware.py         # SCI_HW dataclass
│   ├── sci_dynamic.py          # dynamic SCI(h,n,p) + Sobol sensitivity
│   ├── sci_calculator.py       # interactive SCI calculator (CLI)
│   ├── sci_calculator_gui.py   # Tk dialog version of the calculator
│   ├── crm_metric.py           # CRM definition + per-chip computation
│   ├── crm_forecast.py         # CRM table for 11 chips + forecast chart
│   ├── gui.py                  # main Tk GUI
│   ├── ui_theme.py             # shared palette, ttk styles, app icon
│   ├── csv_preview.py          # live CSV → chart preview engine for the GUI
│   └── utils.py                # statistics, hardware detect, I/O helpers
├── tools/
│   └── replot_charts.py        # re-render charts from saved results, no re-run
└── results/
    ├── 1_pqc_benchmarks/       # PQC algos vs RSA/ECC + threat matrix + roadmap
    ├── 2_quantum_hardware/     # Snowdrop chip experiments + GPU + CRM
    └── _archive_initial_jan_2026/  # original main.py output, kept as historical
```

## What this measures

### `results/1_pqc_benchmarks/` — software-side PQC analysis
| File | Content |
|------|---------|
| `pqc_benchmarks.csv/.json` | Kyber-{512,768,1024}, Dilithium-{2,3,5} timing with 95% CI |
| `comparative_analysis.csv` | Classical baselines (RSA-2048/4096, ECC P-256/P-384, AES, SHA) and hybrid RSA+Kyber |
| `quantum_threat_analysis.csv` | Vulnerability matrix (algorithm → quantum attack → status) |
| `factorization_results.json` | Shor simulation results for small N |
| `sci_analysis.csv` | Original 3-factor SCI (Scientific journal 'Bulletin of the CAA' 2026) |
| `pqc_performance_comparison.png` | 4-panel chart: KeyGen / Encaps / Decaps / Sizes |
| `quantum_threat_timeline.png` | Vulnerability windows per algorithm |
| `migration_roadmap.png` | 5-phase 2024–2035 plan with budget allocation |
| `migration_strategy.txt` | Detailed phase descriptions |
| `report.txt` | Full scientific report (~5–7 pages) |
| `benchmark.log` | Timestamped execution log |

### `results/2_quantum_hardware/` — real chip + GPU + CRM
| File | Content |
|------|---------|
| `quantum_hardware_runs.json` | Raw counts and timing per job_id (source of truth) |
| `quantum_hardware_summary.csv` | mean ± stdev, 95% CI, lag-1 autocorrelation |
| `quantum_hardware_timing.csv` | transpile / submit / queue / execute breakdown |
| `quantum_hardware_chip.csv` | Per-qubit T1, T2, F_1q, F_RO, frequency |
| `quantum_hardware_chip_2q.csv` | Per-pair F_cz |
| `quantum_sci_hw.csv` | SCI_HW per (experiment, backend) |
| `fidelity_comparison.png` | 4 experiments × 4 backends with error bars |
| `fidelity_timeline.png` | Per-run fidelity trajectories |
| `counts_overlay.png` | Observed vs expected probability distributions |
| `chip_calibration.png` | Snowdrop topology with annotated calibration |
| `timing_breakdown.png` | Stacked-bar wall-clock breakdown |
| `ghz_scaling.csv/.json/.png` | GHZ-N scaling: CPU vs GPU vs Snowdrop ceiling |
| `crm_table.csv/.json` | CRM per chip (Snowdrop + 10 published) |
| `crm_forecast.png` | Historical CRM trajectory + RSA-2048 crossing |
| `crm_report.md` | CRM definition, positioning, full table, caveats |
| `report.md` | Quantum hardware scientific report (Markdown) |

## Original contributions

1. **CRM (Cryptographically Reachable Modulus)** — chip-specific
   cryptanalytic benchmark defined in `src/crm_metric.py`. For a chip with
   measured {F_1q, F_cz, F_RO}, returns the largest N for which compiled
   Shor for factoring N succeeds with probability ≥ τ. Fills the gap
   between Quantum Volume (method-agnostic) and NIST IR 8547 fault-tolerant
   estimates (long-term). See `results/2_quantum_hardware/crm_report.md`.

2. **SCI_HW** — operationalization of the dynamic SCI(h⃗, n⃗, p⃗) postulated
   in Zhailin et al. (Scientific journal 'Bulletin of the CAA' №1(40), 2026, DOI 10.53364/24138614_2026_40_1_11).
   Concrete computable form:
   `SCI_HW = (T_obs / T_ideal) × |M_obs − M_ideal| × (D_t / D_l)`
   See `src/sci_hardware.py` and `results/2_quantum_hardware/quantum_sci_hw.csv`.

3. **Six-backend unified framework** (`src/experiments/runner.py`) running
   the same circuits through ideal CPU sim, GPU sim, three chip-noise
   emulators (Snowdrop 4q v1, 4q v2, 8q v1 projection), and the real QPU —
   enabling attribution of fidelity loss to specific causes.

4. **First independent benchmarks of Bauman Octillion Snowdrop 4q ver2**
   on canonical algorithms with full reproducibility (job IDs in `Bauman/runs/`).

## Usage

### GUI

```
python run_gui.py
```

Buttons in the main panel:
- **Refresh from chip** — pull live calibration
- **▶ Run** — single experiment on the selected backend
- **▶▶ Run on ALL 3 backends** — ideal / emulator / real comparison
- **📊 Benchmark (N repeats)** — N-run statistical run with mean ± std + 95% CI
- **🔬 Full benchmark suite** — all experiments × selected backends + auto CRM
- **📈 GPU scaling demo** — GHZ-N for N=2..25 (CPU vs GPU vs chip ceiling)
- **🔐 CRM forecast** — chip → Shor capability + RSA-2048 forecast
- **🧮 SCI calculator** — interactive paper-formula and SCI_HW side by side
- **📦 PQC benchmark suite** — Kyber/Dilithium/RSA/ECC + threat matrix + roadmap

#### Live CSV previews

While any pipeline is running, the *Results* panel follows the CSV files it
writes and re-draws them as native charts right inside the window — no need
to open the generated PNGs to see what a run produced. Powered by
`src/csv_preview.py`, which has a dedicated renderer for every CSV family
the pipelines emit (PQC timings, fidelity-by-backend, GHZ-N scaling, CRM
table, SCI tables, chip calibration, …) and a generic chart/table fallback
for anything else.

- the **Live CSV preview** picker above the chart lists every CSV written
  since the GUI started; pick one to inspect it
- **Follow latest** keeps switching to the newest CSV as the run progresses
  (picking a file manually turns it off)
- **📂 Open results folder** opens the run's artifact directory on demand —
  pipelines no longer pop a file-explorer window over the GUI
- the pipeline-generated PNG charts are unchanged: they are still written
  to `results/` for the dissertation, and `tools/replot_charts.py` still
  re-styles them (see below)

### CLI

```bash
# PQC software benchmarks → results/1_pqc_benchmarks/
python -m src.pqc_pipeline                    # quick (100 iter)
python -m src.pqc_pipeline --full             # publication run (1000 iter)

# Quantum hardware experiments → results/2_quantum_hardware/
python -m src.full_benchmark --repeats 5
python -m src.full_benchmark --backends ideal gpu emulator --repeats 10
python -m src.full_benchmark --extras         # +GHZ-N sweep, +BV secrets
python -m src.full_benchmark --no-real        # skip real chip

# GPU scaling demo
python -m src.scaling_demo --n-min 2 --n-max 24 --shots 1024

# CRM forecast (uses saved calibration, fast)
python -m src.crm_forecast --use-saved-calibration

# SCI calculator (interactive CLI)
python -m src.sci_calculator
python -m src.sci_calculator --examples       # show pre-computed examples
python -m src.sci_calculator_gui              # standalone Tk window
```

### Re-styling charts from saved results

Every chart in a results directory can be regenerated from the saved
`quantum_hardware_runs.json` artifact — different layout, fonts, colors,
error bars — without re-submitting anything to the chip or re-running
the benchmark:

```bash
python tools/replot_charts.py "results/2_quantum_hardware/14052026 17 40"
python tools/replot_charts.py                 # picks the latest run automatically
```

This rewrites `fidelity_comparison.png`, `fidelity_timeline.png`,
`timing_breakdown.png`, `python_overhead.png`, `counts_overlay.png` and
`chip_calibration.png` in place with a publication-grade style:
per-experiment panel grids, 95% CI error bars, log-scale timing axes and
a colour-blind-safe palette. Useful when a figure needs a visual tweak
after an expensive benchmark has already been collected.

## Setup

### Required
- Python 3.13+
- Windows, Linux or macOS
- Bauman Octillion API token (only for `--backends real` runs)

### Install
```bash
python -m venv .venv
.venv/Scripts/activate      # Windows
# source .venv/bin/activate # Linux/macOS

pip install -r requirements.txt

# Octillion client — separate index
pip install octillion_client \
  -i http://public:public@projects.iu5.bmstu.ru:8081/repository/pip_all/simple \
  --trusted-host projects.iu5.bmstu.ru

# Optional: GPU acceleration (Windows + CUDA 12.x toolkit recommended)
pip install cupy-cuda12x \
            nvidia-cublas-cu12 \
            nvidia-cuda-runtime-cu12 \
            nvidia-cuda-nvrtc-cu12 \
            nvidia-nvjitlink-cu12
```

### Token setup

Create `token.env` in the project root:
```
BAUMAN_OCTILLION_TOKEN=your_token_here
```
Add `token.env` to `.gitignore` (already done). The token is read at runtime
and never logged.

## Reproducibility

- All raw counts and timings persist in `Bauman/runs/<job_id>.json`
  (per-job, with QASM, calibration snapshot, and per-shot results).
- `results/2_quantum_hardware/quantum_hardware_runs.json` contains the
  full benchmark dataset with per-run timings.
- Lag-1 autocorrelation of fidelity time-series is computed and reported;
  values close to zero validate the iid assumption underlying 95% CI.
- Transpilation uses `layout_method='trivial'` and `optimization_level=1`
  to keep logical→physical qubit mapping identity (essential because
  `qiskit.qasm2.dumps()` discards layout metadata when sent to the real chip).
- Published-chip parameters in `crm_table.csv` cite their primary sources.

## Citation

Primary publication for the SCI methodology:

> Zhailin A.G., Bekarystankyzy A., Aktanova B.M. (2026).
> "The Cryptographic Impact of Quantum Computing — Benchmarking of Classical
> and Post-Quantum Algorithms." *Scientific journal 'Bulletin of the CAA'* №1(40), pp. 124–139.
> DOI: [10.53364/24138614_2026_40_1_11](https://doi.org/10.53364/24138614_2026_40_1_11)

CRM and SCI_HW extensions: this work.

## License

MIT.
