# ============================================================================
# QCCB v2.0 - Visualization Module
# Publication-Quality Chart Generation
# ============================================================================
"""
Visualization module for generating publication-quality charts and plots.

Charts generated:
1. Quantum Threat Timeline - Algorithm vulnerability windows
2. PQC Performance Comparison - 4-panel performance metrics
3. Migration Roadmap - 5-phase investment chart

All charts are designed for direct inclusion in Master's thesis.

Author: Amir
Date: 2026
Thesis: Data Protection in Quantum Computing Context: Cryptography Resilience Study
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger('QCCB.visualization')


def setup_matplotlib():
    """Configure matplotlib for publication-quality output."""
    import matplotlib.pyplot as plt
    import matplotlib
    
    # Use a clean style
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        try:
            plt.style.use('seaborn-whitegrid')
        except OSError:
            plt.style.use('ggplot')
    
    # Configure for publication
    matplotlib.rcParams.update({
        'font.size': 10,
        'font.family': 'sans-serif',
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
    })
    
    return plt


# ============================================================================
# Color Palettes
# ============================================================================

COLORS = {
    'classical': '#E74C3C',      # Red - vulnerable
    'pqc': '#27AE60',            # Green - quantum-safe
    'hybrid': '#3498DB',         # Blue - transitional
    'warning': '#F39C12',        # Orange - caution
    'safe': '#2ECC71',           # Light green - safe
    'danger': '#C0392B',         # Dark red - critical
    'neutral': '#7F8C8D',        # Gray - neutral
    'lattice': '#9B59B6',        # Purple - lattice-based
    'hash': '#1ABC9C',           # Teal - hash-based
}


# ============================================================================
# Quantum Threat Timeline Chart
# ============================================================================

def create_quantum_threat_timeline(
    threat_results: list,
    config: dict[str, Any],
    output_path: Path
) -> None:
    """
    Create the Quantum Threat Timeline visualization.
    
    Shows vulnerability windows for different cryptographic algorithms
    with threat zones and current year marker.
    
    Args:
        threat_results: List of QuantumThreatResult objects.
        config: Configuration dictionary.
        output_path: Path to save the chart.
    """
    plt = setup_matplotlib()
    
    # Get timeline parameters
    threat_config = config.get('quantum_threat', {}).get('threat_timeline', {})
    current_year = threat_config.get('current_year', 2026)
    opt_year = threat_config.get('rsa_2048_break_year_optimistic', 2027)
    pess_year = threat_config.get('rsa_2048_break_year_pessimistic', 2039)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Define algorithms and their threat windows
    algorithms = [
        # Classical asymmetric (vulnerable)
        {'name': 'RSA-2048', 'start': opt_year, 'end': pess_year, 
         'color': COLORS['danger'], 'category': 'Classical Asymmetric'},
        {'name': 'RSA-4096', 'start': opt_year + 2, 'end': pess_year + 3, 
         'color': COLORS['danger'], 'category': 'Classical Asymmetric'},
        {'name': 'ECC P-256', 'start': opt_year, 'end': pess_year, 
         'color': COLORS['danger'], 'category': 'Classical Asymmetric'},
        {'name': 'ECC P-384', 'start': opt_year + 1, 'end': pess_year + 1, 
         'color': COLORS['danger'], 'category': 'Classical Asymmetric'},
        
        # Symmetric (safe)
        {'name': 'AES-128', 'start': 2050, 'end': 2060, 
         'color': COLORS['warning'], 'category': 'Symmetric'},
        {'name': 'AES-256', 'start': 2100, 'end': 2100, 
         'color': COLORS['safe'], 'category': 'Symmetric'},
        
        # Hash (safe)
        {'name': 'SHA-256', 'start': 2100, 'end': 2100, 
         'color': COLORS['safe'], 'category': 'Hash Functions'},
        {'name': 'SHA3-256', 'start': 2100, 'end': 2100, 
         'color': COLORS['safe'], 'category': 'Hash Functions'},
        
        # PQC (safe)
        {'name': 'ML-KEM (Kyber)', 'start': 2100, 'end': 2100, 
         'color': COLORS['pqc'], 'category': 'Post-Quantum'},
        {'name': 'ML-DSA (Dilithium)', 'start': 2100, 'end': 2100, 
         'color': COLORS['pqc'], 'category': 'Post-Quantum'},
        {'name': 'SLH-DSA (SPHINCS+)', 'start': 2100, 'end': 2100, 
         'color': COLORS['pqc'], 'category': 'Post-Quantum'},
        
        # Hybrid (safe)
        {'name': 'RSA+Kyber Hybrid', 'start': 2100, 'end': 2100, 
         'color': COLORS['hybrid'], 'category': 'Hybrid Schemes'},
    ]
    
    # Time range
    start_year = 2024
    end_year = 2045
    
    # Plot each algorithm
    y_positions = list(range(len(algorithms)))
    
    for i, algo in enumerate(algorithms):
        y = len(algorithms) - 1 - i
        
        # Draw timeline bar
        if algo['start'] < end_year:
            bar_start = max(algo['start'], start_year)
            bar_end = min(algo['end'], end_year)
            
            # Safe zone (before vulnerability)
            ax.barh(y, algo['start'] - start_year, left=start_year, 
                   height=0.6, color=COLORS['safe'], alpha=0.6)
            
            # Vulnerability window
            if bar_end > bar_start:
                ax.barh(y, bar_end - bar_start, left=bar_start,
                       height=0.6, color=algo['color'], alpha=0.8)
            
            # Add text annotation for vulnerability window
            if algo['start'] < end_year:
                ax.annotate(
                    f'{algo["start"]}-{algo["end"]}' if algo['end'] < 2100 else 'Quantum-Safe',
                    xy=(bar_start + 0.5, y),
                    fontsize=8,
                    va='center',
                    color='white' if algo['color'] in [COLORS['danger'], COLORS['pqc']] else 'black'
                )
        else:
            # Algorithm is safe - full green bar
            ax.barh(y, end_year - start_year, left=start_year,
                   height=0.6, color=COLORS['safe'], alpha=0.6)
            ax.annotate('Quantum-Safe', xy=(start_year + 1, y),
                       fontsize=8, va='center', color='darkgreen')
    
    # Current year marker
    ax.axvline(x=current_year, color='blue', linestyle='--', linewidth=2,
               label=f'Current Year ({current_year})')
    
    # HNDL threat zone
    ax.axvspan(current_year, opt_year, alpha=0.2, color='orange',
               label='HNDL Attack Window')
    
    # Optimistic threat zone
    ax.axvspan(opt_year, pess_year, alpha=0.15, color='red',
               label='Primary Threat Window')
    
    # Configure axes
    ax.set_xlim(start_year, end_year)
    ax.set_ylim(-0.5, len(algorithms) - 0.5)
    ax.set_yticks(list(range(len(algorithms))))
    ax.set_yticklabels([a['name'] for a in reversed(algorithms)])
    ax.set_xlabel('Year', fontsize=12)
    ax.set_title('Quantum Threat Timeline: Cryptographic Algorithm Vulnerability Windows\n'
                 '(Based on Gidney & Ekerå, 2021 - RSA-2048 breakage estimates)',
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS['safe'], alpha=0.6, label='Quantum-Safe'),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS['danger'], alpha=0.8, label='Vulnerable'),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS['warning'], alpha=0.6, label='Reduced Security'),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS['pqc'], alpha=0.8, label='Post-Quantum'),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS['hybrid'], alpha=0.8, label='Hybrid'),
        plt.Line2D([0], [0], color='blue', linestyle='--', linewidth=2, label=f'Current ({current_year})'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)
    
    # Add category labels on the right
    categories = {}
    for i, algo in enumerate(reversed(algorithms)):
        cat = algo['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(i)
    
    # Grid
    ax.grid(axis='x', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    logger.info(f"  Saved: {output_path}")


# ============================================================================
# PQC Performance Comparison Chart
# ============================================================================

def create_pqc_performance_chart(
    kem_results: list,
    sig_results: list,
    config: dict[str, Any],
    output_path: Path
) -> None:
    """
    Create 4-panel PQC performance comparison chart.
    
    Panels:
    1. Key Generation Time
    2. Operations Time (Encaps/Decaps, Sign/Verify)
    3. Key Sizes
    4. Ciphertext/Signature Sizes
    
    Args:
        kem_results: List of KEMBenchmarkResult objects.
        sig_results: List of SignatureBenchmarkResult objects.
        config: Configuration dictionary.
        output_path: Path to save the chart.
    """
    plt = setup_matplotlib()
    
    if not kem_results and not sig_results:
        logger.warning("No PQC results to visualize")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Prepare data
    kem_names = [r.algorithm for r in kem_results] if kem_results else []
    sig_names = [r.algorithm for r in sig_results] if sig_results else []
    all_names = kem_names + sig_names
    
    # Colors by family
    colors = []
    for r in kem_results:
        colors.append(COLORS['lattice'] if r.family == 'lattice' else COLORS['hash'])
    for r in sig_results:
        colors.append(COLORS['lattice'] if r.family == 'lattice' else COLORS['hash'])
    
    x = np.arange(len(all_names))
    
    # Panel 1: Key Generation Time
    ax_keygen = axes[0, 0]
    keygen_times = []
    keygen_errors = []
    
    for r in kem_results:
        keygen_times.append(r.keygen_stats.mean)
        keygen_errors.append(r.keygen_stats.std)
    for r in sig_results:
        keygen_times.append(r.keygen_stats.mean)
        keygen_errors.append(r.keygen_stats.std)
    
    keygen_bars = ax_keygen.bar(x, keygen_times, yerr=keygen_errors, capsize=3, 
                    color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax_keygen.set_ylabel('Time (ms)')
    ax_keygen.set_title('Key Generation Time', fontweight='bold')
    ax_keygen.set_xticks(x)
    ax_keygen.set_xticklabels(all_names, rotation=45, ha='right')
    ax_keygen.set_yscale('log')
    
    # Add value labels
    for bar, val in zip(keygen_bars, keygen_times):
        ax_keygen.annotate(f'{val:.2f}', 
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=8)
    
    # Panel 2: Operations Time
    ax_ops = axes[0, 1]
    width = 0.35
    
    op1_times = []  # encaps/sign
    op2_times = []  # decaps/verify
    op1_labels = []
    op2_labels = []
    
    for r in kem_results:
        op1_times.append(r.encaps_stats.mean)
        op2_times.append(r.decaps_stats.mean)
        op1_labels.append('Encaps')
        op2_labels.append('Decaps')
    for r in sig_results:
        op1_times.append(r.sign_stats.mean)
        op2_times.append(r.verify_stats.mean)
        op1_labels.append('Sign')
        op2_labels.append('Verify')
    
    x2 = np.arange(len(all_names))
    op1_bars = ax_ops.bar(x2 - width/2, op1_times, width, label='Encaps/Sign',
                     color=COLORS['pqc'], alpha=0.8, edgecolor='black', linewidth=0.5)
    op2_bars = ax_ops.bar(x2 + width/2, op2_times, width, label='Decaps/Verify',
                     color=COLORS['hybrid'], alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax_ops.set_ylabel('Time (ms)')
    ax_ops.set_title('Operation Time (Encaps/Sign vs Decaps/Verify)', fontweight='bold')
    ax_ops.set_xticks(x2)
    ax_ops.set_xticklabels(all_names, rotation=45, ha='right')
    ax_ops.legend(loc='upper right')
    ax_ops.set_yscale('log')
    
    # Panel 3: Key Sizes
    ax_keys = axes[1, 0]
    
    pk_sizes = []
    sk_sizes = []
    
    for r in kem_results:
        pk_sizes.append(r.public_key_size)
        sk_sizes.append(r.secret_key_size)
    for r in sig_results:
        pk_sizes.append(r.public_key_size)
        sk_sizes.append(r.secret_key_size)
    
    x3 = np.arange(len(all_names))
    pk_bars = ax_keys.bar(x3 - width/2, pk_sizes, width, label='Public Key',
                     color=COLORS['pqc'], alpha=0.8, edgecolor='black', linewidth=0.5)
    sk_bars = ax_keys.bar(x3 + width/2, sk_sizes, width, label='Secret Key',
                     color=COLORS['warning'], alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax_keys.set_ylabel('Size (bytes)')
    ax_keys.set_title('Key Sizes', fontweight='bold')
    ax_keys.set_xticks(x3)
    ax_keys.set_xticklabels(all_names, rotation=45, ha='right')
    ax_keys.legend(loc='upper right')
    ax_keys.set_yscale('log')
    
    # Panel 4: Ciphertext/Signature Sizes
    ax_output = axes[1, 1]
    
    output_sizes = []
    output_labels = []
    
    for r in kem_results:
        output_sizes.append(r.ciphertext_size)
        output_labels.append('Ciphertext')
    for r in sig_results:
        output_sizes.append(r.signature_size)
        output_labels.append('Signature')
    
    x4 = np.arange(len(all_names))
    bar_colors = [COLORS['hybrid'] if 'Ciphertext' in l else COLORS['lattice'] 
                  for l in output_labels]
    output_bars = ax_output.bar(x4, output_sizes, color=bar_colors, alpha=0.8,
                    edgecolor='black', linewidth=0.5)
    
    ax_output.set_ylabel('Size (bytes)')
    ax_output.set_title('Ciphertext/Signature Sizes', fontweight='bold')
    ax_output.set_xticks(x4)
    ax_output.set_xticklabels(all_names, rotation=45, ha='right')
    ax_output.set_yscale('log')
    
    # Add value labels
    for bar, val in zip(output_bars, output_sizes):
        ax_output.annotate(f'{val}', 
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=8)
    
    # Overall title
    fig.suptitle('Post-Quantum Cryptography Performance Comparison\n'
                 '(NIST ML-KEM and ML-DSA Standards)',
                 fontsize=14, fontweight='bold', y=1.02)
    
    # Legend for algorithm families
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS['lattice'], alpha=0.8, label='Lattice-based'),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS['hash'], alpha=0.8, label='Hash-based'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, 
               bbox_to_anchor=(0.5, -0.02), framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    logger.info(f"  Saved: {output_path}")


# ============================================================================
# Migration Roadmap Chart
# ============================================================================

def create_migration_roadmap(
    config: dict[str, Any],
    output_path: Path
) -> None:
    """
    Create 5-phase migration roadmap visualization.
    
    Shows investment percentages and timeline for PQC migration.
    
    Args:
        config: Configuration dictionary.
        output_path: Path to save the chart.
    """
    plt = setup_matplotlib()
    
    # Get migration phases from config
    phases = config.get('migration', {}).get('phases', [])
    
    if not phases:
        # Default phases
        phases = [
            {'name': 'Assessment', 'start_year': 2024, 'end_year': 2025, 
             'budget_percentage': '5-10%', 'description': 'Cryptographic inventory'},
            {'name': 'Hybrid Deployment', 'start_year': 2025, 'end_year': 2028,
             'budget_percentage': '15-20%', 'description': 'Deploy hybrid solutions'},
            {'name': 'Transition', 'start_year': 2028, 'end_year': 2030,
             'budget_percentage': '20-25%', 'description': 'Migrate critical systems'},
            {'name': 'Full PQC', 'start_year': 2030, 'end_year': 2035,
             'budget_percentage': '10-15%', 'description': 'Complete migration'},
            {'name': 'Quantum-Safe Era', 'start_year': 2035, 'end_year': 2040,
             'budget_percentage': '5-10%', 'description': 'Maintenance'},
        ]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Colors for phases
    phase_colors = [
        '#3498DB',  # Blue - Assessment
        '#F39C12',  # Orange - Hybrid
        '#E74C3C',  # Red - Transition (peak)
        '#27AE60',  # Green - Full PQC
        '#2ECC71',  # Light green - Safe
    ]
    
    # Parse budget percentages (take max value)
    budgets = []
    for phase in phases:
        budget_str = phase.get('budget_percentage', '10%')
        # Extract numbers from string like "15-20%"
        numbers = [int(x) for x in budget_str.replace('%', '').split('-')]
        budgets.append(max(numbers))
    
    # Create timeline bars
    y_base = 0
    bar_height = 0.6
    
    for i, phase in enumerate(phases):
        start = phase['start_year']
        end = phase['end_year']
        duration = end - start
        
        # Main bar
        ax.barh(i, duration, left=start, height=bar_height,
                color=phase_colors[i % len(phase_colors)], alpha=0.8,
                edgecolor='black', linewidth=1)
        
        # Phase name and duration
        ax.text(start + duration/2, i, f"{phase['name']}\n({start}-{end})",
                ha='center', va='center', fontweight='bold', fontsize=10,
                color='white')
    
    # Add budget overlay as connected line
    ax2 = ax.twinx()
    
    # Calculate x positions (midpoint of each phase)
    x_positions = [(phase['start_year'] + phase['end_year']) / 2 for phase in phases]
    
    ax2.plot(x_positions, budgets, 'ko-', linewidth=2, markersize=10,
             label='IT Budget Allocation')
    ax2.fill_between(x_positions, budgets, alpha=0.3, color='gray')
    
    # Add budget labels
    for x, budget in zip(x_positions, budgets):
        ax2.annotate(f'{budget}%', xy=(x, budget), xytext=(0, 10),
                    textcoords='offset points', ha='center', fontsize=10,
                    fontweight='bold')
    
    # Configure axes
    ax.set_xlim(2023, 2041)
    ax.set_ylim(-0.5, len(phases) - 0.5)
    ax.set_yticks(range(len(phases)))
    ax.set_yticklabels([f"Phase {i+1}" for i in range(len(phases))])
    ax.set_xlabel('Year', fontsize=12)
    ax.set_title('Post-Quantum Cryptography Migration Roadmap (2024-2035+)\n'
                 'Recommended IT Budget Allocation by Phase',
                 fontsize=14, fontweight='bold', pad=20)
    
    ax2.set_ylabel('IT Budget Allocation (%)', fontsize=12)
    ax2.set_ylim(0, 35)
    
    # Add legend
    legend_elements = []
    for i, phase in enumerate(phases):
        legend_elements.append(
            plt.Rectangle((0, 0), 1, 1, facecolor=phase_colors[i % len(phase_colors)],
                          alpha=0.8, label=phase['name'])
        )
    
    ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9)
    
    # Add current year marker
    current_year = config.get('quantum_threat', {}).get('threat_timeline', {}).get('current_year', 2026)
    ax.axvline(x=current_year, color='red', linestyle='--', linewidth=2)
    ax.text(current_year, len(phases) - 0.3, f'Current\n({current_year})',
            ha='center', fontsize=9, color='red')
    
    # Add threat deadline marker
    threat_year = 2030
    ax.axvline(x=threat_year, color='darkred', linestyle=':', linewidth=2)
    ax.text(threat_year, -0.4, f'Recommended\nCompletion', ha='center',
            fontsize=9, color='darkred')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    logger.info(f"  Saved: {output_path}")


# ============================================================================
# Main Visualization Class
# ============================================================================

class Visualizer:
    """
    Main class for generating all visualizations.
    """
    
    def __init__(self, config: dict[str, Any]):
        """
        Initialize the visualizer.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.vis_config = config.get('visualization', {})
    
    def generate_all(
        self,
        output_dir: Path,
        threat_results: list = None,
        kem_results: list = None,
        sig_results: list = None
    ) -> None:
        """
        Generate all visualizations.
        
        Args:
            output_dir: Directory for output files.
            threat_results: Quantum threat analysis results.
            kem_results: PQC KEM benchmark results.
            sig_results: PQC signature benchmark results.
        """
        logger.info("\n" + "=" * 60)
        logger.info("GENERATING VISUALIZATIONS")
        logger.info("=" * 60)
        
        # Quantum Threat Timeline
        create_quantum_threat_timeline(
            threat_results or [],
            self.config,
            output_dir / 'quantum_threat_timeline.png'
        )
        
        # PQC Performance Comparison
        if kem_results or sig_results:
            create_pqc_performance_chart(
                kem_results or [],
                sig_results or [],
                self.config,
                output_dir / 'pqc_performance_comparison.png'
            )
        
        # Migration Roadmap
        create_migration_roadmap(
            self.config,
            output_dir / 'migration_roadmap.png'
        )
        
        logger.info("All visualizations generated successfully")

