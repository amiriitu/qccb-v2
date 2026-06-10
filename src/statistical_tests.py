"""
Non-parametric statistical tests for QCCB v2 benchmark comparisons.

The published QCCB protocol (Scientific journal 'Bulletin of the CAA' 2026)
declares the following methodology for cross-backend comparisons:

  - Kruskal-Wallis H-test       — global N-sample location test
  - Mann-Whitney U with Bonferroni — pairwise post-hoc
  - BCa bootstrap               — bias-corrected accelerated 95% CI

This module implements all three on top of scipy.stats. Used by full_benchmark
to attach a `statistical_tests.csv` artifact to every benchmark folder, so a
thesis defender can answer the committee's "are these means actually different?"
question with a non-parametric test rather than just µ ± σ.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Sequence

import numpy as np
from scipy import stats


# ============================================================================
# Result containers
# ============================================================================

@dataclass(frozen=True)
class KruskalWallisResult:
    """Outcome of a Kruskal-Wallis test across N >= 2 groups."""
    h_statistic: float
    p_value: float
    n_groups: int
    n_total: int
    significant_at_05: bool
    interpretation: str

    def to_dict(self) -> dict:
        return {
            "test": "Kruskal-Wallis H-test",
            "H": self.h_statistic,
            "p_value": self.p_value,
            "n_groups": self.n_groups,
            "n_total": self.n_total,
            "significant": self.significant_at_05,
            "interpretation": self.interpretation,
        }


@dataclass(frozen=True)
class PairwiseMWResult:
    """One pair from the Mann-Whitney post-hoc, after Bonferroni correction."""
    group_a: str
    group_b: str
    u_statistic: float
    p_raw: float
    p_bonferroni: float
    significant_at_05: bool

    def to_dict(self) -> dict:
        return {
            "test": "Mann-Whitney U + Bonferroni",
            "group_a": self.group_a,
            "group_b": self.group_b,
            "U": self.u_statistic,
            "p_raw": self.p_raw,
            "p_bonferroni": self.p_bonferroni,
            "significant": self.significant_at_05,
        }


@dataclass(frozen=True)
class BCaBootstrapResult:
    """Bias-corrected accelerated bootstrap CI for a sample statistic."""
    point_estimate: float
    ci_low: float
    ci_high: float
    confidence_level: float
    n_bootstrap: int
    bias: float
    acceleration: float

    def to_dict(self) -> dict:
        return {
            "test": "BCa bootstrap",
            "point_estimate": self.point_estimate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "confidence_level": self.confidence_level,
            "n_bootstrap": self.n_bootstrap,
            "bias": self.bias,
            "acceleration": self.acceleration,
        }


# ============================================================================
# Tests
# ============================================================================

def kruskal_wallis(groups: dict[str, Sequence[float]]) -> KruskalWallisResult:
    """
    H = global non-parametric ANOVA equivalent. Tests H0: all groups have the
    same distribution. Robust to non-normality and unequal group sizes.

    Requires at least two groups, each with at least one sample.
    """
    arrays = [np.asarray(v, dtype=float) for v in groups.values() if len(v) > 0]
    if len(arrays) < 2:
        raise ValueError(f"Kruskal-Wallis requires ≥ 2 non-empty groups, got "
                         f"{len(arrays)}")
    h, p = stats.kruskal(*arrays)
    n_total = sum(a.size for a in arrays)
    if p < 0.001:
        interp = "groups differ very strongly (p < 0.001)"
    elif p < 0.01:
        interp = "groups differ strongly (p < 0.01)"
    elif p < 0.05:
        interp = "groups differ (p < 0.05)"
    else:
        interp = "no significant difference between groups (p ≥ 0.05)"
    return KruskalWallisResult(
        h_statistic=float(h), p_value=float(p),
        n_groups=len(arrays), n_total=int(n_total),
        significant_at_05=bool(p < 0.05),
        interpretation=interp,
    )


def mann_whitney_pairwise(groups: dict[str, Sequence[float]],
                            alpha: float = 0.05
                            ) -> list[PairwiseMWResult]:
    """
    Mann-Whitney U test for every pair of groups, with Bonferroni correction
    over the (n choose 2) comparisons. Significance reported at the corrected
    threshold so type-I error is controlled family-wise at `alpha`.
    """
    names = [k for k, v in groups.items() if len(v) > 0]
    pairs = list(combinations(names, 2))
    if not pairs:
        return []
    results: list[PairwiseMWResult] = []
    for a, b in pairs:
        arr_a = np.asarray(groups[a], dtype=float)
        arr_b = np.asarray(groups[b], dtype=float)
        try:
            u, p_raw = stats.mannwhitneyu(arr_a, arr_b, alternative="two-sided")
        except ValueError:
            # all-tied or all-equal samples — degenerate
            u, p_raw = float("nan"), 1.0
        p_bonf = min(1.0, float(p_raw) * len(pairs))
        results.append(PairwiseMWResult(
            group_a=a, group_b=b,
            u_statistic=float(u), p_raw=float(p_raw),
            p_bonferroni=p_bonf,
            significant_at_05=bool(p_bonf < alpha),
        ))
    return results


def bca_bootstrap(samples: Sequence[float],
                    statistic=None,
                    n_bootstrap: int = 10_000,
                    confidence_level: float = 0.95,
                    rng_seed: int | None = 42,
                    ) -> BCaBootstrapResult:
    """
    BCa (Bias-Corrected accelerated) bootstrap confidence interval for an
    arbitrary statistic — defaults to the mean.

    BCa adjusts standard percentile bootstrap CIs for both bias and skewness
    (acceleration), giving the standard non-parametric CI used in modern
    statistical practice. References: Efron-Tibshirani 1993, §14.3.
    """
    if statistic is None:
        statistic = np.mean
    arr = np.asarray(samples, dtype=float)
    n = len(arr)
    if n < 2:
        raise ValueError(f"BCa bootstrap needs ≥ 2 samples, got {n}")
    rng = np.random.default_rng(rng_seed)

    # Bootstrap distribution of the statistic
    boot_idx = rng.integers(0, n, size=(n_bootstrap, n))
    boot_stats = np.array([statistic(arr[idx]) for idx in boot_idx])

    theta_hat = float(statistic(arr))

    # Bias correction factor z0
    fraction_below = np.mean(boot_stats < theta_hat)
    fraction_below = min(max(fraction_below, 1.0 / (2 * n_bootstrap)),
                          1 - 1.0 / (2 * n_bootstrap))
    z0 = float(stats.norm.ppf(fraction_below))

    # Acceleration via jackknife
    jack = np.empty(n)
    for i in range(n):
        jack[i] = statistic(np.delete(arr, i))
    jack_mean = jack.mean()
    num = ((jack_mean - jack) ** 3).sum()
    den = 6.0 * ((jack_mean - jack) ** 2).sum() ** 1.5
    a = float(num / den) if den > 0 else 0.0

    # BCa-adjusted percentiles
    alpha = 1 - confidence_level
    z_lo, z_hi = stats.norm.ppf(alpha / 2), stats.norm.ppf(1 - alpha / 2)
    p_lo = stats.norm.cdf(z0 + (z0 + z_lo) / max(1 - a * (z0 + z_lo), 1e-9))
    p_hi = stats.norm.cdf(z0 + (z0 + z_hi) / max(1 - a * (z0 + z_hi), 1e-9))
    ci_low = float(np.quantile(boot_stats, p_lo))
    ci_high = float(np.quantile(boot_stats, p_hi))

    return BCaBootstrapResult(
        point_estimate=theta_hat,
        ci_low=ci_low, ci_high=ci_high,
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
        bias=z0, acceleration=a,
    )


# ============================================================================
# Convenience: full report for a benchmark
# ============================================================================

@dataclass
class StatisticalReport:
    """Wraps the three tests for a single experiment with multiple backends."""
    experiment: str
    metric: str
    kruskal: KruskalWallisResult | None
    pairwise: list[PairwiseMWResult] = field(default_factory=list)
    bca_per_group: dict[str, BCaBootstrapResult] = field(default_factory=dict)


def full_report(experiment: str, metric: str,
                  groups: dict[str, Sequence[float]]) -> StatisticalReport:
    """
    Compute Kruskal-Wallis, Mann-Whitney+Bonferroni, BCa per group.
    Returns a StatisticalReport ready for CSV serialization.
    """
    nonempty = {k: list(v) for k, v in groups.items() if len(v) > 0}
    if len(nonempty) < 2:
        kw = None
        pw = []
    else:
        kw = kruskal_wallis(nonempty)
        pw = mann_whitney_pairwise(nonempty)

    bca = {}
    for name, samples in nonempty.items():
        if len(samples) >= 2:
            bca[name] = bca_bootstrap(samples)

    return StatisticalReport(
        experiment=experiment, metric=metric,
        kruskal=kw, pairwise=pw, bca_per_group=bca,
    )

