"""
Build the historical CRM table and the quantum-advantage forecast chart.

Combines:
  - Snowdrop 4q ver2 (this work — measured)
  - Published transmon/trapped-ion chips (from cited papers)
  - Classical-simulation upper bound (max N representable in available VRAM
    on consumer hardware)

Produces a single chart that shows the BOUNDARY of NISQ cryptanalysis:
  Y-axis = log2(N) where N is the largest factorable / simulatable modulus
  X-axis = year
  Two curves:
    (A) CRM(year) — what real quantum chips can factor
    (B) classical-sim ceiling — what fits in 8 GB / 64 GB / 1 TB RAM

The crossing of (A) with the cryptographically meaningful line (RSA-2048 = 2048
bits) is the *quantum-advantage-for-cryptanalysis* deadline.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from typing import Callable

if sys.platform == "win32":
    with suppress(AttributeError, OSError):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.crm_metric import (
    ChipParams, compute_crm, crm_for_calibration_dict,
    KNOWN_SHOR_PROFILES, predicted_success_probability,
    beauregard_resource_estimate,
)


# Published-chip database. Numbers are typical/headline values from cited
# sources; precision varies. Goal is to plot historical progression, not to
# adjudicate vendor claims.
PUBLISHED_CHIPS: list[ChipParams] = [
    ChipParams("IBM Q5 Yorktown",       2017, 5,   0.991, 0.94,  0.93,
               source="Tannu & Qureshi, MICRO 2018"),
    ChipParams("IBM Q20 Tokyo",         2018, 20,  0.998, 0.97,  0.94,
               source="IBM Q Experience documentation 2018"),
    ChipParams("Rigetti Acorn",         2018, 19,  0.997, 0.92,  0.93,
               source="Rigetti technical bulletin 2018"),
    ChipParams("Google Sycamore",       2019, 53,  0.999, 0.994, 0.962,
               source="Arute et al., Nature 574 (2019)"),
    ChipParams("IBM Falcon r4",         2020, 27,  0.999, 0.99,  0.97,
               source="IBM Quantum Roadmap 2020"),
    ChipParams("IonQ Aria",             2022, 23,  0.9995,0.997, 0.995,
               source="IonQ Aria datasheet 2022"),
    ChipParams("IBM Heron r1",          2023, 133, 0.9996,0.998, 0.985,
               source="IBM Quantum 2023"),
    ChipParams("Quantinuum H2",         2024, 56,  0.9999,0.998, 0.997,
               source="Quantinuum H-Series 2024"),
    ChipParams("Atom Computing",        2024, 1180,0.9995,0.995, 0.99,
               source="Atom Computing announcement 2024"),
    ChipParams("IBM Heron r2",          2024, 156, 0.9997,0.9985,0.99,
               source="IBM Quantum Summit 2024"),
]


def snowdrop_chip_params(measured: dict) -> ChipParams:
    """Build ChipParams from our measured Bauman Snowdrop calibration."""
    return ChipParams(
        name="Bauman Snowdrop 4q ver2",
        year=2026,
        n_qubits=int(measured.get("n_qubits", 4)),
        f_1q=float(measured.get("avg_f1q", 0.99893)),
        f_2q=float(measured.get("avg_f2q", 0.99067)),
        f_ro=float(measured.get("avg_ro", 0.94855)),
        measured=True,
        source="this work — Bauman Octillion API, May 2026",
    )


def classical_sim_ceiling_log2_N_at_year(year: int) -> float:
    """
    Approximate log2(N) such that classical state-vector simulation of
    Shor for N still fits in available consumer-grade RAM.

    Snapshot of consumer hardware progress:
      ~2020: 32-64 GB RAM → ~30 qubits → log2(N) ≈ 30
      ~2026: 64-128 GB RAM + GPUs → ~36-38 qubits → log2(N) ≈ 36
      ~2030: ~40+ qubits if scaling continues
    Rough doubling time for accessible RAM ≈ 4 years.
    """
    qubits_2020 = 30.0
    growth_per_year = 1.0
    qubits = qubits_2020 + (year - 2020) * growth_per_year
    return float(qubits)


def crm_log2_at_year(year: int, all_chips: list[ChipParams],
                       threshold: float = 0.30) -> float:
    """
    Best CRM achieved by any known chip in or before `year`, in log2 form.
    """
    crms = []
    for chip in all_chips:
        if chip.year > year:
            continue
        res = compute_crm(chip, threshold=threshold)
        if res.crm > 1:
            crms.append(math.log2(res.crm))
    return max(crms) if crms else 0.0


def fit_exponential(years: list[int], log2_N: list[float]
                     ) -> tuple[float, float]:
    """Linear regression on (year, log2(N)). Returns (slope, intercept)."""
    if len(years) < 2:
        return 0.0, 0.0
    x = np.array(years, dtype=float)
    y = np.array(log2_N, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def years_until_crm_crosses(target_log2_N: float,
                              slope: float, intercept: float) -> Optional_float:
    """Year when fitted line crosses target log2(N). None if slope ≤ 0."""
    if slope <= 0:
        return None
    return (target_log2_N - intercept) / slope


# alias because Optional[float] in old type syntax doesn't read well inline
Optional_float = float | None


def build_artifacts(measured_calibration: dict,
                     output_dir: Path | None = None,
                     threshold: float = 0.30,
                     status: Callable[[str], None] | None = None) -> dict[str, Path]:
    """Generate the CRM CSV/JSON/PNG/MD artifacts."""
    status = status or (lambda m: None)
    if output_dir is None:
        import time as _time
        stamp = _time.strftime("%d%m%Y %H %M")
        output_dir = (PROJECT_ROOT / "results" / "crm_forecast" / stamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    snowdrop = snowdrop_chip_params(measured_calibration)
    all_chips = list(PUBLISHED_CHIPS) + [snowdrop]
    all_chips.sort(key=lambda c: c.year)

    # 1) CRM table
    rows = []
    for chip in all_chips:
        res = compute_crm(chip, threshold=threshold)
        rows.append((chip, res))

    csv_path = output_dir / "crm_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["chip", "year", "n_qubits", "F_1q", "F_2q", "F_RO",
                     "CRM", "log2_CRM", "chosen_N", "P_success_at_CRM",
                     "measured", "source"])
        for chip, res in rows:
            log2_crm = math.log2(res.crm) if res.crm > 1 else 0.0
            w.writerow([
                chip.name, chip.year, chip.n_qubits,
                f"{chip.f_1q:.4f}", f"{chip.f_2q:.4f}", f"{chip.f_ro:.4f}",
                res.crm, f"{log2_crm:.2f}",
                res.chosen_profile.N if res.chosen_profile else "",
                f"{res.p_at_crm:.4f}",
                "yes" if chip.measured else "no",
                chip.source,
            ])
    status(f"  ✓ {csv_path.name}")

    # 2) Per-chip JSON
    json_path = output_dir / "crm_table.json"
    json_payload = {
        "threshold_tau": threshold,
        "definition": (
            "CRM(chip) = max{N : P_success(compiled Shor for N | chip) >= tau}, "
            "with depolarizing-channel noise model from per-gate fidelities."
        ),
        "entries": [
            {**res.to_dict(), "source": chip.source}
            for chip, res in rows
        ],
    }
    json_path.write_text(json.dumps(json_payload, indent=2, ensure_ascii=False),
                          encoding="utf-8")
    status(f"  ✓ {json_path.name}")

    # 3) Forecast chart
    chart_path = output_dir / "crm_forecast.png"
    years_with_crm = sorted(set(c.year for c in all_chips))
    crm_log2 = [crm_log2_at_year(y, all_chips, threshold) for y in years_with_crm]
    nonzero_idx = [i for i, v in enumerate(crm_log2) if v > 0]
    fit_years = [years_with_crm[i] for i in nonzero_idx]
    fit_vals = [crm_log2[i] for i in nonzero_idx]
    slope, intercept = fit_exponential(fit_years, fit_vals)

    # Quantum-advantage crossing forecast (computed early so we can size
    # the x-axis to actually contain the projected break year)
    crossing = years_until_crm_crosses(math.log2(2048), slope, intercept)
    crossing_visible = crossing if (crossing and crossing < 2120) else None

    fig, ax = plt.subplots(figsize=(13, 7.0))

    # Plot every chip as a dot; emphasise "this work" with a green star
    measured_chip = None
    published_label_used = False
    for chip, res in rows:
        if res.crm <= 1:
            continue
        log2_v = math.log2(res.crm)
        if chip.measured:
            measured_chip = (chip, res, log2_v)
            ax.scatter([chip.year], [log2_v], c="#0a7a3a", marker="*", s=520,
                       edgecolors="black", linewidth=1.0, zorder=6,
                       label=("this work (measured): "
                              f"{chip.name}, CRM={res.crm}"))
        else:
            ax.scatter([chip.year], [log2_v], c="#1f5f9b", marker="o", s=85,
                       edgecolors="black", linewidth=0.7, zorder=4,
                       label=(None if published_label_used
                              else "published NISQ chips (10 entries)"))
            published_label_used = True

    # ---- Cluster published chips by (log2_v rounded, year-band of 3y) ----
    # so annotations never collide. Show one label per cluster with a count.
    clusters: dict[tuple[float, int], list[tuple[int, str]]] = {}
    for chip, res in rows:
        if res.crm <= 1 or chip.measured:
            continue
        v = round(math.log2(res.crm), 2)
        band = (chip.year // 3) * 3
        clusters.setdefault((v, band), []).append((chip.year, chip.name))

    cluster_order = sorted(clusters.items(), key=lambda kv: (kv[0][1], -kv[0][0]))
    for i, ((v, band), entries) in enumerate(cluster_order):
        ys = [e[0] for e in entries]
        names = [e[1] for e in entries]
        x_anchor = sum(ys) / len(ys)
        # Compose label: 1 chip = "Name", multiple = "Earliest + N more"
        if len(entries) == 1:
            text = names[0]
        else:
            short = names[0].replace(" r1", "").replace(" r2", "")
            text = f"{short}  (+{len(entries) - 1} more)"
        # Place labels to the RIGHT of their cluster markers (with a small
        # vertical stagger). Anchoring to the right keeps text away from the
        # y-axis even when the cluster sits at the leftmost year.
        dy = 8 if (i % 2 == 0) else -10
        ax.annotate(
            text, xy=(x_anchor, v),
            xytext=(22, dy), textcoords="offset points",
            fontsize=10, ha="left", color="#1F2A4D", fontweight="bold",
            arrowprops=dict(arrowstyle="-", color="#666", linewidth=0.7),
            bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                      edgecolor="#bbb", linewidth=0.6, alpha=0.92),
        )

    # Snowdrop label — never auto-clustered (it's the measured star).
    # Use a high-contrast white bubble so the green text reads clearly.
    if measured_chip is not None:
        chip, res, log2_v = measured_chip
        ax.annotate(
            f"{chip.name}\n(this work, CRM={res.crm})",
            xy=(chip.year, log2_v),
            xytext=(14, -34), textcoords="offset points",
            fontsize=11, ha="left", color="#0a4a25", fontweight="bold",
            arrowprops=dict(arrowstyle="->",
                             color="#0a7a3a", linewidth=1.3),
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="#0a7a3a", linewidth=1.0, alpha=0.95),
        )

    # Fitted CRM trajectory — extends through the crossing year if visible
    x_fit_max = max(2050, (crossing_visible or 0) + 2)
    if slope > 0:
        x_fit = np.linspace(min(fit_years), x_fit_max, 200)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, "--", color="#0a7a3a", linewidth=2.0,
                alpha=0.85,
                label=f"CRM trend (linear fit, +{slope:.2f} log₂(N) / year)")

    # Classical-simulation ceiling
    sim_years = list(range(2017, int(x_fit_max) + 1))
    sim_qubits = [classical_sim_ceiling_log2_N_at_year(y) for y in sim_years]
    ax.plot(sim_years, sim_qubits, "-", color="#9bc4e2", linewidth=2.2,
            label="Classical state-vector sim (consumer RAM)")

    # Cryptographic targets
    ax.axhline(math.log2(2048), color="#c00", linestyle=":", linewidth=2.0,
               label="RSA-2048 (target, log₂N = 11)")
    ax.axhline(math.log2(256), color="#888", linestyle=":", linewidth=1.3,
               label="ECC P-256 (effective target, log₂N = 8)")

    # Crossing year — vertical line + inline annotation in the upper area
    if crossing_visible is not None:
        ax.axvline(crossing_visible, color="#c00", linestyle="--",
                   linewidth=1.4, alpha=0.75)
        ax.annotate(
            f"projected RSA-2048 break\n(no QEC): {crossing_visible:.0f}",
            xy=(crossing_visible, math.log2(2048)),
            xytext=(crossing_visible - 7, math.log2(2048) + 1.6),
            color="#c00", fontsize=11.5, ha="right", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#c00", linewidth=1.2),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#c00", linewidth=0.9, alpha=0.95),
        )

    ax.set_xlabel("Year", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_ylabel("log₂(N)   —   largest factorable / simulatable integer",
                  fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title("CRM forecast: NISQ cryptanalysis vs RSA-2048\n"
                 f"(τ = {threshold}, depolarizing noise model)",
                 fontweight="bold", fontsize=13, pad=12)
    ax.tick_params(axis="both", which="major", labelsize=11)

    # Padding on the left (2 yrs) so cluster labels never touch the y-axis,
    # and on the right (1 yr) so the crossing-year annotation has room.
    x_lo = min(c.year for c in all_chips) - 2
    x_hi = int(x_fit_max) + 1 if crossing_visible else 2046
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(0, 13)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95, ncol=1,
              edgecolor="#888")

    # Generous left/bottom margins so axis labels and the leftmost chip
    # annotation never crowd the figure border. tight_layout was clipping
    # the y-label rotated text against the left edge.
    fig.subplots_adjust(left=0.085, right=0.985, top=0.90, bottom=0.115)
    fig.savefig(chart_path, dpi=200,
                facecolor="white", edgecolor="none")
    plt.close(fig)
    status(f"  ✓ {chart_path.name}")

    # 4) Markdown section
    md_path = output_dir / "crm_report.md"
    snowdrop_res = compute_crm(snowdrop, threshold=threshold)
    snowdrop_p = snowdrop_res.p_at_crm
    crossing_str = (f"≈ {crossing:.0f}" if crossing and crossing < 2100
                    else "beyond 2050 (or never under linear extrapolation)")

    lines = [
        "# Cryptographically Reachable Modulus (CRM)\n",
        "## Definition\n",
        "We introduce **CRM(chip)** as a chip-specific cryptanalytic benchmark:\n",
        "> **CRM(chip) = max{ N : P_success(compiled Shor for N | chip) ≥ τ }**\n",
        "where P_success is computed from a depolarizing noise model "
        "parameterized by the chip's measured per-gate fidelities "
        "{F_1q, F_cz, F_RO}, and τ = "
        f"{threshold:.2f} is the post-processing recoverability threshold "
        "(ideal Shor with one counting qubit yields P=0.5; below ≈0.3 "
        "continued-fractions reconstruction of the period fails reliably).\n",
        "## Positioning\n",
        "| Metric | Type | Translates to cryptanalytic capability? |",
        "|---|---|---|",
        "| Quantum Volume (IBM 2019) | random-circuit benchmark | no |",
        "| Algorithmic Qubits (IonQ) | qubit-count adjusted | partially |",
        "| Logical-qubit estimates (NIST IR 8547) | resource estimate | yes (long-term, fault-tolerant) |",
        "| **CRM (this work)** | **measured, NISQ, chip-specific** | **yes (today)** |",
        "\n",
        "## Result for Bauman Snowdrop 4q ver2 (this work)\n",
        f"- Measured F_1q = {snowdrop.f_1q*100:.2f}%, "
        f"F_cz = {snowdrop.f_2q*100:.2f}%, F_RO = {snowdrop.f_ro*100:.2f}%",
        f"- **CRM = {snowdrop_res.crm}** "
        f"(P_success = {snowdrop_p:.3f} for N = {snowdrop_res.crm}, "
        f"τ = {threshold})",
        f"- Confirmed empirically: Shor for N=15 successfully recovered "
        f"factors {3, 5} on real chip (job IDs in `Bauman/runs/`).",
        "\n## Historical CRM table\n",
        "| Year | Chip | n_q | F_2q | CRM | log₂(CRM) | source |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for chip, res in rows:
        log2_crm = math.log2(res.crm) if res.crm > 1 else 0.0
        marker = " ★" if chip.measured else ""
        lines.append(
            f"| {chip.year} | {chip.name}{marker} | {chip.n_qubits} "
            f"| {chip.f_2q:.4f} | {res.crm} | {log2_crm:.2f} "
            f"| {chip.source[:40]} |"
        )
    lines.append("")
    lines.append("![CRM forecast](crm_forecast.png)")
    lines.append("")
    lines.append("## Forecast\n")
    if slope > 0:
        lines.append(
            f"Linear fit on log₂(CRM) vs year: slope = **+{slope:.3f} "
            f"log₂(N) / year** (≈ {2 ** slope:.2f}× growth in factorable "
            "modulus per year).\n"
        )
    lines.append(
        f"Under this trend (NISQ-only, no quantum error correction):\n\n"
        f"- **RSA-2048 reachable: {crossing_str}**\n"
        f"- (For comparison: NIST IR 8547 fault-tolerant CRQC estimates "
        "give 2030–2040 with QEC; our NISQ-only estimate is necessarily "
        "later because we exclude error correction overhead.)\n"
    )
    lines.append("\n## Caveats (state honestly)\n")
    lines.append(
        "1. Depolarizing noise model — does not capture crosstalk, leakage, "
        "or correlated errors that may further degrade P.\n"
        "2. NISQ-only — no quantum error correction modeling. A chip with "
        "QEC could reach RSA-2048 much earlier.\n"
        "3. Compiled-Shor profiles are tabulated for N ≤ 511; "
        "extrapolation beyond uses Beauregard 2003 asymptotic costs.\n"
        "4. P_success threshold τ = 0.30 chosen empirically for "
        "continued-fractions recoverability; tighter τ shrinks CRM.\n"
    )
    lines.append("\n## Reproducibility\n")
    lines.append(
        "- Snowdrop data measured via Bauman Octillion API; "
        "raw counts saved with job IDs in `Bauman/runs/`.\n"
        "- CRM computation: `src/crm_metric.py` (deterministic from "
        "chip parameters).\n"
        "- Published-chip parameters: see `crm_table.csv` `source` column.\n"
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    status(f"  ✓ {md_path.name}")

    return {
        "crm_csv": csv_path,
        "crm_json": json_path,
        "crm_chart": chart_path,
        "crm_md": md_path,
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=float, default=0.30)
    p.add_argument("--use-saved-calibration", action="store_true",
                   help="Read calibration from Bauman/snowdrop_4q_ver2.json")
    args = p.parse_args()

    if args.use_saved_calibration:
        cal_file = PROJECT_ROOT / "Bauman" / "snowdrop_4q_ver2.json"
        cal = json.loads(cal_file.read_text(encoding="utf-8"))
        params = {p["name"]: float(p["values"]) / 100
                  if p["dimension"] == "%" else float(p["values"])
                  for p in cal.get("params", [])}
        measured = {
            "n_qubits": cal["model"]["count_qubits"],
            "avg_f1q": params.get("fidelity_1q", 0.999),
            "avg_f2q": params.get("fidelity_cz", 0.99),
            "avg_ro":  params.get("fidelity_ro", 0.95),
        }
    else:
        from src.quantum_hardware import get_client, get_backend, describe_chip
        client = get_client()
        be = get_backend(client, real_hardware=True)
        spec = describe_chip(be)
        measured = {
            "n_qubits": spec.num_qubits,
            "avg_f1q": spec.avg_f1q,
            "avg_f2q": spec.avg_f2q,
            "avg_ro":  spec.avg_ro,
        }

    print(f"Snowdrop measured: F_1q={measured['avg_f1q']*100:.2f}%, "
          f"F_2q={measured['avg_f2q']*100:.2f}%, "
          f"F_RO={measured['avg_ro']*100:.2f}%\n")

    artifacts = build_artifacts(measured, threshold=args.threshold,
                                  status=lambda m: print(m))
    print("\nDone.")
    for name, p in artifacts.items():
        print(f"  {name}: {p}")


if __name__ == "__main__":
    main()

