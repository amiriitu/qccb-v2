"""
Tk dialog wrapping the four thesis-defence calculators in one window:
  Tab 1: Quantum resource estimator (Shor RSA/ECC, Grover AES/SHA)
  Tab 2: Dynamic SCI(h⃗,n⃗,p⃗) + Sobol sensitivity
  Tab 3: HNDL residual-risk calculator
  Tab 4: TLS 1.3 hybrid handshake comparator

Used standalone (`python -m src.thesis_tools_gui`) or opened from the main GUI.
"""
from __future__ import annotations

import sys
from contextlib import suppress
import tkinter as tk
from pathlib import Path
from tkinter import ttk

if sys.platform == "win32":
    with suppress(AttributeError, OSError):
        sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.quantum_resource_estimator import (
    build_threat_table, crqc_year_estimate,
)
from src.sci_dynamic import (
    DynamicSCIInputs, compute_dynamic_sci, sobol_sensitivity,
)
from src.hndl_calculator import (
    HNDLInputs, compute_hndl, CRQCDistribution,
    DATA_CLASS_LIFETIME, P_BROKEN_GIVEN_CRQC,
)
from src.tls_hybrid_handshake import compare_handshakes


MONO = ("Consolas", 9)


class ThesisToolsWindow:
    def __init__(self, root: tk.Tk, embedded: bool = False):
        self.root = root
        if not embedded:
            self.root.title("Thesis tools")
            self.root.geometry("1100x780")

        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=8, pady=8)

        self._build_qres_tab(nb)
        self._build_sci_tab(nb)
        self._build_hndl_tab(nb)
        self._build_tls_tab(nb)

    # ------------------------------------------------------------------
    # Tab 1 — Quantum resource estimator
    # ------------------------------------------------------------------
    def _build_qres_tab(self, nb: ttk.Notebook):
        f = ttk.Frame(nb, padding=10)
        nb.add(f, text="🔬 Q-resources (Shor + Grover)")

        ttk.Label(f,
            text=("Resource estimation for Shor / Grover attacks against "
                  "RSA, ECC, AES, SHA.\n"
                  "Sources: Gidney-Ekerå '21 (RSA), Roetteler-Naehrig-Svore '17 "
                  "(ECC), Jaques et al. '20 (AES), Amy et al. '16 (SHA)."),
            justify="left", font=("TkDefaultFont", 9, "italic"),
            foreground="#555",
        ).pack(anchor="w")

        cols = ("target", "alg", "n_log", "n_phys", "T_gates", "Toffoli",
                "depth", "runtime_h", "CRQC_year", "source")
        tv = ttk.Treeview(f, columns=cols, show="headings", height=14)
        widths = (130, 70, 90, 130, 110, 110, 110, 90, 80, 220)
        for c, w in zip(cols, widths):
            tv.heading(c, text=c)
            tv.column(c, width=w, anchor="e" if c not in ("target", "source") else "w")
        tv.pack(fill="both", expand=True, pady=(8, 0))

        for r in build_threat_table():
            d = r.to_dict()
            tv.insert("", "end", values=(
                d["target"], d["algorithm"], f'{r.n_logical_qubits:,}',
                f'{r.n_physical_qubits:,}',
                f'10^{d["t_gates_log10"]:.1f}',
                f'10^{d["toffoli_gates_log10"]:.1f}' if d["toffoli_gates_log10"] > 0 else "—",
                f'10^{d["circuit_depth_log10"]:.1f}',
                f'{d["runtime_hours"]:.1f}',
                f'{crqc_year_estimate(r)}',
                d["source"],
            ))

        ttk.Label(f, text=(
            "Reading: 'CRQC_year' assumes Moore-style doubling of physical "
            "qubits every 2 yr from a 2024 baseline of 1000 qubits. Compare "
            "against your data's confidentiality lifetime to assess HNDL risk "
            "(see HNDL tab)."),
            justify="left", font=("TkDefaultFont", 8), foreground="#666",
            wraplength=1050,
        ).pack(anchor="w", pady=(8, 0))

    # ------------------------------------------------------------------
    # Tab 2 — Dynamic SCI + Sobol
    # ------------------------------------------------------------------
    def _build_sci_tab(self, nb: ttk.Notebook):
        f = ttk.Frame(nb, padding=10)
        nb.add(f, text="📐 Dynamic SCI + Sobol")

        top = ttk.Frame(f)
        top.pack(fill="x", expand=False)

        ttk.Label(top, text="SCI(h⃗, n⃗, p⃗) = SCI_static · f_hw · f_net · f_proto",
                   font=("TkDefaultFont", 10, "bold")).grid(row=0, column=0,
                                                              columnspan=4,
                                                              sticky="w")

        self.sci_vars = {
            "sci_static":          tk.DoubleVar(value=1.0),
            "cpu_freq_ghz":        tk.DoubleVar(value=3.5),
            "cores":               tk.IntVar(value=8),
            "aes_ni":              tk.BooleanVar(value=True),
            "ram_gb":              tk.DoubleVar(value=16.0),
            "rtt_ms":              tk.DoubleVar(value=1.0),
            "mtu":                 tk.IntVar(value=1500),
            "packet_loss":         tk.DoubleVar(value=0.0),
            "bandwidth_mbps":      tk.DoubleVar(value=1000.0),
            "payload_bytes":       tk.IntVar(value=1500),
            "handshake_hz":        tk.DoubleVar(value=0.1),
            "session_lifetime_s":  tk.DoubleVar(value=3600.0),
            "extra_round_trips":   tk.IntVar(value=0),
        }
        labels = [
            ("Static SCI",       "sci_static",          "from primary SCI calc"),
            ("CPU freq (GHz)",   "cpu_freq_ghz",        "h⃗"),
            ("CPU cores",        "cores",               "h⃗"),
            ("AES-NI",           "aes_ni",              "h⃗"),
            ("RAM (GB)",         "ram_gb",              "h⃗"),
            ("RTT (ms)",         "rtt_ms",              "n⃗ — LAN=1, WAN=30"),
            ("MTU (bytes)",      "mtu",                 "n⃗"),
            ("Packet loss",      "packet_loss",         "n⃗ — 0.001=0.1%"),
            ("Bandwidth (Mbps)", "bandwidth_mbps",      "n⃗"),
            ("Payload (bytes)",  "payload_bytes",       "n⃗"),
            ("Handshakes/sec",   "handshake_hz",        "p⃗"),
            ("Session life (s)", "session_lifetime_s",  "p⃗"),
            ("Extra round-trips","extra_round_trips",   "p⃗"),
        ]
        for i, (lbl, key, hint) in enumerate(labels):
            ttk.Label(top, text=lbl).grid(row=1 + i, column=0, sticky="w", pady=1)
            if key == "aes_ni":
                ttk.Checkbutton(top, variable=self.sci_vars[key]
                                  ).grid(row=1 + i, column=1, sticky="w")
            else:
                ttk.Entry(top, textvariable=self.sci_vars[key], width=14
                            ).grid(row=1 + i, column=1, sticky="w")
            ttk.Label(top, text=hint, font=("TkDefaultFont", 8),
                       foreground="#888").grid(row=1 + i, column=2,
                                                 sticky="w", padx=(8, 0))
            self.sci_vars[key].trace_add("write", lambda *a: self._recalc_sci())

        ttk.Button(top, text="Run Sobol sensitivity (S_i, S_Ti)",
                    command=self._run_sobol
                    ).grid(row=14, column=0, columnspan=3,
                            sticky="ew", pady=(8, 0))

        self.sci_out = tk.Text(f, height=18, wrap="word", font=MONO)
        self.sci_out.pack(fill="both", expand=True, pady=(8, 0))
        self.sci_out.configure(state="disabled")
        self._recalc_sci()

    def _read_sci_inputs(self) -> DynamicSCIInputs:
        v = {k: var.get() for k, var in self.sci_vars.items()}
        return DynamicSCIInputs(
            sci_static=v["sci_static"],
            cpu_freq_ghz=v["cpu_freq_ghz"], cores=int(v["cores"]),
            aes_ni=v["aes_ni"], ram_gb=v["ram_gb"],
            rtt_ms=v["rtt_ms"], mtu=int(v["mtu"]),
            packet_loss=v["packet_loss"],
            bandwidth_mbps=v["bandwidth_mbps"],
            payload_bytes=int(v["payload_bytes"]),
            handshake_hz=v["handshake_hz"],
            session_lifetime_s=v["session_lifetime_s"],
            extra_round_trips=int(v["extra_round_trips"]),
        )

    def _recalc_sci(self):
        try:
            inp = self._read_sci_inputs()
            r = compute_dynamic_sci(inp)
            txt = (
                f"  f_hw    = {r.f_hw:.3f}\n"
                f"  f_net   = {r.f_net:.3f}\n"
                f"  f_proto = {r.f_proto:.3f}\n"
                f"  ──────────────────\n"
                f"  SCI_static  = {r.sci_static:.4f}\n"
                f"  SCI_dynamic = {r.sci_dynamic:.4f}\n\n"
                f"  {r.interpretation}\n"
            )
        except Exception as e:
            txt = f"  (input error: {e})\n"
        self.sci_out.configure(state="normal")
        self.sci_out.delete("1.0", "end")
        self.sci_out.insert("1.0", txt)
        self.sci_out.configure(state="disabled")

    def _run_sobol(self):
        try:
            inp = self._read_sci_inputs()
            # We sweep each parameter ±50% around its current value.
            def model(rtt_ms, packet_loss, cpu_freq_ghz, payload_bytes):
                trial = DynamicSCIInputs(
                    sci_static=inp.sci_static,
                    cpu_freq_ghz=cpu_freq_ghz, cores=inp.cores,
                    aes_ni=inp.aes_ni, ram_gb=inp.ram_gb,
                    rtt_ms=rtt_ms, mtu=inp.mtu,
                    packet_loss=packet_loss,
                    bandwidth_mbps=inp.bandwidth_mbps,
                    payload_bytes=int(payload_bytes),
                    handshake_hz=inp.handshake_hz,
                    session_lifetime_s=inp.session_lifetime_s,
                    extra_round_trips=inp.extra_round_trips,
                )
                return compute_dynamic_sci(trial).sci_dynamic
            bounds = {
                "rtt_ms":         (1.0, 100.0),
                "packet_loss":    (0.0, 0.05),
                "cpu_freq_ghz":   (0.8, 4.5),
                "payload_bytes":  (256, 16000),
            }
            indices = sobol_sensitivity(model, bounds, n_base=256)
            lines = ["", "Sobol sensitivity over LAN→WAN regime:", ""]
            lines.append(f"  {'parameter':18s}  S_i      S_Ti")
            lines.append(f"  {'─' * 18}  ───────  ───────")
            for ix in indices:
                lines.append(f"  {ix.parameter:18s}  {ix.s_first:7.3f}  "
                              f"{ix.s_total:7.3f}")
            lines.append("")
            lines.append("Reading: parameters with high S_Ti are the levers "
                         "that really matter; ΣS_i ≈ 1 for additive models, "
                         "less for interaction-heavy ones.")
            self.sci_out.configure(state="normal")
            self.sci_out.insert("end", "\n".join(lines))
            self.sci_out.configure(state="disabled")
        except Exception as e:
            self.sci_out.configure(state="normal")
            self.sci_out.insert("end", f"\n(Sobol error: {e})")
            self.sci_out.configure(state="disabled")

    # ------------------------------------------------------------------
    # Tab 3 — HNDL
    # ------------------------------------------------------------------
    def _build_hndl_tab(self, nb: ttk.Notebook):
        f = ttk.Frame(nb, padding=10)
        nb.add(f, text="🕒 HNDL — Harvest Now, Decrypt Later")

        ttk.Label(f, text=(
            "Residual risk = P(CRQC arrives before data expires) · "
            "P(crypto broken | CRQC).\n"
            "Default CRQC distribution: log-normal, median 2034, P95 2050 "
            "(NIST IR 8547 Aug 2024 consensus)."),
            justify="left", font=("TkDefaultFont", 9, "italic"),
            foreground="#555",
        ).pack(anchor="w")

        top = ttk.Frame(f)
        top.pack(fill="x", pady=(8, 0))

        self.hndl_vars = {
            "capture_year":           tk.IntVar(value=2026),
            "data_class":             tk.StringVar(value="medical-records"),
            "confidentiality_years":  tk.IntVar(value=30),
            "crypto":                 tk.StringVar(value="RSA-2048"),
            "crqc_median":            tk.IntVar(value=2034),
            "crqc_p95":               tk.IntVar(value=2050),
        }
        ttk.Label(top, text="Capture year").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.hndl_vars["capture_year"], width=10
                    ).grid(row=0, column=1, sticky="w")
        ttk.Label(top, text="Data class").grid(row=0, column=2, sticky="w",
                                                 padx=(20, 0))
        cls_combo = ttk.Combobox(top, textvariable=self.hndl_vars["data_class"],
                                  values=list(DATA_CLASS_LIFETIME.keys()),
                                  state="readonly", width=22)
        cls_combo.grid(row=0, column=3, sticky="w")

        ttk.Label(top, text="Confidentiality (yrs)").grid(row=1, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.hndl_vars["confidentiality_years"], width=10
                    ).grid(row=1, column=1, sticky="w")
        ttk.Label(top, text="Crypto").grid(row=1, column=2, sticky="w", padx=(20, 0))
        crypto_combo = ttk.Combobox(top, textvariable=self.hndl_vars["crypto"],
                                      values=list(P_BROKEN_GIVEN_CRQC.keys()),
                                      state="readonly", width=22)
        crypto_combo.grid(row=1, column=3, sticky="w")

        ttk.Label(top, text="CRQC median yr").grid(row=2, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.hndl_vars["crqc_median"], width=10
                    ).grid(row=2, column=1, sticky="w")
        ttk.Label(top, text="CRQC P95 yr").grid(row=2, column=2, sticky="w",
                                                 padx=(20, 0))
        ttk.Entry(top, textvariable=self.hndl_vars["crqc_p95"], width=10
                    ).grid(row=2, column=3, sticky="w")

        for var in self.hndl_vars.values():
            var.trace_add("write", lambda *a: self._recalc_hndl())

        self.hndl_out = tk.Text(f, height=16, wrap="word", font=MONO)
        self.hndl_out.pack(fill="both", expand=True, pady=(8, 0))
        self.hndl_out.configure(state="disabled")
        self._recalc_hndl()

    def _recalc_hndl(self):
        try:
            v = {k: var.get() for k, var in self.hndl_vars.items()}
            dist = CRQCDistribution(median_year=int(v["crqc_median"]),
                                      p95_year=int(v["crqc_p95"]))
            res = compute_hndl(HNDLInputs(
                capture_year=int(v["capture_year"]),
                confidentiality_years=int(v["confidentiality_years"]),
                crypto=v["crypto"],
                data_class=v["data_class"],
                crqc_dist=dist,
            ))
            d = res.to_dict()
            txt = (
                f"  data class             = {d['data_class']}\n"
                f"  capture → expiry year  = {d['capture_year']} → {d['expiry_year']}\n"
                f"  crypto in use          = {d['crypto']}\n"
                f"  ─────────────────────────────────────\n"
                f"  P(CRQC by expiry)      = {d['p_crqc_in_lifetime']*100:6.2f}%\n"
                f"  P(broken | CRQC)       = {d['p_broken_if_crqc']*100:6.2f}%\n"
                f"  Residual HNDL risk     = {d['residual_risk']*100:6.2f}%\n"
                f"  Safe-until capture year ≤ {d['safe_until_year']}\n\n"
                f"  {d['interpretation']}\n"
            )
        except Exception as e:
            txt = f"  (input error: {e})\n"
        self.hndl_out.configure(state="normal")
        self.hndl_out.delete("1.0", "end")
        self.hndl_out.insert("1.0", txt)
        self.hndl_out.configure(state="disabled")

    # ------------------------------------------------------------------
    # Tab 4 — TLS hybrid handshake
    # ------------------------------------------------------------------
    def _build_tls_tab(self, nb: ttk.Notebook):
        f = ttk.Frame(nb, padding=10)
        nb.add(f, text="🤝 TLS 1.3 hybrid handshake")

        ttk.Label(f, text=(
            "Comparison of TLS 1.3 handshake variants under LAN/WAN. "
            "Sizes follow RFC 8446 + draft-ietf-tls-hybrid-design."),
            justify="left", font=("TkDefaultFont", 9, "italic"),
            foreground="#555",
        ).pack(anchor="w")

        top = ttk.Frame(f)
        top.pack(fill="x", pady=(8, 0))

        self.tls_rtt = tk.DoubleVar(value=1.0)
        self.tls_mtu = tk.IntVar(value=1500)
        ttk.Label(top, text="RTT (ms)").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.tls_rtt, width=10).grid(row=0, column=1)
        ttk.Label(top, text="MTU").grid(row=0, column=2, sticky="w", padx=(20, 0))
        ttk.Entry(top, textvariable=self.tls_mtu, width=10).grid(row=0, column=3)
        ttk.Button(top, text="Recompute", command=self._recalc_tls
                    ).grid(row=0, column=4, padx=(20, 0))

        self.tls_out = tk.Text(f, wrap="none", font=MONO)
        self.tls_out.pack(fill="both", expand=True, pady=(8, 0))
        self.tls_out.configure(state="disabled")
        self._recalc_tls()

    def _recalc_tls(self):
        try:
            results = compare_handshakes(rtt_ms=float(self.tls_rtt.get()),
                                           mtu_bytes=int(self.tls_mtu.get()))
            baseline = results[0].handshake_ms
            lines = []
            lines.append(f"{'Variant':50s} {'bytes':>7} {'MTU':>4} "
                          f"{'compute':>9} {'network':>9} {'total ms':>9} "
                          f"{'overhead':>9}")
            lines.append("─" * 110)
            for r in results:
                ovh = (r.handshake_ms - baseline) / baseline * 100
                lines.append(
                    f"{r.label:50s} {r.bytes_total:>7d} "
                    f"{('  ✓ ' if r.fits_in_mtu else '  ✗ '):>4} "
                    f"{r.compute_ms:>8.2f}  "
                    f"{r.network_ms:>8.1f}  "
                    f"{r.handshake_ms:>8.2f}  "
                    f"{ovh:>+7.0f}%"
                )
            txt = "\n".join(lines)
        except Exception as e:
            txt = f"(input error: {e})"
        self.tls_out.configure(state="normal")
        self.tls_out.delete("1.0", "end")
        self.tls_out.insert("1.0", txt)
        self.tls_out.configure(state="disabled")


def main():
    root = tk.Tk()
    from src.ui_theme import apply_theme, apply_app_icon
    apply_theme(root)
    apply_app_icon(root)
    ThesisToolsWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()

