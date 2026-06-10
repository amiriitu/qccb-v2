"""
Tk dialog window for the SCI calculator. Both formulas side by side, live update.
Used standalone (`python -m src.sci_calculator_gui`) or opened from the main GUI.
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

from src.sci_calculator import (
    SCIInputs, SCIHWInputs, compute_sci, compute_sci_hw, EXAMPLES,
)


class SCICalculatorWindow:
    def __init__(self, root: tk.Tk, embedded: bool = False):
        self.root = root
        self.embedded = embedded
        if not embedded:
            self.root.title("SCI Calculator — Scientific journal 'Bulletin of the CAA' 2026 + SCI_HW extension")
            self.root.geometry("960x720")

        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.columnconfigure(1, weight=1)

        self._build_paper_panel(outer).grid(row=0, column=0, sticky="nsew",
                                              padx=(0, 4))
        self._build_hw_panel(outer).grid(row=0, column=1, sticky="nsew",
                                           padx=(4, 0))

        examples_frame = ttk.LabelFrame(outer, text="Pre-computed examples (click to fill)",
                                          padding=8)
        examples_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        for i, (name, _ex) in enumerate(EXAMPLES.items()):
            ttk.Button(examples_frame, text=name,
                        command=lambda n=name: self._fill_example(n)).grid(
                row=i // 2, column=i % 2, sticky="ew", padx=2, pady=2
            )
        examples_frame.columnconfigure(0, weight=1)
        examples_frame.columnconfigure(1, weight=1)

        ref_lbl = ttk.Label(
            outer,
            text=("References:\n"
                  "  • Zhailin A.G., Bekarystankyzy A., Aktanova B.M., "
                  "Scientific journal 'Bulletin of the CAA' №1(40), 2026,\n"
                  "    DOI 10.53364/24138614_2026_40_1_11 — original SCI definition.\n"
                  "  • SCI_HW: extension of the dynamic SCI(h⃗) postulate from §2 "
                  "of the above paper,\n"
                  "    operationalized with measured chip parameters in this work."),
            justify="left",
            font=("TkDefaultFont", 8),
            foreground="#444",
        )
        ref_lbl.grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))

        self._recompute()

    def _build_paper_panel(self, parent) -> ttk.LabelFrame:
        f = ttk.LabelFrame(parent, text="Original SCI  (Scientific journal 'Bulletin of the CAA' 2026)",
                            padding=10)
        f.columnconfigure(1, weight=1)

        self.paper_vars = {
            "t_pqc":         tk.DoubleVar(value=568.16),
            "t_classical":   tk.DoubleVar(value=29.65),
            "size_pqc":      tk.DoubleVar(value=1184.0),
            "size_classical":tk.DoubleVar(value=256.0),
            "complexity":    tk.DoubleVar(value=1.0),
            "nist_level":    tk.IntVar(value=3),
        }
        labels = {
            "t_pqc":         "T_pqc       (ms)",
            "t_classical":   "T_classical (ms)",
            "size_pqc":      "Size_pqc       (B)",
            "size_classical":"Size_classical (B)",
            "complexity":    "Complexity   (1.0=lattice … 2.5=code)",
            "nist_level":    "NIST_level   (1, 2, 3, or 5)",
        }
        for i, (key, lbl) in enumerate(labels.items()):
            ttk.Label(f, text=lbl).grid(row=i, column=0, sticky="w", pady=2)
            e = ttk.Entry(f, textvariable=self.paper_vars[key], width=14)
            e.grid(row=i, column=1, sticky="ew", pady=2)
            self.paper_vars[key].trace_add("write", lambda *a: self._recompute_paper())

        ttk.Separator(f, orient="horizontal").grid(row=10, column=0, columnspan=2,
                                                    sticky="ew", pady=8)

        ttk.Label(f, text="Formula:",
                   font=("TkDefaultFont", 9, "bold")).grid(row=11, column=0,
                                                            sticky="w")
        ttk.Label(f,
                   text="SCI = (1+log10(Ovh)/5) × (1+log10(Sz)/5) × Cx × (3/NIST)",
                   font=("Consolas", 8)).grid(row=11, column=1, sticky="w")

        self.paper_out = tk.Text(f, height=10, wrap="word",
                                   font=("Consolas", 9))
        self.paper_out.grid(row=12, column=0, columnspan=2, sticky="nsew",
                              pady=(8, 0))
        self.paper_out.configure(state="disabled")
        f.rowconfigure(12, weight=1)

        return f

    def _build_hw_panel(self, parent) -> ttk.LabelFrame:
        f = ttk.LabelFrame(parent, text="SCI_HW  (this work, hardware-aware)",
                            padding=10)
        f.columnconfigure(1, weight=1)

        self.hw_vars = {
            "t_obs_s":       tk.DoubleVar(value=1.10),
            "t_ideal_s":     tk.DoubleVar(value=0.09),
            "metric_obs":    tk.DoubleVar(value=0.962),
            "metric_ideal":  tk.DoubleVar(value=1.0),
            "depth_t":       tk.IntVar(value=10),
            "depth_l":       tk.IntVar(value=3),
        }
        labels = {
            "t_obs_s":      "T_obs    (s)",
            "t_ideal_s":    "T_ideal  (s)",
            "metric_obs":   "M_obs    (observed metric, e.g. fidelity)",
            "metric_ideal": "M_ideal  (ideal-sim baseline)",
            "depth_t":      "D_transpiled (gate depth on chip)",
            "depth_l":      "D_logical    (gate depth pre-transpile)",
        }
        for i, (key, lbl) in enumerate(labels.items()):
            ttk.Label(f, text=lbl).grid(row=i, column=0, sticky="w", pady=2)
            e = ttk.Entry(f, textvariable=self.hw_vars[key], width=14)
            e.grid(row=i, column=1, sticky="ew", pady=2)
            self.hw_vars[key].trace_add("write", lambda *a: self._recompute_hw())

        ttk.Separator(f, orient="horizontal").grid(row=10, column=0, columnspan=2,
                                                    sticky="ew", pady=8)

        ttk.Label(f, text="Formula:",
                   font=("TkDefaultFont", 9, "bold")).grid(row=11, column=0,
                                                            sticky="w")
        ttk.Label(f,
                   text="SCI_HW = (T_obs/T_ideal) × |M_obs − M_ideal| × (D_t/D_l)",
                   font=("Consolas", 9)).grid(row=11, column=1, sticky="w")

        self.hw_out = tk.Text(f, height=10, wrap="word",
                                font=("Consolas", 9))
        self.hw_out.grid(row=12, column=0, columnspan=2, sticky="nsew",
                          pady=(8, 0))
        self.hw_out.configure(state="disabled")
        f.rowconfigure(12, weight=1)

        return f

    def _recompute_paper(self):
        try:
            v = {k: var.get() for k, var in self.paper_vars.items()}
            res = compute_sci(SCIInputs(**v))
            text = (
                f"  Overhead factor    = {res.overhead_factor:.4f}\n"
                f"  Size penalty       = {res.size_penalty:.4f}\n"
                f"  Complexity score   = {res.complexity_score:.4f}\n"
                f"  NIST factor        = {res.nist_factor:.4f}\n"
                f"  ─────────────────\n"
                f"  SCI                = {res.sci_raw:.4f}\n\n"
                f"  {res.interpretation}"
            )
        except Exception as e:
            text = f"  (input error: {e})"
        self.paper_out.configure(state="normal")
        self.paper_out.delete("1.0", "end")
        self.paper_out.insert("1.0", text)
        self.paper_out.configure(state="disabled")

    def _recompute_hw(self):
        try:
            v = {k: var.get() for k, var in self.hw_vars.items()}
            inp = SCIHWInputs(
                t_obs_s=v["t_obs_s"], t_ideal_s=v["t_ideal_s"],
                metric_obs=v["metric_obs"], metric_ideal=v["metric_ideal"],
                depth_transpiled=int(v["depth_t"]),
                depth_logical=int(v["depth_l"]),
            )
            res = compute_sci_hw(inp)
            text = (
                f"  Time factor      = ×{res.time_factor:.2f}\n"
                f"  Error factor     = {res.error_factor:.4f}\n"
                f"  Routing factor   = ×{res.routing_factor:.2f}\n"
                f"  ─────────────────\n"
                f"  SCI_HW           = {res.sci_hw:.4f}\n\n"
                f"  {res.interpretation}"
            )
        except Exception as e:
            text = f"  (input error: {e})"
        self.hw_out.configure(state="normal")
        self.hw_out.delete("1.0", "end")
        self.hw_out.insert("1.0", text)
        self.hw_out.configure(state="disabled")

    def _recompute(self):
        self._recompute_paper()
        self._recompute_hw()

    def _fill_example(self, name: str):
        ex = EXAMPLES[name]
        if ex["formula"] == "paper_sci":
            for k, v in ex["inputs"].items():
                if k in self.paper_vars:
                    self.paper_vars[k].set(v)
        else:
            mapping = {
                "t_obs_s": "t_obs_s",
                "t_ideal_s": "t_ideal_s",
                "metric_obs": "metric_obs",
                "metric_ideal": "metric_ideal",
                "depth_transpiled": "depth_t",
                "depth_logical": "depth_l",
            }
            for src_key, gui_key in mapping.items():
                if src_key in ex["inputs"]:
                    self.hw_vars[gui_key].set(ex["inputs"][src_key])


def main():
    root = tk.Tk()
    from src.ui_theme import apply_theme
    apply_theme(root)
    SCICalculatorWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()

