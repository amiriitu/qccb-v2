"""
QCCB v2 — Quantum Hardware Experiments GUI

Tkinter front-end for running parameterizable experiments on the
Bauman Octillion Snowdrop 4q ver2 chip (or its noisy/ideal simulators).

Layout:
  - Top:    Chip status panel (refreshes from live backend)
  - Middle: Experiment selector + dynamic parameter form + backend + shots + repeats
  - Buttons: Run / Run on all 3 backends / Benchmark (N repeats)
  - Status: Live job status feed (thread-safe)
  - Bottom: Matplotlib results panel (auto-switches between counts / 3-way / benchmark views)
"""
from __future__ import annotations

import json
import queue
import sys
from contextlib import suppress
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox, filedialog
from typing import Any

if sys.platform == "win32":
    with suppress(AttributeError, OSError):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments import list_experiments, ExperimentDef, ParameterSpec
from src.experiments.runner import (
    run_circuit, ExperimentResult, BenchmarkResult, benchmark,
)
from src.experiments.shor_n15 import interpret as shor_interpret
from src.quantum_hardware import get_client, get_backend, describe_chip
from src.full_benchmark import run_full_benchmark
from src.ui_theme import (
    apply_theme, apply_app_icon, style_console, style_matplotlib,
    PALETTE, CHART_COLORS,
)
from src.csv_preview import render_csv, scan_new_csvs


BACKENDS = [
    ("Ideal simulator (CPU, no noise)", "ideal"),
    ("GPU simulator (CuPy, RTX 4060)", "gpu"),
    ("Local emulator — Snowdrop 4q ver2 (offline JSON)", "emulator_4q_v2"),
    ("Local emulator — Snowdrop 4q ver1 (offline JSON)", "emulator_4q_v1"),
    ("Local emulator — Snowdrop 8q ver1 (offline JSON)", "emulator_8q_v1"),
    ("Real hardware — Snowdrop 4q ver2", "real"),
]


class QCCBGui:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(
            "QCCB v2 — Quantum Computing Cryptography Benchmark"
            "  ·  Snowdrop 4q ver2"
        )
        self.root.geometry("1380x980")
        self.root.minsize(1200, 860)
        # Open maximized — the layout is information-dense and is meant to
        # be shown on a projector.
        with suppress(tk.TclError):
            self.root.state("zoomed")

        self.experiments: list[ExperimentDef] = list_experiments()
        self.experiment_keys = [e.key for e in self.experiments]
        self.experiment_by_key = {e.key: e for e in self.experiments}

        self.msg_queue: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None
        self.last_results: dict[str, ExperimentResult] = {}
        self.last_benchmark: BenchmarkResult | None = None

        # Live CSV preview state (see _start_preview_watch)
        self._preview_paths: dict[str, Path] = {}
        self._preview_state: tuple[Path, float, bool] | None = None
        self._preview_polling = False
        self._preview_t0 = 0.0
        self._last_out_dir: Path | None = None

        self.experiment_var = tk.StringVar(value=self.experiment_keys[0])
        self.backend_var = tk.StringVar(value="real")
        self.shots_exp_var = tk.IntVar(value=10)
        self.repeats_var = tk.IntVar(value=5)
        self.param_vars: dict[str, tk.Variable] = {}
        self._params_holder: ttk.Frame | None = None

        self._build_layout()
        self._rebuild_param_form()
        self._poll_queue()

        # Auto-fetch live chip status (queue + ready/maintenance) right after
        # startup, then every 30 s — without ever blocking the UI.
        self.root.after(800, self._auto_refresh_status)
        self._schedule_status_refresh()

    def _build_layout(self):
        outer = ttk.Frame(self.root, padding=12, style="App.TFrame")
        outer.pack(fill="both", expand=True)
        # Two-column main area: controls on the left, chart on the right.
        # Chip + Status span both columns (full width).
        outer.columnconfigure(0, weight=0, minsize=620)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(1, weight=1)

        self._build_chip_panel(outer).grid(
            row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        self._build_controls_panel(outer).grid(
            row=1, column=0, sticky="nsew", pady=(0, 8), padx=(0, 4))
        self._build_chart_panel(outer).grid(
            row=1, column=1, sticky="nsew", pady=(0, 8), padx=(4, 0))
        self._build_status_panel(outer).grid(
            row=2, column=0, columnspan=2, sticky="ew")

    def _build_chip_panel(self, parent) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text="Chip status", padding=10)
        frame.columnconfigure(0, weight=1)

        # === Top row: chip name + Bauman-style status pills + logo ===
        top = ttk.Frame(frame)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(0, weight=1)

        name_pills = ttk.Frame(top)
        name_pills.grid(row=0, column=0, sticky="w")
        self.chip_name_lbl = ttk.Label(
            name_pills, text="Snowdrop 4q ver2", style="Title.TLabel",
        )
        self.chip_name_lbl.grid(row=0, column=0, sticky="w")

        # Status pill (Готов / Сервис / Не готов) — green / yellow / red
        self.status_pill = tk.Label(
            name_pills, text=" — ",
            fg="#888888", bg="#ECECEC",
            font=("TkDefaultFont", 9, "bold"),
            padx=14, pady=4, borderwidth=0,
        )
        self.status_pill.grid(row=0, column=1, sticky="w", padx=(14, 0))

        # Queue pill (Очередь: N) — dark slate to match Bauman site palette
        self.queue_pill = tk.Label(
            name_pills, text="Очередь: —",
            fg="#A6B1BD", bg="#1F2937",
            font=("TkDefaultFont", 9, "bold"),
            padx=14, pady=4, borderwidth=0,
        )
        self.queue_pill.grid(row=0, column=2, sticky="w", padx=(8, 0))

        # Logo (clickable — triggers refresh)
        logo_path = PROJECT_ROOT / "Bauman" / "Bauman Octillion black.png"
        self._logo_widget: tk.Widget
        try:
            from PIL import Image, ImageTk
            img = Image.open(logo_path)
            target_h = 38
            ratio = target_h / img.height
            target_w = int(img.width * ratio)
            img = img.resize((target_w, target_h), Image.LANCZOS)
            self._logo_imgtk = ImageTk.PhotoImage(img)
            logo = tk.Label(top, image=self._logo_imgtk, cursor="hand2",
                             borderwidth=0, highlightthickness=0)
            logo.bind("<Button-1>", lambda _e: self._refresh_chip_async())
            self._logo_widget = logo
        except Exception:
            self._logo_widget = ttk.Button(
                top, text="Powered by Bauman Octillion  (refresh chip)",
                command=self._refresh_chip_async,
            )
        self._logo_widget.grid(row=0, column=1, sticky="e", padx=(12, 0))

        # === Bottom row: full metrics text ===
        self.chip_metrics_lbl = ttk.Label(
            frame,
            text="Click the Bauman Octillion logo to load live calibration.",
            style="Muted.TLabel",
        )
        self.chip_metrics_lbl.grid(row=1, column=0, sticky="w", pady=(6, 0))

        return frame

    # Pill colors — match the Bauman chip-card badges:
    #   Готов        : solid bright green background, white text
    #   Не готов     : muted gray-green background, gray text
    #   Очередь: N   : very dark green background, accent-green text
    PILL_COLORS = {
        "ready":    {"fg": "#FFFFFF", "bg": PALETTE["success"]},
        "notready": {"fg": "#9AA6B5", "bg": "#212B36"},
        "unknown":  {"fg": "#9AA6B5", "bg": "#212B36"},
        "queue":    {"fg": "#34D27B", "bg": "#0B2D20"},
    }

    def _set_status_pill(self, kind: str, text: str):
        c = self.PILL_COLORS.get(kind, self.PILL_COLORS["unknown"])
        self.status_pill.configure(text=f" {text} ", fg=c["fg"], bg=c["bg"])

    def _set_queue_pill(self, n: int | None):
        c = self.PILL_COLORS["queue"]
        txt = f"Очередь: {n}" if n is not None else "Очередь: —"
        self.queue_pill.configure(text=f" {txt} ", fg=c["fg"], bg=c["bg"])

    def _build_controls_panel(self, parent) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text="Experiment", padding=10)
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Experiment:",
                  font=("TkDefaultFont", 10, "bold")).grid(row=0, column=0,
                                                            sticky="nw", pady=2)
        exp_box = ttk.Frame(frame)
        exp_box.grid(row=0, column=1, sticky="ew", pady=2)
        for i, exp in enumerate(self.experiments):
            ttk.Radiobutton(
                exp_box, text=exp.title, value=exp.key,
                variable=self.experiment_var, command=self._on_experiment_change,
            ).grid(row=i, column=0, sticky="w")

        ttk.Label(frame, text="Description:").grid(row=1, column=0,
                                                    sticky="nw", pady=(8, 2))
        self.desc_lbl = ttk.Label(frame, text=self.experiments[0].description,
                                   wraplength=900, justify="left")
        self.desc_lbl.grid(row=1, column=1, sticky="ew", pady=(8, 2))

        ttk.Label(frame, text="Parameters:",
                  font=("TkDefaultFont", 10, "bold")).grid(row=2, column=0,
                                                            sticky="nw", pady=(8, 2))
        self._params_holder = ttk.Frame(frame)
        self._params_holder.grid(row=2, column=1, sticky="ew", pady=(8, 2))

        ttk.Label(frame, text="Backend:",
                  font=("TkDefaultFont", 10, "bold")).grid(row=3, column=0,
                                                            sticky="nw", pady=(8, 2))
        be_box = ttk.Frame(frame)
        be_box.grid(row=3, column=1, sticky="ew", pady=(8, 2))
        for i, (label, value) in enumerate(BACKENDS):
            ttk.Radiobutton(be_box, text=label, value=value,
                            variable=self.backend_var).grid(row=i, column=0,
                                                            sticky="w")

        ttk.Label(frame, text="Shots:").grid(row=4, column=0, sticky="w",
                                              pady=(8, 2))
        nums_box = ttk.Frame(frame)
        nums_box.grid(row=4, column=1, sticky="ew", pady=(8, 2))
        ttk.Spinbox(nums_box, from_=4, to=14, textvariable=self.shots_exp_var,
                    width=5).grid(row=0, column=0)
        self.shots_summary = ttk.Label(nums_box, text="(2^10 = 1024)")
        self.shots_summary.grid(row=0, column=1, padx=(8, 16))
        self.shots_exp_var.trace_add("write", self._update_shots_summary)

        ttk.Label(nums_box, text="Repeats (for Benchmark):").grid(row=0, column=2)
        ttk.Spinbox(nums_box, from_=2, to=50, textvariable=self.repeats_var,
                    width=5).grid(row=0, column=3, padx=(8, 0))

        btn_box = ttk.Frame(frame)
        btn_box.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(14, 0))
        btn_box.columnconfigure(0, weight=1, uniform="btn")
        btn_box.columnconfigure(1, weight=1, uniform="btn")
        pad = {"sticky": "ew", "pady": 3}

        # --- row 0-2: run / benchmark actions -------------------------------
        self.run_btn = ttk.Button(btn_box, text="▶  Run selected experiment",
                                  style="Accent.TButton",
                                  command=self._run_single)
        self.run_btn.grid(row=0, column=0, padx=(0, 4), **pad)
        self.run_all_tests_btn = ttk.Button(
            btn_box, text="▶▶  Run all 4 tests on selected backend",
            command=self._run_all_tests,
        )
        self.run_all_tests_btn.grid(row=0, column=1, padx=(4, 0), **pad)

        self.run_all_btn = ttk.Button(btn_box,
                                      text="⇄  Compare on 3 backends",
                                      command=self._run_three_way)
        self.run_all_btn.grid(row=1, column=0, padx=(0, 4), **pad)
        self.bench_btn = ttk.Button(btn_box,
                                    text="📊  Benchmark this experiment (N×)",
                                    command=self._run_benchmark)
        self.bench_btn.grid(row=1, column=1, padx=(4, 0), **pad)

        self.bench_all_btn = ttk.Button(
            btn_box, text="📊  Benchmark all 4 experiments (N×)",
            command=self._run_benchmark_all,
        )
        self.bench_all_btn.grid(row=2, column=0, padx=(0, 4), **pad)
        self.full_btn = ttk.Button(
            btn_box,
            text="🔬  Full benchmark suite → results/",
            style="Accent.TButton",
            command=self._run_full_suite,
        )
        self.full_btn.grid(row=2, column=1, padx=(4, 0), **pad)

        ttk.Separator(btn_box, orient="horizontal").grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=(10, 7))

        # --- row 4-6: analysis tools ----------------------------------------
        self.scaling_btn = ttk.Button(
            btn_box,
            text="📈  GPU scaling demo (GHZ-N up to 25)",
            command=self._run_scaling_demo,
        )
        self.scaling_btn.grid(row=4, column=0, padx=(0, 4), **pad)

        self.crm_btn = ttk.Button(
            btn_box,
            text="🔐  CRM forecast (chip → Shor capability)",
            command=self._run_crm_forecast,
        )
        self.crm_btn.grid(row=4, column=1, padx=(4, 0), **pad)

        self.calc_btn = ttk.Button(
            btn_box,
            text="🧮  SCI calculator (paper formula + SCI_HW)",
            command=self._open_sci_calculator,
        )
        self.calc_btn.grid(row=5, column=0, padx=(0, 4), **pad)

        self.pqc_btn = ttk.Button(
            btn_box,
            text="📦  PQC benchmark suite (Kyber · Dilithium · Falcon …)",
            command=self._run_pqc_suite,
        )
        self.pqc_btn.grid(row=5, column=1, padx=(4, 0), **pad)

        # Thesis-defence toolkit: Q-resource estimator + dynamic SCI + Sobol +
        # HNDL risk + TLS hybrid handshake — all in one dialog so the committee
        # can poke at every formula referenced in the paper.
        self.thesis_btn = ttk.Button(
            btn_box,
            text="📚  Thesis tools (Q-resources · Dynamic SCI · HNDL · TLS hybrid)",
            command=self._open_thesis_tools,
        )
        self.thesis_btn.grid(row=6, column=0, columnspan=2, **pad)

        # ⏹ Stop button — always active. Cancels the running job (if any)
        # and/or any user-entered Bauman job ID via client.cancel().
        self.stop_btn = ttk.Button(
            btn_box,
            text="⏹  Stop / Cancel Bauman job (running or by ID)",
            style="Danger.TButton",
            command=self._on_stop,
        )
        self.stop_btn.grid(row=7, column=0, columnspan=2,
                            sticky="ew", pady=(10, 0))

        return frame

    def _rebuild_param_form(self):
        for w in self._params_holder.winfo_children():
            w.destroy()
        self.param_vars.clear()

        exp = self.experiment_by_key[self.experiment_var.get()]
        if not exp.parameters:
            ttk.Label(self._params_holder, text="(no parameters)",
                      foreground="#888").grid(row=0, column=0, sticky="w")
            return

        for i, p in enumerate(exp.parameters):
            ttk.Label(self._params_holder, text=p.label + ":").grid(
                row=i, column=0, sticky="w", padx=(0, 8), pady=2
            )
            self._make_param_widget(p, parent=self._params_holder, row=i)

    def _make_param_widget(self, p: ParameterSpec, parent, row: int):
        if p.kind == "choice":
            var = tk.StringVar(value=str(p.default))
            self.param_vars[p.name] = var
            cb = ttk.Combobox(parent, textvariable=var,
                              values=[str(c) for c in p.choices],
                              state="readonly", width=14)
            cb.grid(row=row, column=1, sticky="w", pady=2)
        elif p.kind == "int":
            var = tk.IntVar(value=int(p.default))
            self.param_vars[p.name] = var
            sb_kwargs: dict[str, Any] = {"textvariable": var, "width": 8}
            if p.min_value is not None:
                sb_kwargs["from_"] = p.min_value
            if p.max_value is not None:
                sb_kwargs["to"] = p.max_value
            ttk.Spinbox(parent, **sb_kwargs).grid(row=row, column=1,
                                                   sticky="w", pady=2)
        elif p.kind == "bool":
            var = tk.BooleanVar(value=bool(p.default))
            self.param_vars[p.name] = var
            ttk.Checkbutton(parent, variable=var).grid(row=row, column=1,
                                                        sticky="w", pady=2)
        else:
            var = tk.StringVar(value=str(p.default))
            self.param_vars[p.name] = var
            ttk.Entry(parent, textvariable=var, width=20).grid(
                row=row, column=1, sticky="w", pady=2
            )

        if p.help:
            ttk.Label(parent, text=p.help, foreground="#666",
                      font=("TkDefaultFont", 8)).grid(
                row=row, column=2, sticky="w", padx=(8, 0), pady=2
            )

    def _collect_params(self) -> dict:
        exp = self.experiment_by_key[self.experiment_var.get()]
        params: dict[str, Any] = {}
        for p in exp.parameters:
            var = self.param_vars.get(p.name)
            raw = var.get() if var is not None else p.default
            params[p.name] = p.coerce(raw)
        return params

    def _build_chart_panel(self, parent) -> ttk.LabelFrame:
        chart_frame = ttk.LabelFrame(parent, text="Results", padding=6)
        chart_frame.columnconfigure(0, weight=1)
        chart_frame.rowconfigure(1, weight=1)

        # Live CSV preview bar: while a pipeline runs, every CSV it writes
        # shows up in this picker and the newest one is re-drawn natively
        # on the canvas below (vector-crisp, unlike the old PNG previews).
        bar = ttk.Frame(chart_frame)
        bar.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        bar.columnconfigure(1, weight=1)
        ttk.Label(bar, text="Live CSV preview:",
                  style="Heading.TLabel").grid(row=0, column=0, sticky="w")
        self.preview_combo = ttk.Combobox(bar, state="readonly", values=())
        self.preview_combo.grid(row=0, column=1, sticky="ew", padx=8)
        self.preview_combo.bind("<<ComboboxSelected>>", self._on_preview_pick)
        self.follow_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Follow latest",
                        variable=self.follow_var).grid(row=0, column=2,
                                                       padx=(0, 8))
        self.open_folder_btn = ttk.Button(bar, text="📂 Open results folder",
                                          command=self._open_results_folder,
                                          state="disabled")
        self.open_folder_btn.grid(row=0, column=3, sticky="e")

        # Figsize calibrated for the right-column slot (≈ half the window
        # width after chip+status grow). Matplotlib re-fits on Tk resize.
        self.fig = Figure(figsize=(8, 5.5), tight_layout=True)
        self._draw_empty_chart()

        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        toolbar_holder = ttk.Frame(chart_frame)
        toolbar_holder.grid(row=2, column=0, sticky="ew")
        NavigationToolbar2Tk(self.canvas, toolbar_holder)

        self.metric_lbl = ttk.Label(chart_frame, text="",
                                    font=("TkDefaultFont", 11, "bold"))
        self.metric_lbl.grid(row=3, column=0, sticky="w", pady=(4, 0))

        self.verdict_lbl = ttk.Label(chart_frame, text="",
                                      foreground=PALETTE["success"],
                                      font=("TkDefaultFont", 10, "bold"),
                                      wraplength=900, justify="left")
        self.verdict_lbl.grid(row=4, column=0, sticky="w", pady=(0, 4))

        save_btn = ttk.Button(chart_frame, text="Save report (JSON)…",
                              command=self._save_report)
        save_btn.grid(row=5, column=0, sticky="e")

        return chart_frame

    # ---------- live CSV preview ----------

    RESULTS_ROOT = PROJECT_ROOT / "results"

    def _start_preview_watch(self):
        """Follow CSV artifacts of the run that just started.

        The poller lives on the Tk main loop and only does cheap mtime
        scans; it stops by itself once the worker thread exits.
        """
        self._preview_t0 = time.time()
        # A new run is an unambiguous signal that following is wanted
        # again, even if the user pinned a CSV manually last time.
        self.follow_var.set(True)
        if not self._preview_polling:
            self._preview_polling = True
            self.root.after(1500, self._poll_preview)

    def _poll_preview(self):
        # Liveness FIRST: once the worker exits, the *_done handler picks
        # the headline chart via _finish_preview, and the poller's last
        # tick must not override it with whatever CSV happens to be newest.
        busy = bool(self.worker and self.worker.is_alive())
        try:
            fresh = self._register_csvs()
            if busy and fresh and self.follow_var.get():
                self._render_csv_preview(fresh[-1])
        except tk.TclError:
            busy = False  # widgets are being torn down
        if busy:
            self.root.after(1500, self._poll_preview)
        else:
            self._preview_polling = False

    def _register_csvs(self) -> list[Path]:
        """Scan results/ for fresh CSVs and add them to the picker."""
        if not self._preview_t0:
            return []
        fresh = scan_new_csvs(self.RESULTS_ROOT, self._preview_t0)
        new_names = False
        for p in fresh:
            name = f"{p.parent.name} / {p.name}"
            if name not in self._preview_paths:
                new_names = True
            self._preview_paths[name] = p
        if new_names:
            self.preview_combo["values"] = list(self._preview_paths)
        return fresh

    def _render_csv_preview(self, path: Path, announce: bool = False) -> bool:
        path = Path(path)
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return False
        state = self._preview_state
        if not announce and state and state[:2] == (path, mtime):
            return state[2]
        try:
            render_csv(self.fig, path)
            self.canvas.draw_idle()
            self._preview_state = (path, mtime, True)
            name = f"{path.parent.name} / {path.name}"
            if name in self._preview_paths:
                self.preview_combo.set(name)
            return True
        except Exception as e:
            # Half-written files are normal while a pipeline is mid-flight;
            # the next poll retries as soon as the file's mtime moves on.
            self._preview_state = (path, mtime, False)
            if announce:
                self._log(f"(preview {path.name}: {e!r})")
            return False

    def _on_preview_pick(self, _event=None):
        path = self._preview_paths.get(self.preview_combo.get())
        if path is None:
            return
        # A manual pick wins over auto-follow, otherwise the next poll
        # would overwrite the user's choice within two seconds.
        self.follow_var.set(False)
        self._render_csv_preview(path, announce=True)

    def _set_out_dir(self, d):
        if not d:
            return
        self._last_out_dir = Path(d)
        self.open_folder_btn.configure(state="normal")

    def _open_results_folder(self):
        if self._last_out_dir and self._last_out_dir.exists():
            import os
            # AttributeError: os.startfile is Windows-only
            with suppress(OSError, AttributeError):
                os.startfile(str(self._last_out_dir))
        else:
            self._log("No results folder yet — run a pipeline first.")

    def _finish_preview(self, csv_path, fallback_png=None, title=""):
        """Final render once a pipeline completes: prefer the native CSV
        preview, fall back to the pipeline's saved PNG chart."""
        self._register_csvs()
        if csv_path and Path(csv_path).exists() \
                and self._render_csv_preview(csv_path, announce=True):
            return
        if fallback_png and Path(fallback_png).exists():
            try:
                self.fig.clear()
                ax = self.fig.add_subplot(111)
                img = matplotlib.image.imread(str(fallback_png))
                ax.imshow(img)
                ax.axis("off")
                if title:
                    ax.set_title(title)
                self.canvas.draw_idle()
            except Exception as e:
                self._log(f"(chart preview failed: {e!r})")

    def _build_status_panel(self, parent) -> ttk.LabelFrame:
        status_frame = ttk.LabelFrame(parent, text="Status", padding=6)
        status_frame.columnconfigure(0, weight=1)

        # Header row with Copy button on the right of the "Status" title area
        header = ttk.Frame(status_frame)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        header.columnconfigure(0, weight=1)
        self._status_copy_state = ttk.Label(header, text="",
                                              foreground=PALETTE["success"],
                                              font=("TkDefaultFont", 9))
        self._status_copy_state.grid(row=0, column=0, sticky="e", padx=(0, 6))
        copy_btn = ttk.Button(header, text="📋 Copy",
                               command=self._copy_status_log, width=10)
        copy_btn.grid(row=0, column=1, sticky="e")

        self.status_text = tk.Text(status_frame, height=8, wrap="none")
        style_console(self.status_text)
        self.status_text.grid(row=1, column=0, sticky="ew")
        self.status_text.configure(state="disabled")

        # === Two progress bars: overall + per-task phase ===
        bars_frame = ttk.Frame(status_frame)
        bars_frame.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        bars_frame.columnconfigure(1, weight=1)

        ttk.Label(bars_frame, text="Overall:", width=8).grid(
            row=0, column=0, sticky="w")
        self.overall_progress = ttk.Progressbar(bars_frame, mode="determinate")
        self.overall_progress.grid(row=0, column=1, sticky="ew", padx=(6, 6))
        self.overall_label = ttk.Label(bars_frame, text="idle", width=18,
                                          anchor="e",
                                          font=("Consolas", 9))
        self.overall_label.grid(row=0, column=2, sticky="e")

        ttk.Label(bars_frame, text="Task:", width=8).grid(
            row=1, column=0, sticky="w", pady=(2, 0))
        self.task_progress = ttk.Progressbar(bars_frame, mode="determinate",
                                                maximum=6)
        self.task_progress.grid(row=1, column=1, sticky="ew",
                                  padx=(6, 6), pady=(2, 0))
        self.task_label = ttk.Label(bars_frame, text="—", width=28,
                                       anchor="e",
                                       font=("Consolas", 9))
        self.task_label.grid(row=1, column=2, sticky="e", pady=(2, 0))

        self._overall_max = 0
        self._overall_done = 0
        return status_frame

    def _copy_status_log(self):
        """Copy the full Status text feed to the clipboard."""
        text = self.status_text.get("1.0", "end").rstrip()
        if not text:
            self._status_copy_state.config(text="(nothing to copy)",
                                             foreground="#888")
            self.root.after(2000, lambda: self._status_copy_state.config(text=""))
            return
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            # Force clipboard sync to OS clipboard (Windows otherwise drops it
            # when the Tk app exits).
            self.root.update()
            n_lines = text.count("\n") + 1
            self._status_copy_state.config(
                text=f"copied {n_lines} lines ({len(text)} chars) ✓",
                foreground=PALETTE["success"])
            self.root.after(3000, lambda: self._status_copy_state.config(text=""))
        except tk.TclError as e:
            self._status_copy_state.config(text=f"copy failed: {e}",
                                             foreground="#a00")
            self.root.after(4000, lambda: self._status_copy_state.config(text=""))

    def _draw_empty_chart(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_xlabel("Outcome")
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1)
        ax.grid(True, axis="y", alpha=0.3)
        ax.text(0.5, 0.55, "Run an experiment to see results",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=15, color=PALETTE["text_muted"])
        ax.text(0.5, 0.42, "pick an experiment + backend on the left",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color=PALETTE["text_muted"])

    def _on_experiment_change(self):
        exp = self.experiment_by_key[self.experiment_var.get()]
        self.desc_lbl.config(text=exp.description)
        self._rebuild_param_form()

    def _update_shots_summary(self, *_):
        try:
            n = int(self.shots_exp_var.get())
            self.shots_summary.config(text=f"(2^{n} = {2 ** n})")
        except (ValueError, tk.TclError):
            self.shots_summary.config(text="(?)")

    def _log(self, msg: str):
        self.status_text.configure(state="normal")
        self.status_text.insert("end", msg + "\n")
        self.status_text.see("end")
        self.status_text.configure(state="disabled")
        # Drive the per-task progress bar from the live status feed
        phase = self._detect_phase(msg)
        if phase is not None:
            self._set_task_phase(phase)

    def _set_running(self, is_running: bool):
        state = "disabled" if is_running else "normal"
        self.run_btn.configure(state=state)
        self.run_all_btn.configure(state=state)
        self.bench_btn.configure(state=state)
        self.bench_all_btn.configure(state=state)
        self.full_btn.configure(state=state)
        self.scaling_btn.configure(state=state)
        self.crm_btn.configure(state=state)
        self.calc_btn.configure(state=state)
        self.pqc_btn.configure(state=state)
        # Stop button is ALWAYS enabled — it can cancel by ID even when idle
        self.stop_btn.configure(state="normal")
        self.run_all_tests_btn.configure(state=state)
        if is_running:
            with suppress(Exception):
                from src.experiments.runner import reset_cancel
                reset_cancel()
            self._start_preview_watch()
        else:
            self._stop_overall()

    def _on_stop(self):
        """
        Always-on Stop dialog. Lets the user:
          1. Cancel the currently in-flight job (if a worker is running).
          2. Cancel any arbitrary Bauman job by typing its UUID — useful for
             jobs left over from previous sessions or kicked off through the
             Bauman web UI.
        """
        from src.experiments.runner import (
            request_cancel, is_cancel_requested, _current_remote_job,
        )

        dlg = tk.Toplevel(self.root)
        dlg.title("Stop / Cancel Bauman job")
        dlg.transient(self.root)
        dlg.grab_set()
        dlg.geometry("560x310")

        outer = ttk.Frame(dlg, padding=12)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)

        # === Section 1: cancel current in-flight job ===
        cur = ttk.LabelFrame(outer, text="Currently running", padding=10)
        cur.grid(row=0, column=0, sticky="ew")
        cur.columnconfigure(0, weight=1)

        in_flight = _current_remote_job is not None
        cur_id = _current_remote_job[1] if in_flight else None
        if in_flight:
            cur_lbl_text = f"In-flight job: {cur_id}"
            cur_lbl_color = "#0a6b00"
        elif self._worker_busy(silent=True):
            cur_lbl_text = ("Worker thread is running but no Bauman job is "
                             "in queue right now (local-only run, e.g. ideal/gpu).")
            cur_lbl_color = "#666666"
        else:
            cur_lbl_text = "No job is currently running."
            cur_lbl_color = "#888888"

        ttk.Label(cur, text=cur_lbl_text, foreground=cur_lbl_color,
                  wraplength=520).grid(row=0, column=0, sticky="w")

        def cancel_current():
            if is_cancel_requested():
                self._log("⏹  STOP already requested — waiting for worker...")
                dlg.destroy()
                return
            self._log("⏹  STOP requested — canceling current job and "
                       "breaking out of benchmark loop...")
            cancelled_remote = request_cancel()
            if cancelled_remote:
                self._log(f"    server-side: Bauman job {cur_id} CANCELED via API")
            else:
                self._log("    no in-flight Bauman job to cancel; flag set, "
                           "loop will exit at next cooperative check")
            dlg.destroy()

        cur_btn = ttk.Button(cur, text="⏹  Stop current job",
                              command=cancel_current,
                              state=("normal" if (in_flight or self._worker_busy(silent=True))
                                     else "disabled"))
        cur_btn.grid(row=1, column=0, sticky="w", pady=(8, 0))

        # === Section 2: cancel by arbitrary ID ===
        by_id = ttk.LabelFrame(outer, text="Cancel any Bauman job by ID",
                                padding=10)
        by_id.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        by_id.columnconfigure(0, weight=1)

        ttk.Label(by_id, text="Job UUID (from octillion.bmstu.ru/runs):",
                  foreground="#444").grid(row=0, column=0, sticky="w")

        id_var = tk.StringVar()
        entry = ttk.Entry(by_id, textvariable=id_var, font=("Consolas", 9))
        entry.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        entry.focus_set()

        # Try clipboard auto-paste — convenient flow:
        # user copies UUID from Bauman web UI, opens dialog, it's pre-filled.
        # Empty/non-text clipboard raises TclError — simply skip pre-fill.
        with suppress(tk.TclError):
            clip = self.root.clipboard_get().strip()
            if 30 < len(clip) < 50 and clip.count("-") == 4:
                id_var.set(clip)
                ttk.Label(by_id, text="(auto-pasted from clipboard)",
                          foreground="#0a6b00",
                          font=("TkDefaultFont", 8)).grid(
                    row=2, column=0, sticky="w", pady=(2, 0))

        def cancel_by_id():
            jid = id_var.get().strip()
            if not jid:
                self._log("    cancel-by-id: no ID entered")
                return

            self._log(f"⏹  Sending DELETE /v2/job/{jid} to Bauman...")
            try:
                from src.quantum_hardware import get_client
                client = get_client()
                # Probe status first
                try:
                    before = client._api.job(jid)
                    self._log(f"    before: status={before.get('status')}")
                except Exception as e:
                    self._log(f"    pre-check failed: {e!r}")
                # Cancel
                try:
                    res = client.cancel(jid)
                    self._log(f"    cancel returned: status={res.status}")
                except Exception as e:
                    self._log(f"    cancel failed: {e!r}")
                    return
                # Verify
                try:
                    after = client._api.job(jid)
                    self._log(f"    after:  status={after.get('status')}")
                except Exception as e:
                    self._log(f"    post-check failed: {e!r}")
            finally:
                dlg.destroy()

        ttk.Button(by_id, text="⏹  Cancel this ID",
                    command=cancel_by_id).grid(row=3, column=0,
                                                  sticky="e", pady=(8, 0))

        # === Bottom: close ===
        ttk.Button(outer, text="Close",
                    command=dlg.destroy).grid(row=2, column=0,
                                                  sticky="e", pady=(10, 0))

        # Allow Enter to submit by-ID form
        entry.bind("<Return>", lambda _e: cancel_by_id())

    # ---------- progress helpers ----------

    PHASE_ORDER = ["setup", "transpile", "submit", "queue",
                    "executing", "post"]
    PHASE_LABELS = {
        "setup":     "1/6  setup (Python)",
        "transpile": "2/6  transpile (Qiskit)",
        "submit":    "3/6  submit (network)",
        "queue":     "4/6  queue (Bauman)",
        "executing": "5/6  executing",
        "post":      "6/6  post-processing",
    }

    def _start_overall(self, total: int = 0, label_prefix: str = "running"):
        if total > 0:
            self.overall_progress.config(mode="determinate",
                                            maximum=total, value=0)
            self._overall_max = total
            self._overall_done = 0
            self.overall_label.config(text=f"0/{total}")
        else:
            self.overall_progress.config(mode="indeterminate")
            self.overall_progress.start(10)
            self._overall_max = 0
            self._overall_done = 0
            self.overall_label.config(text=label_prefix)
        self.task_progress.config(value=0)
        self.task_label.config(text="—")

    def _step_overall(self, increment: int = 1):
        if self._overall_max > 0:
            self._overall_done = min(self._overall_done + increment,
                                       self._overall_max)
            self.overall_progress.config(value=self._overall_done)
            self.overall_label.config(
                text=f"{self._overall_done}/{self._overall_max}"
            )

    def _stop_overall(self):
        with suppress(tk.TclError):
            self.overall_progress.stop()
        self.overall_progress.config(mode="determinate", maximum=100, value=0)
        self.overall_label.config(text="idle")
        self.task_progress.config(value=0)
        self.task_label.config(text="—")
        self._overall_max = 0
        self._overall_done = 0

    def _set_task_phase(self, phase: str | None):
        if phase is None:
            self.task_progress.config(value=0)
            self.task_label.config(text="—")
            return
        if phase not in self.PHASE_ORDER:
            return
        idx = self.PHASE_ORDER.index(phase) + 1
        self.task_progress.config(value=idx)
        self.task_label.config(text=self.PHASE_LABELS[phase])

    @staticmethod
    def _detect_phase(msg: str) -> str | None:
        m = msg.lower()
        if "status=complete" in m or m.startswith("benchmark: run") and "→" in m:
            return "post"
        if "status=executing" in m:
            return "executing"
        if "status=queue" in m:
            return "queue"
        if "waiting for job" in m:
            return "queue"
        if "submitting" in m:
            return "submit"
        if "transpil" in m:
            return "transpile"
        if "running" in m and "shots" in m:
            return "executing"
        if "connecting" in m or "querying" in m:
            return "setup"
        return None

    def _auto_refresh_status(self):
        """Lightweight queue/maintenance check (does NOT call describe_chip).
        Runs in background; updates only the two pills near the logo.

        While a benchmark worker is in flight on the real chip, we SKIP the
        API ping entirely — Bauman's chip-info GET adds load while the chip
        is running our jobs, and the pill state will refresh as soon as the
        worker finishes anyway. Silent skip (no log entry) so the chip log
        stays clean during long benchmarks.
        """
        if self._worker_busy(silent=True):
            return

        def task():
            # Silent on failure: pills keep their last known state and the
            # next scheduled ping retries anyway.
            with suppress(Exception):
                from src.quantum_hardware import get_client, get_chip_queue_info
                client = get_client()
                qinfo = get_chip_queue_info(client, "Snowdrop 4q ver2")
                self.msg_queue.put(("status_ping", qinfo))

        threading.Thread(target=task, daemon=True).start()

    def _schedule_status_refresh(self):
        # Re-poll every 60s. During a real-hw benchmark _auto_refresh_status
        # silently skips, so this just paces background pings when the GUI
        # is idle. Increased from 30s → 60s after observing that 30s ticks
        # interleave with chip-side calibration cycles on busy weeks.
        self.root.after(60_000, self._auto_refresh_status_then_reschedule)

    def _auto_refresh_status_then_reschedule(self):
        self._auto_refresh_status()
        self._schedule_status_refresh()

    def _refresh_chip_async(self):
        if self._worker_busy():
            return
        self._set_running(True)
        self._start_overall(total=0, label_prefix="querying chip")
        self._log("=== Querying chip status... ===")

        def task():
            try:
                client = get_client()
                backend = get_backend(client, real_hardware=True)
                spec = describe_chip(backend)
                # Best-effort queue/status snapshot
                from src.quantum_hardware import get_chip_queue_info
                qinfo = get_chip_queue_info(client, "Snowdrop 4q ver2")
                self.msg_queue.put(("chip_spec", (spec, qinfo)))
            except Exception as e:
                self.msg_queue.put(("error", f"Chip status query failed: {e!r}"))
            finally:
                self.msg_queue.put(("idle", None))

        self.worker = threading.Thread(target=task, daemon=True)
        self.worker.start()

    def _worker_busy(self, silent: bool = False) -> bool:
        """Returns True if a benchmark worker is in flight.

        `silent=True` is for background callers (auto-refresh pills, etc.):
        they need the answer but should NOT spam the status log. User-initiated
        clicks should pass silent=False (default) so we log a visible reason
        for why the click was ignored.
        """
        if self.worker and self.worker.is_alive():
            if not silent:
                self._log("⚠ Another job is already running — request ignored.")
            return True
        return False

    def _run_single(self):
        if self._worker_busy():
            return
        exp = self.experiment_by_key[self.experiment_var.get()]
        backend = self.backend_var.get()
        shots_exp = int(self.shots_exp_var.get())
        params = self._collect_params()

        self._log(f"\n=== {exp.title_with_params(params)} on {backend} ===")
        self._set_running(True)
        self._start_overall(total=1)
        self._kick_off_runs([(exp, backend, params)], shots_exp, mode="single")

    def _run_all_tests(self):
        """One-click: run all 4 experiments on the currently selected backend."""
        if self._worker_busy():
            return
        backend = self.backend_var.get()
        shots_exp = int(self.shots_exp_var.get())

        # Default params for each experiment
        defaults = {
            "bell":  {},
            "ghz":   {"n": 3},
            "bv":    {"secret": "101"},
            "shor":  {"a": 4},
        }
        jobs: list[tuple[ExperimentDef, str, dict]] = []
        for exp in self.experiments:
            params = defaults.get(exp.key, exp.default_params())
            jobs.append((exp, backend, params))

        self._log(f"\n=== ▶▶▶ ALL 4 TESTS on {backend} ===")
        self._set_running(True)
        self._kick_off_runs(jobs, shots_exp, mode="all_tests")

    def _run_three_way(self):
        if self._worker_busy():
            return
        exp = self.experiment_by_key[self.experiment_var.get()]
        shots_exp = int(self.shots_exp_var.get())
        params = self._collect_params()

        self._log(f"\n=== {exp.title_with_params(params)} — 3-WAY COMPARISON ===")
        self._set_running(True)
        self._start_overall(total=3)
        self._kick_off_runs([
            (exp, "ideal", params),
            (exp, "emulator", params),
            (exp, "real", params),
        ], shots_exp, mode="three_way")

    def _run_benchmark(self):
        if self._worker_busy():
            return
        exp = self.experiment_by_key[self.experiment_var.get()]
        backend = self.backend_var.get()
        shots_exp = int(self.shots_exp_var.get())
        params = self._collect_params()
        repeats = int(self.repeats_var.get())

        if backend == "real" and repeats > 10:
            ok = messagebox.askyesno(
                "Long benchmark",
                f"You're about to submit {repeats} jobs to real hardware.\n"
                f"Each job has queue + execute time. Proceed?"
            )
            if not ok:
                return

        self._log(f"\n=== BENCHMARK: {exp.title_with_params(params)} on "
                  f"{backend} × {repeats} repeats ===")
        self._set_running(True)
        self._start_overall(total=repeats)
        self._kick_off_benchmark(exp, backend, params, shots_exp, repeats)

    def _kick_off_runs(self, jobs, shots_exp, mode: str):
        results: dict[str, ExperimentResult] = {}

        def task():
            try:
                for exp, backend, params in jobs:
                    qc = exp.build(params)
                    res = run_circuit(
                        qc, label=exp.key,
                        backend_kind=backend,
                        shots_exponent=shots_exp,
                        expected_distribution=exp.expected(params),
                        metric_name=exp.metric_name,
                        metric_fn=lambda c, _e=exp, _p=params: _e.metric_fn(c, _p),
                        status=lambda m: self.msg_queue.put(("log", m)),
                    )
                    self.msg_queue.put(("partial_result", (exp, backend, params, res)))
                    results[f"{exp.key}_{backend}"] = res
                self.msg_queue.put(("all_done", (jobs, results, mode)))
            except Exception as e:
                import traceback
                self.msg_queue.put(("error", f"{e!r}\n{traceback.format_exc()}"))
            finally:
                self.msg_queue.put(("idle", None))

        self.worker = threading.Thread(target=task, daemon=True)
        self.worker.start()

    def _open_sci_calculator(self):
        from src.sci_calculator_gui import SCICalculatorWindow
        win = tk.Toplevel(self.root)
        win.title("SCI Calculator")
        win.geometry("960x720")
        win.transient(self.root)
        SCICalculatorWindow(win, embedded=True)

    def _open_thesis_tools(self):
        from src.thesis_tools_gui import ThesisToolsWindow
        win = tk.Toplevel(self.root)
        win.title("Thesis Tools — Q-resources · Dynamic SCI · HNDL · TLS hybrid")
        win.geometry("1100x780")
        win.transient(self.root)
        ThesisToolsWindow(win, embedded=True)

    def _run_pqc_suite(self):
        if self._worker_busy():
            return
        self._log("\n=== PQC benchmark suite (Kyber/Dilithium/Classical) ===")
        self._set_running(True)
        self._start_overall(total=0, label_prefix="PQC pipeline")

        def task():
            try:
                from src.pqc_pipeline import run_pqc_pipeline
                paths = run_pqc_pipeline(
                    status=lambda m: self.msg_queue.put(("log", m)),
                )
                self.msg_queue.put(("pqc_done", paths))
            except Exception as e:
                import traceback
                self.msg_queue.put(("error", f"{e!r}\n{traceback.format_exc()}"))
            finally:
                self.msg_queue.put(("idle", None))

        self.worker = threading.Thread(target=task, daemon=True)
        self.worker.start()

    def _run_crm_forecast(self):
        if self._worker_busy():
            return

        self._log("\n=== CRM forecast: Snowdrop + 10 published chips ===")
        self._set_running(True)
        self._start_overall(total=0, label_prefix="CRM forecast")

        def task():
            try:
                from src.crm_forecast import build_artifacts as build_crm
                from src.quantum_hardware import (
                    get_client, get_backend, describe_chip,
                )
                client = get_client()
                be = get_backend(client, real_hardware=True)
                spec = describe_chip(be)
                measured = {
                    "n_qubits": spec.num_qubits,
                    "avg_f1q": spec.avg_f1q,
                    "avg_f2q": spec.avg_f2q,
                    "avg_ro":  spec.avg_ro,
                }
                paths = build_crm(
                    measured,
                    status=lambda m: self.msg_queue.put(("log", m)),
                )
                self.msg_queue.put(("crm_done", paths))
            except Exception as e:
                import traceback
                self.msg_queue.put(("error", f"{e!r}\n{traceback.format_exc()}"))
            finally:
                self.msg_queue.put(("idle", None))

        self.worker = threading.Thread(target=task, daemon=True)
        self.worker.start()

    def _run_scaling_demo(self):
        if self._worker_busy():
            return

        n_max = self._ask_simple_int(
            "GPU scaling demo",
            "Max N qubits for GHZ-N scaling (CPU runs up to 22, GPU up to 25):",
            default=22, lo=4, hi=28,
        )
        if n_max is None:
            return

        self._log(f"\n=== GPU SCALING DEMO: GHZ-N for N=2..{n_max} ===")
        self._set_running(True)
        self._start_overall(total=2 * (n_max - 1),
                              label_prefix="GHZ-N scaling")

        def task():
            try:
                from src.scaling_demo import run_scaling
                paths = run_scaling(
                    n_range=range(2, n_max + 1),
                    shots=1024,
                    do_cpu=True,
                    do_gpu=True,
                    status=lambda m: self.msg_queue.put(("log", m)),
                )
                self.msg_queue.put(("scaling_done", paths))
            except Exception as e:
                import traceback
                self.msg_queue.put(("error", f"{e!r}\n{traceback.format_exc()}"))
            finally:
                self.msg_queue.put(("idle", None))

        self.worker = threading.Thread(target=task, daemon=True)
        self.worker.start()

    def _ask_simple_int(self, title: str, prompt: str,
                          default: int, lo: int, hi: int) -> int | None:
        dlg = tk.Toplevel(self.root)
        dlg.title(title)
        dlg.transient(self.root)
        dlg.grab_set()
        dlg.geometry("420x130")

        ttk.Label(dlg, text=prompt, wraplength=380).pack(padx=12, pady=(12, 6))
        var = tk.IntVar(value=default)
        ttk.Spinbox(dlg, from_=lo, to=hi, textvariable=var, width=8).pack(pady=4)

        result: dict[str, int] = {}
        bb = ttk.Frame(dlg)
        bb.pack(side="bottom", pady=8)
        ttk.Button(bb, text="Cancel", command=dlg.destroy).pack(side="right", padx=4)
        def ok():
            result["v"] = int(var.get()); dlg.destroy()
        ttk.Button(bb, text="Run", command=ok).pack(side="right", padx=4)
        self.root.wait_window(dlg)
        return result.get("v")

    def _run_benchmark_all(self):
        """Statistical benchmark of ALL 4 experiments on the current backend."""
        if self._worker_busy():
            return
        backend = self.backend_var.get()
        shots_exp = int(self.shots_exp_var.get())
        repeats = int(self.repeats_var.get())

        if backend == "real" and repeats > 5:
            ok = messagebox.askyesno(
                "Long benchmark",
                f"You're about to submit {repeats} × 4 = {repeats * 4} jobs "
                f"to real hardware. Each carries queue + execute time. "
                f"Proceed?"
            )
            if not ok:
                return

        defaults = {
            "bell":  {},
            "ghz":   {"n": 3},
            "bv":    {"secret": "101"},
            "shor":  {"a": 4},
        }

        self._log(f"\n=== 📊📊 BENCHMARK ALL 4 × {repeats} on {backend} ===")
        self._set_running(True)

        def task():
            from src.experiments.runner import benchmark
            results: dict[str, BenchmarkResult] = {}
            try:
                for exp in self.experiments:
                    params = defaults.get(exp.key, exp.default_params())
                    self.msg_queue.put(("log", f"\n--- {exp.title} ---"))
                    bench = benchmark(
                        exp,
                        backend_kind=backend,
                        params=params,
                        shots_exponent=shots_exp,
                        repeats=repeats,
                        status=lambda m: self.msg_queue.put(("log", f"  {m}")),
                    )
                    results[exp.key] = bench
                    self.msg_queue.put(("bench_done", (exp, params, bench)))
                self.msg_queue.put(("log",
                    f"\n=== ALL DONE — {len(results)} experiments × "
                    f"{repeats} repeats on {backend} ==="))
            except Exception as e:
                import traceback
                self.msg_queue.put(("error",
                    f"{e!r}\n{traceback.format_exc()}"))
            finally:
                self.msg_queue.put(("idle", None))

        self.worker = threading.Thread(target=task, daemon=True)
        self.worker.start()

    def _run_full_suite(self):
        if self._worker_busy():
            return
        cfg = self._ask_full_suite_config()
        if cfg is None:
            return
        backends, repeats, shots_exp, extras, real_repeats, force_env = cfg

        # Estimate total run count for the overall progress bar.
        # When real_repeats is overridden, real-hw configs use that count
        # while other backends still use the main repeats value.
        from src.full_benchmark import build_default_plan
        plan_size = len(build_default_plan(extras=extras))
        has_real = "real" in backends
        n_local_configs = plan_size * (len(backends) - (1 if has_real else 0))
        total_runs = n_local_configs * repeats
        if has_real:
            total_runs += plan_size * (real_repeats if real_repeats is not None else repeats)

        real_repeats_log = (f", real_repeats={real_repeats}"
                            if real_repeats is not None else "")
        force_log = ", force_envelope_red=True" if force_env else ""
        self._log(f"\n=== FULL BENCHMARK SUITE: backends={backends}, "
                  f"repeats={repeats}{real_repeats_log}{force_log}, "
                  f"shots=2^{shots_exp}, extras={extras} "
                  f"({total_runs} total runs) ===")
        self._set_running(True)
        self._start_overall(total=total_runs)

        def task():
            try:
                artifacts = run_full_benchmark(
                    backends=backends,
                    repeats=repeats,
                    real_repeats=real_repeats,
                    shots_exponent=shots_exp,
                    extras=extras,
                    force_envelope_red=force_env,
                    status=lambda m: self.msg_queue.put(("log", m)),
                    on_run_complete=lambda i, total, bench:
                        self.msg_queue.put(("suite_progress", (i, total, bench))),
                )
                self.msg_queue.put(("full_done", artifacts))
            except Exception as e:
                import traceback
                self.msg_queue.put(("error", f"{e!r}\n{traceback.format_exc()}"))
            finally:
                self.msg_queue.put(("idle", None))

        self.worker = threading.Thread(target=task, daemon=True)
        self.worker.start()

    def _ask_full_suite_config(self):
        dlg = tk.Toplevel(self.root)
        dlg.title("Full benchmark suite — configuration")
        dlg.transient(self.root)
        dlg.grab_set()
        dlg.geometry("520x460")

        result: dict[str, Any] = {}

        ideal_var = tk.BooleanVar(value=True)
        gpu_var = tk.BooleanVar(value=True)
        emu_var = tk.BooleanVar(value=True)
        emu_4qv1_var = tk.BooleanVar(value=False)
        emu_8qv1_var = tk.BooleanVar(value=False)
        real_var = tk.BooleanVar(value=False)
        repeats_var = tk.IntVar(value=int(self.repeats_var.get()))
        real_repeats_var = tk.IntVar(value=int(self.repeats_var.get()))
        shots_var = tk.IntVar(value=int(self.shots_exp_var.get()))
        extras_var = tk.BooleanVar(value=False)
        force_envelope_var = tk.BooleanVar(value=False)

        ttk.Label(dlg, text="Backends to benchmark:",
                  font=("TkDefaultFont", 10, "bold")).pack(anchor="w", padx=12, pady=(12, 2))
        ttk.Checkbutton(dlg, text="Ideal simulator (CPU, no noise)",
                         variable=ideal_var).pack(anchor="w", padx=24)
        ttk.Checkbutton(dlg, text="GPU simulator (CuPy, RTX 4060)",
                         variable=gpu_var).pack(anchor="w", padx=24)
        ttk.Checkbutton(dlg, text="Local emulator — Snowdrop 4q ver2 (offline JSON)",
                         variable=emu_var).pack(anchor="w", padx=24)
        ttk.Checkbutton(dlg, text="Local emulator — Snowdrop 4q ver1 (offline JSON)",
                         variable=emu_4qv1_var).pack(anchor="w", padx=24)
        ttk.Checkbutton(dlg, text="Local emulator — Snowdrop 8q ver1 (offline JSON)",
                         variable=emu_8qv1_var).pack(anchor="w", padx=24)
        ttk.Checkbutton(dlg, text="Real hardware (Snowdrop 4q ver2)",
                         variable=real_var).pack(anchor="w", padx=24)

        sp = ttk.Frame(dlg)
        sp.pack(fill="x", padx=12, pady=(12, 0))
        ttk.Label(sp, text="Repeats per (experiment, backend):").grid(
            row=0, column=0, sticky="w")
        ttk.Spinbox(sp, from_=1, to=50, textvariable=repeats_var, width=6).grid(
            row=0, column=1, padx=(8, 0))
        ttk.Label(sp, text="Real-hardware repeats (override):").grid(
            row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Spinbox(sp, from_=1, to=50, textvariable=real_repeats_var, width=6).grid(
            row=1, column=1, padx=(8, 0), pady=(6, 0))
        ttk.Label(sp, text="Shots exponent (2^N):").grid(
            row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Spinbox(sp, from_=4, to=14, textvariable=shots_var, width=6).grid(
            row=2, column=1, padx=(8, 0), pady=(6, 0))

        ttk.Checkbutton(
            dlg, text="Extras: sweep GHZ-N (2,3,4) and 4 BV secrets",
            variable=extras_var
        ).pack(anchor="w", padx=12, pady=(8, 0))

        ttk.Checkbutton(
            dlg,
            text=("Force-run envelope-red real-hw configs at end "
                  "(bypass safety check; risky)"),
            variable=force_envelope_var,
        ).pack(anchor="w", padx=12, pady=(2, 0))

        warn_lbl = ttk.Label(dlg, text="", foreground="#a00", wraplength=460)
        warn_lbl.pack(anchor="w", padx=12, pady=(8, 0))

        def _safe_int(var, default=1):
            """Read a Tk variable that may transiently hold '' (during typing
            or trace_add fires before the user has typed anything)."""
            try:
                v = var.get()
            except Exception:
                return default
            if isinstance(v, int):
                return v
            s = str(v).strip()
            if not s:
                return default
            try:
                return int(float(s))
            except (TypeError, ValueError):
                return default

        def update_warning(*_):
            chosen_backends = sum(map(bool, [
                ideal_var.get(), gpu_var.get(),
                emu_var.get(), emu_4qv1_var.get(), emu_8qv1_var.get(),
                real_var.get(),
            ]))
            # plan: Bell (1) + GHZ-N (3 if extras else 1) + BV (4 if extras else 1) + Shor (1)
            n_configs = (1 + (3 if extras_var.get() else 1) +
                          (4 if extras_var.get() else 1) + 1)
            repeats_n = _safe_int(repeats_var, default=1)
            real_r = _safe_int(real_repeats_var, default=repeats_n)
            # Local backends (everything but real) all run at repeats_n.
            n_local = chosen_backends - (1 if real_var.get() else 0)
            n_runs = n_local * n_configs * repeats_n
            if real_var.get():
                n_runs += n_configs * real_r
            est_min = 0
            if real_var.get():
                # GHZ-N=4 (extras mode) always skips on real — envelope-red
                # (depth=20, CZ=9 known to trigger chip protection).
                real_configs = n_configs - (1 if extras_var.get() else 0)
                # Per-batched-repeat ≈ 17s wall-clock observed (batch=5 at N=85s);
                # plus ~10s queue/setup overhead per algorithm.
                est_sec = real_configs * (17 * real_r + 10)
                est_min = est_sec / 60
            warn_lbl.config(
                text=f"≈ {n_runs} total runs across {chosen_backends} backends. "
                f"Est. real-hardware time: ~{est_min:.0f} min."
            )

        for v in (ideal_var, gpu_var, emu_var, emu_4qv1_var, emu_8qv1_var,
                   real_var, extras_var, repeats_var, real_repeats_var):
            v.trace_add("write", lambda *_a: update_warning())
        update_warning()

        btn_row = ttk.Frame(dlg)
        btn_row.pack(side="bottom", fill="x", padx=12, pady=12)

        def on_run():
            backends = []
            if ideal_var.get():       backends.append("ideal")
            if gpu_var.get():         backends.append("gpu")
            if emu_var.get():         backends.append("emulator_4q_v2")
            if emu_4qv1_var.get():    backends.append("emulator_4q_v1")
            if emu_8qv1_var.get():    backends.append("emulator_8q_v1")
            if real_var.get():        backends.append("real")
            if not backends:
                messagebox.showwarning("No backends", "Pick at least one backend.")
                return
            result["backends"] = backends
            result["repeats"] = int(repeats_var.get())
            result["shots_exp"] = int(shots_var.get())
            result["extras"] = bool(extras_var.get())
            # Only treat as override if user changed it relative to main repeats.
            rr = int(real_repeats_var.get())
            result["real_repeats"] = rr if rr != int(repeats_var.get()) else None
            result["force_envelope_red"] = bool(force_envelope_var.get())
            dlg.destroy()

        def on_cancel():
            dlg.destroy()

        ttk.Button(btn_row, text="Cancel", command=on_cancel).pack(side="right")
        ttk.Button(btn_row, text="Run suite", command=on_run).pack(
            side="right", padx=(0, 8))

        self.root.wait_window(dlg)
        if not result:
            return None
        return (result["backends"], result["repeats"],
                result["shots_exp"], result["extras"],
                result["real_repeats"],
                result["force_envelope_red"])

    def _kick_off_benchmark(self, exp, backend, params, shots_exp, repeats):
        def task():
            try:
                def on_run_complete(i, res):
                    self.msg_queue.put(("bench_progress", (i + 1, repeats, res)))

                bench = benchmark(
                    exp,
                    backend_kind=backend,
                    params=params,
                    shots_exponent=shots_exp,
                    repeats=repeats,
                    status=lambda m: self.msg_queue.put(("log", m)),
                    on_run_complete=on_run_complete,
                )
                self.msg_queue.put(("bench_done", (exp, params, bench)))
            except Exception as e:
                import traceback
                self.msg_queue.put(("error", f"{e!r}\n{traceback.format_exc()}"))
            finally:
                self.msg_queue.put(("idle", None))

        self.worker = threading.Thread(target=task, daemon=True)
        self.worker.start()

    def _poll_queue(self):
        with suppress(queue.Empty):
            while True:
                kind, payload = self.msg_queue.get_nowait()
                if kind == "log":
                    self._log(payload)
                elif kind == "chip_spec":
                    self._render_chip(payload)
                elif kind == "status_ping":
                    # Lightweight pill-only update (no full chip spec)
                    qinfo = payload
                    qstatus = qinfo.get("status", "unknown")
                    if qstatus == "READY":
                        self._set_status_pill("ready", "Готов")
                    elif qstatus == "NOTREADY":
                        self._set_status_pill("notready", "Не готов")
                    else:
                        self._set_status_pill("unknown", qstatus or "—")
                    self._set_queue_pill(qinfo.get("queue"))
                elif kind == "partial_result":
                    self._render_partial(*payload)
                    self._step_overall(1)
                elif kind == "all_done":
                    jobs, results, mode = payload
                    if mode == "three_way":
                        self._render_three_way(jobs, results)
                elif kind == "bench_progress":
                    i, total, res = payload
                    self._log(f"  → bench [{i}/{total}] "
                              f"{res.metric_name}={res.metric_value:.4f} "
                              f"(total={res.timing.total_s:.2f}s)")
                    self._step_overall(1)
                elif kind == "suite_progress":
                    config_idx, total_configs, bench = payload
                    # bench.repeats successful runs done in this config
                    self._step_overall(bench.repeats)
                elif kind == "bench_done":
                    exp, params, bench = payload
                    self._render_benchmark(exp, params, bench)
                elif kind == "full_done":
                    self._render_full_suite(payload)
                elif kind == "scaling_done":
                    self._render_scaling(payload)
                elif kind == "crm_done":
                    self._render_crm(payload)
                elif kind == "pqc_done":
                    self._render_pqc(payload)
                elif kind == "idle":
                    self._set_running(False)
                elif kind == "error":
                    self._log("ERROR: " + str(payload))
                    messagebox.showerror("Experiment failed", str(payload))
        self.root.after(150, self._poll_queue)

    def _render_chip(self, payload):
        # Backwards-compatible: accept either ChipSpec alone, or (spec, qinfo)
        if isinstance(payload, tuple) and len(payload) == 2:
            spec, qinfo = payload
        else:
            spec, qinfo = payload, None

        if qinfo is not None:
            qstatus = qinfo.get("status", "unknown")
            # Maintenance flag is intentionally ignored — pill reflects only
            # whether the chip API reports it as runnable.
            if qstatus == "READY":
                self._set_status_pill("ready", "Готов")
            elif qstatus == "NOTREADY":
                self._set_status_pill("notready", "Не готов")
            else:
                self._set_status_pill("unknown", qstatus or "—")
            self._set_queue_pill(qinfo.get("queue"))

        text = (
            f"{spec.num_qubits} qubits  |  basis: {', '.join(spec.basis_gates)}    "
            f"F_1q = {spec.avg_f1q*100:.2f}%   "
            f"F_2q (CZ) = {spec.avg_f2q*100:.2f}%   "
            f"F_RO = {spec.avg_ro*100:.2f}%\n"
            f"T1 = {spec.avg_t1_us:.1f} us   T2 = {spec.avg_t2_us:.1f} us   "
            f"coupling = {spec.coupling_map}"
        )
        self.chip_metrics_lbl.config(text=text, foreground=PALETTE["text"])
        self._log(text.replace("\n", " | "))

    def _render_partial(self, exp: ExperimentDef, backend: str,
                        params: dict, res: ExperimentResult):
        t = res.timing
        self._log(
            f"  → {backend}: {exp.metric_name} = {res.metric_value:.4f}\n"
            f"      total      = {t.total_s*1000:7.1f} ms\n"
            f"      ├─ setup   = {t.python_setup_s*1000:7.1f} ms  (Python overhead)\n"
            f"      ├─ transpile = {t.transpile_s*1000:7.1f} ms  (Qiskit Rust+Py)\n"
            f"      ├─ submit  = {t.submit_s*1000:7.1f} ms  (network)\n"
            f"      ├─ queue   = {t.queue_s*1000:7.1f} ms  (Bauman scheduler)\n"
            f"      ├─ execute = {t.execute_s*1000:7.1f} ms  (chip/Aer/GPU)\n"
            f"      └─ post    = {t.python_post_s*1000:7.1f} ms  (Python overhead)\n"
            f"      ► algorithmic     = {t.algorithmic_s*1000:7.1f} ms\n"
            f"      ► python overhead = {t.python_overhead_s*1000:7.1f} ms "
            f"({t.python_overhead_pct:.1f}%)"
        )
        self.last_results[f"{exp.key}_{backend}"] = res
        self.last_benchmark = None

        self._draw_single(exp, params, res)
        self.metric_lbl.config(
            text=f"{exp.metric_name}: {res.metric_value:.4f}  "
                 f"(backend = {backend}, depth={res.transpiled_depth}, "
                 f"shots={res.shots}, total={t.total_s*1000:.0f}ms, "
                 f"algo={t.algorithmic_s*1000:.0f}ms, "
                 f"py-overhead={t.python_overhead_pct:.1f}%)"
        )
        if exp.key == "shor":
            interp = shor_interpret(res.counts, params)
            self.verdict_lbl.config(text=interp["verdict"])
        else:
            self.verdict_lbl.config(text=exp.interpretation_hint)
        self.canvas.draw_idle()

    def _render_three_way(self, jobs, results):
        if len(jobs) <= 1:
            return
        exp = jobs[0][0]
        params = jobs[0][2]
        triple = {b: results.get(f"{exp.key}_{b}") for _, b, _ in jobs}
        self._draw_three_way(exp, params, triple)

        line = " | ".join(
            f"{b}: {r.metric_value:.4f}" for b, r in triple.items() if r is not None
        )
        self.metric_lbl.config(text=f"{exp.metric_name} — {line}")
        if exp.key == "shor":
            real_res = triple.get("real")
            if real_res:
                interp = shor_interpret(real_res.counts, params)
                self.verdict_lbl.config(text=f"On real device: {interp['verdict']}")
        self.canvas.draw_idle()

    def _render_pqc(self, paths: dict):
        self._log("\n=== PQC suite complete ===")
        for k, p in paths.items():
            self._log(f"  ✓ {p}")
        self._set_out_dir(paths.get("out_dir"))
        self.metric_lbl.config(text=f"PQC artifacts: {paths.get('out_dir', '')}")
        self.verdict_lbl.config(text=(
            "Kyber/Dilithium/SPHINCS+ benchmarks vs RSA/ECC baselines, "
            "hybrid-scheme overhead, classical SCI table, threat matrix, "
            "and 5-phase migration roadmap — all in results/1_pqc_benchmarks/. "
            "Use the CSV picker above to flip through every table."
        ))
        primary = (paths.get("pqc_benchmarks.csv")
                   or paths.get("comparative_analysis.csv"))
        self._finish_preview(primary)

    def _render_crm(self, paths: dict):
        self._log("\n=== CRM forecast complete ===")
        for k, p in paths.items():
            self._log(f"  ✓ {p}")

        if paths.get("crm_md"):
            self._set_out_dir(Path(paths["crm_md"]).parent)
        self.metric_lbl.config(text=f"CRM artifacts: {paths.get('crm_md')}")
        self.verdict_lbl.config(text=(
            "CRM = chip-specific cryptanalytic benchmark. The forecast chart "
            "shows where the NISQ→RSA-2048 boundary will likely cross."
        ))
        self._finish_preview(
            paths.get("crm_csv"), fallback_png=paths.get("crm_chart"),
            title="CRM forecast: NISQ cryptanalysis vs RSA-2048")

    def _render_scaling(self, paths: dict):
        self._log("\n=== Scaling demo done ===")
        for k, p in paths.items():
            self._log(f"  ✓ {p}")

        if paths.get("chart"):
            self._set_out_dir(Path(paths["chart"]).parent)
        self.metric_lbl.config(text=f"Scaling chart saved: {paths.get('chart')}")
        self.verdict_lbl.config(text=(
            "Compare with Snowdrop's 4-qubit ceiling. The cross-over point shows "
            "where GPU starts beating CPU and where both go beyond NISQ hardware."
        ))
        self._finish_preview(
            paths.get("csv"), fallback_png=paths.get("chart"),
            title="GHZ-N scaling: CPU vs GPU vs Snowdrop chip")

    def _render_full_suite(self, artifacts: dict):
        self._log("\n=== Full benchmark complete. Artifacts: ===")
        for name, p in artifacts.items():
            self._log(f"  ✓ {p}")

        report = artifacts.get("report_md")
        if report:
            self._set_out_dir(Path(report).parent)
            self.metric_lbl.config(
                text=f"Full suite complete. Report: {Path(report).resolve()}"
            )
            self.verdict_lbl.config(
                text=(f"Open report.md for the full thesis-ready summary, "
                      f"including SCI_HW table and chip calibration. All raw "
                      f"data preserved in quantum_hardware_runs.json. The CSV "
                      f"picker above flips through every table of this run.")
            )

        self._finish_preview(
            artifacts.get("summary_csv"),
            fallback_png=artifacts.get("fidelity_chart"),
            title="Full benchmark suite — fidelity comparison")

    def _render_benchmark(self, exp: ExperimentDef, params: dict,
                          bench: BenchmarkResult):
        self.last_benchmark = bench
        self._log("\n  ===> " + bench.summary())
        self._draw_benchmark(exp, params, bench)
        lo, hi = bench.ci_95
        self.metric_lbl.config(
            text=f"{exp.metric_name}: mean = {bench.mean:.4f} "
                 f"± {bench.stdev:.4f} (95% CI [{lo:.4f}, {hi:.4f}])  "
                 f"| N = {bench.repeats}, backend = {bench.backend}"
        )
        self.verdict_lbl.config(text=(
            f"avg total {bench.avg_total_s:.2f}s "
            f"(queue {bench.avg_queue_s:.2f}s + exec {bench.avg_execute_s:.2f}s)"
        ))
        self.canvas.draw_idle()

    def _draw_single(self, exp: ExperimentDef, params: dict,
                      res: ExperimentResult):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        probs = res.probabilities()
        expected = exp.expected(params)
        keys = sorted(set(list(probs.keys()) + list(expected.keys())))
        observed = [probs.get(k, 0.0) for k in keys]
        ideal = [expected.get(k, 0.0) for k in keys]

        import numpy as np
        x = np.arange(len(keys))
        w = 0.4
        ax.bar(x - w/2, ideal, w, label="ideal",
               color=CHART_COLORS["expected"])
        ax.bar(x + w/2, observed, w,
               label=f"observed ({res.backend})", color=CHART_COLORS["gpu"])
        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=0)
        ax.set_ylim(0, max(1.0, max(observed + ideal) * 1.15))
        # Explicit projector-friendly sizes: global rcParams stay at
        # publication scale because the pipelines share this process.
        ax.set_xlabel("Outcome (bitstring)", fontsize=13)
        ax.set_ylabel("Probability", fontsize=13)
        ax.set_title(f"{exp.title_with_params(params)} — {res.backend}",
                     fontsize=15)
        ax.tick_params(labelsize=12)
        ax.legend(loc="upper right", fontsize=12)
        ax.grid(True, axis="y", alpha=0.3)

    def _draw_three_way(self, exp: ExperimentDef, params: dict, triple: dict):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        expected = exp.expected(params)
        all_keys = set(expected.keys())
        for r in triple.values():
            if r is not None:
                all_keys.update(r.counts.keys())
        keys = sorted(all_keys)

        import numpy as np
        x = np.arange(len(keys))
        n = sum(1 for r in triple.values() if r is not None)
        width = 0.8 / max(n, 1)

        colors = {
            "ideal":    CHART_COLORS["ideal"],
            "gpu":      CHART_COLORS["gpu"],
            "emulator": CHART_COLORS["emulator"],
            "real":     CHART_COLORS["real"],
        }
        offset = -0.4 + width / 2
        for backend, res in triple.items():
            if res is None:
                continue
            probs = res.probabilities()
            vals = [probs.get(k, 0) for k in keys]
            label = f"{backend} ({exp.metric_name}={res.metric_value:.3f})"
            ax.bar(x + offset, vals, width, label=label,
                   color=colors.get(backend, "gray"))
            offset += width

        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=0)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Outcome (bitstring)", fontsize=13)
        ax.set_ylabel("Probability", fontsize=13)
        ax.set_title(f"{exp.title_with_params(params)} — ideal / emulator / real",
                     fontsize=15)
        ax.tick_params(labelsize=12)
        ax.legend(loc="upper right", fontsize=12)
        ax.grid(True, axis="y", alpha=0.3)

    def _draw_benchmark(self, exp: ExperimentDef, params: dict,
                          bench: BenchmarkResult):
        import numpy as np
        self.fig.clear()
        gs = self.fig.add_gridspec(1, 3, width_ratios=[1.4, 1.0, 1.0])

        ax_run = self.fig.add_subplot(gs[0, 0])
        runs_x = np.arange(1, bench.repeats + 1)
        fids = bench.fidelities
        ax_run.plot(runs_x, fids, marker="o", color=CHART_COLORS["gpu"],
                    linewidth=1.4)
        ax_run.axhline(bench.mean, color="#666", linestyle="--",
                       label=f"mean = {bench.mean:.4f}")
        lo, hi = bench.ci_95
        ax_run.fill_between(runs_x, lo, hi, alpha=0.15,
                            color=CHART_COLORS["gpu"], label=f"95% CI")
        ax_run.set_xlabel("Run #", fontsize=11)
        ax_run.set_ylabel(exp.metric_name, fontsize=11)
        ax_run.set_title(f"Per-run {exp.metric_name}", fontsize=12)
        ax_run.set_ylim(0, 1.05)
        ax_run.set_xticks(runs_x)
        ax_run.tick_params(labelsize=10)
        ax_run.grid(True, alpha=0.3)
        ax_run.legend(loc="lower right", fontsize=10)

        ax_hist = self.fig.add_subplot(gs[0, 1])
        bins = max(5, min(15, bench.repeats // 2 + 2))
        ax_hist.hist(fids, bins=bins, color=CHART_COLORS["gpu"], alpha=0.75,
                     edgecolor="white")
        ax_hist.axvline(bench.mean, color="#c00", linestyle="--", linewidth=2,
                        label=f"mean")
        ax_hist.set_xlabel(exp.metric_name, fontsize=11)
        ax_hist.set_ylabel("Count", fontsize=11)
        ax_hist.set_title(f"Distribution (σ = {bench.stdev:.4f})", fontsize=12)
        ax_hist.tick_params(labelsize=10)
        ax_hist.grid(True, alpha=0.3)
        ax_hist.legend(loc="upper right", fontsize=10)

        ax_time = self.fig.add_subplot(gs[0, 2])
        timings = [r.timing for r in bench.runs]
        labels = ["transpile", "submit", "queue", "execute"]
        means = [
            float(np.mean([t.transpile_s for t in timings])),
            float(np.mean([t.submit_s for t in timings])),
            float(np.mean([t.queue_s for t in timings])),
            float(np.mean([t.execute_s for t in timings])),
        ]
        bars = ax_time.bar(labels, means, color=["#9bc4e2", "#f0a847",
                                                  "#c0a8e8", "#0a7a3a"])
        for b, v in zip(bars, means):
            ax_time.text(b.get_x() + b.get_width()/2, b.get_height() * 1.02,
                         f"{v:.2f}s", ha="center", va="bottom", fontsize=9)
        ax_time.set_ylabel("Seconds (avg per run)", fontsize=11)
        ax_time.set_title(f"Timing breakdown (total {bench.avg_total_s:.1f}s)",
                          fontsize=12)
        ax_time.tick_params(labelsize=10)
        ax_time.grid(True, axis="y", alpha=0.3)

        self.fig.suptitle(
            f"Benchmark: {exp.title_with_params(params)} on "
            f"{bench.backend} × {bench.repeats}",
            y=1.02, fontsize=13, fontweight="bold",
        )

    def _save_report(self):
        if not self.last_results and not self.last_benchmark:
            messagebox.showinfo("Nothing to save",
                                "Run at least one experiment first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
            initialdir=PROJECT_ROOT / "results",
            initialfile="qccb_gui_report.json",
        )
        if not path:
            return
        payload: dict[str, Any] = {"single_runs": {}, "benchmark": None}
        for k, r in self.last_results.items():
            payload["single_runs"][k] = {
                "label": r.label,
                "backend": r.backend,
                "shots": r.shots,
                "counts": r.counts,
                "probabilities": r.probabilities(),
                "metric_name": r.metric_name,
                "metric_value": r.metric_value,
                "expected_distribution": r.expected_distribution,
                "transpiled_depth": r.transpiled_depth,
                "transpiled_ops": r.transpiled_ops,
                "timing_s": r.timing.to_dict(),
                "job_id": r.job_id,
            }
        if self.last_benchmark:
            b = self.last_benchmark
            lo, hi = b.ci_95
            payload["benchmark"] = {
                "label": b.label,
                "backend": b.backend,
                "shots_per_run": b.shots_per_run,
                "repeats": b.repeats,
                "fidelities": b.fidelities,
                "mean": b.mean,
                "stdev": b.stdev,
                "ci_95": [lo, hi],
                "avg_total_s": b.avg_total_s,
                "avg_queue_s": b.avg_queue_s,
                "avg_execute_s": b.avg_execute_s,
                "per_run_timings": [r.timing.to_dict() for r in b.runs],
                "per_run_counts": [r.counts for r in b.runs],
            }
        Path(path).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self._log(f"Report saved: {path}")


def main():
    root = tk.Tk()
    apply_theme(root)
    apply_app_icon(root)
    style_matplotlib()
    QCCBGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()

