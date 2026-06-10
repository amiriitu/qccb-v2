"""
Shared visual theme for every QCCB Tk window.

One place controls the palette, fonts and ttk styles, so the main GUI,
the SCI calculator and the thesis-tools dialog all look like parts of the
same application. Built on the 'clam' ttk theme because it is the only
built-in theme whose colors can be fully overridden on Windows.

Usage:
    from src.ui_theme import apply_theme, PALETTE
    root = tk.Tk()
    apply_theme(root)
"""
from __future__ import annotations

import tkinter as tk
import tkinter.font as tkfont
from contextlib import suppress
from tkinter import ttk

# --------------------------------------------------------------------- palette
PALETTE = {
    "bg":          "#EDF0F4",   # app background — soft cool gray
    "surface":     "#FFFFFF",   # cards / panels
    "border":      "#D5DBE3",
    "text":        "#1B2430",
    "text_muted":  "#5D6878",
    "accent":      "#2563EB",   # primary actions
    "accent_dark": "#1D4ED8",
    "accent_soft": "#EAF1FE",   # hover tint
    "success":     "#0E9F4F",
    "success_soft": "#DCFCE7",
    "warning":     "#C77800",
    "danger":      "#D6453D",
    "danger_dark": "#B3362F",
    "console_bg":  "#101826",   # status log — dark terminal
    "console_fg":  "#D7E1EE",
    "console_dim": "#7C8AA0",
}

# Bar/series colors used by the embedded matplotlib charts. Color-blind-safe
# and consistent with tools/replot_charts.py.
CHART_COLORS = {
    "ideal":    "#7FA8D9",
    "expected": "#B9C6D8",
    "gpu":      "#0E9F4F",
    "emulator": "#E8A33D",
    "real":     "#1B2430",
    "series":   ["#2563EB", "#0E9F4F", "#E8A33D", "#9A5BD2", "#D6453D",
                  "#5D6878"],
}

_BASE_SIZE = 10          # default UI font size (Segoe UI), demo-readable
_MONO_FAMILY = "Consolas"


def _pick_family(root: tk.Misc) -> str:
    families = set(tkfont.families(root))
    for name in ("Segoe UI Variable Text", "Segoe UI", "Helvetica Neue",
                  "Arial"):
        if name in families:
            return name
    return "TkDefaultFont"


def apply_theme(root: tk.Misc, base_size: int = _BASE_SIZE) -> ttk.Style:
    """Apply the QCCB look to a Tk root. Returns the configured Style."""
    family = _pick_family(root)
    p = PALETTE

    # ---- named fonts: resizing these cascades through every ttk widget
    for font_name, opts in {
        "TkDefaultFont":  {"family": family, "size": base_size},
        "TkTextFont":     {"family": family, "size": base_size},
        "TkMenuFont":     {"family": family, "size": base_size},
        "TkHeadingFont":  {"family": family, "size": base_size, "weight": "bold"},
        "TkTooltipFont":  {"family": family, "size": base_size - 1},
        "TkFixedFont":    {"family": _MONO_FAMILY, "size": base_size},
    }.items():
        with suppress(tk.TclError):
            tkfont.nametofont(font_name).configure(**opts)

    root.option_add("*Toplevel.background", p["surface"])

    style = ttk.Style(root)
    style.theme_use("clam")

    # ---- base containers
    if isinstance(root, (tk.Tk, tk.Toplevel)):
        root.configure(bg=p["bg"])
    style.configure(".", background=p["surface"], foreground=p["text"],
                    bordercolor=p["border"], font=("TkDefaultFont",))
    style.configure("TFrame", background=p["surface"])
    style.configure("App.TFrame", background=p["bg"])

    style.configure(
        "TLabelframe", background=p["surface"], bordercolor=p["border"],
        relief="solid", borderwidth=1, padding=4,
    )
    style.configure(
        "TLabelframe.Label", background=p["surface"],
        foreground=p["text_muted"],
        font=(family, base_size, "bold"),
    )

    # ---- labels
    style.configure("TLabel", background=p["surface"], foreground=p["text"])
    style.configure("Muted.TLabel", foreground=p["text_muted"])
    style.configure("Success.TLabel", foreground=p["success"])
    style.configure("Danger.TLabel", foreground=p["danger"])
    style.configure("Title.TLabel", font=(family, base_size + 4, "bold"))
    style.configure("Heading.TLabel", font=(family, base_size, "bold"))

    # ---- buttons
    common_btn = {
        "padding": (12, 7),
        "relief": "flat",
        "borderwidth": 1,
        "focusthickness": 1,
    }
    style.configure("TButton", background=p["surface"],
                    foreground=p["text"], bordercolor=p["border"],
                    focuscolor=p["accent"], **common_btn)
    style.map(
        "TButton",
        background=[("disabled", p["surface"]), ("pressed", p["accent_soft"]),
                    ("active", p["accent_soft"])],
        foreground=[("disabled", "#A8B0BC")],
        bordercolor=[("active", p["accent"]), ("pressed", p["accent"])],
    )

    style.configure("Accent.TButton", background=p["accent"],
                    foreground="#FFFFFF", bordercolor=p["accent"],
                    **common_btn)
    style.map(
        "Accent.TButton",
        background=[("disabled", "#A9C3F5"), ("pressed", p["accent_dark"]),
                    ("active", p["accent_dark"])],
        foreground=[("disabled", "#F2F6FE")],
        bordercolor=[("active", p["accent_dark"])],
    )

    style.configure("Danger.TButton", background=p["danger"],
                    foreground="#FFFFFF", bordercolor=p["danger"],
                    **common_btn)
    style.map(
        "Danger.TButton",
        background=[("pressed", p["danger_dark"]), ("active", p["danger_dark"])],
        bordercolor=[("active", p["danger_dark"])],
    )

    # ---- selection controls
    for widget in ("TRadiobutton", "TCheckbutton"):
        style.configure(widget, background=p["surface"],
                        foreground=p["text"], padding=(2, 2),
                        focuscolor=p["accent"])
        style.map(widget,
                  background=[("active", p["surface"])],
                  foreground=[("disabled", "#A8B0BC")])

    # ---- inputs
    field = {
        "fieldbackground": p["surface"],
        "bordercolor": p["border"],
        "lightcolor": p["border"],
        "darkcolor": p["border"],
        "padding": 4,
    }
    style.configure("TEntry", **field)
    style.configure("TSpinbox", arrowcolor=p["text_muted"], **field)
    style.configure("TCombobox", arrowcolor=p["text_muted"], **field)
    for widget in ("TEntry", "TSpinbox", "TCombobox"):
        style.map(widget,
                  bordercolor=[("focus", p["accent"])],
                  lightcolor=[("focus", p["accent"])],
                  darkcolor=[("focus", p["accent"])])

    # ---- progress bars
    style.configure(
        "Horizontal.TProgressbar",
        background=p["accent"], troughcolor="#E2E7EE",
        bordercolor="#E2E7EE", lightcolor=p["accent"], darkcolor=p["accent"],
        thickness=12,
    )

    # ---- notebook (thesis tools dialog)
    style.configure("TNotebook", background=p["surface"],
                    bordercolor=p["border"], tabmargins=(8, 6, 8, 0))
    style.configure("TNotebook.Tab", background=p["bg"],
                    foreground=p["text_muted"], padding=(14, 7),
                    font=(family, base_size))
    style.map(
        "TNotebook.Tab",
        background=[("selected", p["surface"])],
        foreground=[("selected", p["accent"])],
        expand=[("selected", (0, 0, 0, 1))],
    )

    style.configure("TSeparator", background=p["border"])
    style.configure("Treeview", fieldbackground=p["surface"],
                    background=p["surface"], bordercolor=p["border"])
    style.configure("Treeview.Heading", background=p["bg"],
                    foreground=p["text_muted"], relief="flat")

    return style


def style_console(text_widget: tk.Text, base_size: int = _BASE_SIZE) -> None:
    """Dark terminal look for a tk.Text status feed."""
    text_widget.configure(
        background=PALETTE["console_bg"],
        foreground=PALETTE["console_fg"],
        insertbackground=PALETTE["console_fg"],
        selectbackground="#27415F",
        relief="flat", borderwidth=0,
        padx=10, pady=8,
        font=(_MONO_FAMILY, base_size),
    )


def style_matplotlib() -> None:
    """Match embedded matplotlib charts to the UI palette."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.edgecolor": PALETTE["border"],
        "axes.labelcolor": PALETTE["text"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": PALETTE["border"],
        "grid.alpha": 0.45,
        "grid.linewidth": 0.7,
        "xtick.color": PALETTE["text_muted"],
        "ytick.color": PALETTE["text_muted"],
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "figure.facecolor": PALETTE["surface"],
        "axes.facecolor": PALETTE["surface"],
        "axes.prop_cycle": mpl.cycler(color=CHART_COLORS["series"]),
    })

