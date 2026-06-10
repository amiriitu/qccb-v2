"""
Entry point for the QCCB v2 quantum experiments GUI.

Usage:
    python run_gui.py
"""
from __future__ import annotations

import os
import sys
from contextlib import suppress
from pathlib import Path

# CRITICAL: must be set BEFORE any module that imports oqs/liboqs.
# liboqs-python's __init__ tries to git-clone-and-build liboqs on Windows,
# which fails on a transient branch mismatch (current pin: 0.14.1, while
# upstream has moved to 0.15.0). The simulator path produces NIST-spec key
# / ciphertext / signature sizes and reference timings, which is what the
# thesis pipeline actually uses.
os.environ.setdefault("QCCB_FORCE_SIMULATOR", "1")

if sys.platform == "win32":
    with suppress(AttributeError, OSError):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.gui import main

if __name__ == "__main__":
    main()

