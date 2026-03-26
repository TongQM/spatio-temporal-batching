"""Deprecated wrapper for the uniform FR-detour diagnostic.

Use `fr_detour_comparison.py`; this wrapper is kept only for compatibility.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from scripts.plots.fr_detour_comparison import run_uniform_diagnostic


if __name__ == "__main__":
    run_uniform_diagnostic()
