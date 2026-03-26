"""Lazy package exports for optional heavy dependencies."""

from __future__ import annotations

from importlib import import_module

__all__ = ["GeoData", "load_data", "Partition"]


def __getattr__(name: str):
    if name in {"GeoData", "load_data"}:
        module = import_module(".data", __name__)
        return getattr(module, name)
    if name == "Partition":
        module = import_module(".algorithm", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
