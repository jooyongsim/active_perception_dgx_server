"""Grasp backend registry — lazy, cached, fail-soft.

`analytic` is always available (numpy/scipy). `cgn` (Contact-GraspNet) needs an
extra install + checkpoint; if absent it reports unavailable and per-request
calls return an actionable error instead of crashing.
"""

from __future__ import annotations

from .base import Grasp, GraspBackend

AVAILABLE = ("analytic", "cgn")

_cache: dict[str, GraspBackend] = {}
_errors: dict[str, str] = {}


def _build(name: str) -> GraspBackend:
    if name == "analytic":
        from .analytic import AnalyticGrasp
        return AnalyticGrasp()
    if name == "cgn":
        from .contact_graspnet import ContactGraspNet
        return ContactGraspNet()
    raise ValueError(f"unknown grasp backend '{name}'; choose from {AVAILABLE}")


def get_grasp_backend(name: str) -> GraspBackend:
    if name in _cache:
        return _cache[name]
    backend = _build(name)
    _cache[name] = backend
    _errors.pop(name, None)
    return backend


def status() -> dict:
    out = {}
    for name in AVAILABLE:
        if name in _cache:
            out[name] = {"loaded": True, **_cache[name].info()}
        else:
            entry = {"loaded": False}
            if name in _errors:
                entry["error"] = _errors[name]
            out[name] = entry
    return out


def try_load(name: str) -> bool:
    try:
        get_grasp_backend(name)
        return True
    except Exception as e:  # noqa: BLE001
        _errors[name] = f"{type(e).__name__}: {e}"
        return False


__all__ = ["Grasp", "GraspBackend", "get_grasp_backend", "status", "try_load",
           "AVAILABLE"]
