"""Segmentation backend registry — lazy, cached, fail-soft.

`get_segmenter("gsam"|"sam3")` loads the backend once and caches it. A backend
that fails to load (missing/gated weights) raises a RuntimeError captured per
request; other backends keep working.
"""

from __future__ import annotations

from .base import Detection, SegBackend

AVAILABLE = ("gsam", "sam3")

_cache: dict[str, SegBackend] = {}
_errors: dict[str, str] = {}


def _build(name: str) -> SegBackend:
    if name == "gsam":
        from .gsam import GroundedSAM
        return GroundedSAM()
    if name == "sam3":
        from .sam3 import SAM3
        return SAM3()
    raise ValueError(f"unknown segmentation backend '{name}'; "
                     f"choose from {AVAILABLE}")


def get_segmenter(name: str) -> SegBackend:
    if name in _cache:
        return _cache[name]
    backend = _build(name)        # may raise — surfaced to the caller
    _cache[name] = backend
    _errors.pop(name, None)
    return backend


def status() -> dict:
    """Per-backend availability for /health (does not force-load)."""
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
    """Eagerly attempt to load a backend, recording any error. Returns success."""
    try:
        get_segmenter(name)
        return True
    except Exception as e:  # noqa: BLE001 — we want to record any failure
        _errors[name] = f"{type(e).__name__}: {e}"
        return False


__all__ = ["Detection", "SegBackend", "get_segmenter", "status", "try_load",
           "AVAILABLE"]
