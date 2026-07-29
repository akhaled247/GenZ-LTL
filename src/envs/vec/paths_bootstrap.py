"""Spawn-safe PYTHONPATH bootstrap for GenZ multiprocess env workers."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _repo_roots() -> tuple[Path, Path]:
    """Return (GenZ-LTL/src, SpecRLBench) assuming standard RISE-2026 layout."""
    src = Path(__file__).resolve().parents[2]
    genz_src = src
    specrl = src.parent.parent / "SpecRLBench"
    if not specrl.is_dir():
        # GenZ-LTL/src -> GenZ-LTL -> RISE-2026
        specrl = src.parent.parent / "SpecRLBench"
    return genz_src, specrl


def ensure_genz_paths() -> None:
    """Ensure worker interpreters can import GenZ `src` and SpecRLBench."""
    genz_src, specrl = _repo_roots()
    paths = [str(genz_src)]
    if specrl.is_dir():
        paths.append(str(specrl))
    existing = os.environ.get("PYTHONPATH", "")
    parts = [p for p in existing.split(os.pathsep) if p]
    for path in paths:
        if path not in parts:
            parts.insert(0, path)
    os.environ["PYTHONPATH"] = os.pathsep.join(parts)
    for path in paths:
        if path not in sys.path:
            sys.path.insert(0, path)
