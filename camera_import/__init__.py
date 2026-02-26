"""
Camera Import Plugin System

Dispatches camera file imports to the appropriate importer based on file extension.
All importers return a camera_path.json-compatible dict.

Usage:
    from camera_import import import_camera, SUPPORTED_EXTENSIONS

    result = import_camera("/path/to/shot.fbx", fps=24.0)

To add a new format:
    1. Create camera_import/yourformat.py with import_camera(file_path, fps) -> dict
    2. Add one line to IMPORTERS below
    3. Done — the extension appears in the UI file picker automatically
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .fbx import import_fbx
from .json_passthrough import import_json
from .usd import import_usd

# Map lowercase file extension → import function
IMPORTERS: dict[str, Callable[[str, float], dict]] = {
    ".fbx": import_fbx,
    ".usd": import_usd,
    ".usda": import_usd,
    ".usdc": import_usd,
    ".usdz": import_usd,
    ".json": import_json,
}

SUPPORTED_EXTENSIONS: list[str] = sorted(IMPORTERS.keys())


def import_camera(file_path: str, fps: float = 24.0) -> dict:
    """
    Import a camera file and return a camera_path.json-compatible dict.

    Returns dict with: total_frames, fps, solve_error, source, frames[]
    Each frame: {frame_index, registered, position, rotation_matrix}

    Raises:
        ValueError: if the file extension is not supported
        RuntimeError: if the importer fails
    """
    ext = Path(file_path).suffix.lower()
    if ext not in IMPORTERS:
        raise ValueError(
            f"Unsupported camera format: '{ext}'. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    return IMPORTERS[ext](file_path, fps)
