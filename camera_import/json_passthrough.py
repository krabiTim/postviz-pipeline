"""
JSON Passthrough Camera Importer

Accepts a pre-made camera_path.json directly — useful when a matchmove artist
or technical director has already exported camera data in the PostViz format,
or when a custom script has been used to generate camera data.
"""

from __future__ import annotations

import json
from pathlib import Path


def import_json(file_path: str, fps: float = 24.0) -> dict:
    """
    Load a camera_path.json directly and validate its structure.

    The file must contain:
      - "frames": list of frame dicts with frame_index, position, rotation_matrix
      - "total_frames": int (computed from len(frames) if missing)

    fps is used only if not present in the file.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Camera JSON not found: {file_path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    if "frames" not in data:
        raise ValueError(f"File must contain a 'frames' list: {file_path}")

    frames = data["frames"]
    if not frames:
        raise ValueError(f"'frames' list is empty: {file_path}")

    required = {"frame_index", "position", "rotation_matrix"}
    for i, frame in enumerate(frames):
        missing = required - set(frame.keys())
        if missing:
            raise ValueError(f"Frame {i} missing keys: {missing}")
        if "registered" not in frame:
            frame["registered"] = True

    result = {
        "total_frames": data.get("total_frames", len(frames)),
        "fps": data.get("fps", fps),
        "solve_error": data.get("solve_error", 0.0),
        "source": "json",
        "frames": frames,
    }

    print(f"[camera_import.json] Loaded {len(frames)} frames from {path.name}")
    return result
