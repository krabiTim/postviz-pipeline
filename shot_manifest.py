"""
Shot Manifest — reads and writes shot.json in each shot directory.

shot.json tracks stage completion status, camera source, and basic metadata.
It is the authoritative record of what has been done for a shot.

Usage:
    from shot_manifest import load_manifest, update_stage, create_manifest

    create_manifest(shot_dir, video_source="clip.mov", fps=24.0,
                    frame_count=240, width=1920, height=1080)

    update_stage(shot_dir, "camera", done=True, solve_error=0.42, source="colmap")

    m = load_manifest(shot_dir)
    if m["stages"]["camera"]["done"]:
        print("Camera solved")
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

MANIFEST_NAME = "shot.json"

_DEFAULT_STAGES: dict[str, dict] = {
    "frames":     {"done": False, "timestamp": None},
    "camera":     {"done": False, "timestamp": None, "source": None, "solve_error": None},
    "mask":       {"done": False, "timestamp": None, "workflow": None},
    "background": {"done": False, "timestamp": None, "workflow": None},
    "composite":  {"done": False, "timestamp": None, "relight": False},
    "delivery":   {"done": False, "timestamp": None, "path": None},
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _manifest_path(shot_dir: str | Path) -> Path:
    return Path(shot_dir) / MANIFEST_NAME


def create_manifest(
    shot_dir: str | Path,
    video_source: str = "",
    fps: float = 0.0,
    frame_count: int = 0,
    width: int = 0,
    height: int = 0,
) -> dict:
    """
    Create a fresh shot.json in shot_dir.
    Call this immediately after make_shot_dir().
    Returns the manifest dict.
    """
    shot_dir = Path(shot_dir)
    manifest = {
        "shot_name": shot_dir.name,
        "created": _now(),
        "video_source": video_source,
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "stages": {k: dict(v) for k, v in _DEFAULT_STAGES.items()},
    }
    _write(shot_dir, manifest)
    return manifest


def load_manifest(shot_dir: str | Path) -> dict | None:
    """
    Load shot.json from shot_dir.
    Returns None if the file does not exist.
    """
    p = _manifest_path(shot_dir)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning(f"[shot_manifest] Could not read {p}: {e}")
        return None


def update_stage(shot_dir: str | Path, stage: str, done: bool = True, **kwargs: Any) -> dict:
    """
    Update a stage entry in shot.json and persist.

    Examples:
        update_stage(shot_dir, "camera", done=True, solve_error=0.42, source="colmap")
        update_stage(shot_dir, "mask", done=True, workflow="sam2_mask")
        update_stage(shot_dir, "frames", done=True)

    Returns the updated manifest dict.
    """
    shot_dir = Path(shot_dir)
    manifest = load_manifest(shot_dir) or create_manifest(shot_dir)

    if stage not in manifest["stages"]:
        log.warning(f"[shot_manifest] Unknown stage '{stage}' — skipping")
        return manifest

    manifest["stages"][stage]["done"] = done
    manifest["stages"][stage]["timestamp"] = _now() if done else None
    for k, v in kwargs.items():
        manifest["stages"][stage][k] = v

    _write(shot_dir, manifest)
    return manifest


def update_meta(shot_dir: str | Path, **kwargs: Any) -> dict:
    """
    Update top-level fields in shot.json (fps, frame_count, etc.).
    Returns the updated manifest.
    """
    shot_dir = Path(shot_dir)
    manifest = load_manifest(shot_dir) or create_manifest(shot_dir)
    for k, v in kwargs.items():
        manifest[k] = v
    _write(shot_dir, manifest)
    return manifest


def pipeline_log(shot_dir: str | Path, stage: str, message: str) -> None:
    """
    Append a timestamped line to projects/{shot}/pipeline.log.
    Format: [2026-02-26 11:22:33] [STAGE     ] message
    """
    log_path = Path(shot_dir) / "pipeline.log"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [{stage.upper():10s}] {message}\n"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line)


def _write(shot_dir: Path, manifest: dict) -> None:
    p = _manifest_path(shot_dir)
    p.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
