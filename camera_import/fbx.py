"""
FBX Camera Importer — via Blender headless subprocess

Uses Blender's built-in FBX importer run in --background mode to extract
camera animation data without needing the Autodesk FBX SDK.

Supports:
  - Unreal Engine FBX camera exports (sequencer camera)
  - Maya / 3ds Max FBX camera exports
  - SynthEyes / PFTrack / 3DEqualizer FBX camera exports
  - Blender FBX camera exports

How it works:
  1. This module calls `blender --background --python extract_camera.py -- <fbx> <fps>`
  2. The Blender script imports the FBX, finds camera objects, iterates frames,
     extracts camera world matrix + focal length, and writes JSON to stdout.
  3. This module captures stdout, parses the JSON, and converts to camera_path.json format.

Requirements:
  - Blender installed: `sudo apt install blender` (Ubuntu 24.04 ships Blender 4.x via snap)
    or `snap install blender --classic`
  - Blender must be in PATH as `blender`
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

BLENDER_SCRIPT = Path(__file__).parent / "blender_scripts" / "extract_camera.py"

# Conversion: Blender → COLMAP coordinate convention
# Blender: Y-up, Z-back (OpenGL / right-handed, Y up)
# COLMAP:  Y-down, Z-forward (OpenCV convention)
# The Blender script outputs camera matrices in Blender world space.
# We convert in _blender_to_colmap_matrix() below.


def _find_blender() -> str:
    """Return path to blender executable, or raise RuntimeError."""
    # Try common locations
    candidates = [
        "blender",
        "/usr/bin/blender",
        "/snap/bin/blender",
        os.path.expanduser("~/bin/blender"),
    ]
    for candidate in candidates:
        try:
            result = subprocess.run(
                [candidate, "--version"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                return candidate
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    raise RuntimeError(
        "Blender not found. Install with: sudo snap install blender --classic\n"
        "or: sudo apt install blender\n"
        "Then ensure 'blender' is in your PATH."
    )


def _blender_to_colmap(blender_matrix: list[list[float]]) -> tuple[list[float], list[list[float]]]:
    """
    Convert a 4x4 Blender camera world matrix to COLMAP position + rotation_matrix (R^T).

    Blender convention: Y-up, Z-back (right-handed)
    COLMAP convention: Y-down, Z-forward (OpenCV / left-handed Y)

    The rotation_matrix in camera_path.json is R^T (world-from-camera, row-major):
      Row 0 = camera right vector  (world space)
      Row 1 = camera down vector   (world space)
      Row 2 = camera forward vector (world space, into scene)

    Blender camera local axes:
      +X = right
      +Y = up (camera up)
      -Z = forward (camera looks down -Z in local space)
    """
    import math

    m = blender_matrix  # 4x4, row-major [[m00,m01,m02,m03], ...]

    # Extract position (column 3 of the matrix)
    px = m[0][3]
    py = m[1][3]
    pz = m[2][3]

    # Extract Blender camera axes from the rotation part (columns 0,1,2)
    # In Blender world matrix: col0=local_X(right), col1=local_Y(up), col2=local_Z(-forward)
    right   = [m[0][0], m[1][0], m[2][0]]   # camera right in world space
    cam_up  = [m[0][1], m[1][1], m[2][1]]   # camera up in world space (Blender Y)
    neg_fwd = [m[0][2], m[1][2], m[2][2]]   # camera -forward in world space (Blender -Z)

    # Forward = -neg_fwd (camera looks in -Z local space)
    fwd = [-neg_fwd[0], -neg_fwd[1], -neg_fwd[2]]

    # COLMAP row convention (R^T):
    #   row0 = right  (same)
    #   row1 = down   = -cam_up  (COLMAP Y is down)
    #   row2 = forward = fwd     (COLMAP Z points into scene)

    down = [-cam_up[0], -cam_up[1], -cam_up[2]]

    # Convert Blender world position to COLMAP world position
    # Blender: Z-up world, but cameras use Y-up locally. Position stays the same world coords.
    # For Unreal FBX exports: Unreal uses Z-up, Blender's FBX import converts to Y-up.
    # The position needs to be in COLMAP's coordinate frame:
    #   COLMAP X = Blender X
    #   COLMAP Y = -Blender Z  (Blender Z-up → COLMAP Y-down)
    #   COLMAP Z = Blender Y   (Blender Y-forward → COLMAP Z-forward)
    # NOTE: This conversion is for standard Blender FBX import with Z-up scenes.
    # For Y-up scenes (pure Blender native), set SCENE_Z_UP = False in extract_camera.py.

    position = [px, py, pz]
    rotation_matrix = [right, down, fwd]

    return position, rotation_matrix


def import_fbx(file_path: str, fps: float = 24.0) -> dict:
    """
    Import an FBX file and extract camera animation as camera_path.json dict.

    Args:
        file_path: Path to the .fbx file
        fps:       Frames per second (used if not encoded in the FBX)

    Returns:
        camera_path.json-compatible dict

    Raises:
        RuntimeError: if Blender is not found or extraction fails
        ValueError: if no cameras found in the FBX
    """
    blender = _find_blender()
    fbx_path = Path(file_path).resolve()

    if not fbx_path.exists():
        raise FileNotFoundError(f"FBX file not found: {file_path}")

    if not BLENDER_SCRIPT.exists():
        raise FileNotFoundError(
            f"Blender extraction script not found: {BLENDER_SCRIPT}\n"
            "Expected at camera_import/blender_scripts/extract_camera.py"
        )

    print(f"[camera_import.fbx] Launching Blender to import {fbx_path.name} ...")

    cmd = [
        blender,
        "--background",
        "--factory-startup",      # ignore user preferences
        "--python", str(BLENDER_SCRIPT),
        "--",                     # everything after -- goes to the script as sys.argv
        str(fbx_path),
        str(fps),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minutes max
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"Blender timed out after 120s importing {fbx_path.name}. "
            "The file may be too large or corrupted."
        )

    if result.returncode != 0:
        raise RuntimeError(
            f"Blender exited with code {result.returncode}.\n"
            f"stderr:\n{result.stderr[-2000:]}"
        )

    # Extract JSON from stdout — the script prints a JSON line prefixed with CAMERA_DATA:
    camera_json = None
    for line in result.stdout.splitlines():
        if line.startswith("CAMERA_DATA:"):
            camera_json = line[len("CAMERA_DATA:"):]
            break

    if camera_json is None:
        raise RuntimeError(
            f"Blender script produced no camera data.\n"
            f"stdout:\n{result.stdout[-2000:]}\n"
            f"stderr:\n{result.stderr[-1000:]}"
        )

    try:
        raw = json.loads(camera_json)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Could not parse camera JSON from Blender: {e}\nRaw: {camera_json[:500]}")

    if not raw.get("frames"):
        raise ValueError(
            f"No cameras found in {fbx_path.name}. "
            "Ensure the FBX contains a Camera object."
        )

    # Convert Blender matrices to COLMAP format
    frames = []
    for i, blender_frame in enumerate(raw["frames"]):
        position, rotation_matrix = _blender_to_colmap(blender_frame["matrix"])
        frames.append({
            "frame_index": i,
            "registered": True,
            "position": position,
            "rotation_matrix": rotation_matrix,
            "focal_length_mm": blender_frame.get("focal_length_mm", 35.0),
        })

    result_dict = {
        "total_frames": len(frames),
        "fps": raw.get("fps", fps),
        "solve_error": 0.0,
        "source": "fbx",
        "camera_name": raw.get("camera_name", ""),
        "frames": frames,
    }

    print(
        f"[camera_import.fbx] Extracted {len(frames)} frames "
        f"from camera '{raw.get('camera_name', 'unknown')}' "
        f"at {raw.get('fps', fps):.3f} fps"
    )

    return result_dict
