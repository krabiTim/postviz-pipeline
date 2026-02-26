"""
Blender Camera Extraction Script

Run headless via:
  blender --background --factory-startup --python extract_camera.py -- /path/to/file.fbx 24.0

Supports: FBX (.fbx), and anything Blender can import (via format detection).

Outputs a single line to stdout: CAMERA_DATA:{json}
All other output goes to stderr (Blender's default log stream).

JSON output format:
{
  "camera_name": "Camera",
  "fps": 24.0,
  "frame_start": 1,
  "frame_end": 240,
  "frames": [
    {
      "frame": 1,
      "matrix": [[m00,m01,m02,m03],[m10,...],[m20,...],[m30,...]],
      "focal_length_mm": 35.0,
      "sensor_width_mm": 36.0
    },
    ...
  ]
}
"""

import sys
import os
import json

import bpy
import mathutils


def log(msg: str) -> None:
    """Print to stderr so it doesn't contaminate the JSON stdout."""
    print(msg, file=sys.stderr)


def find_fbx_path_and_fps() -> tuple[str, float]:
    """Parse arguments passed after -- on the command line."""
    argv = sys.argv
    try:
        sep = argv.index("--")
        args = argv[sep + 1:]
    except ValueError:
        log("ERROR: No arguments after --. Usage: blender ... -- <file_path> [fps]")
        sys.exit(1)

    if not args:
        log("ERROR: No file path provided.")
        sys.exit(1)

    file_path = args[0]
    fps = float(args[1]) if len(args) > 1 else 24.0

    return file_path, fps


def clear_scene() -> None:
    """Remove all objects from the default Blender scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    # Clear orphaned data
    for block in bpy.data.meshes:
        bpy.data.meshes.remove(block)
    for block in bpy.data.cameras:
        bpy.data.cameras.remove(block)


def import_file(file_path: str) -> None:
    """Import the file using the appropriate Blender operator."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".fbx":
        bpy.ops.import_scene.fbx(
            filepath=file_path,
            use_anim=True,
            anim_offset=0.0,
            ignore_leaf_bones=True,
            force_connect_children=False,
            automatic_bone_orientation=False,
            primary_bone_axis="Y",
            secondary_bone_axis="X",
            use_custom_props=True,
            decal_offset=0.0,
            use_image_search=False,
        )
    elif ext in (".usd", ".usda", ".usdc", ".usdz"):
        bpy.ops.wm.usd_import(filepath=file_path)
    elif ext in (".abc",):
        bpy.ops.wm.alembic_import(filepath=file_path)
    else:
        log(f"ERROR: Unsupported format: {ext}")
        sys.exit(1)

    log(f"Imported: {file_path}")


def find_camera(prefer_name: str = "") -> bpy.types.Object | None:
    """Find the best camera object in the scene."""
    cameras = [obj for obj in bpy.data.objects if obj.type == "CAMERA"]

    if not cameras:
        return None

    # If only one camera, use it
    if len(cameras) == 1:
        return cameras[0]

    # Prefer camera matching the hint name
    if prefer_name:
        for cam in cameras:
            if prefer_name.lower() in cam.name.lower():
                return cam

    # Prefer cameras with "render" or "main" in the name
    for keyword in ("render", "main", "shot", "cam"):
        for cam in cameras:
            if keyword in cam.name.lower():
                return cam

    # Fall back to first camera
    log(f"Multiple cameras found: {[c.name for c in cameras]}. Using first: {cameras[0].name}")
    return cameras[0]


def extract_camera_animation(camera_obj: bpy.types.Object, fps: float) -> dict:
    """
    Extract per-frame camera world matrix and lens data.

    Returns the raw Blender data (no coordinate conversion — done in fbx.py).
    """
    scene = bpy.context.scene

    # Use scene FPS if available, otherwise use the provided fps
    scene_fps = scene.render.fps / scene.render.fps_base
    actual_fps = scene_fps if scene_fps > 0 else fps

    frame_start = scene.frame_start
    frame_end = scene.frame_end

    log(f"Camera: {camera_obj.name}")
    log(f"FPS: {actual_fps:.3f}")
    log(f"Frames: {frame_start} - {frame_end} ({frame_end - frame_start + 1} total)")

    cam_data = camera_obj.data
    frames = []

    for frame_num in range(frame_start, frame_end + 1):
        scene.frame_set(frame_num)

        # Get world matrix at this frame
        world_matrix = camera_obj.matrix_world

        # Convert to list of lists (row-major 4x4)
        matrix_list = [list(row) for row in world_matrix]

        # Get focal length (may be animated)
        focal_length_mm = cam_data.lens
        sensor_width_mm = cam_data.sensor_width

        frames.append({
            "frame": frame_num,
            "matrix": matrix_list,
            "focal_length_mm": focal_length_mm,
            "sensor_width_mm": sensor_width_mm,
        })

    return {
        "camera_name": camera_obj.name,
        "fps": actual_fps,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "frames": frames,
    }


def main() -> None:
    file_path, fps = find_fbx_path_and_fps()

    if not os.path.exists(file_path):
        log(f"ERROR: File not found: {file_path}")
        sys.exit(1)

    log(f"Starting camera extraction for: {file_path}")
    log(f"Requested FPS: {fps}")

    clear_scene()
    import_file(file_path)

    camera_obj = find_camera()
    if camera_obj is None:
        log("ERROR: No camera object found in the scene after import.")
        sys.exit(1)

    data = extract_camera_animation(camera_obj, fps)

    # Output to stdout with a prefix so the caller can find the JSON line
    json_str = json.dumps(data, separators=(",", ":"))
    print(f"CAMERA_DATA:{json_str}", flush=True)
    log(f"Done. Extracted {len(data['frames'])} frames.")


if __name__ == "__main__":
    main()
